"""Durable deduplication and an ordered outbox, owned by one bridge process."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .configs import ConfigurationError, open_private, private_directory


class CapacityError(RuntimeError):
    pass


class InputConflict(ValueError):
    pass


class Store:
    def __init__(self, config):
        self.config = config
        self.db = None
        self.lock = None
        private_directory(Path(config.state_path).parent)
        try:
            self.lock = open_private(config.state_path + '.lock')
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fd = open_private(config.state_path)
            os.close(fd)
            # Small synchronous local transactions deliberately include fsync:
            # acknowledging before durable commit would lose events on a crash.
            # No network operations or database worker threads live here.
            self.db = sqlite3.connect(config.state_path, timeout=0)
            self.db.row_factory = sqlite3.Row
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('PRAGMA synchronous=FULL')
            page_size = self.db.execute('PRAGMA page_size').fetchone()[0]
            pages = config.max_state_bytes // page_size
            actual = self.db.execute(f'PRAGMA max_page_count={pages}').fetchone()[0]
            if actual > pages:
                raise CapacityError('Existing database exceeds max_state_bytes')
            version = self.db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ConfigurationError('Unsupported state database version')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance TEXT UNIQUE NOT NULL, conversation TEXT NOT NULL,
                    frontend TEXT NOT NULL, last_seq INTEGER NOT NULL DEFAULT 0,
                    snapshot_seq INTEGER NOT NULL DEFAULT 0,
                    paused INTEGER NOT NULL DEFAULT 0, active TEXT,
                    online INTEGER NOT NULL DEFAULT 0,
                    closing INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE NOT NULL, author TEXT NOT NULL,
                    origin_id TEXT, input_id TEXT UNIQUE NOT NULL,
                    body TEXT NOT NULL, target TEXT,
                    status TEXT NOT NULL DEFAULT 'recorded', outcome TEXT,
                    UNIQUE(author, origin_id));
                CREATE TABLE IF NOT EXISTS events (
                    instance TEXT NOT NULL, seq INTEGER NOT NULL,
                    input_id TEXT, origin TEXT, turn_id TEXT, type TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(instance, seq));
                CREATE INDEX IF NOT EXISTS event_inputs
                    ON events(instance, input_id, origin, turn_id, type, seq);
                CREATE TABLE IF NOT EXISTS outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    origin_id TEXT UNIQUE NOT NULL, body TEXT NOT NULL);
            ''')
            with self.transaction():
                identity = json.dumps([config.jid, config.room])
                previous = self.db.execute(
                    "SELECT value FROM metadata WHERE key='destination'").fetchone()
                if previous is not None and previous[0] != identity:
                    raise ConfigurationError('Database belongs to another bot/room')
                self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('destination', ?)",
                                (identity,))
                self.db.execute('PRAGMA user_version=1')
                self.db.execute('UPDATE sessions SET online=0')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None

    @contextmanager
    def transaction(self):
        with self.db:
            yield

    @staticmethod
    def alias(session):
        return f"s{session['id']}"

    def session(self, instance):
        return self.db.execute('SELECT * FROM sessions WHERE instance=?',
                               (instance,)).fetchone()

    def lookup(self, alias):
        if not alias.startswith('s') or not alias[1:].isascii() or not alias[1:].isdecimal():
            return None
        number = alias[1:]
        if len(number) > 18 or str(int(number)) != number:
            return None
        return self.db.execute('SELECT * FROM sessions WHERE id=?',
                               (int(number),)).fetchone()

    def sessions(self):
        return self.db.execute('SELECT * FROM sessions ORDER BY id').fetchall()

    def register(self, message):
        instance = message['instance_id']
        with self.transaction():
            previous = self.session(instance)
            if previous and (previous['last_seq'] > message['event_seq']
                             or previous['conversation'] != message['conversation_id']):
                raise ValueError('Live instance identity/sequence changed')
            self.db.execute('''INSERT INTO sessions(instance, conversation, frontend)
                               VALUES (?, ?, ?) ON CONFLICT(instance) DO NOTHING''',
                            (instance, message['conversation_id'], message['frontend']))
            session = self.session(instance)
            baseline = max(session['last_seq'], message['acknowledged'],
                           message['replay_from'] - 1)
            if baseline > session['last_seq']:
                self.add_outbox(f"[{self.alias(session)}] Missing Loki events "
                                f"{session['last_seq'] + 1}..{baseline}; results may be incomplete.")
            active = json.dumps(message['active']) if message['active'] is not None else None
            self.db.execute('''UPDATE sessions SET last_seq=?, snapshot_seq=?, paused=?,
                               active=?, online=1, closing=0 WHERE instance=?''',
                            (baseline, message['event_seq'], message['paused'], active, instance))
            retained = {item['input_id']: item for item in message['inputs']}
            for chat in self.db.execute('''SELECT * FROM chats WHERE target=?
                                          AND status IN ('submitted','accepted','running','uncertain')''',
                                        (instance,)).fetchall():
                item = retained.get(chat['input_id'])
                if item is not None:
                    status = {'queued': 'accepted', 'running': 'running',
                              'finished': 'finished'}[item['status']]
                    self.db.execute('UPDATE chats SET status=?, outcome=? WHERE id=?',
                                    (status, item.get('outcome'), chat['id']))
                elif chat['status'] != 'uncertain':
                    self.db.execute("UPDATE chats SET status='uncertain' WHERE id=?", (chat['id'],))
                    self.add_outbox(f"[{self.alias(session)}] Request {chat['input_id'][:8]} "
                                    "has uncertain execution status; it will NOT be resubmitted.")
            self.add_outbox(f"[{self.alias(session)}] Online "
                            f"(conversation {message['conversation_id'][:8]}).")
        return self.session(instance)

    def offline(self, instance):
        session = self.session(instance)
        if session is None or not session['online']:
            return
        with self.transaction():
            self.db.execute('UPDATE sessions SET online=0 WHERE instance=?', (instance,))
            if not session['closing']:
                self.add_outbox(f'[{self.alias(session)}] Offline.')

    def record_chat(self, message_id, author, origin_id, body, target):
        with self.transaction():
            old = self.db.execute('SELECT * FROM chats WHERE message_id=?', (message_id,)).fetchone()
            if old is None and origin_id is not None:
                old = self.db.execute('SELECT * FROM chats WHERE author=? AND origin_id=?',
                                      (author, origin_id)).fetchone()
            if old is not None:
                if old['author'] != author or old['body'] != body:
                    raise InputConflict('Conflicting reuse of an XMPP message ID')
                return False
            pending = self.db.execute("SELECT COUNT(*) FROM chats WHERE status='recorded'").fetchone()[0]
            status = 'recorded' if pending < self.config.max_pending_commands else 'rejected'
            self.db.execute('''INSERT INTO chats
                               (message_id, author, origin_id, input_id, body, target, status)
                               VALUES (?, ?, ?, ?, ?, ?, ?)''',
                            (message_id, author, origin_id, str(uuid.uuid4()), body, target, status))
            if status == 'rejected':
                self.add_outbox('[bridge] Command queue full; command not submitted.')
        return status == 'recorded'

    def next_chat(self):
        return self.db.execute(
            "SELECT * FROM chats WHERE status='recorded' ORDER BY id LIMIT 1").fetchone()

    def chat_status(self, chat, status, *, notice=None):
        with self.transaction():
            self.db.execute('UPDATE chats SET status=? WHERE id=?', (status, chat['id']))
            if notice:
                self.add_outbox(notice)

    def add_outbox(self, body):
        if len(body.encode('utf-8')) > self.config.message_bytes:
            raise CapacityError('Outgoing message exceeds message_bytes')
        count = self.db.execute('SELECT COUNT(*) FROM outbox').fetchone()[0]
        if count >= self.config.max_outbox_messages:
            raise CapacityError('XMPP outbox is full')
        self.db.execute('INSERT INTO outbox(origin_id, body) VALUES (?, ?)',
                        (str(uuid.uuid4()), body))

    def queue_messages(self, messages):
        with self.transaction():
            for message in messages:
                self.add_outbox(message)

    def next_outgoing(self):
        return self.db.execute('SELECT * FROM outbox ORDER BY id LIMIT 1').fetchone()

    def delivered(self, origin_id, body):
        with self.transaction():
            cursor = self.db.execute('DELETE FROM outbox WHERE origin_id=? AND body=?',
                                     (origin_id, body))
        return bool(cursor.rowcount)

    def chunks(self, instance, message, kind):
        rows = self.db.execute('''SELECT payload FROM events WHERE instance=? AND input_id=?
                                 AND origin=? AND turn_id IS ? AND type=? ORDER BY seq''',
                               (instance, message['input_id'], message['origin'],
                                message.get('turn_id'), kind))
        parts = []
        size = 0
        incomplete = False
        for index, row in enumerate(rows):
            chunk = json.loads(row[0])
            incomplete |= chunk['chunk_index'] != index
            size += len(chunk['text'].encode('utf-8'))
            if size > self.config.max_result_bytes:
                return '[Output exceeds max_result_bytes; recorded in local state.]', True
            parts.append(chunk['text'])
        return ''.join(parts), incomplete

    def event(self, instance, message, render):
        sequence = message['event_seq']
        with self.transaction():
            session = self.session(instance)
            if sequence <= session['last_seq']:
                return session['last_seq']
            if sequence != session['last_seq'] + 1:
                raise ValueError('Unexpected event sequence gap')
            self.db.execute('''INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)''',
                            (instance, sequence, message.get('input_id'), message.get('origin'),
                             message.get('turn_id'), message['type'], json.dumps(message)))
            kind = message['type']
            if sequence > session['snapshot_seq']:
                if kind == 'turn_started':
                    self.db.execute('UPDATE sessions SET active=? WHERE instance=?',
                                    (json.dumps(message), instance))
                elif kind == 'turn_finished':
                    self.db.execute('UPDATE sessions SET active=NULL, paused=? WHERE instance=?',
                                    (message['paused'], instance))
                elif kind == 'session_state':
                    self.db.execute('UPDATE sessions SET paused=? WHERE instance=?',
                                    (message['paused'], instance))
                elif kind == 'session_closing':
                    self.db.execute('UPDATE sessions SET closing=1 WHERE instance=?', (instance,))
            status = {'prompt_accepted': 'accepted', 'prompt_rejected': 'rejected',
                      'turn_started': 'running', 'turn_finished': 'finished',
                      'command_finished': 'finished'}.get(kind)
            if kind == 'prompt_accepted':
                status = {'queued': 'accepted', 'running': 'running',
                          'finished': 'finished'}[message['status']]
            terminal = status in ('finished', 'rejected')
            if status and (terminal or sequence > session['snapshot_seq']):
                self.db.execute('''UPDATE chats SET status=?, outcome=? WHERE target=? AND input_id=?
                                   AND status NOT IN ('finished','rejected')''',
                                (status, message.get('outcome') or message.get('reason'),
                                 instance, message.get('input_id')))
            for text in render(self.alias(session), message, self):
                self.add_outbox(text)
            self.db.execute('UPDATE sessions SET last_seq=? WHERE instance=?', (sequence, instance))
        return sequence
