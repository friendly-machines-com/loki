"""Explicit instance routing; remote input never broadcasts or follows a nickname."""

import asyncio
import logging
import re
import sqlite3

from . import messages
from .stores import CapacityError, InputConflict


logger = logging.getLogger(__name__)
HELP = 'Use /sessions, or /to sNUMBER PROMPT (including Loki slash commands).'
ADDRESS = re.compile(r'^/to\s+(s[1-9][0-9]*)\s+(.+)$', re.DOTALL)


class Router:
    def __init__(self, config, store):
        self.config = config
        self.store = store
        self.peers = {}
        self.incoming = asyncio.Event()
        self.outgoing = asyncio.Event()
        self.failed = asyncio.Event()
        self.failure = None

    def fail(self, error):
        if self.failure is None:
            self.failure = error
            logger.error('Bridge processing/storage failed (%s); stopping.', type(error).__name__)
            self.failed.set()

    def notice(self, text):
        self.store.queue_messages(messages.split_text('[bridge] ', text, self.config.message_bytes))
        self.outgoing.set()

    def register(self, peer, registration):
        instance = registration['instance_id']
        if instance in self.peers:
            raise ValueError('Live instance already has a connection')
        session = self.store.register(registration)
        peer.instance = instance
        peer.accepts_prompts = registration['capabilities']['accepts_prompts']
        self.peers[instance] = peer
        self.outgoing.set()
        self.incoming.set()
        return session['last_seq']

    def closed(self, peer):
        if peer.instance is not None and self.peers.get(peer.instance) is peer:
            del self.peers[peer.instance]
            self.store.offline(peer.instance)
            self.outgoing.set()

    def event(self, peer, event):
        if event['type'] == 'session_closing':
            peer.closing = True
        sequence = self.store.event(peer.instance, event, messages.render)
        self.outgoing.set()
        return sequence

    def receive_chat(self, author, message_id, origin_id, body):
        # Authentication is repeated here so a test/future adapter cannot bypass
        # the XMPP adapter's identity boundary. Targets are bound at receipt.
        if author not in self.config.allowed_jids:
            return
        body = body.strip()
        if len(body.encode('utf-8')) > 65536:
            self.notice('Command exceeds 64 KiB; not submitted.')
            return
        first = body.split(maxsplit=1)[0] if body else ''
        if first not in ('/sessions', '/help', '/to'):
            return
        match = ADDRESS.fullmatch(body)
        session = self.store.lookup(match[1]) if match else None
        target = session['instance'] if session else None
        try:
            recorded = self.store.record_chat(message_id, author, origin_id, body, target)
        except InputConflict:
            self.notice('Conflicting reuse of a message ID; command not submitted.')
            return
        if recorded:
            self.incoming.set()
        self.outgoing.set()

    async def worker(self):
        try:
            while True:
                self.incoming.clear()
                chat = self.store.next_chat()
                if chat is None:
                    await self.incoming.wait()
                    continue
                await self._dispatch(chat)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.fail(error)

    async def _dispatch(self, chat):
        if chat['author'] not in self.config.allowed_jids:
            self.store.chat_status(chat, 'rejected', notice='[bridge] Command author no longer authorized.')
        elif chat['body'] == '/sessions':
            lines = []
            for session in self.store.sessions():
                state = 'offline'
                if session['instance'] in self.peers:
                    state = 'running' if session['active'] else 'idle'
                    if session['paused']:
                        state += ', remote paused'
                lines.append(f"{self.store.alias(session)}: {state}; "
                             f"{session['frontend']}; conversation {session['conversation'][:8]}")
            with self.store.transaction():
                self.store.db.execute("UPDATE chats SET status='finished' WHERE id=?", (chat['id'],))
                for body in messages.split_text('[bridge] Sessions:\n',
                                                '\n'.join(lines) or 'No sessions yet.',
                                                self.config.message_bytes):
                    self.store.add_outbox(body)
        elif chat['body'] == '/help':
            self.store.chat_status(chat, 'finished', notice='[bridge] ' + HELP)
        else:
            match = ADDRESS.fullmatch(chat['body'])
            peer = self.peers.get(chat['target'])
            if not match or chat['target'] is None:
                self.store.chat_status(chat, 'rejected', notice='[bridge] Unknown/missing session. ' + HELP)
            elif peer is None or peer.closing:
                self.store.chat_status(
                    chat, 'rejected', notice=f'[{match[1]}] Offline; command not submitted.')
            elif not peer.accepts_prompts:
                self.store.chat_status(chat, 'rejected', notice=f'[{match[1]}] Remote input is disabled.')
            else:
                # Commit before attempting the write. After a crash/write error,
                # submitted is uncertain, never an instruction to retransmit.
                self.store.chat_status(chat, 'submitted')
                try:
                    await peer.send({'type': 'submit_prompt', 'instance_id': chat['target'],
                                     'input_id': chat['input_id'], 'text': match[2]})
                except (OSError, ConnectionError, TimeoutError):
                    current = self.store.db.execute('SELECT status FROM chats WHERE id=?',
                                                    (chat['id'],)).fetchone()[0]
                    if current == 'submitted':
                        self.store.chat_status(
                            chat, 'uncertain',
                            notice=f'[{match[1]}] Submission status uncertain; '
                            'NOT resubmitting. Check the session before issuing more work.')
        self.outgoing.set()


STORAGE_ERRORS = (sqlite3.Error, CapacityError)
