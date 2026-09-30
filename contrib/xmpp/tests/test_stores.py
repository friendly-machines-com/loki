from dataclasses import replace
import json
import sqlite3
import unittest
import uuid

from loki_xmpp_bridge import messages
from loki_xmpp_bridge.configs import ConfigurationError
from loki_xmpp_bridge.stores import CapacityError, InputConflict, Store
from fixtures import config_directory, registration


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory, self.config = config_directory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(self.config)
        self.addCleanup(lambda: self.store.close())
        self.reg = registration()
        self.session = self.store.register(self.reg)

    def output(self):
        return [row[0] for row in self.store.db.execute('SELECT body FROM outbox ORDER BY id')]

    def event(self, kind, sequence, **fields):
        return {'type': kind, 'instance_id': self.reg['instance_id'],
                'event_seq': sequence, **fields}

    def test_aliases_survive_restart_and_are_never_reused_for_resumed_conversation(self):
        alias = self.store.alias(self.session)
        self.store.close()
        self.store = Store(self.config)
        session = self.store.register(self.reg)
        self.assertEqual(self.store.alias(session), alias)
        another = self.store.register(registration(conversation=self.reg['conversation_id']))
        self.assertNotEqual(self.store.alias(another), alias)
        self.assertFalse(self.store.lookup('s01'))
        self.assertFalse(self.store.lookup('s' + '9' * 100))

    def test_database_has_single_owner_and_is_bound_to_destination(self):
        with self.assertRaises(BlockingIOError):
            Store(self.config)
        self.store.close()
        with self.assertRaisesRegex(ConfigurationError, 'another bot/room'):
            Store(replace(self.config, room='other@conference.example.test'))

    def test_incoming_ids_are_durable_and_origin_ids_are_author_scoped(self):
        args = ('message-1', 'alice@example.test', 'origin-1', '/to s1 hello', self.reg['instance_id'])
        self.assertTrue(self.store.record_chat(*args))
        self.assertFalse(self.store.record_chat(*args))
        self.assertFalse(self.store.record_chat('message-2', *args[1:]))
        with self.assertRaises(InputConflict):
            self.store.record_chat('message-3', 'alice@example.test', 'origin-1',
                                   '/to s1 different', self.reg['instance_id'])
        self.assertTrue(self.store.record_chat(
            'message-4', 'bob@example.test', 'origin-1', '/to s1 different', self.reg['instance_id']))
        self.store.close()
        self.store = Store(self.config)
        self.assertFalse(self.store.record_chat(*args))

    def test_incoming_queue_overflow_records_rejection_without_execution(self):
        self.store.config = replace(self.config, max_pending_commands=1)
        self.store.record_chat('first', 'alice@example.test', None, '/help', None)
        self.assertFalse(self.store.record_chat('second', 'alice@example.test', None, '/help', None))
        row = self.store.db.execute("SELECT status FROM chats WHERE message_id='second'").fetchone()
        self.assertEqual(row[0], 'rejected')
        self.assertIn('queue full', self.output()[-1])

    def test_outbox_requires_matching_echo_body_and_survives_restart(self):
        outgoing = dict(self.store.next_outgoing())
        self.assertFalse(self.store.delivered(outgoing['origin_id'], 'forged body'))
        self.store.close()
        self.store = Store(self.config)
        self.assertEqual(dict(self.store.next_outgoing()), outgoing)
        self.assertTrue(self.store.delivered(outgoing['origin_id'], outgoing['body']))
        self.assertIsNone(self.store.next_outgoing())

    def test_event_commit_replay_and_generation_are_atomic(self):
        event = self.event('session_state', 1, paused=False)
        self.store.event(self.reg['instance_id'], event, messages.render)
        self.assertEqual(self.store.session(self.reg['instance_id'])['last_seq'], 1)
        count = len(self.output())
        self.store.close()
        self.store = Store(self.config)
        self.store.event(self.reg['instance_id'], event, messages.render)
        self.assertEqual(len(self.output()), count)
        self.store.config = replace(self.config, max_outbox_messages=count)
        with self.assertRaises(CapacityError):
            self.store.event(
                self.reg['instance_id'], self.event('session_state', 2, paused=True), messages.render)
        self.assertEqual(self.store.session(self.reg['instance_id'])['last_seq'], 1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 1)

    def test_registration_gaps_and_snapshot_state_are_not_rolled_back_by_replay(self):
        instance = self.reg['instance_id']
        updated = {**self.reg, 'event_seq': 3, 'replay_from': 2, 'paused': True}
        self.store.register(updated)
        self.assertIn('Missing Loki events 1..1', self.output()[-2])
        self.store.event(instance, self.event('session_state', 2, paused=False), messages.render)
        self.assertTrue(self.store.session(instance)['paused'])
        with self.assertRaisesRegex(ValueError, 'sequence gap'):
            self.store.event(instance, self.event('session_state', 4, paused=False), messages.render)

    def test_submitted_commands_reconcile_without_retransmission(self):
        instance = self.reg['instance_id']
        self.store.record_chat('request', 'alice@example.test', None, '/to s1 hi', instance)
        chat = self.store.next_chat()
        self.store.chat_status(chat, 'submitted')
        self.store.register(self.reg)
        status = self.store.db.execute('SELECT status FROM chats').fetchone()[0]
        self.assertEqual(status, 'uncertain')
        self.assertIsNone(self.store.next_chat())
        inputs = [{'input_id': chat['input_id'], 'status': 'finished', 'outcome': 'completed'}]
        self.store.register({**self.reg, 'inputs': inputs})
        self.assertEqual(self.store.db.execute('SELECT status FROM chats').fetchone()[0], 'finished')

    def test_chunk_assembly_gap_labels_and_keyboard_mirroring(self):
        instance, turn, input_id = self.reg['instance_id'], str(uuid.uuid4()), str(uuid.uuid4())
        base = {'origin': 'keyboard', 'turn_id': turn, 'input_id': input_id}
        events = [self.event('input_chunk', 1, **base, chunk_index=0, text='keyboard prompt'),
                  self.event('output_chunk', 2, **base, chunk_index=1, text='partial reply'),
                  self.event('turn_finished', 3, **base, outcome='completed', paused=False)]
        for event in events:
            self.store.event(instance, event, messages.render)
        self.assertIn('Keyboard prompt', self.output()[-2])
        self.assertIn('keyboard prompt', self.output()[-2])
        self.assertIn('output incomplete', self.output()[-1])
        self.assertIn('partial reply', self.output()[-1])

    def test_payloads_are_retained_and_result_limit_is_explicit(self):
        self.store.config = replace(self.config, max_result_bytes=4)
        instance = self.reg['instance_id']
        base = {'origin': 'bridge', 'turn_id': None, 'input_id': str(uuid.uuid4())}
        chunk = self.event('output_chunk', 1, **base, chunk_index=0, text='long reply')
        self.store.event(instance, chunk, messages.render)
        finish = self.event('command_finished', 2, **base, outcome='completed')
        self.store.event(instance, finish, messages.render)
        self.assertIn('max_result_bytes', self.output()[-1])
        payload = self.store.db.execute('SELECT payload FROM events WHERE seq=1').fetchone()[0]
        self.assertEqual(json.loads(payload)['text'], 'long reply')

    def test_database_quota_is_enforced_without_advancing_cursor(self):
        self.store.close()
        self.store = Store(replace(self.config, max_state_bytes=1024 * 1024))
        with self.assertRaises(sqlite3.DatabaseError):
            with self.store.transaction():
                self.store.db.execute('INSERT INTO outbox(origin_id, body) VALUES (?, ?)',
                                      (str(uuid.uuid4()), 'x' * (2 * 1024 * 1024)))
        self.assertEqual(self.store.session(self.reg['instance_id'])['last_seq'], 0)


class MessageTests(unittest.TestCase):
    def test_unicode_splitting_is_byte_bounded_and_preserves_content(self):
        value = '\U0001f680' * 1000
        pieces = messages.split_text('[s1] Output:\n', value, 256)
        self.assertTrue(all(len(piece.encode('utf-8')) <= 256 for piece in pieces))
        content = ''.join(piece.split('\n', 2)[2] for piece in pieces)
        self.assertEqual(content, value)

    def test_xml_line_endings_are_canonical_for_echo_confirmation(self):
        self.assertEqual(messages.xml_text('a\r\nb\rc'), 'a\nb\nc')

    def test_xml_control_characters_are_replaced(self):
        self.assertEqual(messages.xml_text('a\x00\x1bb\n'), 'a\ufffd\ufffdb\n')
