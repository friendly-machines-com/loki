import asyncio
from dataclasses import replace
import unittest
from unittest import mock
from xml.etree import ElementTree as ET

from slixmpp import JID
from slixmpp.stanza import Message, StreamFeatures

from loki_xmpp_bridge import xmpps
from loki_xmpp_bridge.routes import Router
from loki_xmpp_bridge.stores import Store
from fixtures import config_directory, registration, FakePeer


class XmppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory, self.config = config_directory()
        self.store = Store(self.config)
        self.router = Router(self.config, self.store)
        self.peer = FakePeer()
        self.router.register(self.peer, registration())
        self.client = xmpps.RoomClient(self.config, 'test-password', self.router)
        self.client.init_plugins()
        self.client.ready.set()
        self.occupant('Alice', 'alice@example.test/phone')
        self.occupant('Mallory', 'mallory@example.test/client')
        self.occupant(self.config.nick, self.config.jid + '/bridge')

    async def asyncTearDown(self):
        await self.client.close()
        self.store.close()
        self.directory.cleanup()

    def occupant(self, nick, jid, role='participant'):
        self.client.plugin['xep_0045'].rooms[None].setdefault(JID(self.config.room), {})[nick] = {
            'jid': JID(jid) if jid else None, 'role': role}

    def message(self, body='/to s1 hello', *, nick='Alice', message_id='room-id',
                origin=None, room=None, by=None, delay=False):
        stanza = self.client.make_message(
            mto=self.config.jid, mfrom=(room or self.config.room) + '/' + nick,
            mbody=body, mtype='groupchat')
        if message_id is not None:
            ET.SubElement(stanza.xml, '{urn:xmpp:sid:0}stanza-id',
                          {'id': message_id, 'by': by or self.config.room})
        if origin is not None:
            ET.SubElement(stanza.xml, '{urn:xmpp:sid:0}origin-id', {'id': origin})
        if delay:
            ET.SubElement(stanza.xml, '{urn:xmpp:delay}delay', {'stamp': '2026-01-01T00:00:00Z'})
        return Message(xml=ET.fromstring(ET.tostring(stanza.xml)), stream=self.client)

    async def test_authenticated_live_message_routes_and_deduplicates(self):
        stanza = self.message(origin='client-id')
        self.client.receive_message(stanza)
        self.client.receive_message(stanza)
        chat = self.store.next_chat()
        self.assertEqual(chat['author'], 'alice@example.test')
        await self.router._dispatch(chat)
        self.assertEqual(len(self.peer.sent), 1)
        self.assertEqual(self.peer.sent[0]['text'], 'hello')
        self.client.receive_message(self.message(message_id='another-room-id', origin='client-id'))
        self.assertIsNone(self.store.next_chat())

    async def test_nickname_reuse_after_receipt_cannot_change_captured_author(self):
        self.client.receive_message(self.message())
        self.occupant('Alice', 'mallory@example.test/client')
        chat = self.store.next_chat()
        await self.router._dispatch(chat)
        self.assertEqual(chat['author'], 'alice@example.test')
        self.assertEqual(len(self.peer.sent), 1)
        self.client.receive_message(self.message(message_id='later-message'))
        self.assertIsNone(self.store.next_chat())

    async def test_unauthorized_missing_identity_wrong_room_and_spoofed_jid_are_ignored(self):
        candidates = [self.message(nick='Mallory'),
                      self.message(nick='Unknown'),
                      self.message(room='other@conference.example.test')]
        spoofed = self.message(nick='Mallory')
        x = ET.SubElement(spoofed.xml, '{http://jabber.org/protocol/muc#user}x')
        ET.SubElement(x, '{http://jabber.org/protocol/muc#user}item', {'jid': 'alice@example.test'})
        candidates.append(spoofed)
        for stanza in candidates:
            self.client.receive_message(stanza)
        self.assertIsNone(self.store.next_chat())

    async def test_history_mam_forwarded_and_missing_or_untrusted_ids_are_ignored(self):
        forwarded = self.message(message_id='forwarded')
        ET.SubElement(forwarded.xml, '{urn:xmpp:forward:0}forwarded')
        mam = self.message(message_id='mam')
        ET.SubElement(mam.xml, '{urn:xmpp:mam:2}result')
        duplicate_id = self.message(message_id='duplicate-id')
        ET.SubElement(duplicate_id.xml, '{urn:xmpp:sid:0}stanza-id',
                      {'id': 'second-id', 'by': self.config.room})
        for stanza in (self.message(delay=True), forwarded, mam,
                       self.message(message_id=None), self.message(by='attacker.example.test'), duplicate_id):
            self.client.receive_message(stanza)
        self.assertIsNone(self.store.next_chat())

    async def test_bot_echo_and_unaddressed_chat_never_execute(self):
        for stanza in (self.message(nick=self.config.nick),
                       self.message(body='ordinary conversation'), self.message(body='/tools')):
            self.client.receive_message(stanza)
        self.assertIsNone(self.store.next_chat())

    async def test_outgoing_confirmation_requires_authenticated_live_matching_self_echo(self):
        row = dict(self.store.next_outgoing())
        sent = []
        with mock.patch.object(self.client, 'send', side_effect=sent.append):
            task = asyncio.create_task(self.client.send_confirmed(row))
            await asyncio.sleep(0)
            self.assertEqual(sent[0].xml.find('{urn:xmpp:sid:0}origin-id').get('id'), row['origin_id'])
            for stanza in (self.message(row['body'], nick='Mallory', origin=row['origin_id']),
                           self.message('wrong body', nick=self.config.nick, origin=row['origin_id']),
                           self.message(row['body'], nick=self.config.nick,
                                        origin=row['origin_id'], delay=True)):
                self.client.receive_message(stanza)
            self.assertFalse(task.done())
            self.assertIsNotNone(self.store.next_outgoing())
            self.client.receive_message(self.message(row['body'], nick=self.config.nick,
                                                     origin=row['origin_id']))
            await asyncio.wait_for(task, 1)
        self.assertIsNone(self.store.next_outgoing())

    async def test_disconnect_preserves_outbox_and_releases_pending_confirmation(self):
        row = dict(self.store.next_outgoing())
        with mock.patch.object(self.client, 'send'):
            task = asyncio.create_task(self.client.send_confirmed(row))
            await asyncio.sleep(0)
            self.client.connection_stopped(None)
            with self.assertRaises(ConnectionError):
                await task
        self.assertEqual(dict(self.store.next_outgoing()), row)
        self.assertFalse(self.client.ready.is_set())

    async def test_echo_timeout_retains_same_origin_id_for_retry(self):
        self.client.config = replace(self.config, echo_timeout=0.01)
        row = dict(self.store.next_outgoing())
        with mock.patch.object(self.client, 'send'), self.assertRaises(TimeoutError):
            await self.client.send_confirmed(row)
        self.assertEqual(dict(self.store.next_outgoing()), row)
        self.assertFalse(self.client.echoes)

    async def test_actual_plaintext_server_receives_no_auth_and_client_workers_close(self):
        captured = asyncio.get_running_loop().create_future()
        peers = []

        async def plaintext_server(reader, writer):
            peers.append(writer)
            data = await reader.read(4096)
            writer.write(
                b'<stream:stream from="example.test" id="test" xmlns="jabber:client" '
                b'xmlns:stream="http://etherx.jabber.org/streams" version="1.0">'
                b'<stream:features><mechanisms xmlns="urn:ietf:params:xml:ns:xmpp-sasl">'
                b'<mechanism>PLAIN</mechanism></mechanisms></stream:features>')
            await writer.drain()
            data += await reader.read()
            if not captured.done():
                captured.set_result(data)
            writer.close()

        server = await asyncio.start_server(plaintext_server, '127.0.0.1', 0)
        self.client.ready.clear()
        try:
            await self.client.connect('127.0.0.1', server.sockets[0].getsockname()[1])
            await asyncio.wait_for(self.client.stopped.wait(), 2)
            data = await asyncio.wait_for(captured, 2)
            self.assertNotIn(b'<auth', data)
            await self.client.close()
            self.assertIsNone(self.client.filter_task)
        finally:
            server.close()
            await server.wait_closed()
            for writer in peers:
                writer.close()
                await writer.wait_closed()

    async def test_tls_is_required_before_password_authentication(self):
        features = StreamFeatures(xml=ET.fromstring(
            '<features xmlns="http://etherx.jabber.org/streams">'
            '<mechanisms xmlns="urn:ietf:params:xml:ns:xmpp-sasl">'
            '<mechanism>PLAIN</mechanism></mechanisms></features>'), stream=self.client)
        with mock.patch.object(self.client, 'abort') as abort:
            await self.client._handle_stream_features(features)
        abort.assert_called_once()
        self.assertTrue(self.client.stopped.is_set())
        self.assertTrue(self.client.ssl_context.check_hostname)
        self.assertFalse(self.client.enable_plaintext)
        self.assertFalse(self.client.enable_direct_tls)

    async def test_feature_validation_and_room_change_fail_closed(self):
        plugin = self.client.plugin['xep_0030']
        info = {'disco_info': {'features': xmpps.REQUIRED_FEATURES - {'muc_nonanonymous'}}}
        with mock.patch.object(plugin, 'get_info', mock.AsyncMock(return_value=info)):
            with self.assertRaisesRegex(ValueError, 'muc_nonanonymous'):
                await self.client.room_features()
        change = self.message()
        change['from'] = self.config.room
        with mock.patch.object(self.client, 'abort') as abort:
            self.client.room_changed(change)
        abort.assert_called_once()
        self.assertFalse(self.client.ready.is_set())
        self.client.receive_message(self.message())
        self.assertIsNone(self.store.next_chat())

    async def test_occupant_cannot_forge_room_configuration_change(self):
        with mock.patch.object(self.client, 'abort') as abort:
            self.client.room_changed(self.message())
        abort.assert_not_called()
        self.assertTrue(self.client.ready.is_set())

    async def test_join_does_not_silently_create_a_room(self):
        presence = {'muc': {'status_codes': {201}}, 'from': JID(self.config.room + '/' + self.config.nick)}
        with (
            mock.patch.object(self.client, 'room_features', mock.AsyncMock()),
            mock.patch.object(self.client.plugin['xep_0045'], 'join_muc_wait',
                              mock.AsyncMock(return_value=(presence, None, [], []))),
            mock.patch.object(self.client, 'abort') as abort,
        ):
            self.client.ready.clear()
            await self.client.join_room()
        self.assertFalse(self.client.ready.is_set())
        abort.assert_called_once()


class ReconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconnect_preserves_outbox_identity_order_and_closes_clients(self):
        directory, config = config_directory()
        store = Store(config)
        router = Router(config, store)
        store.queue_messages(['first', 'second'])
        original_id = store.next_outgoing()['origin_id']
        clients = []
        delivered = asyncio.Event()
        observed = []

        class Client:
            def __init__(self, *args):
                self.ready = asyncio.Event()
                self.ready.set()
                self.stopped = asyncio.Event()
                self.closed = False
                clients.append(self)

            async def connect(self, *args):
                pass

            async def pump_outbox(self):
                if len(clients) == 1:
                    raise ConnectionError('temporary outage')
                while row := store.next_outgoing():
                    observed.append((row['origin_id'], row['body']))
                    store.delivered(row['origin_id'], row['body'])
                delivered.set()
                await asyncio.Event().wait()

            async def close(self):
                self.closed = True

        task = None
        try:
            with mock.patch.object(xmpps.random, 'uniform', return_value=0):
                task = asyncio.create_task(xmpps.run(config, 'password', router, client_factory=Client))
                await asyncio.wait_for(delivered.wait(), 2)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.assertEqual([body for _id, body in observed], ['first', 'second'])
            self.assertEqual(observed[0][0], original_id)
            self.assertEqual(len(clients), 2)
            self.assertTrue(all(client.closed for client in clients))
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            store.close()
            directory.cleanup()


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory, self.config = config_directory()
        self.store = Store(self.config)
        self.router = Router(self.config, self.store)
        self.peer = FakePeer()
        self.router.register(self.peer, registration())

    async def asyncTearDown(self):
        self.store.close()
        self.directory.cleanup()

    async def dispatch(self, body, message_id='request'):
        self.router.receive_chat('alice@example.test', message_id, None, body)
        chat = self.store.next_chat()
        if chat:
            await self.router._dispatch(chat)
        return chat

    async def test_sessions_query_is_local_and_lists_online_offline_paused(self):
        await self.dispatch('/sessions')
        self.assertFalse(self.peer.sent)
        output = self.store.db.execute('SELECT body FROM outbox ORDER BY id DESC LIMIT 1').fetchone()[0]
        self.assertIn('s1: idle', output)
        self.router.closed(self.peer)
        await self.dispatch('/sessions', 'request-2')
        output = self.store.db.execute('SELECT body FROM outbox ORDER BY id DESC LIMIT 1').fetchone()[0]
        self.assertIn('s1: offline', output)

    async def test_offline_unknown_and_disabled_targets_are_not_forwarded(self):
        self.peer.accepts_prompts = False
        await self.dispatch('/to s1 hello')
        self.router.closed(self.peer)
        await self.dispatch('/to s1 hello', 'request-2')
        await self.dispatch('/to s999 hello', 'request-3')
        self.assertFalse(self.peer.sent)
        states = self.store.db.execute('SELECT status FROM chats').fetchall()
        self.assertTrue(all(row[0] == 'rejected' for row in states))

    async def test_socket_write_failure_becomes_uncertain_not_retryable(self):
        self.peer.error = ConnectionError('lost connection')
        await self.dispatch('/to s1 hello')
        self.assertEqual(self.store.db.execute('SELECT status FROM chats').fetchone()[0], 'uncertain')
        self.assertIsNone(self.store.next_chat())

    async def test_target_is_bound_when_message_is_received(self):
        self.router.receive_chat('alice@example.test', 'request', None, '/to s1 hello')
        chat = self.store.next_chat()
        self.router.closed(self.peer)
        second = FakePeer()
        self.router.register(second, registration())
        await self.router._dispatch(chat)
        self.assertFalse(second.sent)
        self.assertEqual(chat['target'], self.peer.instance)

    async def test_unauthorized_adapter_cannot_bypass_author_allowlist(self):
        self.router.receive_chat('mallory@example.test', 'request', None, '/to s1 hello')
        self.assertIsNone(self.store.next_chat())
