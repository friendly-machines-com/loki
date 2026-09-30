import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import unittest
import uuid

from loki_xmpp_bridge.routes import Router
from loki_xmpp_bridge.sockets import FRAME_BYTES, SocketServer, read_frame
from loki_xmpp_bridge.stores import Store
from fixtures import config_directory, registration


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory, self.config = config_directory()
        self.store = Store(self.config)
        self.router = Router(self.config, self.store)
        self.server = SocketServer(self.config, self.router)
        await self.server.__aenter__()
        self.clients = []

    async def asyncTearDown(self):
        for writer in self.clients:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
        await self.server.__aexit__()
        self.store.close()
        self.directory.cleanup()

    async def write(self, writer, message):
        writer.write((json.dumps(message) + '\n').encode('utf-8'))
        await writer.drain()

    async def connect(self, reg=None):
        reg = reg or registration()
        reader, writer = await asyncio.open_unix_connection(self.config.socket_path, limit=FRAME_BYTES)
        self.clients.append(writer)
        await self.write(writer, {'type': 'hello', 'version': 1})
        self.assertEqual(await read_frame(reader), {'type': 'hello', 'version': 1})
        await self.write(writer, reg)
        async with asyncio.timeout(2):
            while reg['instance_id'] not in self.router.peers:
                await asyncio.sleep(0.001)
        return reader, writer, reg

    async def test_multiple_connections_route_only_to_explicit_instance(self):
        conversation = str(uuid.uuid4())
        first_reader, _first_writer, first = await self.connect(registration(conversation=conversation))
        second_reader, _second_writer, second = await self.connect(registration(conversation=conversation))
        first_alias = self.store.alias(self.store.session(first['instance_id']))
        second_alias = self.store.alias(self.store.session(second['instance_id']))
        self.assertNotEqual(first_alias, second_alias)
        self.router.receive_chat('alice@example.test', 'room-id', 'origin',
                                 f'/to {second_alias} /model model --provider provider')
        chat = self.store.next_chat()
        await self.router._dispatch(chat)
        command = await asyncio.wait_for(read_frame(second_reader), 1)
        self.assertEqual(command['instance_id'], second['instance_id'])
        self.assertEqual(command['text'], '/model model --provider provider')
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(read_frame(first_reader), 0.02)

    async def test_ack_follows_durable_event_and_outbox_commit(self):
        reader, writer, reg = await self.connect()
        event = {'type': 'session_state', 'instance_id': reg['instance_id'],
                 'event_seq': 1, 'paused': True}
        await self.write(writer, event)
        self.assertEqual((await read_frame(reader))['event_seq'], 1)
        self.assertEqual(self.store.session(reg['instance_id'])['last_seq'], 1)
        self.assertIn('paused', self.store.db.execute(
            'SELECT body FROM outbox ORDER BY id DESC LIMIT 1').fetchone()[0])

    async def test_reconnect_acknowledges_replay_without_regenerating_output(self):
        reader, writer, reg = await self.connect()
        event = {'type': 'session_state', 'instance_id': reg['instance_id'],
                 'event_seq': 1, 'paused': False}
        await self.write(writer, event)
        await read_frame(reader)
        count = self.store.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        writer.close()
        await writer.wait_closed()
        async with asyncio.timeout(2):
            while reg['instance_id'] in self.router.peers:
                await asyncio.sleep(0.001)
        reader, writer, _reg = await self.connect({**reg, 'event_seq': 1})
        self.assertEqual((await read_frame(reader))['event_seq'], 1)
        await self.write(writer, event)
        self.assertEqual((await read_frame(reader))['event_seq'], 1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM events').fetchone()[0], count)

    async def test_gap_metadata_is_handled_before_replayed_events(self):
        reader, writer, reg = await self.connect(registration(latest=5, replay_from=5))
        self.assertEqual((await read_frame(reader))['event_seq'], 4)
        await self.write(writer, {'type': 'event_gap', 'instance_id': reg['instance_id'],
                                  'from_seq': 1, 'to_seq': 4})
        await self.write(writer, {'type': 'session_state', 'instance_id': reg['instance_id'],
                                  'event_seq': 5, 'paused': False})
        self.assertEqual((await read_frame(reader))['event_seq'], 5)
        gaps = self.store.db.execute("SELECT COUNT(*) FROM outbox WHERE body LIKE '%Missing Loki events%'")
        self.assertEqual(gaps.fetchone()[0], 1)

    async def test_duplicate_instance_connection_cannot_displace_original(self):
        _reader, _writer, reg = await self.connect()
        original = self.router.peers[reg['instance_id']]
        reader, writer = await asyncio.open_unix_connection(self.config.socket_path)
        self.clients.append(writer)
        await self.write(writer, {'type': 'hello', 'version': 1})
        await read_frame(reader)
        await self.write(writer, reg)
        self.assertEqual(await asyncio.wait_for(reader.read(), 1), b'')
        self.assertIs(self.router.peers[reg['instance_id']], original)

    async def test_invalid_version_and_oversized_frames_do_not_stop_other_sessions(self):
        for data in (b'{"type":"hello","version":true}\n', b'x' * (FRAME_BYTES + 1) + b'\n'):
            reader, writer = await asyncio.open_unix_connection(self.config.socket_path)
            self.clients.append(writer)
            writer.write(data)
            try:
                await writer.drain()
            except ConnectionError:
                pass
            try:
                self.assertEqual(await asyncio.wait_for(reader.read(), 1), b'')
            except ConnectionResetError:
                pass
        self.assertFalse(self.router.failed.is_set())
        await self.connect()

    async def test_bad_events_do_not_advance_durable_cursor(self):
        reader, writer, reg = await self.connect()
        await self.write(writer, {'type': 'session_state', 'instance_id': reg['instance_id'],
                                  'event_seq': 1, 'paused': 'not-bool'})
        self.assertEqual(await asyncio.wait_for(reader.read(), 1), b'')
        self.assertEqual(self.store.session(reg['instance_id'])['last_seq'], 0)

    async def test_invalid_optional_sql_fields_are_protocol_errors_not_storage_failures(self):
        reader, writer, reg = await self.connect()
        await self.write(writer, {'type': 'session_state', 'instance_id': reg['instance_id'],
                                  'event_seq': 1, 'paused': False, 'origin': {'invalid': 'binding'}})
        self.assertEqual(await asyncio.wait_for(reader.read(), 1), b'')
        self.assertFalse(self.router.failed.is_set())
        self.assertEqual(self.store.session(reg['instance_id'])['last_seq'], 0)

    async def test_outbox_failure_does_not_ack_event(self):
        reader, writer, reg = await self.connect()
        count = self.store.db.execute('SELECT COUNT(*) FROM outbox').fetchone()[0]
        self.store.config = replace(self.config, max_outbox_messages=count)
        await self.write(writer, {'type': 'session_state', 'instance_id': reg['instance_id'],
                                  'event_seq': 1, 'paused': False})
        self.assertEqual(await asyncio.wait_for(reader.read(), 1), b'')
        self.assertTrue(self.router.failed.is_set())
        self.assertEqual(self.store.session(reg['instance_id'])['last_seq'], 0)

    async def test_listener_lock_and_permissions(self):
        self.assertEqual(Path(self.config.socket_path).stat().st_mode & 0o777, 0o600)
        self.assertFalse(os.get_inheritable(self.server.server.sockets[0].fileno()))
        another = SocketServer(self.config, self.router)
        with self.assertRaises(BlockingIOError):
            await another.__aenter__()
        self.assertTrue(Path(self.config.socket_path).exists())

    async def test_replacement_path_is_not_removed_on_shutdown(self):
        path = Path(self.config.socket_path)
        path.unlink()
        path.write_text('replacement')
        await self.server.__aexit__()
        self.assertEqual(path.read_text(), 'replacement')

    async def test_unsafe_existing_socket_path_is_not_removed(self):
        await self.server.__aexit__()
        path = Path(self.config.socket_path)
        path.write_text('keep this file')
        with self.assertRaisesRegex(ValueError, 'unsafe socket'):
            await SocketServer(self.config, self.router).__aenter__()
        self.assertEqual(path.read_text(), 'keep this file')
        path.unlink()
        path.symlink_to(self.config.password_file)
        with self.assertRaisesRegex(ValueError, 'unsafe socket'):
            await SocketServer(self.config, self.router).__aenter__()
        self.assertTrue(path.is_symlink())

    async def test_stale_socket_is_replaced(self):
        await self.server.__aexit__()
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(self.config.socket_path)
        stale.close()
        self.server = SocketServer(self.config, self.router)
        await self.server.__aenter__()
        await self.connect()
