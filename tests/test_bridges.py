import asyncio
from dataclasses import replace
import json
import os
import socket
import tempfile
import unittest

from loki_agent.bridge_sessions import BridgeSession
from loki_agent.bridge_transports import (
    BridgeTransport, Limits, ProtocolError, read_message)
from loki_agent.submissions import Submission


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix sockets required")
class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "bridge.sock")
        self.limits = replace(Limits(), retry_min=0.01, retry_max=0.02,
                              connect_timeout=0.2, write_timeout=0.2)
        self.connections = asyncio.Queue()
        self.reports = []
        self.sessions = []
        self.server = None
        self.writers = []

    async def accept(self, reader, writer):
        self.writers.append(writer)
        try:
            hello = await read_message(reader, self.limits)
        except ConnectionError:
            return  # peer-validation tests close before sending a handshake
        self.assertEqual(hello, {"type": "hello", "version": 1})
        writer.write(b'{"type":"hello","version":1}\n')
        await writer.drain()
        registration = await read_message(reader, self.limits)
        await self.connections.put((reader, writer, registration))

    async def start_server(self):
        self.server = await asyncio.start_unix_server(
            self.accept, path=self.path)

    async def start_session(self, limits=None):
        queue = asyncio.Queue()
        session = BridgeSession(
            "conversation", frontend="terminal", enqueue=queue.put_nowait,
            path=self.path,
            limits=limits or self.limits, report=self.reports.append)
        await session.__aenter__()
        self.sessions.append(session)
        return session, queue

    async def asyncTearDown(self):
        for session in self.sessions:
            await session.__aexit__()
        for writer in self.writers:
            writer.close()
            await writer.wait_closed()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.directory.cleanup()

    async def connection(self):
        return await asyncio.wait_for(self.connections.get(), 1)

    def submit(self, session, input_id="input-1", text="hello", **extra):
        session.receive({"type": "submit_prompt",
                         "instance_id": session.instance_id,
                         "input_id": input_id, "text": text, **extra})

    async def test_session_capabilities_do_not_require_an_input_adapter(self):
        session = BridgeSession('conversation', frontend='terminal',
                                path=self.path, peer_uid=os.getuid())
        self.assertFalse(session.registration()[0]['capabilities']['accepts_prompts'])
        self.submit(session)
        self.assertEqual(session.journal[-1][0]['reason'], 'input_not_supported')
        submission = Submission('keyboard')
        turn_id = session.turn_started(submission)
        session.turn_finished(submission, turn_id, 'completed', 'answer')
        self.assertEqual(session.journal[-1][0]['type'], 'turn_finished')

    async def test_multiple_connections_are_independently_addressed(self):
        await self.start_server()
        first, first_queue = await self.start_session()
        second, second_queue = await self.start_session()
        connections = [await self.connection(), await self.connection()]
        self.assertEqual({value[2]["instance_id"] for value in connections},
                         {first.instance_id, second.instance_id})
        self.submit(first, instance_id=second.instance_id)
        self.assertTrue(first_queue.empty())
        self.submit(second)
        self.assertEqual((await second_queue.get()).origin, "bridge")
        self.assertNotEqual(first.instance_id, second.instance_id)

    async def test_absent_bridge_recovers_with_registration(self):
        session, _queue = await self.start_session()
        await asyncio.sleep(0.04)
        self.assertEqual(len(self.reports), 1)
        session.publish("session_state", paused=False)
        await self.start_server()
        reader, _writer, registration = await self.connection()
        self.assertEqual(registration["instance_id"], session.instance_id)
        event = await read_message(reader, self.limits)
        self.assertEqual(event["type"], "session_state")
        self.assertEqual(self.reports[-1], "Bridge connection restored.")

    async def test_deduplication_conflicts_and_limits(self):
        session, queue = await self.start_session(
            replace(self.limits, pending_inputs=1, prompt_bytes=8))
        self.submit(session)
        self.submit(session)
        self.assertEqual(queue.qsize(), 1)
        self.assertTrue(session.journal[-1][0]["duplicate"])
        self.submit(session, text="other")
        self.assertEqual(session.journal[-1][0]["reason"], "input_id_conflict")
        self.submit(session, input_id="input-2")
        self.assertEqual(session.journal[-1][0]["reason"], "queue_full")
        self.submit(session, text="a" * 9)
        self.assertEqual(session.journal[-1][0]["reason"], "prompt_too_large")
        submission = await queue.get()
        turn_id = session.turn_started(submission)
        session.turn_finished(submission, turn_id, "completed", "result")
        self.submit(session)
        self.assertTrue(queue.empty())
        self.assertEqual(session.journal[-1][0]["status"], "finished")

    async def test_reconnect_replays_unacknowledged_events(self):
        await self.start_server()
        session, queue = await self.start_session()
        reader, writer, _registration = await self.connection()
        self.submit(session)
        accepted = await read_message(reader, self.limits)
        writer.write((json.dumps({"type": "ack",
                                  "instance_id": session.instance_id,
                                  "event_seq": accepted["event_seq"]})
                      + "\n").encode())
        await writer.drain()
        while session.acknowledged != accepted["event_seq"]:
            await asyncio.sleep(0.001)
        writer.close()
        await writer.wait_closed()
        submission = await queue.get()
        turn_id = session.turn_started(submission)
        session.turn_finished(submission, turn_id, "completed", "done")
        reader, _writer, registration = await self.connection()
        self.assertEqual(registration["instance_id"], session.instance_id)
        events = [await read_message(reader, self.limits) for _ in range(4)]
        self.assertEqual([event["type"] for event in events],
                         ["turn_started", "input_chunk", "output_chunk",
                          "turn_finished"])

    async def test_gap_chunking_and_completed_record_retention(self):
        session, queue = await self.start_session(
            replace(self.limits, events=3, completed_inputs=1))
        self.submit(session)
        submission = await queue.get()
        turn_id = session.turn_started(submission)
        session.turn_finished(submission, turn_id, "completed", "x" * 10000)
        messages = session.registration()
        self.assertEqual(messages[1]["type"], "event_gap")
        self.assertEqual(messages[-1]["type"], "turn_finished")
        self.submit(session, "input-2")
        session.command_finished(await queue.get(), "completed", "ok")
        self.assertNotIn("input-1", session.inputs)

    async def test_cancel_pauses_remote_inputs_until_resume(self):
        session, queue = await self.start_session()
        self.submit(session)
        submission = await queue.get()
        turn_id = session.turn_started(submission)
        session.turn_finished(submission, turn_id, "cancelled", "partial")
        self.submit(session, "input-2")
        self.assertTrue(queue.empty())
        self.assertEqual(session.journal[-1][0]["reason"],
                         "remote_execution_paused")
        self.submit(session, "resume", "/bridge resume")
        self.assertEqual((await queue.get()).text, "/bridge resume")
        session.resume()
        self.submit(session, "input-3")
        self.assertFalse(queue.empty())

    async def test_outbound_overflow_disconnects_without_blocking_execution(self):
        await self.start_server()
        session, _queue = await self.start_session()
        await self.connection()
        transport = session.transport
        self.assertFalse(os.get_inheritable(
            transport._writer.get_extra_info('socket').fileno()))
        # A synchronous burst fills the writer queue without yielding to it.
        for _ in range(self.limits.events + 1):
            session.publish('session_state', paused=False)
        self.assertFalse(transport.connected)
        self.assertLessEqual(len(session.journal), self.limits.events)
        _reader, _writer, registration = await self.connection()
        self.assertEqual(registration['instance_id'], session.instance_id)
        self.assertGreater(registration['replay_from'], 1)

    async def test_events_published_during_registration_are_not_lost(self):
        await self.start_server()
        session, _queue = await self.start_session()
        # Delay the registration write while the frontend publishes an event.
        original = session.transport._write
        registered = asyncio.Event()
        release = asyncio.Event()

        async def write(writer, data):
            if b'session_registered' in data:
                registered.set()
                await release.wait()
            await original(writer, data)

        session.transport._write = write
        await asyncio.wait_for(registered.wait(), 1)
        session.publish('session_state', paused=True)
        release.set()
        reader, _writer, _registration = await self.connection()
        self.assertEqual((await read_message(reader, self.limits))['paused'], True)

    async def test_invalid_handshake_retries_and_warns_once(self):
        async def bad_server(reader, writer):
            self.writers.append(writer)
            await read_message(reader, self.limits)
            writer.write(b'{"type":"hello","version":true}\n')
            await writer.drain()

        self.server = await asyncio.start_unix_server(bad_server, path=self.path)
        session, _queue = await self.start_session()
        await asyncio.sleep(0.05)
        self.assertFalse(session.transport.connected)
        self.assertEqual(len(self.reports), 1)
        self.assertIn('handshake', self.reports[0])

    async def test_peer_uid_validation(self):
        if not hasattr(socket, 'SO_PEERCRED'):
            self.skipTest('SO_PEERCRED unavailable')
        await self.start_server()
        transport = BridgeTransport(
            self.path, register=lambda: [], receive=lambda message: None,
            peer_uid=os.getuid() + 1, limits=self.limits,
            report=self.reports.append)
        # Peer validation occurs before any protocol or session data is sent.
        reader, writer = await asyncio.open_unix_connection(self.path)
        try:
            with self.assertRaisesRegex(ProtocolError, 'peer UID'):
                transport._check_peer(writer)
        finally:
            writer.close()
            await writer.wait_closed()

    async def test_frame_validation(self):
        for data in [b'{}\n', b'[]\n', b'broken\n', b'{}',
                     b'{"type":"x","value":NaN}\n']:
            reader = asyncio.StreamReader()
            reader.feed_data(data)
            reader.feed_eof()
            with self.assertRaises(ProtocolError):
                await read_message(reader, self.limits)
        reader = asyncio.StreamReader(limit=32)
        reader.feed_data(b'x' * 64 + b'\n')
        with self.assertRaises(ProtocolError):
            await read_message(reader, replace(self.limits, frame_bytes=32))

    async def test_shutdown_reports_queued_inputs_and_closes_tasks(self):
        session, queue = await self.start_session()
        self.submit(session)
        await session.__aexit__()
        self.assertEqual(session.inputs["input-1"].outcome, "unexecuted")
        self.assertIsNone(session.transport._task)
        self.sessions.remove(session)
        self.assertEqual(queue.qsize(), 1)  # frontend owns disposal


class SubmissionTests(unittest.TestCase):
    def test_keyboard_normalization_preserves_envelopes(self):
        keyboard = Submission.normalize("text")
        self.assertEqual(keyboard.text, "text")
        self.assertEqual(keyboard.origin, "keyboard")
        self.assertIs(Submission.normalize(keyboard), keyboard)
