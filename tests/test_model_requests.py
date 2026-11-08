"""An outstanding request keeps its selected connection across awaits."""

import asyncio
import contextlib
import os
import tempfile
import time
import types
import unittest
from unittest import mock

from process_lifecycle_fixtures import observe_fixture
from loki_agent import connections, formats, http_client, loki, model_requests, protocols, recoveries, response_headers
from loki_agent.sessions import Session


class DummyStreamGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_controller_release_is_not_raced_by_an_elapsed_gate_watchdog(self):
        await self._gate_journey('release')

    async def test_cancel_predicate_retires_the_held_dummy_request(self):
        await self._gate_journey('predicate')

    async def test_task_cancellation_retires_the_held_dummy_request(self):
        await self._gate_journey('task')

    async def test_cancellation_wins_simultaneous_gate_release(self):
        await self._gate_journey('release-and-cancel')

    async def _gate_journey(self, outcome):
        # Retain the actual request while a module-local clock jumps past the
        # old cutoff. The event loop's clock remains real. Each refusal path is
        # separately runnable and owns its request join before Store/directory
        # cleanup; no readiness-before-an-elapsed-duration assertion is needed.
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as cleanup:
            gate = os.path.join(directory, 'release')
            selected = connections.make_runtime_config(
                'http://dummy.invalid/v1', protocols.DUMMY, model='dummy', stream=True)
            session = Session(runtime_config=selected, response_headers=response_headers.Store(
                os.path.join(directory, 'observations.json')))
            cleanup.callback(session.response_headers.close)
            first = asyncio.Event()
            deltas = []
            cancelled = False

            def delta(text):
                deltas.append(text)
                first.set()

            clock = types.SimpleNamespace(**vars(time))
            clock.monotonic = mock.Mock(side_effect=[0, 1000])
            with mock.patch.object(loki, '_DEFAULT_SESSION', session), \
                    mock.patch.object(model_requests, 'time', clock), \
                    mock.patch.dict(os.environ, {
                        'LOKI_DUMMY_STREAM_CHUNKS': '["held", "released"]',
                        'LOKI_DUMMY_STREAM_GATE': gate, 'LOKI_DUMMY_REPLY': 'complete',
                        'LOKI_DUMMY_TOOL_CALL': '',
                    }):
                operation = asyncio.create_task(loki.async_chat_completion(
                    [formats.message_item('user', 'question')], tools=[],
                    on_text_delta=delta, cancel_check=lambda: cancelled))
                try:
                    await observe_fixture(first.wait(), operation)
                    self.assertFalse(operation.done())
                    self.assertEqual(deltas, ['held'])
                    if outcome in ['predicate', 'release-and-cancel']:
                        cancelled = True
                    if outcome in ['release', 'release-and-cancel']:
                        with open(gate, 'a'):
                            pass
                    if outcome == 'task':
                        operation.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await operation
                    elif outcome != 'release':
                        with self.assertRaises(recoveries.Cancelled):
                            await operation
                    else:
                        turn = await operation
                        self.assertEqual(formats.item_text(turn.items[0]), 'complete')
                        self.assertEqual(deltas, ['held', 'released'])
                    if outcome != 'release':
                        self.assertEqual(deltas, ['held'])
                    self.assertTrue(operation.done())
                finally:
                    if not operation.done():
                        operation.cancel()
                    await asyncio.gather(operation, return_exceptions=True)


class ModelRequestSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_parser_and_metadata_keep_the_admitted_connection(self):
        selected = connections.make_runtime_config(
            'https://first.test/v1', protocols.OPENAI_CHAT, model='first', stream=True)
        replacement = connections.make_runtime_config(
            'https://second.test/v1', protocols.ANTHROPIC_MESSAGES, model='second')
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as cleanup:
            session = Session(runtime_config=selected, response_headers=response_headers.Store(
                directory + '/observations.json'))
            cleanup.callback(session.response_headers.close)

            async def body():
                # This await boundary is before stream-accumulator construction.
                # A later selection must not change how the outstanding reply
                # is parsed or which connection its observations describe.
                session.runtime_config = replacement
                yield (b'data: {"choices":[{"delta":{"content":"answer"},'
                       b'"finish_reason":null}]}\n\n'
                       b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                       b'data: [DONE]\n\n')

            @contextlib.asynccontextmanager
            async def stream(method, url, **kwargs):
                self.assertEqual(method, 'POST')
                self.assertEqual(url, selected.chat_provider.chat_url)
                headers = {'content-type': 'text/event-stream'}
                yield types.SimpleNamespace(
                    status=200, headers=headers, body=body(),
                    header=lambda name: headers.get(name.lower(), ''))

            with mock.patch.object(loki, '_DEFAULT_SESSION', session), \
                    mock.patch.object(http_client, 'async_http_stream', stream):
                turn = await loki.async_chat_completion(
                    [formats.message_item('user', 'question')], tools=[])
            self.assertEqual(''.join(formats.item_text(item) for item in turn.items), 'answer')
            self.assertEqual(turn.metadata['protocol'], protocols.OPENAI_CHAT)
            self.assertEqual(turn.metadata['endpoint'], selected.chat_provider.chat_url)
            self.assertEqual(turn.metadata['requested_model'], 'first')
            self.assertIs(session.runtime_config, replacement)
            observations = session.response_headers.snapshot()['endpoints']
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0]['endpoint'], selected.chat_provider.chat_url)
            self.assertEqual(observations[0]['latest']['model'], 'first')
