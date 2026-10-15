"""An outstanding request keeps its selected connection across awaits."""

import contextlib
import tempfile
import types
import unittest
from unittest import mock

from loki_agent import connections, formats, http_client, loki, protocols, response_headers
from loki_agent.sessions import Session


class ModelRequestSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_parser_and_metadata_keep_the_admitted_connection(self):
        selected = connections.make_runtime_config(
            'https://first.test/v1', protocols.OPENAI_CHAT, model='first', stream=True)
        replacement = connections.make_runtime_config(
            'https://second.test/v1', protocols.ANTHROPIC_MESSAGES, model='second')
        with tempfile.TemporaryDirectory() as directory:
            session = Session(runtime_config=selected, response_headers=response_headers.Store(
                directory + '/observations.json'))

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
