import asyncio
import json
import os
import tempfile
import types
import unittest
from unittest import mock
from response_header_fixtures import setUpModule  # noqa: F401 - unittest hook
from test_http_client import FakeConnector, PatchedOpenConnection

from loki_agent import formats
from loki_agent import loki
from loki_agent import protocols
from loki_agent.credentials import CredentialStore
from loki_agent.sessions import Session


class OpenCodeSessionHeaderTests(unittest.TestCase):
    def setUp(self):
        self.previous_session = loki._DEFAULT_SESSION
        loki._DEFAULT_SESSION = Session()

    def tearDown(self):
        loki._DEFAULT_SESSION = self.previous_session

    @staticmethod
    def _config(url, *, stream=False):
        config = loki.make_runtime_config(
            url,
            protocols.OPENAI_CHAT,
            model="test-model",
            stream=stream,
        )
        loki.apply_runtime_config(config)
        return config

    @staticmethod
    def _response(answer, *, stream=False):
        # External response bytes only: application decoding stays real.
        if stream:
            chunks = []
            for content in (answer[:3], answer[3:]):
                chunks.append('data: ' + json.dumps({
                    'object': 'chat.completion.chunk',
                    'choices': [{'index': 0, 'delta': {'content': content},
                                 'finish_reason': None}]}) + '\n\n')
            chunks.append('data: ' + json.dumps({
                'object': 'chat.completion.chunk',
                'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}) + '\n\n')
            body = ''.join(chunks).encode() + b'data: [DONE]\n\n'
            content_type = b'text/event-stream'
        else:
            body = json.dumps({'choices': [{
                'index': 0, 'message': {'role': 'assistant', 'content': answer},
                'finish_reason': 'stop'}]}).encode()
            content_type = b'application/json'
        return (b'HTTP/1.1 200 OK\r\nContent-Type: ' + content_type
                + b'\r\nContent-Length: ' + str(len(body)).encode()
                + b'\r\n\r\n' + body)

    @staticmethod
    def _pairs(events):
        pairs = []
        for event in events:
            items = event.get('items', []) if event['type'] == 'model_response' else [event]
            for item in items:
                if item.get('type') == 'message' and item.get('role') in ('user', 'assistant'):
                    pairs.append((item['role'], formats.item_text(item)))
        return pairs

    def _assert_transport_closed(self, connector):
        for writer in connector.writers:
            self.assertTrue(writer.closed)
            self.assertTrue(writer.wait_closed_called)

    def test_recognizes_only_canonical_go_inference_targets(self):
        session_id = "conversation"
        accepted = [
            "https://opencode.ai/zen/go/v1/chat/completions",
            "https://opencode.ai/zen/go/v1/messages/",
            "https://opencode.ai:443/zen/go/v1/responses",
        ]
        rejected = [
            "http://opencode.ai/zen/go/v1/chat/completions",
            "https://opencode.ai:444/zen/go/v1/chat/completions",
            "https://opencode.ai.evil/zen/go/v1/chat/completions",
            "https://opencode.ai/zen/v1/chat/completions",
            "https://opencode.ai/zen/go/v1/models",
        ]
        for url in accepted:
            with self.subTest(url=url):
                config = types.SimpleNamespace(
                    chat_provider=types.SimpleNamespace(chat_url=url))
                self.assertEqual(
                    loki._opencode_session_id_for_request(
                        config, url, session_id),
                    session_id,
                )
        for url in rejected:
            with self.subTest(url=url):
                config = types.SimpleNamespace(
                    chat_provider=types.SimpleNamespace(chat_url=url))
                self.assertIsNone(
                    loki._opencode_session_id_for_request(
                        config, url, session_id))

    def test_canonical_target_requires_a_conversation_identity(self):
        async def scenario():
            async with asyncio.timeout(5):
                connector = FakeConnector([])
                with PatchedOpenConnection(connector):
                    for stream in (False, True):
                        config = self._config(
                            'https://opencode.ai/zen/go/v1/chat/completions', stream=stream)
                        with self.assertRaisesRegex(ValueError, 'require a conversation identity'):
                            loki._opencode_session_id_for_request(config, config.chat_provider.chat_url)
                        for invalid in (None, '', 0, 1, False, True, [], ['id'], {}, {'id': 'x'}):
                            with self.subTest(stream=stream, identity=invalid):
                                loki.current_session().conversation_id = invalid
                                with self.assertRaisesRegex(ValueError, 'require a conversation identity'):
                                    await loki.async_chat_completion(
                                        [formats.message_item('user', 'must not send')], tools=[])
                                self.assertIs(loki.current_session().runtime_config, config)
                                self.assertEqual(connector.calls, [])
                                self.assertEqual(connector.writers, [])
        asyncio.run(scenario())

    def test_conversation_identity_survives_resume_and_switch_on_both_transports(self):
        async def scenario(directory, stream):
            async with asyncio.timeout(10):
                tasks_before = asyncio.all_tasks()
                url = 'https://opencode.ai/zen/go/v1/chat/completions'
                identity_a = '11111111-1111-4111-8111-111111111111'
                identity_b = '22222222-2222-4222-8222-222222222222'
                path_a = os.path.join(directory, f'chat-{identity_a}.json')
                path_b = os.path.join(directory, f'chat-{identity_b}.json')
                answers = ('first A answer', 'resumed A answer', 'unrelated B answer')
                connector = FakeConnector([self._response(answer, stream=stream) for answer in answers])
                hostile = {'X-OpenCode-Session': 'untrusted-mixed',
                           'X-OPENCODE-SESSION': 'untrusted-upper'}
                header_sources = []
                with PatchedOpenConnection(connector):
                    try:
                        self._config(url, stream=stream)
                        loki.new_chat_log(path_a)
                        first_session = loki.current_session()
                        self.assertEqual(first_session.conversation_id, identity_a)
                        saved_a = None
                        first_bytes = None
                        resumed_bytes = None
                        expected_a = []

                        def complete(items, on_text_delta=None, *, codex_turn_state):
                            return loki.async_chat_completion(
                                items, tools=[], on_text_delta=on_text_delta,
                                codex_turn_state=codex_turn_state)
                        for index, (prompt, answer) in enumerate(zip(
                                ('first A prompt', 'resumed A prompt', 'unrelated B prompt'), answers)):
                            if index == 1:
                                resumed = Session(shell_cwd=directory)
                                self.assertIsNot(resumed, first_session)
                                self.assertNotEqual(resumed.conversation_id, identity_a)
                                loki._DEFAULT_SESSION = resumed
                                loki.load_chat_log(path_a, apply_shell_cwd=False)
                                self.assertEqual(resumed.conversation_id, identity_a)
                                self.assertEqual(resumed.transcript_items, saved_a['events'])
                                self.assertEqual(resumed.session_state, saved_a['session_state'])
                                descriptor = loki.connection_from_session_state(resumed.session_state)
                                loki.apply_runtime_config(loki.config_from_connection_descriptor(
                                    descriptor, loki.CREDENTIALS))
                                with open(path_a, 'rb') as saved:
                                    self.assertEqual(saved.read(), first_bytes)
                            elif index == 2:
                                unrelated = Session(shell_cwd=directory)
                                self.assertIsNot(unrelated, resumed)
                                loki._DEFAULT_SESSION = unrelated
                                self._config(url, stream=stream)
                                loki.new_chat_log(path_b)
                                self.assertEqual(unrelated.conversation_id, identity_b)
                            session = loki.current_session()
                            identity = identity_b if index == 2 else identity_a
                            path = path_b if index == 2 else path_a
                            self.assertIs(session.runtime_config.stream, stream)
                            source = session.runtime_config.chat_provider.headers
                            source.update(hostile)
                            original_headers = dict(source)
                            header_sources.append((source, original_headers))
                            history = [] if index == 2 else list(expected_a)
                            session.transcript_items.append(formats.message_item('user', prompt))
                            events = []
                            result = await loki.run_tool_loop_async(
                                session.transcript_items, allowed=set(), max_loops=2,
                                chat_fn=complete, stream_chat=stream, on_event=events.append)
                            self.assertEqual(result, answer)
                            self.assertFalse(any(event['type'] == 'provider_error' for event in events))
                            if stream:
                                self.assertEqual([event['content'] for event in events
                                                  if event['type'] == 'assistant_delta'],
                                                 [answer[:3], answer[3:]])
                            loki.mark_chat_log_dirty()
                            self.assertTrue(loki.save_chat_log())
                            self.assertEqual(session.conversation_id, identity)
                            self.assertEqual(len(connector.writers), index + 1)
                            packet = bytes(connector.writers[index].data)
                            headers, body = packet.split(b'\r\n\r\n', 1)
                            self.assertTrue(headers.startswith(b'POST /zen/go/v1/chat/completions HTTP/1.1\r\n'))
                            identity_headers = [line for line in headers.split(b'\r\n')
                                                if line.partition(b':')[0].lower() == b'x-opencode-session']
                            self.assertEqual(identity_headers, [b'x-opencode-session: ' + identity.encode()])
                            for marker in hostile.values():
                                self.assertNotIn(marker.encode(), packet)
                            payload = json.loads(body)
                            self.assertEqual(payload['model'], 'test-model')
                            if stream:
                                self.assertIs(payload['stream'], True)
                            else:
                                self.assertNotIn('stream', payload)
                            self.assertEqual([(message['role'], message['content'])
                                              for message in payload['messages']
                                              if message['role'] in ('user', 'assistant')],
                                             history + [('user', prompt)])
                            expected = history + [('user', prompt), ('assistant', answer)]
                            with open(path, 'rb') as saved:
                                durable = saved.read()
                            blob = json.loads(durable)
                            formats.validate_events(blob['events'])
                            self.assertEqual(self._pairs(blob['events']), expected)
                            self.assertNotIn('conversation_id', blob['session_state'])
                            self.assertEqual(blob['session_state']['shell_cwd'], directory)
                            descriptor = blob['session_state']['connection']
                            self.assertEqual(descriptor['chat_url'], url)
                            self.assertEqual(descriptor['model'], 'test-model')
                            self.assertEqual(descriptor['protocol'], 'openai_chat')
                            self.assertIs(descriptor['stream'], stream)
                            self.assertIsNone(descriptor['credential'])
                            if index == 0:
                                saved_a, first_bytes = blob, durable
                            elif index == 1:
                                self.assertEqual(blob['events'][:len(saved_a['events'])], saved_a['events'])
                                resumed_bytes = durable
                            else:
                                with open(path_a, 'rb') as saved:
                                    self.assertEqual(saved.read(), resumed_bytes)
                            if index != 2:
                                expected_a = expected
                            for source, original_headers in header_sources:
                                self.assertEqual(source, original_headers)
                            self._assert_transport_closed(connector)
                        self.assertEqual([call['host'] for call in connector.calls], ['opencode.ai'] * 3)
                        self.assertEqual(connector.responses, [])
                        self.assertFalse(asyncio.all_tasks() - tasks_before)
                    finally:
                        # Backstops cannot satisfy transport-release assertions.
                        for writer in connector.writers:
                            if not writer.closed:
                                writer.close()
                                await writer.wait_closed()

        for stream in (False, True):
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as directory:
                with mock.patch.object(loki, '_DEFAULT_SESSION', Session(shell_cwd=directory)), \
                        mock.patch.object(loki, 'CREDENTIALS', CredentialStore({})):
                    asyncio.run(scenario(directory, stream))

    def test_non_go_and_model_requests_do_not_receive_session_header(self):
        async def scenario():
            async with asyncio.timeout(10):
                canonical = 'https://opencode.ai/zen/go/v1/chat/completions'
                non_go = 'https://opencode.ai/zen/v1/chat/completions'
                mismatch = 'https://opencode.ai/zen/go/v1/responses'
                for stream in (False, True):
                    for configured, requested in ((non_go, non_go), (canonical, mismatch)):
                        with self.subTest(stream=stream, configured=configured, requested=requested):
                            self._config(configured, stream=stream)
                            connector = FakeConnector([self._response('boundary answer', stream=stream)])
                            with PatchedOpenConnection(connector):
                                try:
                                    kwargs = {'opencode_session_id': loki.current_session().conversation_id}
                                    if stream:
                                        await loki.async_chat_stream_request(requested, {}, **kwargs)
                                    else:
                                        await loki.async_provider_request('POST', requested, {}, **kwargs)
                                    self.assertEqual(len(connector.writers), 1)
                                    headers = bytes(connector.writers[0].data).split(b'\r\n\r\n', 1)[0]
                                    self.assertFalse(any(line.partition(b':')[0].lower() == b'x-opencode-session'
                                                         for line in headers.split(b'\r\n')))
                                    self._assert_transport_closed(connector)
                                finally:
                                    for writer in connector.writers:
                                        if not writer.closed:
                                            writer.close()
                                            await writer.wait_closed()
                config = self._config(canonical)
                body = b'{"data": []}'
                connector = FakeConnector([b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '
                                           + str(len(body)).encode() + b'\r\n\r\n' + body])
                with PatchedOpenConnection(connector):
                    try:
                        response = await loki.async_provider_request(
                            'GET', config.chat_provider.models_url,
                            opencode_session_id=loki.current_session().conversation_id)
                        self.assertEqual(response.payload, {'data': []})
                        self.assertEqual(len(connector.writers), 1)
                        headers = bytes(connector.writers[0].data).split(b'\r\n\r\n', 1)[0]
                        self.assertTrue(headers.startswith(b'GET /zen/go/v1/models HTTP/1.1\r\n'))
                        self.assertFalse(any(line.partition(b':')[0].lower() == b'x-opencode-session'
                                             for line in headers.split(b'\r\n')))
                        self._assert_transport_closed(connector)
                    finally:
                        for writer in connector.writers:
                            if not writer.closed:
                                writer.close()
                                await writer.wait_closed()
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
