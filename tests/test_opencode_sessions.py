import asyncio
import json
import types
import unittest
from unittest import mock


from loki_agent import http_client
from loki_agent import formats
from loki_agent import loki
from loki_agent import protocols
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
        url = "https://opencode.ai/zen/go/v1/chat/completions"
        config = types.SimpleNamespace(
            chat_provider=types.SimpleNamespace(chat_url=url))

        with self.assertRaisesRegex(
                ValueError, "require a conversation identity"):
            loki._opencode_session_id_for_request(config, url)

    def test_buffered_request_installs_authoritative_session_header(self):
        config = self._config(
            "https://opencode.ai/zen/go/v1/chat/completions")
        requests = []

        async def request(method, url, **kwargs):
            requests.append((method, url, kwargs))
            return http_client.HttpResponse(
                url, 200, "OK",
                {"content-type": "application/json"},
                b"{}",
            )

        with mock.patch.object(
                http_client, "async_http_request", new=request):
            asyncio.run(loki.async_provider_request(
                "POST",
                config.chat_provider.chat_url,
                {},
                request_headers={"X-OpenCode-Session": "untrusted"},
                opencode_session_id="owned-conversation",
            ))

        headers = requests[0][2]["headers_in"]
        self.assertEqual(
            headers[loki.OPENCODE_SESSION_HEADER],
            "owned-conversation",
        )
        self.assertNotIn("X-OpenCode-Session", headers)

    def test_streaming_request_installs_authoritative_session_header(self):
        config = self._config(
            "https://opencode.ai/zen/go/v1/chat/completions",
            stream=True,
        )
        requests = []

        async def request_once(
                url, payload, headers, on_text_delta, cancel_check,
                codex_turn_state=None):
            requests.append(dict(headers))
            return protocols.ProviderResponse({})

        with mock.patch.object(
                loki, "_async_chat_stream_request_once",
                new=request_once):
            asyncio.run(loki.async_chat_stream_request(
                config.chat_provider.chat_url,
                {},
                request_headers={"X-OPENCODE-SESSION": "untrusted"},
                opencode_session_id="owned-conversation",
            ))

        self.assertEqual(
            requests[0][loki.OPENCODE_SESSION_HEADER],
            "owned-conversation",
        )
        self.assertNotIn("X-OPENCODE-SESSION", requests[0])

    def test_completion_passes_one_conversation_id_to_both_transports(self):
        response = protocols.ProviderResponse({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "ok",
                },
            }],
        })
        for stream, function_name in [
                (False, "async_provider_request"),
                (True, "async_chat_stream_request")]:
            with self.subTest(stream=stream):
                self._config(
                    "https://opencode.ai/zen/go/v1/chat/completions",
                    stream=stream,
                )
                loki.current_session().conversation_id = "conversation"
                request = mock.AsyncMock(return_value=response)
                with mock.patch.object(loki, function_name, new=request):
                    asyncio.run(loki.async_chat_completion(
                        [formats.message_item("user", "hello")],
                        tools=[],
                    ))

                self.assertEqual(
                    request.await_args.kwargs["opencode_session_id"],
                    "conversation",
                )

    def test_non_go_and_model_requests_do_not_receive_session_header(self):
        config = self._config("https://opencode.ai/zen/v1/chat/completions")
        requests = []

        async def request(method, url, **kwargs):
            requests.append((method, url, kwargs))
            return http_client.HttpResponse(
                url, 200, "OK",
                {"content-type": "application/json"},
                json.dumps({"data": []}).encode("utf-8"),
            )

        with mock.patch.object(
                http_client, "async_http_request", new=request):
            asyncio.run(loki.async_provider_request(
                "POST",
                config.chat_provider.chat_url,
                {},
                opencode_session_id="must-not-leak",
            ))

            go_config = self._config(
                "https://opencode.ai/zen/go/v1/chat/completions")
            asyncio.run(loki.async_provider_request(
                "GET",
                go_config.chat_provider.models_url,
                opencode_session_id="must-not-leak",
            ))

        for _method, _url, kwargs in requests:
            self.assertNotIn(
                loki.OPENCODE_SESSION_HEADER,
                {
                    str(name).lower(): value
                    for name, value in kwargs["headers_in"].items()
                },
            )


if __name__ == "__main__":
    unittest.main()
