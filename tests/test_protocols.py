import json
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from loki_agent import formats
from loki_agent import protocols
from loki_agent import sse


class OpenAIResponsesProviderTests(unittest.TestCase):
    def test_responses_provider_derives_endpoint_from_v1_root(self):
        provider = protocols.make_provider(
            "https://api.openai.com/v1",
            provider=protocols.OPENAI_RESPONSES,
            api_key="test-key",
        )

        self.assertEqual(provider.kind, protocols.OPENAI_RESPONSES)
        self.assertEqual(provider.chat_url, "https://api.openai.com/v1/responses")
        self.assertEqual(provider.models_url, "https://api.openai.com/v1/models")
        self.assertIn("https://api.openai.com/v1/models", provider.model_urls)
        self.assertEqual(provider.headers["Authorization"], "Bearer test-key")

    def test_provider_with_empty_key_omits_authentication_header(self):
        provider = protocols.make_provider(
            "http://localhost:8000/v1",
            provider=protocols.OPENAI_RESPONSES,
            api_key="",
        )

        self.assertNotIn("Authorization", provider.headers)
        self.assertNotIn("x-api-key", provider.headers)

    def test_responses_provider_keeps_explicit_endpoint_literal(self):
        provider = protocols.make_provider(
            "https://example.test/prefix/v1/responses?trace=1",
            provider=protocols.AUTO,
            api_key="test-key",
        )

        self.assertEqual(provider.kind, protocols.OPENAI_RESPONSES)
        self.assertEqual(provider.chat_url, "https://example.test/prefix/v1/responses?trace=1")
        self.assertEqual(provider.models_url, "https://example.test/prefix/v1/models")

    def test_responses_payload_uses_responses_wire_format(self):
        provider = protocols.make_provider(
            "https://api.openai.com/v1/responses",
            provider=protocols.OPENAI_RESPONSES,
            api_key="test-key",
            max_tokens=1234,
        )
        items = [
            formats.instruction_item("system marker"),
            formats.message_item("user", "read marker"),
            formats.message_item("assistant", "need file"),
            formats.tool_call_item("call_1", "Read", {"file_path": "README.md"}),
            formats.tool_result_item("call_1", "contents marker"),
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                    "strict": True,
                },
            }
        ]

        payload = provider.chat_payload(items, tools, "gpt-test")

        self.assertEqual(payload["model"], "gpt-test")
        self.assertEqual(payload["instructions"], "system marker")
        self.assertEqual(payload["max_output_tokens"], 1234)
        self.assertEqual(
            [item.get("type") for item in payload["input"]],
            ["message", "message", "function_call", "function_call_output"],
        )
        self.assertEqual(payload["input"][0]["role"], "user")
        self.assertEqual(payload["input"][0]["content"][0]["text"], "read marker")
        self.assertEqual(payload["input"][2]["type"], "function_call")
        self.assertEqual(payload["input"][2]["call_id"], "call_1")
        self.assertEqual(payload["input"][2]["name"], "Read")
        self.assertEqual(payload["input"][3]["type"], "function_call_output")
        self.assertEqual(payload["input"][3]["call_id"], "call_1")
        self.assertEqual(payload["input"][3]["output"], "contents marker")
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["name"], "Read")
        self.assertEqual(payload["tools"][0]["parameters"]["required"], ["file_path"])
        self.assertTrue(payload["tools"][0]["strict"])
        self.assertNotIn("messages", payload)

    def test_responses_payload_omits_tools_when_empty(self):
        provider = protocols.make_provider(
            "https://api.openai.com/v1/responses",
            provider=protocols.OPENAI_RESPONSES,
            api_key="test-key",
        )

        payload = provider.chat_payload([formats.message_item("user", "hi")], [], "gpt-test")

        self.assertNotIn("tools", payload)
        self.assertEqual(payload["input"][0]["content"][0]["text"], "hi")

    def test_responses_parse_response_returns_v2_items(self):
        provider = protocols.make_provider(
            "https://api.openai.com/v1/responses",
            provider=protocols.OPENAI_RESPONSES,
            api_key="test-key",
        )
        response = {
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "model": "gpt-test",
            "output": [
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file_path":"README.md"}',
                }
            ],
        }

        items = provider.parse_chat_response(response)

        self.assertEqual([item.get("type") for item in items], ["response_metadata", "tool_call"])
        self.assertEqual(items[0]["protocol"], protocols.OPENAI_RESPONSES)
        self.assertEqual(items[1]["call_id"], "call_1")
        self.assertEqual(items[1]["name"], "Read")
        self.assertEqual(items[1]["input"], {"file_path": "README.md"})


class AnthropicMessagesProviderTests(unittest.TestCase):
    def test_credentialless_provider_keeps_protocol_version_header(self):
        provider = protocols.make_provider(
            "http://localhost:8000/v1/messages",
            provider=protocols.ANTHROPIC_MESSAGES,
            api_key="",
            anthropic_version="2024-01-01",
        )

        self.assertEqual(
            provider.headers["anthropic-version"], "2024-01-01")
        self.assertNotIn("x-api-key", provider.headers)
        self.assertNotIn("Authorization", provider.headers)


class StreamAccumulatorTests(unittest.TestCase):
    def event(self, data, event="message"):
        return sse.SseEvent(event, data)

    def test_openai_chat_accumulates_text_and_parallel_tool_calls(self):
        deltas = []
        accumulator = protocols.OpenAIChatStreamAccumulator(deltas.append)
        chunks = [
            {
                "id": "chat_1",
                "object": "chat.completion.chunk",
                "model": "local-model",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": None,
                }],
            },
            {
                "id": "chat_1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "lo",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"file_',
                                },
                            },
                            {
                                "index": 1,
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "Glob",
                                    "arguments": '{"pattern":',
                                },
                            },
                        ],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": 'path":"README.md"}',
                                },
                            },
                            {
                                "index": 1,
                                "function": {
                                    "arguments": '"*.py"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }],
            },
        ]
        for chunk in chunks:
            accumulator.feed(self.event(json.dumps(chunk)))
        accumulator.feed(self.event("[DONE]"))

        response = accumulator.finish()
        items = formats.openai_chat_response_to_items(response)

        self.assertEqual(deltas, ["Hel", "lo"])
        self.assertEqual(formats.item_text(items[1]), "Hello")
        self.assertEqual(
            [item["name"] for item in items[2:]], ["Read", "Glob"])
        self.assertEqual(
            items[2]["input"], {"file_path": "README.md"})
        self.assertEqual(items[3]["input"], {"pattern": "*.py"})

    def test_openai_chat_requires_done_marker(self):
        accumulator = protocols.OpenAIChatStreamAccumulator(lambda text: None)
        accumulator.feed(self.event(
            '{"choices":[{"index":0,"delta":{"content":"partial"}}]}'))

        with self.assertRaisesRegex(
                protocols.StreamProtocolError, "before data: \\[DONE\\]"):
            accumulator.finish()

    def test_anthropic_accumulates_text_and_tool_json(self):
        deltas = []
        accumulator = protocols.AnthropicMessagesStreamAccumulator(
            deltas.append)
        events = [
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 4},
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            }),
            ("content_block_stop", {
                "type": "content_block_stop", "index": 0,
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {},
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"file_path":',
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '"README.md"}',
                },
            }),
            ("content_block_stop", {
                "type": "content_block_stop", "index": 1,
            }),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 8},
            }),
            ("message_stop", {"type": "message_stop"}),
        ]
        for event_name, data in events:
            accumulator.feed(self.event(
                json.dumps(data), event=event_name))

        response = accumulator.finish()
        items = formats.anthropic_response_to_items(response)

        self.assertEqual(deltas, ["hello"])
        self.assertEqual(formats.item_text(items[1]), "hello")
        self.assertEqual(items[2]["name"], "Read")
        self.assertEqual(
            items[2]["input"], {"file_path": "README.md"})
        self.assertEqual(
            response["usage"], {"input_tokens": 4, "output_tokens": 8})

    def test_openai_responses_uses_completed_response_as_authority(self):
        deltas = []
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            deltas.append)
        accumulator.feed(self.event(
            '{"type":"response.output_text.delta","delta":"hel"}'))
        accumulator.feed(self.event(
            '{"type":"response.output_text.delta","delta":"lo"}'))
        accumulator.feed(self.event(
            '{"type":"response.completed","response":'
            '{"id":"resp_1","object":"response","status":"completed",'
            '"output":[{"id":"msg_1","type":"message",'
            '"status":"completed","role":"assistant","content":'
            '[{"type":"output_text","text":"hello"}]}]}}'))

        response = accumulator.finish()
        items = formats.openai_responses_response_to_items(response)

        self.assertEqual(deltas, ["hel", "lo"])
        self.assertEqual(formats.item_text(items[1]), "hello")

    def test_streaming_payload_is_opt_in_at_call_site(self):
        provider = protocols.make_provider(
            "http://localhost:8000/v1",
            provider=protocols.OPENAI_CHAT,
            api_key="",
        )
        items = [formats.message_item("user", "hello")]

        self.assertNotIn(
            "stream", provider.chat_payload(items, [], "local-model"))
        self.assertIs(
            provider.streaming_chat_payload(
                items, [], "local-model")["stream"],
            True,
        )


if __name__ == "__main__":
    unittest.main()
