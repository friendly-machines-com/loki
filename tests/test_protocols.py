import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import formats
import protocols


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


if __name__ == "__main__":
    unittest.main()
