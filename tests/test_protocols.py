import contextlib
import io
import json
import unittest

from loki_agent import formats
from loki_agent import openai_models
from loki_agent import protocols
from loki_agent import sse


def _codex_model(slug="gpt-test", **overrides):
    value = {
        "slug": slug,
        "display_name": slug,
        "visibility": "list",
        "input_modalities": ["text", "image"],
        "supported_reasoning_levels": [
            {"effort": "high", "description": "High"},
        ],
        "default_reasoning_level": "high",
        "supports_reasoning_summaries": True,
        "default_reasoning_summary": "detailed",
        "support_verbosity": True,
        "default_verbosity": "medium",
        "supports_parallel_tool_calls": True,
        "shell_type": "shell_command",
    }
    value.update(overrides)
    return openai_models.CodexModelRequestProfile.from_catalog_model(value)


class OpenAIResponsesProviderTests(unittest.TestCase):
    def test_codex_request_profile_ignores_non_request_model_fields(
            self):
        profile = _codex_model(
            tool_mode="code_mode_only",
            service_tiers={"malformed": "and ignored"},
            context_window={"malformed": "and ignored"},
            truncation_policy="malformed but ignored",
            base_instructions="Codex application prompt",
            model_messages={"instructions_template": "large UI prompt"},
            server_extension={"future": True},
        )

        encoded = profile.to_dict()
        decoded = openai_models.CodexModelRequestProfile.from_dict(encoded)

        self.assertEqual(decoded, profile)
        self.assertEqual(set(encoded), {
            "use_responses_lite",
            "supports_parallel_tool_calls",
            "supports_reasoning_summaries",
            "default_reasoning_level",
            "default_reasoning_summary",
            "supports_verbosity",
            "default_verbosity",
        })
        self.assertNotIn("base_instructions", encoded)
        self.assertNotIn("model_messages", encoded)
        self.assertNotIn("server_extension", encoded)

    def test_responses_provider_derives_endpoint_from_v1_root(self):
        provider = protocols.make_provider(
            "https://api.openai.com/v1",
            provider=protocols.OPENAI_RESPONSES,
        )

        self.assertEqual(provider.kind, protocols.OPENAI_RESPONSES)
        self.assertEqual(provider.chat_url, "https://api.openai.com/v1/responses")
        self.assertEqual(provider.models_url, "https://api.openai.com/v1/models")
        self.assertIn("https://api.openai.com/v1/models", provider.model_urls)
        self.assertNotIn("Authorization", provider.headers)

    def test_provider_with_empty_key_omits_authentication_header(self):
        provider = protocols.make_provider(
            "http://localhost:8000/v1",
            provider=protocols.OPENAI_RESPONSES,
        )

        self.assertNotIn("Authorization", provider.headers)
        self.assertNotIn("x-api-key", provider.headers)

    def test_responses_provider_keeps_explicit_endpoint_literal(self):
        provider = protocols.make_provider(
            "https://example.test/prefix/v1/responses?trace=1",
            provider=protocols.AUTO,
        )

        self.assertEqual(provider.kind, protocols.OPENAI_RESPONSES)
        self.assertEqual(provider.chat_url, "https://example.test/prefix/v1/responses?trace=1")
        self.assertEqual(provider.models_url, "https://example.test/prefix/v1/models")

    def test_responses_payload_uses_responses_wire_format(self):
        provider = protocols.make_provider(
            "https://api.openai.com/v1/responses",
            provider=protocols.OPENAI_RESPONSES,
            max_tokens=1234,
        )
        items = [
            formats.instruction_item("system marker"),
            formats.message_item("user", "read marker"),
            formats.model_response_event(
                protocols.OPENAI_CHAT,
                [
                    formats.message_item("assistant", "need file"),
                    formats.tool_call_item(
                        "call_1", "Read",
                        {"file_path": "README.md"}),
                ],
            ),
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
        )

        payload = provider.chat_payload([formats.message_item("user", "hi")], [], "gpt-test")

        self.assertNotIn("tools", payload)
        self.assertEqual(payload["input"][0]["content"][0]["text"], "hi")

    def test_subscription_payload_uses_codex_backend_contract(self):
        provider = protocols.make_provider(
            "https://chatgpt.com/backend-api/codex/responses",
            provider=protocols.OPENAI_RESPONSES,
            provider_id="openai-subscription",
            max_tokens=1234,
            openai_request_profile=_codex_model(),
        )

        payload = provider.streaming_chat_payload(
            [formats.message_item("user", "hi")],
            [],
            "gpt-test",
            prompt_cache_key="cache-key",
        )

        self.assertEqual(payload["tool_choice"], "auto")
        self.assertTrue(payload["parallel_tool_calls"])
        self.assertFalse(payload["store"])
        self.assertTrue(payload["stream"])
        self.assertEqual(
            payload["include"], ["reasoning.encrypted_content"])
        self.assertEqual(payload["reasoning"], {
            "effort": "high",
            "summary": "detailed",
        })
        self.assertEqual(payload["text"], {"verbosity": "medium"})
        self.assertEqual(payload["prompt_cache_key"], "cache-key")
        self.assertNotIn("max_output_tokens", payload)

    def test_subscription_parallel_tool_calls_follow_request_profile(self):
        provider = protocols.make_provider(
            "https://chatgpt.com/backend-api/codex/responses",
            provider=protocols.OPENAI_RESPONSES,
            provider_id="openai-subscription",
            openai_request_profile=_codex_model(
                supports_parallel_tool_calls=False),
        )

        payload = provider.streaming_chat_payload(
            [formats.message_item("user", "hi")],
            [],
            "gpt-test",
        )

        self.assertFalse(payload["parallel_tool_calls"])

    def test_unsupported_reasoning_and_verbosity_are_not_requested(self):
        provider = protocols.make_provider(
            "https://chatgpt.com/backend-api/codex/responses",
            provider=protocols.OPENAI_RESPONSES,
            provider_id="openai-subscription",
            openai_request_profile=_codex_model(
                supports_reasoning_summaries=False,
                default_reasoning_level={"ignored": True},
                default_reasoning_summary={"ignored": True},
                support_verbosity=False,
                default_verbosity={"ignored": True},
            ),
        )

        payload = provider.streaming_chat_payload(
            [formats.message_item("user", "hi")],
            [],
            "gpt-test",
        )

        self.assertNotIn("reasoning", payload)
        self.assertNotIn("include", payload)
        self.assertNotIn("text", payload)

    def test_subscription_maps_catalog_ultra_reasoning_to_wire_max(self):
        provider = protocols.make_provider(
            "https://chatgpt.com/backend-api/codex/responses",
            provider=protocols.OPENAI_RESPONSES,
            provider_id="openai-subscription",
            openai_request_profile=_codex_model(
                default_reasoning_level="ultra"),
        )

        payload = provider.streaming_chat_payload(
            [formats.message_item("user", "hi")],
            [],
            "gpt-test",
        )

        self.assertEqual(payload["reasoning"]["effort"], "max")

    def test_subscription_explicit_effort_replaces_catalog_default(self):
        provider = protocols.make_provider(
            "https://chatgpt.com/backend-api/codex/responses",
            provider=protocols.OPENAI_RESPONSES,
            provider_id="openai-subscription",
            openai_request_profile=_codex_model(
                default_reasoning_level="high"),
        )

        payload = provider.chat_payload(
            [formats.message_item("user", "hi")],
            [],
            "gpt-test",
            reasoning_effort="low",
        )

        self.assertEqual(payload["reasoning"], {
            "effort": "low",
            "summary": "detailed",
        })
        self.assertEqual(
            payload["include"], ["reasoning.encrypted_content"])

    def test_responses_lite_uses_input_items_for_instructions_and_tools(self):
        provider = protocols.make_provider(
            "https://chatgpt.com/backend-api/codex/responses",
            provider=protocols.OPENAI_RESPONSES,
            provider_id="openai-subscription",
            openai_request_profile=_codex_model(
                "gpt-5.6-sol",
                use_responses_lite=True,
                tool_mode="code_mode_only",
                default_reasoning_level="low",
                default_reasoning_summary="none",
            ),
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            },
        }]

        payload = provider.streaming_chat_payload(
            [
                formats.instruction_item("system marker"),
                formats.message_item("user", "read marker"),
            ],
            tools,
            "gpt-5.6-sol",
        )

        self.assertEqual(
            provider.headers[protocols.RESPONSES_LITE_HEADER], "true")
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(payload["reasoning"], {
            "effort": "low",
            "context": "all_turns",
        })
        self.assertNotIn("instructions", payload)
        self.assertNotIn("tools", payload)
        self.assertEqual(
            [item["type"] for item in payload["input"]],
            ["additional_tools", "message", "message"],
        )
        additional = payload["input"][0]
        self.assertEqual(additional["role"], "developer")
        namespace = additional["tools"][0]
        self.assertEqual(namespace["type"], "namespace")
        self.assertEqual(namespace["name"], "functions")
        self.assertTrue(namespace["description"])
        self.assertEqual(namespace["tools"][0]["type"], "function")
        self.assertEqual(namespace["tools"][0]["name"], "Read")
        self.assertEqual(payload["input"][1]["role"], "developer")
        self.assertEqual(
            payload["input"][1]["content"][0]["text"], "system marker")
        self.assertEqual(payload["input"][2]["role"], "user")

    def test_responses_lite_is_confined_to_chatgpt_subscription(self):
        with self.assertRaisesRegex(
                protocols.ProtocolError,
                "only valid for the OpenAI ChatGPT subscription"):
            protocols.make_provider(
                "https://example.test/v1/responses",
                provider=protocols.OPENAI_RESPONSES,
                openai_request_profile=_codex_model(),
            )

    def test_subscription_parses_codex_model_catalog(self):
        provider = protocols.make_provider(
            "https://chatgpt.com/backend-api/codex/responses",
            provider=protocols.OPENAI_RESPONSES,
            provider_id="openai-subscription",
            openai_request_profile=_codex_model("visible"),
        )

        model_ids = provider.parse_model_ids({
            "models": [
                {"slug": "visible", "visibility": "list"},
                {"slug": "hidden", "visibility": "hide"},
            ],
        })

        self.assertEqual(model_ids, ["visible"])

    def test_generic_responses_payload_omits_codex_backend_controls(self):
        provider = protocols.make_provider(
            "https://example.test/v1/responses",
            provider=protocols.OPENAI_RESPONSES,
            provider_id="compatible-provider",
            max_tokens=1234,
        )

        payload = provider.chat_payload(
            [formats.message_item("user", "hi")],
            [],
            "gpt-test",
            prompt_cache_key="subscription-only",
        )

        self.assertEqual(payload["max_output_tokens"], 1234)
        for field in [
                "tool_choice", "parallel_tool_calls", "store", "include",
                "prompt_cache_key"]:
            self.assertNotIn(field, payload)

    def test_generic_responses_sends_only_explicit_reasoning_effort(self):
        provider = protocols.make_provider(
            "https://api.openai.com/v1/responses",
            provider=protocols.OPENAI_RESPONSES,
        )

        default_payload = provider.chat_payload(
            [formats.message_item("user", "hi")], [], "gpt-test")
        explicit_payload = provider.chat_payload(
            [formats.message_item("user", "hi")],
            [],
            "gpt-test",
            reasoning_effort="high",
        )

        self.assertNotIn("reasoning", default_payload)
        self.assertEqual(
            explicit_payload["reasoning"], {"effort": "high"})

    def test_responses_parse_response_separates_items_from_envelope(self):
        provider = protocols.make_provider(
            "https://api.openai.com/v1/responses",
            provider=protocols.OPENAI_RESPONSES,
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

        turn = provider.parse_chat_response(response)

        self.assertEqual(
            [item.get("type") for item in turn.items],
            ["function_call"],
        )
        self.assertEqual(
            turn.metadata["protocol"], protocols.OPENAI_RESPONSES)
        self.assertEqual(turn.items[0]["call_id"], "call_1")
        self.assertEqual(turn.items[0]["name"], "Read")
        self.assertEqual(
            formats.tool_call_input(turn.items[0]),
            {"file_path": "README.md"},
        )
        event = turn.to_event()
        self.assertEqual(event["protocol"], protocols.OPENAI_RESPONSES)
        self.assertEqual(event["items"], turn.items)
        self.assertNotIn("response", turn.metadata)


class OpenAIChatReasoningTests(unittest.TestCase):
    def test_openrouter_payload_uses_nested_reasoning_effort(self):
        provider = protocols.make_provider(
            "https://openrouter.ai/api/v1/chat/completions",
            provider=protocols.OPENAI_CHAT,
            provider_id="openrouter",
        )

        default_payload = provider.chat_payload(
            [formats.message_item("user", "hello")], [], "model")
        explicit_payload = provider.streaming_chat_payload(
            [formats.message_item("user", "hello")],
            [],
            "model",
            reasoning_effort="max",
        )

        self.assertNotIn("reasoning_effort", default_payload)
        self.assertEqual(
            explicit_payload["reasoning"], {"effort": "max"})
        self.assertTrue(explicit_payload["stream"])

    def test_zai_effort_explicitly_enables_thinking(self):
        for provider_id, url in [
                ("zai", "https://api.z.ai/api/paas/v4/chat/completions"),
                (
                    "zai-coding-plan",
                    "https://api.z.ai/api/coding/paas/v4/chat/completions",
                ),
                (
                    "zhipuai",
                    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                ),
                (
                    "zhipuai-coding-plan",
                    "https://open.bigmodel.cn/api/coding/paas/v4/"
                    "chat/completions",
                ),
        ]:
            with self.subTest(provider_id=provider_id):
                provider = protocols.make_provider(
                    url,
                    provider=protocols.OPENAI_CHAT,
                    provider_id=provider_id,
                )

                default_payload = provider.chat_payload(
                    [formats.message_item("user", "hello")],
                    [],
                    "glm",
                )
                explicit_payload = provider.chat_payload(
                    [formats.message_item("user", "hello")],
                    [],
                    "glm",
                    reasoning_effort="max",
                )

                self.assertNotIn("reasoning_effort", default_payload)
                self.assertNotIn("thinking", default_payload)
                self.assertEqual(
                    explicit_payload["reasoning_effort"], "max")
                self.assertEqual(
                    explicit_payload["thinking"], {"type": "enabled"})

    def test_zai_none_disables_thinking_and_omits_effort(self):
        provider = protocols.make_provider(
            "https://api.z.ai/api/paas/v4/chat/completions",
            provider=protocols.OPENAI_CHAT,
            provider_id="zai",
        )

        payload = provider.chat_payload(
            [formats.message_item("user", "hello")],
            [],
            "glm",
            reasoning_effort="none",
        )

        self.assertEqual(
            payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)


class AnthropicMessagesProviderTests(unittest.TestCase):
    def test_credentialless_provider_keeps_protocol_version_header(self):
        provider = protocols.make_provider(
            "http://localhost:8000/v1/messages",
            provider=protocols.ANTHROPIC_MESSAGES,
            anthropic_version="2024-01-01",
        )

        self.assertEqual(
            provider.headers["anthropic-version"], "2024-01-01")
        self.assertNotIn("x-api-key", provider.headers)
        self.assertNotIn("Authorization", provider.headers)

    def test_payload_enables_automatic_prompt_caching(self):
        provider = protocols.make_provider(
            "https://api.anthropic.com/v1/messages",
            provider=protocols.ANTHROPIC_MESSAGES,
            prompt_cache=True,
        )

        payload = provider.chat_payload(
            [
                formats.instruction_item("system"),
                formats.message_item("user", "hello"),
            ],
            [],
            "claude-test",
        )

        self.assertEqual(
            payload["cache_control"], {"type": "ephemeral"})
        self.assertEqual(
            payload["system"], [{"type": "text", "text": "system"}])

    def test_payload_omits_prompt_cache_for_compatible_server_by_default(self):
        provider = protocols.make_provider(
            "http://localhost:8000/v1/messages",
            provider=protocols.ANTHROPIC_MESSAGES,
        )

        payload = provider.chat_payload(
            [formats.message_item("user", "hello")],
            [],
            "local-model",
        )

        self.assertNotIn("cache_control", payload)

    def test_payload_uses_output_config_for_explicit_effort(self):
        provider = protocols.make_provider(
            "https://api.anthropic.com/v1/messages",
            provider=protocols.ANTHROPIC_MESSAGES,
        )

        default_payload = provider.chat_payload(
            [formats.message_item("user", "hello")],
            [],
            "claude-test",
        )
        explicit_payload = provider.chat_payload(
            [formats.message_item("user", "hello")],
            [],
            "claude-test",
            reasoning_effort="medium",
        )

        self.assertNotIn("output_config", default_payload)
        self.assertEqual(
            explicit_payload["output_config"], {"effort": "medium"})


class StreamAccumulatorTests(unittest.TestCase):
    def event(self, data, event="message"):
        return sse.SseEvent(event, data)

    def test_openai_chat_semantic_strings_are_split_invariant(self):
        values = {
            "content": "Hello world",
            "reasoning_content": "Think first",
            "arguments": '{"file_path":"README.md"}',
        }
        buffered = {
            "id": "chat_split",
            "object": "chat.completion",
            "model": "chat-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": values["content"],
                    "reasoning_content": values["reasoning_content"],
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": values["arguments"],
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        expected = formats.openai_chat_response_to_items(
            buffered).to_event()

        for target, value in values.items():
            for split in range(len(value) + 1):
                with self.subTest(target=target, split=split):
                    parts = {
                        key: [item]
                        for key, item in values.items()
                    }
                    parts[target] = [value[:split], value[split:]]
                    accumulator = (
                        protocols.OpenAIChatStreamAccumulator(
                            lambda text: None))
                    for position in range(2):
                        delta = {}
                        if position == 0:
                            delta["role"] = "assistant"
                        if position < len(parts["content"]):
                            delta["content"] = (
                                parts["content"][position])
                        if position < len(
                                parts["reasoning_content"]):
                            delta["reasoning_content"] = (
                                parts["reasoning_content"][position])
                        if position < len(parts["arguments"]):
                            call = {
                                "index": 0,
                                "function": {
                                    "arguments":
                                        parts["arguments"][position],
                                },
                            }
                            if position == 0:
                                call.update({
                                    "id": "call_1",
                                    "type": "function",
                                })
                                call["function"]["name"] = "Read"
                            delta["tool_calls"] = [call]
                        accumulator.feed(self.event(json.dumps({
                            "id": "chat_split",
                            "object": "chat.completion.chunk",
                            "model": "chat-model",
                            "choices": [{
                                "index": 0,
                                "delta": delta,
                                "finish_reason": (
                                    "tool_calls"
                                    if position == 1 else None),
                            }],
                        })))
                    accumulator.feed(self.event("[DONE]"))
                    streamed = accumulator.finish()

                    self.assertEqual(streamed, buffered)
                    self.assertEqual(
                        formats.openai_chat_response_to_items(
                            streamed).to_event(),
                        expected,
                    )

    def test_openai_chat_stream_matches_buffered_canonical_turn(self):
        buffered = {
            "id": "chat_equivalent",
            "object": "chat.completion",
            "model": "chat-model",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "refusal": "No",
                    "reasoning_content": "Think first",
                    "audio": {
                        "id": "audio_1",
                        "data": "AB",
                        "transcript": "spoken",
                        "expires_at": 123,
                    },
                    "phase": "final",
                    "provider_trace": "trace",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "provider_call": "call-extra",
                        "function": {
                            "name": "Read",
                            "arguments":
                                '{"file_path":"README.md"}',
                            "provider_function": "function-extra",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
                "logprobs": {
                    "content": [
                        {"token": "Hel", "logprob": -0.1},
                        {"token": "lo", "logprob": -0.2},
                    ],
                    "refusal": [
                        {"token": "No", "logprob": -0.3},
                    ],
                },
            }],
        }
        chunks = [
            {
                "id": "chat_equivalent",
                "object": "chat.completion.chunk",
                "model": "chat-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Hel",
                        "reasoning_content": "Think ",
                        "audio": {
                            "id": "audio_1",
                            "data": "A",
                            "transcript": "spo",
                        },
                        "phase": "final",
                        "provider_trace": "tr",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "provider_call": "call-",
                            "function": {
                                "name": "Read",
                                "arguments": '{"file_path":',
                                "provider_function": "function-",
                            },
                        }],
                    },
                    "finish_reason": None,
                    "logprobs": {
                        "content": [{
                            "token": "Hel", "logprob": -0.1,
                        }],
                    },
                }],
            },
            {
                "id": "chat_equivalent",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "lo",
                        "refusal": "N",
                        "reasoning_content": "first",
                        "audio": {
                            "data": "B",
                            "transcript": "ken",
                            "expires_at": 123,
                        },
                        "provider_trace": "ace",
                        "tool_calls": [{
                            "index": 0,
                            "provider_call": "extra",
                            "function": {
                                "arguments": '"README.md"}',
                                "provider_function": "extra",
                            },
                        }],
                    },
                    "finish_reason": None,
                    "logprobs": {
                        "content": [{
                            "token": "lo", "logprob": -0.2,
                        }],
                        "refusal": [{
                            "token": "No", "logprob": -0.3,
                        }],
                    },
                }],
            },
            {
                "usage": buffered["usage"],
                "choices": [{
                    "index": 0,
                    "delta": {"refusal": "o"},
                    "finish_reason": "tool_calls",
                }],
            },
        ]
        accumulator = protocols.OpenAIChatStreamAccumulator(
            lambda text: None)
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            buffered_turn = formats.openai_chat_response_to_items(
                buffered)
            for chunk in chunks:
                accumulator.feed(self.event(json.dumps(chunk)))
            accumulator.feed(self.event("[DONE]"))
            streamed = accumulator.finish()
            streamed_turn = formats.openai_chat_response_to_items(
                streamed)

        self.assertEqual(
            streamed_turn.to_event(), buffered_turn.to_event())
        self.assertEqual(
            streamed["choices"][0]["logprobs"],
            buffered["choices"][0]["logprobs"],
        )
        self.assertEqual(
            streamed["choices"][0]["message"],
            buffered["choices"][0]["message"],
        )

    def test_openai_chat_cost_is_preserved_without_unknown_diagnostic(self):
        buffered = {
            "id": "chat_cost",
            "object": "chat.completion",
            "model": "deepseek-v4-flash",
            "cost": "0",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Working.",
                },
                "finish_reason": "stop",
            }],
        }
        chunks = [
            {
                "id": "chat_cost",
                "object": "chat.completion.chunk",
                "model": "deepseek-v4-flash",
                "cost": "0",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Work",
                    },
                    "finish_reason": None,
                }],
            },
            {
                "cost": "0",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "ing."},
                    "finish_reason": "stop",
                }],
            },
        ]
        accumulator = protocols.OpenAIChatStreamAccumulator(
            lambda text: None)
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            buffered_turn = formats.openai_chat_response_to_items(
                buffered)
            for chunk in chunks:
                accumulator.feed(self.event(json.dumps(chunk)))
            accumulator.feed(self.event("[DONE]"))
            streamed_turn = formats.openai_chat_response_to_items(
                accumulator.finish())

        expected = buffered_turn.to_event()
        self.assertEqual(streamed_turn.to_event(), expected)
        self.assertEqual(
            expected["protocol_data"][protocols.OPENAI_CHAT]["cost"],
            "0",
        )
        self.assertEqual(diagnostics.getvalue(), "")

    def test_openai_chat_tool_only_stream_matches_null_buffered_content(self):
        buffered = {
            "id": "chat_tool_only",
            "object": "chat.completion",
            "model": "chat-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments":
                                '{"file_path":"README.md"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        chunks = [
            {
                "id": "chat_tool_only",
                "object": "chat.completion.chunk",
                "model": "chat-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": '{"file_path":',
                            },
                        }],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": '"README.md"}',
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            },
        ]
        accumulator = protocols.OpenAIChatStreamAccumulator(
            lambda text: None)
        for chunk in chunks:
            accumulator.feed(self.event(json.dumps(chunk)))
        accumulator.feed(self.event("[DONE]"))
        streamed = accumulator.finish()

        self.assertIsNone(
            streamed["choices"][0]["message"]["content"])
        self.assertEqual(
            formats.openai_chat_response_to_items(
                streamed).to_event(),
            formats.openai_chat_response_to_items(
                buffered).to_event(),
        )

    def test_anthropic_stream_matches_buffered_canonical_turn(self):
        buffered = {
            "id": "message_equivalent",
            "type": "message",
            "role": "assistant",
            "model": "claude-model",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Think first",
                    "signature": "signature",
                },
                {
                    "type": "text",
                    "text": "Answer",
                    "citations": [{
                        "type": "char_location",
                        "start_char_index": 0,
                        "end_char_index": 6,
                    }],
                },
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"file_path": "README.md"},
                },
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "Loki"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [{
                        "type": "web_search_result",
                        "title": "Loki",
                        "url": "https://example.test/loki",
                        "encrypted_content": "encrypted",
                    }],
                },
                {
                    "type": "redacted_thinking",
                    "data": "redacted",
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": "STOP",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 34,
            },
        }
        events = [
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "message_equivalent",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-model",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 1,
                    },
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": "Think ",
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": "first",
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "signature_delta",
                    "signature": "signature",
                },
            }),
            ("content_block_stop", {
                "type": "content_block_stop", "index": 0,
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Ans"},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "wer"},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "citations_delta",
                    "citation": buffered["content"][1]["citations"][0],
                },
            }),
            ("content_block_stop", {
                "type": "content_block_stop", "index": 1,
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {},
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"file_path":',
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '"README.md"}',
                },
            }),
            ("content_block_stop", {
                "type": "content_block_stop", "index": 2,
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 3,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {},
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 3,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"query":"Loki"}',
                },
            }),
            ("content_block_stop", {
                "type": "content_block_stop", "index": 3,
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 4,
                "content_block": buffered["content"][4],
            }),
            ("content_block_stop", {
                "type": "content_block_stop", "index": 4,
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 5,
                "content_block": buffered["content"][5],
            }),
            ("content_block_stop", {
                "type": "content_block_stop", "index": 5,
            }),
            ("message_delta", {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use",
                    "stop_sequence": "STOP",
                },
                "usage": {"output_tokens": 34},
            }),
            ("message_stop", {"type": "message_stop"}),
        ]
        accumulator = protocols.AnthropicMessagesStreamAccumulator(
            lambda text: None)
        for event_name, data in events:
            accumulator.feed(self.event(
                json.dumps(data), event=event_name))
        streamed = accumulator.finish()

        buffered_turn = formats.anthropic_response_to_items(buffered)
        streamed_turn = formats.anthropic_response_to_items(streamed)

        self.assertEqual(streamed, buffered)
        self.assertEqual(
            streamed_turn.to_event(), buffered_turn.to_event())

    def test_anthropic_semantic_strings_are_split_invariant(self):
        cases = [
            (
                "text",
                "Answer text",
                {"type": "text", "text": "Answer text"},
                {"type": "text", "text": ""},
                "text_delta",
                "text",
                [],
            ),
            (
                "thinking",
                "Think first",
                {
                    "type": "thinking",
                    "thinking": "Think first",
                    "signature": "signature",
                },
                {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                },
                "thinking_delta",
                "thinking",
                [],
            ),
            (
                "signature",
                "signature",
                {
                    "type": "thinking",
                    "thinking": "Think first",
                    "signature": "signature",
                },
                {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                },
                "signature_delta",
                "signature",
                [("thinking_delta", "thinking", "Think first")],
            ),
            (
                "tool_input",
                '{"file_path":"README.md"}',
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"file_path": "README.md"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {},
                },
                "input_json_delta",
                "partial_json",
                [],
            ),
        ]
        for (case_name, value, final_block, start_block,
             delta_type, delta_key, preceding) in cases:
            buffered = {
                "id": "message_split",
                "type": "message",
                "role": "assistant",
                "model": "claude-model",
                "content": [final_block],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 5,
                },
            }
            expected = formats.anthropic_response_to_items(
                buffered).to_event()
            for split in range(len(value) + 1):
                with self.subTest(
                        case=case_name, split=split):
                    accumulator = (
                        protocols.AnthropicMessagesStreamAccumulator(
                            lambda text: None))
                    accumulator.feed(self.event(json.dumps({
                        "type": "message_start",
                        "message": {
                            "id": "message_split",
                            "type": "message",
                            "role": "assistant",
                            "model": "claude-model",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                            },
                        },
                    }), event="message_start"))
                    accumulator.feed(self.event(json.dumps({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": start_block,
                    }), event="content_block_start"))
                    for kind, key, part in preceding:
                        accumulator.feed(self.event(json.dumps({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": kind,
                                key: part,
                            },
                        }), event="content_block_delta"))
                    for part in [value[:split], value[split:]]:
                        accumulator.feed(self.event(json.dumps({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": delta_type,
                                delta_key: part,
                            },
                        }), event="content_block_delta"))
                    if case_name == "thinking":
                        accumulator.feed(self.event(json.dumps({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "signature_delta",
                                "signature": "signature",
                            },
                        }), event="content_block_delta"))
                    accumulator.feed(self.event(json.dumps({
                        "type": "content_block_stop",
                        "index": 0,
                    }), event="content_block_stop"))
                    accumulator.feed(self.event(json.dumps({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": 5},
                    }), event="message_delta"))
                    accumulator.feed(self.event(json.dumps({
                        "type": "message_stop",
                    }), event="message_stop"))
                    streamed = accumulator.finish()

                    self.assertEqual(streamed, buffered)
                    self.assertEqual(
                        formats.anthropic_response_to_items(
                            streamed).to_event(),
                        expected,
                    )

    def test_responses_terminal_stream_matches_buffered_canonical_turn(self):
        cases = [
            (
                "response.completed",
                {
                    "id": "response_completed",
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {
                            "type": "reasoning",
                            "id": "reasoning_1",
                            "summary": [],
                            "encrypted_content": "encrypted",
                        },
                        {
                            "type": "message",
                            "id": "message_1",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": "answer",
                                "annotations": [],
                            }],
                        },
                        {
                            "type": "function_call",
                            "id": "function_1",
                            "status": "completed",
                            "call_id": "call_1",
                            "name": "Read",
                            "arguments":
                                '{"file_path":"README.md"}',
                        },
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            ),
            (
                "response.incomplete",
                {
                    "id": "response_incomplete",
                    "object": "response",
                    "status": "incomplete",
                    "incomplete_details": {
                        "reason": "max_output_tokens",
                    },
                    "output": [{
                        "type": "message",
                        "id": "message_partial",
                        "status": "incomplete",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": "partial",
                            "annotations": [],
                        }],
                    }],
                },
            ),
        ]
        for terminal_type, buffered in cases:
            with self.subTest(terminal_type=terminal_type):
                accumulator = (
                    protocols.OpenAIResponsesStreamAccumulator(
                        lambda text: None))
                accumulator.feed(self.event(json.dumps({
                    "type": "response.output_text.delta",
                    "delta": "ignored for final authority",
                })))
                accumulator.feed(self.event(json.dumps({
                    "type": terminal_type,
                    "response": buffered,
                })))
                streamed = accumulator.finish()

                buffered_turn = (
                    formats.openai_responses_response_to_items(
                        buffered))
                streamed_turn = (
                    formats.openai_responses_response_to_items(
                        streamed))

                self.assertEqual(streamed, buffered)
                self.assertEqual(
                    streamed_turn.to_event(),
                    buffered_turn.to_event(),
                )

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
        self.assertEqual(formats.item_text(items[0]), "Hello")
        self.assertEqual(
            [item["name"] for item in items[1:]], ["Read", "Glob"])
        self.assertEqual(
            formats.tool_call_input(items[1]),
            {"file_path": "README.md"})
        self.assertEqual(
            formats.tool_call_input(items[2]), {"pattern": "*.py"})
        self.assertEqual(response["object"], "chat.completion")

    def test_openai_chat_requires_done_marker(self):
        accumulator = protocols.OpenAIChatStreamAccumulator(lambda text: None)
        accumulator.feed(self.event(
            '{"choices":[{"index":0,"delta":{"content":"partial"}}]}'))

        with self.assertRaisesRegex(
                protocols.StreamProtocolError, "before data: \\[DONE\\]"):
            accumulator.finish()

    def test_openai_chat_assembles_reasoning_content_without_token_diagnostics(
            self):
        accumulator = protocols.OpenAIChatStreamAccumulator(
            lambda text: None)
        diagnostics = io.StringIO()
        chunks = [
            {"choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "reasoning_content": "The",
                },
                "finish_reason": None,
            }]},
            {"choices": [{
                "index": 0,
                "delta": {"reasoning_content": " user"},
                "finish_reason": None,
            }]},
            {"choices": [{
                "index": 0,
                "delta": {"content": "Working."},
                "finish_reason": "stop",
            }]},
        ]
        with contextlib.redirect_stderr(diagnostics):
            for chunk in chunks:
                accumulator.feed(self.event(json.dumps(chunk)))
            accumulator.feed(self.event("[DONE]"))
            response = accumulator.finish()
            turn = formats.openai_chat_response_to_items(response)

        self.assertEqual(diagnostics.getvalue(), "")
        self.assertEqual(
            response["choices"][0]["message"]["reasoning_content"],
            "The user",
        )
        self.assertEqual(formats.item_text(turn.items[0]), "Working.")
        self.assertEqual(
            turn.items[0]["protocol_data"][protocols.OPENAI_CHAT]
            ["fields"]["reasoning_content"],
            "The user",
        )

    def test_openai_chat_aggregates_unknown_string_delta_before_diagnostic(
            self):
        accumulator = protocols.OpenAIChatStreamAccumulator(
            lambda text: None)
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            for value in ["one", " two", " three"]:
                accumulator.feed(self.event(json.dumps({
                    "choices": [{
                        "index": 0,
                        "delta": {"future_delta": value},
                        "finish_reason": None,
                    }],
                })))
            accumulator.feed(self.event(json.dumps({
                "choices": [{
                    "index": 0,
                    "delta": {"content": "answer"},
                    "finish_reason": "stop",
                }],
            })))
            accumulator.feed(self.event("[DONE]"))
            response = accumulator.finish()
            formats.openai_chat_response_to_items(response)

        self.assertEqual(
            response["choices"][0]["message"]["future_delta"],
            "one two three",
        )
        self.assertEqual(
            diagnostics.getvalue().count(
                "Unknown openai_chat message fields"),
            1,
        )

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
        self.assertEqual(formats.item_text(items[0]), "hello")
        call = formats.response_tool_calls(items.items)[0]
        self.assertEqual(call["name"], "Read")
        self.assertEqual(
            formats.tool_call_input(call), {"file_path": "README.md"})
        self.assertEqual(
            response["usage"], {"input_tokens": 4, "output_tokens": 8})

    def test_anthropic_accumulates_streamed_server_tool_json(self):
        accumulator = protocols.AnthropicMessagesStreamAccumulator(
            lambda text: None)
        events = [
            ("message_start", {
                "type": "message_start",
                "message": {
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"query":"current news"}',
                },
            }),
            ("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            }),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "pause_turn"},
            }),
            ("message_stop", {"type": "message_stop"}),
        ]
        for event_name, data in events:
            accumulator.feed(self.event(
                json.dumps(data), event=event_name))

        response = accumulator.finish()
        turn = formats.anthropic_response_to_items(response)
        call = turn.items[0]

        self.assertEqual(
            formats.tool_call_input(call), {"query": "current news"})
        self.assertEqual(call["execution"], "provider")

    def test_anthropic_preserves_server_tool_input_from_block_start(self):
        accumulator = protocols.AnthropicMessagesStreamAccumulator(
            lambda text: None)
        events = [
            ("message_start", {
                "type": "message_start",
                "message": {
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "already complete"},
                },
            }),
            ("content_block_stop", {
                "type": "content_block_stop", "index": 0,
            }),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "pause_turn"},
            }),
            ("message_stop", {"type": "message_stop"}),
        ]
        for event_name, data in events:
            accumulator.feed(self.event(
                json.dumps(data), event=event_name))

        response = accumulator.finish()

        self.assertEqual(
            response["content"][0]["input"],
            {"query": "already complete"},
        )

    def test_openai_responses_uses_output_item_done_as_authority(self):
        deltas = []
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            deltas.append)
        accumulator.feed(self.event(
            '{"type":"response.output_text.delta","delta":"hel"}'))
        accumulator.feed(self.event(
            '{"type":"response.output_text.delta","delta":"lo"}'))
        accumulator.feed(self.event(
            '{"type":"response.output_item.done","output_index":0,'
            '"item":{"type":"message","role":"assistant","content":'
            '[{"type":"output_text","text":"hello"}]}}'))
        accumulator.feed(self.event(
            '{"type":"response.completed","response":'
            '{"id":"resp_1","object":"response","status":"completed",'
            '"frequency_penalty":0.0,"presence_penalty":0.0,'
            '"moderation":null,"tool_usage":{"web_search":'
            '{"num_requests":0}},'
            '"output":[{"id":"msg_1","type":"message",'
            '"status":"completed","role":"assistant","content":'
            '[{"type":"output_text","text":"compatibility fallback"}]}]}}'))

        response = accumulator.finish()
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            items = formats.openai_responses_response_to_items(response)

        self.assertEqual(deltas, ["hel", "lo"])
        self.assertEqual(formats.item_text(items[0]), "hello")
        self.assertEqual(diagnostics.getvalue(), "")

    def test_responses_effective_model_prefers_nested_event_headers(self):
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            lambda text: None)
        accumulator.feed(self.event(json.dumps({
            "type": "response.created",
            "headers": {"OPENAI-MODEL": "top-level-model"},
        })))
        accumulator.feed(self.event(json.dumps({
            "type": "response.completed",
            "headers": {"openai-model": "later-top-level-model"},
            "response": {
                "id": "response_1",
                "headers": {"OpenAI-Model": "nested-model"},
            },
        })))

        response = accumulator.finish()

        self.assertEqual(accumulator.effective_model, "nested-model")
        self.assertNotIn("headers", response)

    def test_responses_metadata_retains_only_known_notice_codes(self):
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            lambda text: None)
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            accumulator.feed(self.event(json.dumps({
                "type": "response.metadata",
                "metadata": {
                    "openai_verification_recommendation": [
                        formats.TRUSTED_ACCESS_FOR_CYBER,
                        formats.TRUSTED_ACCESS_FOR_CYBER,
                    ],
                    "openai_chatgpt_moderation_metadata": {
                        "opaque": True,
                    },
                },
            })))
            accumulator.feed(self.event(json.dumps({
                "type": "response.metadata",
                "metadata": {
                    "type": "safety_buffering",
                    "retry_model": "opaque",
                },
                "safety_buffering": {"opaque": True},
            })))
            accumulator.feed(self.event(json.dumps({
                "type": "response.safety_buffering",
                "safety_buffering": {"opaque": True},
            })))
            accumulator.feed(self.event(
                '{"type":"response.completed","response":'
                '{"id":"response_1"}}'))

        self.assertEqual(
            accumulator.notice_codes,
            [formats.TRUSTED_ACCESS_FOR_CYBER],
        )
        self.assertEqual(diagnostics.getvalue(), "")
        self.assertNotIn(
            "metadata", json.dumps(accumulator.finish()))

    def test_unknown_responses_metadata_is_still_diagnosed(self):
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            lambda text: None)
        diagnostics = io.StringIO()

        with contextlib.redirect_stderr(diagnostics):
            accumulator.feed(self.event(json.dumps({
                "type": "response.metadata",
                "metadata": {"future_metadata": {"value": 1}},
            })))

        self.assertIn("future_metadata", diagnostics.getvalue())

    def test_openai_responses_collects_items_with_metadata_only_completion(
            self):
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            lambda text: None)
        accumulator.feed(self.event(
            '{"type":"response.output_item.done","output_index":0,'
            '"item":{"type":"reasoning","encrypted_content":"opaque"}}'))
        accumulator.feed(self.event(
            '{"type":"response.output_item.done","output_index":1,'
            '"item":{"type":"function_call","call_id":"call_1",'
            '"name":"Read","arguments":"{\\"file_path\\":\\"README.md\\"}",'
            '"status":"completed"}}'))
        accumulator.feed(self.event(
            '{"type":"response.completed","response":'
            '{"id":"resp_1"}}'))

        response = accumulator.finish()
        turn = formats.openai_responses_response_to_items(response)

        self.assertEqual(
            [item["type"] for item in response["output"]],
            ["reasoning", "function_call"],
        )
        calls = formats.response_tool_calls(turn.items)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Read")
        self.assertEqual(
            formats.tool_call_input(calls[0]),
            {"file_path": "README.md"},
        )

    def test_openai_responses_incomplete_stream_returns_partial_turn(self):
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            lambda text: None)
        accumulator.feed(self.event(
            '{"type":"response.incomplete","response":'
            '{"id":"resp_1","object":"response","status":"incomplete",'
            '"incomplete_details":{"reason":"max_output_tokens"},'
            '"output":[{"id":"msg_1","type":"message",'
            '"status":"incomplete","role":"assistant","content":'
            '[{"type":"output_text","text":"partial"}]}]}}'))

        response = accumulator.finish()
        turn = formats.openai_responses_response_to_items(response)

        self.assertFalse(turn.complete)
        self.assertEqual(formats.item_text(turn.items[0]), "partial")
        self.assertEqual(
            turn.metadata["protocol_data"][protocols.OPENAI_RESPONSES]
            ["incomplete_details"]["reason"],
            "max_output_tokens",
        )

    def test_openai_responses_failed_stream_raises_classified_api_error(self):
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            lambda text: None)
        with self.assertRaises(protocols.ResponseApiError) as raised:
            accumulator.feed(self.event(
                '{"type":"response.failed","response":'
                '{"id":"resp_1","object":"response","status":"failed",'
                '"error":{"code":"server_error","message":"failed"},'
                '"output":[]}}'))

        self.assertEqual(raised.exception.code, "server_error")
        self.assertEqual(raised.exception.category, "retryable")
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(
            raised.exception.payload["response"]["id"], "resp_1")

    def test_response_failure_classes_match_codex(self):
        cases = [
            ("context_length_exceeded", "context_window", False),
            ("insufficient_quota", "quota", False),
            ("usage_not_included", "subscription", False),
            ("cyber_policy", "cyber_policy", False),
            ("invalid_prompt", "invalid_request", False),
            ("bio_policy", "invalid_request", False),
            ("server_is_overloaded", "overloaded", False),
            ("slow_down", "overloaded", False),
            ("unknown_transient_error", "retryable", True),
        ]
        for code, category, retryable in cases:
            with self.subTest(code=code):
                accumulator = protocols.OpenAIResponsesStreamAccumulator(
                    lambda text: None)
                with self.assertRaises(
                        protocols.ResponseApiError) as raised:
                    accumulator.feed(self.event(json.dumps({
                        "type": "response.failed",
                        "response": {
                            "error": {
                                "code": code,
                                "message": "failure",
                            },
                        },
                    })))
                self.assertEqual(raised.exception.category, category)
                self.assertIs(raised.exception.retryable, retryable)

    def test_rate_limit_failure_carries_retry_delay(self):
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            lambda text: None)
        with self.assertRaises(protocols.ResponseApiError) as raised:
            accumulator.feed(self.event(json.dumps({
                "type": "response.failed",
                "response": {
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": "Please try again in 125 ms.",
                    },
                },
            })))

        self.assertTrue(raised.exception.retryable)
        self.assertEqual(raised.exception.retry_after, 0.125)

    def test_openai_responses_error_event_is_a_stream_error(self):
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            lambda text: None)
        accumulator.feed(self.event(
            '{"type":"error","code":"stream_error",'
            '"message":"transport failed"}'))

        with self.assertRaises(protocols.StreamProtocolError) as raised:
            accumulator.finish()

        self.assertEqual(
            raised.exception.payload["code"], "stream_error")

    def test_unknown_stream_event_is_diagnosed_not_in_conversation(self):
        accumulator = protocols.OpenAIResponsesStreamAccumulator(
            lambda text: None)
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            accumulator.feed(self.event(
                '{"type":"future.event","payload":{"marker":true}}'))
            accumulator.feed(self.event(
                '{"type":"response.completed","response":'
                '{"id":"resp_1","object":"response",'
                '"status":"completed","output":[]}}'))
            response = accumulator.finish()
            turn = formats.openai_responses_response_to_items(response)

        self.assertIn('"marker": true', diagnostics.getvalue())
        self.assertEqual(turn.items, [])
        self.assertNotIn("response", turn.metadata)
        self.assertNotIn(
            "_loki_stream_extensions", json.dumps(turn.to_event()))

    def test_streaming_payload_is_opt_in_at_call_site(self):
        provider = protocols.make_provider(
            "http://localhost:8000/v1",
            provider=protocols.OPENAI_CHAT,
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
