import contextlib
import copy
import io
import json
import unittest

from loki_agent import formats


def mixed_session():
    return [
        formats.instruction_item("You are Loki."),
        formats.message_item("user", "Read a.txt and b.txt."),
        formats.model_response_event(
            formats.ANTHROPIC_MESSAGES,
            [
                {
                    "type": "anthropic_thinking",
                    "thinking": "I should read both.",
                    "signature": "sig_abc",
                },
                formats.message_item(
                    "assistant", "I will read both files."),
                formats.tool_call_item(
                    "toolu_a", "read", {"path": "a.txt"}),
                formats.tool_call_item(
                    "toolu_b", "read", {"path": "b.txt"}),
            ],
            provider="anthropic",
            model="claude-test",
            stop_reason="tool_use",
        ),
        formats.tool_result_item("toolu_a", "AAA"),
        formats.tool_result_item("toolu_b", "BBB"),
        formats.model_response_event(
            formats.OPENAI_RESPONSES,
            [
                {
                    "type": "openai_reasoning",
                    "value": {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                        "encrypted_content": "encrypted",
                    },
                },
                formats.message_item(
                    "assistant",
                    "I found a matching line.",
                    protocol_data={
                        formats.OPENAI_RESPONSES: {
                            "id": "msg_1",
                            "status": "completed",
                            "phase": "final",
                        },
                    },
                ),
                formats.tool_call_item(
                    "call_grep",
                    "grep",
                    {"pattern": "needle"},
                    protocol_data={
                        formats.OPENAI_RESPONSES: {
                            "native_type": "function_call",
                            "id": "fc_1",
                            "status": "completed",
                        },
                    },
                ),
            ],
            provider="openai",
            model="gpt-test",
        ),
        formats.tool_result_item(
            "call_grep", "a.txt:9:needle"),
        formats.model_response_event(
            formats.OPENAI_CHAT,
            [formats.message_item(
                "assistant", "The work is complete.")],
            provider="chat-provider",
            model="chat-test",
        ),
        formats.message_item("user", "Summarize it."),
    ]


class SessionV4Tests(unittest.TestCase):
    def test_roundtrip_contains_editable_events_without_call_ranges(self):
        events = mixed_session()
        blob = formats.new_log_blob(
            events, [{"content": "todo"}], toolsets=[[{"name": "Read"}]])

        self.assertEqual(blob["schema"], formats.TRANSCRIPT_SCHEMA)
        self.assertIn("events", blob)
        self.assertNotIn("items", blob)
        self.assertNotIn("calls", blob)
        self.assertNotIn("id", blob["events"][2])
        loaded, todos = formats.load_log_blob(blob)
        self.assertEqual(loaded, events)
        self.assertEqual(todos, [{"content": "todo"}])
        self.assertEqual(
            formats.log_toolsets(blob), [[{"name": "Read"}]])

    def test_inserting_and_moving_complete_events_needs_no_reindexing(self):
        events = mixed_session()
        inserted = formats.message_item("user", "Inserted")
        events.insert(1, inserted)
        response = events.pop(3)
        events.insert(2, response)

        blob = formats.new_log_blob(events, [])
        loaded, _ = formats.load_log_blob(blob)
        self.assertEqual(loaded, events)
        rendered = json.dumps(blob)
        self.assertNotIn('"start"', rendered)
        self.assertNotIn('"end"', rendered)
        self.assertNotIn('"output_items"', rendered)

    def test_invalid_tool_result_reports_the_event_and_call(self):
        events = [
            formats.message_item("user", "hello"),
            formats.tool_result_item("missing", "result"),
        ]
        with self.assertRaisesRegex(
                formats.TranscriptFormatError,
                "event 1.*missing"):
            formats.validate_events(events)

    def test_duplicate_call_id_in_one_response_is_invalid(self):
        event = formats.model_response_event(
            formats.OPENAI_CHAT,
            [
                formats.tool_call_item("same", "one", {}),
                formats.tool_call_item("same", "two", {}),
            ],
        )
        with self.assertRaisesRegex(
                formats.TranscriptFormatError,
                "duplicate function call 'same'"):
            formats.validate_events([event])

    def test_old_flat_schema_is_rejected(self):
        with self.assertRaises(formats.TranscriptFormatError):
            formats.load_log_blob({
                "schema": "day-agent.transcript.v3",
                "items": [],
            })

    def test_many_turns_store_instruction_and_toolset_once(self):
        instruction = "unique system instruction marker"
        events = [formats.instruction_item(instruction)]
        for index in range(300):
            events.append(formats.message_item(
                "user", f"question {index}"))
            events.append(formats.model_response_event(
                formats.OPENAI_RESPONSES,
                [formats.message_item(
                    "assistant", f"answer {index}")],
            ))
        toolset = [[{
            "type": "function",
            "function": {"name": "Read"},
        }]]

        serialized = json.dumps(
            formats.new_log_blob(events, [], toolsets=toolset))

        self.assertEqual(serialized.count(instruction), 1)
        self.assertEqual(serialized.count('"name": "Read"'), 1)


class MixedProtocolProjectionTests(unittest.TestCase):
    def test_chat_projection_contains_all_portable_history(self):
        messages = formats.items_to_openai_chat_messages(mixed_session())

        self.assertEqual(
            [message["role"] for message in messages],
            [
                "system", "user", "assistant", "tool", "tool",
                "assistant", "tool", "assistant", "user",
            ],
        )
        self.assertEqual(
            [call["id"] for call in messages[2]["tool_calls"]],
            ["toolu_a", "toolu_b"],
        )
        self.assertEqual(
            messages[5]["tool_calls"][0]["id"], "call_grep")
        self.assertNotIn("thinking", json.dumps(messages))
        self.assertNotIn("encrypted", json.dumps(messages))
        self.assertNotIn("phase", json.dumps(messages))

    def test_anthropic_projection_restores_native_thinking(self):
        system, messages = formats.items_to_anthropic_parts(
            mixed_session())

        self.assertEqual(
            system, [{"type": "text", "text": "You are Loki."}])
        self.assertEqual(
            messages[1]["content"][0],
            {
                "type": "thinking",
                "thinking": "I should read both.",
                "signature": "sig_abc",
            },
        )
        self.assertEqual(
            [block["tool_use_id"]
             for block in messages[2]["content"]],
            ["toolu_a", "toolu_b"],
        )
        self.assertEqual(
            messages[3]["content"][1]["id"], "call_grep")
        self.assertNotIn("encrypted", json.dumps(messages))

    def test_responses_projection_restores_native_reasoning(self):
        instructions, inputs = (
            formats.items_to_openai_responses_parts(mixed_session()))

        self.assertEqual(instructions, "You are Loki.")
        self.assertEqual(
            [item["type"] for item in inputs],
            [
                "message", "message", "function_call", "function_call",
                "function_call_output", "function_call_output",
                "reasoning", "message", "function_call",
                "function_call_output", "message", "message",
            ],
        )
        self.assertEqual(inputs[6]["encrypted_content"], "encrypted")
        self.assertEqual(inputs[7]["phase"], "final")
        self.assertEqual(inputs[8]["id"], "fc_1")
        self.assertNotIn("sig_abc", json.dumps(inputs))

    def test_every_origin_projects_to_every_target(self):
        sessions = {
            protocol: [
                formats.instruction_item("system"),
                formats.message_item("user", "hello"),
                response,
            ]
            for protocol, response in [
                (
                    formats.OPENAI_CHAT,
                    formats.openai_chat_response_to_items({
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "chat",
                            },
                            "finish_reason": "stop",
                        }],
                    }).to_event(),
                ),
                (
                    formats.ANTHROPIC_MESSAGES,
                    formats.anthropic_response_to_items({
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": "anthropic",
                        }],
                        "stop_reason": "end_turn",
                    }).to_event(),
                ),
                (
                    formats.OPENAI_RESPONSES,
                    formats.openai_responses_response_to_items({
                        "object": "response",
                        "status": "completed",
                        "output": [{
                            "type": "message",
                            "id": "msg_1",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": "responses",
                                "annotations": [],
                            }],
                        }],
                    }).to_event(),
                ),
            ]
        }
        for source, events in sessions.items():
            with self.subTest(source=source, target="chat"):
                formats.items_to_openai_chat_messages(events)
            with self.subTest(source=source, target="anthropic"):
                formats.items_to_anthropic_parts(events)
            with self.subTest(source=source, target="responses"):
                formats.items_to_openai_responses_parts(events)

    def test_projection_is_pure(self):
        events = mixed_session()
        before = copy.deepcopy(events)
        formats.items_to_openai_chat_messages(events)
        formats.items_to_anthropic_parts(events)
        formats.items_to_openai_responses_parts(events)
        self.assertEqual(events, before)

    def test_append_changes_only_projected_tail(self):
        events = mixed_session()[:-1]
        chat_before = formats.items_to_openai_chat_messages(events)
        anth_system_before, anth_before = (
            formats.items_to_anthropic_parts(events))
        resp_instructions_before, resp_before = (
            formats.items_to_openai_responses_parts(events))

        events.append(formats.message_item("user", "new tail"))

        self.assertEqual(
            formats.items_to_openai_chat_messages(events)[:-1],
            chat_before,
        )
        anth_system_after, anth_after = (
            formats.items_to_anthropic_parts(events))
        self.assertEqual(anth_system_after, anth_system_before)
        self.assertEqual(anth_after[:-1], anth_before)
        resp_instructions_after, resp_after = (
            formats.items_to_openai_responses_parts(events))
        self.assertEqual(
            resp_instructions_after, resp_instructions_before)
        self.assertEqual(resp_after[:-1], resp_before)

    def test_late_responses_instruction_stays_at_the_tail(self):
        events = [
            formats.instruction_item("initial"),
            formats.message_item("user", "before"),
            formats.model_response_event(
                formats.OPENAI_CHAT,
                [formats.message_item("assistant", "answer")],
            ),
        ]
        instructions_before, inputs_before = (
            formats.items_to_openai_responses_parts(events))

        events.append(formats.instruction_item("late"))
        instructions_after, inputs_after = (
            formats.items_to_openai_responses_parts(events))

        self.assertEqual(instructions_before, "initial")
        self.assertEqual(instructions_after, instructions_before)
        self.assertEqual(inputs_after[:-1], inputs_before)
        self.assertEqual(inputs_after[-1]["role"], "system")
        self.assertEqual(
            inputs_after[-1]["content"],
            [{"type": "input_text", "text": "late"}],
        )


class SameProtocolReplayTests(unittest.TestCase):
    def test_chat_message_fields_calls_and_null_refusal_replay(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "calling",
                        "future_text_field": "marker",
                    }
                ],
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": "{\"file_path\":\"README.md\"}",
                        "future_function_field": "marker",
                    },
                    "future_call_field": "marker",
                }],
                "future_message_field": "marker",
            },
            {
                "role": "assistant",
                "content": None,
                "refusal": "I cannot do that.",
            },
        ]
        for message in messages:
            with self.subTest(message=message):
                with contextlib.redirect_stderr(io.StringIO()):
                    turn = formats.openai_chat_response_to_items({
                        "choices": [{"message": message}],
                    })
                rendered = formats.items_to_openai_chat_messages(
                    [turn.to_event()])
                self.assertEqual(rendered, [message])

    def test_chat_array_content_block_boundaries_replay_exactly(self):
        message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
        turn = formats.openai_chat_response_to_items({
            "choices": [{"message": message}],
        })

        rendered = formats.items_to_openai_chat_messages([
            turn.to_event(),
        ])

        self.assertEqual(rendered, [message])

    def test_chat_reasoning_content_is_native_without_unknown_diagnostic(self):
        message = {
            "role": "assistant",
            "reasoning_content": "private reasoning",
            "content": "visible answer",
        }
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            turn = formats.openai_chat_response_to_items({
                "choices": [{"message": message}],
            })
        event = turn.to_event()
        event.update({
            "provider": "provider-a",
            "endpoint": "https://provider-a.example/v1/chat/completions",
            "model": "model-a",
        })
        origin = formats.projection_target(
            formats.OPENAI_CHAT,
            provider_id="provider-a",
            endpoint="https://provider-a.example/v1/chat/completions",
            model="model-a",
        )
        foreign = formats.projection_target(
            formats.OPENAI_CHAT,
            provider_id="provider-b",
            endpoint="https://provider-b.example/v1/chat/completions",
            model="model-b",
        )

        exact = formats.items_to_openai_chat_messages(
            [event], target=origin)
        portable = formats.items_to_openai_chat_messages(
            [event], target=foreign)

        self.assertEqual(diagnostics.getvalue(), "")
        self.assertEqual(exact, [message])
        self.assertEqual(portable, [{
            "role": "assistant",
            "content": "visible answer",
        }])

    def test_legacy_chat_call_and_result_replay_as_legacy(self):
        turn = formats.openai_chat_response_to_items({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "Read",
                        "arguments": "{\"file_path\":\"README.md\"}",
                    },
                },
            }],
        })
        call = formats.response_tool_calls(turn)[0]
        events = [
            turn.to_event(),
            formats.tool_result_for_call(call, "contents"),
        ]
        rendered = formats.items_to_openai_chat_messages(events)
        self.assertIn("function_call", rendered[0])
        self.assertNotIn("tool_calls", rendered[0])
        self.assertEqual(rendered[1]["role"], "function")
        self.assertEqual(rendered[1]["name"], "Read")

    def test_anthropic_order_and_native_fields_replay(self):
        content = [
            {
                "type": "thinking",
                "thinking": "consider",
                "signature": "sig",
            },
            {
                "type": "text",
                "text": "before",
                "citations": [{"type": "char_location"}],
            },
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Read",
                "input": {"file_path": "README.md"},
                "caller": {"type": "direct"},
            },
            {"type": "text", "text": "after"},
        ]
        turn = formats.anthropic_response_to_items({
            "type": "message",
            "role": "assistant",
            "content": content,
            "stop_reason": "tool_use",
        })
        system, messages = formats.items_to_anthropic_parts(
            [turn.to_event()])
        self.assertEqual(system, [])
        self.assertEqual(messages, [{
            "role": "assistant",
            "content": content,
        }])

    def test_anthropic_thinking_extensions_and_container_replay(self):
        content = [
            {
                "type": "thinking",
                "thinking": "consider",
                "signature": "sig",
                "future_field": "preserved",
            },
            {
                "type": "container_upload",
                "file_id": "file_1",
            },
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            turn = formats.anthropic_response_to_items({
                "type": "message",
                "role": "assistant",
                "content": content,
                "stop_reason": "end_turn",
            })

        _system, messages = formats.items_to_anthropic_parts([
            turn.to_event(),
        ])

        self.assertEqual(messages, [{
            "role": "assistant",
            "content": content,
        }])

    def test_responses_output_items_replay_exactly(self):
        output = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{
                    "type": "summary_text",
                    "text": "summary",
                }],
                "encrypted_content": "encrypted",
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "phase": "final",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "answer",
                    "annotations": [{"type": "future_annotation"}],
                    "logprobs": [],
                }],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "status": "completed",
                "call_id": "call_1",
                "name": "Read",
                "arguments": "{ \"file_path\": \"README.md\" }",
            },
        ]
        turn = formats.openai_responses_response_to_items({
            "object": "response",
            "status": "completed",
            "output": output,
        })
        _instructions, rendered = (
            formats.items_to_openai_responses_parts([
                turn.to_event(),
            ]))
        self.assertEqual(rendered, output)

    def test_responses_end_turn_is_known_and_persisted(self):
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            turn = formats.openai_responses_response_to_items({
                "object": "response",
                "status": "completed",
                "end_turn": False,
                "output": [],
            })

        event = turn.to_event()

        self.assertEqual(diagnostics.getvalue(), "")
        self.assertIs(turn.metadata["end_turn"], False)
        self.assertIs(event["end_turn"], False)
        formats.validate_events([event])

    def test_responses_end_turn_rejects_non_boolean_values(self):
        with self.assertRaisesRegex(
                formats.TranscriptFormatError, "end_turn"):
            formats.openai_responses_response_to_items({
                "object": "response",
                "status": "completed",
                "end_turn": "false",
                "output": [],
            })

    def test_native_replay_requires_the_originating_connection(self):
        event = formats.model_response_event(
            formats.OPENAI_RESPONSES,
            [
                {
                    "type": "openai_reasoning",
                    "value": {
                        "type": "reasoning",
                        "id": "reasoning_1",
                        "encrypted_content": "provider-a-only",
                        "summary": [],
                    },
                },
                formats.message_item("assistant", "portable answer"),
            ],
            provider="provider-a",
            endpoint="https://a.example/v1/responses",
            model="model-a",
        )
        same_target = formats.projection_target(
            formats.OPENAI_RESPONSES,
            provider_id="provider-a",
            endpoint="https://a.example/v1/responses",
            model="model-a",
        )
        other_target = formats.projection_target(
            formats.OPENAI_RESPONSES,
            provider_id="provider-b",
            endpoint="https://b.example/v1/responses",
            model="model-b",
        )
        other_model_target = formats.projection_target(
            formats.OPENAI_RESPONSES,
            provider_id="provider-a",
            endpoint="https://a.example/v1/responses",
            model="model-b",
        )
        aliased_endpoint_target = formats.projection_target(
            formats.OPENAI_RESPONSES,
            provider_id="provider-b",
            endpoint="https://a.example/v1/responses",
            model="model-a",
        )

        _instructions, same = (
            formats.items_to_openai_responses_parts(
                [event], target=same_target))
        _instructions, other = (
            formats.items_to_openai_responses_parts(
                [event], target=other_target))
        _instructions, other_model = (
            formats.items_to_openai_responses_parts(
                [event], target=other_model_target))
        _instructions, aliased_endpoint = (
            formats.items_to_openai_responses_parts(
                [event], target=aliased_endpoint_target))

        self.assertEqual(
            [item["type"] for item in same],
            ["reasoning", "message"],
        )
        self.assertEqual(
            [item["type"] for item in other],
            ["message"],
        )
        self.assertEqual(
            [item["type"] for item in other_model],
            ["message"],
        )
        self.assertEqual(
            [item["type"] for item in aliased_endpoint],
            ["message"],
        )
        self.assertIn("portable answer", json.dumps(other))
        self.assertNotIn("provider-a-only", json.dumps(other))
        self.assertNotIn("provider-a-only", json.dumps(other_model))
        self.assertNotIn(
            "provider-a-only", json.dumps(aliased_endpoint))

    def test_anthropic_provider_data_is_exact_at_origin_and_sanitized_elsewhere(
            self):
        response = {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "portable history"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [
                        {
                            "type": "web_search_result",
                            "title": "Portable result",
                            "url": "https://example.test/result",
                            "encrypted_content": "provider-a-only",
                        },
                    ],
                },
                {
                    "type": "text",
                    "text": "The result is portable.",
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "url": "https://example.test/result",
                            "title": "Portable result",
                            "encrypted_index": "provider-a-index",
                        },
                    ],
                },
            ],
            "stop_reason": "end_turn",
        }
        event = formats.anthropic_response_to_items(
            response).to_event()
        event["provider"] = "provider-a"
        event["endpoint"] = "https://a.example/v1/messages"
        same_target = formats.projection_target(
            formats.ANTHROPIC_MESSAGES,
            provider_id="provider-a",
            endpoint="https://a.example/v1/messages",
        )
        other_target = formats.projection_target(
            formats.ANTHROPIC_MESSAGES,
            provider_id="provider-b",
            endpoint="https://b.example/v1/messages",
        )

        _system, same = formats.items_to_anthropic_parts(
            [event], target=same_target)
        _system, other = formats.items_to_anthropic_parts(
            [event], target=other_target)

        self.assertEqual(same[0]["content"], response["content"])
        serialized = json.dumps(other)
        self.assertIn("Portable result", serialized)
        self.assertIn("The result is portable.", serialized)
        self.assertNotIn("provider-a-only", serialized)
        self.assertNotIn("provider-a-index", serialized)
        self.assertNotIn("citations", serialized)

    def test_responses_provider_operation_is_exact_at_origin_and_portable_elsewhere(
            self):
        output = {
            "type": "code_interpreter_call",
            "id": "ci_1",
            "status": "completed",
            "code": "print(42)",
            "container_id": "container_1",
            "outputs": [{
                "type": "logs",
                "logs": "42",
                "encrypted_content": "provider-a-only",
            }],
        }
        turn = formats.openai_responses_response_to_items({
            "object": "response",
            "status": "completed",
            "output": [output],
        })
        event = turn.to_event()

        _instructions, same = (
            formats.items_to_openai_responses_parts([event]))
        chat = formats.items_to_openai_chat_messages([event])
        _system, anthropic = formats.items_to_anthropic_parts([event])

        self.assertEqual(same, [output])
        self.assertEqual(
            [message["role"] for message in chat],
            ["assistant", "tool"],
        )
        self.assertEqual(
            chat[0]["tool_calls"][0]["function"]["name"],
            "code_interpreter",
        )
        self.assertIn("42", chat[1]["content"])
        self.assertEqual(
            [message["role"] for message in anthropic],
            ["assistant", "user"],
        )
        self.assertIn(
            "42", anthropic[1]["content"][0]["content"])
        self.assertNotIn("provider-a-only", json.dumps(chat))
        self.assertNotIn("provider-a-only", json.dumps(anthropic))

    def test_large_native_payload_is_not_duplicated_in_session(self):
        marker = "x" * 10000
        turn = formats.anthropic_response_to_items({
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "marker"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": marker,
                },
            ],
            "stop_reason": "end_turn",
        })

        serialized = json.dumps(turn.to_event())
        self.assertEqual(serialized.count(marker), 1)

    def test_unknown_output_is_printed_retained_and_not_sent_foreign(self):
        output = {
            "type": "future_output",
            "payload": {"marker": True},
        }
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            turn = formats.openai_responses_response_to_items({
                "object": "response",
                "status": "completed",
                "output": [output],
            })
        event = turn.to_event()
        self.assertIn(json.dumps({"marker": True}), stderr.getvalue())
        self.assertEqual(
            formats.items_to_openai_responses_parts([event])[1],
            [output],
        )
        self.assertEqual(
            formats.items_to_openai_chat_messages([event]), [])
        self.assertEqual(
            formats.items_to_anthropic_parts([event]), ([], []))

    def test_unsupported_known_provider_output_is_also_diagnosed(self):
        output = {
            "type": "computer_call",
            "id": "computer_1",
            "call_id": "call_1",
            "status": "completed",
            "action": {"type": "screenshot"},
        }
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            turn = formats.openai_responses_response_to_items({
                "object": "response",
                "status": "completed",
                "output": [output],
            })
        event = turn.to_event()

        self.assertIn(
            "unsupported provider output", stderr.getvalue())
        self.assertEqual(
            formats.items_to_openai_responses_parts([event])[1],
            [output],
        )
        self.assertEqual(
            formats.items_to_openai_chat_messages([event]), [])
        self.assertEqual(
            formats.items_to_anthropic_parts([event]), ([], []))

    def test_provider_executed_tool_result_survives_switch(self):
        content = [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "marker"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [{
                    "type": "web_search_result",
                    "title": "Result",
                    "url": "https://example.test/",
                }],
            },
            {
                "type": "text",
                "text": "The result says marker.",
            },
        ]
        turn = formats.anthropic_response_to_items({
            "type": "message",
            "role": "assistant",
            "content": content,
            "stop_reason": "end_turn",
        })
        event = turn.to_event()

        _system, anthropic = formats.items_to_anthropic_parts([event])
        self.assertEqual(anthropic, [{
            "role": "assistant",
            "content": content,
        }])

        chat = formats.items_to_openai_chat_messages([event])
        self.assertEqual(chat[0]["tool_calls"][0]["id"], "srvtoolu_1")
        self.assertEqual(chat[1]["role"], "tool")
        self.assertIn("Result", chat[1]["content"])
        self.assertIn("marker", chat[2]["content"])

        _instructions, responses = (
            formats.items_to_openai_responses_parts([event]))
        self.assertEqual(
            [item["type"] for item in responses],
            ["function_call", "function_call_output", "message"],
        )
        self.assertEqual(
            responses[1]["call_id"], "srvtoolu_1")
        self.assertIn("Result", responses[1]["output"])


class ToolAndMediaTests(unittest.TestCase):
    def test_pending_calls_are_derived_without_mutation(self):
        first = formats.tool_call_item("call_1", "Read", {"path": "a"})
        second = formats.tool_call_item("call_2", "Read", {"path": "b"})
        events = [
            formats.model_response_event(
                formats.OPENAI_RESPONSES, [first, second]),
            formats.tool_result_for_call(first, "A"),
        ]
        before = copy.deepcopy(events)
        pending = formats.pending_tool_calls(events)
        self.assertEqual(
            [formats.tool_call_id(call) for call in pending],
            ["call_2"],
        )
        self.assertEqual(events, before)

    def test_chat_image_url_maps_to_both_other_protocols(self):
        items = formats.openai_chat_message_to_items({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": "https://example.test/image.png",
                    "detail": "high",
                },
            }],
        })
        message = items[0]
        _instructions, responses = (
            formats.items_to_openai_responses_parts([message]))
        self.assertEqual(
            responses[0]["content"][0],
            {
                "type": "input_image",
                "image_url": "https://example.test/image.png",
                "detail": "high",
            },
        )
        _system, anthropic = formats.items_to_anthropic_parts([message])
        self.assertEqual(
            anthropic[0]["content"][0],
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.test/image.png",
                },
            },
        )

    def test_base64_image_maps_to_all_protocols(self):
        message = formats.message_item("user", [{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAAA",
            },
        }])

        chat = formats.items_to_openai_chat_messages([message])
        _instructions, responses = (
            formats.items_to_openai_responses_parts([message]))
        _system, anthropic = formats.items_to_anthropic_parts([message])

        self.assertEqual(
            chat[0]["content"][0]["image_url"]["url"],
            "data:image/png;base64,AAAA",
        )
        self.assertEqual(
            responses[0]["content"][0]["image_url"],
            "data:image/png;base64,AAAA",
        )
        self.assertEqual(
            anthropic[0]["content"][0]["source"],
            {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAAA",
            },
        )

    def test_chat_data_url_decodes_to_portable_base64(self):
        message = formats.openai_chat_message_to_items({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,AAAA",
                },
            }],
        })[0]

        _system, anthropic = formats.items_to_anthropic_parts([message])

        self.assertEqual(
            anthropic[0]["content"][0]["source"],
            {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAAA",
            },
        )

    def test_unrepresentable_user_media_fails_request_not_selection(self):
        events = [
            formats.message_item("user", [{
                "type": "audio",
                "value": None,
            }]),
        ]
        with self.assertRaisesRegex(
                formats.ProjectionError, "cannot encode this audio"):
            formats.items_to_openai_responses_parts(events)

    def test_image_tool_result_maps_to_responses_and_anthropic(self):
        events = [
            formats.model_response_event(
                formats.OPENAI_CHAT,
                [formats.tool_call_item(
                    "call_image", "Screenshot", {})],
            ),
            formats.tool_result_item(
                "call_image",
                [{
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.test/result.png",
                    },
                }],
            ),
        ]

        _instructions, responses = (
            formats.items_to_openai_responses_parts(events))
        self.assertEqual(
            responses[1]["output"],
            [{
                "type": "input_image",
                "image_url": "https://example.test/result.png",
            }],
        )
        _system, anthropic = formats.items_to_anthropic_parts(events)
        self.assertEqual(
            anthropic[1]["content"][0]["content"],
            [{
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.test/result.png",
                },
            }],
        )


if __name__ == "__main__":
    unittest.main()
