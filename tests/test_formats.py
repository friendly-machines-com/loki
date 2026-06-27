import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from day_agent import formats


def rendered_text(value):
    return repr(value)


def item_types(items):
    return [item.get("type") for item in items]


class TranscriptFormatTests(unittest.TestCase):
    def test_current_log_schema_roundtrips_without_migration(self):
        items = [
            formats.message_item("assistant", "Need the file."),
            formats.tool_call_item("call_1", "Read", {"file_path": "README.md"}),
        ]
        blob = formats.new_log_blob(items, ["keep"])

        loaded_items, loaded_todos = formats.load_log_blob(blob)

        self.assertEqual(blob["schema"], "day-agent.transcript.v2")
        self.assertEqual(loaded_todos, ["keep"])
        self.assertEqual(item_types(loaded_items), ["message", "tool_call"])
        self.assertEqual(loaded_items[1]["call_id"], "call_1")
        self.assertEqual(loaded_items[1]["input"], {"file_path": "README.md"})

    def test_old_log_formats_are_rejected(self):
        old_blobs = [
            {
                "schema": "day-agent.transcript.v1",
                "items": [{"type": "message", "role": "user", "content": "old"}],
            },
            {"messages": [{"role": "user", "content": "old"}]},
            [{"role": "user", "content": "old"}],
            {"items": [{"type": "message", "role": "user", "content": "old"}]},
        ]

        for blob in old_blobs:
            with self.subTest(blob=blob):
                with self.assertRaises(formats.TranscriptFormatError):
                    formats.load_log_blob(blob)

    def test_invalid_current_log_shapes_are_rejected(self):
        invalid_blobs = [
            {"schema": "day-agent.transcript.v2", "items": {}},
            {"schema": "day-agent.transcript.v2", "items": [], "session_todos": {}},
        ]

        for blob in invalid_blobs:
            with self.subTest(blob=blob):
                with self.assertRaises(formats.TranscriptFormatError):
                    formats.load_log_blob(blob)

    def test_openai_chat_tool_calls_roundtrip_through_v2(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": "Need the file.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path":"README.md"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
        ]

        items = formats.openai_chat_messages_to_items(messages)
        rendered = formats.items_to_openai_chat_messages(items)

        self.assertEqual(item_types(items), ["instruction", "message", "message", "tool_call", "tool_result"])
        self.assertEqual(rendered[2]["role"], "assistant")
        self.assertEqual(rendered[2]["content"], "Need the file.")
        self.assertEqual(rendered[2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(rendered[2]["tool_calls"][0]["function"]["name"], "Read")
        self.assertEqual(rendered[3], {"role": "tool", "tool_call_id": "call_1", "content": "contents"})

    def test_openai_chat_v2_projects_to_anthropic_tool_use(self):
        items = [
            formats.instruction_item("sys"),
            formats.message_item("user", "read it"),
            formats.message_item("assistant", "Need the file."),
            formats.tool_call_item("call_1", "Read", {"file_path": "README.md"}),
            formats.tool_result_item("call_1", "contents"),
        ]

        system, messages = formats.items_to_anthropic_parts(items)

        self.assertEqual(system, "sys")
        self.assertEqual(messages[0], {"role": "user", "content": [{"type": "text", "text": "read it"}]})
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"][0], {"type": "text", "text": "Need the file."})
        self.assertEqual(
            messages[1]["content"][1],
            {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "README.md"}},
        )
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[2]["content"][0]["type"], "tool_result")
        self.assertEqual(messages[2]["content"][0]["tool_use_id"], "call_1")

    def test_anthropic_tool_use_projects_to_openai_chat_tool_call(self):
        response = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Need the file."},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "README.md"}},
            ],
        }

        items = formats.anthropic_response_to_items(response)
        rendered = formats.items_to_openai_chat_messages(items)

        self.assertEqual(item_types(items), ["response_metadata", "message", "tool_call"])
        self.assertIn("Transcript metadata preserved", rendered[0]["content"])
        self.assertEqual(rendered[1]["role"], "assistant")
        self.assertEqual(rendered[1]["content"], "Need the file.")
        self.assertEqual(rendered[1]["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(rendered[1]["tool_calls"][0]["function"]["name"], "Read")

    def test_openai_responses_response_parses_known_item_kinds(self):
        response = {
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "model": "gpt-test",
            "output": [
                {"id": "rs_1", "type": "reasoning", "summary": [{"text": "reason_marker"}]},
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                },
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file_path":"README.md"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "contents"},
                {"id": "cc_1", "type": "custom_tool_call", "call_id": "call_2", "name": "custom", "input": "raw"},
                {"type": "custom_tool_call_output", "call_id": "call_2", "output": "custom result"},
                {"id": "ws_1", "type": "web_search_call", "status": "completed", "query": "marker"},
                {"id": "unknown_1", "type": "future_output_item", "future_marker": True},
            ],
        }

        items = formats.openai_responses_response_to_items(response)

        self.assertEqual(
            item_types(items),
            [
                "response_metadata",
                "reasoning",
                "message",
                "tool_call",
                "tool_result",
                "tool_call",
                "tool_result",
                "tool_call",
                "provider_item",
            ],
        )
        self.assertEqual(items[3]["tool_kind"], "function")
        self.assertEqual(items[5]["tool_kind"], "custom")
        self.assertEqual(items[7]["tool_kind"], "web_search_call")
        self.assertEqual(items[8]["provider"], "openai_responses")

    def test_v2_renders_to_openai_responses_input_items(self):
        items = [
            formats.instruction_item("sys"),
            formats.message_item("user", "hi"),
            formats.message_item("assistant", "Need file."),
            formats.tool_call_item("call_1", "Read", {"file_path": "README.md"}),
            formats.tool_result_item("call_1", "contents"),
            formats.tool_call_item("call_2", "custom", "raw input", tool_kind="custom"),
            formats.tool_result_item("call_2", "custom output", tool_kind="custom"),
            formats.reasoning_item(summary=[{"text": "reason_marker"}]),
            formats.provider_item("openai_responses", {"type": "future_item", "provider_marker": True}),
        ]

        instructions, input_items = formats.items_to_openai_responses_parts(items)

        self.assertEqual(instructions, "sys")
        self.assertEqual(
            [item.get("type") for item in input_items],
            [
                "message",
                "message",
                "function_call",
                "function_call_output",
                "custom_tool_call",
                "custom_tool_call_output",
                "reasoning",
                "future_item",
            ],
        )
        self.assertEqual(input_items[2]["name"], "Read")
        self.assertEqual(input_items[3]["output"], "contents")
        self.assertEqual(input_items[6]["summary"], [{"text": "reason_marker"}])
        self.assertTrue(input_items[7]["provider_marker"])

    def non_native_items(self):
        return [
            formats.response_metadata_item(
                "openai",
                "openai_responses",
                {"id": "resp_marker", "object": "response", "status": "incomplete", "model": "gpt-test"},
            ),
            formats.reasoning_item(summary=[{"text": "reason_marker"}]),
            formats.message_item(
                "user",
                [
                    {"type": "image", "provider": "openai_responses", "value": {"image_marker": True}},
                    {"type": "weird_block", "value": "weird_marker"},
                ],
            ),
            formats.message_item("assistant", "about to use a built-in tool"),
            formats.tool_call_item(
                "call_builtin",
                "web_search",
                {"query": "builtin_marker"},
                tool_kind="web_search_call",
            ),
            formats.tool_result_item("call_builtin", "result_marker", tool_kind="web_search_call"),
            formats.provider_item("other_provider", {"provider_marker": True}),
            {"type": "future_semantic_item", "payload": "unknown_marker"},
        ]

    def assert_all_markers_visible(self, value):
        text = rendered_text(value)
        for marker in [
            "resp_marker",
            "reason_marker",
            "image_marker",
            "weird_marker",
            "builtin_marker",
            "result_marker",
            "provider_marker",
            "unknown_marker",
        ]:
            self.assertIn(marker, text)

    def test_non_native_items_render_as_explicit_text_for_openai_chat(self):
        rendered = formats.items_to_openai_chat_messages(self.non_native_items())

        self.assert_all_markers_visible(rendered)
        text = rendered_text(rendered)
        self.assertIn("Transcript metadata preserved", text)
        self.assertIn("Previous assistant reasoning item", text)
        self.assertIn("Previous assistant tool call", text)
        self.assertIn("Previous tool result", text)
        self.assertIn("Provider-specific transcript item", text)
        self.assertIn("Transcript item not native to openai_chat", text)

    def test_non_native_items_render_as_explicit_text_for_anthropic(self):
        _system, messages = formats.items_to_anthropic_parts(self.non_native_items())

        self.assert_all_markers_visible(messages)
        text = rendered_text(messages)
        self.assertIn("Transcript metadata preserved", text)
        self.assertIn("Previous assistant reasoning item", text)
        self.assertIn("Previous assistant tool call", text)
        self.assertIn("Previous tool result", text)
        self.assertIn("Provider-specific transcript item", text)
        self.assertIn("Transcript item not native to anthropic_messages", text)

    def test_non_native_items_render_as_explicit_input_text_for_responses(self):
        _instructions, input_items = formats.items_to_openai_responses_parts(self.non_native_items())

        self.assert_all_markers_visible(input_items)
        text = rendered_text(input_items)
        self.assertIn("Transcript metadata preserved", text)
        self.assertTrue(any(item.get("type") == "reasoning" for item in input_items))
        self.assertIn("Previous tool result", text)
        self.assertIn("Provider-specific transcript item", text)
        self.assertIn("Transcript item not native to openai_responses", text)

    def test_unknown_content_blocks_do_not_crash_or_disappear(self):
        items = [
            formats.message_item(
                "user",
                [{"type": "unknown_block", "payload": {"content_marker": True}}],
            )
        ]

        chat = formats.items_to_openai_chat_messages(items)
        _system, anthropic = formats.items_to_anthropic_parts(items)
        _instructions, responses = formats.items_to_openai_responses_parts(items)

        self.assertIn("content_marker", rendered_text(chat))
        self.assertIn("content_marker", rendered_text(anthropic))
        self.assertIn("content_marker", rendered_text(responses))
        self.assertIn("Transcript content not native", rendered_text(chat))
        self.assertIn("Transcript content not native", rendered_text(anthropic))
        self.assertIn("Transcript content not native", rendered_text(responses))


if __name__ == "__main__":
    unittest.main()
