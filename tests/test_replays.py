"""Replay-fidelity tests: exotic transcript items must never vanish.

The contract under test: `replays.classify_transcript` consumes the same
canonical transcript list that ResumeTranscriptRenderer renders for the
terminal, so every user-visible thing the terminal resume shows must appear
in the replay classification -- possibly as degraded text, never as
silence -- and everything the terminal silently skips must be skipped here
too.
"""

import unittest

from loki_agent import formats, replays


def user_message(text):
    return formats.message_item("user", text)


def assistant_message(text):
    return formats.message_item("assistant", text)


class ReplayFidelityTests(unittest.TestCase):
    def test_plain_conversation_round_trips(self):
        events = [
            user_message("hello"),
            assistant_message("hi there"),
            user_message("do a thing"),
            assistant_message("done"),
        ]
        classified = replays.classify_transcript(events)
        kinds = [b[0] for b in classified]
        self.assertEqual(kinds, ["user", "agent", "user", "agent"])
        self.assertEqual(classified[0][1], "hello")
        self.assertEqual(classified[1][1], "hi there")

    def test_system_instruction_is_silent(self):
        events = [
            formats.instruction_item("You are a helpful system agent."),
            user_message("hello"),
        ]
        classified = replays.classify_transcript(events)
        self.assertEqual([b[1] for b in classified], ["hello"])

    def test_unknown_session_event_degrades_to_visible_text(self):
        exotic = {"type": "quantum_thought_bubble", "payload": {"x": 1}}
        classified = replays.classify_transcript([exotic, user_message("hi")])
        self.assertEqual(len(classified), 2)
        self.assertIn("quantum_thought_bubble", classified[0][1])
        self.assertEqual(classified[0][0], "agent")

    def test_image_and_audio_content_is_placeholder_not_silence(self):
        events = [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image", "source": {"data": "..."}},
            ],
        }]
        classified = replays.classify_transcript(events)
        texts = [b[1] for b in classified]
        self.assertIn("what is this", texts)
        self.assertIn("[Image content]", texts)

    def test_provider_reasoning_is_silent_on_replay(self):
        # Mirrors ResumeTranscriptRenderer: openai_reasoning without a
        # summary and anthropic thinking items are not replayed.
        events = [{
            "type": "model_response",
            "model": "m",
            "items": [
                {"type": "openai_reasoning", "value": {"summary": None}},
                {"type": "anthropic_thinking", "value": "hidden"},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "text", "text": "answer"}]},
            ],
        }]
        classified = replays.classify_transcript(events)
        self.assertEqual([b[1] for b in classified], ["answer"])

    def test_provider_notice_is_not_fabricated_as_acp_agent_text(self):
        event = formats.model_response_event(
            formats.OPENAI_RESPONSES,
            [],
            protocol_data={
                "loki": {
                    "provider_notices": [
                        formats.TRUSTED_ACCESS_FOR_CYBER,
                    ],
                },
            },
        )

        self.assertEqual(replays.classify_transcript([event]), [])

    def test_incomplete_response_status_is_visible(self):
        events = [{
            "type": "model_response",
            "model": "m",
            "status": "incomplete",
            "items": [
                {"type": "message", "role": "assistant",
                 "content": [{"type": "text", "text": "partial"}]},
            ],
        }]
        classified = replays.classify_transcript(events)
        texts = [b[1] for b in classified]
        self.assertIn("[Model response incomplete]", texts)
        self.assertIn("partial", texts)

    def test_tool_call_and_result_are_tool_blocks(self):
        events = [
            user_message("list files"),
            {
                "type": "model_response",
                "model": "m",
                "items": [
                    {"type": "function_call", "name": "Bash",
                     "arguments": {"command": "ls"}, "call_id": "c1"},
                ],
            },
            {
                "type": "tool_result",
                "call_id": "c1",
                "name": "Bash",
                "content": [{"type": "text", "text": "file1\nfile2"}],
            },
        ]
        classified = replays.classify_transcript(events)
        tool_blocks = [b for b in classified if b[0] == "tool"]
        self.assertEqual(len(tool_blocks), 2)
        self.assertEqual(tool_blocks[0][1], "Bash")
        self.assertEqual(tool_blocks[0][2], "c1")
        self.assertIn("file1", tool_blocks[1][1])
        self.assertEqual(tool_blocks[1][2], "c1")

    def test_tool_error_result_is_labeled(self):
        events = [{
            "type": "tool_result",
            "call_id": "c9",
            "name": "Write",
            "is_error": True,
            "content": [{"type": "text", "text": "boom"}],
        }]
        classified = replays.classify_transcript(events)
        self.assertEqual(classified[0][0], "tool")
        self.assertIn("Tool error", classified[0][1])
        self.assertIn("boom", classified[0][1])

    def test_provider_tool_call_variant_renders_name(self):
        # Provider-executed tools (execution != "client") still classify as
        # tool blocks with the tool name; the ACP replay must show them.
        events = [{
            "type": "model_response",
            "model": "m",
            "items": [
                {"type": "function_call", "name": "web_search",
                 "arguments": {"query": "x"}, "call_id": "c2",
                 "execution": "provider"},
            ],
        }]
        classified = replays.classify_transcript(events)
        self.assertEqual(classified[0][0], "tool")
        self.assertEqual(classified[0][1], "web_search")

    def test_every_terminal_visible_block_has_replay_counterpart(self):
        # Cross-renderer invariant: for a transcript with one of everything,
        # the union of visible text in the terminal renderer is covered by
        # the replay classification (degraded forms allowed, silence not).
        from loki_agent.savefiles import ResumeTranscriptRenderer
        events = [
            formats.instruction_item("system preamble"),
            user_message("go"),
            {
                "type": "model_response",
                "model": "m",
                "status": "completed",
                "items": [
                    {"type": "message", "role": "assistant",
                     "content": [{"type": "text", "text": "doing"}]},
                    {"type": "function_call", "name": "Read",
                     "arguments": {"file_path": "/tmp/x"},
                     "call_id": "c3"},
                ],
            },
            {
                "type": "tool_result",
                "call_id": "c3",
                "name": "Read",
                "content": [{"type": "text", "text": "contents"}],
            },
            {"type": "future_event_kind", "anything": True},
        ]
        terminal_text = ResumeTranscriptRenderer(
            assistant_label="Assistant").render(events)
        replay_blocks = replays.classify_transcript(events)
        replay_text = "\n".join(b[1] for b in replay_blocks)

        for needle in ["go", "doing", "Read", "contents",
                       "future_event_kind"]:
            self.assertIn(needle, terminal_text, f"terminal lost {needle}")
            self.assertIn(needle, replay_text, f"replay lost {needle}")
        # System preamble must be silent in BOTH renderers.
        self.assertNotIn("system preamble", terminal_text)
        self.assertNotIn("system preamble", replay_text)


if __name__ == "__main__":
    unittest.main()
