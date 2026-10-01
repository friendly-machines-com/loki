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


class ReplayFidelityTests(unittest.TestCase):
    def test_system_instruction_is_silent(self):
        events = [
            formats.instruction_item("You are a helpful system agent."),
            user_message("hello"),
        ]
        classified = replays.classify_transcript(events)
        self.assertEqual([b[1] for b in classified], ["hello"])

    def test_unknown_session_events_have_exact_terminal_and_replay_fallback(self):
        from loki_agent.savefiles import ResumeTranscriptRenderer
        for event, text in (
                ({'type': 'quantum_thought_bubble', 'payload': {'x': 1}},
                 "[Session event: quantum_thought_bubble]\n"
                 "{'payload': {'x': 1}, 'type': 'quantum_thought_bubble'}"),
                ({'type': 'future_event_kind', 'anything': True},
                 "[Session event: future_event_kind]\n"
                 "{'anything': True, 'type': 'future_event_kind'}")):
            with self.subTest(event=event['type']):
                events = [formats.instruction_item('system preamble'), event,
                          user_message('hi')]
                self.assertEqual(replays.classify_transcript(events), [
                    ('agent', text, ('session_event', event['type'])),
                    ('user', 'hi', ('message', 'user'))])
                presentation = ResumeTranscriptRenderer('Assistant').presentation(events)
                self.assertEqual([(kind, ''.join(part for _, part in segments))
                                  for kind, segments in presentation],
                                 [('other', text), ('message', 'User: hi')])

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


if __name__ == "__main__":
    unittest.main()
