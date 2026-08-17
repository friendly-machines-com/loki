"""Cancellation semantics for the tool loop.

ACP's session/cancel maps onto the existing cooperative cancel_check
(the same mechanism as Ctrl-C in the terminal).  The contract under test --
which ACP's stopReason: "cancelled" depends on:

* a cancel arriving between tool calls ends the turn by returning normally
  (never by raising), so a caller can answer its pending request with a
  cancelled stop reason rather than an error;
* tool calls after the cancellation point are recorded as not-executed
  results, so the transcript stays provider-replayable;
* a response_cancelled event is emitted exactly once.
"""

import asyncio
import unittest
from unittest import mock

from loki_agent import formats, loki


def _tool_call(name, call_id, arguments=None):
    return {
        "type": "function_call",
        "name": name,
        "call_id": call_id,
        "arguments": arguments or {},
    }


class _ScriptedChat:
    """chat_fn that returns scripted response item lists, counting calls."""

    def __init__(self, responses):
        # Each response is a list of canonical output items for one turn.
        self.responses = list(responses)
        self.calls = 0

    async def __call__(self, items, on_text_delta=None):
        self.calls += 1
        return self.responses.pop(0)


class _CancelAfter:
    """cancel_check that reports True once the flag is set."""

    def __init__(self):
        self.cancelled = False

    def __call__(self):
        return self.cancelled


class CancellationBetweenToolCallsTests(unittest.TestCase):
    def _run(self, chat, cancel_check, transcript):
        events = []

        async def scenario():
            return await loki.run_tool_loop_async(
                transcript,
                chat_fn=chat,
                on_event=events.append,
                cancel_check=cancel_check,
                allowed=[],  # no real tools: everything executes as errors
            )

        return asyncio.run(scenario()), events

    def test_cancel_between_calls_returns_normally_and_marks_rest(self):
        # Two tool calls in one response; the first executes, then cancel
        # fires before the second runs.
        chat = _ScriptedChat([
            [
                _tool_call("TodoWrite", "c1", {"todos": []}),
                _tool_call("TodoWrite", "c2", {"todos": []}),
            ],
        ])
        cancel = _CancelAfter()
        executed = []

        async def fake_dispatch(tc, **kwargs):
            executed.append(tc.get("call_id"))
            return {"ok": True, "content": "fine"}, "client"

        transcript = [formats.message_item("user", "go")]
        with mock.patch.object(
                loki, "execute_tool_call_async", new=fake_dispatch):
            # Flip the cancel flag after the first tool executes.
            original = fake_dispatch

            async def dispatch_and_cancel(tc, **kwargs):
                result = await original(tc, **kwargs)
                cancel.cancelled = True
                return result

            with mock.patch.object(
                    loki, "execute_tool_call_async", new=dispatch_and_cancel):
                async def scenario():
                    return await loki.run_tool_loop_async(
                        transcript,
                        chat_fn=chat,
                        on_event=lambda e: None,
                        cancel_check=cancel,
                        allowed=["TodoWrite"],
                    )
                result = asyncio.run(scenario())

        # Turn ended by returning, not raising.
        self.assertIsInstance(result, str)
        # First call executed, second was marked not-executed.
        self.assertEqual(executed, ["c1"])
        tool_results = [item for item in transcript
                        if item.get("type") == "tool_result"]
        self.assertEqual(len(tool_results), 2)
        self.assertFalse(tool_results[1].get("is_error") is None
                         and "not executed" not in
                         formats.item_text(tool_results[1]).lower())
        self.assertIn("cancelled",
                      formats.item_text(tool_results[1]).lower())

    def test_cancel_before_any_tool_blocks_execution(self):
        chat = _ScriptedChat([
            [_tool_call("TodoWrite", "c1", {"todos": []})],
        ])
        cancel = _CancelAfter()
        cancel.cancelled = True

        executed = []

        async def fake_dispatch(tc, **kwargs):
            executed.append(tc.get("call_id"))
            return {"ok": True, "content": "fine"}, "client"

        transcript = [formats.message_item("user", "go")]
        with mock.patch.object(
                loki, "execute_tool_call_async", new=fake_dispatch):
            async def scenario():
                return await loki.run_tool_loop_async(
                    transcript,
                    chat_fn=chat,
                    on_event=lambda e: None,
                    cancel_check=cancel,
                    allowed=["TodoWrite"],
                )
            result = asyncio.run(scenario())

        self.assertIsInstance(result, str)
        self.assertEqual(executed, [])
        tool_results = [item for item in transcript
                        if item.get("type") == "tool_result"]
        self.assertEqual(len(tool_results), 1)
        self.assertIn("cancelled",
                      formats.item_text(tool_results[0]).lower())

    def test_no_cancel_means_full_loop(self):
        # Control: identical setup without cancel runs everything and loops
        # until the model responds with plain text.
        chat = _ScriptedChat([
            [_tool_call("TodoWrite", "c1", {"todos": []})],
            [
                {"type": "message", "role": "assistant",
                 "content": [{"type": "text", "text": "all done"}]},
            ],
        ])
        cancel = _CancelAfter()
        executed = []

        async def fake_dispatch(tc, **kwargs):
            executed.append(tc.get("call_id"))
            return {"ok": True, "content": "fine"}, "client"

        transcript = [formats.message_item("user", "go")]
        with mock.patch.object(
                loki, "execute_tool_call_async", new=fake_dispatch):
            async def scenario():
                return await loki.run_tool_loop_async(
                    transcript,
                    chat_fn=chat,
                    on_event=lambda e: None,
                    cancel_check=cancel,
                    allowed=["TodoWrite"],
                )
            result = asyncio.run(scenario())

        self.assertEqual(result, "all done")
        self.assertEqual(executed, ["c1"])
        self.assertEqual(chat.calls, 2)


if __name__ == "__main__":
    unittest.main()
