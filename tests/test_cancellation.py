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
import json
import signal
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
        # A cancellation already pending before the model request wins before
        # any provider output is accepted. There is therefore no call to pair
        # and no synthetic tool result to invent.
        self.assertEqual(tool_results, [])
        self.assertEqual(chat.calls, 0)
        self.assertEqual(
            [item.get("type") for item in transcript],
            ["message"],
        )

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

    def test_cancel_that_races_with_nonstream_response_discards_response(self):
        cancel = _CancelAfter()
        events = []
        transcript = [formats.message_item("user", "go")]

        async def response_and_cancel(_items):
            cancel.cancelled = True
            return [
                formats.message_item(
                    "assistant", "must not be accepted"),
            ]

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=response_and_cancel,
            on_event=events.append,
            cancel_check=cancel,
        ))

        self.assertEqual(result, "")
        self.assertEqual(
            [item.get("role") for item in transcript], ["user"])
        self.assertEqual(
            [event["type"] for event in events],
            ["response_cancelled"],
        )

    def test_cancel_dominates_simultaneous_transport_failure(self):
        cancel = _CancelAfter()
        events = []

        async def failure_and_cancel(_items):
            cancel.cancelled = True
            raise ConnectionResetError("socket closed during cancellation")

        result = asyncio.run(loki.run_tool_loop_async(
            [formats.message_item("user", "go")],
            chat_fn=failure_and_cancel,
            on_event=events.append,
            cancel_check=cancel,
        ))

        self.assertEqual(result, "")
        self.assertEqual(
            [event["type"] for event in events],
            ["response_cancelled"],
        )

    def test_provider_refusal_has_distinct_terminal_event(self):
        events = []

        async def refusal(_items):
            return formats.DecodedTurn([
                formats.message_item(
                    "assistant",
                    [{"type": "refusal", "text": "I cannot help."}],
                ),
            ])

        result = asyncio.run(loki.run_tool_loop_async(
            [formats.message_item("user", "request")],
            chat_fn=refusal,
            on_event=events.append,
        ))

        self.assertEqual(result, "I cannot help.")
        self.assertEqual(
            [event["type"] for event in events],
            ["assistant_message", "response_refusal"],
        )


class ForegroundJobCancelTests(unittest.TestCase):
    """Ctrl-C must interrupt the foreground job a turn is awaiting."""

    def test_cancel_event_kills_sleep_quickly(self):
        import time as _time
        from loki_agent.loki import JobManager

        async def run():
            manager = JobManager("/tmp/loki-cancel-test-jobs2")
            cancel = asyncio.Event()
            start = _time.monotonic()

            async def scenario():
                return await manager.run_foreground(
                    ["sleep", "30"], "sleep 30", 60000,
                    description="cancel test", shell=False, cwd="/tmp",
                    cancel_event=cancel)

            task = asyncio.get_running_loop().create_task(scenario())
            await asyncio.sleep(0.7)  # let the job start
            cancel.set()
            job, status, _stdout, _stderr = await task
            with open(job.metadata_path, encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            return (
                job,
                status,
                metadata,
                _time.monotonic() - start,
            )

        job, status, metadata, elapsed = asyncio.run(run())
        self.assertEqual(status, "cancelled")
        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.exit_code, -signal.SIGINT)
        self.assertEqual(job.signal, signal.SIGINT)
        self.assertEqual(metadata["status"], "cancelled")
        self.assertEqual(metadata["exit_code"], -signal.SIGINT)
        self.assertEqual(metadata["signal"], signal.SIGINT)
        self.assertLess(elapsed, 5.0)  # SIGINT kills sleep immediately

    def test_no_cancel_event_uses_timeout_path(self):
        from loki_agent.loki import JobManager

        async def run():
            manager = JobManager("/tmp/loki-cancel-test-jobs3")
            return await manager.run_foreground(
                ["sleep", "5"], "sleep 5", 300,  # 300ms timeout
                description="timeout test", shell=False, cwd="/tmp")

        job, status, stdout, stderr = asyncio.run(run())
        self.assertEqual(status, "timed_out")


if __name__ == "__main__":
    unittest.main()
