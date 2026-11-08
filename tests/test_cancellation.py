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
import contextlib
import json
import os
import signal
import sys
import tempfile
import unittest
from unittest import mock
from response_header_fixtures import isolated_response_headers, setUpModule  # noqa: F401 - unittest hook
from process_lifecycle_fixtures import observe_fixture

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

    async def __call__(
            self, items, on_text_delta=None, *, codex_turn_state):
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

    def test_cancel_that_races_with_nonstream_response_discards_response(self):
        cancel = _CancelAfter()
        events = []
        transcript = [formats.message_item("user", "go")]

        async def response_and_cancel(
                _items, *, codex_turn_state):
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

        async def failure_and_cancel(
                _items, *, codex_turn_state):
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

        async def refusal(_items, *, codex_turn_state):
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


class SamplingOwnerCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def _held_sampling(self, mode, broker_loss):
        from loki_agent import authentications, credential_capabilities, model_requests, recoveries, terminal_frontend
        from loki_agent.acp_worker import Worker

        with isolated_response_headers():
            async with contextlib.AsyncExitStack() as cleanup:
                session = loki.current_session()
                cleanup.push_async_callback(session.close_state)
                if broker_loss:
                    server, child = await credential_capabilities.CredentialCapabilityServer.create(
                        authentications.CredentialBroker())
                    cleanup.push_async_callback(server.close)
                    client = await credential_capabilities.CredentialClient.from_fd(child)
                    cleanup.push_async_callback(client.close)
                    session.credential_authority = client
                entered, release = asyncio.Event(), asyncio.Event()
                events = []
                transcript = [formats.message_item('user', 'held request')]
                session.transcript_items = transcript
                worker = Worker(session, lambda _message: None)
                user_cancel = worker.cancel_event if mode == 'acp' else asyncio.Event()

                async def completion(_items, *_args, cancel_check, on_text_delta=None, **_kwargs):
                    if on_text_delta is not None:
                        on_text_delta('partial')
                    entered.set()
                    await release.wait()
                    self.assertTrue(cancel_check(), 'sampler lost its Session/broker cancellation predicate')
                    raise recoveries.Cancelled()

                async def run():
                    if mode == 'terminal':
                        return await terminal_frontend.run_terminal_turn_async(
                            transcript, cancel_event=user_cancel, turn_events=events)
                    if mode == 'acp':
                        return await worker._run_turn(events.append, model_requests.TurnThinkingSettings())
                    return await loki.run_tool_loop_async(
                        transcript, cancel_event=user_cancel, stream_chat=mode == 'streamed',
                        on_event=events.append)

                with mock.patch.object(loki, 'async_chat_completion', new=completion), \
                        mock.patch.object(loki, 'current_model', return_value='held-model'), \
                        mock.patch.object(terminal_frontend, '_redraw_status'), \
                        mock.patch.object(terminal_frontend, '_terminal_agent_event'):
                    task = asyncio.create_task(run())
                    try:
                        await observe_fixture(entered.wait(), task)
                        if broker_loss:
                            await server.close()
                            await client.wait_closed()
                        else:
                            session.begin_close()
                        self.assertFalse(user_cancel.is_set())
                        release.set()
                        self.assertEqual(await task, '')
                    finally:
                        release.set()
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                self.assertEqual([item['type'] for item in transcript], ['message'])
                self.assertEqual(sum(event['type'] == 'response_cancelled' for event in events), 1)

    async def test_owner_loss_reaches_each_frontend_sampler_without_user_cancel(self):
        for mode in ['buffered', 'streamed', 'terminal', 'acp']:
            with self.subTest(mode=mode):
                await self._held_sampling(mode, False)

    async def test_broker_loss_reaches_each_frontend_sampler_without_user_cancel(self):
        for mode in ['buffered', 'streamed', 'terminal', 'acp']:
            with self.subTest(mode=mode):
                await self._held_sampling(mode, True)

    async def test_repeated_task_cancellation_joins_stream_read_retirement(self):
        from loki_agent import http_client, model_requests

        started = asyncio.Event()
        retired = []

        class Reader(asyncio.StreamReader):
            async def read(self, size):
                started.set()
                try:
                    return await super().read(size)
                finally:
                    retired.append(True)

        iterator = http_client.HttpBodyStream(
            Reader(), {}, timeout=20, max_bytes=100).__aiter__()
        task = asyncio.create_task(model_requests._next_stream_chunk(iterator, lambda: False))
        try:
            await observe_fixture(started.wait(), task)
            task.cancel()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(retired, [True])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await iterator.aclose()

    async def test_owner_loss_wins_a_simultaneously_ready_stream_chunk(self):
        from loki_agent import http_client, model_requests

        with isolated_response_headers():
            session = loki.current_session()

            class Reader(asyncio.StreamReader):
                async def read(self, size):
                    session.begin_close()
                    return await super().read(size)

            reader = Reader()
            reader.feed_data(b'must not be published')
            iterator = http_client.HttpBodyStream(
                reader, {}, timeout=20, max_bytes=100,
                cancel_check=lambda: not session.active).__aiter__()
            try:
                with self.assertRaises(http_client.HttpRequestCancelled):
                    await model_requests._next_stream_chunk(iterator, lambda: not session.active)
            finally:
                await iterator.aclose()
                await session.close_state()


class ForegroundJobCancelTests(unittest.TestCase):
    """Ctrl-C must interrupt the foreground job a turn is awaiting."""

    def test_cancel_event_reaps_a_running_command(self):
        from job_fixtures import managed_jobs

        async def run(root):
            async with managed_jobs(root) as manager:
                cancel = asyncio.Event()
                # No natural completion or timeout may impersonate cancellation.
                script = "import time\nprint('ready',flush=True)\nwhile True: time.sleep(1)"
                task = asyncio.create_task(manager.run_foreground(
                    [sys.executable, '-c', script], 'held command', None,
                    description='cancel test', shell=False, cwd=root, cancel_event=cancel))
                try:
                    while True:
                        job = next(iter(manager.jobs.values()), None)
                        if job is not None and job.process is not None:
                            if 'ready' in manager._spool_tail(job, 'stdout.log'):
                                break
                            self.assertIsNone(job.process.returncode,
                                              manager._spool_tail(job, 'stderr.log'))
                        self.assertFalse(task.done(), 'command ended before readiness')
                        await asyncio.sleep(.01)
                    cancel.set()
                    job, status, _stdout, _stderr = await task
                    with open(job.metadata_path, encoding='utf-8') as stream:
                        return job, status, json.load(stream)
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        with tempfile.TemporaryDirectory() as root:
            job, status, metadata = asyncio.run(run(root))
        self.assertEqual(status, 'cancelled')
        self.assertEqual(job.status, 'cancelled')
        self.assertEqual(metadata['status'], 'cancelled')
        self.assertIsNotNone(job.exit_code)
        self.assertNotEqual(job.exit_code, 0)
        self.assertEqual(metadata['exit_code'], job.exit_code)
        if os.name == 'posix':
            self.assertEqual(job.exit_code, -signal.SIGINT)
            self.assertEqual(job.signal, signal.SIGINT)
            self.assertEqual(metadata['signal'], signal.SIGINT)

    def test_no_cancel_event_uses_timeout_path(self):
        from job_fixtures import managed_jobs

        async def run(root):
            async with managed_jobs(root) as manager:
                return await manager.run_foreground(
                    [sys.executable, '-c', 'import threading;threading.Event().wait()'], 'held command', 300,
                    description='timeout test', shell=False, cwd=root)

        with tempfile.TemporaryDirectory() as root:
            job, status, stdout, stderr = asyncio.run(run(root))
        self.assertEqual(status, 'timed_out')


if __name__ == "__main__":
    unittest.main()
