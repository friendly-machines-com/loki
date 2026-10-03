"""Terminal and ACP use one model-control state and one admission snapshot."""

import contextlib
import io
import unittest
from unittest import mock

from loki_agent import acp_commands, acp_events, acp_worker, formats, loki, protocols, terminal_frontend
from loki_agent.sessions import Session


class ThinkingSurfaceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = Session(runtime_config=loki.make_runtime_config(
            "https://api.anthropic.com/v1", protocols.ANTHROPIC_MESSAGES, model="claude-opus-4-5"))
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)

    def worker(self, write=None):
        worker = acp_worker.Worker(self.session, write or (lambda message: None))
        worker._model_options = [{"value": "model", "name": "Model"}]
        worker._current_option_value = "model"
        return worker

    async def test_manual_picker_collects_allowance_atomically_and_cancels_without_change(self):
        for replies, expected in [[['1', '2', '2048'], 2048], [['1', '2', ''], None]]:
            modal = mock.Mock(prompt=mock.AsyncMock(side_effect=replies))

            @contextlib.asynccontextmanager
            async def scope():
                yield modal

            self.session.thinking_mode = None
            self.session.thinking_budget = None
            with contextlib.redirect_stdout(io.StringIO()):
                text = await terminal_frontend.run_thinking_picker_async(mock.Mock(modal=scope))
            self.assertEqual(self.session.thinking_budget, expected)
            if expected is None:
                self.assertIsNone(self.session.thinking_mode)
                self.assertIn("cancelled", text)
            else:
                self.assertEqual(self.session.thinking_mode, "manual")

    async def test_local_commands_do_not_consume_trials_or_enter_transcript(self):
        self.session.runtime_config = loki.make_runtime_config(
            "https://relay.example/v1/messages", protocols.ANTHROPIC_MESSAGES, model="custom")
        loki.thinking_command("mode adaptive")
        pending = self.session.thinking_request
        messages = []
        worker = self.worker(messages.append)
        for text in ["/trace thinking on", "/thinking", "/thinking on", "/trace thinking wrong"]:
            await worker.handle({"id": 1, "method": "session/prompt", "params": {
                "prompt": [{"type": "text", "text": text}]}}, concurrent=True)
            await worker._prompt_task
            self.assertIs(self.session.thinking_request, pending)
        self.assertEqual(self.session.transcript_items, [])
        with self.assertRaises(acp_worker.acps.TransportError):
            await worker.prompt({"prompt": [{"type": "text", "text": 1}]})
        self.assertIs(self.session.thinking_request, pending)

    async def test_acp_snapshot_is_captured_before_scheduling_and_not_mutated(self):
        loki.thinking_command("mode manual budget 2048")
        worker = self.worker()
        captured = []

        async def run(on_event, thinking):
            captured.append(thinking)
            return ""

        worker._run_turn = run
        with mock.patch.object(loki, "run_turn_end_hooks_async", mock.AsyncMock()):
            await worker.handle({"id": 1, "method": "session/prompt", "params": {
                "prompt": [{"type": "text", "text": "hello"}]}}, concurrent=True)
            worker.set_config_option({"configId": "thinking_mode", "value": "off"})
            worker.set_config_option({"configId": "reasoning_traces", "value": "on"})
            await worker._prompt_task
        self.assertEqual(captured[0].mode, "manual")
        self.assertEqual(captured[0].budget, 2048)
        self.assertEqual(captured[0].traces, "off")
        self.assertEqual(self.session.thinking_mode, "off")
        self.assertEqual(self.session.reasoning_traces, "on")

    async def test_terminal_continuations_receive_same_snapshot(self):
        loki.thinking_command("mode manual budget 2048")
        snapshots = []

        async def completion(items, *args, **kwargs):
            snapshots.append(kwargs["thinking"])
            if len(snapshots) == 1:
                self.session.thinking_mode = "off"
                self.session.reasoning_traces = "on"
                return formats.DecodedTurn([formats.tool_call_item("call", "Read", {"file_path": "x"})])
            return formats.DecodedTurn([formats.message_item("assistant", "answer")])

        execute = mock.AsyncMock(return_value=({"ok": True, "content": "ok"}, {}))
        with mock.patch.object(terminal_frontend, "async_chat_completion", completion), mock.patch.object(
                loki, "execute_tool_call_async", execute), mock.patch.object(
                terminal_frontend.terminals, "redraw_status_bar"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(await terminal_frontend.run_terminal_turn_async([]), "answer")
        self.assertIs(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[1].mode, "manual")
        self.assertEqual(snapshots[1].budget, 2048)
        self.assertEqual(snapshots[1].traces, "off")

    async def test_trace_controls_and_retention_default_are_separate(self):
        self.session.runtime_config = loki.make_runtime_config(
            "https://api.z.ai/api/paas/v4", protocols.OPENAI_CHAT, model="glm-5.2")
        loki.thinking_command("retention preserve")
        loki.thinking_command("retention default")
        self.assertEqual(self.session.reasoning_retention, "default")
        outcome = await acp_commands.run("/trace thinking on", self.session)
        self.assertIn("Thinking traces: on", outcome.text)
        self.assertIsNone(self.session.thinking_mode)
        self.assertIsNone(self.session.thinking_budget)
        self.assertEqual(self.session.transcript_items, [])

    async def test_thought_updates_and_control_sequence_neutralization(self):
        state = {}
        chunks = []
        for text in ["first", "second"]:
            for event in [{"type": "reasoning_start"}, {"type": "reasoning_delta", "text": text}]:
                chunks.extend(acp_events.map_event("s", event, state))
        self.assertEqual("".join(chunk["update"]["content"]["text"] for chunk in chunks), "first\n\nsecond")
        self.assertTrue(all(chunk["update"]["sessionUpdate"] == "agent_thought_chunk" for chunk in chunks))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            terminal_frontend._terminal_agent_event({"type": "reasoning_start", "kind": "thinking"})
            terminal_frontend._terminal_agent_event({"type": "reasoning_delta", "text": "trace\x1b[2J"})
            terminal_frontend._terminal_agent_event({"type": "reasoning_end", "complete": True})
        self.assertIn("Thinking:", output.getvalue())
        self.assertNotIn("\x1b[2J", output.getvalue())
