"""Ctrl+C dismisses command dialogs without ending the input session."""

import asyncio
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from response_header_fixtures import setUpModule as set_up_response_headers
from settings_fixtures import setUpModule as set_up_default_settings
from loki_agent import endpoint_pins, loki, models, protocols, provider_controls
from loki_agent import terminal_frontend, terminals
from loki_agent.credentials import CredentialStore


def setUpModule():
    set_up_response_headers()
    set_up_default_settings()


def answer(text, *, prefilled=False):
    events = [terminals.KeyEvent("BACKSPACE_WORD")] if prefilled else []
    return events + [terminals.KeyEvent("TEXT", text), terminals.KeyEvent("ENTER")]


class KeyboardSession(terminals.InputSession):
    """Real modal, producer and prompt; replace only terminal resources/keys."""

    def __init__(self, events, command=None, interrupt=KeyboardInterrupt):
        super().__init__(fd=0)
        self.dialog_events = list(events)
        self.normal_events = asyncio.Queue()
        self.modals = []
        self.command = command
        self.interrupt = interrupt
        self.reader = self
        self.cancel_requested = False
        self.cancel_event = asyncio.Event()
        self.normal_keys_after_interrupt = 0

    async def __aenter__(self):
        if self.command:
            self.user_messages.put_nowait(self.command)
        self._producer = asyncio.create_task(self._produce())
        return self

    async def read_key(self):
        if self._modal is not None:
            if self._modal not in self.modals:
                self.modals.append(self._modal)
            event = self.dialog_events.pop(0)
            if event.kind == "CTRL_C":
                self.cancel_requested = True
                self.cancel_event.set()
                # Only a resumed normal producer can deliver this next input.
                for key in answer("/quit" if self.command else "next message"):
                    self.normal_events.put_nowait(key)
                if self.interrupt is not KeyboardInterrupt:
                    raise self.interrupt()
            return event
        event = await self.normal_events.get()
        if self.cancel_requested:
            self.normal_keys_after_interrupt += 1
        return event


class DialogCancellationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.agent_session = loki.Session(
            shell_cwd=self.directory.name,
            job_manager=loki.JobManager(self.directory.name),
            runtime_config=loki.make_runtime_config(
                "https://api.anthropic.com/v1", protocols.ANTHROPIC_MESSAGES,
                model="claude-opus-4-5"),
        )
        self.addAsyncCleanup(self.agent_session.job_manager.close_session_owned)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(loki, "_DEFAULT_SESSION", self.agent_session))
        self.stack.enter_context(mock.patch.object(terminals.os, "isatty", return_value=False))
        self.stack.enter_context(mock.patch.object(terminals, "open_terminal_stdin"))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    async def run_dialog(self, operation):
        try:
            return await operation
        except KeyboardInterrupt:
            # Report a regression without letting asyncio's Task re-raise
            # KeyboardInterrupt into the test runner's event loop.
            self.fail("Ctrl+C escaped the command workflow")

    def assert_modal_released(self, inputs):
        self.assertIsNone(inputs._modal)
        self.assertTrue(inputs.modals)
        for modal in inputs.modals:
            self.assertFalse(modal.active)
            self.assertFalse(modal.reading)
        self.assertFalse(inputs.dialog_events)

    async def assert_next_input(self, inputs):
        self.assert_modal_released(inputs)
        self.assertEqual(await asyncio.wait_for(inputs.user_messages.get(), 1), "next message")
        self.assertGreater(inputs.normal_keys_after_interrupt, 0)

    async def run_model_dialog(self, events, *, offline=False, interrupt=KeyboardInterrupt):
        credentials = CredentialStore({
            "LOKI_API_BASE": "https://current.invalid/v1/chat/completions",
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_API_KEY": "test-key", "LOKI_MODEL": "current-model",
            "ACME_API_KEY": "catalog-key",
        })
        catalog = {"acme": {
            "id": "acme", "name": "Acme", "npm": "@ai-sdk/openai-compatible",
            "api": "https://acme.invalid/v1", "env": ["ACME_API_KEY"],
            "models": {"m": {"id": "m", "name": "M"}},
        }}
        index = mock.AsyncMock(return_value=(catalog, models.build_groups(catalog)))
        if offline:
            index.side_effect = OSError("offline")
        inputs = KeyboardSession(events, command="/model", interrupt=interrupt)
        with mock.patch.object(loki, "CREDENTIALS", credentials), \
                mock.patch.dict(os.environ, {"XDG_STATE_HOME": self.directory.name}), \
                mock.patch.object(terminal_frontend, "input_session", return_value=inputs), \
                mock.patch.object(terminal_frontend, "new_chat_log_path",
                                  return_value=os.path.join(self.directory.name, "chat.json")), \
                mock.patch.object(terminal_frontend, "restore_output_area_after_input"), \
                mock.patch.object(terminal_frontend, "explicit_connection_option", return_value=None), \
                mock.patch.object(models, "ensure_index", index), \
                mock.patch.object(terminal_frontend, "load_models_async",
                                  mock.AsyncMock(return_value=["current-model", "other-model"])), \
                mock.patch.object(endpoint_pins, "record") as record, \
                mock.patch.object(terminal_frontend, "run_terminal_turn_async") as turn:
            if interrupt is asyncio.CancelledError:
                with self.assertRaises(asyncio.CancelledError):
                    await terminal_frontend.async_main([])
            else:
                self.assertEqual(await asyncio.wait_for(
                    self.run_dialog(terminal_frontend.async_main([])), 2), 0)
                self.assertGreater(inputs.normal_keys_after_interrupt, 0)
                self.assertEqual(self.stderr.getvalue().count("Model selection cancelled."), 1)
            self.assertEqual(loki.current_model(), "current-model")
            self.assertEqual(loki.current_config().model, "current-model")
            self.assertEqual(self.agent_session.session_state["connection"]["model"], "current-model")
            record.assert_not_called()
            turn.assert_not_called()
            self.assert_modal_released(inputs)

    async def test_model_ctrl_c_at_each_selection_stage(self):
        for prior in [[], answer("1", prefilled=True),
                      answer("1", prefilled=True) + answer("1")]:
            with self.subTest(stage=len(prior)):
                self.stderr.seek(0)
                self.stderr.truncate()
                await self.run_model_dialog(prior + [terminals.KeyEvent("CTRL_C")])

    async def test_model_ctrl_c_in_offline_fallback(self):
        await self.run_model_dialog([terminals.KeyEvent("CTRL_C")], offline=True)

    async def test_model_task_cancellation_propagates(self):
        for offline in [False, True]:
            with self.subTest(offline=offline):
                await self.run_model_dialog([terminals.KeyEvent("CTRL_C")],
                                            offline=offline, interrupt=asyncio.CancelledError)

    async def test_thinking_ctrl_c_at_each_stage_preserves_controls(self):
        # Mode choice, manual allowance, and the direct budget control.
        for prior in [[], answer("1"), answer("1") + answer("2"), answer("2")]:
            with self.subTest(stage=prior):
                inputs = KeyboardSession(prior + [terminals.KeyEvent("CTRL_C")])
                async with inputs:
                    text = await self.run_dialog(
                        terminal_frontend.run_thinking_picker_async(inputs))
                    self.assertEqual(text, "Thinking selection cancelled.")
                    self.assertIsNone(self.agent_session.thinking_mode)
                    self.assertIsNone(self.agent_session.thinking_budget)
                    await self.assert_next_input(inputs)

    def account_control(self):
        run = mock.AsyncMock(return_value=provider_controls.ControlResult(lines=["done"]))
        action = provider_controls.ControlAction("reset", "Reset", "Reset account?", run)
        spec = provider_controls.ControlSpec(
            "usage", "Usage", "Account usage", lambda context: True,
            mock.AsyncMock(return_value=provider_controls.ControlResult(lines=["usage"], actions=[action])),
        )
        return spec, run

    async def test_account_ctrl_c_at_each_stage_never_runs_action(self):
        cases = [("/account", []), ("/account", answer("1")),
                 ("/account", answer("1") + answer("1")),
                 ("/account usage reset", [])]
        for command, prior in cases:
            with self.subTest(command=command, stage=prior):
                spec, run = self.account_control()
                inputs = KeyboardSession(prior + [terminals.KeyEvent("CTRL_C")])
                with mock.patch.object(provider_controls, "available_controls", return_value=[spec]), \
                        mock.patch.object(provider_controls, "find_control", return_value=spec):
                    async with inputs:
                        await self.run_dialog(
                            terminal_frontend.run_account_controls_async(command, inputs))
                        run.assert_not_awaited()
                        await self.assert_next_input(inputs)

    async def test_thinking_and_account_task_cancellation_propagates(self):
        spec, run = self.account_control()
        with mock.patch.object(provider_controls, "available_controls", return_value=[spec]):
            for command in ["/thinking", "/account"]:
                inputs = KeyboardSession([terminals.KeyEvent("CTRL_C")], interrupt=asyncio.CancelledError)
                async with inputs:
                    with self.assertRaises(asyncio.CancelledError):
                        if command == "/thinking":
                            await terminal_frontend.run_thinking_picker_async(inputs)
                        else:
                            await terminal_frontend.run_account_controls_async(command, inputs)
                    self.assert_modal_released(inputs)
        run.assert_not_awaited()
