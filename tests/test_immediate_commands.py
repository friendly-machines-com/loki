"""Immediate (non-queueing) commands answer without dequeuing or turns.

/status and the read-only /account form share /ps's monitor-plane delivery:
consumed by the input owner before the FIFO, rendered to the display only,
never appended to the transcript, and never starting a turn -- even while a
turn is busy.
"""

import asyncio
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from loki_agent import formats, loki, terminal_frontend, terminals
from loki_agent.provider_controls import ControlAction, ControlResult
from loki_agent.response_headers import Store
from loki_agent.sessions import Session


class _FakeReader:
    def __init__(self):
        self.keys = asyncio.Queue()
        self.cancel_requested = False
        self.cancel_event = asyncio.Event()

    async def read_key(self):
        return await self.keys.get()


class ImmediateCommandTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = directory.name
        self.session = Session(
            shell_cwd=self.root,
            transcript_items=[formats.message_item("user", "busy")])
        self.initial = list(self.session.transcript_items)
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)
        # Module-global delivery state must not leak between tests when one
        # fails mid-scenario.
        self.addCleanup(terminal_frontend._immediate_tasks.clear)
        self.addCleanup(terminal_frontend._queued_inputs.reset, None)
        self.output = io.StringIO()
        self.handled = []

    def test_every_immediate_command_has_exactly_one_handler(self):
        # The delivery table and the handler registry live in different
        # modules: a declaration without a handler would raise KeyError
        # inside the input owner and silently kill the producer. Pin the
        # two tables together, in both directions.
        from loki_agent import command_deliveries
        declared = sorted(
            name for name, spec in command_deliveries._COMMANDS.items()
            if spec.delivery.terminal == command_deliveries.IMMEDIATE)
        self.assertEqual(
            sorted(terminal_frontend._IMMEDIATE_HANDLERS), declared)

    def make_session(self):
        def submit(text):
            consumed = terminal_frontend._submit_immediate(text)
            if consumed:
                self.handled.append(text)
            return consumed

        session = terminals.InputSession(
            fd=0, on_submit=submit, on_queue_size_change=lambda count: None)
        session.reader = _FakeReader()
        return session

    def feed(self, session, text):
        session.reader.keys.put_nowait(terminals.KeyEvent("TEXT", text))
        session.reader.keys.put_nowait(terminals.KeyEvent("ENTER"))

    async def command(self, session, text):
        previous = len(self.handled)
        self.feed(session, text)
        async with asyncio.timeout(3):
            while len(self.handled) == previous:
                await asyncio.sleep(0)
            # Async handlers are scheduled; let them finish before asserting.
            pending = list(terminal_frontend._immediate_tasks)
            if pending:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), 3)
        self.assertEqual(self.handled[-1], text)
        self.assertEqual(session.user_messages.message_count, 0)
        self.assertEqual(self.session.transcript_items, self.initial)

    def test_status_answers_without_queueing_or_touching_the_turn(self):
        async def scenario():
            store = Store(os.path.join(self.root, "status.json"))
            store.observer("https://example.test/chat", None, "model")(
                200, {"x-remaining": "8"})
            self.session.response_headers = store
            with mock.patch.object(terminals.os, "isatty",
                                   return_value=False), \
                    mock.patch.object(
                        terminal_frontend,
                        "restore_output_area_after_input"), \
                    mock.patch.object(
                        loki, "async_chat_completion",
                        side_effect=AssertionError("turn started")), \
                    contextlib.redirect_stdout(self.output):
                session = self.make_session()
                session._producer = asyncio.create_task(session._produce())
                try:
                    await self.command(session, "/status")
                    await self.command(session, "/status all")
                    await self.command(session, "/status --json")
                finally:
                    await session._pause()
            rendered = self.output.getvalue()
            self.assertIn("User: /status", rendered)
            self.assertIn("x-remaining", rendered)
            self.assertIn("All known connections", rendered)
            self.assertIn('"endpoints": []', rendered)

        asyncio.run(scenario())

    def test_status_save_runs_as_a_scheduled_task(self):
        async def scenario():
            store = Store(os.path.join(self.root, "status.json"))
            store.observer("https://example.test/chat", None, "model")(
                200, {"x-remaining": "8"})
            self.session.response_headers = store
            with mock.patch.object(terminals.os, "isatty",
                                   return_value=False), \
                    mock.patch.object(
                        terminal_frontend,
                        "restore_output_area_after_input"), \
                    contextlib.redirect_stdout(self.output):
                session = self.make_session()
                session._producer = asyncio.create_task(session._produce())
                try:
                    await self.command(session, "/status save")
                finally:
                    await session._pause()
            self.assertIn("Response status saved", self.output.getvalue())
            saved = Store(store.path).snapshot()["endpoints"]
            self.assertEqual(saved[0]["headers"]["x-remaining"]["value"], "8")

        asyncio.run(scenario())

    def test_account_read_is_immediate_modal_free_and_hints_actions(self):
        async def scenario():
            action = ControlAction(
                id="reset", title="Reset", confirm="Reset?", run=None)
            spec = mock.Mock()
            spec.read = mock.AsyncMock(return_value=ControlResult(
                lines=("usage: fine",), actions=(action,)))

            from loki_agent import provider_controls
            with mock.patch.object(terminals.os, "isatty",
                                   return_value=False), \
                    mock.patch.object(
                        terminal_frontend,
                        "restore_output_area_after_input"), \
                    mock.patch.object(
                        provider_controls, "find_control",
                        return_value=spec), \
                    contextlib.redirect_stdout(self.output):
                session = self.make_session()
                session._producer = asyncio.create_task(session._produce())
                try:
                    await self.command(session, "/account usage")
                finally:
                    await session._pause()
            spec.read.assert_awaited_once()
            rendered = self.output.getvalue()
            self.assertIn("usage: fine", rendered)
            self.assertIn(
                "Actions (run /account usage ACTION to perform one;", rendered)
            self.assertIn("  reset - Reset", rendered)
            # The monitor plane never takes the reader: no modal prompt line.
            self.assertNotIn("[y/N]", rendered)

        asyncio.run(scenario())

    def test_account_read_failure_is_reported_not_fatal(self):
        async def scenario():
            from loki_agent import provider_controls, authentications
            spec = mock.Mock()
            spec.read = mock.AsyncMock(
                side_effect=authentications.CredentialError("no key"))
            errors = io.StringIO()
            with mock.patch.object(terminals.os, "isatty",
                                   return_value=False), \
                    mock.patch.object(
                        terminal_frontend,
                        "restore_output_area_after_input"), \
                    mock.patch.object(
                        provider_controls, "find_control",
                        return_value=spec), \
                    contextlib.redirect_stdout(self.output), \
                    contextlib.redirect_stderr(errors):
                session = self.make_session()
                session._producer = asyncio.create_task(session._produce())
                try:
                    await self.command(session, "/account usage")
                finally:
                    await session._pause()
            self.assertIn("Could not read account usage", errors.getvalue())

        asyncio.run(scenario())

    def test_extra_account_operands_never_crash_the_dispatch(self):
        from loki_agent import provider_controls
        with mock.patch.object(
                provider_controls, "find_control", return_value=None), \
                mock.patch.object(
                    provider_controls, "available_controls",
                    return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            # Three operands must be ignored past the action id, not raise
            # an unpack ValueError out of the queued dispatch.
            asyncio.run(terminal_frontend.run_account_controls_async(
                "/account usage reset extra", self.session))

    def test_account_read_captures_the_connection_at_admission(self):
        from loki_agent import command_deliveries
        from loki_agent.provider_controls import ControlResult
        admitted, later = object(), object()
        holder = {"config": admitted}
        seen = {}

        async def fake_read(context, chosen, as_json):
            seen["config"] = context.config
            return ControlResult(lines=("ok",))

        async def scenario():
            with mock.patch.object(terminal_frontend, "current_config",
                                   lambda: holder["config"]), \
                    mock.patch.object(
                        terminal_frontend, "_read_account_control",
                        fake_read), \
                    mock.patch.object(terminals.os, "isatty",
                                      return_value=False), \
                    mock.patch.object(
                        terminal_frontend,
                        "restore_output_area_after_input"), \
                    contextlib.redirect_stdout(io.StringIO()):
                parsed = command_deliveries.terminal_immediate("/account usage")
                outcome = terminal_frontend._dispatch_immediate(parsed)
                # Admission has happened; the connection moves on before
                # the scheduled task first runs.
                holder["config"] = later
                await outcome

        asyncio.run(scenario())
        self.assertIs(seen["config"], admitted)

    def test_pending_immediate_read_does_not_survive_frontend_exit(self):
        from test_loki_tool_loop import ScriptedInputSession

        async def slow_read(context, chosen, as_json):
            await asyncio.sleep(30)
            self.fail("the read outlived the frontend")

        async def scenario():
            from loki_agent.credentials import CredentialStore
            loki.CREDENTIALS = CredentialStore({})
            session = ScriptedInputSession(["/account usage", "/quit"])
            session.on_submit = terminal_frontend._submit_immediate
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch.object(
                        terminal_frontend, "input_session",
                        return_value=session), \
                        mock.patch.object(
                            terminal_frontend, "new_chat_log_path",
                            return_value=os.path.join(tmpdir, "chat.json")), \
                        mock.patch.object(
                            terminal_frontend,
                            "restore_output_area_after_input"), \
                        mock.patch.object(
                            terminal_frontend, "_read_account_control",
                            slow_read), \
                        mock.patch.object(
                            terminal_frontend, "run_terminal_turn_async",
                            mock.AsyncMock()), \
                        contextlib.redirect_stdout(io.StringIO()):
                    status = await terminal_frontend.async_main([])
            self.assertEqual(status, 0)
            self.assertFalse(terminal_frontend._immediate_tasks)
            leftover = [
                task for task in asyncio.all_tasks()
                if task is not asyncio.current_task() and not task.done()]
            self.assertEqual(leftover, [])

        asyncio.run(scenario())

    def test_interactive_account_forms_still_queue(self):
        # The bare listing and CONTROL ACTION stay queued: they interact and
        # must not take the reader from a running turn.
        for text in ["/account", "/account usage reset"]:
            with self.subTest(text=text):
                self.assertFalse(terminal_frontend._submit_immediate(text))


if __name__ == "__main__":
    unittest.main()
