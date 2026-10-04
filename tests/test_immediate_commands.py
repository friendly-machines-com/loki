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
import types
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


class ImmediateQueueTests(unittest.TestCase):
    """/queue snapshots the FIFO and staged images without consuming."""

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
        self.output = io.StringIO()
        self.handled = []
        self.addCleanup(terminal_frontend._immediate_tasks.clear)
        self.addCleanup(terminal_frontend._queued_inputs.reset, None)

    async def enqueue(self, session, text):
        session.reader.keys.put_nowait(terminals.KeyEvent("TEXT", text))
        session.reader.keys.put_nowait(terminals.KeyEvent("ENTER"))
        async with asyncio.timeout(3):
            while text not in session.user_messages.pending_texts():
                await asyncio.sleep(0)

    async def command(self, session, text):
        previous = len(self.handled)
        session.reader.keys.put_nowait(terminals.KeyEvent("TEXT", text))
        session.reader.keys.put_nowait(terminals.KeyEvent("ENTER"))
        async with asyncio.timeout(3):
            while len(self.handled) == previous:
                await asyncio.sleep(0)
        self.assertEqual(self.handled[-1], text)
        self.assertEqual(self.session.transcript_items, self.initial)

    def run_queue_scenario(self, script):
        async def scenario():
            def submit(text):
                consumed = terminal_frontend._submit_immediate(text)
                if consumed:
                    self.handled.append(text)
                return consumed

            with mock.patch.object(terminals.os, "isatty",
                                   return_value=False), \
                    mock.patch.object(
                        terminal_frontend,
                        "restore_output_area_after_input"), \
                    contextlib.redirect_stdout(self.output):
                session = terminals.InputSession(
                    fd=0, on_submit=submit,
                    on_queue_size_change=lambda count: None)
                session.reader = _FakeReader()
                terminal_frontend._queued_inputs.reset(session)
                session._producer = asyncio.create_task(session._produce())
                try:
                    await script(session)
                finally:
                    await session._pause()

        asyncio.run(scenario())

    def test_bare_queue_lists_counts_and_subcommands(self):
        async def script(session):
            await self.enqueue(session, "first")
            await self.enqueue(session, "second")
            terminal_frontend._queued_inputs.staged_images.append(
                types.SimpleNamespace(
                    path="/tmp/x.png", media_type="image/png",
                    byte_size=12))
            await self.command(session, "/queue")
            # Listing never consumes: both prompts are still queued.
            self.assertEqual(session.user_messages.message_count, 2)
            self.assertEqual(
                session.user_messages.pending_texts(),
                ["first", "second"])

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertIn("Queued texts: 2; staged images: 1.", rendered)
        self.assertIn("Subcommands: /queue texts, /queue images.", rendered)

    def test_queue_texts_numbers_by_send_order_and_escapes_ansi(self):
        async def script(session):
            await self.enqueue(session, "first")
            await self.enqueue(session, "evil\x1b]777;QUEUE_ATTACK\x07tail")
            await self.command(session, "/queue texts")
            self.assertEqual(session.user_messages.message_count, 2)

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertIn("Queued texts (1 = next sent):", rendered)
        self.assertIn("1. first", rendered)
        self.assertIn("2. evil^[]777;QUEUE_ATTACK^Gtail", rendered)
        self.assertNotIn("\x1b]777", rendered)

    def test_queue_images_lists_staged_entries_and_empty_state(self):
        async def script(session):
            await self.command(session, "/queue images")
            terminal_frontend._queued_inputs.staged_images.append(
                types.SimpleNamespace(
                    path="/home/dannym/x.webp", media_type="image/webp",
                    byte_size=4096))
            await self.command(session, "/queue images")

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertIn("No staged images.", rendered)
        self.assertIn("Staged images (sent with the next prompt):", rendered)
        self.assertIn(
            "1. /home/dannym/x.webp (image/webp, 4096 bytes)", rendered)

    def test_queue_texts_empty_state_and_unknown_subcommand(self):
        async def script(session):
            await self.command(session, "/queue texts")
            await self.command(session, "/queue bogus")

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertIn("No queued texts.", rendered)
        self.assertIn(
            "usage: /queue [texts | images] "
            "[delete N | move N M | edit N TEXT]", rendered)
        self.assertEqual(len(self.handled), 2)

    def test_queue_texts_delete_removes_and_relists(self):
        async def script(session):
            for text in ["first", "second", "third"]:
                await self.enqueue(session, text)
            await self.command(session, "/queue texts delete 2")
            self.assertEqual(
                session.user_messages.pending_texts(), ["first", "third"])
            self.assertEqual(session.user_messages.message_count, 2)

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertNotIn("2. second\n", rendered)
        self.assertIn("1. first", rendered)
        self.assertIn("2. third", rendered)
        self.assertNotIn("3. third", rendered)

    def test_queue_texts_move_reorders_and_relists(self):
        async def script(session):
            for text in ["first", "second", "third"]:
                await self.enqueue(session, text)
            await self.command(session, "/queue texts move 1 3")
            self.assertEqual(
                session.user_messages.pending_texts(),
                ["second", "third", "first"])

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertIn("1. second", rendered)
        self.assertIn("3. first", rendered)

    def test_queue_texts_edit_replaces_text_only(self):
        async def script(session):
            for text in ["first", "second"]:
                await self.enqueue(session, text)
            await self.command(
                session, "/queue texts edit 2 replacement text")
            self.assertEqual(
                session.user_messages.pending_texts(),
                ["first", "replacement text"])
            self.assertEqual(session.user_messages.message_count, 2)

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertIn("2. replacement text", rendered)
        self.assertNotIn("2. second\n", rendered)
        self.assertIn("1. first", rendered)

    def test_queue_texts_edit_escapes_replacement_ansi(self):
        async def script(session):
            await self.enqueue(session, "first")
            await self.command(session, "/queue texts edit 1 x\x1b]777;EDIT^G")
            self.assertEqual(
                session.user_messages.pending_texts(),
                ["x\x1b]777;EDIT^G"])

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertIn("x^[]777;EDIT^G", rendered)
        self.assertNotIn("\x1b]777", rendered)

    def test_queue_texts_positions_name_real_entries(self):
        async def script(session):
            await self.enqueue(session, "first")
            await self.command(session, "/queue texts delete 9")
            await self.command(session, "/queue texts move 1 nope")
            await self.command(session, "/queue texts edit 1")
            self.assertEqual(
                session.user_messages.pending_texts(), ["first"])

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertIn("No queued text 9.", rendered)
        self.assertIn(
            "usage: /queue [texts | images] "
            "[delete N | move N M | edit N TEXT]", rendered)
        # Nothing changed, so no listing was re-printed either.
        self.assertNotIn("Queued texts (1 = next sent):", rendered)

    def test_queue_images_delete_and_move_keep_the_status_count(self):
        async def script(session):
            staged = terminal_frontend._queued_inputs.staged_images
            for name in ("a.png", "b.png", "c.png"):
                staged.append(types.SimpleNamespace(
                    path=f"/tmp/{name}", media_type="image/png",
                    byte_size=len(name)))
            await self.command(session, "/queue images delete 1")
            self.assertEqual([image.path for image in staged],
                             ["/tmp/b.png", "/tmp/c.png"])
            self.assertEqual(
                terminal_frontend._terminal_activity.queued_images, 2)
            await self.command(session, "/queue images move 2 1")
            self.assertEqual([image.path for image in staged],
                             ["/tmp/c.png", "/tmp/b.png"])

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertIn("1. /tmp/c.png (image/png, 5 bytes)", rendered)
        self.assertIn("2. /tmp/b.png (image/png, 5 bytes)", rendered)

    def test_queue_images_edit_is_refused_with_guidance(self):
        async def script(session):
            await self.command(session, "/queue images edit 1 x.png")

        self.run_queue_scenario(script)
        rendered = self.output.getvalue()
        self.assertIn(
            "A staged image cannot be edited; delete it and stage another "
            "with /image PATH.", rendered)

    def test_queue_positions_out_of_range_for_images(self):
        async def script(session):
            await self.command(session, "/queue images delete 3")

        self.run_queue_scenario(script)
        self.assertIn("No staged image 3.", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
