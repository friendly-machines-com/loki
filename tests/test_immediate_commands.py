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
        self.addCleanup(terminal_frontend._immediate_tasks.clear)
        self.addCleanup(terminal_frontend.terminal.assistant_markdown.reset)
        activity_patch = mock.patch.object(
            terminal_frontend, "_terminal_activity",
            terminal_frontend.TerminalActivityStatus())
        activity_patch.start()
        self.addCleanup(activity_patch.stop)
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

    def assert_delayed_account_block(self, *, error=None):
        from loki_agent import provider_controls

        async def scenario():
            read_started = asyncio.Event()
            release_read = asyncio.Event()
            turn_started = asyncio.Event()
            release_turn = asyncio.Event()

            async def gated_read(context):
                read_started.set()
                await release_read.wait()
                if error is not None:
                    raise error
                return ControlResult(lines=["ACCOUNT-RESULT"])

            async def streaming_turn():
                terminal_frontend._terminal_activity.set_turn_running(True)
                terminal_frontend._terminal_agent_event({
                    "type": "assistant_start"})
                turn_started.set()
                await release_turn.wait()
                terminal_frontend._terminal_agent_event({
                    "type": "assistant_delta", "content": "ASSISTANT-PART-B"})
                terminal_frontend._terminal_agent_event({"type": "assistant_end"})
                terminal_frontend._terminal_activity.set_turn_running(False)

            spec = mock.Mock(read=gated_read)
            with mock.patch.object(terminals.os, "isatty", return_value=False), \
                    mock.patch.object(terminal_frontend.terminal,
                                      "markdown_style", False), \
                    mock.patch.object(terminal_frontend, "current_model",
                                      return_value="model"), \
                    mock.patch.object(terminal_frontend,
                                      "restore_output_area_after_input"), \
                    mock.patch.object(provider_controls, "find_control",
                                      return_value=spec), \
                    contextlib.redirect_stdout(self.output), \
                    contextlib.redirect_stderr(self.output):
                session = self.make_session()
                session._producer = asyncio.create_task(session._produce())
                turn = asyncio.create_task(streaming_turn())
                try:
                    async with asyncio.timeout(3):
                        await turn_started.wait()
                        self.feed(session, "/account usage")
                        await read_started.wait()
                        # Move the streaming cursor AFTER the admission echo,
                        # before completing the delayed network read.
                        terminal_frontend._terminal_agent_event({
                            "type": "assistant_delta",
                            "content": "ASSISTANT-PART-A"})
                        release_read.set()
                        if error is None:
                            answer = "ACCOUNT-RESULT"
                        elif isinstance(error, OSError):
                            answer = f"Could not read account usage: {error}"
                        else:
                            answer = f"Command failed: {error}"
                        while answer not in self.output.getvalue():
                            await asyncio.sleep(0)
                        # Assert visibility while the turn is STILL BLOCKED,
                        # without draining tasks or releasing that turn.
                        self.assertFalse(turn.done())
                        self.assertTrue(terminal_frontend._terminal_activity.turn_running)
                        self.assertIn(
                            f"ASSISTANT-PART-A\n/account usage:\n{answer}\n",
                            self.output.getvalue())
                        self.assertTrue(terminal_frontend.terminal.assistant_markdown.active)
                        self.assertEqual(session.user_messages.message_count, 0)
                        self.assertEqual(self.session.transcript_items, self.initial)
                        self.assertFalse(session.reader.cancel_event.is_set())
                finally:
                    release_turn.set()
                    await turn
                    await session._pause()
                    await terminal_frontend._cancel_immediate_tasks(session)
            self.assertIn("\nASSISTANT-PART-B", self.output.getvalue())

        asyncio.run(scenario())

    def test_delayed_account_answer_is_isolated_and_visible_during_turn(self):
        self.assert_delayed_account_block()

    def test_delayed_account_error_is_isolated_and_visible_during_turn(self):
        self.assert_delayed_account_block(error=OSError("account unavailable"))

    def test_unexpected_delayed_error_is_isolated_and_visible_during_turn(self):
        self.assert_delayed_account_block(error=RuntimeError("account failed"))

    def test_monitor_output_preserves_split_markdown_and_escapes_ansi(self):
        with mock.patch.object(terminal_frontend.terminal, "markdown_style", True), \
                mock.patch.object(terminal_frontend, "current_model",
                                  return_value="model"), \
                mock.patch.object(terminal_frontend,
                                  "restore_output_area_after_input"), \
                contextlib.redirect_stdout(self.output):
            terminal_frontend._terminal_agent_event({"type": "assistant_start"})
            terminal_frontend._terminal_agent_event({
                "type": "assistant_delta", "content": "before **pen"})
            terminal_frontend._emit_immediate_output(
                "/account usage", "ACCOUNT\x1b]777;ATTACK\x07")
            terminal_frontend._terminal_agent_event({
                "type": "assistant_delta", "content": "ding** after"})
            terminal_frontend._terminal_agent_event({"type": "assistant_end"})
        rendered = self.output.getvalue()
        self.assertIn("before \n/account usage:\nACCOUNT^[]777;ATTACK^G\n", rendered)
        self.assertNotIn("\x1b]777", rendered)
        self.assertIn(terminals.BOLD + "pending" + terminals.RESET, rendered)

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
            self.assertIn(
                "/account usage:\nCould not read account usage: no key\n",
                self.output.getvalue())

        asyncio.run(scenario())

    def test_extra_account_operands_report_usage_without_reading_or_acting(self):
        from loki_agent import provider_controls
        for command in ["/account usage reset extra",
                        "/account usage reset extra --json"]:
            with self.subTest(command=command), \
                    mock.patch.object(provider_controls, "find_control") as lookup, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                asyncio.run(terminal_frontend.run_account_controls_async(
                    command, self.session))
                self.assertIn("usage: /account [CONTROL [ACTION]] [--json]",
                              output.getvalue())
                lookup.assert_not_called()

    def test_account_read_captures_connection_and_authority_before_scheduling(self):
        from loki_agent import provider_controls
        admitted, later = object(), object()
        authority_a, authority_b = object(), object()
        holder = {"config": admitted}
        seen = {}

        async def read(context):
            seen["config"] = context.config
            seen["authority"] = context.credential_authority
            return ControlResult(lines=["ACCOUNT-A"])

        async def scenario():
            self.session.credential_authority = authority_a
            spec = mock.Mock(read=read)
            with mock.patch.object(terminal_frontend, "current_config",
                                   lambda: holder["config"]), \
                    mock.patch.object(provider_controls, "find_control",
                                      return_value=spec), \
                    mock.patch.object(terminal_frontend,
                                      "restore_output_area_after_input"), \
                    contextlib.redirect_stdout(self.output):
                self.assertTrue(terminal_frontend._submit_immediate("/account usage"))
                # Change both before yielding: the submitted task cannot
                # have started yet.
                holder["config"] = later
                self.session.credential_authority = authority_b
                async with asyncio.timeout(3):
                    while "ACCOUNT-A" not in self.output.getvalue():
                        await asyncio.sleep(0)
            self.assertIs(seen["config"], admitted)
            self.assertIs(seen["authority"], authority_a)

        asyncio.run(scenario())

    def test_status_save_captures_store_before_scheduling(self):
        async def scenario():
            first = types.SimpleNamespace(save=mock.AsyncMock())
            later = types.SimpleNamespace(save=mock.AsyncMock())
            self.session.response_headers = first
            with mock.patch.object(terminal_frontend,
                                   "restore_output_area_after_input"), \
                    contextlib.redirect_stdout(self.output):
                self.assertTrue(terminal_frontend._submit_immediate("/status save"))
                self.session.response_headers = later
                async with asyncio.timeout(3):
                    while "Response status saved" not in self.output.getvalue():
                        await asyncio.sleep(0)
            first.save.assert_awaited_once()
            later.save.assert_not_awaited()

        asyncio.run(scenario())

    def test_pending_reads_are_cancelled_and_joined_before_every_frontend_exit(self):
        from loki_agent import provider_controls
        from loki_agent.credentials import CredentialStore
        from test_loki_tool_loop import ScriptedInputSession

        async def scenario(mode):
            started = asyncio.Event()
            release = asyncio.Event()
            cancellation = asyncio.Event()
            events = []
            attempted_admissions = []

            class GatedInput(ScriptedInputSession):
                async def get(inner):
                    self.assertTrue(inner.on_submit("/account usage"))
                    await started.wait()
                    if mode == "exception":
                        raise RuntimeError("input failed")
                    if mode == "cancel":
                        await asyncio.Event().wait()
                    return None if mode == "eof" else "/quit"

                async def __aexit__(inner, *args):
                    events.append("input closed")

            input_owner = GatedInput([])

            async def read(context):
                started.set()
                try:
                    await release.wait()
                    return ControlResult(lines=["LATE-ANSWER"])
                except asyncio.CancelledError:
                    cancellation.set()
                    # Shutdown must close admission before it awaits reads;
                    # otherwise the still-running producer can add a task.
                    attempted_admissions.append(
                        input_owner.on_submit("/account usage"))
                    raise
                finally:
                    events.append("read ended")

            def open_input(**kwargs):
                input_owner.on_submit = kwargs["on_submit"]
                return input_owner

            spec = mock.Mock(read=read)
            with mock.patch.object(loki, "CREDENTIALS", CredentialStore({})), \
                    mock.patch.object(terminal_frontend, "input_session",
                                      side_effect=open_input), \
                    mock.patch.object(terminal_frontend, "new_chat_log_path",
                                      return_value=os.path.join(self.root, "chat.json")), \
                    mock.patch.object(terminal_frontend,
                                      "restore_output_area_after_input"), \
                    mock.patch.object(provider_controls, "find_control",
                                      return_value=spec), \
                    contextlib.redirect_stdout(self.output), \
                    contextlib.redirect_stderr(io.StringIO()):
                frontend = asyncio.create_task(terminal_frontend.async_main([]))
                async with asyncio.timeout(3):
                    if mode == "cancel":
                        await started.wait()
                        frontend.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await frontend
                    elif mode == "exception":
                        with self.assertRaisesRegex(RuntimeError, "input failed"):
                            await frontend
                    else:
                        self.assertEqual(await frontend, 0)
                self.assertTrue(cancellation.is_set())
                self.assertEqual(events, ["read ended", "input closed"])
                self.assertEqual(attempted_admissions, [False])
                self.assertFalse(terminal_frontend._immediate_tasks)
                output_at_exit = self.output.getvalue()
                release.set()
                await asyncio.sleep(0)
                self.assertEqual(self.output.getvalue(), output_at_exit)
                self.assertNotIn("LATE-ANSWER", output_at_exit)

        for mode in ["quit", "eof", "exception", "cancel"]:
            with self.subTest(mode=mode):
                asyncio.run(scenario(mode))

    def test_cancellation_before_task_start_closes_the_handler_coroutine(self):
        import gc
        import warnings

        async def scenario():
            input_owner = types.SimpleNamespace(
                on_submit=terminal_frontend._submit_immediate)
            with mock.patch.object(terminal_frontend,
                                   "restore_output_area_after_input"), \
                    contextlib.redirect_stdout(self.output):
                self.assertTrue(input_owner.on_submit("/account usage"))
                # No event-loop tick between admission and shutdown.
                await terminal_frontend._cancel_immediate_tasks(input_owner)
            self.assertFalse(terminal_frontend._immediate_tasks)

        with warnings.catch_warnings(record=True) as warnings_seen:
            warnings.simplefilter("always", RuntimeWarning)
            asyncio.run(scenario())
            gc.collect()
        self.assertFalse(any("was never awaited" in str(item.message)
                             for item in warnings_seen))

    def test_output_failure_is_observed_without_affecting_the_turn(self):
        from loki_agent import provider_controls

        async def scenario():
            spec = mock.Mock(read=mock.AsyncMock(
                return_value=ControlResult(lines=["answer"])))
            with mock.patch.object(provider_controls, "find_control",
                                   return_value=spec), \
                    mock.patch.object(terminal_frontend,
                                      "restore_output_area_after_input"), \
                    mock.patch.object(terminal_frontend, "_emit_immediate_output",
                                      side_effect=OSError("display closed")), \
                    contextlib.redirect_stdout(self.output), \
                    self.assertLogs("loki_agent.terminal_frontend", level="ERROR") as logs:
                self.assertTrue(terminal_frontend._submit_immediate("/account usage"))
                async with asyncio.timeout(3):
                    while terminal_frontend._immediate_tasks:
                        await asyncio.sleep(0)
            self.assertIn("display closed", logs.output[0])
            self.assertEqual(self.session.transcript_items, self.initial)

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

    def test_oversized_queue_number_does_not_kill_input(self):
        async def script(session):
            await self.enqueue(session, "first")
            await self.command(session, "/queue texts delete " + "9" * 5000)
            await self.command(session, "/queue texts")
            self.assertEqual(session.user_messages.pending_texts(), ["first"])

        self.run_queue_scenario(script)
        self.assertIn("1. first", self.output.getvalue())

    def test_queue_view_is_released_after_abnormal_frontend_exit(self):
        from loki_agent.credentials import CredentialStore
        from test_loki_tool_loop import ScriptedInputSession

        class FailingInput(ScriptedInputSession):
            async def get(inner):
                self.assertIs(terminal_frontend._queued_inputs.session, inner)
                terminal_frontend._queued_inputs.staged_images.append(object())
                raise RuntimeError("input failed")

        async def scenario():
            input_owner = FailingInput([])
            with mock.patch.object(loki, "CREDENTIALS", CredentialStore({})), \
                    mock.patch.object(terminal_frontend, "input_session",
                                      return_value=input_owner), \
                    mock.patch.object(terminal_frontend, "new_chat_log_path",
                                      return_value=os.path.join(self.root, "chat.json")), \
                    mock.patch.object(terminal_frontend,
                                      "restore_output_area_after_input"), \
                    contextlib.redirect_stdout(self.output), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "input failed"):
                    await terminal_frontend.async_main([])
                self.assertIsNone(terminal_frontend._queued_inputs.session)
                self.assertEqual(terminal_frontend._queued_inputs.staged_images, [])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
