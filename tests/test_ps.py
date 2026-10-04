"""User job commands reuse the job manager without changing job lifetimes."""

import asyncio
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import acp_commands, loki
from loki_agent.sessions import Session


class PsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.manager = loki.JobManager(directory.name)
        self.session = Session(job_manager=self.manager)
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)

    def add_job(self, job_id, status, started_at="2026-10-04T10:00:00Z"):
        spool = os.path.join(self.manager.session_dir, job_id)
        os.makedirs(spool)
        job = loki.Job(
            id=job_id, command="command", argv=["command"], shell=False,
            description="test", background=True, spool_dir=spool,
            stdout_path=os.path.join(spool, "stdout.log"),
            stderr_path=os.path.join(spool, "stderr.log"),
            metadata_path=os.path.join(spool, "job.json"),
            started_at_iso=started_at, status=status,
            process=mock.Mock(returncode=None))
        self.manager.jobs[job_id] = job
        return job

    def test_job_changes_notify_after_the_state_is_updated(self):
        job = self.add_job("1", "running")
        observed = []
        self.manager.on_change = lambda: observed.append(job.status)
        with mock.patch.object(loki.host_process, "signal_group"):
            loki.run_ps("stop 1")
        job.process.returncode = -15
        self.manager._refresh_job(job)
        self.assertEqual(observed, ["stopping", "stopped"])

    def test_display_failure_does_not_replace_job_or_metadata_outcomes(self):
        job = self.add_job("1", "running")
        self.manager.on_change = mock.Mock(side_effect=RuntimeError("display failed"))
        job.process.returncode = 0
        with self.assertLogs("loki_agent.loki", level="ERROR"):
            self.manager._refresh_job(job)
        self.assertEqual(job.status, "exited")
        self.assertEqual(job.exit_code, 0)
        with mock.patch.object(loki, "_atomic_write_text", side_effect=OSError("metadata failed")), \
                self.assertLogs("loki_agent.loki", level="ERROR"), \
                self.assertRaisesRegex(OSError, "metadata failed"):
            self.manager._write_metadata(job)

    def listed_ids(self, text):
        return [line.split(".", 1)[0] for line in text.splitlines()
                if line.partition(". ")[0].isdecimal()]

    def test_list_views_explain_their_empty_result_and_offer_commands(self):
        self.add_job("1", "exited")
        for argument in ["", "all"]:
            with self.subTest(argument=argument):
                text = loki.run_ps(argument)
                for command in [
                        "/ps all", "/ps ID", "/ps stop ID", "/ps kill ID"]:
                    self.assertIn(command, text)
                if not argument:
                    self.assertIn("No running, starting, or failed jobs.", text)
                    self.assertIn(
                        "failed = job setup/launch failure, not command exit code.",
                        text)
                    self.assertNotIn("No jobs.", text)
                else:
                    self.assertNotIn("failed =", text)
        self.manager.jobs.clear()
        self.assertIn("No jobs.", loki.run_ps("all"))
        self.assertEqual(loki.run_jobs(), "No jobs.")

    def test_nonempty_list_keeps_terminal_commands_discoverable(self):
        self.add_job("1", "running")
        for argument in ["", "all"]:
            with self.subTest(argument=argument):
                text = loki.run_ps(argument)
                self.assertIn("/ps ID", text)
                self.assertIn("/ps stop ID", text)
                self.assertIn("/ps kill ID", text)
                if not argument:
                    self.assertIn(
                        "failed = job setup/launch failure, not command exit code.",
                        text)
        self.assertNotIn("/ps", loki.run_jobs())

    def test_default_filters_history_but_all_and_jobs_tool_keep_it(self):
        for job_id, status in [
                ["1", "exited"], ["2", "running"], ["3", "failed"],
                ["4", "starting"], ["5", "stopped"], ["6", "signaled"],
                ["7", "timed_out"], ["8", "cancelled"],
                ["9", "owner_closed"], ["10", "stopping"]]:
            self.add_job(job_id, status)
        self.assertEqual(
            self.listed_ids(loki.run_ps("")), ["2", "3", "4", "6", "7"])
        history = [str(n) for n in range(1, 11)]
        self.assertEqual(self.listed_ids(loki.run_ps("all")), history)
        self.assertEqual(self.listed_ids(loki.run_jobs()), history)

    def test_all_list_views_sort_by_creation_time_then_numeric_id(self):
        self.add_job("10", "running")
        self.add_job("2", "running")
        # Time takes precedence over ID and dict insertion order.
        self.add_job("11", "starting", "2026-10-04T09:59:59Z")
        self.add_job("1", "failed", "2026-10-04T10:01:00Z")
        for argument in ["", "all"]:
            with self.subTest(argument=argument):
                self.assertEqual(
                    self.listed_ids(loki.run_ps(argument)),
                    ["11", "2", "10", "1"])
        self.assertEqual(
            self.listed_ids(loki.run_jobs()), ["11", "2", "10", "1"])

    def test_refresh_happens_before_filtering(self):
        job = self.add_job("1", "running")
        job.process.returncode = 0
        self.assertIn("No running, starting, or failed jobs.", loki.run_ps(""))
        self.assertEqual(job.status, "exited")
        self.assertEqual(self.listed_ids(loki.run_ps("all")), ["1"])

    def test_exited_and_stopped_jobs_are_history_only(self):
        for job_id, status, exit_code in [
                ["1", "running", 0], ["2", "running", 42],
                ["3", "stopping", -9], ["4", "stopping", -15]]:
            job = self.add_job(job_id, status)
            job.process.returncode = exit_code
        self.assertEqual(self.listed_ids(loki.run_ps("")), [])
        self.assertEqual(self.manager.jobs["2"].status, "exited")
        self.assertEqual(self.manager.jobs["2"].exit_code, 42)
        self.assertEqual(self.listed_ids(loki.run_ps("all")), ["1", "2", "3", "4"])

    def test_tail_includes_both_streams_of_a_finished_job(self):
        job = self.add_job("1", "exited")
        with open(job.stdout_path, "w", encoding="utf-8") as stream:
            stream.write("old-prefix" + "x" * loki.JOB_TAIL_CHARS + "answer")
        with open(job.stderr_path, "w", encoding="utf-8") as stream:
            stream.write("diagnostic")
        text = loki.run_ps("1")
        self.assertIn("status: exited", text)
        self.assertIn("answer", text)
        self.assertIn("[stderr_tail]\ndiagnostic", text)
        self.assertNotIn("old-prefix", text)

    def test_stop_and_kill_use_existing_signal_and_state_transitions(self):
        job = self.add_job("1", "running")
        with mock.patch.object(loki.host_process, "signal_group") as send:
            loki.run_ps("stop 1")
            send.assert_called_once_with(job.process, job.pgid, loki.signal.SIGTERM)
            self.assertEqual(job.status, "stopping")
            send.reset_mock()
            loki.run_ps("kill 1")
            send.assert_called_once_with(job.process, job.pgid, loki.host_process.FORCE)
            self.assertEqual(job.status, "stopping")

    def test_unknown_ids_and_finished_jobs_do_not_signal(self):
        self.add_job("1", "exited")
        with mock.patch.object(loki.host_process, "signal_group") as send:
            for argument in ["99", "stop 99", "kill 99"]:
                self.assertIn("unknown job id", loki.run_ps(argument))
            for argument in ["stop 1", "kill 1"]:
                self.assertIn("not running", loki.run_ps(argument))
            send.assert_not_called()

    def test_invalid_arguments_are_local_usage_errors(self):
        with mock.patch.object(self.manager, "stop_job") as stop:
            for argument in [
                    "bogus", "stop", "kill", "stop x y", "all extra",
                    "1 extra", "stop all", "kill bogus"]:
                with self.subTest(argument=argument):
                    self.assertTrue(loki.run_ps(argument).startswith("usage:"))
            stop.assert_not_called()

    def test_terminal_ps_recognition_does_not_intercept_paths_or_other_commands(self):
        for text, expected in [
                [" /ps ", ""], ["/ps all", "all"], ["/ps stop 1", "stop 1"],
                ["/ps kill 1", "kill 1"], ["/ps bad argument", "bad argument"]]:
            with self.subTest(text=text):
                self.assertEqual(loki.ps_argument(text), expected)
        for text in ["/ps/file.py", "/ps-extra", "/PS", "/ps\t1", "/status", "prompt"]:
            with self.subTest(text=text):
                self.assertIsNone(loki.ps_argument(text))

    def test_acp_does_not_advertise_or_handle_ps(self):
        with mock.patch.object(loki, "LOKI_CONFIG_DIR", self.manager.base_dir):
            self.assertNotIn("ps", [command["name"]
                                    for command in acp_commands.advertised_commands()])
        for text in ["/ps", "/ps 1", "/ps all", "/ps stop 1", "/ps kill 1", "/ps stop"]:
            with self.subTest(text=text), mock.patch.object(loki, "run_ps") as run_ps:
                self.assertIsNone(acp_commands.parse(text))
                self.assertIsNone(asyncio.run(acp_commands.run(text, self.session)))
                run_ps.assert_not_called()
        self.assertEqual(self.session.transcript_items, [])


class ImmediateTerminalPsTests(unittest.TestCase):
    def test_input_owner_lists_tails_stops_and_kills_without_dequeueing(self):
        from loki_agent import formats, terminal_frontend, terminals
        from process_lifecycle_fixtures import ProcessResources

        async def scenario(root):
            manager = loki.JobManager(os.path.join(root, "jobs"))
            conversation = Session(job_manager=manager, shell_cwd=root,
                                   transcript_items=[formats.message_item("user", "busy")])
            initial = list(conversation.transcript_items)
            output = io.StringIO()
            counts = []
            handled = []

            class Reader:
                def __init__(self):
                    self.keys = asyncio.Queue()
                    self.cancel_requested = False
                    self.cancel_event = asyncio.Event()

                async def read_key(self):
                    return await self.keys.get()

            def submit(text):
                consumed = terminal_frontend._submit_job_control(text)
                if consumed:
                    handled.append(text)
                return consumed

            with mock.patch.object(loki, "_DEFAULT_SESSION", conversation), \
                    mock.patch.object(terminals.os, "isatty", return_value=False), \
                    contextlib.redirect_stdout(output):
                session = terminals.InputSession(
                    fd=0, on_submit=submit, on_queue_size_change=counts.append)
                session.reader = Reader()
                session._producer = asyncio.create_task(session._produce())
                script = (
                    "import signal,time\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "if hasattr(signal, 'SIGBREAK'):\n"
                    "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
                    "print('job-ready\\x1b]777;PS_TAIL_ATTACK\\x07', flush=True)\n"
                    "time.sleep(60)\n"
                )
                foreground = asyncio.create_task(manager.run_foreground(
                    [sys.executable, "-c", script], "controlled job", 60_000, cwd=root))
                resources = None

                def feed(text):
                    session.reader.keys.put_nowait(terminals.KeyEvent("TEXT", text))
                    session.reader.keys.put_nowait(terminals.KeyEvent("ENTER"))

                async def command(text):
                    previous = len(handled)
                    feed(text)
                    async with asyncio.timeout(3):
                        while len(handled) == previous:
                            await asyncio.sleep(0)
                    self.assertEqual(handled[-1], text)
                    self.assertEqual(session.user_messages.message_count, 1)
                    self.assertEqual(counts, [1])
                    self.assertEqual(conversation.transcript_items, initial)

                try:
                    async with asyncio.timeout(5):
                        while True:
                            job = next(iter(manager.jobs.values()), None)
                            if job is not None and job.process is not None:
                                if "job-ready" in loki._read_spool_tail(job.stdout_path):
                                    break
                            await asyncio.sleep(.01)
                    resources = ProcessResources(job.process)
                    feed("ordinary prompt")
                    async with asyncio.timeout(3):
                        while session.user_messages.message_count != 1:
                            await asyncio.sleep(0)
                    await command("/ps")
                    await command("/ps all")
                    self.assertIn(f"{job.id}. status=running", output.getvalue())
                    await command("/ps " + job.id)
                    self.assertIn("[stdout_tail]\njob-ready^[]777;PS_TAIL_ATTACK^G", output.getvalue())
                    self.assertNotIn("\x1b]777", output.getvalue())
                    await command("/ps stop " + job.id)
                    self.assertEqual(job.status, "stopping")
                    self.assertFalse(session.reader.cancel_requested)
                    self.assertFalse(session.reader.cancel_event.is_set())
                    await command("/ps kill " + job.id)
                    result_job, outcome, _stdout, _stderr = await asyncio.wait_for(foreground, 5)
                    self.assertIs(result_job, job)
                    self.assertEqual(outcome, "completed")
                    self.assertEqual(job.status, "stopped")
                    self.assertIsNotNone(job.exit_code)
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    resources.assert_released(self)

                    # /ps remains immediate during exclusive modal input, and
                    # must neither answer the modal nor consume pending input.
                    session.reader.cancel_requested = True
                    session.reader.cancel_event.set()
                    async with session.modal() as modal:
                        answer = asyncio.create_task(modal.prompt("Confirm: "))
                        try:
                            await command("/ps all")
                            self.assertFalse(answer.done())
                            self.assertTrue(session.reader.cancel_requested)
                            self.assertTrue(session.reader.cancel_event.is_set())
                            feed("no")
                            self.assertEqual(await asyncio.wait_for(answer, 3), "no")
                        finally:
                            if not answer.done():
                                answer.cancel()
                            await asyncio.gather(answer, return_exceptions=True)
                    with mock.patch.object(loki, "run_ps", side_effect=OSError("inspect failed\x1b[2J")):
                        await command("/ps")
                    self.assertIn("Could not inspect or control jobs: inspect failed^[[2J", output.getvalue())
                    await command("/ps all")
                    self.assertEqual(session.user_messages.get_nowait(), "ordinary prompt")
                    self.assertEqual(counts, [1, 0])
                finally:
                    await session._pause()
                    for job in manager.jobs.values():
                        if job.process is not None and job.process.returncode is None:
                            loki.host_process.signal_group(job.process, job.pgid, loki.host_process.FORCE)
                            await asyncio.wait_for(job.process.wait(), 3)
                    if not foreground.done():
                        foreground.cancel()
                    await asyncio.wait_for(asyncio.gather(foreground, return_exceptions=True), 3)
                    if resources is not None:
                        await resources.cleanup()

        with tempfile.TemporaryDirectory() as root:
            asyncio.run(scenario(root))


if __name__ == "__main__":
    unittest.main()
