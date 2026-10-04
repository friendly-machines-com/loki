"""User job commands reuse the job manager without changing job lifetimes."""

import asyncio
import os
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

    def test_nonempty_list_keeps_commands_discoverable_in_both_frontends(self):
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
                outcome = asyncio.run(acp_commands.run(
                    "/ps " + argument, self.session))
                self.assertEqual(outcome.text, text)
                self.assertIsNone(outcome.model_text)
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
        outcome = asyncio.run(acp_commands.run("/ps", self.session))
        self.assertEqual(self.listed_ids(outcome.text), [])
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

    def test_acp_routes_all_variants_as_local_commands(self):
        for argument in ["", "1", "all", "stop 1", "kill 1", "stop"]:
            with self.subTest(argument=argument), mock.patch.object(
                    loki, "run_ps", return_value="result") as run_ps:
                outcome = asyncio.run(acp_commands.run(
                    "/ps " + argument, self.session))
                run_ps.assert_called_once_with(argument)
                self.assertEqual(outcome.text, "result")
                self.assertIsNone(outcome.model_text)
        self.assertEqual(self.session.transcript_items, [])


if __name__ == "__main__":
    unittest.main()
