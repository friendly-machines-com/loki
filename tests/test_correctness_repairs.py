import asyncio
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

from loki_agent import formats, http_client, loki, protocols, terminal_frontend


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PackageContractTests(unittest.TestCase):
    def test_console_script_targets_the_terminal_frontend(self):
        with open(ROOT / "pyproject.toml", "rb") as stream:
            project = tomllib.load(stream)
        self.assertEqual(
            project["project"]["scripts"]["loki"],
            "loki_agent.terminal_frontend:main",
        )
        self.assertTrue(callable(loki.main))

    def test_historical_module_invocation_still_reaches_frontend(self):
        env = dict(os.environ)
        env.update({
            "PYTHONPATH": str(ROOT),
            "LOKI_PROVIDER": "dummy",
            "LOKI_API_BASE": "http://dummy.invalid/v1",
            "LOKI_MODEL": "dummy-model",
            "LOKI_DUMMY_REPLY": "module-entry-ok",
        })
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "loki_agent.loki",
                "--headless",
                "--prompt",
                "probe",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "module-entry-ok")


class ProviderResponseContractTests(unittest.TestCase):
    def test_successful_http_response_with_invalid_json_is_protocol_error(self):
        response = http_client.HttpResponse(
            "https://example.test/v1/chat/completions",
            200,
            "OK",
            {"content-type": "text/plain"},
            b"<html>not json</html>",
        )

        with mock.patch.object(
                http_client, "async_http_request",
                new=mock.AsyncMock(return_value=response)):
            with self.assertRaises(protocols.ProtocolError):
                asyncio.run(loki.async_chat_request(
                    response.url, {"model": "x"}, request_headers={}))

    def test_chat_posts_and_model_gets_use_separate_timeouts(self):
        response = http_client.HttpResponse(
            "https://example.test/v1/chat/completions",
            200,
            "OK",
            {"content-type": "application/json"},
            b"{}",
        )

        for payload, expected_timeout in [
                ({"model": "x"}, loki.LLM_REQUEST_TIMEOUT_S),
                (None, loki.WEBFETCH_TIMEOUT_S)]:
            transport = mock.AsyncMock(return_value=response)
            with self.subTest(payload=payload), mock.patch.object(
                    http_client, "async_http_request", new=transport):
                asyncio.run(loki.async_chat_request(
                    response.url, payload, request_headers={}))

            self.assertEqual(
                transport.await_args.kwargs["timeout"],
                expected_timeout,
            )

    def test_tool_loop_reports_provider_protocol_error_without_appending(self):
        events = []
        transcript = [
            loki.formats.message_item("user", "hello"),
        ]

        async def broken_chat(_items):
            raise protocols.ProtocolError("malformed provider JSON")

        answer = asyncio.run(loki.run_tool_loop_async(
            transcript, chat_fn=broken_chat, on_event=events.append))

        self.assertEqual(answer, "")
        self.assertEqual(len(transcript), 1)
        self.assertIn("provider_error", [event["type"] for event in events])


class FileObservationContractTests(unittest.TestCase):
    def setUp(self):
        loki.file_state.clear()

    def test_binary_change_after_read_blocks_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "payload.bin")
            pathlib.Path(path).write_bytes(b"\x00old")
            self.assertIn("binary", loki.run_read(path))

            pathlib.Path(path).write_bytes(b"\x00new")
            result = loki.run_write(path, "replacement")

            self.assertIn("changed on disk", result)
            self.assertEqual(pathlib.Path(path).read_bytes(), b"\x00new")

    def test_deletion_after_read_blocks_recreation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "notes.txt")
            pathlib.Path(path).write_text("old", encoding="utf-8")
            loki.run_read(path)
            os.unlink(path)

            result = loki.run_write(path, "replacement")

            self.assertIn("checking current file contents", result)
            self.assertFalse(os.path.exists(path))

    def test_empty_string_is_a_valid_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.txt")
            result = loki.run_write(path, "")
            self.assertIn("Successfully wrote", result)
            self.assertEqual(pathlib.Path(path).read_text(encoding="utf-8"), "")


class JobOwnershipContractTests(unittest.TestCase):
    def test_job_state_uses_event_loop_ownership_without_a_thread_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))

        self.assertFalse(hasattr(manager, "_lock"))
        self.assertEqual(manager._next_job_id(), "1")
        self.assertEqual(manager._next_job_id(), "2")

    def test_force_stop_escalates_a_job_already_stopping(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            script = (
                "import signal,time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "print('ready', flush=True)\n"
                "time.sleep(30)\n"
            )
            job = await manager.run_background_exec(
                [sys.executable, "-c", script], cwd=tmpdir)
            try:
                deadline = asyncio.get_running_loop().time() + 3
                while "ready" not in loki._read_spool_tail(job.stdout_path):
                    if asyncio.get_running_loop().time() >= deadline:
                        self.fail("background process did not become ready")
                    await asyncio.sleep(0.01)
                first = manager.stop_job(job.id)
                await asyncio.sleep(0.05)
                self.assertIsNone(job.process.returncode)
                second = manager.stop_job(job.id, force=True)
                await asyncio.wait_for(job.process.wait(), timeout=3)
                deadline = asyncio.get_running_loop().time() + 3
                while job.status == "stopping":
                    if asyncio.get_running_loop().time() >= deadline:
                        self.fail("background reaper did not finalize job")
                    await asyncio.sleep(0.01)
                with open(
                        job.metadata_path, encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)
                return first, second, job, metadata
            finally:
                if job.process.returncode is None:
                    os.killpg(job.pgid, signal.SIGKILL)
                    await job.process.wait()

        with tempfile.TemporaryDirectory() as tmpdir:
            first, second, job, metadata = asyncio.run(scenario(tmpdir))
        self.assertIn("SIGTERM", first)
        self.assertIn("SIGKILL", second)
        self.assertEqual(job.status, "stopped")
        self.assertEqual(job.exit_code, -signal.SIGKILL)
        self.assertEqual(job.signal, signal.SIGKILL)
        self.assertEqual(metadata["status"], "stopped")
        self.assertEqual(metadata["exit_code"], -signal.SIGKILL)
        self.assertEqual(metadata["signal"], signal.SIGKILL)

    def test_cancelling_owner_task_reaps_foreground_process(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            task = asyncio.create_task(manager.run_exec(
                ["sleep", "30"], 60_000, cwd=tmpdir))
            while not manager.jobs:
                await asyncio.sleep(0)
            job = next(iter(manager.jobs.values()))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with open(job.metadata_path, encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            return job, metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            job, metadata = asyncio.run(scenario(tmpdir))
        self.assertEqual(job.status, "cancelled")
        self.assertIsNotNone(job.process.returncode)
        self.assertEqual(metadata["status"], "cancelled")
        self.assertEqual(metadata["exit_code"], job.process.returncode)

    def test_bash_timeout_is_a_failed_tool_result(self):
        async def scenario(tmpdir):
            old_manager = loki.current_session().job_manager
            loki.current_session().job_manager = loki.JobManager(
                os.path.join(tmpdir, "jobs"))
            try:
                return await loki.dispatch_tool_async(
                    "Bash",
                    {
                        "command": "sleep 1",
                        "timeout": 10,
                        "description": "timeout contract",
                    },
                )
            finally:
                loki.current_session().job_manager = old_manager

        with tempfile.TemporaryDirectory() as tmpdir:
            result = asyncio.run(scenario(tmpdir))
        self.assertFalse(result["ok"])
        self.assertIn("timed_out", result["content"])


class AgentModeContractTests(unittest.TestCase):
    def test_explore_and_plan_expose_only_nonmutating_tools(self):
        self.assertNotIn("Bash", loki.EXPLORE_TOOLS)
        self.assertNotIn("Write", loki.EXPLORE_TOOLS)
        self.assertNotIn("Edit", loki.EXPLORE_TOOLS)
        self.assertNotIn("TodoWrite", loki.EXPLORE_TOOLS)
        self.assertNotIn("JobStop", loki.EXPLORE_TOOLS)
        self.assertNotIn("Agent", loki.EXPLORE_TOOLS)
        self.assertIn("Read", loki.EXPLORE_TOOLS)
        self.assertIn("Grep", loki.EXPLORE_TOOLS)

        old_mode = loki.current_session().agent_mode
        try:
            for mode in ("explore", "plan"):
                loki.current_session().agent_mode = mode
                context = loki.get_tool_loop_extra_context([])
                self.assertIsNotNone(
                    loki._tool_access_error(
                        "Write", extra_context=context))
                self.assertIsNone(
                    loki._tool_access_error(
                        "Read", extra_context=context))
        finally:
            loki.current_session().agent_mode = old_mode

    def test_question_punctuation_does_not_revoke_implementation(self):
        old_mode = loki.current_session().agent_mode
        try:
            loki.current_session().agent_mode = "normal"
            transcript = [
                loki.formats.message_item(
                    "user", "Can you fix the broken resume?"),
            ]
            self.assertFalse(
                loki.get_tool_loop_extra_context(transcript)[
                    "inhibit_edits"])
        finally:
            loki.current_session().agent_mode = old_mode

    def test_terminal_advertises_only_read_only_tools_in_plan_mode(self):
        captured = []
        old_mode = loki.current_session().agent_mode
        old_toolsets = list(loki.current_session().session_toolsets)

        async def fake_completion(items, tools, *args, **kwargs):
            captured.extend(
                tool["function"]["name"] for tool in tools)
            return formats.DecodedTurn([
                formats.message_item("assistant", "plan"),
            ])

        try:
            loki.current_session().agent_mode = "plan"
            with (
                    mock.patch.object(
                        terminal_frontend, "async_chat_completion",
                        new=fake_completion),
                    mock.patch.object(
                        terminal_frontend, "_terminal_agent_event")):
                asyncio.run(terminal_frontend.run_terminal_turn_async([
                    formats.message_item("user", "plan this"),
                ]))
        finally:
            loki.current_session().agent_mode = old_mode
            loki.current_session().session_toolsets = old_toolsets

        self.assertEqual(set(captured), loki.EXPLORE_TOOLS)


if __name__ == "__main__":
    unittest.main()
