import asyncio
import json
import os
import pathlib
import signal
import sys
import tempfile
import types
import unittest
from unittest import mock

from loki_agent import (
    authentications,
    formats,
    http_client,
    loki,
    protocols,
    terminal_frontend,
)
from loki_agent import __main__ as terminal_entrypoint
from loki_agent.credentials import CredentialStore


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
                asyncio.run(loki.async_provider_request(
                    "POST",
                    response.url,
                    {"model": "x"},
                    request_headers={},
                ))

    def test_chat_posts_and_model_gets_use_separate_timeouts(self):
        response = http_client.HttpResponse(
            "https://example.test/v1/chat/completions",
            200,
            "OK",
            {"content-type": "application/json"},
            b"{}",
        )

        for method, payload, expected_timeout in [
                ("POST", {"model": "x"}, loki.LLM_REQUEST_TIMEOUT_S),
                ("GET", None, loki.WEBFETCH_TIMEOUT_S)]:
            transport = mock.AsyncMock(return_value=response)
            with self.subTest(method=method), mock.patch.object(
                    http_client, "async_http_request", new=transport):
                provider_response = asyncio.run(loki.async_provider_request(
                    method,
                    response.url,
                    payload,
                    request_headers={},
                ))

            self.assertIsInstance(
                provider_response, protocols.ProviderResponse)
            self.assertEqual(provider_response.payload, {})
            self.assertEqual(transport.await_args.args[0], method)
            self.assertEqual(
                transport.await_args.kwargs["timeout"],
                expected_timeout,
            )

    def test_tool_loop_reports_provider_protocol_error_without_appending(self):
        events = []
        transcript = [
            loki.formats.message_item("user", "hello"),
        ]

        async def broken_chat(_items, *, codex_turn_state):
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
    def test_exit_revokes_then_awaits_credential_capability_cleanup(self):
        class Capability:
            def __init__(self):
                self.revoked = False
                self.closed = False

            def close_now(self):
                self.revoked = True

            async def close(self):
                self.assert_revoked()
                await asyncio.sleep(0)
                self.closed = True

            def assert_revoked(self):
                if not self.revoked:
                    raise AssertionError("cleanup started before revocation")

        async def scenario():
            manager = loki.JobManager("/tmp/loki-job-cleanup-test")
            capability = Capability()
            job = types.SimpleNamespace(
                status="running",
                signal=None,
                exit_code=None,
                finished_at=None,
                finished_at_iso=None,
                owner_signal_fd=None,
                credential_capability=capability,
            )
            with mock.patch.object(manager, "_write_metadata"):
                manager._record_exit(job, 0)
                self.assertTrue(capability.revoked)
                self.assertFalse(capability.closed)
                self.assertIs(job.credential_capability, capability)
                await manager._close_credential_capability(job)
            return job, capability

        job, capability = asyncio.run(scenario())

        self.assertTrue(capability.closed)
        self.assertIsNone(job.credential_capability)

    def test_subagent_slots_bound_concurrent_children_and_release_on_exit(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            command = [
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            ]
            first = await manager.run_background_exec(
                command, cwd=tmpdir, session_owned=True, subagent=True)
            second = await manager.run_background_exec(
                command, cwd=tmpdir, session_owned=True, subagent=True)
            third = None
            try:
                with self.assertRaises(loki.SubagentCapacityError):
                    await manager.run_background_exec(
                        command,
                        cwd=tmpdir,
                        session_owned=True,
                        subagent=True,
                    )

                os.killpg(first.pgid, signal.SIGTERM)
                await asyncio.wait_for(first.process.wait(), timeout=3)
                manager._refresh_job(first)
                third = await manager.run_background_exec(
                    command,
                    cwd=tmpdir,
                    session_owned=True,
                    subagent=True,
                )
                return manager, first, second, third
            finally:
                for job in (first, second, third):
                    if (job is not None
                            and job.process.returncode is None):
                        os.killpg(job.pgid, signal.SIGKILL)
                        await job.process.wait()
                        manager._refresh_job(job)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager, first, second, third = asyncio.run(scenario(tmpdir))

        self.assertFalse(first.subagent_slot)
        self.assertFalse(second.subagent_slot)
        self.assertFalse(third.subagent_slot)
        self.assertEqual(manager._active_subagents, 0)

    def test_failed_subagent_spawn_releases_its_slot(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            with mock.patch.object(
                    loki.asyncio,
                    "create_subprocess_exec",
                    new=mock.AsyncMock(
                        side_effect=OSError("spawn failed")),
            ):
                with self.assertRaisesRegex(OSError, "spawn failed"):
                    await manager.run_background_exec(
                        ["missing"],
                        cwd=tmpdir,
                        session_owned=True,
                        subagent=True,
                    )
            return manager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = asyncio.run(scenario(tmpdir))

        self.assertEqual(manager._active_subagents, 0)

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

    def test_failed_credential_relay_setup_closes_owner_pipe(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            session = loki.current_session()
            old_authority = session.credential_authority
            session.credential_authority = (
                authentications.CredentialBroker())
            pipe_fds = []
            real_pipe = loki.os.pipe

            def recording_pipe():
                pair = real_pipe()
                pipe_fds.extend(pair)
                return pair

            try:
                with mock.patch.object(
                        loki.os, "pipe", side_effect=recording_pipe), \
                        mock.patch.object(
                            loki.credential_capabilities.
                            CredentialCapabilityServer,
                            "create",
                            new=mock.AsyncMock(
                                side_effect=RuntimeError("relay failed"))):
                    with self.assertRaisesRegex(
                            RuntimeError, "relay failed"):
                        await manager.run_background_exec(
                            [sys.executable, "-c", "pass"],
                            cwd=tmpdir,
                            session_owned=True,
                            credential_refs=frozenset(),
                        )
            finally:
                session.credential_authority = old_authority
            return pipe_fds

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe_fds = asyncio.run(scenario(tmpdir))

        self.assertEqual(len(pipe_fds), 2)
        for fd in pipe_fds:
            with self.assertRaises(OSError):
                os.fstat(fd)

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

    def test_session_close_reaps_owned_jobs_only(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            owned_script = (
                "import os,sys\n"
                "owner_fd = int(sys.argv[-1])\n"
                "print('owned-ready', flush=True)\n"
                "os.read(owner_fd, 1)\n"
            )
            ordinary_script = (
                "import time\n"
                "print('ordinary-ready', flush=True)\n"
                "time.sleep(30)\n"
            )
            owned = await manager.run_background_exec(
                [sys.executable, "-c", owned_script],
                cwd=tmpdir,
                session_owned=True,
            )
            ordinary = await manager.run_background_exec(
                [sys.executable, "-c", ordinary_script],
                cwd=tmpdir,
            )
            try:
                deadline = asyncio.get_running_loop().time() + 3
                while (
                        "owned-ready" not in loki._read_spool_tail(
                            owned.stdout_path)
                        or "ordinary-ready" not in loki._read_spool_tail(
                            ordinary.stdout_path)
                ):
                    if asyncio.get_running_loop().time() >= deadline:
                        self.fail("background processes did not become ready")
                    await asyncio.sleep(0.01)

                await manager.close_session_owned()
                self.assertIsNotNone(owned.process.returncode)
                self.assertIsNone(ordinary.process.returncode)
                with open(
                        owned.metadata_path,
                        encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)
                return owned, ordinary, metadata
            finally:
                if ordinary.process.returncode is None:
                    os.killpg(ordinary.pgid, signal.SIGKILL)
                    await ordinary.process.wait()

        with tempfile.TemporaryDirectory() as tmpdir:
            owned, ordinary, metadata = asyncio.run(scenario(tmpdir))
        self.assertEqual(owned.status, "owner_closed")
        self.assertEqual(owned.exit_code, 0)
        self.assertTrue(metadata["session_owned"])
        self.assertEqual(metadata["status"], "owner_closed")
        self.assertIsNotNone(ordinary.process.returncode)

    def test_delegated_credential_fd_is_not_inherited_by_child_command(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            credential = authentications.CredentialRef.environment(
                "EXAMPLE_API_KEY")
            broker = authentications.CredentialBroker()
            broker.install_static(credential, "delegated-secret")
            session = loki.current_session()
            old_authority = session.credential_authority
            session.credential_authority = broker
            probe_code = (
                "import os,sys\n"
                "try:\n"
                "    os.fstat(int(sys.argv[1]))\n"
                "except OSError:\n"
                "    print('closed')\n"
                "else:\n"
                "    print('inherited')\n"
            )
            script = (
                "import asyncio,os,subprocess,sys\n"
                "from loki_agent import authentications\n"
                "from loki_agent import credential_capabilities\n"
                "async def main():\n"
                "    fd = int(sys.argv[-1])\n"
                "    client = await "
                "credential_capabilities.CredentialClient.from_fd(fd)\n"
                "    lease = await client.lease("
                "authentications.CredentialRef.environment("
                "'EXAMPLE_API_KEY'))\n"
                "    probe = subprocess.check_output([\n"
                f"        sys.executable, '-c', {probe_code!r},\n"
                "        str(fd)], stderr=subprocess.STDOUT, text=True)\n"
                "    print(lease.value)\n"
                "    print(probe.strip())\n"
                "    await client.close()\n"
                "asyncio.run(main())\n"
            )
            try:
                return await manager.run_exec(
                    [sys.executable, "-c", script],
                    10_000,
                    cwd=os.path.dirname(os.path.dirname(__file__)),
                    credential_refs={credential},
                )
            finally:
                session.credential_authority = old_authority

        with tempfile.TemporaryDirectory() as tmpdir:
            job, status, stdout, stderr = asyncio.run(scenario(tmpdir))

        self.assertEqual(status, "completed", stderr)
        self.assertEqual(job.exit_code, 0, stderr)
        self.assertIn("delegated-secret", stdout)
        self.assertIn("closed", stdout)
        self.assertNotIn("inherited", stdout)

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


class TerminalEntrypointContractTests(unittest.TestCase):
    def test_public_entrypoint_protects_credential_supervisor(self):
        credentials = CredentialStore({})
        storage = mock.Mock()
        supervisor = mock.Mock()
        supervisor.run_terminal_runtime = mock.AsyncMock(
            return_value=17)
        with mock.patch.object(
                terminal_entrypoint,
                "capture_process_credentials",
                return_value=credentials) as capture, mock.patch.object(
                    terminal_entrypoint,
                    "protect_credential_process") as protect, mock.patch(
                            "loki_agent.credential_supervisors."
                            "CredentialSupervisor",
                            return_value=supervisor) as supervisor_class, \
                mock.patch(
                    "loki_agent.credential_storages."
                    "JsonCredentialStorage",
                    return_value=storage) as storage_class, \
                mock.patch.object(
                    sys, "argv", ["/checkout/loki.py", "--headless"]):
            status = terminal_entrypoint.main()

        self.assertEqual(status, 17)
        capture.assert_called_once_with()
        protect.assert_called_once_with()
        storage_class.assert_called_once_with()
        supervisor_class.assert_called_once_with(
            credentials, storage)
        supervisor.run_terminal_runtime.assert_awaited_once_with(
            "/checkout/loki.py", ["--headless"])

    def test_internal_runtime_never_captures_root_credentials(self):
        owner_read, owner_write = os.pipe()
        capability_read, capability_write = os.pipe()
        try:
            with mock.patch.object(
                    terminal_entrypoint,
                    "capture_process_credentials") as capture, \
                    mock.patch.object(
                        terminal_entrypoint,
                        "isolate_credential_directory") as isolate, \
                    mock.patch.object(
                        terminal_entrypoint,
                        "protect_credential_process") as protect, \
                    mock.patch.object(
                        terminal_frontend,
                        "main",
                        return_value=19) as terminal_main, \
                    mock.patch.object(sys, "argv", [
                        "/checkout/loki.py",
                        "--runtime",
                        "--session-owner-fd", str(owner_read),
                        "--credential-capability-fd",
                        str(capability_read),
                        "--",
                        "--headless",
                    ]):
                status = terminal_entrypoint.main()

            self.assertEqual(status, 19)
            capture.assert_not_called()
            isolate.assert_called_once_with()
            protect.assert_called_once_with()
            terminal_main.assert_called_once_with(
                ["--headless"], owner_read, capability_read)
        finally:
            os.close(owner_read)
            os.close(owner_write)
            os.close(capability_read)
            os.close(capability_write)

    def test_subagent_inherits_parent_isolation_and_never_captures(self):
        with mock.patch.object(
                terminal_entrypoint,
                "capture_process_credentials") as capture, \
                mock.patch.object(
                    terminal_entrypoint,
                    "isolate_credential_directory") as isolate, \
                mock.patch.object(
                    terminal_entrypoint,
                    "protect_credential_process") as protect, \
                mock.patch(
                    "loki_agent.subagents.main",
                    return_value=29) as subagent_main, \
                mock.patch.object(sys, "argv", [
                    "/checkout/loki.py",
                    "--subagent",
                    "Explore",
                    "--session-owner-fd", "7",
                    "--credential-capability-fd", "8",
                ]):
            status = terminal_entrypoint.main()

        self.assertEqual(status, 29)
        capture.assert_not_called()
        isolate.assert_not_called()
        protect.assert_called_once_with()
        subagent_main.assert_called_once_with([
            "Explore",
            "--session-owner-fd", "7",
            "--credential-capability-fd", "8",
        ])


class AgentModeContractTests(unittest.TestCase):
    def test_terminal_advertises_plan_toolset_in_plan_mode(self):
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

        # Plan mode advertises the read-only set plus TodoWrite.
        self.assertEqual(set(captured), loki.PLAN_TOOLS)


if __name__ == "__main__":
    unittest.main()
