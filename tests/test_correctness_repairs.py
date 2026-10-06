import asyncio
import contextlib
import errno
import json
import os
import pathlib
import signal
import sys
import tempfile
import types
import unittest
from unittest import mock
from response_header_fixtures import setUpModule  # noqa: F401 - unittest hook

from loki_agent import (
    authentications,
    formats,
    host_process,
    http_client,
    loki,
    protocols,
    terminal_frontend,
)
from loki_agent import __main__ as terminal_entrypoint
from loki_agent.credentials import CredentialStore


async def _finish_within(coroutine, seconds, label):
    """Await ``coroutine``, dumping every task's stack if it does not finish.

    A lost asyncio wakeup leaves the loop idle: the suite hangs with a bare
    test name and no frame, which is how a bug here became a job timeout.
    Bounding the await turns that into a failure naming the await each task is
    stuck on.
    """
    task = asyncio.ensure_future(coroutine)
    _done, pending = await asyncio.wait({task}, timeout=seconds)
    if not pending:
        return task.result()
    for other in asyncio.all_tasks():
        if other is not asyncio.current_task():
            other.print_stack()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    raise AssertionError(
        f"{label} did not finish within {seconds}s; task stacks above")


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

                host_process.signal_group(
                    first.process, first.pgid, signal.SIGTERM)
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
                for job in [first, second, third]:
                    if (job is not None
                            and job.process.returncode is None):
                        host_process.signal_group(
                            job.process, job.pgid, host_process.FORCE)
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
                # A non-forced stop is a signal on POSIX and CTRL_BREAK_EVENT
                # on Windows, so ignore both to make the first stop genuinely
                # ineffective on either platform.
                "if hasattr(signal, 'SIGBREAK'):\n"
                "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
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
                    host_process.signal_group(
                        job.process, job.pgid, host_process.FORCE)
                    await job.process.wait()

        with tempfile.TemporaryDirectory() as tmpdir:
            first, second, job, metadata = asyncio.run(scenario(tmpdir))
        self.assertIn(host_process.label(signal.SIGTERM), first)
        self.assertIn(host_process.label(host_process.FORCE), second)
        self.assertEqual(job.status, "stopped")
        self.assertIsNotNone(job.exit_code)
        self.assertEqual(metadata["status"], "stopped")
        self.assertEqual(metadata["exit_code"], job.exit_code)
        if os.name == "posix":
            # A forced stop is SIGKILL on POSIX.  Windows terminates instead,
            # which has an ordinary exit code and no signal number.
            self.assertEqual(job.exit_code, -signal.SIGKILL)
            self.assertEqual(job.signal, signal.SIGKILL)
            self.assertEqual(metadata["signal"], signal.SIGKILL)

    def test_stopped_foreground_still_cancels_times_out_and_unwinds(self):
        from process_lifecycle_fixtures import ProcessResources

        async def scenario(tmpdir, cause):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            cancel_event = asyncio.Event()
            release_spawn = asyncio.Event()
            tasks_before = asyncio.all_tasks()
            script = (
                "import signal,time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                "if hasattr(signal, 'SIGBREAK'):\n"
                "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
                "print('ready', flush=True)\n"
                "time.sleep(60)\n"
            )
            real_spawn = manager._spawn

            async def held_spawn(*args, **kwargs):
                job = await real_spawn(*args, **kwargs)
                await release_spawn.wait()
                return job

            spawn_gate = (
                mock.patch.object(manager, "_spawn", side_effect=held_spawn)
                if cause == "spawn_cancel" else contextlib.nullcontext())
            with spawn_gate:
                task = asyncio.create_task(manager.run_foreground(
                    [sys.executable, "-c", script], "uncooperative command",
                    5_000 if cause == "timeout" else 60_000,
                    cwd=tmpdir, cancel_event=cancel_event))
                try:
                    deadline = asyncio.get_running_loop().time() + 4
                    while True:
                        job = next(iter(manager.jobs.values()), None)
                        if job is not None and job.process is not None:
                            self.assertIsNone(job.process.returncode)
                            if "ready" in loki._read_spool_tail(job.stdout_path):
                                break
                        if asyncio.get_running_loop().time() >= deadline:
                            self.fail("foreground process did not become ready")
                        await asyncio.sleep(.01)
                    resources = ProcessResources(job.process)
                    with mock.patch.object(
                            loki.current_session(), "job_manager", manager):
                        self.assertIn("Sent", loki.run_ps("stop " + job.id))
                    self.assertEqual(job.status, "stopping")
                    self.assertIsNone(job.process.returncode)
                    if cause in ["cancel", "spawn_cancel"]:
                        cancel_event.set()
                    elif cause == "unwind":
                        task.cancel()
                    release_spawn.set()
                    done, _pending = await asyncio.wait({task}, timeout=9)
                    self.assertTrue(done, "stopping job blocked foreground cleanup")
                    expected = "timed_out" if cause == "timeout" else "cancelled"
                    if cause == "unwind":
                        with self.assertRaises(asyncio.CancelledError):
                            task.result()
                    else:
                        result_job, outcome, stdout, stderr = task.result()
                        self.assertIs(result_job, job)
                        self.assertEqual(outcome, expected)
                        self.assertIn("ready", stdout)
                    self.assertEqual(job.status, expected)
                    self.assertIsNotNone(job.process.returncode)
                    if os.name == "posix":
                        self.assertEqual(job.exit_code, -signal.SIGKILL)
                        self.assertEqual(job.signal, signal.SIGKILL)
                    with open(job.metadata_path, encoding="utf-8") as stream:
                        metadata = json.load(stream)
                    self.assertEqual(metadata["status"], expected)
                    self.assertEqual(metadata["exit_code"], job.process.returncode)
                    self.assertEqual(metadata["signal"], job.signal)
                    self.assertEqual(cancel_event.is_set(),
                                     cause in ["cancel", "spawn_cancel"])
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    resources.assert_released(self)
                    self.assertFalse(asyncio.all_tasks() - tasks_before)
                finally:
                    # Backstop after the assertions, including a broken waiter.
                    for job in manager.jobs.values():
                        if job.process is not None and job.process.returncode is None:
                            host_process.signal_group(job.process, job.pgid,
                                                      host_process.FORCE)
                            await asyncio.wait_for(job.process.wait(), 3)
                    release_spawn.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.wait_for(
                        asyncio.gather(task, return_exceptions=True), 3)

        for cause in ["cancel", "timeout", "unwind", "spawn_cancel"]:
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as tmpdir:
                asyncio.run(scenario(tmpdir, cause))

    def test_foreground_completion_and_simultaneous_cancellation(self):
        async def scenario(tmpdir, simultaneous_cancel):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            cancel_event = asyncio.Event()
            real_wait = manager._wait_for_job

            async def exit_then_cancel(job):
                exit_code = await real_wait(job)
                cancel_event.set()
                return exit_code

            waiter = (
                mock.patch.object(manager, "_wait_for_job",
                                  side_effect=exit_then_cancel)
                if simultaneous_cancel else contextlib.nullcontext())
            with waiter:
                job, outcome, stdout, stderr = await asyncio.wait_for(
                    manager.run_foreground(
                        [sys.executable, "-c", "print('finished')"],
                        "finite command", 5_000, cwd=tmpdir,
                        cancel_event=cancel_event), 8)
            self.assertEqual(outcome,
                             "cancelled" if simultaneous_cancel else "completed")
            self.assertEqual(job.status,
                             "cancelled" if simultaneous_cancel else "exited")
            self.assertEqual(job.exit_code, 0)
            self.assertIn("finished", stdout)
            self.assertEqual(stderr, "")
            self.assertEqual(cancel_event.is_set(), simultaneous_cancel)
            with open(job.metadata_path, encoding="utf-8") as stream:
                self.assertEqual(json.load(stream)["status"], job.status)

        for simultaneous_cancel in [False, True]:
            with self.subTest(cancel=simultaneous_cancel), tempfile.TemporaryDirectory() as tmpdir:
                asyncio.run(scenario(tmpdir, simultaneous_cancel))

    def test_failed_credential_relay_setup_closes_its_ends(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            session = loki.current_session()
            old_authority = session.credential_authority
            session.credential_authority = (
                authentications.CredentialBroker())
            ends = []
            real_channel = loki.host_ipc.owner_channel

            def recording_channel():
                pair = real_channel()
                ends.extend(pair)
                return pair

            try:
                with mock.patch.object(
                        loki.host_ipc, "owner_channel",
                        side_effect=recording_channel), \
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
            return ends

        with tempfile.TemporaryDirectory() as tmpdir:
            ends = asyncio.run(scenario(tmpdir))

        self.assertEqual(len(ends), 2)
        for end in ends:
            if isinstance(end, int):
                # A POSIX descriptor; nothing here observes its closure.
                continue
            if loki.host_ipc.is_endpoint(end):
                # A closed Windows pipe endpoint has released its handles.
                self.assertEqual(end.handles(), [])
            else:
                # A closed socket has no handle.
                self.assertEqual(end.fileno(), -1)

    def test_session_owned_job_does_not_hold_the_end_the_runtime_keeps(self):
        """Closing the runtime's end reaches the job, so no copy leaked to it.

        The runtime keeps one end of the channel and hands the other to the
        job.  A read on the handed end returns only once every write end of
        the pipe is closed, so closing the runtime's end and watching the job
        observe that proves the job holds no write end of that pipe.  A job
        holding the kept end would keep the pipe open and never be revoked.
        """
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            script = (
                "import asyncio, sys\n"
                "from loki_agent import host_ipc\n"
                "async def main():\n"
                "    end = host_ipc.child_endpoint(\n"
                "        sys.argv[sys.argv.index('--session-owner-fd') + 1])\n"
                "    await host_ipc.watch_closed(end)\n"
                "    print('closed', flush=True)\n"
                "asyncio.run(main())\n"
            )
            job = await manager.run_background_exec(
                [sys.executable, "-c", script],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                session_owned=True,
            )
            try:
                manager._close_owner_signal(job)
                deadline = asyncio.get_running_loop().time() + 5
                while True:
                    reported = loki._read_spool_tail(job.stdout_path)
                    if "closed" in reported:
                        return reported
                    if asyncio.get_running_loop().time() >= deadline:
                        self.fail(
                            "the job did not observe the owner end close")
                    await asyncio.sleep(0.01)
            finally:
                await manager.close_session_owned()

        with tempfile.TemporaryDirectory() as tmpdir:
            reported = asyncio.run(scenario(tmpdir))

        self.assertIn("closed", reported)

    def test_command_spawned_by_a_job_does_not_hold_the_job_owner_end(self):
        """A command the job spawns holds no write end of the job's channel.

        Closing the runtime's end must reach the job while the command it
        spawned is still alive.  If the command held a write end of that pipe,
        the job's read would stay blocked and revocation would never reach it.
        """
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            script = (
                "import asyncio, sys, tempfile\n"
                "from loki_agent import host_ipc, loki\n"
                "async def main():\n"
                "    end = host_ipc.child_endpoint(\n"
                "        sys.argv[sys.argv.index('--session-owner-fd') + 1])\n"
                "    manager = loki.JobManager(tempfile.mkdtemp())\n"
                "    command = await manager.run_background_exec(\n"
                "        [sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                "    await host_ipc.watch_closed(end)\n"
                "    command.process.kill()\n"
                "    print('closed', flush=True)\n"
                "asyncio.run(main())\n"
            )
            job = await manager.run_background_exec(
                [sys.executable, "-c", script],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                session_owned=True,
            )
            try:
                manager._close_owner_signal(job)
                deadline = asyncio.get_running_loop().time() + 5
                while True:
                    reported = loki._read_spool_tail(job.stdout_path)
                    if "closed" in reported:
                        return reported
                    if asyncio.get_running_loop().time() >= deadline:
                        self.fail("a command held the job's owner end")
                    await asyncio.sleep(0.01)
            finally:
                await manager.close_session_owned()

        with tempfile.TemporaryDirectory() as tmpdir:
            reported = asyncio.run(scenario(tmpdir))

        self.assertIn("closed", reported)

    def test_cancelling_owner_task_reaps_foreground_process(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            task = asyncio.create_task(manager.run_exec(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                60_000, cwd=tmpdir))
            # A job is registered before it is launched, so wait (bounded) for
            # a live process rather than for the job to appear, or a cancelled
            # launch would be observed instead of a cancelled running job.
            deadline = asyncio.get_running_loop().time() + 5
            while not any(job.process is not None
                          for job in manager.jobs.values()):
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("job never started")
                await asyncio.sleep(0.01)
            job = next(job for job in manager.jobs.values()
                       if job.process is not None)
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

    def test_worker_owned_and_ordinary_job_lifecycle(self):
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session
        from process_lifecycle_fixtures import ProcessResources

        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            session = Session(shell_cwd=tmpdir)
            session.job_manager = manager
            worker = Worker(session, lambda message: None, "session")
            resources = []
            tasks_before = asyncio.all_tasks()
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            owned_cwd = os.path.join(tmpdir, "owned")
            ordinary_cwd = os.path.join(tmpdir, "ordinary")
            os.mkdir(owned_cwd)
            os.mkdir(ordinary_cwd)
            owned_script = (
                "import asyncio,sys,os\n"
                f"sys.path.insert(0, {root!r})\n"
                "from loki_agent import credential_runtimes, host_ipc\n"
                "async def main():\n"
                "    owner = credential_runtimes.SessionOwner(\n"
                "        host_ipc.child_endpoint(sys.argv[-1]))\n"
                "    print('owned-ready:' + os.getcwd(), flush=True)\n"
                "    await owner.closed_task\n"
                "    print('owner-eof', flush=True)\n"
                "asyncio.run(main())\n"
            )
            try:
                with mock.patch.object(loki, '_DEFAULT_SESSION', session):
                    owned = await manager.run_background_exec(
                        [sys.executable, "-c", owned_script], cwd=owned_cwd,
                        session_owned=True, subagent=True)
                    resources.append(ProcessResources(owned.process))
                    ordinary = await manager.run_background_exec(
                        [sys.executable, "-c",
                         "import os,time; print('ordinary-ready:' + os.getcwd(), "
                         "flush=True); time.sleep(60)"], cwd=ordinary_cwd)
                    resources.append(ProcessResources(ordinary.process))
                    self.assertEqual(owned.cwd, owned_cwd)
                    self.assertEqual(ordinary.cwd, ordinary_cwd)
                    self.assertNotEqual(owned.id, ordinary.id)
                    deadline = asyncio.get_running_loop().time() + 5
                    for job, marker in ((owned, 'owned-ready:'),
                                        (ordinary, 'ordinary-ready:')):
                        while marker + job.cwd not in loki._read_spool_tail(job.stdout_path):
                            self.assertIsNone(job.process.returncode,
                                              loki._read_spool_tail(job.stderr_path))
                            if asyncio.get_running_loop().time() >= deadline:
                                self.fail('children did not become ready')
                            await asyncio.sleep(.01)
                    self.assertEqual(manager._active_subagents, 1)
                    listing = loki.run_ps()
                    self.assertTrue(listing.startswith(manager.list_jobs() + '\n'))
                    self.assertIn('/ps ID', listing)
                    self.assertEqual(listing.splitlines()[0], 'Jobs:')
                    for job in [owned, ordinary]:
                        self.assertIsNone(job.process.returncode)
                        self.assertEqual(job.status, 'running')
                        lines = [line for line in listing.splitlines()
                                 if line.startswith(f'{job.id}. ')]
                        self.assertEqual(len(lines), 1)
                        line, = lines
                        self.assertIn('status=running', line)
                        self.assertIn(f'pid={job.process.pid}', line)
                        self.assertIn(f'cwd={job.cwd!r}', line)
                        status = manager.job_status(job.id)
                        self.assertIn(f'cwd: {job.cwd}', status)
                        self.assertIn('status: running', status)
                        with open(job.metadata_path, encoding='utf-8') as stream:
                            metadata = json.load(stream)
                        self.assertEqual(metadata['cwd'], job.cwd)
                        self.assertEqual(metadata['status'], 'running')
                        self.assertEqual(metadata['session_owned'], job is owned)
                    await _finish_within(worker.close(), 8, 'Worker close')
                    self.assertEqual(owned.process.returncode, 0)
                    self.assertEqual(owned.exit_code, 0)
                    self.assertEqual(owned.status, 'owner_closed')
                    self.assertIn('owner-eof', loki._read_spool_tail(owned.stdout_path))
                    self.assertIsNone(owned.owner_signal_fd)
                    self.assertFalse(owned.subagent_slot)
                    self.assertEqual(manager._active_subagents, 0)
                    with open(owned.metadata_path, encoding='utf-8') as stream:
                        metadata = json.load(stream)
                    self.assertEqual(metadata['status'], 'owner_closed')
                    self.assertEqual(metadata['exit_code'], 0)
                    self.assertTrue(metadata['session_owned'])
                    await asyncio.sleep(0)
                    resources[0].assert_released(self)
                    await _finish_within(worker.close(), 8, 'repeat Worker close')
                    self.assertIsNone(ordinary.process.returncode)
                    self.assertEqual(ordinary.status, 'running')
                    self.assertFalse(ordinary.session_owned)
                    with open(ordinary.metadata_path, encoding='utf-8') as stream:
                        self.assertEqual(json.load(stream)['status'], 'running')
                    host_process.signal_group(ordinary.process, ordinary.pgid,
                                              host_process.FORCE)
                    await _finish_within(ordinary.process.wait(), 5, 'ordinary reap')
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    resources[1].assert_released(self)
                    self.assertFalse(asyncio.all_tasks() - tasks_before)
            finally:
                for resource in resources:
                    await resource.cleanup()

        with tempfile.TemporaryDirectory() as tmpdir:
            asyncio.run(_finish_within(scenario(tmpdir), 30, 'job lifecycle'))

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

    def test_a_signal_failure_is_not_reported_as_a_gone_process(self):
        manager = loki.JobManager(
            os.path.join(tempfile.gettempdir(), "loki-signal-audit"))
        job = types.SimpleNamespace(
            process=types.SimpleNamespace(returncode=None, pid=1), pgid=1)

        with mock.patch.object(
                loki.host_process, "signal_group",
                side_effect=PermissionError(errno.EPERM, "denied")):
            with self.assertRaises(PermissionError):
                manager._signal_process_group(job, signal.SIGTERM)

        with mock.patch.object(
                loki.host_process, "signal_group",
                side_effect=ProcessLookupError(errno.ESRCH, "gone")):
            self.assertFalse(manager._signal_process_group(job, signal.SIGTERM))

    def test_a_post_spawn_failure_reaps_the_child_and_frees_its_slot(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            with mock.patch.object(manager, "_write_metadata",
                                   side_effect=OSError("metadata unavailable")):
                with self.assertRaisesRegex(OSError, "metadata unavailable"):
                    await manager.run_exec(
                        [sys.executable, "-c",
                         "import time; time.sleep(30)"],
                        5_000, cwd=tmpdir, session_owned=True, subagent=True)
            return manager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = asyncio.run(scenario(tmpdir))

        job = next(iter(manager.jobs.values()))
        self.assertEqual(job.status, "failed")
        self.assertIsNotNone(job.process.returncode)
        self.assertFalse(job.subagent_slot)
        self.assertEqual(manager._active_subagents, 0)

    def test_recording_a_failure_does_not_replace_it(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            with mock.patch.object(
                    asyncio, "create_subprocess_exec",
                    side_effect=FileNotFoundError("no such program")), \
                    mock.patch.object(
                        manager, "_write_metadata",
                        side_effect=PermissionError("cannot record")):
                with self.assertRaises(FileNotFoundError):
                    await manager.run_exec(
                        ["definitely-not-a-program"], 1_000, cwd=tmpdir)

        with tempfile.TemporaryDirectory() as tmpdir:
            asyncio.run(scenario(tmpdir))

    def test_session_close_during_launch_does_not_publish_a_live_child(self):
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session
        from process_lifecycle_fixtures import ProcessResources

        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            session = Session(shell_cwd=tmpdir)
            session.job_manager = manager
            worker = Worker(session, lambda message: None, "session")
            entered, release = asyncio.Event(), asyncio.Event()
            real = asyncio.create_subprocess_exec
            resources = []
            tasks_before = asyncio.all_tasks()

            async def delayed(*args, **kwargs):
                entered.set()
                await release.wait()
                process = await real(*args, **kwargs)
                resources.append(ProcessResources(process))
                return process

            task = None
            try:
                with mock.patch.object(asyncio, "create_subprocess_exec", new=delayed):
                    task = asyncio.create_task(manager.run_background_exec(
                        [sys.executable, "-c", "import time; time.sleep(60)"],
                        cwd=tmpdir, session_owned=True, subagent=True))
                    await _finish_within(entered.wait(), 5, 'launch barrier')
                    self.assertEqual(manager._active_subagents, 1)
                    await _finish_within(worker.close(), 5, 'close during launch')
                    release.set()
                    with self.assertRaises(loki._JobRevokedDuringLaunch) as raised:
                        await _finish_within(task, 5, 'revoked launch')
                    self.assertIs(type(raised.exception), loki._JobRevokedDuringLaunch)
                self.assertEqual(len(resources), 1)
                # Observe the captured unpublished child before any backstop.
                self.assertIsNotNone(resources[0].process.returncode)
                await asyncio.sleep(0)
                resources[0].assert_released(self)
                job, = manager.jobs.values()
                self.assertIsNone(job.process)
                self.assertIsNone(job.owner_signal_fd)
                self.assertEqual(job.status, 'cancelled')
                self.assertFalse(job.subagent_slot)
                self.assertEqual(manager._active_subagents, 0)
                with open(job.metadata_path, encoding='utf-8') as stream:
                    self.assertEqual(json.load(stream)['status'], 'cancelled')
                self.assertTrue(task.done())
                await _finish_within(worker.close(), 5, 'repeat race close')
                self.assertFalse(asyncio.all_tasks() - tasks_before)
            finally:
                release.set()
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                for resource in resources:
                    await resource.cleanup()

        with tempfile.TemporaryDirectory() as tmpdir:
            asyncio.run(_finish_within(scenario(tmpdir), 20, 'launch race'))

    def test_a_failed_spawn_is_left_as_a_recorded_failed_job(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            with self.assertRaises(OSError):
                await manager.run_exec(
                    [os.path.join(tmpdir, "not-a-program")], 5_000,
                    cwd=tmpdir)
            return manager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = asyncio.run(scenario(tmpdir))

            jobs = list(manager.jobs.values())
            self.assertEqual(len(jobs), 1)
            job = jobs[0]
            self.assertEqual(job.status, "failed")
            self.assertIsInstance(job.error, str)
            self.assertIsNone(job.process)
            with open(job.metadata_path, encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["error"], job.error)
            with open(job.stderr_path, encoding="utf-8") as stderr_file:
                self.assertIn("launch failed", stderr_file.read())

    def test_a_refused_subagent_is_recorded_and_frees_its_reservation(self):
        async def scenario(tmpdir):
            manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
            manager._active_subagents = loki.MAX_ACTIVE_DIRECT_SUBAGENTS
            with self.assertRaises(loki.SubagentCapacityError):
                await manager.run_exec(
                    [os.path.join(tmpdir, "not-a-program")], 5_000,
                    cwd=tmpdir, session_owned=True, subagent=True)
            return manager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = asyncio.run(scenario(tmpdir))

        self.assertEqual(manager._active_subagents,
                         loki.MAX_ACTIVE_DIRECT_SUBAGENTS)
        job = next(iter(manager.jobs.values()))
        self.assertEqual(job.status, "failed")
        self.assertIn("SubagentCapacityError", job.error)
        self.assertFalse(job.subagent_slot)


class TerminalEntrypointContractTests(unittest.TestCase):
    def setUp(self):
        # The runtime_isolation seam gates the entrypoints on the platform
        # module before the work under test: windows_runtime on Windows,
        # runtime_isolations on POSIX.  Mock the one this host selects so the
        # shared credential routing is what these tests exercise.
        self.isolation = None
        self.windows_steps = {}
        if os.name == "posix":
            from loki_agent import runtime_isolations
            patcher = mock.patch.object(
                runtime_isolations, "isolate_credential_directory")
            self.isolation = patcher.start()
            self.addCleanup(patcher.stop)
        else:
            from loki_agent import windows_runtime
            for name, value in (("configured_workspace", "/workspace"),
                                ("verify_runtime", None),
                                ("ensure_runtime_temp", None)):
                patcher = mock.patch.object(
                    windows_runtime, name, return_value=value)
                self.windows_steps[name] = patcher.start()
                self.addCleanup(patcher.stop)

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
        with mock.patch.object(
                terminal_entrypoint,
                "capture_process_credentials") as capture, \
                mock.patch.object(
                    terminal_entrypoint,
                    "protect_credential_process") as protect, \
                mock.patch.object(
                    terminal_entrypoint,
                    "_terminal_runtime_arguments",
                    return_value=(11, 12, ["--headless"])) as descriptors, \
                mock.patch.object(
                    terminal_frontend,
                    "main",
                    return_value=19) as terminal_main, \
                mock.patch.object(
                    sys, "argv", ["/checkout/loki.py", "--runtime"]):
            status = terminal_entrypoint.main()

        self.assertEqual(status, 19)
        capture.assert_not_called()
        protect.assert_called_once_with()
        # This is the routing contract: the descriptors the seam produced
        # reach the frontend, and the runtime never captures root credentials.
        descriptors.assert_called_once_with([])
        terminal_main.assert_called_once_with(["--headless"], 11, 12)
        if os.name == "posix":
            self.isolation.assert_called_once_with()
        else:
            self.windows_steps["verify_runtime"].assert_called_once_with()

    def test_subagent_inherits_parent_isolation_and_never_captures(self):
        with mock.patch.object(
                terminal_entrypoint,
                "capture_process_credentials") as capture, \
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
        protect.assert_called_once_with()
        if os.name == "posix":
            # A subagent inherits the runtime's view; it does not isolate again.
            self.isolation.assert_not_called()
        else:
            self.windows_steps["verify_runtime"].assert_called_once_with()
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
                asyncio.run(terminal_frontend.run_terminal_turn_async(
                    [formats.message_item("user", "plan this")],
                    # Plan mode may ask the user clarifying questions; an
                    # asker must be present for Ask to be advertised.
                    ask_user=mock.AsyncMock(return_value=None)))
        finally:
            loki.current_session().agent_mode = old_mode
            loki.current_session().session_toolsets = old_toolsets

        # Plan mode advertises the read-only set plus TodoWrite and Ask.
        self.assertEqual(set(captured), loki.PLAN_TOOLS)


if __name__ == "__main__":
    unittest.main()
