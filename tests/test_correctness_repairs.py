import asyncio
import contextlib
import errno
import json
import os
import pathlib
import shlex
import signal
import sys
import tempfile
import types
import unittest
from unittest import mock
from response_header_fixtures import setUpModule  # noqa: F401 - unittest hook
from job_fixtures import job_manager, managed_jobs
from process_lifecycle_fixtures import observe_fixture

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
from loki_agent import jobs as job_runtime, tool_contracts
from loki_agent import model_requests


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
                ("POST", {"model": "x"}, model_requests.LLM_REQUEST_TIMEOUT_S),
                ("GET", None, http_client.HTTP_DEFAULT_TIMEOUT_S)]:
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
        loki.current_session().file_observations.clear()

    def test_binary_change_after_read_blocks_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "payload.bin")
            pathlib.Path(path).write_bytes(b"\x00old")
            self.assertIn("binary", loki.run_read(path))

            pathlib.Path(path).write_bytes(b"\x00new")
            result = asyncio.run(loki.dispatch_tool_async('Write', {'file_path': path, 'content': 'replacement'}))

            self.assertFalse(result['ok'], result)
            self.assertIn('changed on disk', result['content'])
            self.assertEqual(pathlib.Path(path).read_bytes(), b"\x00new")

    def test_deletion_after_read_blocks_recreation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "notes.txt")
            pathlib.Path(path).write_text("old", encoding="utf-8")
            loki.run_read(path)
            os.unlink(path)

            result = asyncio.run(loki.dispatch_tool_async('Write', {'file_path': path, 'content': 'replacement'}))

            self.assertFalse(result['ok'], result)
            self.assertIn('checking current file contents', result['content'])
            self.assertFalse(os.path.exists(path))

    def test_empty_string_is_a_valid_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.txt")
            result = loki.run_write(path, "")
            self.assertIn("Successfully wrote", result)
            self.assertEqual(pathlib.Path(path).read_text(encoding="utf-8"), "")


class JobOwnershipContractTests(unittest.IsolatedAsyncioTestCase):
    """Keep actors live until the controller's stop, cancellation or expiry.

    Finite sleeps can finish while the controller is paused, substituting
    natural exit for the ownership transition under test. These actors block
    indefinitely; each existing job/worker scope or explicit backstop owns
    termination and joins the real child before releasing its state directory.
    Product cleanup assertions run before managed_jobs retires fixture resources;
    ordinary-child backstops remain explicit because manager closure is not kill.
    """
    async def failed_launch_wait(self, boundary):
        from loki_agent import windows_api, windows_runtime, windows_subprocesses

        native = windows_runtime.ContainedProcess(
            windows_api.ProcessInformation(71, 72, 42, 0), job=44)
        observer = asyncio.create_task(native.wait())
        protocol = windows_subprocesses.streams.ProcessProtocol()
        transport = mock.Mock(_exit_task=observer)
        transport.get_pid.return_value = 42
        transport.get_returncode.side_effect = lambda: native.returncode
        process = windows_subprocesses.Process(transport, protocol)
        process._family = native
        manager = job_runtime.JobManager.__new__(job_runtime.JobManager)
        manager._closed, manager._session_fd = False, None
        manager._counter = manager._active_subagents = 0
        manager._monitors, manager.jobs = set(), {}
        manager.session_dir = 'substituted-job-storage'
        manager._create_exec = mock.AsyncMock(return_value=process)
        manager._job_directory = mock.Mock(return_value=99)
        manager._open_spool_stream = mock.Mock(side_effect=lambda *args: contextlib.nullcontext(object()))

        def record(_job):
            if boundary == 'metadata' and manager._write_metadata.call_count == 1:
                raise OSError('metadata unavailable')

        manager._write_metadata = mock.Mock(side_effect=record)
        manager._append_spool = mock.Mock()
        closed = []
        owner = job_runtime.host_ipc.PipeEndpoint(write=201)
        child = job_runtime.host_ipc.PipeEndpoint(read=202)
        with mock.patch.object(windows_api, 'wait_for_single_object', return_value=0xffffffff), \
                mock.patch.object(windows_api, 'close_handle', side_effect=closed.append), \
                mock.patch.object(host_process, 'sys', types.SimpleNamespace(platform='win32')), \
                mock.patch.object(host_process, 'signal_group', side_effect=OSError('termination refused')) as signal_group, \
                mock.patch.object(host_process, 'process_group', return_value=42), \
                mock.patch.object(job_runtime.private_files, 'close', side_effect=(
                    OSError('directory close failed') if boundary == 'directory' else None)), \
                mock.patch.object(job_runtime.host_ipc, 'eof_stdin', return_value=(123, lambda: None)), \
                mock.patch.object(job_runtime.host_ipc, 'owner_channel', return_value=(owner, child)), \
                mock.patch.object(job_runtime.host_ipc, 'reference', return_value=202), \
                mock.patch.object(job_runtime.host_ipc, 'spawn_kwargs', return_value={}):
            try:
                with self.assertRaises(Exception) as caught:
                    await manager.run_background_exec(
                        ['chosen-image'], session_owned=True, subagent=True)
                job = next(iter(manager.jobs.values()))
                self.assertEqual(native.job, 44)
                self.assertEqual(native.information.hProcess, 71)
                self.assertIsNone(native.returncode)
                self.assertIs(job.process, process)
                self.assertIs(caught.exception, observer.exception())
                self.assertEqual(job.status, 'monitor_error')
                self.assertIsNone(job.finished_at)
                self.assertIsNone(job.exit_code)
                self.assertTrue(job.subagent_slot)
                self.assertEqual(manager._active_subagents, 1)
                self.assertIsNone(job.owner_signal_fd)
                self.assertCountEqual(closed, [201, 202])
                manager._session_fd = 99
                with mock.patch.object(manager, '_finish_owned_job', new=mock.AsyncMock(
                        side_effect=observer.exception())):
                    with self.assertRaises(ExceptionGroup):
                        await manager.aclose()
                self.assertEqual(manager._session_fd, 99)
                self.assertEqual(native.job, 44)
                # An explicit stop must remain available without treating its
                # request as an observed exit or enabling manager detachment.
                signal_group.side_effect = None
                self.assertIn('Sent', manager.stop_job(job.id, force=True))
                self.assertEqual(job.status, 'monitor_error')
                self.assertTrue(job.group_active)
                self.assertEqual(native.job, 44)
                with self.assertRaises(RuntimeError):
                    manager.close()
            finally:
                manager._session_fd = None
                native.close()  # Only substituted records; explicit fixture backstop.
                await asyncio.gather(observer, return_exceptions=True)

    async def test_repeated_cancellation_joins_failed_launch_rollback(self):
        stopping = asyncio.Event()
        signal_group = host_process.signal_group
        with tempfile.TemporaryDirectory() as directory:
            async with managed_jobs(directory) as manager:
                process = None
                with mock.patch.object(manager, '_write_metadata', side_effect=OSError('metadata unavailable')), \
                        mock.patch.object(host_process, 'signal_group', side_effect=lambda *args: stopping.set()):
                    task = asyncio.create_task(manager.run_background_exec(
                        [sys.executable, '-c', 'import threading; threading.Event().wait()'],
                        session_owned=True, subagent=True))
                    try:
                        await observe_fixture(stopping.wait(), task)
                        job = next(iter(manager.jobs.values()))
                        process = job.process
                        for _ in range(2):
                            task.cancel()
                            await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        self.assertIsNone(process.returncode)
                        self.assertTrue(job.group_active)
                        self.assertTrue(job.subagent_slot)
                        signal_group(process, job.pgid, host_process.FORCE)
                        with self.assertRaisesRegex(OSError, 'metadata unavailable'):
                            await task
                        self.assertIsNotNone(process.returncode)
                        self.assertFalse(job.group_active)
                        self.assertFalse(job.subagent_slot)
                        self.assertEqual(manager._active_subagents, 0)
                    finally:
                        if process is not None:
                            if process.returncode is None:
                                signal_group(process, job.pgid, host_process.FORCE)
                            await process.aclose()
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

    async def test_post_publication_wait_failure_retains_family_and_capacity(self):
        await self.failed_launch_wait('metadata')

    async def test_pre_publication_wait_failure_retains_the_created_process(self):
        await self.failed_launch_wait('directory')

    async def _output_until(self, manager, job, marker):
        while marker not in manager._spool_tail(job, 'stdout.log'):
            self.assertIsNone(job.process.returncode,
                              manager._spool_tail(job, 'stderr.log'))
            await asyncio.sleep(.01)
        return manager._spool_tail(job, 'stdout.log')

    async def test_owner_close_failure_still_revokes_releases_capacity_and_records_exit(self):
        from loki_agent import windows_api

        failure = windows_api.WindowsApiError('CloseHandle failed', status=6)
        with tempfile.TemporaryDirectory() as root:
            async with managed_jobs(root) as manager:
                capability = mock.Mock()
                job = types.SimpleNamespace(
                    status='running', group_active=False, signal=None, exit_code=None,
                    owner_signal_fd=object(), credential_capability=capability,
                    subagent_slot=True, error=None)
                manager._active_subagents = 1
                with mock.patch.object(job_runtime.host_ipc, 'close_end', side_effect=failure), \
                        mock.patch.object(manager, '_write_metadata') as record:
                    with self.assertRaises(windows_api.WindowsApiError) as caught:
                        manager._record_exit(job, 0)
                    self.assertIs(caught.exception, failure)
                    capability.close_now.assert_called_once_with()
                    self.assertFalse(job.subagent_slot)
                    self.assertEqual(manager._active_subagents, 0)
                    self.assertIsNone(job.owner_signal_fd)
                    self.assertEqual(job.exit_code, 0)
                    self.assertIn('CloseHandle failed', job.error)
                    record.assert_called_once_with(job)

    async def test_exit_revokes_then_awaits_credential_capability_cleanup(self):
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

        with tempfile.TemporaryDirectory() as root:
            async with managed_jobs(root) as manager:
                capability = Capability()
                job = types.SimpleNamespace(
                    status="running",
                    group_active=False,
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

                self.assertTrue(capability.closed)
                self.assertIsNone(job.credential_capability)

    async def test_subagent_slots_bound_concurrent_children_and_release_on_exit(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                command = [
                    sys.executable,
                    "-c",
                    "import threading; threading.Event().wait()",
                ]
                first = await manager.run_background_exec(
                    command, cwd=tmpdir, session_owned=True, subagent=True)
                second = await manager.run_background_exec(
                    command, cwd=tmpdir, session_owned=True, subagent=True)
                third = None
                try:
                    with self.assertRaises(job_runtime.SubagentCapacityError):
                        await manager.run_background_exec(
                            command,
                            cwd=tmpdir,
                            session_owned=True,
                            subagent=True,
                        )

                    host_process.signal_group(
                        first.process, first.pgid, signal.SIGTERM)
                    await first.process.wait()
                    manager._refresh_job(first)
                    third = await manager.run_background_exec(
                        command,
                        cwd=tmpdir,
                        session_owned=True,
                        subagent=True,
                    )
                finally:
                    for job in [first, second, third]:
                        if (job is not None
                                and job.process.returncode is None):
                            host_process.signal_group(
                                job.process, job.pgid, host_process.FORCE)
                            await job.process.wait()
                            manager._refresh_job(job)

                self.assertFalse(first.subagent_slot)
                self.assertFalse(second.subagent_slot)
                self.assertFalse(third.subagent_slot)
                self.assertEqual(manager._active_subagents, 0)

    async def test_failed_subagent_spawn_releases_its_slot(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                with mock.patch.object(
                        job_runtime.host_process,
                        "create_exec",
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

                self.assertEqual(manager._active_subagents, 0)

    async def test_force_stop_escalates_a_job_already_stopping(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                script = (
                    "import signal,threading\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    # A non-forced stop is a signal on POSIX and CTRL_BREAK_EVENT
                    # on Windows, so ignore both to make the first stop genuinely
                    # ineffective on either platform.
                    "if hasattr(signal, 'SIGBREAK'):\n"
                    "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
                    "print('ready', flush=True)\n"
                    "threading.Event().wait()\n"
                )
                job = await manager.run_background_exec(
                    [sys.executable, "-c", script], cwd=tmpdir)
                try:
                    await self._output_until(manager, job, 'ready')
                    first = manager.stop_job(job.id)
                    await asyncio.sleep(0.05)
                    self.assertIsNone(job.process.returncode)
                    second = manager.stop_job(job.id, force=True)
                    await job.process.wait()
                    while job.status == "stopping":
                        await asyncio.sleep(0.01)
                    with open(
                            job.metadata_path, encoding="utf-8") as metadata_file:
                        metadata = json.load(metadata_file)
                finally:
                    if job.process.returncode is None:
                        host_process.signal_group(
                            job.process, job.pgid, host_process.FORCE)
                        await job.process.wait()

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

    async def test_stopped_foreground_observes_cancellation(self):
        await self._stopped_foreground_journey("cancel")

    async def test_stopped_foreground_observes_requested_expiry(self):
        await self._stopped_foreground_journey("timeout")

    async def test_stopped_foreground_unwind_joins_retirement(self):
        await self._stopped_foreground_journey("unwind")

    async def test_stopped_foreground_observes_cancellation_during_spawn(self):
        await self._stopped_foreground_journey("spawn_cancel")

    async def _stopped_foreground_journey(self, cause):
        # Each failure path is independently selectable; the shared journey
        # retains the same real child, metadata and cleanup assertions.
        from process_lifecycle_fixtures import ProcessResources

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                cancel_event = asyncio.Event()
                release_spawn = asyncio.Event()
                tasks_before = asyncio.all_tasks()
                script = (
                    "import signal,threading\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                    "if hasattr(signal, 'SIGBREAK'):\n"
                    "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
                    "print('ready', flush=True)\n"
                    "threading.Event().wait()\n"
                )
                real_spawn = manager._spawn

                async def held_spawn(*args, **kwargs):
                    job = await real_spawn(*args, **kwargs)
                    await release_spawn.wait()
                    return job

                waiting = asyncio.Event()
                real_wait = manager._wait_for_job

                async def observed_wait(job):
                    waiting.set()
                    return await real_wait(job)

                # The real execution budget starts after _spawn returns. Hold that
                # existing seam until signal handlers/readiness and /ps stop are
                # established; never require startup output before an elapsed limit.
                # Cancellation-only cases have no execution deadline at all.
                with mock.patch.object(manager, "_spawn", side_effect=held_spawn), \
                        mock.patch.object(manager, "_wait_for_job", side_effect=observed_wait):
                    task = asyncio.create_task(manager.run_foreground(
                        [sys.executable, "-c", script], "uncooperative command",
                        5_000 if cause == "timeout" else None,
                        cwd=tmpdir, cancel_event=cancel_event))
                    try:
                        while True:
                            job = next(iter(manager.jobs.values()), None)
                            if job is not None and job.process is not None:
                                break
                            self.assertFalse(task.done(), 'foreground launch ended before publication')
                            await asyncio.sleep(.01)
                        await self._output_until(manager, job, 'ready')
                        resources = ProcessResources(job.process)
                        with mock.patch.object(
                                loki.current_session(), "job_manager", manager):
                            self.assertIn("Sent", loki.run_ps("stop " + job.id))
                        self.assertEqual(job.status, "stopping")
                        self.assertIsNone(job.process.returncode)
                        if cause == "spawn_cancel":
                            cancel_event.set()
                        release_spawn.set()
                        if cause in ["cancel", "unwind"]:
                            await observe_fixture(waiting.wait(), task)
                            if cause == "cancel":
                                cancel_event.set()
                            else:
                                task.cancel()
                        expected = "timed_out" if cause == "timeout" else "cancelled"
                        if cause == "unwind":
                            with self.assertRaises(asyncio.CancelledError):
                                await task
                        else:
                            result_job, outcome, stdout, stderr = await task
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
                                await job.process.wait()
                        release_spawn.set()
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

    async def test_foreground_completion_and_simultaneous_cancellation(self):

        for simultaneous_cancel in [False, True]:
            with self.subTest(cancel=simultaneous_cancel), tempfile.TemporaryDirectory() as tmpdir:
                async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
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
                        job, outcome, stdout, stderr = await manager.run_foreground(
                            [sys.executable, "-c", "print('finished')"],
                            "finite command", 5_000, cwd=tmpdir,
                            cancel_event=cancel_event)
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

    async def test_logical_session_close_refuses_launches_but_keeps_cleanup_storage(self):
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as root:
            async with managed_jobs(root) as manager:
                session = Session(job_manager=manager)
                try:
                    session.begin_close()
                    self.assertFalse(session.active)
                    self.assertIsNotNone(session.response_headers.directory_handle)
                    self.assertIsNotNone(manager._session_directory())
                    with mock.patch.object(manager, '_create_exec', side_effect=AssertionError('new launch')):
                        with self.assertRaisesRegex(RuntimeError, 'conversation has closed'):
                            await manager.run_background_exec([sys.executable, '-c', 'pass'])
                    self.assertEqual(manager.jobs, {})
                finally:
                    await session.close_state()
                with self.assertRaisesRegex(ValueError, 'closed'):
                    manager._session_directory()
                with self.assertRaisesRegex(ValueError, 'closed'):
                    _ = session.response_headers.directory_handle

    async def test_session_retirement_reaps_all_owned_jobs_despite_metadata_failures(self):
        from loki_agent.sessions import Session
        from process_lifecycle_fixtures import ProcessResources

        with tempfile.TemporaryDirectory() as root:
            async with managed_jobs(root) as manager:
                session = Session(job_manager=manager)
                resources = []
                try:
                    for _ in range(2):
                        job = await manager.run_background_exec(
                            [sys.executable, '-c',
                             'import threading; print("ready", flush=True); threading.Event().wait()'],
                            session_owned=True)
                        resources.append(ProcessResources(job.process))
                        await self._output_until(manager, job, 'ready')
                    record = manager._write_metadata

                    def failing_record(job):
                        if job.status == 'owner_closing':
                            raise OSError('metadata unavailable during retirement')
                        return record(job)

                    with mock.patch.object(manager, '_write_metadata', side_effect=failing_record):
                        with self.assertRaises(ExceptionGroup) as failed:
                            await session.close_state()
                    self.assertEqual(len(failed.exception.exceptions), 2)
                    for resource in resources:
                        resource.assert_released(self)
                    self.assertFalse(manager._monitors)
                    self.assertIsNone(manager._session_fd)
                    with self.assertRaisesRegex(ValueError, 'closed'):
                        _ = session.response_headers.directory_handle
                finally:
                    for resource in resources:
                        await resource.cleanup()
                    await session.close_state()

    async def test_session_final_save_survives_repeated_cancellation(self):
        from loki_agent import response_headers
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as root:
            async with managed_jobs(root) as manager:
                session = Session(job_manager=manager)
                session.response_headers.observer('https://final-save.test/v1', None, 'model')(
                    200, {'X-Retirement': 'saved'})
                entered, release = asyncio.Event(), asyncio.Event()
                original = session.response_headers.save_on_exit

                async def save():
                    entered.set()
                    await release.wait()
                    await original()

                with mock.patch.object(session.response_headers, 'save_on_exit', side_effect=save):
                    task = asyncio.create_task(session.close_state())
                    try:
                        await observe_fixture(entered.wait(), task)
                        self.assertIsNone(manager._session_fd)
                        self.assertIsNotNone(session.response_headers.directory_handle)
                        for _ in range(2):
                            task.cancel()
                            await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        release.set()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        with self.assertRaisesRegex(ValueError, 'closed'):
                            _ = session.response_headers.directory_handle
                        readback = response_headers.Store()
                        try:
                            entry = readback.snapshot('https://final-save.test/v1')['endpoints'][0]
                            self.assertEqual(entry['headers']['x-retirement']['value'], 'saved')
                        finally:
                            readback.close()
                    finally:
                        release.set()
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                await session.close_state()

    async def test_final_retirement_observes_a_completed_monitor_failure(self):
        from process_lifecycle_fixtures import ProcessResources

        with tempfile.TemporaryDirectory() as root:
            async with managed_jobs(root) as manager:
                async def failed_monitor(job):
                    await job.process.wait()
                    raise OSError('monitor retirement failed')

                with mock.patch.object(manager, '_monitor_background_job', side_effect=failed_monitor):
                    job = await manager.run_background_exec([sys.executable, '-c', 'pass'])
                    resources = ProcessResources(job.process)
                    try:
                        monitor = next(iter(manager._monitors))
                        with self.assertRaisesRegex(OSError, 'monitor retirement failed'):
                            await monitor
                        with self.assertRaisesRegex(OSError, 'monitor retirement failed'):
                            await manager.aclose()
                        resources.assert_released(self)
                        self.assertFalse(manager._monitors)
                        self.assertIsNone(manager._session_fd)
                    finally:
                        await resources.cleanup()

    async def test_launch_role_comes_from_trusted_subagent_kind_not_argv_or_lifetime(self):

        with tempfile.TemporaryDirectory() as root:
            async with managed_jobs(os.path.join(root, 'jobs')) as manager:
                calls = []

                class Child:
                    pid = 999999
                    returncode = 0

                    async def wait(self):
                        return 0

                    async def aclose(self):
                        # Invocation-only seam: no native resources acquired.
                        pass

                async def create(*command, **kwargs):
                    kwargs['before_resume']()
                    calls.append([list(command), kwargs['owning'], kwargs['cwd']])
                    return Child()

                with mock.patch.object(job_runtime.host_process, 'create_exec', side_effect=create), \
                        mock.patch.object(job_runtime.host_process, 'process_group', return_value=999999), \
                        mock.patch.object(job_runtime.host_process, 'group_exists', return_value=False):
                    await manager.run_background_exec(['selected-loki', '--subagent'], cwd=os.getcwd(),
                                                      session_owned=True, subagent=True)
                    await manager.run_background_exec(['selected-tool', '--subagent'], cwd=root)
                    await manager.run_background_exec(['selected-tool'], cwd=root, session_owned=True)
                    with self.assertRaises(ValueError):
                        await manager.run_background_exec(['selected-tool'], credential_refs=frozenset(),
                                                          credential_authority=object())
                    await manager.aclose()
                self.assertEqual([call[1] for call in calls], [True, False, False])
                self.assertEqual([call[2] for call in calls], [os.getcwd(), root, root])

    async def test_failed_credential_relay_setup_closes_its_ends(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                authority = authentications.CredentialBroker()
                ends = []
                real_channel = job_runtime.host_ipc.owner_channel

                def recording_channel():
                    pair = real_channel()
                    ends.extend(pair)
                    return pair

                with mock.patch.object(
                        job_runtime.host_ipc, "owner_channel",
                        side_effect=recording_channel), \
                        mock.patch.object(
                            job_runtime.credential_capabilities.
                            CredentialCapabilityServer,
                            "create",
                            new=mock.AsyncMock(
                                side_effect=RuntimeError("relay failed"))):
                    with self.assertRaisesRegex(RuntimeError, "relay failed"):
                        await manager.run_background_exec(
                            [sys.executable, "-c", "pass"],
                            cwd=tmpdir,
                            session_owned=True,
                            subagent=True,
                            credential_refs=frozenset(),
                            credential_authority=authority,
                        )

                self.assertEqual(len(ends), 2)
                for end in ends:
                    if isinstance(end, int):
                        # A POSIX descriptor; nothing here observes its closure.
                        continue
                    if job_runtime.host_ipc.is_endpoint(end):
                        # A closed Windows pipe endpoint has released its handles.
                        self.assertEqual(end.handles(), [])
                    else:
                        # A closed socket has no handle.
                        self.assertEqual(end.fileno(), -1)

    def test_session_owned_job_does_not_hold_the_end_the_runtime_keeps(self):
        """Owner EOF reaches the job, including while its ordinary command lives.

        The nested arm's independent socket reply comes AFTER the job reports
        owner EOF. A missing actor/startup error is not proof of descriptor isolation.
        """
        child_script = (
            'import asyncio,sys\n'
            'async def main():\n'
            '    reader,writer=await asyncio.open_connection("127.0.0.1",int(sys.argv[1]))\n'
            '    print("command-ready",flush=True)\n'
            '    while data:=await reader.read(1):\n'
            '        writer.write(data);await writer.drain()\n'
            '    writer.close();await writer.wait_closed()\n'
            'asyncio.run(main())\n')
        script = (
            'import asyncio,sys,tempfile\n'
            'from contextlib import AsyncExitStack\n'
            'from loki_agent import host_ipc\n'
            'sys.path.insert(0,"tests")\n'
            'from job_fixtures import managed_jobs\n'
            'async def main():\n'
            '    end=host_ipc.child_endpoint(sys.argv[sys.argv.index("--session-owner-fd")+1])\n'
            '    async with AsyncExitStack() as cleanup:\n'
            '        command=None\n'
            '        if int(sys.argv[1]):\n'
            '            directory=cleanup.enter_context(tempfile.TemporaryDirectory())\n'
            '            manager=await cleanup.enter_async_context(managed_jobs(directory))\n'
            '            command=await manager.run_background_exec([sys.executable,"-c",sys.argv[2],sys.argv[1]])\n'
            '            while "command-ready" not in manager._spool_tail(command,"stdout.log"):\n'
            '                if command.process.returncode is not None:\n'
            '                    raise RuntimeError(manager._spool_tail(command,"stderr.log"))\n'
            '                await asyncio.sleep(.01)\n'
            '        print("job-ready",flush=True)\n'
            '        await host_ipc.watch_closed(end)\n'
            '        if command is not None and command.process.returncode is not None:\n'
            '            raise RuntimeError("ordinary command exited before owner EOF")\n'
            '        print("closed",flush=True)\n'
            '        if command is not None: await command.process.wait()\n'
            'asyncio.run(main())\n')

        async def scenario(tmpdir, nested):
            accepted = asyncio.get_running_loop().create_future()
            closing = False

            def accept(reader, writer):
                if closing or accepted.done():
                    writer.close()
                    return
                accepted.set_result((reader, writer))

            async with await asyncio.start_server(accept, '127.0.0.1', 0) as server, \
                    managed_jobs(os.path.join(tmpdir, 'jobs')) as manager:
                port = server.sockets[0].getsockname()[1] if nested else 0
                job = await manager.run_background_exec(
                    [sys.executable, '-c', script, str(port), child_script],
                    cwd='.', session_owned=True)
                try:
                    await self._output_until(manager, job, 'job-ready')
                    manager._close_owner_signal(job)
                    self.assertIn('closed', await self._output_until(manager, job, 'closed'))
                    if nested:
                        reader, writer = await accepted
                        writer.write(b'p')
                        await writer.drain()
                        self.assertEqual(await reader.readexactly(1), b'p')
                        writer.close()
                        await writer.wait_closed()
                    self.assertEqual(await job.process.wait(), 0,
                                     manager._spool_tail(job, 'stderr.log'))
                finally:
                    closing = True
                    if accepted.done():
                        _reader, writer = accepted.result()
                        writer.close()
                        await writer.wait_closed()

        for nested in [False, True]:
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as tmpdir:
                asyncio.run(scenario(tmpdir, nested))

    async def test_cancelling_owner_task_reaps_foreground_process(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                task = asyncio.create_task(manager.run_exec(
                    [sys.executable, "-c", "import threading; threading.Event().wait()"],
                    cwd=tmpdir))
                # Registration precedes launch. Observe a live process so this
                # tests cancellation of running work, not cancellation before spawn.
                while not any(job.process is not None
                              for job in manager.jobs.values()):
                    self.assertFalse(task.done(), 'job ended before process publication')
                    await asyncio.sleep(0.01)
                job = next(job for job in manager.jobs.values()
                           if job.process is not None)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                with open(job.metadata_path, encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)

                self.assertEqual(job.status, "cancelled")
                self.assertIsNotNone(job.process.returncode)
                self.assertEqual(metadata["status"], "cancelled")
                self.assertEqual(metadata["exit_code"], job.process.returncode)

    async def test_worker_owned_and_ordinary_job_lifecycle(self):
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session
        from process_lifecycle_fixtures import ProcessResources

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                session = Session(shell_cwd=tmpdir)
                self.addCleanup(session.response_headers.close)
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
                             "import os,threading; print('ordinary-ready:' + os.getcwd(), "
                             "flush=True); threading.Event().wait()"], cwd=ordinary_cwd)
                        resources.append(ProcessResources(ordinary.process))
                        self.assertEqual(owned.cwd, owned_cwd)
                        self.assertEqual(ordinary.cwd, ordinary_cwd)
                        self.assertNotEqual(owned.id, ordinary.id)
                        for job, marker in [[owned, 'owned-ready:'],
                                            [ordinary, 'ordinary-ready:']]:
                            await self._output_until(manager, job, marker + job.cwd)
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
                        await worker.close()
                        self.assertFalse(session.active)
                        self.assertIsNotNone(session.response_headers.directory_handle)
                        await session.close_state()
                        self.assertEqual(owned.process.returncode, 0)
                        self.assertEqual(owned.exit_code, 0)
                        self.assertEqual(owned.status, 'owner_closed')
                        with open(owned.stdout_path, encoding='utf-8') as stream:
                            self.assertIn('owner-eof', stream.read())
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
                        await worker.close()
                        self.assertIsNone(ordinary.process.returncode)
                        self.assertEqual(ordinary.status, 'running')
                        self.assertFalse(ordinary.session_owned)
                        with open(ordinary.metadata_path, encoding='utf-8') as stream:
                            self.assertEqual(json.load(stream)['status'], 'running')
                        host_process.signal_group(ordinary.process, ordinary.pgid,
                                                  host_process.FORCE)
                        await ordinary.process.wait()
                        await asyncio.sleep(0)
                        await asyncio.sleep(0)
                        resources[1].assert_released(self)
                        self.assertFalse(asyncio.all_tasks() - tasks_before)
                finally:
                    for resource in resources:
                        await resource.cleanup()

    async def test_bash_timeout_is_a_failed_tool_result(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                with mock.patch.object(loki.current_session(), 'job_manager', manager):
                    # This checks the tool's real process-timeout result, not Bash
                    # parsing. Windows uses a native -c actor for this unit contract;
                    # actual Bash integration remains a separate qualification.
                    actor = (mock.patch.object(job_runtime.executables, 'BASH', sys.executable)
                             if os.name == 'nt' else contextlib.nullcontext())
                    script = "import threading; threading.Event().wait()"
                    command = (script if os.name == 'nt' else
                               f"exec {shlex.quote(sys.executable)} -c {shlex.quote(script)}")
                    with actor:
                        result = await loki.dispatch_tool_async(
                            "Bash",
                            {
                                "command": command,
                                "timeout": 10,
                                "description": "timeout contract",
                            },
                        )

                self.assertFalse(result["ok"])
                self.assertIn("timed_out", result["content"])

    def test_a_signal_failure_is_not_reported_as_a_gone_process(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        manager = job_manager(root, cleanup=self.addCleanup)
        job = types.SimpleNamespace(
            process=types.SimpleNamespace(returncode=None, pid=1), pgid=1,
            group_active=True)

        with mock.patch.object(
                job_runtime.host_process, "signal_group",
                side_effect=PermissionError(errno.EPERM, "denied")):
            with self.assertRaises(PermissionError):
                manager._signal_process_group(job, signal.SIGTERM)

        with mock.patch.object(
                job_runtime.host_process, "signal_group",
                side_effect=ProcessLookupError(errno.ESRCH, "gone")):
            self.assertFalse(manager._signal_process_group(job, signal.SIGTERM))

    async def test_a_post_spawn_failure_reaps_the_child_and_frees_its_slot(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                with mock.patch.object(manager, "_write_metadata",
                                       side_effect=OSError("metadata unavailable")):
                    with self.assertRaisesRegex(OSError, "metadata unavailable"):
                        await manager.run_exec(
                            [sys.executable, "-c",
                             "import threading; threading.Event().wait()"],
                            cwd=tmpdir, session_owned=True, subagent=True)

                job = next(iter(manager.jobs.values()))
                self.assertEqual(job.status, "failed")
                self.assertIsNotNone(job.process.returncode)
                self.assertFalse(job.subagent_slot)
                self.assertEqual(manager._active_subagents, 0)

    async def test_recording_a_failure_does_not_replace_it(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                with mock.patch.object(
                        job_runtime.host_process, "create_exec",
                        side_effect=FileNotFoundError("no such program")), \
                        mock.patch.object(
                            manager, "_write_metadata",
                            side_effect=PermissionError("cannot record")):
                    with self.assertRaises(FileNotFoundError):
                        await manager.run_exec(
                            ["definitely-not-a-program"], cwd=tmpdir)

    async def test_session_close_during_launch_does_not_publish_a_live_child(self):
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session
        from process_lifecycle_fixtures import ProcessResources

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                session = Session(shell_cwd=tmpdir)
                self.addCleanup(session.response_headers.close)
                session.job_manager = manager
                worker = Worker(session, lambda message: None, "session")
                entered, release = asyncio.Event(), asyncio.Event()
                real = job_runtime.host_process.create_exec
                resources = []
                tasks_before = asyncio.all_tasks()

                async def delayed(*args, **kwargs):
                    process = await real(*args, **kwargs)
                    resources.append(ProcessResources(process))
                    # Hold the successful launch before JobManager can publish it.
                    # Native pre-resume revocation is covered at the lower seam too.
                    entered.set()
                    await release.wait()
                    return process

                task = None
                try:
                    with mock.patch.object(job_runtime.host_process, "create_exec", new=delayed):
                        task = asyncio.create_task(manager.run_background_exec(
                            [sys.executable, "-c", "import threading; threading.Event().wait()"],
                            cwd=tmpdir, session_owned=True, subagent=True))
                        await observe_fixture(entered.wait(), task)
                        self.assertEqual(manager._active_subagents, 1)
                        await worker.close()
                        release.set()
                        with self.assertRaises(job_runtime._JobRevokedDuringLaunch) as raised:
                            await task
                        self.assertIs(type(raised.exception), job_runtime._JobRevokedDuringLaunch)
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
                    await worker.close()
                    self.assertFalse(asyncio.all_tasks() - tasks_before)
                finally:
                    release.set()
                    if task is not None and not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    for resource in resources:
                        await resource.cleanup()

    async def test_a_failed_spawn_is_left_as_a_recorded_failed_job(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                with self.assertRaises(OSError):
                    await manager.run_exec(
                        [os.path.join(tmpdir, "not-a-program")], 5_000,
                        cwd=tmpdir)
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

    async def test_a_refused_subagent_is_recorded_and_frees_its_reservation(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            async with managed_jobs(os.path.join(tmpdir, "jobs")) as manager:
                manager._active_subagents = tool_contracts.MAX_ACTIVE_DIRECT_SUBAGENTS
                with self.assertRaises(job_runtime.SubagentCapacityError):
                    await manager.run_exec(
                        [os.path.join(tmpdir, "not-a-program")], 5_000,
                        cwd=tmpdir, session_owned=True, subagent=True)

                self.assertEqual(manager._active_subagents,
                                 tool_contracts.MAX_ACTIVE_DIRECT_SUBAGENTS)
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
            self.isolation = self.enterContext(mock.patch.object(
                runtime_isolations, "isolate_credential_directory"))
        else:
            from loki_agent import windows_runtime
            for name, value in (("configured_workspace", "/workspace"),
                                ("verify_runtime", None),
                                ("ensure_runtime_temp", None)):
                self.windows_steps[name] = self.enterContext(mock.patch.object(
                    windows_runtime, name, return_value=value))

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
                    return_value=(11, 12, 13 if os.name == 'posix' else None, ["--headless"])) as descriptors, \
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
            self.isolation.assert_called_once_with(13)
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
                        loki, "async_chat_completion",
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
