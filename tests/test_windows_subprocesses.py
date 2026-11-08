"""Portable worker lifecycle tests with native handles and pipe I/O substituted."""

import asyncio
import contextlib
import ctypes
import sys
import unittest
from unittest import mock

from loki_agent import windows_subprocesses as workers
from process_lifecycle_fixtures import observe_fixture


@unittest.skipUnless(sys.platform == 'win32', 'Windows named-pipe allocator ABI')
class NamedPipeCreationTests(unittest.TestCase):
    def exercise(self, contained, fail=False):
        names, freed, closed = [], [], []

        def convert(text, revision, output, length):
            self.assertEqual(text, 'explicit-pipe-policy')
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 99
            return True

        def create(name, mode, flags, *args):
            names.append(name)
            self.assertTrue(mode & workers.api.FileFlags.FILE_FLAG_FIRST_PIPE_INSTANCE)
            self.assertTrue(flags & workers.host_ipc.PIPE_REJECT_REMOTE_CLIENTS)
            return workers.host_ipc.INVALID_HANDLE_VALUE if fail else 41

        calls = {'ConvertStringSecurityDescriptorToSecurityDescriptorW': convert,
                 'LocalFree': lambda pointer: freed.append(pointer.value),
                 'CreateNamedPipeW': create, 'ConnectNamedPipe': lambda *args: False}
        with mock.patch.object(workers.host_ipc, 'pipe_policy', return_value={
                'descriptor': 'explicit-pipe-policy', 'package': 'package' if contained else None}), \
                mock.patch.object(workers.api, 'bind', side_effect=lambda lib, symbol, *args: calls[symbol]), \
                mock.patch.object(workers.api, 'open_with_access', return_value=42) as client, \
                mock.patch.object(workers.api, 'close_handle', side_effect=closed.append), \
                mock.patch.object(ctypes, 'get_last_error', return_value=87 if fail else 535, create=True):
            if fail:
                with self.assertRaises(workers.api.WindowsApiError) as raised:
                    workers.host_ipc.overlapped_pipe()
                self.assertEqual(raised.exception.status, 87)
                client.assert_not_called()
            else:
                self.assertEqual(list(workers.host_ipc.overlapped_pipe()), [41, 42])
        self.assertEqual(freed, [99])
        self.assertEqual(closed, [])
        self.assertEqual(len(names), 1)
        expected = r'\\.\pipe\LOCAL\loki-worker-' if contained else r'\\.\pipe\loki-worker-'
        self.assertTrue(names[0].startswith(expected))

    def test_contained_and_front_pipes_use_their_own_namespace(self):
        for contained in [True, False]:
            with self.subTest(contained=contained):
                self.exercise(contained)

    def test_unsigned_native_failure_is_never_a_server_handle(self):
        self.exercise(True, fail=True)


class WorkerCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def native_release_failure(self, handle, *, failed_publication=False):
        loop = asyncio.get_running_loop()
        failure = workers.api.WindowsResourceError('CloseHandle failed', status=6)
        closed, acquired = [], []
        exited = failed_publication

        def prepare(native, *args, **kwargs):
            native.information = workers.api.ProcessInformation(71, 72, 42, 0)
            native.pid, native.job = 42, 44
            acquired.append(native)

        def close(value):
            closed.append(value)
            if value == handle:
                raise failure

        def ready():
            if failed_publication:
                raise RuntimeError('publication refused')

        with mock.patch.object(workers.windows_runtime, 'prepare_local', side_effect=prepare), \
                mock.patch.object(workers.windows_runtime, 'duplicate_standard_handle', side_effect=[11, 12, 13]), \
                mock.patch.object(workers.api, 'close_handle', side_effect=close), \
                mock.patch.object(workers.api, 'resume_thread'), \
                mock.patch.object(workers.api, 'wait_for_single_object', side_effect=lambda *args: (
                    workers.api.WAIT_OBJECT_0 if exited else workers.api.WAIT_TIMEOUT)), \
                mock.patch.object(workers.api, 'get_exit_code_process', return_value=0), \
                mock.patch.object(loop, 'call_exception_handler') as reported:
            try:
                if failed_publication:
                    with self.assertRaisesRegex(OSError, 'CloseHandle failed'):
                        await workers.create_local_process('chosen-image', before_resume=ready)
                else:
                    process = await workers.create_local_process('chosen-image')
                    exited = True
                    self.assertEqual(await process.wait(), 0)
                    with self.assertRaises(workers.api.WindowsResourceError) as caught:
                        await process.aclose()
                    self.assertIs(caught.exception, failure)
                    process.close()
                await asyncio.sleep(0)
                reported.assert_not_called()
                self.assertEqual(closed.count(71), 1)
                self.assertEqual(closed.count(72), 1)
                self.assertIsNone(acquired[0].information.hProcess)
                self.assertIsNone(acquired[0].information.hThread)
            finally:
                # Only substituted native records remain; no production retry.
                for native in acquired:
                    native.close()

    async def test_native_process_release_failure_reaches_the_process_owner(self):
        await self.native_release_failure(71)

    async def test_native_thread_release_failure_reaches_the_process_owner(self):
        await self.native_release_failure(72)

    async def test_failed_publication_observes_already_completed_native_release_failure(self):
        await self.native_release_failure(71, failed_publication=True)

    async def test_loop_reported_pipe_close_failure_reaches_the_process_owner(self):
        loop = asyncio.get_running_loop()
        failure = workers.api.WindowsResourceError('CloseHandle failed', status=6)
        raw = workers.host_ipc.PipeHandle(101)
        protocol = workers.streams.ProcessProtocol()
        protocol.exited.set()
        exited = loop.create_future()
        exited.set_result(None)
        transport = mock.Mock(_exit_task=exited)
        transport.get_returncode.return_value = 0
        transport.get_pipe_transport.return_value = None
        transport._proc = mock.Mock(stdin=None, stdout=raw, stderr=None)

        def lost():
            # CPython's proactor notifies its protocol before raw.close; the
            # latter's exception is delivered only to the loop error handler.
            protocol.connection_lost(None)
            raw.close()

        transport.close.side_effect = lambda: loop.call_soon(lost)
        process = workers.Process(transport, protocol)
        with mock.patch.object(loop, 'call_exception_handler') as reported, \
                mock.patch.object(workers.api, 'close_handle', side_effect=failure) as close:
            with self.assertRaises(workers.api.WindowsResourceError) as caught:
                await process.aclose()
        self.assertIs(caught.exception, failure)
        self.assertIs(reported.call_args.args[0]['exception'], failure)
        close.assert_called_once_with(101)
        self.assertIsNone(raw.handle)

    async def test_observed_exit_does_not_hide_native_observer_failure(self):
        failure = workers.api.WindowsApiError('CloseHandle failed', status=6)

        async def observe():
            raise failure

        observer = asyncio.create_task(observe())
        protocol = workers.streams.ProcessProtocol()
        protocol.exited.set()
        transport = mock.Mock(_exit_task=observer)
        transport.get_returncode.return_value = 0
        process = workers.Process(transport, protocol)
        try:
            with self.assertRaises(workers.api.WindowsApiError) as caught:
                await process.wait()
            self.assertIs(caught.exception, failure)
        finally:
            protocol.connection_lost(None)
            await asyncio.gather(observer, return_exceptions=True)

    async def exercise(self, failure=None, cancel=False, close_early=False, local=False, owning=True, revoke=False,
                       termination_error=False, large_output=False):
        loop = asyncio.get_running_loop()
        exited = asyncio.Event()
        attaching = asyncio.Event()
        stopping = asyncio.Event()
        observed = False
        released = []
        closed_handles = []
        transports = []
        native = mock.Mock(pid=42, returncode=None, job=44, stdin=None, stdout=None, stderr=None,
                           information=workers.api.ProcessInformation(71, 72, 42, 0))

        async def wait():
            nonlocal observed
            await exited.wait()
            observed = True
            native.returncode = 1
            return 1

        def terminate():
            # A native termination request is not synchronous exit observation.
            stopping.set()
            if termination_error:
                raise workers.api.WindowsApiError('termination failed', status=5)
            loop.call_soon(exited.set)

        def release_leader():
            self.assertTrue(observed, 'closed native handles under a live wait')
            if native.information.hProcess or native.information.hThread:
                released.append(True)
                native.information.hProcess = native.information.hThread = None

        def close_family():
            if native.job is not None:
                closed_handles.append(native.job)
                native.job = None

        def release():
            close_family()
            release_leader()

        native.wait = wait
        native.terminate = native.kill = terminate
        native.close = release
        native.release_leader = release_leader
        native.close_family = close_family

        async def connect(number, factory, raw):
            if failure == number:
                attaching.set()
                if cancel:
                    await asyncio.Future()
                raise OSError('attachment failed')
            protocol = factory()

            class Pipe:
                closed = False

                def close(self):
                    if not self.closed:
                        self.closed = True
                        raw.close()
                        loop.call_soon(protocol.connection_lost, None)

                def is_closing(self):
                    return self.closed

                def pause_reading(self):
                    pass

                def resume_reading(self):
                    pass

                def get_extra_info(self, name, default=None):
                    return default

            pipe = Pipe()
            protocol.connection_made(pipe)
            return pipe, protocol

        async def connect_write(factory, raw):
            return await connect(1, factory, raw)

        read_number = 1

        async def connect_read(factory, raw):
            nonlocal read_number
            read_number += 1
            return await connect(read_number, factory, raw)

        transport_class = workers.LocalProcessTransport if local else workers.ContainedWorkerTransport

        def transport(*args, **kwargs):
            result = transport_class(*args, **kwargs)
            transports.append(result)
            if cancel and failure == 0:
                # Cancel before attachment gets its first coroutine step.
                asyncio.current_task().cancel()
            return result

        with (
            mock.patch.object(workers.host_ipc, 'overlapped_pipe', side_effect=[(11, 12), (13, 14), (16, 17)], create=True),
            mock.patch.object(workers.api, 'set_handle_information'),
            mock.patch.object(workers.api, 'close_handle',
                              side_effect=closed_handles.append),
            mock.patch.object(workers.windows_runtime, 'worker_stdout_null',
                              return_value=contextlib.nullcontext(15)),
            mock.patch.object(workers.windows_runtime, 'prepare_contained'),
            mock.patch.object(workers, transport_class.__name__, side_effect=transport),
            mock.patch.object(workers.windows_runtime, 'ContainedProcess', return_value=native),
            mock.patch.object(workers.windows_runtime, 'prepare_local') as prepare,
            mock.patch.object(workers.windows_runtime, 'job_has_processes', return_value=True) as family_exists,
            mock.patch.object(workers.windows_runtime, 'terminate_job') as stop_family,
            mock.patch.object(workers.api, 'resume_thread') as resume,
            mock.patch.object(loop, 'connect_write_pipe',
                              side_effect=connect_write),
            mock.patch.object(loop, 'connect_read_pipe',
                              side_effect=connect_read),
        ):
            def before_resume():
                if revoke:
                    raise RuntimeError('revoked before resume')

            if local:
                operation = workers.create_local_process(
                    '/chosen/image.exe', '--subagent', owning=owning, cwd='/trusted',
                    stdin=workers.PIPE, stdout=workers.PIPE, stderr=workers.PIPE,
                    before_resume=before_resume)
            else:
                operation = workers.create_worker_process(
                    workspace='/workspace', environment={}, arguments=[],
                    inherited_handles=[], current_directory='/installed')
            task = asyncio.create_task(operation)
            if termination_error:
                try:
                    await observe_fixture(stopping.wait(), task)
                    self.assertEqual(native.job, 44, 'lost family control before observed exit')
                    self.assertFalse(transports[0]._exit_task.done())
                    self.assertFalse(task.done())
                    self.assertEqual(native.information.hProcess, 71)
                finally:
                    exited.set()
                    await asyncio.gather(task, transports[0]._exit_task, return_exceptions=True)
            if cancel and failure:
                await attaching.wait()
                task.cancel()
            if failure is not None or revoke:
                with self.assertRaises(RuntimeError if revoke else asyncio.CancelledError if cancel else OSError):
                    await task
            else:
                process = await task
                if large_output:
                    try:
                        line = b'x' * 100000 + b'\n'
                        transports[0]._pipe_data_received(1, line)
                        self.assertEqual(await process.stdout.readline(), line)
                    except BaseException:
                        exited.set()
                        await process.aclose()
                        raise
                if local:
                    transports[0]._pipe_data_received(2, b'actual-stderr')
                    self.assertEqual(await process.stderr.read(13), b'actual-stderr')
                if close_early:
                    waiter = asyncio.create_task(process.wait())
                    cancelled = asyncio.create_task(process.wait())
                    await asyncio.sleep(0)
                    cancelled.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await cancelled
                    self.assertFalse(waiter.done())
                    transports[0].close()
                    self.assertEqual(await waiter, 1)
                else:
                    exited.set()
                    self.assertEqual(await process.wait(), 1)
                    # Root exit alone does not imply pipe EOF.
                    transports[0].close()
                await process.aclose()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            transports[0].close()  # repeated cleanup must not close handles twice
            self.assertEqual(released, [True])
            self.assertIs(transports[0]._proc, native)
            self.assertFalse(native.information.hProcess)
            self.assertFalse(native.information.hThread)
            if failure is None and not revoke:
                self.assertEqual(process.returncode, 1)
                with self.assertRaises(ProcessLookupError):
                    transports[0].send_signal(1)
            if local and failure is None and not revoke:
                self.assertIs(process._family, native)
                self.assertEqual(native.job, 44)
                self.assertTrue(process.family_exists(), 'leader/stream exit lost the surviving family')
                family_exists.assert_called_once_with(44)
                process.terminate()
                stop_family.assert_called_once_with(44)
                process.close_family()
                process.close_family()
            self.assertIsNone(native.job)
            expected = [11, 12, 13, 14, 16, 17] if local else [11, 12, 13, 14]
            self.assertCountEqual(closed_handles, [*expected, 44])
            if local:
                self.assertEqual(prepare.call_args.kwargs['owning'], owning)
                self.assertEqual(prepare.call_args.args[1], ['/chosen/image.exe', '--subagent'])
                self.assertEqual(prepare.call_args.args[3], '/trusted')
                if failure is not None or revoke:
                    resume.assert_not_called()
                else:
                    resume.assert_called_once()
            self.assertTrue(transports[0]._exit_task.done())
            self.assertFalse(transports[0]._exit_task.cancelled())
            self.assertTrue(transports[0]._connect_task.done())

    async def test_first_and_second_pipe_attachment_failure_reap_child(self):
        for stage in [1, 2]:
            with self.subTest(stage=stage):
                await self.exercise(failure=stage)

    async def test_cancelled_launch_joins_attachment_and_reaps_child(self):
        for stage in [0, 1, 2]:
            with self.subTest(stage=stage):
                await self.exercise(failure=stage, cancel=True)

    async def test_close_keeps_exit_observer_and_wakes_existing_waiters(self):
        await self.exercise(close_early=True)

    async def test_normal_exit_releases_handles_once(self):
        await self.exercise()

    async def test_worker_pipe_accepts_frames_larger_than_the_stdio_default(self):
        await self.exercise(large_output=True)

    async def test_local_roles_keep_cwd_and_capture_three_streams(self):
        for owning in [True, False]:
            with self.subTest(owning=owning):
                await self.exercise(local=True, owning=owning)

    async def test_local_attachment_failure_and_cancellation_never_resume(self):
        for stage in [1, 2, 3]:
            for cancel in [True, False]:
                with self.subTest(stage=stage, cancel=cancel):
                    await self.exercise(local=True, failure=stage, cancel=cancel)

    async def test_revocation_after_attachment_reaps_without_resume(self):
        await self.exercise(local=True, revoke=True)

    async def test_failed_termination_retains_family_control_until_actual_exit(self):
        await self.exercise(local=True, failure=1, termination_error=True)
