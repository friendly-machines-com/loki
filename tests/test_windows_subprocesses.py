"""Portable worker lifecycle tests with native handles and pipe I/O substituted."""

import asyncio
import contextlib
import ctypes
import unittest
from unittest import mock

from loki_agent import windows_subprocesses as workers


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
            self.assertTrue(flags & workers.PIPE_REJECT_REMOTE_CLIENTS)
            return workers.INVALID_HANDLE_VALUE if fail else 41

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
                    workers.pipe()
                self.assertEqual(raised.exception.status, 87)
                client.assert_not_called()
            else:
                self.assertEqual(list(workers.pipe()), [41, 42])
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
    async def exercise(self, failure=None, cancel=False, close_early=False, local=False, owning=True, revoke=False):
        loop = asyncio.get_running_loop()
        exited = asyncio.Event()
        attaching = asyncio.Event()
        observed = False
        released = []
        closed_handles = []
        transports = []
        native = mock.Mock(pid=42, returncode=None)

        async def wait():
            nonlocal observed
            await exited.wait()
            observed = True
            native.returncode = 1
            return 1

        def terminate():
            # A native termination request is not synchronous exit observation.
            loop.call_soon(exited.set)

        def release():
            self.assertTrue(observed, 'closed native handles under a live wait')
            released.append(True)

        native.wait = wait
        native.terminate = terminate
        native.close = release

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
            mock.patch.object(workers, 'pipe', side_effect=[(11, 12), (13, 14), (16, 17)]),
            mock.patch.object(workers.api, 'set_handle_information'),
            mock.patch.object(workers.api, 'close_handle',
                              side_effect=closed_handles.append),
            mock.patch.object(workers.windows_runtime, 'worker_stdout_null',
                              return_value=contextlib.nullcontext(15)),
            mock.patch.object(workers.windows_runtime, 'launch', return_value=native),
            mock.patch.object(workers, transport_class.__name__, side_effect=transport),
            mock.patch.object(workers.windows_runtime, 'ContainedProcess', return_value=native),
            mock.patch.object(workers.windows_runtime, 'prepare_local') as prepare,
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
            if cancel and failure:
                await asyncio.wait_for(attaching.wait(), 1)
                task.cancel()
            if failure is not None or revoke:
                with self.assertRaises(RuntimeError if revoke else asyncio.CancelledError if cancel else OSError):
                    await asyncio.wait_for(task, 1)
            else:
                process = await asyncio.wait_for(task, 1)
                if local:
                    transports[0]._pipe_data_received(2, b'actual-stderr')
                    self.assertEqual(await process.stderr.read(13), b'actual-stderr')
                if close_early:
                    waiter = asyncio.create_task(process.wait())
                    transports[0].close()
                    self.assertEqual(await asyncio.wait_for(waiter, 1), 1)
                else:
                    exited.set()
                    self.assertEqual(await asyncio.wait_for(process.wait(), 1), 1)
                    # Root exit alone does not imply pipe EOF.
                    transports[0].close()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            transports[0].close()  # repeated cleanup must not close handles twice
            self.assertEqual(released, [True])
            self.assertCountEqual(closed_handles, [11, 12, 13, 14, 16, 17] if local else [11, 12, 13, 14])
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
