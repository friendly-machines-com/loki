"""Portable worker lifecycle tests with native handles and pipe I/O substituted."""

import asyncio
import contextlib
import unittest
from unittest import mock

from loki_agent import windows_subprocesses as workers


class WorkerCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, failure=None, cancel=False, close_early=False):
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

        async def connect_read(factory, raw):
            return await connect(2, factory, raw)

        transport_class = workers.ContainedWorkerTransport

        def transport(*args, **kwargs):
            result = transport_class(*args, **kwargs)
            transports.append(result)
            if cancel and failure == 0:
                # Cancel before attachment gets its first coroutine step.
                asyncio.current_task().cancel()
            return result

        with (
            mock.patch.object(workers, 'pipe', side_effect=[(11, 12), (13, 14)]),
            mock.patch.object(workers.api, 'set_handle_information'),
            mock.patch.object(workers.api, 'close_handle',
                              side_effect=closed_handles.append),
            mock.patch.object(workers.windows_runtime, 'worker_stdout_null',
                              return_value=contextlib.nullcontext(15)),
            mock.patch.object(workers.windows_runtime, 'launch', return_value=native),
            mock.patch.object(workers, 'ContainedWorkerTransport', side_effect=transport),
            mock.patch.object(loop, 'connect_write_pipe',
                              side_effect=connect_write),
            mock.patch.object(loop, 'connect_read_pipe',
                              side_effect=connect_read),
        ):
            task = asyncio.create_task(workers.create_worker_process(
                workspace='/workspace', environment={}, arguments=[],
                inherited_handles=[], current_directory='/installed'))
            if cancel and failure:
                await asyncio.wait_for(attaching.wait(), 1)
                task.cancel()
            if failure is not None:
                with self.assertRaises(asyncio.CancelledError if cancel else OSError):
                    await asyncio.wait_for(task, 1)
            else:
                process = await asyncio.wait_for(task, 1)
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
            self.assertCountEqual(closed_handles, [11, 12, 13, 14])
            self.assertTrue(transports[0]._exit_task.done())
            self.assertFalse(transports[0]._exit_task.cancelled())
            self.assertTrue(transports[0]._connect_task.done())

    async def test_first_and_second_pipe_attachment_failure_reap_child(self):
        for stage in (1, 2):
            with self.subTest(stage=stage):
                await self.exercise(failure=stage)

    async def test_cancelled_launch_joins_attachment_and_reaps_child(self):
        for stage in (0, 1, 2):
            with self.subTest(stage=stage):
                await self.exercise(failure=stage, cancel=True)

    async def test_close_keeps_exit_observer_and_wakes_existing_waiters(self):
        await self.exercise(close_early=True)

    async def test_normal_exit_releases_handles_once(self):
        await self.exercise()
