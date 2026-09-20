"""Ownership tests plus native public-asyncio pipe integration.

The existing release-entrypoint ACP tests exercise the actual contained worker;
these native transport cases isolate full-pipe/EOF behavior without credentials.
"""

import asyncio
from contextlib import ExitStack
import os
import sys
import subprocess
import types
import unittest
from unittest import mock

from loki_agent import acp, acps, windows_workers as workers
from loki_agent import windows_runtime as runtime, windows_api as api


class PipeCreationTests(unittest.TestCase):
    def plumbing(self, stack):
        native = types.SimpleNamespace(
            PIPE_ACCESS_DUPLEX=3, PIPE_ACCESS_INBOUND=1,
            FILE_FLAG_FIRST_PIPE_INSTANCE=0x80000,
            FILE_FLAG_OVERLAPPED=0x40000000, PIPE_WAIT=0,
            GENERIC_WRITE=0x40000000, GENERIC_READ=0x80000000, OPEN_EXISTING=3,
            CreateNamedPipe=mock.Mock(return_value=101),
            CreateFile=mock.Mock(return_value=102),
            ConnectNamedPipe=mock.Mock())
        stack.enter_context(mock.patch.dict(sys.modules, _winapi=native))
        convert = mock.Mock(return_value=True)
        free = mock.Mock()
        stack.enter_context(mock.patch.object(
            api, 'bind', side_effect=lambda library, symbol, *args:
            free if symbol == 'LocalFree' else convert))
        stack.enter_context(mock.patch.object(api, 'current_user_sid', return_value='USER'))
        inheritance = stack.enter_context(mock.patch.object(api, 'set_handle_information'))
        closed = stack.enter_context(mock.patch.object(api, 'close_handle'))
        return native, convert, free, inheritance, closed

    def test_private_connected_pair_and_only_child_inheritance(self):
        for stdin in (False, True):
            with self.subTest(stdin=stdin), ExitStack() as stack:
                native, convert, free, inherit, closed = self.plumbing(stack)
                front, child = workers._stdio_pair(stdin=stdin)
                self.assertEqual((front, child), (102, 101) if stdin else (101, 102))
                self.assertEqual(convert.call_args.args[0],
                                 'D:P(A;;GA;;;SY)(A;;GA;;;USER)')
                args = native.CreateNamedPipe.call_args.args
                self.assertTrue(args[0].startswith(r'\\.\pipe\loki-worker-'))
                self.assertEqual(args[2] & 8, 8)  # reject remote clients
                self.assertEqual(args[3], 1)
                self.assertTrue(args[1] & native.FILE_FLAG_FIRST_PIPE_INSTANCE)
                self.assertEqual(bool(args[1] & native.FILE_FLAG_OVERLAPPED), not stdin)
                self.assertTrue(args[-1])  # creation-time security attributes
                self.assertEqual(bool(native.CreateFile.call_args.args[5]
                                      & native.FILE_FLAG_OVERLAPPED), stdin)
                inherit.assert_called_once_with(child, api.HANDLE_FLAG_INHERIT,
                                                api.HANDLE_FLAG_INHERIT)
                closed.assert_not_called()
                free.assert_called_once()

    def test_partial_pair_failures_release_only_acquired_handles(self):
        for phase, expected in [('create', []), ('open', [101]),
                                ('connect', [101, 102]), ('inherit', [101, 102])]:
            with self.subTest(phase=phase), ExitStack() as stack:
                native, convert, free, inherit, closed = self.plumbing(stack)
                target = {'create': native.CreateNamedPipe, 'open': native.CreateFile,
                          'connect': native.ConnectNamedPipe, 'inherit': inherit}[phase]
                target.side_effect = OSError('failed')
                with self.assertRaises(OSError):
                    workers._stdio_pair(stdin=True)
                self.assertEqual([c.args[0] for c in closed.call_args_list], expected)
                free.assert_called_once()

    def test_second_pair_failure_releases_first_pair(self):
        with mock.patch.object(workers, '_stdio_pair',
                               side_effect=[(11, 12), OSError('failed')]), \
                mock.patch.object(api, 'close_handle') as close:
            with self.assertRaises(OSError):
                workers._stdio()
        self.assertEqual(close.call_args_list, [mock.call(11), mock.call(12)])


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_backpressure_and_connection_loss_wake_drain(self):
        writer = workers._Writer()
        transport = mock.Mock()
        transport.is_closing.return_value = False
        writer.connection_made(transport)
        writer.write(b'abc')
        writer.pause_writing()
        waiting = asyncio.create_task(writer.drain())
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        writer.resume_writing()
        await waiting
        writer.pause_writing()
        waiting = asyncio.create_task(writer.drain())
        writer.connection_lost(BrokenPipeError('peer closed'))
        with self.assertRaises(BrokenPipeError):
            await waiting
        with self.assertRaises(BrokenPipeError):
            await writer.wait_closed()

    async def test_cancelled_wait_does_not_cancel_connection_lost(self):
        writer = workers._Writer()
        pending = asyncio.create_task(writer.wait_closed())
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertFalse(writer.closed.cancelled())
        writer.connection_lost(None)
        await writer.wait_closed()

    async def test_reader_delivers_final_bytes_before_eof(self):
        reader = workers._Reader()
        reader.data_received(b'last reply\n')
        reader.connection_lost(None)
        self.assertEqual(await reader.stream.readline(), b'last reply\n')
        self.assertEqual(await reader.stream.readline(), b'')

    async def test_partial_attachment_and_cancellation_release_each_end_once(self):
        for phase in ('success', 'first', 'second', 'cancel'):
            with self.subTest(phase=phase):
                loop = asyncio.get_running_loop()
                connecting = asyncio.Event()
                proceed = asyncio.Event()
                count = 0

                async def connect(factory, pipe):
                    nonlocal count
                    count += 1
                    protocol = factory()
                    if (phase == 'first' and count == 1
                            or phase == 'second' and count == 2):
                        raise OSError('attachment failed')
                    if phase == 'cancel' and count == 2:
                        connecting.set()
                        await proceed.wait()
                    transport = mock.Mock()
                    closed = False

                    def close():
                        nonlocal closed
                        if not closed:
                            closed = True
                            pipe.close()
                            protocol.connection_lost(None)
                    transport.close.side_effect = close
                    transport.abort.side_effect = close
                    protocol.connection_made(transport)
                    return transport, protocol

                with mock.patch.object(loop, 'connect_read_pipe', side_effect=connect), \
                        mock.patch.object(loop, 'connect_write_pipe', side_effect=connect), \
                        mock.patch.object(api, 'close_handle') as close:
                    streams = workers._Streams(11, 12)
                    task = asyncio.create_task(streams.connect())
                    if phase == 'cancel':
                        await connecting.wait()
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        proceed.set()
                    elif phase in ('first', 'second'):
                        with self.assertRaises(OSError):
                            await task
                    else:
                        await task
                    await streams.close()
                    await streams.close()
                self.assertCountEqual(close.call_args_list, [mock.call(11), mock.call(12)])


class LaunchOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_gate_refuses_before_any_pipe_or_launch(self):
        with mock.patch.object(runtime, 'required_workspace',
                               side_effect=runtime.RuntimeIsolationError('denied')), \
                mock.patch.object(workers, '_stdio') as pipes, \
                mock.patch.object(runtime, 'launch') as launch:
            with self.assertRaises(runtime.RuntimeIsolationError):
                await workers.start_worker('/workspace', {}, mock.Mock())
        pipes.assert_not_called()
        launch.assert_not_called()

    async def test_launch_handoff_and_failure_cleanup(self):
        for phase in ('success', 'attach', 'launch'):
            with self.subTest(phase=phase), ExitStack() as stack:
                stack.enter_context(mock.patch.object(runtime, 'required_workspace',
                                                      return_value='/registered'))
                stack.enter_context(mock.patch.object(workers, '_stdio',
                                                      return_value=((11, 12), (13, 14))))
                streams = mock.Mock(connect=mock.AsyncMock(), close=mock.AsyncMock())
                stack.enter_context(mock.patch.object(workers, '_Streams', return_value=streams))
                delegation = mock.Mock(owner_child=(21,), credential_child=(22,))
                delegation.child_arguments.return_value = ['--authority', 'test']
                stack.enter_context(mock.patch.object(workers.host_ipc, 'handles', side_effect=tuple))
                null = stack.enter_context(mock.patch.object(runtime, 'worker_stdout_null'))
                null.return_value.__enter__.return_value = 31
                process = mock.Mock(wait=mock.AsyncMock(return_value=0), returncode=0)
                launch = stack.enter_context(mock.patch.object(runtime, 'launch', return_value=process))
                closed = stack.enter_context(mock.patch.object(api, 'close_handle'))
                if phase == 'attach':
                    streams.connect.side_effect = OSError('attachment failed')
                if phase == 'launch':
                    launch.side_effect = OSError('launch failed')
                if phase == 'success':
                    worker = await workers.start_worker('/logical', {'SAFE': 'yes'}, delegation)
                    self.assertEqual(launch.call_args.args[3], '/registered')
                    self.assertEqual(launch.call_args.kwargs['current_directory'], os.getcwd())
                    self.assertEqual(launch.call_args.kwargs['stdio'], (13, 14))
                    self.assertEqual(launch.call_args.args[4], [21, 22, 31])
                    await worker.close()
                    await worker.close()
                    process.close.assert_called_once_with()
                else:
                    with self.assertRaises(OSError):
                        await workers.start_worker('/logical', {}, delegation)
                self.assertCountEqual(closed.call_args_list, [mock.call(13), mock.call(14)])
                streams.close.assert_awaited_once_with()
                if phase == 'attach':
                    launch.assert_not_called()

    async def test_root_exit_closes_job_even_if_stdout_has_not_closed(self):
        exited = asyncio.Event()
        process = mock.Mock(returncode=None)

        async def wait():
            await exited.wait()
            process.returncode = 9
            return 9
        process.wait = wait
        streams = mock.Mock(close=mock.AsyncMock())
        streams.stdout = asyncio.StreamReader()
        worker = workers.Worker(process, streams)
        try:
            exited.set()
            self.assertEqual(await worker.wait(), 9)
            process.close_job.assert_called_once_with()
            self.assertFalse(streams.stdout.at_eof())
        finally:
            await worker.close()
        process.close.assert_called_once_with()

    def test_native_handles_have_idempotent_owners(self):
        information = api.ProcessInformation(11, 12, 13, 14)
        process = runtime.ContainedProcess(information, 15)
        with mock.patch.object(api, 'close_handle') as close:
            process.close_job()
            process.close()
            process.close()
        self.assertCountEqual(close.call_args_list,
                              [mock.call(11), mock.call(12), mock.call(15)])


class ChannelFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_graceful_close_delivers_final_buffered_notification(self):
        process = mock.Mock(returncode=0, stdout=asyncio.StreamReader(),
                            wait=mock.AsyncMock(return_value=0))
        process.stdin.wait_closed = mock.AsyncMock()
        received = []
        with mock.patch.object(acp.runtime_isolation, 'close_runtime_process',
                               new=mock.AsyncMock()):
            channel = acp.WorkerChannel('s', process, received.append)
            closing = asyncio.create_task(channel.close())
            await asyncio.sleep(0)
            process.stdout.feed_data(b'{"method":"last","params":{}}\n')
            process.stdout.feed_eof()
            await closing
        self.assertEqual(received, [{"method": "last", "params": {}}])

    async def test_stdin_cleanup_error_still_releases_process_once(self):
        process = mock.Mock(returncode=0, stdout=asyncio.StreamReader(),
                            wait=mock.AsyncMock(return_value=0))
        process.stdin.wait_closed = mock.AsyncMock(side_effect=RuntimeError('stdin'))
        with mock.patch.object(acp.runtime_isolation, 'close_runtime_process',
                               new=mock.AsyncMock()) as release:
            channel = acp.WorkerChannel('s', process, lambda msg: None)
            results = await asyncio.gather(channel.close(), channel.close(),
                                           return_exceptions=True)
        self.assertTrue(all(isinstance(result, RuntimeError) for result in results))
        release.assert_awaited_once_with(process)

    async def test_cancelling_close_does_not_cancel_resource_cleanup(self):
        entered, exited = asyncio.Event(), asyncio.Event()

        async def wait():
            entered.set()
            await exited.wait()
            return 0
        process = mock.Mock(returncode=None, stdout=asyncio.StreamReader(), wait=wait)
        process.stdin.wait_closed = mock.AsyncMock()
        with mock.patch.object(acp.runtime_isolation, 'close_runtime_process',
                               new=mock.AsyncMock()) as release:
            channel = acp.WorkerChannel('s', process, lambda msg: None)
            closing = asyncio.create_task(channel.close())
            await entered.wait()
            closing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            exited.set()
            await channel.close()
        release.assert_awaited_once_with(process)

    async def test_reader_cancellation_does_not_abandon_credential_cleanup(self):
        started, finish = asyncio.Event(), asyncio.Event()

        async def cleanup():
            started.set()
            await finish.wait()
        process = mock.Mock(returncode=0, stdout=asyncio.StreamReader(),
                            wait=mock.AsyncMock(return_value=0))
        process.stdin.wait_closed = mock.AsyncMock()
        delegation = mock.Mock(close=mock.AsyncMock(side_effect=cleanup))
        with mock.patch.object(acp.runtime_isolation, 'close_runtime_process',
                               new=mock.AsyncMock()):
            channel = acp.WorkerChannel('s', process, lambda msg: None, delegation)
            process.stdout.feed_eof()
            await started.wait()
            channel._reader_task.cancel()
            closing = asyncio.create_task(channel.close())
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            finish.set()
            await closing
        delegation.close.assert_awaited_once_with()

    async def test_eof_fails_request_before_credential_cleanup_finishes(self):
        process = mock.Mock(returncode=None, stdout=asyncio.StreamReader(),
                            wait=mock.AsyncMock(return_value=0))
        process.stdin.drain = mock.AsyncMock()
        cleanup_started, cleanup_finish = asyncio.Event(), asyncio.Event()

        async def cleanup():
            cleanup_started.set()
            await cleanup_finish.wait()
        delegation = mock.Mock(close=mock.AsyncMock(side_effect=cleanup))
        with mock.patch.object(acp.runtime_isolation, 'close_runtime_process',
                               new=mock.AsyncMock()):
            channel = acp.WorkerChannel('s', process, lambda msg: None, delegation)
            pending = asyncio.create_task(channel.request('session/new', {}))
            await asyncio.sleep(0)
            process.stdout.feed_eof()
            await cleanup_started.wait()
            try:
                with self.assertRaises(acps.TransportError):
                    await asyncio.wait_for(pending, 1)
                process.wait.assert_not_awaited()
            finally:
                cleanup_finish.set()
                process.stdin.wait_closed = mock.AsyncMock()
                await channel.close()


@unittest.skipUnless(sys.platform == 'win32', 'native Windows Proactor pipes')
class NativePipeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_child_roundtrip_and_graceful_stdin_eof(self):
        # This is an OS pipe witness, not a substitute Loki entrypoint. The
        # release-entrypoint tests cover the actual AppContainer worker gate.
        front, child = workers._stdio()
        streams = workers._Streams(*front)
        process = None
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {"handle_list": list(child)}
        code = (
            "import msvcrt,os,sys; "
            "r=msvcrt.open_osfhandle(int(sys.argv[1]),os.O_RDONLY|os.O_BINARY); "
            "w=msvcrt.open_osfhandle(int(sys.argv[2]),os.O_WRONLY|os.O_BINARY); "
            "data=os.read(r,3); assert data==b'abc'; assert os.read(r,1)==b''; "
            "os.write(w,data+b'\\n'); os.close(r); os.close(w)"
        )
        try:
            try:
                await streams.connect()
                process = await asyncio.create_subprocess_exec(
                    sys.executable, '-I', '-c', code, *(str(h) for h in child),
                    startupinfo=startup, close_fds=True)
            finally:
                for handle in child:
                    api.close_handle(handle)
            streams.stdin.write(b'abc')
            streams.stdin.close()
            await asyncio.wait_for(streams.stdin.wait_closed(), 2)
            self.assertEqual(await asyncio.wait_for(streams.stdout.read(), 2), b'abc\n')
            self.assertEqual(await asyncio.wait_for(process.wait(), 2), 0)
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            await streams.close()

    async def test_peer_end_closure_delivers_eof_and_broken_pipe(self):
        front, child = workers._stdio()
        streams = workers._Streams(*front)
        try:
            try:
                await streams.connect()
            finally:
                for handle in child:
                    api.close_handle(handle)
            self.assertEqual(await asyncio.wait_for(streams.stdout.read(), 2), b'')
            await asyncio.wait_for(asyncio.shield(streams.stdin.closed), 2)
            with self.assertRaises((BrokenPipeError, ConnectionError, OSError)):
                streams.stdin.write(b'peer has closed')
                await streams.stdin.drain()
        finally:
            await streams.close()

    async def test_full_pipe_abort_keeps_event_loop_responsive(self):
        front, child = workers._stdio()
        streams = workers._Streams(*front)
        try:
            await streams.connect()
            streams.stdin.write(b'x' * (4 * 1024 * 1024))
            waiting = asyncio.create_task(streams.stdin.drain())
            await asyncio.sleep(0.05)
            self.assertFalse(waiting.done())
            # Peer endpoints are deliberately still open and not reading.
            await asyncio.wait_for(streams.close(), 2)
            with self.assertRaises((BrokenPipeError, ConnectionError, OSError)):
                await waiting
        finally:
            await streams.close()
            for handle in child:
                api.close_handle(handle)
