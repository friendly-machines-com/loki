"""ACP transport and front/worker tests.

The end-to-end test runs the real front process and its spawned worker
over pipes with the dummy provider (no network): initialize, session/new
(real subprocess spawn), session/prompt, and the reply's stopReason.
"""

import asyncio
import io
import errno
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from unittest import mock
from loki_entrypoints import child_environment, configure_container, loki_acp_command
from directory_fixtures import directory_alias
from response_header_fixtures import setUpModule as set_up_response_headers
from config_fixtures import setUpModule as set_up_config
from process_lifecycle_fixtures import close_fixture_connection, observe_fixture, observe_process_exit

from loki_agent import (
    __version__,
    acp,
    acp_main,
    acps,
    authentications,
    models,
    protocols,
    streams,
)
from loki_agent.credentials import CredentialStore, is_credential_name
from loki_agent import connections, paths as app_paths
from loki_agent import jobs as job_runtime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setUpModule():
    set_up_config()
    set_up_response_headers()


def _configured_workspace(tmpdir):
    """The session cwd a test must request: the registered container dir.

    Every front-spawning fixture here configures ``<tmpdir>/workspace`` with
    the real ``loki-setup``, and the Windows front refuses a session whose
    cwd is not a configured container.  On POSIX this is inert.
    """
    return os.path.join(tmpdir, "workspace")


def _close_process_streams(process):
    for name in ["stdin", "stdout", "stderr"]:
        stream = getattr(process, name, None)
        if stream is not None and not stream.closed:
            stream.close()


class _FrontInput:
    """One serial test input stream, not an application prompt queue.

    A send acknowledges routing by Front.run, never completion of the request.
    This catches handlers which starve their own reverse responses or EOF.
    """

    def __init__(self, front):
        self.front = front
        self.messages = asyncio.Queue()
        front.read = self.read
        self.task = asyncio.create_task(front.run())

    async def read(self):
        while True:
            message, routed = await self.messages.get()
            if message is None:
                routed.set_result(None)
                return
            yield message
            routed.set_result(None)

    async def send(self, message):
        await self.send_many(message)

    async def send_many(self, *messages):
        routed = []
        for message in messages:
            future = asyncio.get_running_loop().create_future()
            self.messages.put_nowait((message, future))
            routed.append(future)
        for future in routed:
            await asyncio.wait([future, self.task], return_when=asyncio.FIRST_COMPLETED)
            if not future.done():
                await self.task
                raise AssertionError('front exited before acknowledging message routing')
            future.result()

    async def finish(self):
        await self.send(None)
        await self.task

    async def close(self):
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


class FrontInputFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_protocol_write_failure_retires_reader_and_saved_output_copy(self):
        from loki_agent import credential_storages

        reader = acps.AsyncFdLineReader(0)
        source = mock.Mock(closed=False)
        source.aclose = mock.AsyncMock(side_effect=lambda: setattr(source, 'closed', True))
        reader._threaded = source
        reader._chunks.put_nowait(b'{\n')
        storage = mock.Mock()
        storage.load_openai_subscription.return_value = None
        failure = BrokenPipeError('client closed protocol output')
        copies = []
        duplicate = os.dup

        def dup(fd):
            result = duplicate(fd)
            copies.append(result)
            return result

        try:
            with mock.patch.object(acps, 'sys', mock.Mock(platform='win32')), \
                    mock.patch.object(acps, 'AsyncFdLineReader', return_value=reader), \
                    mock.patch.object(acps, 'quarantine_stdout'), \
                    mock.patch.object(acps, 'make_writer', return_value=mock.Mock(side_effect=failure)), \
                    mock.patch.object(credential_storages, 'JsonCredentialStorage', return_value=storage), \
                    mock.patch.object(acp_main.os, 'dup', side_effect=dup):
                with self.assertRaises(BrokenPipeError) as caught:
                    await acp_main.amain(CredentialStore({}))
            self.assertIs(caught.exception, failure)
            source.aclose.assert_awaited_once_with()
            self.assertTrue(source.closed)
            storage.close.assert_called_once_with()
            self.assertEqual(len(copies), 1)
            with self.assertRaises(OSError) as closed:
                os.fstat(copies[0])
            self.assertEqual(closed.exception.errno, errno.EBADF)
        finally:
            await reader.aclose()
            for fd in copies:
                try:
                    os.close(fd)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise

    async def test_routing_observes_front_failure_without_a_watchdog(self):
        failure = OSError('front failed before routing')
        front = mock.Mock(run=mock.AsyncMock(side_effect=failure))
        source = _FrontInput(front)
        try:
            with self.assertRaises(OSError) as caught:
                await source.send({'id': 1, 'method': 'initialize'})
            self.assertIs(caught.exception, failure)
        finally:
            await source.close()

    async def test_event_wait_observes_front_exit(self):
        source = _FrontInput(mock.Mock(run=mock.AsyncMock()))
        before = asyncio.all_tasks()
        try:
            with self.assertRaisesRegex(AssertionError, 'operation ended before expected fixture event'):
                await observe_fixture(asyncio.Event().wait(), source.task)
            self.assertFalse(asyncio.all_tasks() - before)
        finally:
            await source.close()

    async def test_event_wait_propagates_front_failure(self):
        failure = OSError('front failed while awaiting an event')
        source = _FrontInput(mock.Mock(run=mock.AsyncMock(side_effect=failure)))
        before = asyncio.all_tasks()
        try:
            with self.assertRaises(OSError) as caught:
                await observe_fixture(asyncio.Event().wait(), source.task)
            self.assertIs(caught.exception, failure)
            self.assertFalse(asyncio.all_tasks() - before)
        finally:
            await source.close()

    async def test_event_wait_joins_its_observer_without_cancelling_owners(self):
        source = _FrontInput(mock.Mock(run=asyncio.Event().wait))
        operation = asyncio.get_running_loop().create_future()
        event = asyncio.Event()
        before = asyncio.all_tasks()
        try:
            event.set()
            self.assertTrue(await observe_fixture(event.wait(), source.task, operation))
            self.assertFalse(operation.done())
            self.assertFalse(source.task.done())
            self.assertFalse(asyncio.all_tasks() - before)
        finally:
            await source.close()

    async def test_cancelled_event_wait_joins_only_its_observer(self):
        source = _FrontInput(mock.Mock(run=asyncio.Event().wait))
        operation = asyncio.get_running_loop().create_future()
        before = asyncio.all_tasks()
        waiting = asyncio.create_task(observe_fixture(asyncio.Event().wait(), source.task, operation))
        try:
            await asyncio.sleep(0)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiting
            self.assertFalse(operation.done())
            self.assertFalse(source.task.done())
            self.assertFalse(asyncio.all_tasks() - before)
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            await source.close()

    async def test_event_wait_rejects_reply_before_event(self):
        source = _FrontInput(mock.Mock(run=asyncio.Event().wait))
        operation = asyncio.get_running_loop().create_future()
        before = asyncio.all_tasks()
        try:
            operation.set_result('reply before event')
            with self.assertRaisesRegex(AssertionError, 'reply before event'):
                await observe_fixture(asyncio.Event().wait(), source.task, operation)
            self.assertEqual(operation.result(), 'reply before event')
            self.assertFalse(source.task.done())
            self.assertFalse(asyncio.all_tasks() - before)
        finally:
            await source.close()


class AsyncFdLineReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_threaded_native_error_closes_its_reader_and_propagates(self):
        reader = acps.AsyncFdLineReader(7)
        source = mock.Mock(closed=False)
        entered, release = asyncio.Event(), asyncio.Event()

        async def close():
            entered.set()
            await release.wait()
            source.closed = True

        source.aclose = mock.AsyncMock(side_effect=close)
        reader._threaded = source
        error = OSError('native input failed')
        reader._chunks.put_nowait(error)
        with mock.patch.object(acps.sys, 'platform', 'win32'):
            task = asyncio.create_task(reader.readline())
            try:
                await entered.wait()
                self.assertIs(reader._threaded, source)
                task.cancel()  # Another cancellation must not abandon the close.
                release.set()
                with self.assertRaises(OSError) as caught:
                    await task
                self.assertIs(caught.exception, error)
                source.aclose.assert_awaited_once_with()
                self.assertIsNone(reader._threaded)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    async def test_threaded_cancelled_wakeup_retains_bytes_accepted_before_stop(self):
        reader = acps.AsyncFdLineReader(7)
        source = mock.Mock(closed=False)
        source.aclose = mock.AsyncMock(side_effect=lambda: setattr(source, 'closed', True))
        with mock.patch.object(acps.sys, 'platform', 'win32'), \
                mock.patch.object(acps.handle_reader, 'HandleReader',
                                  side_effect=[source, AssertionError('accepted bytes restarted the reader')]):
            pending = asyncio.create_task(reader.readline())
            try:
                await asyncio.sleep(0)
                reader._chunks.put_nowait(b'accepted before stop\n')
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                source.aclose.assert_awaited_once_with()
                self.assertTrue(source.closed)
                self.assertEqual(await reader.readline(), b'accepted before stop\n')
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def test_threaded_startup_failure_joins_the_published_reader(self):
        error = OSError('reader startup failed')
        source = mock.Mock(closed=False, start=mock.Mock(side_effect=error))
        source.aclose = mock.AsyncMock(side_effect=lambda: setattr(source, 'closed', True))
        reader = acps.AsyncFdLineReader(7)
        with mock.patch.object(acps.sys, 'platform', 'win32'), \
                mock.patch.object(acps.handle_reader, 'HandleReader', return_value=source):
            with self.assertRaises(OSError) as caught:
                await reader.readline()
        self.assertIs(caught.exception, error)
        source.aclose.assert_awaited_once_with()
        self.assertIsNone(reader._threaded)

    async def test_reads_buffered_lines_and_eof_without_changing_fd_flags(self):
        read_fd, write_fd = os.pipe()
        reader = acps.AsyncFdLineReader(read_fd, chunk_size=7)
        blocking_before = os.get_blocking(read_fd)
        try:
            os.write(write_fd, b"first\nsecond\npartial")
            os.close(write_fd)
            write_fd = None
            self.assertEqual([line async for line in reader], [b"first\n", b"second\n", b"partial"])
            self.assertEqual(os.get_blocking(read_fd), blocking_before)
        finally:
            try:
                await reader.aclose()
            finally:
                os.close(read_fd)
                if write_fd is not None:
                    os.close(write_fd)

    async def test_cancelled_read_unregisters_cleanly(self):
        read_fd, write_fd = os.pipe()
        reader = acps.AsyncFdLineReader(read_fd)
        try:
            pending = asyncio.create_task(reader.readline())
            await asyncio.sleep(0)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            os.write(write_fd, b"after cancellation\n")
            self.assertEqual(await reader.readline(), b"after cancellation\n")
        finally:
            try:
                await reader.aclose()
            finally:
                os.close(read_fd)
                os.close(write_fd)

    @contextmanager
    def _readable(self, chunks):
        callbacks = []
        loop = asyncio.get_running_loop()
        with mock.patch.object(acps, 'sys', mock.Mock(platform='linux')), \
                mock.patch.object(loop, 'add_reader', side_effect=lambda fd, callback: callbacks.append(callback)), \
                mock.patch.object(loop, 'remove_reader'), \
                mock.patch.object(acps.os, 'read', side_effect=iter(chunks)):
            yield callbacks

    async def test_cancelled_wakeup_retains_large_line_and_eof_tail(self):
        line, tail = b'x' * 65537 + b'\n', b'y' * 65537
        reader = acps.AsyncFdLineReader(7, chunk_size=len(line) + len(tail))
        with self._readable([line + tail, b'']) as callbacks:
            pending = asyncio.create_task(reader.readline())
            try:
                await asyncio.sleep(0)
                callbacks[-1]()
                pending.cancel()  # Bytes were consumed, but the waiter has not resumed.
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                self.assertEqual(await reader.readline(), line)
                self.assertEqual(len(callbacks), 1, 'buffered bytes registered another fd read')
                pending = asyncio.create_task(reader.readline())
                await asyncio.sleep(0)
                callbacks[-1]()
                self.assertEqual(await pending, tail)
                self.assertEqual(await reader.readline(), b'')
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def test_fragmented_read_keeps_admission_and_prefix_through_cancellation(self):
        reader = acps.AsyncFdLineReader(7)
        with self._readable([b'prefix', b'suffix\n']) as callbacks:
            pending = asyncio.create_task(reader.readline())
            try:
                await asyncio.sleep(0)
                callbacks[-1]()
                await asyncio.sleep(0)
                with self.assertRaisesRegex(RuntimeError, 'concurrent reads'):
                    await reader.readline()
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                pending = asyncio.create_task(reader.readline())
                await asyncio.sleep(0)
                callbacks[-1]()
                self.assertEqual(await pending, b'prefixsuffix\n')
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def test_reads_through_event_loop_readiness_without_an_executor(self):
        read_fd, write_fd = os.pipe()
        reader = acps.AsyncFdLineReader(read_fd)
        try:
            os.write(write_fd, b"message\n")
            with mock.patch.object(asyncio.BaseEventLoop, "run_in_executor",
                                   side_effect=AssertionError("ACP stdin used an executor")):
                self.assertEqual(await reader.readline(), b"message\n")
        finally:
            try:
                await reader.aclose()
            finally:
                os.close(read_fd)
                os.close(write_fd)


class WorkerChannelLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_channel_close_joins_the_owned_process_transport(self):
        entered = asyncio.Event()
        pipe = mock.Mock()
        pipe.is_closing.return_value = False
        transport = mock.Mock()
        transport.get_pipe_transport.side_effect = lambda fd: pipe if fd == 1 else None
        transport.get_returncode.return_value = 0
        transport.close.side_effect = entered.set
        protocol = streams.ProcessProtocol()
        protocol.connection_made(transport)
        protocol.exited.set()
        process = streams.Process(transport, protocol)
        channel = acp.WorkerChannel('session', process, mock.Mock())
        closing = asyncio.create_task(channel.close())
        try:
            await observe_fixture(entered.wait(), closing)
            self.assertFalse(closing.done())
            protocol.connection_lost(None)
            await closing
            self.assertTrue(channel._reader_task.done())
        finally:
            protocol.connection_lost(None)
            await asyncio.gather(closing, return_exceptions=True)

    async def test_revoke_failure_still_closes_and_joins_every_session(self):
        from loki_agent import windows_api

        front = acp.Front(lambda: None, mock.Mock(), CredentialStore({}))
        first = front._reserve_session('first')
        second = front._reserve_session('second')
        failure = windows_api.WindowsApiError('CloseHandle failed', status=6)
        first.delegation = mock.Mock(revoke_now=mock.Mock(side_effect=failure), close=mock.AsyncMock())
        second.delegation = mock.Mock(close=mock.AsyncMock())
        with self.assertRaises(windows_api.WindowsApiError) as caught:
            await front.shutdown()
        self.assertIs(caught.exception, failure)
        first.delegation.close.assert_awaited_once_with()
        second.delegation.close.assert_awaited_once_with()
        self.assertTrue(first.cleanup.done())
        self.assertTrue(second.cleanup.done())
        self.assertFalse(front._sessions)

    async def test_failed_retirement_keeps_its_closing_owner_and_reports_failure(self):
        from loki_agent import windows_api

        front = acp.Front(lambda: None, mock.Mock(), CredentialStore({}))
        owner = front._reserve_session('session')
        failure = windows_api.WindowsApiError('CloseHandle failed', status=6)
        owner.channel = mock.Mock(close=mock.AsyncMock(side_effect=failure))
        with self.assertRaises(windows_api.WindowsApiError) as caught:
            await front.shutdown()
        self.assertIs(caught.exception, failure)
        self.assertTrue(owner.cleanup.done())
        self.assertIs(front._sessions['session'], owner)
        self.assertEqual(owner.state, 'closing')
        owner.channel.close.assert_awaited_once_with()

    async def test_worker_exit_awaits_credential_capability_cleanup(self):
        class Output:
            async def readline(self):
                return b""

        class Process:
            stdout = Output()
            stdin = None
            returncode = 0

            async def wait(self):
                return 0

        delegation = mock.Mock()
        delegation.revoke_now = mock.Mock()
        delegation.close = mock.AsyncMock()
        channel = acp.WorkerChannel(
            "session", Process(), lambda message: None, delegation)

        await channel._reader_task

        delegation.revoke_now.assert_called_once_with()
        delegation.close.assert_awaited_once_with()
        self.assertIsNone(channel.credential_delegation)

    async def _assert_platform_retirement(self, ending):
        from process_lifecycle_fixtures import process_lifecycle

        async with process_lifecycle('loki-acp') as fixture:
            delegation = await fixture.supervisor.delegate()
            process = fixture.record_process(await acp.runtime_isolation.start_worker(
                fixture.workspace, fixture.supervisor.environment, delegation,
                credential_directory=fixture.supervisor.storage.directory_handle))
            delegation.child_spawned()
            channel = acp.WorkerChannel('session', process, lambda message: None, delegation)
            try:
                # A reply proves that the real worker passed its startup gate
                # and established its delegated runtime, not just that it spawned.
                self.assertEqual(await channel.request('session/cancel', {}), {})
                if ending == 'input':
                    process.stdin.close()
                    await process.stdin.wait_closed()
                    self.assertEqual(await process.wait(), 0)
                elif ending == 'revocation':
                    # Keep stdin open: only real authority loss can end this
                    # otherwise-idle worker, before channel.close helps it.
                    delegation.revoke_now()
                    self.assertEqual(await process.wait(), 1)
            finally:
                await channel.close()
            # close joins exit; it does not promise which independently watched
            # EOF won. The ordered controls above assert exact success/failure.
            self.assertIsNotNone(process.returncode)
            self.assertTrue(channel._reader_task.done())
            self.assertTrue(process.stdout.at_eof())
            self.assertTrue(process.stdin.is_closing())
            self.assertIsNone(channel.credential_delegation)
            await fixture.assert_released(self)
            # Repeated caller cleanup must not compete with transport ownership.
            await channel.close()
            await fixture.assert_released(self)

    async def test_close_releases_the_platform_process(self):
        await self._assert_platform_retirement('channel')

    async def test_input_eof_completes_the_platform_worker_successfully(self):
        await self._assert_platform_retirement('input')

    async def test_revocation_interrupts_the_platform_worker(self):
        await self._assert_platform_retirement('revocation')

    async def test_a_dead_worker_fails_the_pending_request(self):
        # A worker that exits without answering must fail the request that
        # waits on it, not leave the caller waiting for a reply that cannot
        # arrive.
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "pass",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        channel = acp.WorkerChannel("dying", process, lambda message: None)
        try:
            with self.assertRaises(acps.TransportError):
                await channel.request("session/new", {})
        finally:
            await channel.close()


class WorkerSpawnGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_child_copy_release_failure_retires_the_created_worker(self):
        from loki_agent import windows_api

        front = acp.Front(lambda: None, mock.Mock(), CredentialStore({}))
        failure = windows_api.WindowsApiError('CloseHandle failed', status=6)
        delegation = mock.Mock(child_spawned=mock.Mock(side_effect=failure), close=mock.AsyncMock())
        process = mock.Mock(stdin=None, stdout=asyncio.StreamReader(), returncode=None)

        async def wait():
            process.returncode = 0
            process.stdout.feed_eof()
            return 0

        process.wait = mock.AsyncMock(side_effect=wait)
        with mock.patch.object(front.credential_supervisor, 'delegate', new=mock.AsyncMock(return_value=delegation)), \
                mock.patch.object(acp.runtime_isolation, 'start_worker', new=mock.AsyncMock(return_value=process)):
            with self.assertRaises(windows_api.WindowsApiError) as caught:
                await front._open_worker(cwd='workspace-operand', open_method='session/new')
            await asyncio.gather(*list(front._tasks), return_exceptions=True)
        self.assertIs(caught.exception, failure)
        process.wait.assert_awaited()
        delegation.close.assert_awaited()
        self.assertEqual(process.returncode, 0)
        self.assertFalse(front._sessions)

    async def test_denied_workspace_fails_closed_and_releases_delegation(self):
        # A session whose workspace has no configured container must not get
        # an unprotected worker: the spawn seam's refusal surfaces as a
        # transport error and the delegation it created is closed.
        front = acp.Front(
            lambda: None, lambda message: None, CredentialStore({}))
        delegation = mock.Mock()
        delegation.close = mock.AsyncMock()
        spawned = mock.AsyncMock(
            side_effect=acp.RuntimeIsolationError(
                "No Windows container configured; run loki-setup"))
        with mock.patch.object(
                front.credential_supervisor,
                "delegate",
                new=mock.AsyncMock(return_value=delegation)), \
                mock.patch.object(
                    acp.runtime_isolation, "start_worker", new=spawned):
            with self.assertRaisesRegex(
                    acps.TransportError, "could not start worker"):
                await front._open_worker(
                    cwd=ROOT, open_method="session/new")
        spawned.assert_awaited_once_with(
            ROOT, front.environment, delegation, credential_directory=None)
        delegation.close.assert_awaited_once_with()
        self.assertFalse(front.workers)
        self.assertFalse(any(owner.state == "opening"
                             for owner in front._sessions.values()))


class FrontPromptOrderingTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffered_restore_close_settles_unstarted_request(self):
        responses = []
        front = acp.Front(lambda: None, responses.append, CredentialStore({}))
        source = _FrontInput(front)
        self.addAsyncCleanup(source.close)
        with mock.patch.object(front.credential_supervisor, 'delegate',
                               new=mock.AsyncMock()) as delegate:
            await source.send_many(
                acps.request(1, 'session/resume', {'sessionId': 'saved', 'cwd': ROOT}),
                acps.request(2, 'session/close', {'sessionId': 'saved'}))
            # The restore is intentionally cancelled before entry; join it and
            # the close/reply owners before checking their observable responses.
            await asyncio.gather(*front._tasks, return_exceptions=True)
            by_id = {message['id']: message for message in responses}
            self.assertEqual(by_id[1]['error']['message'], 'session operation was closed')
            self.assertEqual(by_id[2], acps.response(2, result={}))
            self.assertEqual(len(responses), 2)
            delegate.assert_not_awaited()
            self.assertFalse(front._sessions)
            await source.finish()
            self.assertFalse(front._tasks)

    async def test_buffered_config_cannot_overtake_prompt(self):
        order = []
        responses = []
        prompt_reply = asyncio.get_running_loop().create_future()
        forwarding = asyncio.Event()
        entered = asyncio.Event()
        changed = asyncio.Event()

        def write(message):
            responses.append(message)
            changed.set()

        class Channel:
            _closed = False
            process = mock.Mock(returncode=None)

            async def close(self):
                pass

            async def request(self, method, params, forwarded=None):
                order.append(method)
                changed.set()
                if method == "session/prompt":
                    entered.set()
                    await forwarding.wait()
                if forwarded is not None:
                    forwarded.set()
                if method == "session/prompt":
                    return await prompt_reply
                if method == "session/describe_config_selection":
                    # Nothing to approve for this change; the front's read-only
                    # question is part of forwarding a config option now.
                    return {}
                return {"configOptions": []}

        front = acp.Front(
            lambda: None, write, CredentialStore({}))
        owner = front._reserve_session("session")
        owner.channel = Channel()
        owner.state = "active"
        source = _FrontInput(front)
        self.addAsyncCleanup(source.close)

        await source.send_many({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/prompt",
            "params": {
                "sessionId": "session",
                "prompt": [{"type": "text", "text": "hello"}],
            },
        }, {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/set_config_option",
            "params": {
                "sessionId": "session",
                "configId": "reasoning_effort",
                "value": "effort:low",
            },
        })

        operations = list(owner.tasks)
        self.assertTrue(operations)
        await observe_fixture(entered.wait(), source.task, *operations)
        # Even a worker-pipe forwarding wait belongs outside the sole reader.
        await source.send(acps.request(3, 'unknown-method', {}))
        await asyncio.gather(*(front._tasks - owner.tasks))
        self.assertEqual(order, ['session/prompt'])
        self.assertEqual(next(m for m in responses if m.get('id') == 3)['error']['code'],
                         acps.METHOD_NOT_FOUND)
        forwarding.set()
        while len(order) < 3:
            changed.clear()
            await observe_fixture(changed.wait(), source.task, *operations)
        self.assertEqual(order, [
            "session/prompt",
            "session/describe_config_selection",
            "session/set_config_option",
        ])
        prompt_reply.set_result({"stopReason": "end_turn"})
        await asyncio.gather(*front._tasks)
        self.assertEqual(
            {message["id"] for message in responses}, {1, 2, 3})
        await source.finish()


class SavedConnectionAuthorizationTests(
        unittest.IsolatedAsyncioTestCase):
    def _descriptor(self):
        from loki_agent import protocols
        from loki_agent.connections import ConnectionDescriptor

        return ConnectionDescriptor(
            provider_id=None,
            provider_name="Saved Provider",
            model="saved-model",
            chat_url="https://saved.example/v1/chat/completions",
            models_url="https://saved.example/v1/models",
            protocol=protocols.OPENAI_CHAT,
        )

    async def _restore(
            self, method, action, *, advertise=True,
            authorization_connection=True):
        messages = []
        requests = []
        descriptor = self._descriptor()
        front = None
        test_case = self

        class Delegation:
            def child_arguments(self):
                return ()

            def child_spawn_kwargs(self):
                return {}

            def child_spawned(self):
                return None

            def revoke_now(self):
                return None

            async def close(self):
                return None

        class Channel:
            def __init__(self, session_id, process, forward, delegation,
                         reverse_handler=None):
                self.session_id = session_id
                self.closed = False
                self._closed = False
                self.process = mock.Mock(returncode=None)
                self._reader_task = asyncio.get_running_loop().create_future()

            async def request(self, method, params):
                requests.append((method, params))
                if method == "session/prepare_open":
                    return (
                        {
                            "authorizationConnection":
                                descriptor.to_dict(),
                        }
                        if authorization_connection else {})
                if method == "session/commit_open":
                    test_case.assertNotIn(
                        self.session_id, front.workers)
                    return {"configOptions": []}
                raise AssertionError(method)

            async def close(self):
                self.closed = True
                self._closed = True
                if not self._reader_task.done():
                    self._reader_task.set_result(None)

        def write(message):
            messages.append(message)
            if message.get("method") == "elicitation/create":
                async def respond():
                    front.handle(acps.response(
                        message["id"],
                        result=(
                            {
                                "action": "accept",
                                "content": {"authorize": True},
                            }
                            if action == "accept"
                            else {"action": action}
                        ),
                    ))

                asyncio.create_task(respond())

        front = acp.Front(
            lambda: None, write, CredentialStore({}))
        if advertise:
            front.initialize({
                "clientCapabilities": {
                    "elicitation": {"form": {}},
                },
            })
        delegation = Delegation()
        with mock.patch.object(
                front.credential_supervisor,
                "delegate",
                new=mock.AsyncMock(return_value=delegation)), \
                mock.patch.object(
                    acp.runtime_isolation,
                    "start_worker",
                    new=mock.AsyncMock(return_value=object())), \
                mock.patch.object(acp, "WorkerChannel", Channel):
            try:
                result = await front.restore_session(
                    method,
                    {"sessionId": "saved", "cwd": ROOT},
                    request_id=73,
                )
            except Exception as error:
                return front, requests, messages, error
        return front, requests, messages, result

    async def test_decline_closes_provisional_worker_without_commit(self):
        for method in acp.RESTORE_METHODS:
            with self.subTest(method=method):
                front, requests, _messages, error = await self._restore(
                    method, "decline")

                self.assertIsInstance(error, acps.TransportError)
                self.assertEqual(
                    [name for name, _params in requests],
                    ["session/prepare_open"],
                )
                self.assertNotIn("saved", front.workers)
                self.assertNotIn("saved", front._sessions)

    async def test_restore_fails_closed_without_form_elicitation(self):
        for method in acp.RESTORE_METHODS:
            with self.subTest(method=method):
                front, requests, messages, error = await self._restore(
                    method, "accept", advertise=False)

                self.assertIsInstance(error, acps.TransportError)
                self.assertEqual(
                    [name for name, _params in requests],
                    ["session/prepare_open"],
                )
                self.assertFalse(any(
                    message.get("method") == "elicitation/create"
                    for message in messages))
                self.assertNotIn("saved", front.workers)

    async def test_session_lifecycle_requires_explicit_cwd(self):
        front = acp.Front(
            lambda: None, lambda _message: None, CredentialStore({}))
        with self.assertRaisesRegex(
                acps.TransportError, "cwd must be an absolute path"):
            await front.new_session({})
        for method in acp.RESTORE_METHODS:
            with self.subTest(method=method):
                with self.assertRaisesRegex(
                        acps.TransportError,
                        "cwd must be an absolute path"):
                    await front.restore_session(
                        method, {"sessionId": "saved"}, request_id=73)


class QuarantineTests(unittest.TestCase):
    def test_stdout_is_reserved_for_protocol(self):
        code = (
            "import sys, os, json\n"
            "from loki_agent import acps\n"
            "saved = os.dup(1)\n"
            "acps.quarantine_stdout()\n"
            "write = acps.make_writer(saved)\n"
            "write(acps.response(1, result={'ok': True}))\n"
            "print('stray output')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=ROOT)
        lines = [
            line for line in proc.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]),
                         {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})

        # A contained worker is given the null descriptor to redirect to
        # instead of opening the device itself, which its container refuses.
        # The borrowed descriptor stays the caller's, the protocol keeps its
        # own, and stray output is still discarded.
        borrowed = (
            "import sys, os, json\n"
            "from loki_agent import acps\n"
            "null = os.open(os.devnull, os.O_WRONLY)\n"
            "identity = os.fstat(null)\n"
            "saved = os.dup(1)\n"
            "acps.quarantine_stdout(null)\n"
            # quarantine_stdout only dup2s the descriptor; the caller's fd must
            # still be the same object, or the close below would hit another.
            "assert os.path.samestat(os.fstat(null), identity)\n"
            "write = acps.make_writer(saved)\n"
            "write(acps.response(1, result={'ok': True}))\n"
            "print('stray output')\n"
            "os.close(null)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", borrowed],
            capture_output=True, text=True, cwd=ROOT)
        lines = [
            line for line in proc.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]),
                         {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})


class EntrypointTests(unittest.TestCase):

    def test_subagent_uses_inherited_worker_authority(self):
        from loki_agent import runtime_isolation

        order = []
        with mock.patch.object(runtime_isolation, 'verify_contained_runtime',
                               side_effect=lambda: order.append('verify')) as verify, \
                mock.patch.object(
                acp_main,
                "capture_process_credentials") as capture, \
                mock.patch.object(
                    acp_main,
                    "protect_credential_process", side_effect=lambda: order.append('protect')) as protect, \
                mock.patch(
                    "loki_agent.subagents.main",
                    return_value=31) as subagent_main, \
                mock.patch.object(sys, "argv", [
                    "/installed/bin/loki-acp",
                    "--subagent",
                    "Explore",
                    "--session-owner-fd", "7",
                    "--credential-capability-fd", "8",
                ]):
            status = acp_main.main()

        self.assertEqual(status, 31)
        self.assertEqual(order, ['verify', 'protect'])
        verify.assert_called_once_with()
        capture.assert_not_called()
        protect.assert_called_once_with()
        subagent_main.assert_called_once_with([
            "Explore",
            "--session-owner-fd", "7",
            "--credential-capability-fd", "8",
        ])

    def test_subagent_gate_failure_refuses_dispatch_and_root_capture(self):
        from loki_agent import runtime_isolation, windows_api

        for failure in [runtime_isolation.RuntimeIsolationError('gate refused'),
                        windows_api.WindowsApiError('native verification failed')]:
            with self.subTest(failure=type(failure).__name__), \
                    mock.patch.object(runtime_isolation, 'verify_contained_runtime', side_effect=failure), \
                    mock.patch.object(acp_main, 'protect_credential_process') as protect, \
                    mock.patch.object(acp_main, 'capture_process_credentials') as capture, \
                    mock.patch.object(acp_main, 'configure_logging') as logging, \
                    mock.patch('loki_agent.subagents.main', return_value=31) as dispatch, \
                    mock.patch.object(sys, 'argv', ['loki-acp', '--subagent', 'Explore']), \
                    mock.patch.object(sys, 'stderr', io.StringIO()):
                self.assertEqual(acp_main.main(), 2)
                protect.assert_not_called()
                capture.assert_not_called()
                logging.assert_not_called()
                dispatch.assert_not_called()


class WorkerStartupCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_protocol_write_failure_retires_the_input_reader(self):
        from loki_agent import acp_worker_main, loki, settings

        reader = acps.AsyncFdLineReader(0)
        source = mock.Mock(closed=False)
        source.aclose = mock.AsyncMock(side_effect=lambda: setattr(source, 'closed', True))
        reader._threaded = source
        reader._chunks.put_nowait(b'{"id":1,"method":"unsupported"}\n')
        failure = BrokenPipeError('front closed worker output')

        async def run(operation):
            return True, await operation

        runtime = mock.Mock(install=mock.Mock(return_value=loki.CREDENTIALS), run=run)
        try:
            with mock.patch.object(acps, 'sys', mock.Mock(platform='win32')), \
                    mock.patch.object(acps, 'AsyncFdLineReader', return_value=reader), \
                    mock.patch.object(loki, '_DEFAULT_SESSION', loki._DEFAULT_SESSION), \
                    mock.patch.object(loki, 'apply_runtime_config'), \
                    mock.patch.object(loki, 'configure_tool_hook_pipeline'), \
                    mock.patch.object(settings, 'load_settings', new=mock.AsyncMock()):
                with self.assertRaises(BrokenPipeError) as caught:
                    await acp_worker_main._serve_connected_runtime(
                        runtime, mock.Mock(side_effect=failure))
            self.assertIs(caught.exception, failure)
            source.aclose.assert_awaited_once_with()
            self.assertTrue(source.closed)
        finally:
            await reader.aclose()

    async def test_failed_session_construction_closes_adopted_authority(self):
        """Construction failure must retire real adopted channels on both platforms."""
        from loki_agent import acp_worker_main, credential_supervisors, windows_api

        delegation = await credential_supervisors.RuntimeDelegation.create(
            authentications.CredentialBroker())
        # This is an adoption test, not an exec test: transfer the child-owned
        # ends directly rather than leaving supervisor copies that mask EOF.
        owner, delegation.owner_child = delegation.owner_child, None
        capability, delegation.credential_child = delegation.credential_child, None
        try:
            with mock.patch('loki_agent.sessions.Session', side_effect=(
                    OSError('session construction failed'))):
                with self.assertRaisesRegex(OSError, 'session construction failed'):
                    await acp_worker_main.amain(owner, capability, lambda _message: None)
            # The still-owned parent writer must see no reader, not an invalid
            # parent handle. This checks release without querying a closed ID.
            if os.name == 'posix':
                with self.assertRaises(BrokenPipeError) as closed:
                    os.write(delegation.owner_parent, b'probe')
                self.assertEqual(closed.exception.errno, errno.EPIPE)
            else:
                with self.assertRaises(windows_api.WindowsApiError) as closed:
                    windows_api.write_file(delegation.owner_parent.write, b'probe')
                # A closed anonymous-pipe reader can produce either broken
                # pipe or pipe-closing (NO_DATA), not access/invalid-handle errors.
                self.assertIn(closed.exception.status,
                              [windows_api.ERROR_BROKEN_PIPE, windows_api.ERROR_NO_DATA])
            await delegation.credential_server._reader_task
        finally:
            await delegation.close()


class _ACPFrontFixture:
    async def _provider(self, respond, *, protocol=protocols.OPENAI_CHAT):
        """Script inference only; front, worker, delegation and tools stay real."""
        requests = []
        authorizations = []
        cleanup = AsyncExitStack()

        async def serve(reader, writer):
            try:
                header = await reader.readuntil(b'\r\n\r\n')
                fields = {}
                for line in header.split(b'\r\n')[1:]:
                    name, separator, value = line.partition(b':')
                    if separator:
                        fields[name.lower()] = value.strip()
                body = await reader.readexactly(int(fields.get(b'content-length', b'0')))
                if header.startswith(b'GET '):
                    reply = {'data': [{'id': 'workflow-model'}]}
                else:
                    request = json.loads(body)
                    requests.append(request)
                    authorizations.append(fields.get(b'authorization', b'').decode())
                    message = await respond(request)
                    if protocol == protocols.ANTHROPIC_MESSAGES:
                        reply = {'type': 'message', 'role': 'assistant', 'model': request['model'],
                                 'content': message, 'stop_reason':
                                 'tool_use' if any(block['type'] == 'tool_use' for block in message) else 'end_turn'}
                    else:
                        reply = {'choices': [{'message': message, 'finish_reason':
                                              'tool_calls' if message.get('tool_calls') else 'stop'}]}
                encoded = json.dumps(reply).encode()
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
                             b'Connection: close\r\nX-Workflow: observed\r\nContent-Length: '
                             + str(len(encoded)).encode() + b'\r\n\r\n' + encoded)
                await writer.drain()
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass

        def accept(reader, writer):
            # Register before the coroutine can start or fail; the same exit
            # stack owns completed outcomes and never-started writers too.
            task = asyncio.create_task(serve(reader, writer))
            cleanup.push_async_callback(close_fixture_connection, task, writer)
            return task

        # A VM's native AppContainer cannot be assumed to reach host-local
        # loopback services. An explicit external fixture address/forwarded port
        # lets the SAME journey use a normal remote inference origin, without a
        # loopback exemption or any change to the product's network permissions.
        host = os.environ.get('LOKI_TEST_HTTP_HOST', '127.0.0.1')
        requested_port = int(os.environ.get('LOKI_TEST_HTTP_PORT', '0'))
        server = await asyncio.start_server(accept, '127.0.0.1', requested_port, start_serving=False)

        async def close():
            server.close()
            try:
                await cleanup.aclose()
            finally:
                await server.wait_closed()

        self.addAsyncCleanup(close)
        await server.start_serving()
        port = server.sockets[0].getsockname()[1]
        if host != '127.0.0.1':
            # Establish that the external fixture route works before the
            # contained frontend uses it. Failed routing is not a denial.
            reader, writer = await asyncio.open_connection(host, port)
            try:
                writer.write(b'GET /v1/models HTTP/1.1\r\nHost: fixture\r\n\r\n')
                await writer.drain()
                self.assertTrue((await reader.read()).startswith(b'HTTP/1.1 200 OK'))
            finally:
                writer.close()
                await writer.wait_closed()
        route = 'messages' if protocol == protocols.ANTHROPIC_MESSAGES else 'chat/completions'
        return f'http://{host}:{port}/v1/{route}', requests, authorizations

    async def _search_file(self, workspace, name, owner, *, nonce=None):
        # Initial records need complete JSON; replies must additionally match
        # this request's nonce. In both cases an actual owner is mandatory.
        path = os.path.join(workspace, name)

        async def published():
            while True:
                try:
                    with open(path, encoding='utf-8') as stream:
                        record = json.load(stream)
                    if nonce is None or record['nonce'] == nonce:
                        return record
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                await asyncio.sleep(0.01)

        return await observe_fixture(published(), owner)

    _observe_process_exit = observe_process_exit

    async def _front(self, env, workspace, *, cleanup, closed_stderr=False, lifetime_boundary=False, on_diagnostics=None):
        command = loki_acp_command()
        if lifetime_boundary and sys.platform.startswith('linux'):
            # Trusted outer-fixture setup, before the actual sanctioned front.
            # Reuse the deployment fixture's mount operations, never install
            # protection in a hook/tool payload or change the host's mounts.
            prepare = '''import ctypes, json, os, sys
sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], 'tests')]
from loki_agent import runtime_isolations as isolation
from test_state_isolations import mount_call, retained_mount_flags, MS_BIND, MS_REMOUNT
options = json.loads(sys.argv[2])
libc = ctypes.CDLL(None, use_errno=True)
uid, gid = os.getuid(), os.getgid()
isolation._unshare_user_and_mount_namespaces(libc)
isolation._map_current_identity(uid, gid)
mount_call(libc, None, '/', isolation._MS_REC | isolation._MS_PRIVATE)
if options['lifetime']:
    # POSIX ordinary jobs intentionally outlive worker close. The fixture's
    # outer init supplies root retirement, not a new per-command kill policy.
    if libc.unshare(0x20000000) != 0:
        raise OSError(ctypes.get_errno(), 'outer fixture PID namespace failed')
    child = os.fork()
    if child:
        isolation._drop_namespace_capabilities(libc)
        _pid, status = os.waitpid(child, 0)
        raise SystemExit(os.waitstatus_to_exitcode(status))
isolation._drop_namespace_capabilities(libc)
os.execv(sys.argv[3], sys.argv[3:])
'''
            options = {'lifetime': lifetime_boundary}
            command = [sys.executable, '-c', prepare, ROOT, json.dumps(options), *command]
        if closed_stderr:
            command = [sys.executable, '-c',
                       'import os,sys; os.close(2); os.execv(sys.argv[1], sys.argv[1:])',
                       *command]
        process = await asyncio.create_subprocess_exec(
            *command, env=env, cwd=workspace, limit=64 * 1024,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

        # One reader drains and retains diagnostics through shutdown. Retry wire
        # tests can observe the actual backoff announcement without racing this
        # reader or replacing the real child/HTTP/hook boundary with a mock.
        async def read_diagnostics():
            data = bytearray()
            while chunk := await process.stderr.read(65536):
                data.extend(chunk)
                if on_diagnostics is not None:
                    on_diagnostics(data)
            return bytes(data)

        diagnostics = asyncio.create_task(read_diagnostics())

        cleanup(self._close_front, process, diagnostics)
        return process, diagnostics

    async def _close_front(self, process, diagnostics):
        # EOF already makes the real front cancel operations and join every
        # worker/delegation. Killing only the front can leave a worker's cwd
        # held while the fixture is removing its workspace on Windows.
        if process.returncode is None:
            process.stdin.close()
        # A failed assertion may leave protocol messages unread. Discard them
        # while shutting down so pipe backpressure cannot prevent the join.
        while await process.stdout.read(65536):
            pass
        await process.wait()
        await diagnostics

    async def _response(self, front, request_id):
        preceding = []
        while True:
            line = await front.stdout.readline()
            self.assertTrue(line, "ACP front exited before its response")
            message = json.loads(line)
            if message.get("id") == request_id:
                self.assertNotIn("error", message, message)
                return message["result"], preceding
            preceding.append(message)

    async def _request(self, front, request_id, method, params):
        front.stdin.write((json.dumps(acps.request(
            request_id, method, params)) + "\n").encode())
        await front.stdin.drain()
        return await self._response(front, request_id)

    async def _finish(self, front, diagnostics, session_id):
        result, _ = await self._request(
            front, 90, "session/close", {"sessionId": session_id})
        self.assertEqual(result, {})
        # The original owner must no longer route requests to the closed worker.
        front.stdin.write((json.dumps(acps.request(
            91, "session/prompt", {"sessionId": session_id,
                                   "prompt": [{"type": "text", "text": "forbidden"}]}))
                           + "\n").encode())
        await front.stdin.drain()
        while True:
            line = await front.stdout.readline()
            self.assertTrue(line, "front exited before rejecting closed session")
            message = json.loads(line)
            if message.get("id") == 91:
                self.assertIn("unknown session", message["error"]["message"])
                break
        front.stdin.close()
        await front.wait()
        stderr = await diagnostics
        self.assertEqual(front.returncode, 0, stderr.decode(errors="replace"))
        self.assertEqual(await front.stdout.read(), b"")


class ProviderFixtureOwnershipTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    async def _exercise_cleanup(self, complete):
        server = mock.Mock(start_serving=mock.AsyncMock(), wait_closed=mock.AsyncMock())
        server.sockets = [mock.Mock()]
        server.sockets[0].getsockname.return_value = ('127.0.0.1', 4321)
        reader = mock.Mock(
            readuntil=mock.AsyncMock(return_value=b'POST /chat HTTP/1.1\r\nContent-Length: 2\r\n\r\n'),
            readexactly=mock.AsyncMock(return_value=b'{}'))
        failure = RuntimeError('provider close failed')
        writer = mock.Mock(drain=mock.AsyncMock(), wait_closed=mock.AsyncMock(
            side_effect=failure if complete else None))
        cleanups = []
        with mock.patch.dict(os.environ, {'LOKI_TEST_HTTP_HOST': '127.0.0.1', 'LOKI_TEST_HTTP_PORT': '0'}), \
                mock.patch.object(asyncio, 'start_server', new=mock.AsyncMock(return_value=server)) as start, \
                mock.patch.object(self, 'addAsyncCleanup', side_effect=cleanups.append):
            try:
                await self._provider(mock.AsyncMock(return_value={'role': 'assistant', 'content': 'ok'}))
                operation = start.await_args.args[0](reader, writer)
                # Model start_server's documented scheduling of coroutine callbacks
                # too, so this regression also exposes the former async callback.
                task = asyncio.create_task(operation) if asyncio.iscoroutine(operation) else operation
                if complete:
                    await asyncio.gather(task, return_exceptions=True)
                close = cleanups.pop()
                if complete:
                    with self.assertRaises(RuntimeError) as caught:
                        await close()
                    self.assertIs(caught.exception, failure)
                else:
                    await close()
                    self.assertTrue(task.cancelled())
                    reader.readuntil.assert_not_awaited()
            finally:
                for close in reversed(cleanups):
                    await close()
        writer.close.assert_called()
        writer.wait_closed.assert_awaited()
        server.close.assert_called_once()
        server.wait_closed.assert_awaited_once()

    async def test_completed_handler_close_failure_is_not_lost(self):
        await self._exercise_cleanup(complete=True)

    async def test_never_started_handler_still_closes_its_writer(self):
        await self._exercise_cleanup(complete=False)


class FrontWorkerTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    def _front_env(self, tmpdir):
        env = dict(os.environ)
        env.update({
            "HOME": tmpdir,
            "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
            "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
            "TERM": "dumb",
            "LOKI_PROVIDER": "dummy",
            "LOKI_API_BASE": "http://dummy.invalid/v1",
            "LOKI_MODEL": "dummy-model",
            "LOKI_DUMMY_REPLY": "acp reply text",
        })
        workspace = os.path.join(tmpdir, "workspace")
        os.makedirs(workspace, exist_ok=True)
        configure_container(env, workspace)
        return env

    async def test_ini_logging_reaches_the_front_not_the_contained_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            config = os.path.join(directory, "logging.ini")
            # The handler arg is a Python literal that fileConfig evals, so the
            # path must be repr'd: a raw Windows path would be consumed by
            # backslash escapes before that eval ran, and the front would exit
            # before opening its protocol loop.
            trace_prefix = os.path.join(directory, "trace-")
            with open(config, "w") as stream:
                stream.write("""[loggers]
keys=root
[handlers]
keys=trace
[formatters]
keys=
[logger_root]
level=DEBUG
handlers=trace
[handler_trace]
class=FileHandler
args=(%r + str(__import__('os').getpid()), 'a')
""" % trace_prefix)
            env = self._front_env(directory)
            workspace = _configured_workspace(directory)
            env["LOKI_LOG_CONFIG"] = os.path.relpath(config, workspace)
            front, diagnostics = await self._front(env, workspace, cleanup=self.addAsyncCleanup)
            await self._request(front, 1, "initialize", {"protocolVersion": 1})
            opened, _ = await self._request(front, 2, "session/new", {"cwd": workspace})
            session_id = opened["sessionId"]
            reply, updates = await self._request(front, 3, "session/prompt", {
                "sessionId": session_id, "prompt": [{"type": "text", "text": "logging probe"}],
            })
            self.assertEqual(reply, {"stopReason": "end_turn"})
            self.assertEqual(_tool_updates(updates, session_id), _assistant_chunks("acp reply text"))
            await self._finish(front, diagnostics, session_id)
            # The front is uncontained and loads the INI.  The separately
            # execed worker is contained: it refuses any configuration the
            # invoker names (it cannot be assumed able to read it or write its
            # handlers' targets) and logs to stderr instead, so only the front
            # opens a trace file.
            traces = [name for name in os.listdir(directory)
                      if name.startswith("trace-")]
            self.assertEqual(len(traces), 1)

    def test_front_delegates_credentials_without_worker_environment_values(self):
        credential_name = "LOKI_ACP_BOOTSTRAP_TEST_TOKEN"
        credential_value = "secret-" + ("v" * 137)
        store = CredentialStore({
            "LOKI_API_BASE": "http://dummy.invalid/v1",
            "LOKI_PROVIDER": "dummy",
            "LOKI_MODEL": "dummy-model",
            credential_name: credential_value,
        })
        front = acp.Front(
            lambda: None, lambda message: None, store)
        spawned = {}

        class FakeProcess:
            returncode = None

        class FakeChannel:
            def __init__(
                    self, session_id, process, forward,
                    credential_delegation, reverse_handler=None):
                self.session_id = session_id
                self.credential_delegation = credential_delegation
                self.process = process
                self._closed = False
                self._reader_task = asyncio.get_running_loop().create_future()

            async def request(self, method, params):
                return {}

            async def close(self):
                await self.credential_delegation.close()
                self._closed = True
                if not self._reader_task.done():
                    self._reader_task.set_result(None)

        async def fake_spawn(cwd, environment, delegation, *, credential_directory=None):
            spawned["cwd"] = cwd
            spawned["environment"] = environment
            spawned["delegation"] = delegation
            return FakeProcess()

        async def scenario():
            with mock.patch.object(
                    acp.runtime_isolation, "start_worker",
                    new=fake_spawn), mock.patch.object(
                        acp, "WorkerChannel", FakeChannel):
                session_id, _reply = await front._open_worker(
                    cwd=ROOT, open_method="session/new")
                await front.close_session({'sessionId': session_id})

        asyncio.run(scenario())

        # The spawn seam receives the session's workspace and the sanitized
        # environment -- the spawn-specific kwargs (pipes, descriptors or
        # handle lists) are the seam's own contract, pinned in
        # test_runtime_isolations and test_host_ipc.
        self.assertEqual(spawned["cwd"], ROOT)
        child_environment = spawned["environment"]
        self.assertNotIn(credential_name, child_environment)
        self.assertNotIn(credential_value, repr(child_environment))
        self.assertIsNotNone(spawned["delegation"])
        credential = authentications.CredentialRef.environment(
            credential_name)
        self.assertTrue(front.credentials.has_ref(credential))
        self.assertEqual(front.credentials.get(credential_name), "")
        # Inventory/delegation alone grants no authenticated destination.
        with self.assertRaises(authentications.CredentialUnavailable):
            asyncio.run(front.credential_broker.lease(
                credential, destination="https://example.test/unapproved"))

    def test_real_front_scrubs_and_real_worker_never_receives_credential(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                name: value
                for name, value in self._front_env(tmpdir).items()
                if not is_credential_name(name)
            }
            credential_name = "LOKI_ACP_BOOTSTRAP_TEST_TOKEN"
            credential_value = "secret-" + ("v" * 137)
            env[credential_name] = credential_value
            env["LOKI_ACP_AFTER"] = "visible"

            observer_dir = os.path.join(tmpdir, "observer")
            report_dir = os.path.join(tmpdir, "reports")
            os.makedirs(observer_dir)
            os.makedirs(report_dir)
            sitecustomize = os.path.join(observer_dir, "sitecustomize.py")
            observer = r'''
import ctypes
import json
import os
import sys

from loki_agent import credentials

_capture = credentials.capture_process_credentials


def native_environment():
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        get = kernel.GetEnvironmentStringsW
        get.restype = ctypes.c_void_p
        free = kernel.FreeEnvironmentStringsW
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_int
        pointer = get()
        if not pointer:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entries, address = [], pointer
            while True:
                text = ctypes.wstring_at(address)
                if not text:
                    break
                entries.append(text)
                address += (len(text) + 1) * ctypes.sizeof(ctypes.c_wchar)
            return b"\0".join(entry.encode("utf-8") for entry in entries)
        finally:
            free(pointer)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        libc.sysctl.argtypes = (
            ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t)
        libc.sysctl.restype = ctypes.c_int
        mib = (ctypes.c_int * 3)(1, 49, os.getpid())
        size = ctypes.c_size_t(os.sysconf("SC_ARG_MAX"))
        buffer = ctypes.create_string_buffer(size.value)
        if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), "sysctl failed")
        return buffer.raw[:size.value]
    with open("/proc/self/environ", "rb") as source:
        return source.read()


def report_environment():
    raw = native_environment()
    name = __NAME__.encode("ascii")
    value = __VALUE__.encode("ascii")
    filler = b"x" * len(name + b"=" + value)
    after = b"LOKI_ACP_AFTER=visible\0"
    report = {
        "pid": os.getpid(),
        "worker": "--worker" in sys.argv,
        "name_present": name + b"=" in raw,
        "value_present": value in raw,
        "after_present": after in raw,
    }
    if sys.platform.startswith("linux"):
        # Linux overwrites the original records in place, so the filler and the
        # record that follows it are observable.  Windows removes them instead.
        report["filler_present"] = filler in raw.split(b"\0")
        report["after_follows_filler"] = (
            raw.find(after) > raw.find(filler + b"\0")
            if filler + b"\0" in raw else False)
    path = os.path.join(os.environ["LOKI_TEST_REPORT_DIR"],
                        "%d.json" % os.getpid())
    with open(path, "w", encoding="ascii") as output:
        json.dump(report, output)


def capture_process_credentials():
    store = _capture()
    report_environment()
    return store


credentials.capture_process_credentials = capture_process_credentials
if "--worker" in sys.argv:
    report_environment()
'''
            observer = observer.replace("__NAME__", repr(credential_name))
            observer = observer.replace("__VALUE__", repr(credential_value))
            with open(sitecustomize, "w", encoding="ascii") as stream:
                stream.write(observer)
            env["LOKI_TEST_REPORT_DIR"] = report_dir
            env["PYTHONPATH"] = os.pathsep.join(
                [observer_dir, ROOT])

            # Drive the front the way the product drives a worker channel:
            # an asyncio subprocess read line by line (WorkerChannel), under
            # the default event-loop policy the entrypoints themselves use.
            # stderr stays inherited, so a failing front or worker explains
            # itself in the runner log.
            async def converse(process):
                async def send(message):
                    process.stdin.write(
                        (json.dumps(message) + "\n").encode("utf-8"))
                    await process.stdin.drain()

                async def reply(reply_id):
                    while True:
                        line = await asyncio.wait_for(
                            process.stdout.readline(), 30)
                        self.assertTrue(
                            line,
                            "front closed its protocol output",
                        )
                        message = json.loads(line.decode("utf-8"))
                        if message.get("id") == reply_id:
                            return message

                await send({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": 1},
                })
                self.assertIn("result", await reply(1))
                await send({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/new",
                    "params": {"cwd": _configured_workspace(tmpdir)},
                })
                self.assertIn("result", await reply(2))

            async def run():
                process = await asyncio.create_subprocess_exec(
                    *loki_acp_command(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=None,
                    env=env,
                    cwd=os.path.join(tmpdir, "workspace"),
                )
                try:
                    await converse(process)
                    # Read the observer reports before tearing the
                    # front down: the Linux check below needs the
                    # worker's live /proc entry.
                    report_names = os.listdir(report_dir)
                    self.assertEqual(len(report_names), 2)
                    reports = []
                    for name in report_names:
                        with open(
                                os.path.join(report_dir, name),
                                encoding="ascii") as stream:
                            reports.append(json.load(stream))
                    front_report = next(
                        report for report in reports if not report["worker"])
                    worker_report = next(
                        report for report in reports if report["worker"])

                    for report in reports:
                        self.assertFalse(report["name_present"])
                        self.assertFalse(report["value_present"])
                        self.assertTrue(report["after_present"])
                    if sys.platform.startswith("linux"):
                        # Linux overwrites the credential record in place and
                        # covers the directory with a tmpfs; both are Linux
                        # mechanisms.  The tmpfs cover is containment, whose
                        # Windows counterpart is the AppContainer gate exercised
                        # by test_windows_runtime and test_runtime_gate.  The
                        # in-place overwrite is a scrub-in-place detail: Windows
                        # removes the record instead (see report_environment),
                        # so it has no counterpart assertion here.
                        self.assertTrue(front_report["filler_present"])
                        self.assertTrue(front_report["after_follows_filler"])
                        self.assertFalse(worker_report["filler_present"])
                        self.assertFalse(worker_report["after_follows_filler"])
                        credential_dir = os.path.join(
                            tmpdir, "config", "loki", "credentials")
                        with open(
                                f"/proc/{worker_report['pid']}/mountinfo",
                                encoding="ascii") as stream:
                            worker_mounts = stream.read()
                        self.assertTrue(any(
                            f" {credential_dir} " in line
                            and " - tmpfs " in line
                            for line in worker_mounts.splitlines()
                        ), worker_mounts)
                finally:
                    process.stdin.close()
                    try:
                        await asyncio.wait_for(process.wait(), 5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()

            asyncio.run(run())

    def test_unknown_session_is_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._front_env(tmpdir)
            front = subprocess.Popen(
                loki_acp_command(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=env, cwd=os.path.join(tmpdir, "workspace"))
            self.addCleanup(_close_process_streams, front)
            try:
                front.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": 9,
                    "method": "session/prompt",
                    "params": {"sessionId": "nope",
                               "prompt": [{"type": "text", "text": "x"}]},
                }) + "\n")
                front.stdin.flush()
                line = front.stdout.readline()
                reply = json.loads(line)
                self.assertEqual(reply["id"], 9)
                self.assertIn("error", reply)
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()


class EventMapperTests(unittest.TestCase):
    def test_plan_is_a_full_detached_snapshot_including_empty_clears(self):
        from loki_agent import acp_events
        todos = [
            {"content": "Inspect", "status": "completed", "priority": "high"},
            {"content": "Implement", "status": "in_progress", "priority": "medium"},
            {"content": "Verify", "status": "pending", "priority": "low"},
        ]
        projected = acp_events.plan("session-A", todos)
        self.assertEqual(projected, {
            "sessionId": "session-A", "update": {"sessionUpdate": "plan", "entries": todos}})
        todos[0]["content"] = "Changed later"
        todos.clear()
        self.assertEqual(projected["update"]["entries"][0]["content"], "Inspect")
        self.assertEqual(acp_events.plan("session-B", []), {
            "sessionId": "session-B", "update": {"sessionUpdate": "plan", "entries": []}})

    def test_plan_does_not_send_malformed_saved_entries_or_extra_fields(self):
        import copy
        from loki_agent import acp_events
        valid = {"content": "Restored", "status": "pending", "priority": "high"}
        todos = [
            None, "not a todo", {},
            {**valid, "content": None}, {**valid, "content": "  "},
            {**valid, "status": "unknown"}, {**valid, "status": []},
            {**valid, "priority": "critical"}, {**valid, "priority": {}},
            {**valid, "_meta": {"untrusted": True}},
        ]
        original = copy.deepcopy(todos)
        self.assertEqual(acp_events.plan("restored", todos), {
            "sessionId": "restored", "update": {"sessionUpdate": "plan", "entries": [valid]}})
        self.assertEqual(todos, original)

    def test_rejected_tool_uses_its_real_call_id(self):
        from loki_agent import acp_events
        updates = acp_events.map_event(
            "s",
            {
                "type": "tool_rejected",
                "name": "Write",
                "call_id": "provider-call-77",
                "args": {"file_path": "x"},
            },
            {},
        )
        self.assertEqual(
            updates[0]["update"]["sessionUpdate"], "tool_call")
        self.assertEqual(
            updates[0]["update"]["toolCallId"], "provider-call-77")
        self.assertEqual(updates[0]["update"]["status"], "failed")

    def test_tool_result_error_is_failed(self):
        from loki_agent import acp_events
        updates = acp_events.map_event("s", {"type": "tool_result",
                                             "content": "boom",
                                             "is_error": True},
                                       {"pending_call_id": "call-2"})
        self.assertEqual(updates[0]["update"]["status"], "failed")

    def test_ignored_events_map_to_nothing(self):
        from loki_agent import acp_events
        for kind in ["assistant_end", "response_timing", "max_loops",
                     "assistant_start", "provider_notice"]:
            self.assertEqual(
                acp_events.map_event("s", {"type": kind}, {}), [])


class CancelEndToEndTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    """session/cancel mid-turn must yield stopReason "cancelled"."""

    async def test_cancel_during_streaming_turn(self):
        async with AsyncExitStack() as cleanup:
            tmpdir = cleanup.enter_context(tempfile.TemporaryDirectory())
            gate = os.path.join(tmpdir, "release")
            env = dict(os.environ)
            env.update({
                "HOME": tmpdir,
                "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
                "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
                "TERM": "dumb",
                "LOKI_PROVIDER": "dummy",
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_MODEL": "dummy-model",
                "LOKI_DUMMY_REPLY": "chunked answer",
                "LOKI_STREAM": "1",
                "LOKI_DUMMY_STREAM_CHUNKS":
                    '["first ", "second part"]',
                "LOKI_DUMMY_STREAM_GATE": gate,
            })
            workspace = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace, exist_ok=True)
            configure_container(env, workspace)
            front, _diagnostics = await self._front(
                env, workspace, cleanup=cleanup.push_async_callback)

            async def send(message):
                front.stdin.write((json.dumps(message) + "\n").encode())
                await front.stdin.drain()

            async def recv():
                line = await front.stdout.readline()
                self.assertTrue(line, "front produced no message")
                return json.loads(line)

            await self._request(front, 1, "initialize", {"protocolVersion": 1})
            opened, _ = await self._request(front, 2, "session/new", {"cwd": workspace})
            session_id = opened["sessionId"]
            await send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                        "params": {"sessionId": session_id,
                                   "prompt": [{"type": "text", "text": "hello"}]}})
            # These mid-stream reads cannot use the success-only response helper.
            # The first delta must precede completion while the gate is held.
            while True:
                message = await recv()
                self.assertNotEqual(message.get('id'), 3,
                                    f'prompt finished before its first delta: {message!r}')
                if (message.get("method") == "session/update"
                        and message["params"]["update"].get(
                            "sessionUpdate") == "agent_message_chunk"):
                    break

            await send({"jsonrpc": "2.0", "method": "session/cancel",
                        "params": {"sessionId": session_id}})
            after_cancel = []
            while True:
                reply = await recv()
                if reply.get("id") == 3:
                    break
                after_cancel.append(reply)
            self.assertEqual(reply["result"]["stopReason"], "cancelled")
            self.assertTrue(all(
                message.get("id") is None
                for message in after_cancel
            ), after_cancel)
            # The gate was never released: the cancel, not the gate, ended it.
            self.assertFalse(os.path.exists(gate))


def _conversation_pairs(blob):
    """Inspect canonical durable messages, without a production replay oracle."""
    items = []
    for event in blob["events"]:
        if event["type"] == "model_response":
            items.extend(event["items"])
        else:
            items.append(event)
    return [(item["role"], item["content"]) for item in items
            if item["type"] == "message"
            and item["role"] in ["user", "assistant"]]


def _expected_pairs(*turns):
    return [
        (role, [{"type": "text", "text": text}])
        for prompt, answer in turns
        for role, text in (("user", prompt), ("assistant", answer))
    ]


def _historical_chunks(messages, session_id):
    chunks = []
    for message in messages:
        if message.get("method") != "session/update":
            continue
        params = message["params"]
        update = params["update"]
        kind = update["sessionUpdate"]
        if kind in ["user_message_chunk", "agent_message_chunk"]:
            if params["sessionId"] != session_id:
                raise AssertionError("replay belongs to a different session")
            chunks.append((kind, update["content"]))
    return chunks


class SessionRestoreTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    async def test_failed_request_cleanup_joins_workers_before_removing_workspace(self):
        with self.assertRaisesRegex(AssertionError, 'fixture body failed'):
            async with AsyncExitStack() as cleanup:
                root = cleanup.enter_context(tempfile.TemporaryDirectory())
                workspace = _configured_workspace(root)
                os.mkdir(workspace)
                env = child_environment(
                    HOME=root, PATH=os.environ.get('PATH', ''), TERM='dumb',
                    XDG_CONFIG_HOME=os.path.join(root, 'config'),
                    XDG_STATE_HOME=os.path.join(root, 'state'),
                    LOKI_PROVIDER='dummy', LOKI_API_BASE='http://dummy.invalid/v1',
                    LOKI_MODEL='dummy-model')
                configure_container(env, workspace)
                front, diagnostics = await self._front(env, workspace, cleanup=cleanup.push_async_callback)
                await self._request(front, 1, 'initialize', {'protocolVersion': 1})
                await self._request(front, 2, 'session/new', {'cwd': workspace})
                # The worker is still open when protocol assertions unwind.
                with self.assertRaisesRegex(AssertionError, 'unknown session'):
                    await self._request(front, 3, 'session/prompt', {
                        'sessionId': 'not-open', 'prompt': [{'type': 'text', 'text': 'unused'}]})
                # Leave a response larger than the pipe buffers unread. Scope
                # cleanup must drain it while EOF retires the real worker.
                front.stdin.write((json.dumps(acps.request(4, 'unknown-' + 'x' * 262144, {})) + '\n').encode())
                await front.stdin.drain()
                await front.stdout.readexactly(32)
                raise AssertionError('fixture body failed')
        self.assertEqual(front.returncode, 0, (await diagnostics).decode(errors='replace'))
        self.assertFalse(os.path.exists(root))

    async def test_saved_sessions_and_todo_plans_survive_load_resume_and_continuation(self):
        import datetime
        from loki_agent import formats

        tasks = [
            {"content": "Inspect", "status": "completed", "priority": "high"},
            {"content": "Implement", "status": "in_progress", "priority": "medium"},
        ]
        # These observations share the same new/load/resume lifecycle. Keep the
        # second live worker and fresh fronts: neither isolation nor restoration
        # is proved by rereading state in the worker which created it.
        # Register the directory first so every front joins before its removal.
        root = self.enterContext(tempfile.TemporaryDirectory())
        workspace = _configured_workspace(root)
        os.mkdir(workspace)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("LOKI_")
               and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
        env.update({
            "HOME": root, "XDG_CONFIG_HOME": os.path.join(root, "config"),
            "XDG_STATE_HOME": os.path.join(root, "state"), "TERM": "dumb",
            "LOKI_PROVIDER": "dummy", "LOKI_API_BASE": "http://dummy.invalid/v1",
            "LOKI_MODEL": "dummy-model", "LOKI_DUMMY_REPLY": "first answer",
            "LOKI_DUMMY_TOOL_CALL": json.dumps({"name": "TodoWrite", "arguments": {"todos": tasks}}),
        })
        configure_container(env, workspace)

        async def opened_plan(front, session_id):
            # Read after the reply; _request must not hide premature snapshots.
            for kind in ["available_commands_update", "plan"]:
                raw = await front.stdout.readline()
                self.assertTrue(raw, 'front exited before its plan snapshot')
                message = json.loads(raw)
                self.assertEqual(message["method"], "session/update")
                self.assertEqual(message["params"]["sessionId"], session_id)
                self.assertEqual(message["params"]["update"]["sessionUpdate"], kind)
            return message["params"]["update"]["entries"]

        async def assert_catalogue(front, session_ids):
            for params in [{}, {"cwd": workspace}]:
                listed, _ = await self._request(front, 6, "session/list", params)
                self.assertCountEqual([entry["sessionId"] for entry in listed["sessions"]], session_ids)
                for entry in listed["sessions"]:
                    self.assertEqual(entry["cwd"], workspace)
                    # Independent filesystem observation, not the catalogue's
                    # timestamp helper or a reconstructed pathname operand.
                    with os.scandir(app_paths.chat_log_dir_for(workspace)) as saved:
                        modified = next(item.stat().st_mtime for item in saved
                                        if item.name == f'chat-{entry["sessionId"]}.json')
                    self.assertEqual(datetime.datetime.fromisoformat(entry["updatedAt"]),
                                     datetime.datetime.fromtimestamp(modified, datetime.timezone.utc))

        front, diagnostics = await self._front(env, workspace, cleanup=self.addAsyncCleanup)
        await self._request(front, 1, "initialize", {"protocolVersion": 1})
        opened, preceding = await self._request(front, 2, "session/new", {"cwd": workspace})
        session_id = opened["sessionId"]
        self.assertNotIn("lokiUpdates", opened)
        self.assertEqual(preceding, [])
        self.assertEqual(await opened_plan(front, session_id), [])
        result, updates = await self._request(front, 3, "session/prompt", {
            "sessionId": session_id, "prompt": [{"type": "text", "text": "first prompt"}]})
        self.assertEqual(result, {"stopReason": "end_turn"})
        plans = [message["params"] for message in updates
                 if message.get("method") == "session/update"
                 and message["params"]["update"]["sessionUpdate"] == "plan"]
        self.assertEqual(plans, [{
            "sessionId": session_id, "update": {"sessionUpdate": "plan", "entries": tasks}}])
        await assert_catalogue(front, [session_id])
        second, _ = await self._request(front, 4, "session/new", {"cwd": workspace})
        self.assertNotEqual(second["sessionId"], session_id)
        self.assertEqual(await opened_plan(front, second["sessionId"]), [])
        # Opening an empty worker does not persist a conversation. Make its
        # first turn explicit before requiring both IDs in the durable catalogue.
        await assert_catalogue(front, [session_id])
        result, _ = await self._request(front, 7, "session/prompt", {
            "sessionId": second["sessionId"], "prompt": [{"type": "text", "text": "second prompt"}]})
        self.assertEqual(result, {"stopReason": "end_turn"})
        saved_ids = [session_id, second["sessionId"]]
        await assert_catalogue(front, saved_ids)
        await self._request(front, 5, "session/close", {"sessionId": second["sessionId"]})
        await self._finish(front, diagnostics, session_id)
        path = os.path.join(app_paths.chat_log_dir_for(workspace), f"chat-{session_id}.json")
        expected_permissions = None

        def permissions():
            if sys.platform == 'win32':
                from loki_agent import windows_api
                return windows_api.dacl_sddl(path)
            return os.stat(path).st_mode & 0o7777

        def read_saved(turns):
            with open(path, encoding="utf-8") as stream:
                blob = json.load(stream)
            formats.validate_events(blob["events"])
            self.assertEqual(_conversation_pairs(blob), _expected_pairs(*turns))
            self.assertEqual(blob["session_state"]["shell_cwd"], workspace)
            self.assertEqual(blob["session_todos"], tasks)
            if expected_permissions is not None:
                self.assertEqual(permissions(), expected_permissions)

        turns = [["first prompt", "first answer"]]
        read_saved(turns)
        # Observe permissions independently; fresh workers must preserve them.
        expected_permissions = permissions()
        del env["LOKI_DUMMY_TOOL_CALL"]
        for method, extension, prompt, answer in [
                ["session/load", False, "loaded prompt", "loaded answer"],
                ["session/resume", True, "resumed prompt", "resumed answer"]]:
            with self.subTest(method=method):
                env["LOKI_DUMMY_REPLY"] = answer
                front, diagnostics = await self._front(env, workspace, cleanup=self.addAsyncCleanup)
                initialized, _ = await self._request(front, 1, "initialize", {
                    "protocolVersion": 1,
                    "clientInfo": {
                        "name": "agent-shell", "title": "Emacs Agent Shell",
                        "version": "test",
                    },
                })
                self.assertEqual(initialized["agentCapabilities"]["sessionCapabilities"]["resume"], {})
                await assert_catalogue(front, saved_ids)
                for continuation in [False, True]:
                    params = {"sessionId": session_id, "cwd": workspace, "mcpServers": []}
                    if continuation:
                        params["replay"] = extension
                    restored, replay = await self._request(front, 3, method, params)
                    self.assertNotIn("sessionId", restored)
                    self.assertIn("configOptions", restored)
                    self.assertNotIn("lokiUpdates", restored)
                    self.assertFalse(any(message.get("method") == "session/update"
                                         and message["params"]["update"]["sessionUpdate"] == "plan"
                                         for message in replay))
                    expected_replay = [
                        (kind, {"type": "text", "text": text})
                        for user, assistant in turns
                        for kind, text in [["user_message_chunk", user], ["agent_message_chunk", assistant]]
                    ] if method == "session/load" else []
                    self.assertEqual(_historical_chunks(replay, session_id), expected_replay)
                    if method == "session/resume":
                        self.assertEqual(replay, [])
                    self.assertEqual(await opened_plan(front, session_id), tasks)
                    if not continuation:
                        # Restore-only shutdown must preserve the saved plan;
                        # another prompt must not conceal a close/save defect.
                        await self._request(front, 4, "session/close", {"sessionId": session_id})
                        read_saved(turns)
                        continue
                    result, updates = await self._request(front, 4, "session/prompt", {
                        "sessionId": session_id, "prompt": [{"type": "text", "text": prompt}]})
                    self.assertEqual(result, {"stopReason": "end_turn"})
                    self.assertEqual(_historical_chunks(updates, session_id), [
                        ("agent_message_chunk", {"type": "text", "text": answer})])
                    await self._finish(front, diagnostics, session_id)
                    turns.append([prompt, answer])
                    read_saved(turns)


class SavedApprovalStdioTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    async def test_provisional_approval_rejection_close_and_eof_over_real_pipes(self):
        from loki_agent import protocols
        from loki_agent.connections import ConnectionDescriptor

        async with AsyncExitStack() as cleanup:
            root = cleanup.enter_context(tempfile.TemporaryDirectory())
            workspace = _configured_workspace(root)
            os.mkdir(workspace)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith('LOKI_')
                   and not key.endswith(('_KEY', '_TOKEN', '_PAT'))}
            env.update(HOME=root, XDG_CONFIG_HOME=os.path.join(root, 'config'),
                       XDG_STATE_HOME=os.path.join(root, 'state'), TERM='dumb',
                       LOKI_PROVIDER='dummy', LOKI_API_BASE='http://dummy.invalid/v1',
                       LOKI_MODEL='dummy-model')
            configure_container(env, workspace)
            front, diagnostics = await self._front(env, workspace, cleanup=cleanup.push_async_callback)
            await self._request(front, 1, 'initialize', {'protocolVersion': 1})
            opened, _ = await self._request(front, 2, 'session/new', {'cwd': workspace})
            session_id = opened['sessionId']
            await self._request(front, 3, 'session/prompt', {
                'sessionId': session_id,
                'prompt': [{'type': 'text', 'text': 'seed conversation'}]})
            await self._finish(front, diagnostics, session_id)
            path = os.path.join(app_paths.chat_log_dir_for(workspace), f'chat-{session_id}.json')
            with open(path, encoding='utf-8') as stream:
                saved = json.load(stream)
            saved['session_state']['connection'] = ConnectionDescriptor(
                provider_id=None, provider_name='Saved connection',
                model='untrusted-saved-model', protocol=protocols.OPENAI_CHAT,
                chat_url='https://saved.invalid/v1/chat/completions',
                models_url='https://saved.invalid/v1/models').to_dict()
            with open(path, 'w', encoding='utf-8') as stream:
                json.dump(saved, stream)
            with open(path, 'rb') as stream:
                original = stream.read()
            for name in ['LOKI_PROVIDER', 'LOKI_API_BASE', 'LOKI_MODEL']:
                del env[name]

            for decision in ['accept', 'decline', 'close then accept', 'eof']:
                with self.subTest(decision=decision):
                    front, diagnostics = await self._front(env, workspace, cleanup=cleanup.push_async_callback)
                    await self._request(front, 1, 'initialize', {
                        'protocolVersion': 1,
                        'clientCapabilities': {'elicitation': {'form': {}}}})
                    messages = []

                    async def receive(predicate):
                        while True:
                            raw = await front.stdout.readline()
                            self.assertTrue(raw, 'front exited before expected response')
                            message = json.loads(raw)
                            messages.append(message)
                            if predicate(message):
                                return message

                    async def send(*messages):
                        front.stdin.write(b''.join(
                            (json.dumps(message) + '\n').encode() for message in messages))
                        await front.stdin.drain()

                    await send(acps.request(3, 'session/resume', {
                        'sessionId': session_id, 'cwd': workspace}))
                    ask = await receive(lambda m: m.get('method') == 'elicitation/create' or m.get('id') == 3)
                    self.assertEqual(ask.get('method'), 'elicitation/create', ask)
                    self.assertEqual(ask['params']['requestId'], 3)
                    await send(acps.request(4, 'session/prompt', {
                        'sessionId': session_id,
                        'prompt': [{'type': 'text', 'text': 'must not execute'}]}))
                    rejected = await receive(lambda m: m.get('id') == 4)
                    self.assertIn('request was not executed', rejected['error']['message'])
                    answer = acps.response(ask['id'], result={
                        'action': 'accept', 'content': {'authorize': True}})
                    if decision == 'accept':
                        await send(answer)
                        restored = await receive(lambda m: m.get('id') == 3)
                        self.assertIn('configOptions', restored['result'])
                        await self._finish(front, diagnostics, session_id)
                    else:
                        if decision == 'decline':
                            await send(acps.response(ask['id'], result={'action': 'decline'}))
                            self.assertIn('error', await receive(lambda m: m.get('id') == 3))
                        elif decision == 'close then accept':
                            await send(acps.request(5, 'session/close', {'sessionId': session_id}), answer)
                            await receive(lambda m: m.get('id') == 5)
                            self.assertEqual(next(m for m in messages if m.get('id') == 5)['result'], {})
                            self.assertIn('error', next(m for m in messages if m.get('id') == 3))
                        front.stdin.close()
                        remaining = await front.stdout.read()
                        await front.wait()
                        stderr = await diagnostics
                        self.assertEqual(front.returncode, 0, stderr.decode(errors='replace'))
                        if decision == 'eof':
                            self.assertNotIn(b'"result"', remaining)
                    with open(path, 'rb') as stream:
                        self.assertEqual(stream.read(), original)
                    self.assertFalse(any(m.get('method') == 'session/update' for m in messages))


def _tool_fixture(root):
    workspace = os.path.join(root, "workspace")
    os.mkdir(workspace)
    payload = b"ACP-W3 unique file output"
    with open(os.path.join(workspace, "payload.txt"), "wb") as stream:
        stream.write(payload)
    # A wrong cwd cannot accidentally return the same bytes.
    with open(os.path.join(root, "payload.txt"), "wb") as stream:
        stream.write(b"wrong directory")
    command = "cat payload.txt >> executed.txt && cat executed.txt"
    result = "status: completed\nexit_code: 0\n[stdout]\n" + payload.decode()
    return workspace, payload, command, result


def _tool_updates(messages, session_id):
    updates = []
    for message in messages:
        if message.get("method") != "session/update":
            continue
        params = message["params"]
        if params["sessionId"] != session_id:
            raise AssertionError("tool update belongs to another session")
        update = params["update"]
        if update["sessionUpdate"] in [
                "agent_message_chunk", "user_message_chunk",
                "tool_call", "tool_call_update"]:
            updates.append(update)
    return updates


def _call_and_result(call_id, command, workspace, result):
    # Independent ACP wire oracle, including successful-status omission.
    return [
        {"sessionUpdate": "tool_call", "toolCallId": call_id,
         "title": f"Bash: {command} (cwd: {workspace})",
         "kind": "execute", "status": "in_progress"},
        {"sessionUpdate": "tool_call_update", "toolCallId": call_id,
         "content": [{"type": "content", "content": {
             "type": "text", "text": result}}]},
    ]


def _assistant_chunks(*texts):
    return [{"sessionUpdate": "agent_message_chunk",
             "content": {"type": "text", "text": text}} for text in texts]


class ToolStreamingJourneyTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    async def test_file_witness_propagates_request_failure(self):
        workspace = self.enterContext(tempfile.TemporaryDirectory())
        owner = asyncio.get_running_loop().create_future()
        failure = OSError('search producer failed')
        owner.set_exception(failure)
        before = asyncio.all_tasks()
        with self.assertRaises(OSError) as caught:
            await self._search_file(workspace, 'not-published', owner)
        self.assertIs(caught.exception, failure)
        self.assertFalse(owner.cancelled())
        self.assertEqual(asyncio.all_tasks(), before)

    def _saved_tool_turn(self, path, user, call_id, args, result, answers):
        from loki_agent import formats

        with open(path, encoding="utf-8") as stream:
            blob = json.load(stream)
        formats.validate_events(blob["events"])
        events = blob["events"]
        user_index = next(i for i, event in enumerate(events)
                          if event.get("role") == "user")
        turn = events[user_index:]
        self.assertEqual([event["type"] for event in turn],
                         ["message", "model_response", "tool_result", "model_response"])
        self.assertEqual(_conversation_pairs(blob), [
            ("user", [{"type": "text", "text": user}]),
            *[("assistant", [{"type": "text", "text": answer}]) for answer in answers],
        ])
        calls = [item for event in turn if event["type"] == "model_response"
                 for item in event["items"] if item["type"] == "function_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["call_id"], call_id)
        self.assertEqual(calls[0]["name"], "Bash")
        self.assertEqual(json.loads(calls[0]["arguments"]), args)
        tool_result = turn[2]
        self.assertEqual(tool_result["call_id"], call_id)
        self.assertEqual(tool_result["name"], "Bash")
        self.assertIs(tool_result["is_error"], False)
        self.assertEqual(tool_result["content"], [{"type": "text", "text": result}])
        return blob

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux credential-directory mount boundary')
    async def test_fresh_worker_covers_retained_storage_after_configuration_parent_moves(self):
        from loki_agent import credential_storages

        for relative, closed_stderr in [[False, False], [True, False],
                                        [False, True], [True, True]]:
            async with AsyncExitStack() as cleanup:
                cleanup.enter_context(self.subTest(relative=relative, closed_stderr=closed_stderr))
                root = cleanup.enter_context(tempfile.TemporaryDirectory())
                workspace = os.path.join(root, 'workspace')
                second_workspace = os.path.join(root, 'second-workspace')
                os.mkdir(workspace)
                os.mkdir(second_workspace)
                configuration = os.path.join(workspace if relative else root, 'config')
                moved = configuration + '-moved'
                directory = os.path.join(configuration, 'loki', 'credentials')
                storage = credential_storages.JsonCredentialStorage(directory)
                cleanup.callback(storage.close)
                await storage.store_openai_login(authentications.OpenAITokenSet(
                    access_token='directory-view-access', refresh_token='directory-view-refresh',
                    expires_at=10**12))
                current_name = shlex.quote(configuration)
                moved_name = shlex.quote(moved)
                replacement = shlex.quote(directory)
                original_file = os.path.join(moved, 'loki', 'credentials', 'tokens.json')
                command = (
                    f'if test ! -d {moved_name}; then '
                    f'mv -- {current_name} {moved_name} && mkdir -p -- {replacement} '
                    '&& printf FIRST_VIEW_OK; '
                    f'elif cat -- {shlex.quote(original_file)} >/dev/null 2>&1; then '
                    'printf WRONG_DIRECTORY_VIEW; else printf SECOND_VIEW_OK; fi')
                environment = {key: value for key, value in os.environ.items()
                               if not key.startswith('LOKI_')
                               and not any(key.endswith(suffix)
                                           for suffix in ['_KEY', '_TOKEN', '_PAT'])}
                environment.update({
                    'HOME': root, 'XDG_CONFIG_HOME': 'config' if relative else configuration,
                    'XDG_STATE_HOME': os.path.join(root, 'state'), 'TERM': 'dumb',
                    'LOKI_PROVIDER': 'dummy', 'LOKI_API_BASE': 'http://dummy.invalid/v1',
                    'LOKI_MODEL': 'dummy-model', 'LOKI_DUMMY_REPLY': 'Directory view checked.',
                    'LOKI_DUMMY_TOOL_CALL': json.dumps({'name': 'Bash', 'arguments': {
                        'command': command,
                        'description': 'Check the configured directory view'}}),
                })
                front, diagnostics = await self._front(
                    environment, workspace, closed_stderr=closed_stderr, cleanup=cleanup.push_async_callback)
                await self._request(front, 1, 'initialize', {'protocolVersion': 1})
                opened, _ = await self._request(front, 2, 'session/new', {'cwd': workspace})
                first_session = opened['sessionId']
                reply, updates = await self._request(front, 3, 'session/prompt', {
                    'sessionId': first_session, 'prompt': [{'type': 'text', 'text': 'Check the first view'}]})
                self.assertEqual(reply, {'stopReason': 'end_turn'})
                first_results = [update for update in _tool_updates(updates, first_session)
                                 if update['sessionUpdate'] == 'tool_call_update']
                self.assertEqual(len(first_results), 1)
                self.assertIn('FIRST_VIEW_OK', json.dumps(first_results))
                self.assertTrue(os.path.isdir(directory))
                # Publish after the rename through the same held object. A fresh
                # worker must hide these new bytes too, not just the old file.
                await storage.store_openai_login(authentications.OpenAITokenSet(
                    access_token='later-directory-access', refresh_token='later-directory-refresh',
                    expires_at=10**12))
                with open(original_file) as stream:
                    self.assertIn('later-directory-refresh', stream.read())
                self.assertFalse(os.path.exists(storage.file_path))
                opened, _ = await self._request(front, 4, 'session/new', {'cwd': second_workspace})
                second_session = opened['sessionId']
                self.assertNotEqual(first_session, second_session)
                reply, updates = await self._request(front, 5, 'session/prompt', {
                    'sessionId': second_session, 'prompt': [{'type': 'text', 'text': 'Check the fresh view'}]})
                self.assertEqual(reply, {'stopReason': 'end_turn'})
                results = [update for update in _tool_updates(updates, second_session)
                           if update['sessionUpdate'] == 'tool_call_update']
                self.assertEqual(len(results), 1)
                self.assertIn('SECOND_VIEW_OK', json.dumps(results))
                self.assertNotIn('WRONG_DIRECTORY_VIEW', json.dumps(results))
                await self._request(front, 6, 'session/close', {'sessionId': first_session})
                await self._finish(front, diagnostics, second_session)

    async def test_shipped_front_real_tool_updates_save_and_close(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        workspace, payload, command, result = _tool_fixture(root)
        args = {"command": command, "description": "Read the relative fixture"}
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("LOKI_")
                       and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
        environment.update({
            "HOME": root, "XDG_CONFIG_HOME": os.path.join(root, "config"),
            "XDG_STATE_HOME": os.path.join(root, "state"), "TERM": "dumb",
            "LOKI_PROVIDER": "dummy", "LOKI_API_BASE": "http://dummy.invalid/v1",
            "LOKI_MODEL": "dummy-model", "LOKI_DUMMY_REPLY": "Tool complete.",
            "LOKI_DUMMY_TOOL_CALL": json.dumps({"name": "Bash", "arguments": args}),
        })
        configure_container(environment, workspace)
        front, diagnostics = await self._front(environment, workspace, cleanup=self.addAsyncCleanup)
        await self._request(front, 1, "initialize", {"protocolVersion": 1})
        opened, _ = await self._request(front, 2, "session/new", {"cwd": workspace})
        session_id = opened["sessionId"]
        reply, messages = await self._request(front, 3, "session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "Run the real file proof"}],
        })
        self.assertEqual(reply, {"stopReason": "end_turn"})
        self.assertEqual(_tool_updates(messages, session_id),
                         _call_and_result("dummy-tool-call", command, workspace, result)
                         + _assistant_chunks("Tool complete."))
        # The append-only witness distinguishes skipped/fabricated dispatch
        # and duplicate execution; it is read before shutdown/backstops.
        with open(os.path.join(workspace, "executed.txt"), "rb") as stream:
            self.assertEqual(stream.read(), payload)
        await self._finish(front, diagnostics, session_id)
        path = os.path.join(app_paths.chat_log_dir_for(workspace), f"chat-{session_id}.json")
        self._saved_tool_turn(path, "Run the real file proof",
                              "dummy-tool-call", args, result, ["Tool complete."])

    async def test_resource_real_tool_continuation_save_and_fresh_replay(self):
        await self._resource_tool_continuation()

    async def test_resource_fixture_closes_state_after_configuration_failure(self):
        await self._resource_fixture_failure('loki_agent.loki.apply_runtime_config')

    async def test_resource_fixture_closes_state_after_prompt_failure(self):
        await self._resource_fixture_failure('loki_agent.acp_worker.Worker.prompt')

    async def _resource_fixture_failure(self, target):
        from loki_agent import sessions

        # Reuse the real resource journey and retain its Sessions beyond scope
        # exit: a destructor cannot supply false evidence of fixture cleanup.
        # Each failure is independently selectable and happens before Bash runs.
        original = sessions.Session
        held = []

        def create(*arguments, **keywords):
            session = original(*arguments, **keywords)
            held.append(session)
            return session

        try:
            with mock.patch.object(sessions, 'Session', side_effect=create), \
                    mock.patch(target, side_effect=RuntimeError('resource fixture injected failure')):
                with self.assertRaisesRegex(RuntimeError, 'resource fixture injected failure'):
                    await self._resource_tool_continuation()
            self.assertEqual(len(held), 1, 'failure must occur after Session state acquisition')
            for session in held:
                with self.assertRaisesRegex(ValueError, 'response status storage is closed'):
                    session.response_headers.directory_handle
        finally:
            # Release even a failing red control without relying on its GC.
            for session in held:
                await session.close_state()

    async def _resource_tool_continuation(self):
        import copy
        from contextlib import AsyncExitStack
        from pathlib import Path
        from loki_agent import formats, loki, paths, protocols, response_headers
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        # The fixture, not Worker.close(), owns each Session's retained state.
        # Register it before setup can fail; this outer stack joins state after
        # the worker backstop and before removing the workspace it refers to.
        async with AsyncExitStack() as stack:
            root = stack.enter_context(tempfile.TemporaryDirectory())
            workspace, payload, command, result = _tool_fixture(root)
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("LOKI_")
                           and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
            environment.update({"HOME": root, "XDG_CONFIG_HOME": os.path.join(root, "config"),
                                "XDG_STATE_HOME": os.path.join(root, "state")})
            stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
            credentials = CredentialStore({
                "LOKI_PROVIDER": "openai", "LOKI_API_BASE": "https://provider.invalid/v1",
                "LOKI_MODEL": "tool-model", "LOKI_STREAM": "1",
            })
            stack.enter_context(mock.patch.object(loki, "CREDENTIALS", credentials))
            state_root = paths.loki_state_dir(environment)
            session = Session(shell_cwd=workspace, response_headers=response_headers.Store(
                os.path.join(state_root, 'response-headers.json')))
            stack.push_async_callback(session.close_state)
            stack.enter_context(mock.patch.object(loki, "_DEFAULT_SESSION", session))
            stack.enter_context(mock.patch.object(loki, "LOKI_JOB_STATE_DIR", os.path.join(state_root, "jobs")))
            stack.enter_context(mock.patch.object(models, "ensure_index", new=mock.AsyncMock(return_value=({}, {}))))
            loki.apply_runtime_config(connections.build_config_from_env(credentials=credentials))
            messages = []
            session_id = "tool-workflow"
            worker = Worker(session, messages.append, session_id)
            workers = [worker]
            args = {"command": command, "description": "Read the relative fixture"}
            call_id = "provider-call-73"
            resource = {"name": "payload.txt", "uri": Path(workspace, "payload.txt").as_uri(),
                        "title": "Source \u03b1", "description": "A uniquely populated local file",
                        "mimeType": "text/plain", "size": len(payload)}
            user = "inspect this\n[ACP resource link]\n" + json.dumps(resource, ensure_ascii=False)
            requests = []

            async def completion(items, *, tools, **kwargs):
                self.assertIsInstance(tools, list)
                self.assertFalse(kwargs["cancel_check"]())
                requests.append(copy.deepcopy(items))
                self.assertLessEqual(len(requests), 2, "unexpected additional provider request")
                if len(requests) == 1:
                    for chunk in ["Inspecting ", "resource."]:
                        kwargs["on_text_delta"](chunk)
                    return formats.DecodedTurn([
                        formats.message_item("assistant", "Inspecting resource."),
                        formats.tool_call_item(call_id, "Bash", args),
                    ], {"protocol": protocols.OPENAI_CHAT})
                for chunk in ["Tool ", "complete."]:
                    kwargs["on_text_delta"](chunk)
                return formats.DecodedTurn([
                    formats.message_item("assistant", "Tool complete."),
                ], {"protocol": protocols.OPENAI_CHAT})

            stack.enter_context(mock.patch.object(loki, "async_chat_completion", new=completion))
            try:
                await worker.prepare_open({
                    "sessionId": session_id, "cwd": workspace, "openMethod": "session/new"})
                worker.commit_open()
                messages.clear()
                reply = await worker.prompt({
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "inspect this"},
                               {"type": "resource_link", **resource, "unadvertised": "must not leak"}],
                })
                self.assertEqual(reply, {"stopReason": "end_turn"})
                self.assertEqual(len(requests), 2)
                for request in requests:
                    self.assertEqual([item["content"] for item in request
                                      if item.get("role") == "user"],
                                     [[{"type": "text", "text": user}]])
                followup = requests[1]
                self.assertEqual([item["type"] for item in followup[-3:]],
                                 ["message", "model_response", "tool_result"])
                self.assertEqual(followup[-2]["items"][0], {
                    "type": "message", "role": "assistant",
                    "content": [{"type": "text", "text": "Inspecting resource."}],
                })
                continued_call = followup[-2]["items"][1]
                self.assertEqual(continued_call["call_id"], call_id)
                self.assertEqual(continued_call["name"], "Bash")
                self.assertEqual(json.loads(continued_call["arguments"]), args)
                self.assertEqual(len(followup[-2]["items"]), 2)
                self.assertEqual(followup[-1]["call_id"], call_id)
                self.assertEqual(followup[-1]["content"], [{"type": "text", "text": result}])
                self.assertIs(followup[-1]["is_error"], False)
                self.assertEqual(_tool_updates(messages, session_id),
                                 _assistant_chunks("Inspecting ", "resource.")
                                 + _call_and_result(call_id, command, workspace, result)
                                 + _assistant_chunks("Tool ", "complete."))
                with open(os.path.join(workspace, "executed.txt"), "rb") as stream:
                    self.assertEqual(stream.read(), payload)
                self.assertIsNotNone(session.job_manager)
                self.assertEqual(len(session.job_manager.jobs), 1)
                job = next(iter(session.job_manager.jobs.values()))
                self.assertEqual(job.cwd, workspace)
                self.assertEqual(job.status, "exited")
                self.assertEqual(job.exit_code, 0)
                self.assertEqual(job.process.returncode, 0)
                with open(job.stdout_path, encoding="utf-8") as stream:
                    self.assertEqual(stream.read(), payload.decode())
                with open(job.stderr_path, encoding="utf-8") as stream:
                    self.assertEqual(stream.read(), "")
                path = session.chat_log_path
                self.assertEqual(path, os.path.join(
                    app_paths.chat_log_dir_for(workspace), f'chat-{session_id}.json'))
                saved = self._saved_tool_turn(path, user, call_id, args, result,
                                              ["Inspecting resource.", "Tool complete."])
                await worker.close()
                restored_session = Session(shell_cwd=workspace)
                stack.push_async_callback(restored_session.close_state)
                loki._DEFAULT_SESSION = restored_session
                restored = Worker(restored_session, messages.append, session_id)
                workers.append(restored)
                messages.clear()
                await restored.prepare_open({
                    "sessionId": session_id, "cwd": workspace, "openMethod": "session/load",
                    "replay": False})
                self.assertEqual(restored_session.transcript_items, saved["events"])
                self.assertEqual(_tool_updates(messages, session_id), [])
                restored.commit_open()
                self.assertEqual(_tool_updates(messages, session_id), [
                    {"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": user}},
                    *_assistant_chunks("Inspecting resource."),
                    {"sessionUpdate": "tool_call", "toolCallId": call_id, "title": "Bash",
                     "kind": "other", "status": "completed"},
                    {"sessionUpdate": "tool_call", "toolCallId": call_id, "title": "Tool result: Bash",
                     "kind": "other", "status": "completed"},
                    *_assistant_chunks("Tool complete."),
                ])
                self.assertEqual(len(requests), 2, "load must not perform inference")
                with open(os.path.join(workspace, "executed.txt"), "rb") as stream:
                    self.assertEqual(stream.read(), payload, "load must not execute historical calls")
                await restored.close()
            finally:
                # Backstop only: owned job exit/output was asserted above.
                for owner in workers:
                    await owner.close()
        for owner in workers:
            with self.assertRaisesRegex(ValueError, 'response status storage is closed'):
                owner.session.response_headers.directory_handle


def _command_fixture(root):
    import base64
    workspace = os.path.join(root, "workspace")
    destination = os.path.join(workspace, "changed cwd")
    os.makedirs(destination)
    payload = b"ACP-W1 destination bytes"
    for directory, content in ((workspace, b"wrong initial cwd"), (destination, payload)):
        with open(os.path.join(directory, "payload.txt"), "wb") as stream:
            stream.write(content)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
    with open(os.path.join(destination, "shot.png"), "wb") as stream:
        stream.write(png)
    command = "cat payload.txt >> executed.txt && cat executed.txt"
    output = "status: completed\nexit_code: 0\n[stdout]\n" + payload.decode()
    model_text = f"I ran the local command `{command}`.\nOutput:\n```\n{output}\n```"
    image = {"type": "image", "source": {
        "type": "base64", "media_type": "image/png",
        "data": base64.b64encode(png).decode("ascii"),
    }}
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("LOKI_")
                   and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
    environment.update({
        "HOME": root, "XDG_CONFIG_HOME": os.path.join(root, "config"),
        "XDG_STATE_HOME": os.path.join(root, "state"), "TERM": "dumb",
        "LOKI_PROVIDER": "dummy", "LOKI_API_BASE": "http://dummy.invalid/v1",
        "LOKI_MODEL": "dummy-model", "LOKI_DUMMY_REPLY": "Local command answer.",
    })
    return workspace, destination, payload, png, command, output, model_text, image, environment


async def _command_status(environment, cleanup):
    from loki_agent import paths, response_headers
    path = os.path.join(paths.loki_state_dir(environment), "response-headers.json")
    store = response_headers.Store(path)
    cleanup(store.close)  # Save/observer/read failures occur before a caller receives the Store.
    store.observer("https://unrelated.example/v1/chat/completions", None, "observed-model")(
        200, {"x-remaining": "7"})
    await store.save()
    with open(path, "rb") as stream:
        original = stream.read()
    return store, path, original


class LocalCommandJourneyTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    async def test_command_status_closes_store_when_save_fails(self):
        from loki_agent import response_headers
        store_type = response_headers.Store
        acquired = []

        def acquire(*args, **kwargs):
            store = store_type(*args, **kwargs)
            acquired.append(store)
            return store

        with tempfile.TemporaryDirectory() as root:
            try:
                with ExitStack() as resources:
                    with mock.patch.object(response_headers, 'Store', side_effect=acquire), \
                            mock.patch.object(store_type, 'save', new=mock.AsyncMock(side_effect=OSError('save failed'))):
                        with self.assertRaisesRegex(OSError, 'save failed'):
                            await _command_status({'XDG_STATE_HOME': root}, resources.callback)
                self.assertEqual(len(acquired), 1)
                with self.assertRaisesRegex(ValueError, 'response status storage is closed'):
                    acquired[0].directory_handle
            finally:
                for store in acquired:
                    store.close()

    def _advertisement(self, commands):
        names = [command["name"] for command in commands]
        self.assertEqual(set(names), {"status", "account", "pwd", "cd", "image", "thinking", "trace"})
        self.assertEqual(len(names), 7)
        self.assertNotIn("ps", names)
        self.assertNotIn("todo", names)
        self.assertTrue(all(command["description"] for command in commands))
        self.assertNotIn("model", names)
        self.assertNotIn("effort", names)

    def _status(self, messages, session_id, original):
        updates = _tool_updates(messages, session_id)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["sessionUpdate"], "agent_message_chunk")
        document = json.loads(updates[0]["content"]["text"])
        self.assertEqual(document, json.loads(original))
        self.assertEqual(document["version"], 1)
        self.assertEqual(len(document["endpoints"]), 1)
        entry = document["endpoints"][0]
        self.assertEqual(entry["endpoint"], "https://unrelated.example/v1/chat/completions")
        self.assertEqual(entry["headers"]["x-remaining"]["value"], "7")
        self.assertEqual(entry["latest"]["model"], "observed-model")
        self.assertEqual(entry["latest"]["status"], 200)

    def _saved(self, path, destination, model_text, image, answers):
        from loki_agent import formats
        with open(path, encoding="utf-8") as stream:
            blob = json.load(stream)
        formats.validate_events(blob["events"])
        self.assertEqual(_conversation_pairs(blob), [
            ("user", [{"type": "text", "text": model_text}, image]),
            ("assistant", [{"type": "text", "text": answers[0]}]),
            ("user", [{"type": "text", "text": "Second prompt"}]),
            ("assistant", [{"type": "text", "text": answers[1]}]),
        ])
        start = next(i for i, event in enumerate(blob["events"]) if event.get("role") == "user")
        self.assertEqual([event["type"] for event in blob["events"][start:]],
                         ["message", "model_response", "message", "model_response"])
        self.assertEqual(blob["session_state"]["shell_cwd"], destination)

    async def test_shipped_commands_snapshot_bang_two_turns_and_close(self):
        import socket
        root = self.enterContext(tempfile.TemporaryDirectory())
        workspace, destination, payload, png, command, output, model_text, image, env = _command_fixture(root)
        store, status_path, status_bytes = await _command_status(env, self.addCleanup)
        configure_container(env, workspace)
        # The new virtual cwd is also a supported configured workspace on
        # Windows; no alternative launcher or containment route is invented.
        configure_container(env, destination)
        front, diagnostics = await self._front(env, workspace, cleanup=self.addAsyncCleanup)
        initialized, _ = await self._request(front, 1, "initialize", {"protocolVersion": 1})
        self.assertEqual(initialized["protocolVersion"], 1)
        self.assertEqual(initialized["agentInfo"]["name"], "loki")
        self.assertEqual(initialized["agentInfo"]["version"], __version__)
        for capability in ["close", "list", "resume"]:
            self.assertEqual(initialized["agentCapabilities"]["sessionCapabilities"][capability], {})
        self.assertIs(initialized["agentCapabilities"]["promptCapabilities"]["image"], False)
        opened, preceding = await self._request(front, 2, "session/new", {"cwd": workspace})
        session_id = opened["sessionId"]
        self.assertTrue(session_id)
        self.assertFalse(any(message.get("method") == "session/update" for message in preceding))
        advertised = json.loads(await front.stdout.readline())
        self.assertEqual(advertised["method"], "session/update")
        self.assertEqual(advertised["params"]["sessionId"], session_id)
        self.assertEqual(advertised["params"]["update"]["sessionUpdate"], "available_commands_update")
        self._advertisement(advertised["params"]["update"]["availableCommands"])
        plan = json.loads(await front.stdout.readline())
        self.assertEqual(plan, acps.notification("session/update", {
            "sessionId": session_id,
            "update": {"sessionUpdate": "plan", "entries": []},
        }))
        self.assertNotIn("lokiUpdates", opened)
        path = os.path.join(app_paths.chat_log_dir_for(workspace), f"chat-{session_id}.json")

        async def prompt(request_id, text):
            reply, messages = await self._request(front, request_id, "session/prompt", {
                "sessionId": session_id, "prompt": [{"type": "text", "text": text}],
            })
            self.assertEqual(reply, {"stopReason": "end_turn"})
            return messages

        for request_id, text, expected in (
                (3, "/pwd", f"cwd: {workspace}"),
                (4, f"/cd {destination}", f"cwd: {destination}"),
                (5, "/pwd", f"cwd: {destination}"),
                (6, "/image shot.png", f"Attached image for next prompt: {os.path.join(destination, 'shot.png')} (image/png, {len(png)} bytes)")):
            messages = await prompt(request_id, text)
            self.assertEqual(_tool_updates(messages, session_id), _assistant_chunks(expected))
        self._status(await prompt(7, "/status all --json"), session_id, status_bytes)
        self.assertFalse(os.path.exists(path), "local commands must not save a model turn")
        with open(status_path, "rb") as stream:
            self.assertEqual(stream.read(), status_bytes)
        with open(os.path.join(destination, "shot.png"), "wb") as stream:
            stream.write(b"\xff\xd8\xffreplacement source")
        messages = await prompt(8, "!" + command)
        acknowledgement = f"{socket.gethostname()}: [Running local command: {command}]\n{output}"
        self.assertEqual(_tool_updates(messages, session_id),
                         _assistant_chunks(acknowledgement, "Local command answer."))
        self.assertEqual(_tool_updates(await prompt(9, "Second prompt"), session_id),
                         _assistant_chunks("Local command answer."))
        with open(os.path.join(destination, "executed.txt"), "rb") as stream:
            self.assertEqual(stream.read(), payload)
        self.assertFalse(os.path.exists(os.path.join(workspace, "executed.txt")))
        listed, _ = await self._request(front, 88, 'session/list', {'cwd': workspace})
        self.assertEqual([entry['sessionId'] for entry in listed['sessions']], [session_id])
        self.assertEqual(listed['sessions'][0]['cwd'], destination)
        await self._finish(front, diagnostics, session_id)
        self._saved(path, destination, model_text, image,
                    ("Local command answer.", "Local command answer."))

    async def test_wire_commands_snapshot_bang_and_durable_two_turns(self):
        import copy
        import socket
        from contextlib import ExitStack
        from loki_agent import formats, loki, paths, protocols
        from loki_agent.sessions import Session
        async with AsyncExitStack() as resources:
            root = resources.enter_context(tempfile.TemporaryDirectory())
            stack = resources.enter_context(ExitStack())
            workspace, destination, payload, png, command, output, model_text, image, env = _command_fixture(root)
            store, status_path, status_bytes = await _command_status(env, resources.callback)
            state_root = paths.loki_state_dir(env)
            stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
            stack.enter_context(mock.patch.object(loki, "CREDENTIALS", CredentialStore(env)))
            stack.enter_context(mock.patch.object(loki, "LOKI_CONFIG_DIR", paths.loki_config_dir(env)))
            stack.enter_context(mock.patch.object(loki, "LOKI_JOB_STATE_DIR", os.path.join(state_root, "jobs")))
            # Start at the wrong cwd. Only the actual prepare wire request may
            # install the conversation's initial workspace.
            session = Session(shell_cwd=root, response_headers=store)
            # The fake owns Worker/transport retirement; this fixture owns Session state.
            resources.push_async_callback(session.close_state)
            stack.enter_context(mock.patch.object(loki, "_DEFAULT_SESSION", session))
            stack.enter_context(mock.patch.object(models, "ensure_index", new=mock.AsyncMock(return_value=({}, {}))))
            stack.enter_context(mock.patch.object(acp.runtime_isolation, "close_runtime_process", new=lambda process: None))
            loki.apply_runtime_config(connections.build_config_from_env(credentials=loki.CREDENTIALS))
            messages = []
            requests = []
            session_id = "local-commands"
            process = _LocalWorkerProcess(session, session_id, lambda message: None)
            channel = acp.WorkerChannel(session_id, process, messages.append)
            process_cwd = os.getcwd()

            async def completion(items, *, tools, **kwargs):
                self.assertIsInstance(tools, list)
                self.assertFalse(kwargs["cancel_check"]())
                requests.append(copy.deepcopy(items))
                self.assertLessEqual(len(requests), 2)
                answer = "Image accepted." if len(requests) == 1 else "Second answer."
                return formats.DecodedTurn(
                    [formats.message_item("assistant", answer)],
                    {"protocol": protocols.OPENAI_CHAT})

            stack.enter_context(mock.patch.object(loki, "async_chat_completion", new=completion))

            async def prompt(text):
                messages.clear()
                reply = await channel.request("session/prompt", {
                    "sessionId": session_id, "prompt": [{"type": "text", "text": text}],
                })
                self.assertEqual(reply, {"stopReason": "end_turn"})
                return list(messages)

            try:
                await channel.request("session/prepare_open", {
                    "sessionId": session_id, "cwd": workspace, "openMethod": "session/new"})
                opened = await channel.request("session/commit_open", {})
                self.assertEqual(session.shell_cwd, workspace)
                self._advertisement(opened["lokiUpdates"][0]["availableCommands"])
                self.assertEqual(opened["lokiUpdates"][1], {
                    "sessionUpdate": "plan", "entries": []})
                initial = copy.deepcopy(session.transcript_items)
                for text, expected in (
                        ("/pwd", f"cwd: {workspace}"),
                        (f"/cd {destination}", f"cwd: {destination}"),
                        ("/pwd", f"cwd: {destination}"),
                        ("/image shot.png", f"Attached image for next prompt: {os.path.join(destination, 'shot.png')} (image/png, {len(png)} bytes)")):
                    self.assertEqual(_tool_updates(await prompt(text), session_id), _assistant_chunks(expected))
                    self.assertEqual(requests, [])
                    if text.startswith("/cd "):
                        initial.append({
                            "type": "message", "role": "system", "content": [{
                                "type": "text",
                                "text": (
                                    f"Current Loki cwd changed to: {destination}. "
                                    "Relative tool paths and Bash commands now run from this directory."),
                            }],
                        })
                    self.assertEqual(session.transcript_items, initial)
                self._status(await prompt("/status all --json"), session_id, status_bytes)
                self.assertEqual(requests, [])
                self.assertEqual(session.transcript_items, initial)
                with open(status_path, "rb") as stream:
                    self.assertEqual(stream.read(), status_bytes)
                with open(os.path.join(destination, "shot.png"), "wb") as stream:
                    stream.write(b"\xff\xd8\xffreplacement source")
                acknowledgement = f"{socket.gethostname()}: [Running local command: {command}]\n{output}"
                self.assertEqual(_tool_updates(await prompt("!" + command), session_id),
                                 _assistant_chunks(acknowledgement, "Image accepted."))
                self.assertEqual([item["content"] for item in requests[0] if item.get("role") == "user"],
                                 [[{"type": "text", "text": model_text}, image]])
                self.assertEqual(_tool_updates(await prompt("Second prompt"), session_id),
                                 _assistant_chunks("Second answer."))
                self.assertEqual(len(requests), 2)
                self.assertEqual([item["content"] for item in requests[1] if item.get("role") == "user"],
                                 [[{"type": "text", "text": model_text}, image],
                                  [{"type": "text", "text": "Second prompt"}]])
                self.assertEqual(os.getcwd(), process_cwd)
                self.assertEqual(session.shell_cwd, destination)
                with open(os.path.join(destination, "executed.txt"), "rb") as stream:
                    self.assertEqual(stream.read(), payload)
                self.assertFalse(os.path.exists(os.path.join(workspace, "executed.txt")))
                self.assertEqual(len(session.job_manager.jobs), 1)
                job = next(iter(session.job_manager.jobs.values()))
                self.assertEqual(job.cwd, destination)
                self.assertEqual(job.status, "exited")
                self.assertEqual(job.exit_code, 0)
                self.assertEqual(job.process.returncode, 0)
                for path, expected in ((job.stdout_path, payload.decode()), (job.stderr_path, "")):
                    with open(path, encoding="utf-8") as stream:
                        self.assertEqual(stream.read(), expected)
                self._saved(session.chat_log_path, destination, model_text, image,
                            ("Image accepted.", "Second answer."))
                await channel.close()
                self.assertEqual(process.returncode, 0)
                self.assertTrue(process.close_task.done())
                self.assertIsNone(process.close_task.exception())
                self.assertTrue(channel._reader_task.done())
                self.assertFalse(channel._pending)
                self._saved(session.chat_log_path, destination, model_text, image,
                            ("Image accepted.", "Second answer."))
            finally:
                process.close()
                await process.wait()
                await channel.close()
        with self.assertRaisesRegex(ValueError, 'response status storage is closed'):
            store.directory_handle


class _LocalWorkerProcess:
    """Byte-transport seam only: real Worker and WorkerChannel, no OS child.

    Worker.handle generates every reply and update. EOF drives Worker.close;
    wait observes that operation, rather than supplying cleanup in assertions.
    """

    def __init__(self, session, session_id, before_request):
        from loki_agent.acp_worker import Worker
        self.before_request = before_request
        self.stdout = asyncio.StreamReader(limit=64 * 1024)
        self.stdin = self
        self.returncode = None
        self.buffer = bytearray()
        self.finished = asyncio.Event()
        self.close_task = None
        self.methods = []
        self.worker = Worker(session, lambda message: self.stdout.feed_data(
            (json.dumps(message) + "\n").encode()), session_id)

    def write(self, data):
        self.buffer.extend(data)

    async def drain(self):
        while b"\n" in self.buffer:
            line, _, rest = self.buffer.partition(b"\n")
            self.buffer = bytearray(rest)
            message = json.loads(line)
            self.methods.append(message["method"])
            self.before_request(message)
            # Match the worker entrypoint: prompt replies may be long-running,
            # while ordered config/cancel messages continue to be handled.
            await self.worker.handle(message, concurrent=True)

    def close(self):
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._finish())

    async def _finish(self):
        try:
            await self.worker.close()
            self.returncode = 0
        finally:
            self.stdout.feed_eof()
            self.finished.set()

    async def wait_closed(self):
        await self.wait()

    async def wait(self):
        await self.finished.wait()
        await asyncio.shield(self.close_task)
        return self.returncode


class WorkerPromptOwnershipTests(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def _channel(self, root, messages, completion):
        from loki_agent import formats, loki, response_headers
        from loki_agent.sessions import Session
        state_root = os.path.join(root, 'state')
        session = Session(
            response_headers=response_headers.Store(os.path.join(state_root, 'response-headers.json')),
            shell_cwd=root, chat_log_path=os.path.join(root, "chat.json"),
            transcript_items=[formats.instruction_item("system")])
        tasks_before = asyncio.all_tasks()
        async with AsyncExitStack() as stack:
            stack.push_async_callback(session.close_state)
            stack.enter_context(mock.patch.object(loki, "_DEFAULT_SESSION", session))
            stack.enter_context(mock.patch.object(loki, "CREDENTIALS", CredentialStore({})))
            stack.enter_context(mock.patch.object(loki, "current_model", return_value="model"))
            stack.enter_context(mock.patch.object(loki, "async_chat_completion", new=completion))
            stack.enter_context(mock.patch.object(loki, "LOKI_JOB_STATE_DIR", os.path.join(state_root, "jobs")))
            # The process seam owns byte transport, not an OS process.
            stack.enter_context(mock.patch.object(
                acp.runtime_isolation, "close_runtime_process", new=lambda process: None))
            process = _LocalWorkerProcess(session, "prompt-owner", lambda message: None)
            channel = acp.WorkerChannel("prompt-owner", process, messages.append)
            try:
                yield channel
                await channel.close()
                self.assertEqual(process.returncode, 0)
                self.assertTrue(process.close_task.done())
                self.assertIsNone(process.close_task.exception())
                self.assertFalse(channel._pending)
                self.assertTrue(channel._reader_task.done())
                self.assertFalse(asyncio.all_tasks() - tasks_before)
            finally:
                process.close()
                await process.wait()
                await channel.close()

    def _params(self, text):
        return {"sessionId": "prompt-owner", "prompt": [{"type": "text", "text": text}]}

    async def test_channel_closes_its_store_on_success(self):
        with tempfile.TemporaryDirectory() as root:
            async with self._channel(root, [], mock.AsyncMock()) as channel:
                storage = channel.process.worker.session.response_headers
            with self.assertRaisesRegex(ValueError, 'closed'):
                _ = storage.directory_handle

    async def test_channel_closes_its_store_on_body_failure(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, 'fixture body failed'):
                async with self._channel(root, [], mock.AsyncMock()) as channel:
                    storage = channel.process.worker.session.response_headers
                    raise RuntimeError('fixture body failed')
            with self.assertRaisesRegex(ValueError, 'closed'):
                _ = storage.directory_handle

    async def test_local_worker_wait_observes_cleanup_failure_without_cancelling_it(self):
        from loki_agent.sessions import Session
        session = Session()
        self.addCleanup(session.response_headers.close)
        process = _LocalWorkerProcess(session, 'failing-cleanup', lambda message: None)
        entered, release = asyncio.Event(), asyncio.Event()
        failure = OSError('fixture cleanup failed')

        async def close():
            entered.set()
            await release.wait()
            raise failure

        with mock.patch.object(process.worker, 'close', new=close):
            waiter = asyncio.create_task(process.wait())
            process.close()
            try:
                await observe_fixture(entered.wait(), process.close_task)
                waiter.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiter
                self.assertFalse(process.close_task.done())
                release.set()
                with self.assertRaises(OSError) as caught:
                    await process.wait()
                self.assertIs(caught.exception, failure)
                self.assertIsNone(process.returncode)
                self.assertTrue(process.stdout.at_eof())
                task = process.close_task
                process.close()
                self.assertIs(process.close_task, task)
            finally:
                release.set()
                waiter.cancel()
                await asyncio.gather(waiter, process.close_task, return_exceptions=True)

    async def test_todo_plan_is_visible_before_turn_end_and_survives_cancellation(self):
        from loki_agent import formats, protocols
        proposed = [{"content": "  Inspect the code  ", "status": "in_progress", "priority": "high"}]
        committed = [{"content": "Inspect the code", "status": "in_progress", "priority": "high"}]
        for cancel in [False, True]:
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as root:
                messages = []
                continuation = asyncio.Event()
                release = asyncio.Event()
                calls = 0

                async def completion(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return formats.DecodedTurn([
                            formats.tool_call_item("write-plan", "TodoWrite", {"todos": proposed}),
                        ], {"protocol": protocols.OPENAI_CHAT})
                    continuation.set()
                    await release.wait()
                    return formats.DecodedTurn([
                        formats.message_item("assistant", "Continued."),
                    ], {"protocol": protocols.OPENAI_CHAT})

                async with self._channel(root, messages, completion) as channel:
                    worker = channel.process.worker
                    pending = asyncio.create_task(channel.request(
                        "session/prompt", self._params("Plan the work")))
                    try:
                        await observe_fixture(continuation.wait(), pending)
                        # Also flushes preceding notifications through the byte
                        # channel, without relaxing ACP's active-prompt guard.
                        with self.assertRaisesRegex(acps.TransportError, "a prompt is already running"):
                            await channel.request("session/prompt", self._params("/pwd"))
                        self.assertFalse(pending.done())
                        plans = [message["params"] for message in messages
                                 if message.get("method") == "session/update"
                                 and message["params"]["update"]["sessionUpdate"] == "plan"]
                        self.assertEqual(plans, [{
                            "sessionId": "prompt-owner",
                            "update": {"sessionUpdate": "plan", "entries": committed},
                        }])
                        self.assertEqual(worker.session.session_todos, committed)
                        if cancel:
                            await channel.request("session/cancel", {})
                        release.set()
                        result = await pending
                        self.assertEqual(result, {"stopReason": "cancelled" if cancel else "end_turn"})
                        self.assertEqual(worker.session.session_todos, committed)
                        with open(worker.session.chat_log_path, encoding="utf-8") as stream:
                            saved = json.load(stream)
                        self.assertEqual(saved["session_todos"], committed)
                        self.assertFalse(any(event["type"] == "plan" for event in saved["events"]))
                    finally:
                        release.set()
                        await asyncio.gather(pending, return_exceptions=True)

    async def test_todo_plan_precedes_post_hook_failure_or_task_cancellation(self):
        from loki_agent import formats, loki, protocols, tool_runtime
        tasks = [{"content": "Committed before the hook", "status": "pending", "priority": "medium"}]
        for outcome in ["failure", "cancellation"]:
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as root:
                messages = []
                started = asyncio.Event()
                release = asyncio.Event()
                pipeline = tool_runtime.ToolHookPipeline()

                async def post_hook(invocation, result):
                    started.set()
                    await release.wait()
                    raise RuntimeError("post hook failed")

                pipeline.add_post("blocking-post", post_hook)
                completion = mock.AsyncMock(side_effect=[
                    formats.DecodedTurn([
                        formats.tool_call_item("write-plan", "TodoWrite", {"todos": tasks}),
                    ], {"protocol": protocols.OPENAI_CHAT}),
                    formats.DecodedTurn([formats.message_item("assistant", "Continued.")],
                                        {"protocol": protocols.OPENAI_CHAT}),
                ])
                with mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline):
                    async with self._channel(root, messages, completion) as channel:
                        worker = channel.process.worker
                        pending = asyncio.create_task(channel.request(
                            "session/prompt", self._params("Track the work")))
                        try:
                            await observe_fixture(started.wait(), pending)
                            with self.assertRaisesRegex(acps.TransportError, "a prompt is already running"):
                                await channel.request("session/prompt", self._params("/pwd"))
                            plans = [message["params"] for message in messages
                                     if message.get("method") == "session/update"
                                     and message["params"]["update"]["sessionUpdate"] == "plan"]
                            self.assertEqual(plans, [{
                                "sessionId": "prompt-owner",
                                "update": {"sessionUpdate": "plan", "entries": tasks},
                            }])
                            self.assertFalse(pending.done())
                            self.assertFalse(any(update["sessionUpdate"] == "tool_call_update"
                                                 for update in _tool_updates(messages, "prompt-owner")))
                            if outcome == "cancellation":
                                await channel.request("session/cancel", {})
                                worker._prompt_task.cancel()
                            else:
                                release.set()
                            result = await pending
                            self.assertEqual(result, {
                                "stopReason": "cancelled" if outcome == "cancellation" else "end_turn"})
                            self.assertEqual(worker.session.session_todos, tasks)
                            self.assertEqual(sum(message.get("method") == "session/update"
                                                 and message["params"]["update"]["sessionUpdate"] == "plan"
                                                 for message in messages), 1)
                        finally:
                            release.set()
                            await asyncio.gather(pending, return_exceptions=True)

    async def test_todo_plan_survives_a_failed_provider_continuation(self):
        from loki_agent import formats, protocols
        tasks = [{"content": "Retain this task", "status": "pending", "priority": "low"}]
        completion = mock.AsyncMock(side_effect=[
            formats.DecodedTurn([
                formats.tool_call_item("write-plan", "TodoWrite", {"todos": tasks}),
            ], {"protocol": protocols.OPENAI_CHAT}),
            RuntimeError("continuation failed"),
        ])
        messages = []
        with tempfile.TemporaryDirectory() as root:
            async with self._channel(root, messages, completion) as channel:
                with self.assertRaisesRegex(acps.TransportError, "continuation failed"):
                    await channel.request("session/prompt", self._params("Track this task"))
                plans = [message["params"] for message in messages
                         if message.get("method") == "session/update"
                         and message["params"]["update"]["sessionUpdate"] == "plan"]
                self.assertEqual(plans, [{
                    "sessionId": "prompt-owner", "update": {"sessionUpdate": "plan", "entries": tasks}}])
                worker = channel.process.worker
                self.assertEqual(worker.session.session_todos, tasks)
                with open(worker.session.chat_log_path, encoding="utf-8") as stream:
                    self.assertEqual(json.load(stream)["session_todos"], tasks)

    async def test_todo_reads_and_rejected_writes_do_not_publish_plans_but_clear_does(self):
        from loki_agent import formats, protocols
        original = [{"content": "Existing task", "status": "pending", "priority": "low"}]
        invalid = [
            {"content": "First", "status": "in_progress", "priority": "high"},
            {"content": "Second", "status": "in_progress", "priority": "low"},
        ]
        completion = mock.AsyncMock(side_effect=[
            formats.DecodedTurn([
                formats.tool_call_item("read-plan", "TodoRead", {}),
                formats.tool_call_item("rejected-plan", "TodoWrite", {
                    "todos": [{"content": "Missing priority", "status": "pending"}]}),
                formats.tool_call_item("invalid-plan", "TodoWrite", {"todos": invalid}),
                formats.tool_call_item("clear-plan", "TodoWrite", {"todos": []}),
            ], {"protocol": protocols.OPENAI_CHAT}),
            formats.DecodedTurn([formats.message_item("assistant", "Cleared.")],
                                {"protocol": protocols.OPENAI_CHAT}),
        ])
        messages = []
        with tempfile.TemporaryDirectory() as root:
            async with self._channel(root, messages, completion) as channel:
                worker = channel.process.worker
                worker.session.session_todos = original
                result = await channel.request("session/prompt", self._params("Inspect and clear the plan"))
                self.assertEqual(result, {"stopReason": "end_turn"})
                plans = [message["params"] for message in messages
                         if message.get("method") == "session/update"
                         and message["params"]["update"]["sessionUpdate"] == "plan"]
                self.assertEqual(plans, [{
                    "sessionId": "prompt-owner", "update": {"sessionUpdate": "plan", "entries": []}}])
                self.assertEqual(worker.session.session_todos, [])
                failed = [update["toolCallId"] for update in _tool_updates(messages, "prompt-owner")
                          if update["sessionUpdate"] == "tool_call_update" and update.get("status") == "failed"]
                self.assertEqual(failed, ["rejected-plan", "invalid-plan"])
                self.assertIn("Existing task", json.dumps(_tool_updates(messages, "prompt-owner")))
                completion.assert_awaited()

    async def test_wire_cancellation_before_prompt_task_starts_and_next_turn(self):
        from loki_agent import formats, protocols
        for text in ["ordinary prompt", "!echo should-not-run > executed.txt"]:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as root:
                messages = []
                completion = mock.AsyncMock(return_value=formats.DecodedTurn(
                    [formats.message_item("assistant", "next answer")],
                    {"protocol": protocols.OPENAI_CHAT}))
                async with self._channel(root, messages, completion) as channel:
                    process = channel.process
                    worker = process.worker

                    def before_request(message):
                        if message["method"] == "session/prompt":
                            # Both byte-framed requests are admitted before the
                            # reader yields to the new prompt task.
                            process.write((json.dumps(acps.request(
                                "cancel-before-start", "session/cancel", {})) + "\n").encode())
                            process.before_request = lambda message: None

                    process.before_request = before_request
                    reply = await channel.request("session/prompt", self._params(text))
                    self.assertEqual(reply, {"stopReason": "cancelled"})
                    self.assertEqual(completion.await_count, 0)
                    self.assertTrue(worker.cancel_event.is_set())
                    self.assertIsNone(worker.session.job_manager)
                    self.assertFalse(os.path.exists(os.path.join(root, "executed.txt")))
                    self.assertEqual(_tool_updates(messages, "prompt-owner"),
                                     _assistant_chunks("[turn cancelled by user]"))
                    with open(worker.session.chat_log_path, encoding="utf-8") as stream:
                        self.assertEqual(_conversation_pairs(json.load(stream)),
                                         [("user", [{"type": "text", "text": text}])])
                    messages.clear()
                    reply = await channel.request("session/prompt", self._params("next prompt"))
                    self.assertEqual(reply, {"stopReason": "end_turn"})
                    self.assertFalse(worker.cancel_event.is_set())
                    self.assertEqual(completion.await_count, 1)
                    self.assertEqual(_tool_updates(messages, "prompt-owner"),
                                     _assistant_chunks("next answer"))
                    with open(worker.session.chat_log_path, encoding="utf-8") as stream:
                        self.assertEqual(_conversation_pairs(json.load(stream)), [
                            ("user", [{"type": "text", "text": text}]),
                            ("user", [{"type": "text", "text": "next prompt"}]),
                            ("assistant", [{"type": "text", "text": "next answer"}]),
                        ])

    async def test_local_commands_do_not_reset_cancellation_or_enter_history(self):
        import copy
        messages = []
        completion = mock.AsyncMock(side_effect=AssertionError("local command requested inference"))
        with tempfile.TemporaryDirectory() as root:
            async with self._channel(root, messages, completion) as channel:
                worker = channel.process.worker
                initial = copy.deepcopy(worker.session.transcript_items)
                worker.cancel_event.set()
                for text in ["/status --json", "/pwd"]:
                    messages.clear()
                    reply = await channel.request("session/prompt", self._params(text))
                    self.assertEqual(reply, {"stopReason": "end_turn"})
                    self.assertTrue(worker.cancel_event.is_set())
                    self.assertEqual(worker.session.transcript_items, initial)
                    self.assertEqual(len(_tool_updates(messages, "prompt-owner")), 1)
                self.assertEqual(await worker.prompt(self._params("/pwd")),
                                 {"stopReason": "end_turn"})
                self.assertTrue(worker.cancel_event.is_set())
                self.assertIsNone(worker._prompt_task)
                self.assertFalse(os.path.exists(worker.session.chat_log_path))
                completion.assert_not_awaited()

    async def test_ps_is_ordinary_text_without_local_control_or_fake_tool_updates(self):
        from loki_agent import formats, loki, protocols
        messages = []
        completion = mock.AsyncMock(return_value=formats.DecodedTurn(
            [formats.message_item("assistant", "ordinary text response")],
            {"protocol": protocols.OPENAI_CHAT}))
        with tempfile.TemporaryDirectory() as root:
            async with self._channel(root, messages, completion) as channel:
                worker = channel.process.worker
                with mock.patch.object(loki, "run_ps") as run_ps:
                    result = await channel.request(
                        "session/prompt", self._params("/ps all"))
                run_ps.assert_not_called()
                self.assertEqual(result, {"stopReason": "end_turn"})
                completion.assert_awaited_once()
                self.assertEqual(_tool_updates(messages, "prompt-owner"),
                                 _assistant_chunks("ordinary text response"))
                with open(worker.session.chat_log_path, encoding="utf-8") as stream:
                    self.assertEqual(_conversation_pairs(json.load(stream)), [
                        ("user", [{"type": "text", "text": "/ps all"}]),
                        ("assistant", [{"type": "text", "text": "ordinary text response"}]),
                    ])

    async def test_close_joins_prompt_and_session_owns_final_retirement(self):
        from loki_agent import loki
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        session = Session()
        self.addAsyncCleanup(session.close_state)
        with mock.patch.object(loki, '_DEFAULT_SESSION', session):
            manager = loki.current_job_manager()
        worker = Worker(session, lambda message: None)
        started, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        finished = []

        async def prompt():
            started.set()
            try:
                await asyncio.Future()
            finally:
                entered.set()
                await release.wait()
                finished.append(True)

        worker._prompt_task = asyncio.create_task(prompt())
        first = second = None
        try:
            await observe_fixture(started.wait(), worker._prompt_task)
            with mock.patch.object(manager, 'close_session_owned', wraps=manager.close_session_owned) as retire:
                first = asyncio.create_task(worker.close())
                second = asyncio.create_task(worker.close())
                await observe_fixture(entered.wait(), first, second)
                self.assertFalse(session.active)
                self.assertIsNotNone(session.response_headers.directory_handle)
                retire.assert_not_awaited()
                with self.assertRaisesRegex(acps.TransportError, 'conversation has closed'):
                    await worker.prompt(self._params('/pwd'))
                for _ in range(2):
                    first.cancel()
                    await asyncio.sleep(0)
                self.assertFalse(first.done())
                self.assertFalse(second.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                await second
                self.assertEqual(finished, [True])
                retire.assert_not_awaited()
                await session.close_state()
                retire.assert_awaited_once()
                with self.assertRaisesRegex(ValueError, 'closed'):
                    _ = session.response_headers.directory_handle
        finally:
            release.set()
            tasks = [task for task in [first, second, worker._prompt_task] if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_all_prompt_entry_paths_preserve_active_owner_and_updates(self):
        import copy
        from loki_agent import formats, loki, protocols
        from loki_agent.lifetimes import finish_cleanup
        for origin in ["wire", "inline"]:
            with self.subTest(origin=origin), tempfile.TemporaryDirectory() as root:
                entered = asyncio.Event()
                release = asyncio.Event()
                closing = asyncio.Event()
                messages = []

                async def completion(*args, **kwargs):
                    entered.set()
                    try:
                        await worker.cancel_event.wait()
                        await release.wait()
                    finally:
                        closing.set()
                        await finish_cleanup(release.wait())
                    return formats.DecodedTurn(
                        [formats.message_item("assistant", "must not appear")],
                        {"protocol": protocols.OPENAI_CHAT})

                async with self._channel(root, messages, completion) as channel:
                    worker = channel.process.worker
                    active = asyncio.create_task(
                        channel.request("session/prompt", self._params("active"))
                        if origin == "wire" else worker.prompt(self._params("active")))
                    close = None
                    try:
                        await observe_fixture(entered.wait(), active)
                        owner = worker._prompt_task
                        self.assertIsNotNone(owner)
                        initial = copy.deepcopy(worker.session.transcript_items)
                        await channel.request("session/cancel", {})
                        for text in ["/ps", "/ps stop 1", "/status", "second prompt"]:
                            with self.assertRaisesRegex(acps.TransportError, "already running"):
                                await channel.request("session/prompt", self._params(text))
                            with self.assertRaisesRegex(acps.TransportError, "already running"):
                                await worker.prompt(self._params(text))
                            replies = []
                            write = worker.write

                            def record(message):
                                replies.append(message)
                                write(message)

                            with mock.patch.object(worker, "write", new=record):
                                await worker.handle(acps.request(
                                    "inline-rejected", "session/prompt", self._params(text)))
                            self.assertEqual(replies, [acps.response("inline-rejected", error={
                                "code": acps.INVALID_PARAMS,
                                "message": "a prompt is already running for this session",
                            })])
                            self.assertIs(worker._prompt_task, owner)
                            self.assertFalse(owner.done())
                            self.assertTrue(worker.cancel_event.is_set())
                            self.assertEqual(worker.session.transcript_items, initial)
                            self.assertEqual(messages, [])
                        # Hold the actual prompt's cancellation cleanup, not an
                        # injected timer. Closing must revoke admission before
                        # this cleanup and cannot start a replacement end hook.
                        pipeline = mock.Mock(
                            turn_end_hooks=[object()], finish_turn=mock.AsyncMock())
                        with mock.patch.object(loki, 'TOOL_HOOK_PIPELINE', pipeline):
                            close = asyncio.create_task(worker.close())
                            await observe_fixture(closing.wait(), close)
                            self.assertFalse(worker.session.active)
                            self.assertFalse(close.done())
                            release.set()
                            self.assertEqual(await active,
                                             {"stopReason": "cancelled"})
                            await close
                            pipeline.finish_turn.assert_not_awaited()
                        await asyncio.sleep(0)
                        self.assertEqual(_tool_updates(messages, "prompt-owner"),
                                         _assistant_chunks("[turn cancelled by user]"))
                    finally:
                        worker.cancel_event.set()
                        release.set()
                        owners = [active] if close is None else [active, close]
                        await asyncio.gather(*owners, return_exceptions=True)

    async def test_bang_cancellation_reaps_command_without_requesting_inference(self):
        import shlex
        from process_lifecycle_fixtures import ProcessResources
        messages = []
        completion = mock.AsyncMock(side_effect=AssertionError("cancelled bang requested inference"))
        command_parts = [sys.executable, "-c",
                         "import threading; print('bang-ready', flush=True); threading.Event().wait()"]
        command = shlex.join(command_parts)
        with tempfile.TemporaryDirectory() as root:
            async with self._channel(root, messages, completion) as channel:
                worker = channel.process.worker
                active = asyncio.create_task(channel.request(
                    "session/prompt", self._params("!" + command)))
                resources = None
                try:
                    async def ready_job():
                        while True:
                            manager = worker.session.job_manager
                            job = next(iter(manager.jobs.values()), None) if manager is not None else None
                            if job is not None:
                                self.assertNotEqual(job.status, 'failed', job.error)
                                if job.process is not None:
                                    self.assertIsNone(job.process.returncode)
                                    if "bang-ready" in manager._spool_tail(job, 'stdout.log'):
                                        return job
                            await asyncio.sleep(.01)

                    job = await observe_fixture(ready_job(), active)
                    resources = ProcessResources(job.process)
                    await channel.request("session/cancel", {})
                    self.assertEqual(await active, {"stopReason": "cancelled"})
                    self.assertEqual(job.status, "cancelled")
                    self.assertIsNotNone(job.process.returncode)
                    with open(job.metadata_path, encoding="utf-8") as stream:
                        self.assertEqual(json.load(stream)["status"], "cancelled")
                    completion.assert_not_awaited()
                    updates = _tool_updates(messages, "prompt-owner")
                    self.assertEqual(updates[-1], _assistant_chunks("[turn cancelled by user]")[0])
                    self.assertIn("bang-ready", updates[0]["content"]["text"])
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    resources.assert_released(self)
                finally:
                    if resources is not None:
                        if job.process.returncode is None:
                            job_runtime.host_process.signal_group(
                                job.process, job.pgid, job_runtime.host_process.FORCE)
                            await job.process.wait()
                        await resources.cleanup()
                    if not active.done():
                        active.cancel()
                    await asyncio.gather(active, return_exceptions=True)


class SavedSessionJourneyTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_restore_continues_and_saves(self):
        from contextlib import ExitStack
        from test_http_client import FakeConnector
        from loki_agent import formats, http_client, loki
        from loki_agent.sessions import Session

        for explicit in [False, True]:
            for method in acp.RESTORE_METHODS:
                # The fake owns transport/Worker retirement, not Session state;
                # own state even when launch fails before entering its backstop.
                session_stores = []
                async with AsyncExitStack() as resources:
                    resources.enter_context(self.subTest(explicit=explicit, method=method))
                    root = resources.enter_context(tempfile.TemporaryDirectory())
                    stack = resources.enter_context(ExitStack())
                    workspace = os.path.join(root, 'workspace with spaces')
                    os.mkdir(workspace)
                    environment = {key: value for key, value in os.environ.items()
                                   if not key.startswith("LOKI_")
                                   and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
                    environment.update({"HOME": root, "XDG_CONFIG_HOME": os.path.join(root, "config"),
                                        "XDG_STATE_HOME": os.path.join(root, "state")})
                    stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
                    stack.enter_context(mock.patch.object(models, "ensure_index", new=mock.AsyncMock(return_value=({}, {}))))
                    responses = []
                    for answer in ["original answer", "continued answer"]:
                        body = json.dumps({
                            "id": answer,
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": answer},
                                "finish_reason": "stop",
                            }],
                        }).encode()
                        responses.append(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                                         + str(len(body)).encode() + b"\r\n\r\n" + body)
                    connector = FakeConnector(responses)
                    open_connection = asyncio.open_connection

                    async def connect(*args, **kwargs):
                        if "sock" in kwargs:
                            return await open_connection(*args, **kwargs)
                        return await connector.open_connection(*args, **kwargs)

                    stack.enter_context(mock.patch.object(
                        http_client.asyncio, "open_connection", new=connect))
                    base_session = Session(shell_cwd=workspace)
                    resources.push_async_callback(base_session.close_state)
                    session_stores.append(base_session.response_headers)
                    stack.enter_context(mock.patch.object(loki, "_DEFAULT_SESSION", base_session))
                    startup = {"LOKI_PROVIDER": "openai", "LOKI_API_BASE": "https://saved.example/v1",
                               "LOKI_MODEL": "saved-model", "LOKI_STREAM": "0"}
                    stack.enter_context(mock.patch.object(loki, "CREDENTIALS", CredentialStore(startup)))
                    processes = []
                    delegations = []
                    channels = []
                    messages = []
                    changed = asyncio.Event()
                    front = None
                    sources = []

                    def write(message):
                        messages.append(message)
                        changed.set()

                    async def launch(cwd, environment, delegation, *, credential_directory=None):
                        session = Session(shell_cwd=cwd)
                        resources.push_async_callback(session.close_state)
                        session_stores.append(session.response_headers)
                        session.credential_authority = front.credential_broker
                        loki._DEFAULT_SESSION = session
                        # Match the worker entrypoint's explicit startup config;
                        # prepare_open itself deliberately does not invent it.
                        if connections.explicit_connection_option(loki.CREDENTIALS):
                            loki.apply_runtime_config(connections.build_config_from_env(
                                credentials=loki.CREDENTIALS))

                        def before_request(message):
                            if message["method"] == "session/commit_open":
                                self.assertNotIn(process.worker.session_id, front.workers)

                        process = _LocalWorkerProcess(session, "provisional", before_request)
                        processes.append(process)
                        delegations.append(delegation)
                        return process

                    channel_type = acp.WorkerChannel

                    def channel(*args, **kwargs):
                        result = channel_type(*args, **kwargs)
                        channels.append(result)
                        return result

                    stack.enter_context(mock.patch.object(acp.runtime_isolation, "start_worker", new=launch))
                    stack.enter_context(mock.patch.object(acp, "WorkerChannel", new=channel))
                    # OS process release is outside this byte-transport lane.
                    stack.enter_context(mock.patch.object(acp.runtime_isolation, "close_runtime_process", new=lambda process: None))

                    async def backstop():
                        for source in sources:
                            await source.close()
                        for process in processes:
                            process.close()
                            await process.wait()
                        for channel in channels:
                            await channel.close()
                        if front is not None:
                            for task in list(front._tasks):
                                task.cancel()
                            await asyncio.gather(*front._tasks, return_exceptions=True)

                    # Runs before restoring patched globals, including on failure.
                    async def changed_while_running():
                        # A preceding request may finish normally while close/open
                        # still runs. Watch the active group, not any one completed
                        # request; retiring this aggregate cancels only its shields.
                        operations = [task for task in front._tasks if not task.done()]
                        self.assertTrue(operations, 'request ended without its expected update')
                        finished = asyncio.gather(*[asyncio.shield(task) for task in operations])
                        try:
                            await observe_fixture(changed.wait(), source.task, finished)
                        finally:
                            finished.cancel()
                            await asyncio.gather(finished, return_exceptions=True)

                    async def response(request_id):
                        while True:
                            changed.clear()
                            for message in messages:
                                if message.get("id") == request_id:
                                    self.assertNotIn("error", message, message)
                                    return message["result"]
                            await changed_while_running()

                    async def request(request_id, operation, params):
                        await source.send(acps.request(request_id, operation, params))
                        return await response(request_id)

                    try:
                        front = acp.Front(lambda: None, write, loki.CREDENTIALS)
                        source = _FrontInput(front)
                        sources.append(source)
                        await source.send(acps.request(0, 'initialize', {
                            'clientCapabilities': {'elicitation': {'form': {}}}}))
                        opened = await request(1, "session/new", {"cwd": workspace})
                        session_id = opened["sessionId"]
                        result = await request(2, "session/prompt", {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "original prompt"}]})
                        self.assertEqual(result["stopReason"], "end_turn")
                        path = processes[0].worker.session.chat_log_path
                        self.assertEqual(await request(3, "session/close", {"sessionId": session_id}), {})
                        await source.finish()
                        self.assertEqual(processes[0].returncode, 0)
                        self.assertTrue(channels[0]._reader_task.done())
                        with open(path, "rb") as stream:
                            original = stream.read()
                        self.assertEqual(_conversation_pairs(json.loads(original)),
                                         _expected_pairs(("original prompt", "original answer")))
                        credentials = CredentialStore({**startup, "LOKI_API_BASE": "https://explicit.example/v1",
                                                       "LOKI_MODEL": "explicit-model"} if explicit else {})
                        loki.CREDENTIALS = credentials
                        front = acp.Front(lambda: None, write, credentials)
                        source = _FrontInput(front)
                        sources.append(source)
                        await source.send(acps.request(0, 'initialize', {
                            'clientCapabilities': {'elicitation': {'form': {}}}}))
                        messages.clear()
                        await source.send(acps.request(73, method, {
                            "sessionId": session_id, "cwd": workspace,
                            "replay": method == "session/resume"}))
                        if not explicit:
                            while not any(m.get("method") == "elicitation/create" for m in messages):
                                changed.clear()
                                self.assertFalse(any(m.get('id') == 73 for m in messages),
                                                 f'restore replied before authorization: {messages!r}')
                                await changed_while_running()
                            elicitation = next(m for m in messages if m.get("method") == "elicitation/create")
                            params = elicitation["params"]
                            self.assertEqual(params["requestId"], 73)
                            self.assertEqual(params["mode"], "form")
                            self.assertIs(params["requestedSchema"]["properties"]["authorize"]["default"], False)
                            self.assertIn('"https://saved.example/v1/chat/completions"', params["message"])
                            self.assertIn("Working directory:", params["message"])
                            self.assertIn(json.dumps(workspace, ensure_ascii=True), params["message"])
                            self.assertNotIn(session_id, front.workers)
                            self.assertEqual(front._sessions[session_id].state, "opening")
                            self.assertEqual(processes[-1].methods, ["session/prepare_open"])
                            self.assertIsNone(processes[-1].worker.session.runtime_config)
                            self.assertIsNotNone(processes[-1].worker._pending_open)
                            self.assertEqual(len(connector.writers), 1)
                            self.assertEqual(_historical_chunks(messages, session_id), [])
                            with open(path, "rb") as stream:
                                self.assertEqual(stream.read(), original)
                            await source.send(acps.response(elicitation["id"], result={
                                "action": "accept", "content": {"authorize": True}}))
                        restored = await response(73)
                        self.assertNotIn("sessionId", restored)
                        self.assertIn("configOptions", restored)
                        self.assertIn(session_id, front.workers)
                        self.assertTrue(all(owner.state != "opening"
                                            for owner in front._sessions.values()))
                        self.assertFalse(front._client_requests)
                        self.assertEqual(processes[-1].methods, ["session/prepare_open", "session/commit_open"])
                        if explicit:
                            self.assertFalse(any(m.get("method") == "elicitation/create" for m in messages))
                        expected = [("user_message_chunk", {"type": "text", "text": "original prompt"}),
                                    ("agent_message_chunk", {"type": "text", "text": "original answer"})]
                        self.assertEqual(_historical_chunks(messages, session_id), expected if method == "session/load" else [])
                        continued_start = len(messages)
                        result = await request(74, "session/prompt", {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "continued prompt"}]})
                        self.assertEqual(result["stopReason"], "end_turn")
                        self.assertEqual(_historical_chunks(messages[continued_start:], session_id), [
                            ("agent_message_chunk", {"type": "text", "text": "continued answer"})])
                        self.assertEqual(len(connector.writers), 2)
                        self.assertEqual(connector.responses, [])
                        self.assertEqual([call["host"] for call in connector.calls],
                                         ["saved.example", "explicit.example" if explicit else "saved.example"])
                        packet = bytes(connector.writers[-1].data)
                        headers, body = packet.split(b"\r\n\r\n", 1)
                        host = b"explicit.example" if explicit else b"saved.example"
                        self.assertIn(b"POST /v1/chat/completions HTTP/1.1\r\n", headers)
                        self.assertIn(b"Host: " + host, headers)
                        payload = json.loads(body)
                        self.assertEqual(payload["model"], "explicit-model" if explicit else "saved-model")
                        self.assertEqual([(m["role"], m["content"]) for m in payload["messages"]
                                          if m["role"] in ["user", "assistant"]],
                                         [("user", "original prompt"), ("assistant", "original answer"),
                                          ("user", "continued prompt")])
                        for writer in connector.writers:
                            self.assertTrue(writer.closed)
                            self.assertTrue(writer.wait_closed_called)
                        self.assertEqual(await request(75, "session/close", {"sessionId": session_id}), {})
                        await source.finish()
                        self.assertFalse(front.workers)
                        self.assertEqual(processes[-1].returncode, 0)
                        self.assertTrue(processes[-1].close_task.done())
                        self.assertIsNone(processes[-1].close_task.exception())
                        self.assertTrue(channels[-1]._reader_task.done())
                        for delegation in delegations:
                            self.assertIsNone(delegation.owner_parent)
                            self.assertIsNone(delegation.owner_child)
                            self.assertIsNone(delegation.credential_child)
                            server = delegation.credential_server
                            self.assertTrue(server._reader_task.done())
                            self.assertTrue(server._close_task.done())
                            self.assertIsNone(server._close_task.exception())
                        self.assertIsNone(channels[-1].credential_delegation)
                        self.assertFalse(channels[-1]._pending)
                        with open(path, encoding="utf-8") as stream:
                            blob = json.load(stream)
                        formats.validate_events(blob["events"])
                        self.assertEqual(_conversation_pairs(blob), _expected_pairs(
                            ("original prompt", "original answer"), ("continued prompt", "continued answer")))
                    finally:
                        await backstop()
                for store in session_stores:
                    with self.assertRaisesRegex(ValueError, 'response status storage is closed'):
                        store.directory_handle


class ConfigOptionTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    async def test_session_new_returns_model_options(self):
        # The dummy env has no usable catalog credentials, so the option
        # list carries only the explicit LOKI_* connection -- which is
        # enough to prove the option plumbing flows and is settable.
        async with AsyncExitStack() as cleanup:
            tmpdir = cleanup.enter_context(tempfile.TemporaryDirectory())
            env = dict(os.environ)
            env.update({
                "HOME": tmpdir,
                "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
                "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
                "TERM": "dumb",
                "LOKI_PROVIDER": "dummy",
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_MODEL": "dummy-model",
                "LOKI_DUMMY_REPLY": "x",
            })
            workspace = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace, exist_ok=True)
            configure_container(env, workspace)
            front, _diagnostics = await self._front(
                env, workspace, cleanup=cleanup.push_async_callback)
            await self._request(front, 1, "initialize", {"protocolVersion": 1})
            opened, _ = await self._request(front, 2, "session/new", {"cwd": workspace})
            session_id = opened["sessionId"]
            options = opened["configOptions"]
            self.assertEqual([option["id"] for option in options], ["model", "reasoning_traces"])
            model_option = options[0]
            self.assertEqual(model_option["id"], "model")
            self.assertEqual(model_option["category"], "model")
            self.assertEqual(model_option["type"], "select")
            self.assertEqual(model_option["currentValue"], "loki-explicit")
            values = [o["value"] for o in model_option["options"]]
            self.assertEqual(values, ["loki-explicit"])

            # Setting the value round-trips and echoes full state.
            changed, _ = await self._request(front, 3, "session/set_config_option", {
                "sessionId": session_id, "configId": "model", "value": "loki-explicit"})
            self.assertIn("configOptions", changed)
            self.assertEqual(changed["configOptions"][0]["currentValue"], "loki-explicit")


class WorkerReasoningConfigTests(unittest.TestCase):
    @staticmethod
    def _profile(*values):
        return models.ReasoningEffortProfile(list(values))

    def test_model_changes_return_dependent_agent_shell_option(self):
        from loki_agent import loki
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as cleanup:
                session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                cleanup.callback(session.response_headers.close)
                cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', session))
                cleanup.enter_context(mock.patch.object(loki, 'CREDENTIALS', CredentialStore({
                    "OPENROUTER_API_KEY": "secret",
                })))
                provider = {
                    "id": "openrouter",
                    "name": "OpenRouter",
                    "npm": "@openrouter/ai-sdk-provider",
                    "env": ["OPENROUTER_API_KEY"],
                    "api": "https://openrouter.ai/api/v1",
                }
                high_model = {
                    "id": "high-model",
                    "name": "High model",
                    "reasoning_options": [{
                        "type": "effort",
                        "values": ["low", "high"],
                    }],
                }
                low_model = {
                    "id": "low-model",
                    "name": "Low model",
                    "reasoning_options": [{
                        "type": "effort",
                        "values": ["low"],
                    }],
                }
                no_effort_model = {
                    "id": "plain-model",
                    "name": "Plain model",
                }
                loki.apply_runtime_config(
                    connections.config_from_modelsdev_selection(
                        "openrouter",
                        provider,
                        high_model,
                        loki.CREDENTIALS,
                    ))
                loki.new_chat_log(
                    os.path.join(tmpdir, "chat-reasoning.json"))
                worker = Worker(session, lambda _message: None, "session")
                worker._set_choices([
                    ({
                        "value": "openrouter/high-model",
                        "name": "High model",
                    }, ("openrouter", provider, high_model)),
                    ({
                        "value": "openrouter/low-model",
                        "name": "Low model",
                    }, ("openrouter", provider, low_model)),
                    ({
                        "value": "openrouter/plain-model",
                        "name": "Plain model",
                    }, ("openrouter", provider, no_effort_model)),
                ], "openrouter/high-model")

                initial = worker.config_options()
                thought = initial[1]
                self.assertEqual(
                    [option["category"] for option in initial],
                    ["model", "thought_level", "other"],
                )
                self.assertEqual(thought["id"], "reasoning_effort")
                self.assertEqual(thought["currentValue"], "default")
                self.assertEqual(
                    [value["value"] for value in thought["options"]],
                    ["default", "effort:low", "effort:high"],
                )

                selected = worker.set_config_option({
                    "configId": "reasoning_effort",
                    "value": "effort:high",
                })
                self.assertEqual(
                    selected["configOptions"][1]["currentValue"],
                    "effort:high")

                plain = worker.set_config_option({
                    "configId": "model",
                    "value": "openrouter/plain-model",
                })
                self.assertEqual([option["id"] for option in plain["configOptions"]],
                                 ["model", "reasoning_traces"])
                self.assertEqual(
                    loki.current_reasoning_effort_preference(), "high")
                restored = worker.set_config_option({
                    "configId": "model",
                    "value": "openrouter/high-model",
                })
                self.assertEqual(
                    restored["configOptions"][1]["currentValue"],
                    "effort:high")

                narrowed = worker.set_config_option({
                    "configId": "model",
                    "value": "openrouter/low-model",
                })
                narrowed_thought = narrowed["configOptions"][1]
                self.assertEqual(
                    narrowed_thought["currentValue"], "default")
                self.assertEqual(
                    [value["value"]
                     for value in narrowed_thought["options"]],
                    ["default", "effort:low"],
                )
                self.assertIn(
                    "preferred high is unavailable",
                    narrowed_thought["options"][0]["name"],
                )
                self.assertEqual(
                    loki.current_reasoning_effort_preference(), "high")
                self.assertIsNone(loki.effective_reasoning_effort())
                with self.assertRaises(acps.TransportError) as caught:
                    worker.set_config_option({
                        "configId": "reasoning_effort",
                        "value": "effort:max",
                    })
                self.assertEqual(
                    caught.exception.code, acps.INVALID_PARAMS)
                self.assertEqual(
                    loki.current_reasoning_effort_preference(), "high")

                restored = worker.set_config_option({
                    "configId": "model",
                    "value": "openrouter/high-model",
                })
                self.assertEqual(
                    restored["configOptions"][1]["currentValue"],
                    "effort:high")
                self.assertEqual(loki.effective_reasoning_effort(), "high")

    def test_effort_change_during_prompt_applies_to_next_turn(self):
        from loki_agent import formats, loki, protocols
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            target = os.path.join(directory, 'read.txt')
            with open(target, 'w', encoding='utf-8') as stream:
                stream.write('real tool continuation')
            session = Session(shell_cwd=directory)
            cleanup.callback(session.response_headers.close)
            session.chat_log_path = os.path.join(directory, 'chat.json')
            session.runtime_config = connections.make_runtime_config(
                'https://api.openai.com/v1/responses',
                protocols.OPENAI_RESPONSES, model='gpt-test',
                provider_id='openai',
                reasoning_effort_profile=self._profile('low', 'high'))
            session.reasoning_effort_preference = 'high'
            session.session_state = {'reasoning_effort': 'high'}
            written, snapshots = [], []
            worker = Worker(session, written.append, 'session')
            worker._set_choices([
                ({'value': 'model', 'name': 'Model'}, object())], 'model')
            started, release = asyncio.Event(), asyncio.Event()

            async def completion(items, tools, *args, thinking, **kwargs):
                snapshots.append(thinking.effort)
                if len(snapshots) == 1:
                    started.set()
                    await release.wait()
                    return formats.DecodedTurn([
                        formats.tool_call_item('read', 'Read',
                                               {'file_path': target})])
                if len(snapshots) == 2:
                    result = next(item for item in items
                                  if item.get('type') == 'tool_result')
                    self.assertEqual(result['call_id'], 'read')
                    self.assertFalse(result['is_error'])
                    self.assertIn('real tool continuation',
                                  result['content'][0]['text'])
                else:
                    self.assertEqual(len(snapshots), 3)
                return formats.DecodedTurn([
                    formats.message_item('assistant',
                                         f'answer-{len(snapshots)}')])

            async def prompt(request_id):
                await worker.handle({
                    'jsonrpc': '2.0', 'id': request_id,
                    'method': 'session/prompt', 'params': {
                        'sessionId': 'session',
                        'prompt': [{'type': 'text', 'text': 'continue'}]}},
                    concurrent=True)

            async def scenario():
                try:
                    await prompt(1)
                    await observe_fixture(started.wait(), worker._prompt_task)
                    await worker.handle({
                        'jsonrpc': '2.0', 'id': 2,
                        'method': 'session/set_config_option', 'params': {
                            'sessionId': 'session',
                            'configId': 'reasoning_effort',
                            'value': 'effort:low'}}, concurrent=True)
                    release.set()
                    await worker._prompt_task
                    self.assertEqual(snapshots, ['high', 'high'])
                    await prompt(3)
                    await worker._prompt_task
                finally:
                    release.set()
                    await worker.close()

            with mock.patch.object(loki, '_DEFAULT_SESSION', session), \
                    mock.patch.object(loki, 'async_chat_completion', completion):
                asyncio.run(scenario())
            self.assertEqual(snapshots, ['high', 'high', 'low'])
            response = next(message for message in written
                            if message.get('id') == 2)
            self.assertEqual(response['result']['configOptions'][1]
                             ['currentValue'], 'effort:low')
            for request_id in [1, 3]:
                response = next(message for message in written
                                if message.get('id') == request_id)
                self.assertEqual(response['result']['stopReason'], 'end_turn')
            with open(session.chat_log_path, encoding='utf-8') as stream:
                saved = json.load(stream)
            self.assertEqual(saved['session_state']['reasoning_effort'], 'low')
            results = [item for item in saved['events']
                       if item.get('type') == 'tool_result']
            self.assertEqual([item['call_id'] for item in results], ['read'])
            self.assertFalse(results[0]['is_error'])
            self.assertIn('real tool continuation', results[0]['content'][0]['text'])
            self.assertIn('answer-2', json.dumps(saved['events']))
            self.assertIn('answer-3', json.dumps(saved['events']))


@unittest.skipUnless(os.name == 'posix', 'requires a POSIX controlling pseudo-terminal')
class TtyStdinTests(unittest.TestCase):
    """The front must work when stdin is a tty, not just a pipe.

    ACP owns its protocol stdin and reads fd 0 directly. This test gives
    the shipped front executable a real controlling pty, exactly like an
    interactive manual run.
    """

    def test_front_answers_initialize_with_tty_stdin(self):
        import fcntl
        import pty
        import termios
        master, slave = pty.openpty()

        def child_setup():
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env.update({
                "HOME": tmpdir,
                "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
                "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
                "TERM": "dumb",
                "LOKI_PROVIDER": "dummy",
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_MODEL": "dummy-model",
                "LOKI_DUMMY_REPLY": "x",
            })
            os.makedirs(os.path.join(tmpdir, "workspace"), exist_ok=True)
            proc = subprocess.Popen(
                loki_acp_command(),
                stdin=slave, stdout=subprocess.PIPE,
                env=env, cwd=os.path.join(tmpdir, "workspace"),
                preexec_fn=child_setup, text=True)
            self.addCleanup(_close_process_streams, proc)
            os.close(slave)
            try:
                os.write(master, (json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": 1},
                }) + "\n").encode())
                line = proc.stdout.readline()
                self.assertTrue(line, 'ACP front exited before its initialize reply')
                reply = json.loads(line)
                self.assertEqual(reply["id"], 1)
                self.assertEqual(
                    reply["result"]["agentInfo"]["name"], "loki")
            finally:
                if proc.poll() is None:
                    proc.terminate()
                proc.communicate()
                os.close(master)


class WorkerSessionContractTests(unittest.TestCase):
    def test_worker_rejects_unknown_open_method(self):
        from loki_agent import loki
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        with ExitStack() as cleanup:
            session = Session(shell_cwd=ROOT)
            cleanup.callback(session.response_headers.close)
            cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', session))
            worker = Worker(session, lambda _message: None)
            with self.assertRaisesRegex(
                    acps.TransportError,
                    "unsupported session opening method"):
                asyncio.run(worker.prepare_open({
                    "sessionId": "saved",
                    "cwd": ROOT,
                    "openMethod": "session/unknown",
                }))
            self.assertIsNone(worker._pending_open)

    def test_restore_rejects_a_different_saved_cwd(self):
        from loki_agent import formats, loki
        from loki_agent.acp_worker import Worker
        from loki_agent.credentials import CredentialStore
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as cleanup:
                saved_cwd = os.path.join(tmpdir, "saved-cwd")
                requested_cwd = os.path.join(tmpdir, "requested-cwd")
                # The saved log lives in the workspace the client names.
                chat_dir = app_paths.chat_log_dir_for(requested_cwd)
                os.mkdir(saved_cwd)
                os.mkdir(requested_cwd)
                os.makedirs(chat_dir)
                cleanup.enter_context(mock.patch.object(loki, 'CREDENTIALS', CredentialStore({})))

                for stored_cwd in [saved_cwd, "invalid\x00cwd"]:
                    blob = formats.new_log_blob(
                        loki.initial_transcript_items(), [])
                    blob["session_state"] = {"shell_cwd": stored_cwd}
                    with open(
                            os.path.join(chat_dir, "chat-saved.json"),
                            "w", encoding="utf-8") as stream:
                        json.dump(blob, stream)
                    for method in acp.RESTORE_METHODS:
                        with self.subTest(
                                method=method, stored_cwd=stored_cwd):
                            session = Session(shell_cwd="/")
                            cleanup.callback(session.response_headers.close)
                            cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', session))
                            worker = Worker(session, lambda _message: None)
                            with self.assertRaisesRegex(
                                    acps.TransportError,
                                    ("cwd does not match"
                                     if stored_cwd == saved_cwd
                                     else "could not load saved session")
                            ) as caught:
                                asyncio.run(worker.prepare_open({
                                    "sessionId": "saved",
                                    "cwd": requested_cwd,
                                    "openMethod": method,
                                }))
                            self.assertEqual(
                                caught.exception.code, acps.INVALID_PARAMS)
                            self.assertIsNone(worker._pending_open)

    def test_subscription_option_and_resume_use_brokered_credential(self):
        from loki_agent import formats, http_client, loki, models
        from loki_agent.acp_worker import Worker
        from loki_agent.credentials import CredentialInventory
        from loki_agent.sessions import Session

        credential = (
            authentications.CredentialRef.openai_subscription())
        catalog = models.add_openai_subscription_catalog(
            models.normalize_catalog({
                "openai": {
                    "id": "openai",
                    "name": "OpenAI",
                    "npm": "@ai-sdk/openai",
                    "env": ["OPENAI_API_KEY"],
                    "models": {},
                },
            }),
            {
                "models": [{
                    "slug": "gpt-test",
                    "display_name": "GPT Test",
                    "visibility": "list",
                    "input_modalities": ["text"],
                    "supported_reasoning_levels": [
                        {"effort": "low", "description": "Low"},
                    ],
                    "default_reasoning_level": "low",
                    "supports_reasoning_summaries": True,
                    "default_reasoning_summary": "none",
                    "support_verbosity": True,
                    "default_verbosity": "low",
                    "supports_parallel_tool_calls": False,
                    "use_responses_lite": True,
                    "tool_mode": "code_mode_only",
                    "context_window": 200000,
                    "base_instructions":
                        "must not be copied into the session log",
                }],
            },
        )
        groups = models.build_groups(catalog)
        refreshed_catalog = models.add_openai_subscription_catalog(
            {},
            {
                "models": [{
                    "slug": "gpt-test",
                    "display_name": "GPT Test",
                    "visibility": "list",
                    "input_modalities": ["text"],
                    "supported_reasoning_levels": [
                        {"effort": "high", "description": "High"},
                    ],
                    "default_reasoning_level": "high",
                    "supports_reasoning_summaries": True,
                    "default_reasoning_summary": "detailed",
                    "support_verbosity": True,
                    "default_verbosity": "medium",
                    "supports_parallel_tool_calls": True,
                    "use_responses_lite": False,
                }],
            },
        )
        refreshed_groups = models.build_groups(refreshed_catalog)
        broker = authentications.CredentialBroker()
        broker.install_openai_subscription(
            authentications.OpenAITokenSet(
                access_token="access-secret",
                refresh_token="refresh-secret",
                expires_at=10**12,
            ))
        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as cleanup:
                cleanup.enter_context(mock.patch.object(loki, 'CREDENTIALS', CredentialInventory(
                    {}, {credential})))
                requests = []

                async def request(method, url, **kwargs):
                    requests.append((
                        method,
                        url,
                        dict(kwargs["headers_in"]),
                    ))
                    return http_client.HttpResponse(
                        url, 200, "OK", {}, b"{}")

                async def scenario():
                    first_session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                    cleanup.callback(first_session.response_headers.close)
                    first_session.credential_authority = broker
                    cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', first_session))
                    first_worker = Worker(
                        first_session,
                        lambda _message: None,
                        "first",
                    )
                    with mock.patch.object(
                            models,
                            "ensure_index",
                            new=mock.AsyncMock(
                                return_value=(catalog, groups))):
                        await first_worker.prepare_open({
                            "sessionId": "first",
                            "cwd": tmpdir,
                            "openMethod": "session/new",
                        })
                        opened = first_worker.commit_open()
                    options = opened["configOptions"][0]["options"]
                    value = "openai-subscription/gpt-test"
                    self.assertIn(
                        value,
                        [option["value"] for option in options],
                    )
                    first_worker.set_config_option({
                        "configId": "model",
                        "value": value,
                    })
                    first_provider = (
                        loki.current_config().chat_provider)
                    first_profile = first_provider.openai_request_profile
                    first_payload = (
                        loki.current_config().chat_provider.chat_payload(
                            [formats.message_item("user", "hello")],
                            [],
                            "gpt-test",
                            prompt_cache_key=(
                                first_session.conversation_id),
                        ))
                    with mock.patch.object(
                            http_client,
                            "async_http_request",
                            new=request):
                        await loki.async_provider_request(
                            "POST",
                            loki.current_config().chat_provider.input_url,
                            {})
                    saved_id = "first"
                    with open(
                            first_session.chat_log_path,
                            "r", encoding="utf-8") as stream:
                        saved_text = stream.read()
                    await first_worker.close()

                    resumed_session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                    cleanup.callback(resumed_session.response_headers.close)
                    resumed_session.credential_authority = broker
                    cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', resumed_session))
                    resumed_worker = Worker(
                        resumed_session,
                        lambda _message: None,
                        "resumed",
                    )
                    with mock.patch.object(
                            models,
                            "ensure_index",
                            new=mock.AsyncMock(
                                return_value=(
                                    refreshed_catalog,
                                    refreshed_groups,
                                ))):
                        prepared = await resumed_worker.prepare_open({
                            "sessionId": saved_id,
                            "cwd": tmpdir,
                            "openMethod": "session/resume",
                        })
                        self.assertIsNone(resumed_session.runtime_config)
                        self.assertEqual(
                            prepared["authorizationConnection"]["model"],
                            "gpt-test",
                        )
                        with open(
                                resumed_session.chat_log_path,
                                "r", encoding="utf-8") as stream:
                            self.assertEqual(stream.read(), saved_text)
                        resumed = resumed_worker.commit_open()
                    with open(
                            resumed_session.chat_log_path,
                            "r", encoding="utf-8") as stream:
                        committed_text = stream.read()
                    self.assertNotEqual(committed_text, saved_text)
                    self.assertEqual(
                        resumed["configOptions"][0]["currentValue"],
                        "loki-saved",
                    )
                    resumed_provider = (
                        loki.current_config().chat_provider)
                    resumed_profile = (
                        resumed_provider.openai_request_profile)
                    resumed_payload = (
                        loki.current_config().chat_provider.chat_payload(
                            [formats.message_item("user", "hello")],
                            [],
                            "gpt-test",
                            prompt_cache_key=(
                                resumed_session.conversation_id),
                        ))
                    with mock.patch.object(
                            http_client,
                            "async_http_request",
                            new=request):
                        await loki.async_provider_request(
                            "POST",
                            loki.current_config().chat_provider.input_url,
                            {})
                    await resumed_worker.close()
                    return (
                        first_profile,
                        resumed_profile,
                        first_payload,
                        resumed_payload,
                        saved_text,
                    )

                (
                    first_profile,
                    resumed_profile,
                    first_payload,
                    resumed_payload,
                    saved_text,
                ) = asyncio.run(scenario())

                self.assertNotEqual(first_profile, resumed_profile)
                self.assertFalse(hasattr(first_profile, "tool_mode"))
                self.assertFalse(hasattr(first_profile, "context_window"))
                self.assertEqual(
                    first_payload["reasoning"],
                    {"effort": "low", "context": "all_turns"},
                )
                self.assertEqual(
                    first_payload["text"], {"verbosity": "low"})
                self.assertFalse(first_payload["parallel_tool_calls"])
                self.assertEqual(
                    resumed_payload["reasoning"],
                    {"effort": "high", "summary": "detailed"},
                )
                self.assertEqual(
                    resumed_payload["text"], {"verbosity": "medium"})
                self.assertTrue(resumed_payload["parallel_tool_calls"])
                self.assertNotIn("context", resumed_payload["reasoning"])
                self.assertNotIn(
                    "must not be copied into the session log", saved_text)
                self.assertEqual(len(requests), 2)
                for _method, url, headers in requests:
                    self.assertEqual(
                        url,
                        authentications.OPENAI_CHATGPT_RESPONSES_URL,
                    )
                    self.assertEqual(
                        headers["Authorization"],
                        "Bearer access-secret",
                    )

    def test_restore_accepts_equivalent_client_cwd(self):
        from unittest import mock
        from loki_agent import formats, loki, protocols
        from loki_agent.acp_worker import Worker
        from loki_agent.connections import ConnectionDescriptor
        from loki_agent.credentials import CredentialStore
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as cleanup:
                saved_cwd = os.path.join(tmpdir, "saved-cwd")
                client_cwd = os.path.join(tmpdir, "client-cwd")
                os.mkdir(saved_cwd)
                directory_alias(saved_cwd, client_cwd)
                # The saved log lives in the workspace spelling the client
                # names, which identifies the same directory as the saved cwd.
                chat_dir = app_paths.chat_log_dir_for(client_cwd)
                os.makedirs(chat_dir)
                descriptor = ConnectionDescriptor(
                    provider_id="saved-provider",
                    provider_name="Saved Provider",
                    model="saved-model",
                    chat_url=(
                        "https://saved.example/v1/chat/completions"),
                    models_url="https://saved.example/v1/models",
                    protocol=protocols.OPENAI_CHAT,
                    credential_ref=(
                        authentications.CredentialRef.environment(
                            "SAVED_API_KEY")),
                    max_tokens=777,
                )
                blob = formats.new_log_blob(
                    loki.initial_transcript_items(), [])
                blob["session_state"] = {
                    "shell_cwd": saved_cwd,
                    "connection": descriptor.to_dict(),
                }
                saved_name = "chat-saved.json"
                with open(
                        os.path.join(chat_dir, saved_name),
                        "w", encoding="utf-8") as stream:
                    json.dump(blob, stream)

                session = Session(shell_cwd="/")
                cleanup.callback(session.response_headers.close)
                cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', session))
                cleanup.enter_context(mock.patch.object(loki, 'CREDENTIALS', CredentialStore({
                    "SAVED_API_KEY": "secret",
                })))
                worker = Worker(session, lambda message: None)
                with mock.patch(
                        "loki_agent.acp_worker.modelsdev.ensure_index",
                        new=mock.AsyncMock(return_value=({}, {}))):
                    async def open_worker():
                        await worker.prepare_open({
                            "sessionId": "saved",
                            "cwd": client_cwd,
                            "openMethod": "session/resume",
                        })
                        return worker.commit_open()

                    result = asyncio.run(open_worker())

                self.assertEqual(session.shell_cwd, client_cwd)
                self.assertEqual(session.model, "saved-model")
                self.assertEqual(
                    session.runtime_config.chat_provider.max_tokens, 777)
                self.assertEqual(
                    result["configOptions"][0]["currentValue"],
                    "loki-saved",
                )

    def test_new_sessions_get_distinct_persistent_logs_with_empty_catalog(self):
        from unittest import mock
        from loki_agent import loki, models
        from loki_agent.acp_worker import Worker
        from loki_agent.credentials import CredentialStore
        from loki_agent.sessions import Session

        paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as cleanup:
                cleanup.enter_context(mock.patch.object(loki, 'CREDENTIALS', CredentialStore({})))
                for number in range(2):
                    session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                    cleanup.callback(session.response_headers.close)
                    cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', session))
                    worker = Worker(session, lambda message: None)
                    with mock.patch.object(
                            models, "ensure_index",
                            new=mock.AsyncMock(return_value=({}, {}))):
                        async def open_worker():
                            await worker.prepare_open({
                                "sessionId": f"live-{number}",
                                "cwd": tmpdir,
                                "openMethod": "session/new",
                            })
                            return worker.commit_open()

                        result = asyncio.run(open_worker())
                    paths.append(session.chat_log_path)
                    option = result["configOptions"][0]
                    self.assertEqual(
                        option["currentValue"], "loki-disconnected")
                    self.assertIn(
                        option["currentValue"],
                        [entry["value"] for entry in option["options"]],
                    )
                self.assertNotEqual(paths[0], paths[1])
                log_dir = app_paths.chat_log_dir_for(tmpdir)
                self.assertEqual(paths, [
                    os.path.join(log_dir, f'chat-live-{number}.json') for number in range(2)])

    def test_acp_turn_end_hook_outcomes(self):
        from loki_agent import loki, protocols, tool_runtime
        from loki_agent.acp_worker import TurnFailure, Worker
        from loki_agent.sessions import Session

        for outcome in ["completed", "stream_error", "raised", "cancelled"]:
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmpdir, ExitStack() as cleanup:
                session = Session(shell_cwd=tmpdir)
                cleanup.callback(session.response_headers.close)
                worker = Worker(session, lambda message: None, "s")

                async def run_loop(*args, on_event, **kwargs):
                    if outcome == "stream_error":
                        on_event({
                            "type": "stream_error",
                            "error": protocols.StreamProtocolError("unexpected EOF"),
                        })
                        return ""
                    if outcome == "raised":
                        raise RuntimeError("turn failed")
                    if outcome == "cancelled":
                        worker.cancel_event.set()
                        raise asyncio.CancelledError()
                    return "answer"

                # Real notification-only configuration: an unspecified mock
                # attribute must not accidentally select a recovery policy.
                pipeline = tool_runtime.ToolHookPipeline()
                pipeline.turn_end_hooks.append(object())
                pipeline.finish_turn = mock.AsyncMock()
                with mock.patch.object(loki, "_DEFAULT_SESSION", session), \
                        mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline), \
                        mock.patch.object(loki, "current_model", return_value="test"), \
                        mock.patch.object(loki, "run_tool_loop_async", side_effect=run_loop), \
                        mock.patch.object(loki, "save_chat_log"):
                    prompt = worker.prompt({
                        "sessionId": "s",
                        "prompt": [{"type": "text", "text": "hello"}],
                    })
                    if outcome == "raised":
                        with self.assertRaisesRegex(RuntimeError, "turn failed"):
                            asyncio.run(prompt)
                    elif outcome == "stream_error":
                        with self.assertRaisesRegex(TurnFailure, "unexpected EOF"):
                            asyncio.run(prompt)
                    else:
                        expected_stop = "cancelled" if outcome == "cancelled" else "end_turn"
                        self.assertEqual(asyncio.run(prompt)["stopReason"], expected_stop)
                pipeline.finish_turn.assert_awaited_once()
                payload = pipeline.finish_turn.await_args.args[0]
                self.assertEqual(payload["event"], "turn_end")
                expected = "error" if outcome in ["stream_error", "raised"] else outcome
                self.assertEqual(payload["reason"], expected)
                self.assertEqual(payload["text"], "answer" if outcome == "completed" else "")
                if outcome in ["stream_error", "raised"]:
                    expected_type = "stream_error" if outcome == "stream_error" else "RuntimeError"
                    expected_message = "unexpected EOF" if outcome == "stream_error" else "turn failed"
                    self.assertEqual(payload["error_type"], expected_type)
                    self.assertEqual(payload["error_message"], expected_message)
                else:
                    self.assertNotIn("error_type", payload)
                    self.assertNotIn("error_message", payload)

    def test_provider_failure_is_an_acp_request_failure(self):
        from loki_agent import formats, loki, protocols
        from loki_agent.acp_worker import TurnFailure, Worker
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as cleanup:
                session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                cleanup.callback(session.response_headers.close)
                session.transcript_items = [
                    formats.instruction_item("system"),
                ]
                cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', session))
                worker = Worker(session, lambda message: None, "s")

                async def failed_turn(on_event, _reasoning_effort):
                    on_event({
                        "type": "provider_error",
                        "error": protocols.ProtocolError(
                            "invalid provider JSON"),
                    })

                worker._run_turn = failed_turn
                with self.assertRaisesRegex(
                        TurnFailure, "invalid provider JSON"):
                    asyncio.run(worker.prompt({
                        "sessionId": "s",
                        "prompt": [{"type": "text", "text": "hello"}],
                    }))

    def test_stop_reasons_cover_protocol_terminal_states(self):
        from loki_agent.acp_worker import Worker

        self.assertEqual(
            Worker._stop_reason([{"type": "response_cancelled"}]),
            "cancelled",
        )
        self.assertEqual(
            Worker._stop_reason([{"type": "max_loops"}]),
            "max_turn_requests",
        )
        self.assertEqual(
            Worker._stop_reason([{"type": "response_incomplete"}]),
            "max_tokens",
        )
        self.assertEqual(
            Worker._stop_reason([{"type": "response_refusal"}]),
            "refusal",
        )

    def test_cancelled_prompt_returns_cancelled_even_if_operation_raises(self):
        from loki_agent import loki
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as cleanup:
                session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                cleanup.callback(session.response_headers.close)
                cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', session))
                worker = Worker(session, lambda message: None, "s")

                async def cancelled_failure(
                        on_event, _reasoning_effort):
                    worker.cancel_event.set()
                    raise RuntimeError("underlying close race")

                worker._run_turn = cancelled_failure
                result = asyncio.run(worker.prompt({
                    "sessionId": "s",
                    "prompt": [{"type": "text", "text": "hello"}],
                }))
                self.assertEqual(result["stopReason"], "cancelled")

    def test_refused_prompt_is_retained_persisted_replayed_and_reused(self):
        import copy
        from unittest import mock
        from loki_agent import formats, loki, protocols, savefiles
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as cleanup:
                initial = [formats.instruction_item("system")]
                chat_path = os.path.join(tmpdir, "chat-refusal.json")
                session = Session(
                    shell_cwd=os.path.join(tmpdir, "workspace"),
                    transcript_items=list(initial),
                    chat_log_path=chat_path,
                )
                cleanup.callback(session.response_headers.close)
                cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', session))
                messages = []
                worker = Worker(session, messages.append, "s")
                turns = [
                    formats.DecodedTurn(
                        [formats.message_item(
                            "assistant",
                            [{"type": "refusal", "text": "No"}],
                        )],
                        metadata={
                            "protocol": protocols.OPENAI_CHAT,
                            "stop_reason": "refusal",
                        },
                    ),
                    formats.DecodedTurn(
                        [formats.message_item("assistant", "Continued")],
                        metadata={
                            "protocol": protocols.OPENAI_CHAT,
                            "stop_reason": "end_turn",
                        },
                    ),
                ]
                requests = []

                async def completion(items, *_args, **_kwargs):
                    requests.append(copy.deepcopy(items))
                    return turns.pop(0)

                async def scenario():
                    with mock.patch.object(
                            loki, "current_model", return_value="model"
                    ), mock.patch.object(
                            loki, "async_chat_completion", new=completion
                    ):
                        refused = await worker.prompt({
                            "sessionId": "s",
                            "prompt": [
                                {"type": "text", "text": "unsafe"},
                            ],
                        })
                        with open(
                                chat_path, encoding="utf-8") as chat_file:
                            persisted = savefiles.read_chat_log(chat_file)[0]
                        messages.clear()
                        worker._replay_transcript()
                        replayed = list(messages)
                        messages.clear()
                        continued = await worker.prompt({
                            "sessionId": "s",
                            "prompt": [
                                {"type": "text", "text": "why?"},
                            ],
                        })
                        return refused, continued, persisted, replayed

                refused, continued, persisted, replayed = asyncio.run(
                    scenario())

                self.assertEqual(refused["stopReason"], "refusal")
                self.assertEqual(continued["stopReason"], "end_turn")
                self.assertEqual(
                    [event["type"] for event in persisted],
                    ["message", "message", "model_response"],
                )
                self.assertEqual(formats.item_text(persisted[1]), "unsafe")
                self.assertEqual(formats.item_text(persisted[2]), "No")
                self.assertEqual(
                    [formats.item_text(event) for event in requests[1]],
                    ["system", "unsafe", "No", "why?"],
                )
                replay_updates = [
                    message["params"]["update"]
                    for message in replayed
                    if message.get("method") == "session/update"
                ]
                replay_text = [
                    update["content"]["text"]
                    for update in replay_updates
                    if update.get("sessionUpdate") in [
                        "user_message_chunk", "agent_message_chunk"]
                ]
                self.assertIn("unsafe", replay_text)
                self.assertIn("No", replay_text)

    def test_refusal_does_not_erase_prior_tool_activity_in_prompt(self):
        from loki_agent import formats, loki
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as cleanup:
                session = Session(
                    shell_cwd=os.path.join(tmpdir, "workspace"),
                    transcript_items=[formats.instruction_item("system")],
                )
                cleanup.callback(session.response_headers.close)
                cleanup.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', session))
                worker = Worker(session, lambda message: None, "s")
                call = {
                    "type": "function_call",
                    "name": "Read",
                    "call_id": "call-1",
                    "arguments": {"file_path": "source.py"},
                }

                async def refused_after_tool(
                        on_event, _reasoning_effort):
                    session.transcript_items.append(
                        formats.DecodedTurn([call]).to_event())
                    session.transcript_items.append(
                        formats.tool_result_for_call(call, "file contents"))
                    session.transcript_items.append(
                        formats.DecodedTurn([
                            formats.message_item(
                                "assistant",
                                [{"type": "refusal", "text": "No"}],
                            ),
                        ]).to_event())
                    on_event({"type": "response_refusal"})

                worker._run_turn = refused_after_tool
                result = asyncio.run(worker.prompt({
                    "sessionId": "s",
                    "prompt": [{"type": "text", "text": "inspect"}],
                }))

                self.assertEqual(result["stopReason"], "refusal")
                self.assertEqual(
                    [event["type"] for event in session.transcript_items],
                    [
                        "message",
                        "message",
                        "model_response",
                        "tool_result",
                        "model_response",
                    ],
                )
                formats.validate_events(session.transcript_items)


class ConfigEndpointApprovalTests(unittest.IsolatedAsyncioTestCase):
    """The ACP front asks before a catalog endpoint receives a credential."""

    PAIR = {
        "credential": "env:ACME_API_KEY",
        "destinations": ["https://acme.invalid/v1/chat/completions", "https://acme.invalid/v1/models"],
    }

    def _front(self, *, supports_elicitation=True):
        front = acp.Front(
            lambda: None, lambda message: None, CredentialStore({"ACME_API_KEY": "sample"}))
        front._client_supports_form_elicitation = supports_elicitation
        return front

    def _channel(self, front, selection):
        channel = mock.Mock()
        channel.session_id = "s"
        channel._closed = False
        channel.process.returncode = None
        channel.request = mock.AsyncMock(return_value=selection)
        owner = front._reserve_session('s')
        owner.channel = channel
        owner.state = 'active'
        return channel

    @asynccontextmanager
    async def _approval_journey(self):
        """Real application components; only catalog/HTTP and OS-child seams.

        All client traffic uses one serial Front.run input stream. This is
        component qualification, not a shipped-process catalog/HTTP check.
        """
        from types import SimpleNamespace
        from test_http_client import FakeConnector
        from loki_agent import credential_storages, http_client, loki
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as root:
            async with AsyncExitStack() as stack:
                workspace = os.path.join(root, 'workspace')
                os.mkdir(workspace)
                environment = {key: value for key, value in os.environ.items()
                               if not key.startswith('LOKI_')
                               and not key.endswith(('_KEY', '_TOKEN', '_PAT'))}
                environment.update(HOME=root, XDG_CONFIG_HOME=os.path.join(root, 'config'),
                                   XDG_STATE_HOME=os.path.join(root, 'state'))
                stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
                selected, unrelated, previous = (
                    'approval-selected-secret', 'approval-unrelated-secret',
                    'approval-previous-secret')
                credentials = CredentialStore({
                    'ACME_API_KEY': selected, 'UNRELATED_API_KEY': unrelated,
                    'LOKI_API_KEY': previous, 'LOKI_PROVIDER': 'openai',
                    'LOKI_API_BASE': 'https://previous.example/v1',
                    'LOKI_MODEL': 'previous-model', 'LOKI_STREAM': '0'})
                endpoint = 'https://approved.example/override/v1'
                catalog = {'acme': {
                    'id': 'acme', 'name': 'Acme', 'env': ['ACME_API_KEY'],
                    'npm': '@ai-sdk/openai-compatible', 'api': 'https://provider.example/base/v1',
                    'models': {'chosen-model': {
                        'id': 'chosen-model', 'name': 'Chosen Model',
                        'provider': {'api': endpoint, 'npm': '@ai-sdk/openai-compatible'}}}}}
                groups = models.build_groups(catalog)
                stack.enter_context(mock.patch.object(
                    models, 'ensure_index',
                    new=mock.AsyncMock(return_value=(catalog, groups))))
                packets = []
                for answer in ['approved answer', 'pinned answer']:
                    body = json.dumps({'id': answer, 'choices': [{
                        'index': 0, 'message': {'role': 'assistant', 'content': answer},
                        'finish_reason': 'stop'}]}).encode()
                    packets.append(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '
                                   + str(len(body)).encode() + b'\r\n\r\n' + body)
                f = SimpleNamespace(messages=[], changed=asyncio.Event(), tasks=[],
                                    processes=[], delegations=[], connector=FakeConnector(packets),
                                    trace=[], endpoint=endpoint, workspace=workspace,
                                    secrets=(selected, unrelated, previous),
                                    tasks_before=asyncio.all_tasks())

                def write(message):
                    f.messages.append(message)
                    if message.get('method') == 'elicitation/create':
                        f.trace.append('elicitation')
                    f.changed.set()

                f.storage = credential_storages.JsonCredentialStorage(os.path.join(root, 'config', 'loki', 'credentials'))
                stack.callback(f.storage.close)
                f.front = acp.Front(lambda: None, write, credentials, f.storage)
                stack.enter_context(mock.patch.object(loki, 'CREDENTIALS', f.front.credentials))
                base_session = Session(shell_cwd=workspace)
                stack.push_async_callback(base_session.close_state)
                stack.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', base_session))
                real_connect = asyncio.open_connection

                async def connect(*args, **kwargs):
                    if 'sock' in kwargs:
                        return await real_connect(*args, **kwargs)
                    f.trace.append('http')
                    return await f.connector.open_connection(*args, **kwargs)

                stack.enter_context(mock.patch.object(http_client.asyncio, 'open_connection', new=connect))
                real_lease = f.front.credential_broker.lease
                f.leases = []

                async def lease(ref, **kwargs):
                    f.leases.append(ref.encode())
                    f.trace.append('lease')
                    return await real_lease(ref, **kwargs)

                stack.enter_context(mock.patch.object(f.front.credential_broker, 'lease', new=lease))
                await f.front.credential_broker.approve_destinations(
                    authentications.CredentialRef.environment('UNRELATED_API_KEY'),
                    ['https://untouched.example/v1'])
                f.approvals_path = f.storage.file_path
                with open(f.approvals_path, 'rb') as stream:
                    f.original_approvals_document = stream.read()
                f.expected_approvals = {
                    'env:UNRELATED_API_KEY': ['https://untouched.example/v1'],
                    'env:ACME_API_KEY': [endpoint + '/chat/completions', endpoint + '/models']}
                real_record = f.storage.approve_destinations

                async def record(*args, **kwargs):
                    await real_record(*args, **kwargs)
                    f.trace.append('approval')

                stack.enter_context(mock.patch.object(f.storage, 'approve_destinations', new=record))

                async def launch(cwd, environment, delegation, *, credential_directory=None):
                    session = Session(shell_cwd=cwd)
                    stack.push_async_callback(session.close_state)
                    # Same-process authority, as in the existing selection
                    # journey; no worker credential-IPC claim is made here.
                    session.credential_authority = f.front.credential_broker
                    loki._DEFAULT_SESSION = session
                    loki.apply_runtime_config(connections.build_config_from_env(credentials=loki.CREDENTIALS))

                    def before_request(message):
                        if message['method'] == 'session/describe_config_selection':
                            f.trace.append('describe')
                        if message['method'] == 'session/set_config_option':
                            f.trace.append('switch')
                            with open(f.approvals_path, encoding='utf-8') as stream:
                                self.assertEqual(json.load(stream)["endpoint_approvals"], f.expected_approvals)

                    process = _LocalWorkerProcess(session, 'provisional', before_request)
                    f.processes.append(process)
                    f.delegations.append(delegation)
                    f.session = session
                    return process

                stack.enter_context(mock.patch.object(acp.runtime_isolation, 'start_worker', new=launch))
                stack.enter_context(mock.patch.object(acp.runtime_isolation, 'close_runtime_process', new=lambda process: None))

                async def wait_message(predicate, *finished):
                    while True:
                        f.changed.clear()
                        for message in f.messages:
                            if predicate(message):
                                return message
                        await observe_fixture(f.changed.wait(), f.source.task, *finished)

                async def request(request_id, method, params):
                    await f.source.send(acps.request(request_id, method, params))
                    message = await wait_message(lambda message: message.get('id') == request_id)
                    self.assertNotIn('error', message, message)
                    return message['result']

                f.wait_message, f.request = wait_message, request
                f.source = _FrontInput(f.front)
                try:
                    await f.source.send(acps.request(0, 'initialize', {
                        'clientCapabilities': {'elicitation': {'form': {}}}}))
                    opened = await request(1, 'session/new', {'cwd': workspace})
                    f.session_id = opened['sessionId']
                    f.channel = f.front.workers[f.session_id]
                    choice, = [option for config in opened['configOptions'] if config['id'] == 'model'
                               for option in config['options'] if option['value'] == 'acme/chosen-model']
                    f.value = choice['value']
                    f.params = {'sessionId': f.session_id, 'configId': 'model', 'value': f.value}
                    f.original_config = f.session.runtime_config
                    f.path = f.session.chat_log_path
                    self.assertTrue(loki.save_chat_log())
                    with open(f.path, 'rb') as stream:
                        f.original_chat = stream.read()
                    f.trace.clear()
                    self.assertEqual(f.leases, [])
                    yield f
                finally:
                    for task in f.tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*f.tasks, return_exceptions=True)
                    await f.source.close()
                    # Failure backstop, not release evidence: assertions above
                    # must pass before this fallback can repair a leaked owner.
                    await f.front.shutdown()
                    for delegation in f.delegations:
                        await delegation.close()
                    for task in list(f.front._tasks):
                        task.cancel()
                    await asyncio.gather(*f.front._tasks, return_exceptions=True)

    async def _pending_approval(self, f, request_id):
        task = asyncio.create_task(f.wait_message(
            lambda message: message.get('id') == request_id))
        f.tasks.append(task)
        await f.source.send(acps.request(
            request_id, 'session/set_config_option', f.params))
        elicitation = await f.wait_message(
            lambda message: message.get('method') == 'elicitation/create'
            and message['params'].get('requestId') == request_id, task)
        self.assertFalse(task.done())
        self.assertIn(elicitation['id'], f.front._client_requests)
        self.assertEqual(elicitation['jsonrpc'], '2.0')
        params = elicitation['params']
        self.assertEqual(params['mode'], 'form')
        self.assertIn('Credential: "env:ACME_API_KEY"', params['message'])
        for target in f.expected_approvals['env:ACME_API_KEY']:
            self.assertIn(target, params['message'])
        self.assertEqual(params['requestedSchema']['type'], 'object')
        self.assertEqual(params['requestedSchema']['required'], ['approve'])
        field = params['requestedSchema']['properties']['approve']
        self.assertEqual(field['type'], 'boolean')
        self.assertIs(field['default'], False)
        self.assertEqual(f.trace[-2:], ['describe', 'elicitation'])
        self.assertIs(f.session.runtime_config, f.original_config)
        self.assertEqual(f.session.session_state['connection'],
                         json.loads(f.original_chat)['session_state']['connection'])
        self.assertFalse(f.session.chat_log_dirty)
        self.assertEqual(f.leases, [])
        self.assertEqual(f.connector.calls, [])
        with open(f.approvals_path, 'rb') as stream:
            self.assertEqual(stream.read(), f.original_approvals_document)
        with open(f.path, 'rb') as stream:
            self.assertEqual(stream.read(), f.original_chat)
        return task, elicitation

    async def _close_approval_journey(self, f):
        self.assertEqual(await f.request(90, 'session/close', {'sessionId': f.session_id}), {})
        await f.source.finish()
        await self._assert_approval_released(f)

    async def _assert_approval_released(self, f):
        self.assertFalse(f.front._sessions)
        self.assertFalse(f.front.workers)
        self.assertFalse(f.front._client_requests)
        self.assertFalse(f.channel._pending)
        self.assertTrue(f.channel._reader_task.done())
        self.assertIsNone(f.channel.credential_delegation)
        for process in f.processes:
            self.assertEqual(process.returncode, 0)
            self.assertTrue(process.close_task.done())
            self.assertIsNone(process.close_task.exception())
        for delegation in f.delegations:
            self.assertIsNone(delegation.owner_parent)
            self.assertIsNone(delegation.owner_child)
            self.assertIsNone(delegation.credential_child)
            server = delegation.credential_server
            self.assertTrue(server._reader_task.done())
            self.assertTrue(server._close_task.done())
            self.assertIsNone(server._close_task.exception())
        for writer in f.connector.writers:
            self.assertTrue(writer.closed)
            self.assertTrue(writer.wait_closed_called)
        await asyncio.sleep(0)
        self.assertFalse(f.front._tasks)
        if f.session.job_manager is not None:
            self.assertEqual(f.session.job_manager.jobs, {})
        self.assertFalse(asyncio.all_tasks() - f.tasks_before)
        with open(f.path, 'rb') as stream:
            saved = stream.read()
        with open(f.approvals_path, 'rb') as stream:
            pins = stream.read()
        for secret in f.secrets:
            self.assertNotIn(secret.encode(), saved + pins + json.dumps(f.messages).encode())

    def _assert_approved_descriptor(self, f, blob):
        descriptor = blob['session_state']['connection']
        self.assertEqual(descriptor['provider_id'], 'acme')
        self.assertEqual(descriptor['model'], 'chosen-model')
        self.assertEqual(descriptor['chat_url'], f.endpoint + '/chat/completions')
        self.assertEqual(descriptor['credential'], {'kind': 'env', 'name': 'ACME_API_KEY'})
        self.assertEqual(descriptor['protocol'], 'openai_chat')
        self.assertIs(descriptor['stream'], False)

    async def test_approval_records_switches_infers_and_reuses(self):
        from loki_agent import formats

        async with self._approval_journey() as f:
            task, elicitation = await self._pending_approval(f, 9)
            # A foreign response cannot resolve this approval.
            await f.source.send(acps.response('unrelated-id', result={
                'action': 'accept', 'content': {'approve': True}}))
            self.assertFalse(f.front._client_requests[elicitation['id']].done())
            self.assertFalse(task.done())
            await f.source.send(acps.response(elicitation['id'], result={
                'action': 'accept', 'content': {'approve': True}}))
            await task
            reply = await f.wait_message(lambda message: message.get('id') == 9)
            self.assertNotIn('error', reply, reply)
            self.assertEqual(f.trace, ['describe', 'elicitation', 'approval', 'switch'])
            model_option, = [option for option in reply['result']['configOptions'] if option['id'] == 'model']
            self.assertEqual(model_option['currentValue'], f.value)
            self.assertFalse(f.front._client_requests)
            self.assertEqual(f.session.runtime_config.model, 'chosen-model')
            self.assertEqual(f.session.runtime_config.chat_provider.chat_url, f.endpoint + '/chat/completions')
            with open(f.path, 'rb') as stream:
                switched = json.load(stream)
            self._assert_approved_descriptor(f, switched)
            self.assertEqual(_conversation_pairs(switched), [])
            self.assertEqual(f.storage.load_document()['endpoint_approvals'], f.expected_approvals)
            self.assertEqual(await f.front.credential_broker.unapproved_destinations(
                authentications.CredentialRef.environment('ACME_API_KEY'), f.expected_approvals['env:ACME_API_KEY']), [])
            expected_pairs = []
            for index, (prompt, answer) in enumerate((('approved prompt', 'approved answer'),
                                                      ('pinned prompt', 'pinned answer'))):
                if index:
                    with open(f.approvals_path, 'rb') as stream:
                        pinned_bytes = stream.read()
                    await f.request(12, 'session/set_config_option', f.params)
                    self.assertEqual(sum(message.get('method') == 'elicitation/create'
                                         for message in f.messages), 1)
                    with open(f.approvals_path, 'rb') as stream:
                        self.assertEqual(stream.read(), pinned_bytes)
                start = len(f.messages)
                result = await f.request(20 + index, 'session/prompt', {
                    'sessionId': f.session_id, 'prompt': [{'type': 'text', 'text': prompt}]})
                self.assertEqual(result, {'stopReason': 'end_turn'})
                self.assertEqual(_historical_chunks(f.messages[start:], f.session_id), [
                    ('agent_message_chunk', {'type': 'text', 'text': answer})])
                updates = [message for message in f.messages[start:] if message.get('method') == 'session/update']
                self.assertTrue(updates)
                self.assertTrue(all(message['params']['sessionId'] == f.session_id for message in updates))
                packet = bytes(f.connector.writers[index].data)
                headers, body = packet.split(b'\r\n\r\n', 1)
                self.assertTrue(headers.startswith(b'POST /override/v1/chat/completions HTTP/1.1\r\n'))
                self.assertIn(b'Host: approved.example', headers)
                authorization = [line for line in headers.split(b'\r\n')
                                 if line.lower().startswith(b'authorization:')]
                self.assertEqual(authorization, [b'Authorization: Bearer ' + f.secrets[0].encode()])
                for secret in f.secrets[1:]:
                    self.assertNotIn(secret.encode(), packet)
                payload = json.loads(body)
                self.assertEqual(payload['model'], 'chosen-model')
                self.assertNotIn('stream', payload)
                expected_context = expected_pairs + [('user', prompt)]
                self.assertEqual([(message['role'], message['content']) for message in payload['messages']
                                  if message['role'] in ['user', 'assistant']], expected_context)
                expected_pairs.extend([('user', prompt), ('assistant', answer)])
                with open(f.path, 'rb') as stream:
                    saved = stream.read()
                blob = json.loads(saved)
                formats.validate_events(blob['events'])
                self.assertEqual(_conversation_pairs(blob), [
                    (role, [{'type': 'text', 'text': text}])
                    for role, text in expected_pairs])
                self._assert_approved_descriptor(f, blob)
                with open(f.approvals_path, 'rb') as stream:
                    pins = stream.read()
                for secret in f.secrets:
                    self.assertNotIn(secret.encode(), saved + pins + json.dumps(f.messages).encode())
            self.assertEqual([call['host'] for call in f.connector.calls], ['approved.example'] * 2)
            self.assertEqual(f.leases, ['env:ACME_API_KEY'] * 2)
            self.assertEqual(f.connector.responses, [])
            self.assertEqual(f.trace, ['describe', 'elicitation', 'approval', 'switch', 'lease', 'http',
                                       'describe', 'switch', 'lease', 'http'])
            await self._close_approval_journey(f)

    @contextmanager
    def _record_session_stores(self):
        from loki_agent import sessions
        session_type = sessions.Session
        stores = []

        def create(*args, **kwargs):
            session = session_type(*args, **kwargs)
            stores.append(session.response_headers)
            return session

        # Retain every Store across scope exit so GC cannot supply the closure
        # that this regression requires from the fixture's explicit owner.
        with mock.patch.object(sessions, 'Session', side_effect=create):
            yield stores

    async def test_fixture_closes_session_stores_on_success(self):
        with self._record_session_stores() as stores:
            async with self._approval_journey() as f:
                await self._close_approval_journey(f)
            self.assertEqual(len(stores), 2)
            for storage in stores:
                with self.assertRaisesRegex(ValueError, 'closed'):
                    _ = storage.directory_handle

    async def test_fixture_closes_session_stores_on_pending_failure(self):
        with self._record_session_stores() as stores:
            with self.assertRaisesRegex(RuntimeError, 'fixture body failed'):
                async with self._approval_journey() as f:
                    await self._pending_approval(f, 9)
                    raise RuntimeError('fixture body failed')
            self.assertEqual(len(stores), 2)
            for storage in stores:
                with self.assertRaisesRegex(ValueError, 'closed'):
                    _ = storage.directory_handle

    async def test_fixture_initialization_failure_joins_front_and_closes_storage(self):
        from loki_agent import credential_storages
        before = asyncio.all_tasks()
        failure = OSError('initialize fixture failed')
        storage = credential_storages.JsonCredentialStorage
        with mock.patch.object(_FrontInput, 'send', side_effect=failure), mock.patch.object(
                storage, 'close', autospec=True, side_effect=storage.close) as close:
            with self.assertRaises(OSError) as caught:
                async with self._approval_journey():
                    self.fail('initialization unexpectedly succeeded')
            self.assertIs(caught.exception, failure)
            close.assert_called_once()
        self.assertFalse(asyncio.all_tasks() - before)

    async def test_fixture_observes_reply_before_elicitation(self):
        async with self._approval_journey() as f:
            with mock.patch.object(f.front, '_approve_config_endpoint',
                                   side_effect=acps.TransportError('failed before elicitation')):
                with self.assertRaisesRegex(AssertionError, 'failed before elicitation'):
                    await self._pending_approval(f, 9)
            self._assert_no_approval_or_inference(f)
            await self._close_approval_journey(f)

    async def _assert_approval_refused(self, refusal):
        async with self._approval_journey() as f:
            for request_id in [9, 10]:
                task, elicitation = await self._pending_approval(f, request_id)
                await f.source.send(acps.response(elicitation['id'], result=refusal))
                await task
                reply = await f.wait_message(lambda message: message.get('id') == request_id)
                self.assertIn('error', reply, reply)
                self.assertEqual(reply['error'], {
                    'code': acps.INVALID_PARAMS,
                    'message': 'the provider endpoint was not approved'})
                self.assertIs(f.session.runtime_config, f.original_config)
                self.assertEqual(f.session.session_state['connection'],
                                 json.loads(f.original_chat)['session_state']['connection'])
                self.assertFalse(f.session.chat_log_dirty)
                with open(f.path, 'rb') as stream:
                    self.assertEqual(stream.read(), f.original_chat)
                with open(f.approvals_path, 'rb') as stream:
                    self.assertEqual(stream.read(), f.original_approvals_document)
                self.assertEqual(f.leases, [])
                self.assertEqual(f.connector.calls, [])
                self.assertFalse(f.front._client_requests)
                self.assertNotIn('switch', f.trace)
                self.assertNotIn('approval', f.trace)
            await self._close_approval_journey(f)

    async def test_declined_approval_fails_the_switch(self):
        await self._assert_approval_refused({'action': 'decline'})

    async def test_cancelled_approval_fails_the_switch(self):
        await self._assert_approval_refused({'action': 'cancel'})

    async def test_empty_approval_response_fails_the_switch(self):
        await self._assert_approval_refused({})

    async def test_list_approval_response_fails_the_switch(self):
        await self._assert_approval_refused([])

    async def test_false_approval_fails_the_switch(self):
        await self._assert_approval_refused({'action': 'accept', 'content': {'approve': False}})

    async def test_missing_approval_value_fails_the_switch(self):
        await self._assert_approval_refused({'action': 'accept', 'content': {}})

    async def test_numeric_approval_value_fails_the_switch(self):
        await self._assert_approval_refused({'action': 'accept', 'content': {'approve': 1}})

    async def test_string_approval_value_fails_the_switch(self):
        await self._assert_approval_refused({'action': 'accept', 'content': {'approve': 'true'}})

    def _assert_no_approval_or_inference(self, f):
        with open(f.approvals_path, 'rb') as stream:
            self.assertEqual(stream.read(), f.original_approvals_document)
        self.assertEqual(f.leases, [])
        self.assertEqual(f.connector.calls, [])
        self.assertNotIn('approval', f.trace)
        self.assertNotIn('switch', f.trace)

    async def test_pending_approval_rejects_buffered_work_only_in_its_session(self):
        async with self._approval_journey() as f:
            calls = []

            class OtherChannel:
                _closed = False
                process = mock.Mock(returncode=None)

                async def request(self, method, params, forwarded=None):
                    calls.append(method)
                    if forwarded is not None:
                        forwarded.set()
                    return {'stopReason': 'end_turn'}

                async def close(self):
                    pass

            other = f.front._reserve_session('other')
            other.channel = OtherChannel()
            other.state = 'active'
            await f.source.send_many(
                acps.request(9, 'session/set_config_option', f.params),
                acps.request(10, 'session/prompt', {
                    'sessionId': f.session_id,
                    'prompt': [{'type': 'text', 'text': 'must not run'}]}),
                acps.request(11, 'session/set_config_option', f.params),
                acps.request(12, 'session/prompt', {
                    'sessionId': 'other',
                    'prompt': [{'type': 'text', 'text': 'independent'}]}))
            for request_id in [10, 11]:
                reply = await f.wait_message(lambda m: m.get('id') == request_id)
                self.assertEqual(reply['error']['code'], acps.INVALID_PARAMS)
                self.assertIn('request was not executed', reply['error']['message'])
            self.assertEqual(await f.wait_message(lambda m: m.get('id') == 12),
                             acps.response(12, result={'stopReason': 'end_turn'}))
            self.assertEqual(calls, ['session/prompt'])
            ask = await f.wait_message(lambda m: m.get('method') == 'elicitation/create')
            self.assertEqual(ask['params']['requestId'], 9)
            self.assertEqual(f.processes[0].methods[-1], 'session/describe_config_selection')
            self._assert_no_approval_or_inference(f)
            with open(f.path, 'rb') as stream:
                self.assertEqual(stream.read(), f.original_chat)
            await f.source.send(acps.response(ask['id'], result={'action': 'decline'}))
            reply = await f.wait_message(lambda m: m.get('id') == 9)
            self.assertIn('not approved', reply['error']['message'])
            self.assertEqual(await f.request(80, 'session/close', {'sessionId': 'other'}), {})
            await self._close_approval_journey(f)

    async def test_close_invalidates_approval_before_accept_and_after_reopen(self):
        async with self._approval_journey() as f:
            task, ask = await self._pending_approval(f, 9)
            old_channel = f.channel
            await f.source.send_many(
                acps.request(10, 'session/close', {'sessionId': f.session_id}),
                acps.response(ask['id'], result={
                    'action': 'accept', 'content': {'approve': True}}))
            self.assertIn('error', await task)
            self.assertEqual(await f.wait_message(lambda m: m.get('id') == 10),
                             acps.response(10, result={}))
            self._assert_no_approval_or_inference(f)
            await f.request(11, 'session/resume', {
                'sessionId': f.session_id, 'cwd': f.workspace})
            f.channel = f.front.workers[f.session_id]
            self.assertIsNot(f.channel, old_channel)
            await f.source.send(acps.request(12, 'session/set_config_option', f.params))
            new_ask = await f.wait_message(
                lambda m: m.get('method') == 'elicitation/create'
                and m['params']['requestId'] == 12)
            self.assertNotEqual(new_ask['id'], ask['id'])
            await f.source.send(acps.response(ask['id'], result={
                'action': 'accept', 'content': {'approve': True}}))
            self.assertFalse(f.front._client_requests[new_ask['id']].done())
            self._assert_no_approval_or_inference(f)
            await f.source.send(acps.response(new_ask['id'], result={'action': 'cancel'}))
            self.assertIn('error', await f.wait_message(lambda m: m.get('id') == 12))
            await self._close_approval_journey(f)

    async def test_pending_approval_eof_releases_authority_without_a_reply(self):
        async with self._approval_journey() as f:
            task, _ask = await self._pending_approval(f, 9)
            # No response is owed over a disconnected transport.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await f.source.finish()
            self.assertFalse(any(m.get('id') == 9 for m in f.messages))
            self._assert_no_approval_or_inference(f)
            await self._assert_approval_released(f)

    async def test_pending_approval_worker_exit_releases_authority(self):
        async with self._approval_journey() as f:
            task, _ask = await self._pending_approval(f, 9)
            f.processes[0].close()
            self.assertIn('error', await task)
            await f.source.finish()
            self._assert_no_approval_or_inference(f)
            await self._assert_approval_released(f)

    async def test_active_prompt_rejects_model_change_before_asking(self):
        async with self._approval_journey() as f:
            entered = asyncio.Event()
            worker = f.processes[0].worker

            async def turn(on_event, reasoning_effort):
                entered.set()
                await worker.cancel_event.wait()
                on_event({'type': 'response_cancelled'})
                return ''

            with mock.patch.object(worker, '_run_turn', side_effect=turn):
                pending = asyncio.create_task(f.wait_message(lambda m: m.get('id') == 9))
                f.tasks.append(pending)
                await f.source.send(acps.request(9, 'session/prompt', {
                    'sessionId': f.session_id,
                    'prompt': [{'type': 'text', 'text': 'running turn'}]}))
                await observe_fixture(entered.wait(), pending, f.source.task)
                await f.source.send(acps.request(10, 'session/set_config_option', f.params))
                reply = await f.wait_message(lambda m: m.get('id') == 10)
                self.assertIn('prompt is running', reply['error']['message'])
                self.assertFalse(any(m.get('method') == 'elicitation/create' for m in f.messages))
                self._assert_no_approval_or_inference(f)
                await f.source.send(acps.notification('session/cancel', {'sessionId': f.session_id}))
                reply = await pending
                self.assertEqual(reply['result'], {'stopReason': 'cancelled'})
            await self._close_approval_journey(f)

    async def _assert_client_error_denies_duplicate_approval(self, error, expected):
        async with self._approval_journey() as f:
            task, ask = await self._pending_approval(f, 9)
            await f.source.send_many(
                acps.response(ask['id'], error=error),
                acps.response(ask['id'], result={
                    'action': 'accept', 'content': {'approve': True}}))
            reply = await task
            self.assertEqual(reply['error'], expected)
            self._assert_no_approval_or_inference(f)
            await self._close_approval_journey(f)

    async def test_client_error_and_duplicate_response_do_not_authorize(self):
        error = {'code': -32800, 'message': 'dismissed'}
        await self._assert_client_error_denies_duplicate_approval(error, error)

    async def test_malformed_client_error_and_duplicate_response_do_not_authorize(self):
        await self._assert_client_error_denies_duplicate_approval(['invalid error'], {
            'code': acps.INVALID_PARAMS,
            'message': 'ACP client response has an invalid error object'})

    async def test_unapproved_exact_urls_are_displayed(self):
        front = self._front()
        channel = self._channel(front, dict(self.PAIR))
        front._request_client = mock.AsyncMock(
            return_value={"action": "accept", "content": {"approve": True}})
        await front._approve_config_endpoint(
            channel, {"sessionId": "s"}, 9, owner=front._sessions['s'])
        message = front._request_client.await_args.args[1]["message"]
        for destination in self.PAIR["destinations"]:
            self.assertIn(destination, message)
        self.assertEqual(await front.credential_broker.unapproved_destinations(
            authentications.CredentialRef.environment('ACME_API_KEY'), self.PAIR['destinations']), [])

    async def test_nothing_to_approve_asks_nothing(self):
        front = self._front()
        channel = self._channel(front, {})
        front._request_client = mock.AsyncMock()
        await front._approve_config_endpoint(
            channel, {"sessionId": "s"}, 9, owner=front._sessions['s'])
        front._request_client.assert_not_awaited()

    async def test_client_without_form_elicitation_fails_closed(self):
        front = self._front(supports_elicitation=False)
        channel = self._channel(front, dict(self.PAIR))
        front._request_client = mock.AsyncMock()

        with self.assertRaises(acps.TransportError):
            await front._approve_config_endpoint(
                channel, {"sessionId": "s"}, 9, owner=front._sessions['s'])

        front._request_client.assert_not_awaited()


class _AskLoopProcess(_LocalWorkerProcess):
    """Both transport ends in one object for model-question round trips.

    The real WorkerChannel reads this object as its child process; the real
    Worker writes reverse requests into it and consumes the front's reverse
    responses. Front->worker session/cancel requests are dispatched by the
    real Worker too; the fixture supplies byte transport, not protocol replies.
    """

    def __init__(self, session):
        super().__init__(session, 'worker', lambda message: None)
        self.worker.write = self._feed
        self.replies = []
        self.asks = []
        self.changed = asyncio.Event()

    def ask(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.asks.append(task)
        return task

    def _feed(self, message):
        self.stdout.feed_data(
            (json.dumps(message) + "\n").encode("utf-8"))

    async def drain(self):
        while b"\n" in self.buffer:
            line, _, rest = self.buffer.partition(b"\n")
            self.buffer = bytearray(rest)
            message = json.loads(line)
            if message.get("method") is not None:
                self.methods.append(message["method"])
                if message["method"] != "session/cancel":
                    # A new front->worker method flowing through this harness
                    # would be silently no-oped below; fail loud instead so
                    # the test author answers it deliberately.
                    raise AssertionError(
                        "test harness received unhandled front->worker "
                        f"method {message['method']!r}")
                await self.worker.handle(message, concurrent=True)
            else:
                self.replies.append(message)
                self.worker.resolve_reverse(message["id"], message)
                self.changed.set()


class WorkerAskElicitationTests(unittest.IsolatedAsyncioTestCase):
    """Model questions travel worker -> front -> elicitation/create -> back."""

    def _owner(self, front, session_id="ask-session"):
        owner = front._reserve_session(session_id)
        owner.state = "active"
        return owner

    @asynccontextmanager
    async def _roundtrip(self, *, advertise=True):
        from loki_agent.sessions import Session

        async with AsyncExitStack() as cleanup:
            session = Session()
            cleanup.push_async_callback(session.close_state)
            messages = []
            process = _AskLoopProcess(session)

            def client_write(message):
                messages.append(message)
                process.changed.set()

            front = acp.Front(lambda: None, client_write, CredentialStore({}))
            if advertise:
                front.initialize({
                    "clientCapabilities": {"elicitation": {"form": {}}}})
            owner = self._owner(front)
            channel = acp.WorkerChannel(
                owner.session_id, process, lambda message: None, None,
                reverse_handler=lambda message: front._worker_request(
                    owner, message))
            owner.channel = channel
            self._seen_elicitations = set()
            try:
                yield front, process, messages
            finally:
                for task in process.asks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*process.asks, return_exceptions=True)
                await front.shutdown()
                self.assertEqual(process.returncode, 0)
                self.assertTrue(process.close_task.done())
                self.assertIsNone(process.close_task.exception())
                self.assertTrue(channel._reader_task.done())
                self.assertFalse(front._client_requests)
                self.assertFalse(process.worker._reverse_requests)

    async def _next_elicitation(self, front, process, messages, ask_task):
        while True:
            process.changed.clear()
            for message in messages:
                if (message.get("method") == "elicitation/create"
                        and message["id"] not in self._seen_elicitations):
                    self._seen_elicitations.add(message["id"])
                    return message
            await observe_fixture(
                process.changed.wait(), ask_task, front.workers['ask-session']._reader_task)

    async def _answer(self, front, process, messages, ask_task, result):
        elicitation = await self._next_elicitation(front, process, messages, ask_task)
        front.handle(acps.response(elicitation["id"], result=result))
        return elicitation

    async def test_roundtrip_cleanup_joins_unstarted_and_pending_asks(self):
        before = asyncio.all_tasks()
        for started in [False, True]:
            with self.subTest(started=started):
                async with self._roundtrip() as (front, process, messages):
                    ask_task = process.ask(process.worker._ask_user(
                        'Unanswered?', [('Yes', None), ('No', None)]))
                    if started:
                        await self._next_elicitation(front, process, messages, ask_task)
                self.assertTrue(ask_task.cancelled())
                self.assertFalse(asyncio.all_tasks() - before)

    async def test_roundtrip_closes_its_session_storage(self):
        async with self._roundtrip() as (_front, process, _messages):
            storage = process.worker.session.response_headers
        with self.assertRaisesRegex(ValueError, 'closed'):
            _ = storage.directory_handle

    async def test_ask_roundtrip_builds_the_form_and_returns_the_answer(self):
        async with self._roundtrip() as (front, process, messages):
            ask_task = process.ask(process.worker._ask_user(
                "Which storage?", [("Local", "on disk"), ("Remote", "S3")]))
            elicitation = await self._answer(
                front, process, messages, ask_task,
                {"action": "accept",
                 "content": {"answer": "Remote", "custom": "via VPN"}})
            self.assertEqual(await ask_task, {
                "action": "answered", "answer": "Remote",
                "custom": "via VPN"})

            params = elicitation["params"]
            self.assertEqual(params["sessionId"], "ask-session")
            self.assertEqual(params["mode"], "form")
            self.assertEqual(params["message"], "Which storage?")
            self.assertNotIn("requestId", params)
            self.assertNotIn("url", params)
            schema = params["requestedSchema"]
            self.assertEqual(schema["required"], [])
            answer = schema["properties"]["answer"]
            self.assertEqual(answer["type"], "string")
            self.assertEqual(
                answer["oneOf"],
                [{"const": "Local", "title": "Local",
                  "description": "on disk"},
                 {"const": "Remote", "title": "Remote",
                  "description": "S3"}])
            self.assertEqual(schema["properties"]["custom"]["type"], "string")

    async def test_custom_text_alone_becomes_the_answer(self):
        async with self._roundtrip() as (front, process, messages):
            ask_task = process.ask(process.worker._ask_user(
                "Proceed?", [("Yes", None), ("No", None)]))
            await self._answer(
                front, process, messages, ask_task,
                {"action": "accept", "content": {"custom": " maybe later "}})
            self.assertEqual(await ask_task, {
                "action": "answered", "answer": "maybe later"})

    async def test_multi_select_uses_an_array_schema_and_answer(self):
        async with self._roundtrip() as (front, process, messages):
            ask_task = process.ask(process.worker._ask_user(
                "Which parts?", [("Legs", None), ("Roof", None)],
                multi_select=True))
            elicitation = await self._answer(
                front, process, messages, ask_task,
                {"action": "accept",
                 "content": {"answer": ["Legs", "Roof"]}})
            self.assertEqual(await ask_task, {
                "action": "answered", "answer": ["Legs", "Roof"]})
            items = elicitation["params"]["requestedSchema"]["properties"]["answer"]
            self.assertEqual(items["type"], "array")
            self.assertEqual(len(items["items"]["anyOf"]), 2)

    async def test_decline_is_not_a_cancel(self):
        async with self._roundtrip() as (front, process, messages):
            ask_task = process.ask(process.worker._ask_user(
                "Proceed?", [("Yes", None), ("No", None)]))
            await self._answer(front, process, messages, ask_task, {"action": "decline"})
            self.assertEqual(
                await ask_task, {"action": "declined"})

            ask_task = process.ask(process.worker._ask_user(
                "Proceed?", [("Yes", None), ("No", None)]))
            await self._answer(front, process, messages, ask_task, {"action": "cancel"})
            self.assertEqual(await ask_task, {"action": "cancelled"})

    async def test_no_answer_in_an_accept_is_a_worker_error(self):
        async with self._roundtrip() as (front, process, messages):
            ask_task = process.ask(process.worker._ask_user(
                "Proceed?", [("Yes", None), ("No", None)]))
            await self._answer(front, process, messages, ask_task,
                               {"action": "accept", "content": {}})
            with self.assertRaisesRegex(
                    acps.TransportError, "contained no answer"):
                await ask_task

    async def test_missing_capability_fails_the_ask_closed(self):
        async with self._roundtrip(advertise=False) as (
                front, process, messages):
            with self.assertRaisesRegex(
                    acps.TransportError, "cannot answer model questions"):
                await process.worker._ask_user(
                    "Proceed?", [("Yes", None), ("No", None)])
            self.assertFalse(any(
                message.get("method") == "elicitation/create"
                for message in messages))

    async def test_session_cancel_settles_the_pending_ask(self):
        async with self._roundtrip() as (front, process, messages):
            ask_task = process.ask(process.worker._ask_user(
                "Proceed?", [("Yes", None), ("No", None)]))
            elicitation = await self._next_elicitation(front, process, messages, ask_task)
            pending = set(front._tasks)
            front.handle(acps.request(
                91, "session/cancel", {"sessionId": "ask-session"}))
            await asyncio.gather(*(front._tasks - pending))
            self.assertEqual(
                await ask_task,
                {"action": "cancelled"})
            self.assertTrue(process.worker.cancel_event.is_set())
            # Completion above already witnessed the worker's reverse reply.
            self.assertTrue(any(reply.get("result") == {"action": "cancelled"}
                                for reply in process.replies))
            # A late client answer resolves nothing and raises nothing.
            front.handle(acps.response(
                elicitation["id"],
                result={"action": "accept",
                        "content": {"answer": "Yes"}}))
            self.assertFalse(front._client_requests)

    async def test_unknown_worker_method_is_refused_not_dropped(self):
        async with self._roundtrip() as (front, process, messages):
            process._feed(acps.request(
                "ask-x", "session/plumb", {}))
            await observe_fixture(process.changed.wait(), front.workers['ask-session']._reader_task)
            error = process.replies[0]["error"]
            self.assertEqual(error["code"], acps.METHOD_NOT_FOUND)
            self.assertIn("session/plumb", error["message"])

    async def test_ask_during_cancelled_turn_reports_cancelled(self):
        async with self._roundtrip() as (front, process, messages):
            worker = process.worker
            worker.cancel_event.set()
            ask_task = process.ask(worker._ask_user(
                "Proceed?", [("Yes", None), ("No", None)]))
            elicitation = await self._next_elicitation(front, process, messages, ask_task)
            front.handle(acps.response(
                elicitation["id"],
                result={"action": "accept", "content": {"answer": "Yes"}}))
            # The ask still went out and was answered, but an answer that
            # arrives after the turn's cancellation must not drive the
            # cancelled turn onward.
            self.assertEqual(
                await ask_task, {"action": "cancelled"})

    async def test_worker_request_input_reaches_the_client_session_scoped(self):
        async with self._roundtrip() as (front, process, messages):
            ask_task = process.ask(process.worker.request_input(
                "Proceed?", [("Yes", None), ("No", None)]))
            elicitation = await self._next_elicitation(front, process, messages, ask_task)
            front.handle(acps.response(elicitation["id"], result={
                "action": "accept", "content": {"answer": "Yes"}}))
            self.assertEqual(await ask_task, {
                "action": "answered", "answer": "Yes"})
            self.assertEqual(elicitation["params"]["sessionId"], "ask-session")
            self.assertEqual(elicitation["params"]["mode"], "form")

    async def test_late_reverse_reply_is_discarded(self):
        async with self._roundtrip() as (front, process, messages):
            worker = process.worker
            ask_task = process.ask(worker._ask_user(
                "Proceed?", [("Yes", None), ("No", None)]))
            elicitation = await self._next_elicitation(front, process, messages, ask_task)
            front.handle(acps.response(elicitation["id"], result={
                "action": "accept", "content": {"answer": "Yes"}}))
            self.assertEqual(
                await ask_task,
                {"action": "answered", "answer": "Yes"})
            # The reverse request was settled and removed. A second response
            # for it is a deterministic no-op: nothing raises, nothing stays
            # pending.
            reverse_id = process.replies[0]["id"]
            worker.resolve_reverse(
                reverse_id, {"result": {"action": "answered", "answer": "x"}})
            self.assertEqual(worker._reverse_requests, {})

    async def test_worker_request_without_handler_is_refused(self):
        # A channel built without a reverse handler must answer a worker
        # request with an error, never drop it and leave the worker waiting.
        from loki_agent.sessions import Session

        async with AsyncExitStack() as cleanup:
            session = Session()
            cleanup.push_async_callback(session.close_state)
            process = _AskLoopProcess(session)
            channel = acp.WorkerChannel("s", process, lambda message: None, None)
            try:
                process._feed(acps.request(
                    "ask-7", "session/request_input", {
                        "question": "q?",
                        "options": [{"label": "a"}, {"label": "b"}]}))
                await observe_fixture(process.changed.wait(), channel._reader_task)
                error = process.replies[0]["error"]
                self.assertEqual(error["code"], acps.METHOD_NOT_FOUND)
                self.assertIn("cannot accept", error["message"])
            finally:
                await channel.close()

    async def test_close_fails_a_pending_question(self):
        async with self._roundtrip() as (front, process, messages):
            worker = process.worker
            ask_task = process.ask(worker._ask_user(
                "Proceed?", [("Yes", None), ("No", None)]))
            await self._next_elicitation(front, process, messages, ask_task)
            await worker.close()
            with self.assertRaises(acps.TransportError):
                await ask_task


class WorkerAskAdvertisementTests(unittest.IsolatedAsyncioTestCase):
    async def _offered_tools(self, capable):
        from loki_agent import formats, loki, model_requests
        from loki_agent.acp_worker import Worker
        from response_header_fixtures import isolated_response_headers

        with isolated_response_headers():
            async with AsyncExitStack() as cleanup:
                session = loki.current_session()
                cleanup.push_async_callback(session.close_state)
                worker = Worker(session, lambda message: None)
                worker.can_ask_user = capable
                offered = []

                async def completion(_items, *, tools, **_kwargs):
                    offered.append(tools)
                    return formats.DecodedTurn([formats.message_item('assistant', 'done')])

                with mock.patch.object(loki, 'current_model', return_value='model'), mock.patch.object(
                        loki, 'async_chat_completion', new=completion):
                    result = await worker._run_turn(lambda _event: None, model_requests.TurnThinkingSettings())
                self.assertEqual(result, 'done')
                self.assertEqual(len(offered), 1)
                self.assertEqual(session.session_toolsets, offered)
                return {tool['function']['name'] for tool in offered[0]}

    async def test_ask_is_withheld_without_capability(self):
        self.assertNotIn('Ask', await self._offered_tools(False))

    async def test_ask_is_advertised_with_capability(self):
        self.assertIn('Ask', await self._offered_tools(True))


class FrontAskValidationTests(unittest.IsolatedAsyncioTestCase):
    """The front builds the schema; workers may not set scope or mode."""

    def _front(self, messages, *, advertise=True):
        front = acp.Front(lambda: None, messages.append, CredentialStore({}))
        if advertise:
            front.initialize({
                "clientCapabilities": {"elicitation": {"form": {}}}})
        self.owner = front._reserve_session("s")
        self.owner.state = "active"
        return front

    async def assert_refused(self, front, messages, params, pattern):
        with self.assertRaisesRegex(acps.TransportError, pattern):
            await front._ask_user_via_elicitation(params, owner=self.owner)
        self.assertEqual(
            [message for message in messages
             if message.get("method") == "elicitation/create"],
            [])

    async def test_workers_may_not_set_scope_or_mode(self):
        messages = []
        for params in [
                {"requestId": 5},
                {"mode": "url"},
                {"sessionId": "other"},
                {"url": "https://example.com"},
                {"elicitationId": "e1"}]:
            with self.subTest(params=params):
                front = self._front(messages)
                await self.assert_refused(
                    front, messages, {"question": "q?", "options": [
                        {"label": "a"}, {"label": "b"}], **params},
                    "belong to the front")

    async def test_question_and_option_bounds(self):
        messages = []
        base = {"options": [{"label": "a"}, {"label": "b"}]}
        cases = [
            ({"question": "", **base}, "nonempty string"),
            ({"question": "x" * 4001, **base}, "at most"),
            ({**base}, "nonempty string"),
            ({"question": "q?", "options": [{"label": "a"}]}, "2 to 8"),
            ({"question": "q?", "options": [
                {"label": "a"}, {"label": "a"}]}, "unique"),
            ({"question": "q?", "options": [
                {"label": ""}, {"label": "b"}]}, "nonempty strings"),
            ({"question": "q?", "options": [
                {"label": "a"}, {"label": "b"}], "multiSelect": "yes"},
             "boolean"),
        ]
        for params, pattern in cases:
            with self.subTest(params=params):
                front = self._front(messages)
                await self.assert_refused(front, messages, params, pattern)

    async def test_schema_is_front_built_and_session_scoped(self):
        messages = []

        def write(message):
            messages.append(message)
            if message.get("method") == "elicitation/create":
                front.handle(acps.response(message["id"], result={
                    "action": "accept", "content": {"answer": "b"}}))

        front = acp.Front(lambda: None, write, CredentialStore({}))
        front.initialize({
            "clientCapabilities": {"elicitation": {"form": {}}}})
        self.owner = front._reserve_session("s")
        self.owner.state = "active"
        outcome = await front._ask_user_via_elicitation(
            {"question": "q?", "options": [{"label": "a"}, {"label": "b"}]},
            owner=self.owner)
        self.assertEqual(outcome, {"action": "answered", "answer": "b"})
        elicitation = next(
            message for message in messages
            if message.get("method") == "elicitation/create")
        self.assertEqual(elicitation["params"]["sessionId"], "s")
        self.assertEqual(elicitation["params"]["mode"], "form")


if __name__ == "__main__":
    unittest.main()
