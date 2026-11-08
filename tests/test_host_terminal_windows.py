"""Tests for the Windows console backend.

Console mode checks require Windows. The import-safe handle reader's native
calls are replaced in portable tests so shutdown races are exercised on Linux
as well as by the real Windows PTY tests.
"""

import asyncio
import contextlib
import ctypes
import sys
import threading
import unittest
from ctypes import wintypes
from unittest import mock

from loki_agent import handle_reader
from process_lifecycle_fixtures import observe_fixture

if sys.platform == "win32":
    from loki_agent import host_terminal_windows


@unittest.skipUnless(sys.platform == "win32", "Windows console APIs")
class RawInputModeTests(unittest.TestCase):
    def test_clears_line_echo_and_processed_and_sets_vt_input(self):
        original = (host_terminal_windows.ConsoleInputMode.ENABLE_LINE_INPUT
                    | host_terminal_windows.ConsoleInputMode.ENABLE_ECHO_INPUT
                    | host_terminal_windows.ConsoleInputMode.ENABLE_PROCESSED_INPUT
                    | host_terminal_windows.ConsoleInputMode.ENABLE_EXTENDED_FLAGS)

        mode = host_terminal_windows.raw_input_mode(original)

        self.assertEqual(
            mode, host_terminal_windows.ConsoleInputMode.ENABLE_EXTENDED_FLAGS
            | host_terminal_windows.ConsoleInputMode.ENABLE_VIRTUAL_TERMINAL_INPUT)

    def test_other_bits_are_left_alone(self):
        mode = host_terminal_windows.raw_input_mode(0x0000)
        self.assertEqual(
            mode,
            host_terminal_windows.ConsoleInputMode.ENABLE_VIRTUAL_TERMINAL_INPUT)


@unittest.skipUnless(sys.platform == "win32", "Windows console APIs")
class ControlBytesTests(unittest.TestCase):
    def test_the_caller_defaults_stand(self):
        fallbacks = (frozenset((0x7f,)), frozenset((0x17,)), frozenset((0x03,)))
        self.assertIs(host_terminal_windows.control_bytes(fallbacks), fallbacks)


class _FakeQueue:
    maxsize = 0

    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


class _InlineLoop:
    """Run the posted callback where it is posted from."""

    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


class ReaderBackpressureTests(unittest.IsolatedAsyncioTestCase):
    @contextlib.asynccontextmanager
    async def blocked_delivery(self):
        entered, finished = asyncio.Event(), asyncio.Event()

        class Queue(asyncio.Queue):
            async def put(self, item):
                entered.set()
                await super().put(item)

        queue = Queue(maxsize=1)
        queue.put_nowait(b'accepted')
        loop = asyncio.get_running_loop()
        reader = handle_reader.HandleReader(7, loop, queue)
        reader._accepting = True
        reader._cancel_thread = handle_reader.Future()
        reader._cancel_thread.set_result(None)

        def produce():
            reader._post(b'pending')
            loop.call_soon_threadsafe(finished.set)

        reader.thread = threading.Thread(target=produce)
        with mock.patch.object(loop, 'call_exception_handler') as reported:
            reader.thread.start()
            joined = asyncio.create_task(reader.wait_closed())
            try:
                await observe_fixture(entered.wait(), joined)
                yield reader, queue, finished, joined
                reported.assert_not_called()
            finally:
                await reader.aclose()
                await asyncio.gather(joined, return_exceptions=True)

    async def test_bounded_queue_stops_native_read_ahead_until_consumed(self):
        async with self.blocked_delivery() as (reader, queue, finished, joined):
            self.assertEqual(queue.qsize(), 1)
            self.assertFalse(finished.is_set())
            self.assertEqual(queue.get_nowait(), b'accepted')
            self.assertEqual(await queue.get(), b'pending')
            await joined
            self.assertTrue(finished.is_set())
            self.assertTrue(reader.closed)

    async def test_stop_joins_a_producer_waiting_for_queue_space(self):
        async with self.blocked_delivery() as (reader, queue, finished, joined):
            await reader.aclose()
            await joined
            self.assertTrue(finished.is_set())
            self.assertTrue(reader.closed)
            self.assertEqual(queue.get_nowait(), b'accepted')
            self.assertTrue(queue.empty())


class ReaderThreadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # These authored handles test thread ownership, not native console
        # records; notification/byte ordering has its own cases below.
        patch = mock.patch.object(handle_reader.windows_api, 'discard_console_notification', return_value=False)
        self.notifications = patch.start()
        self.addCleanup(patch.stop)

    @contextlib.asynccontextmanager
    async def source(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(handle_reader.sys, 'platform', 'win32'))
            calls = stack.enter_context(mock.patch.multiple(
                handle_reader, _handle=mock.DEFAULT, _GetFileType=mock.DEFAULT,
                _CreateEventW=mock.DEFAULT, _WaitForMultipleObjects=mock.DEFAULT,
                _ReadFile=mock.DEFAULT, _SetEvent=mock.DEFAULT,
                _DuplicateCurrentThread=mock.DEFAULT, _CancelSynchronousIo=mock.DEFAULT,
                _CloseHandle=mock.DEFAULT))
            stack.enter_context(mock.patch.object(
                handle_reader.ctypes, 'get_last_error', return_value=5, create=True))
            calls['_handle'].return_value = 1
            calls['_GetFileType'].return_value = 2
            calls['_CreateEventW'].return_value = 2
            calls['_WaitForMultipleObjects'].return_value = handle_reader.WAIT_OBJECT_0
            calls['_ReadFile'].return_value = False
            calls['_SetEvent'].return_value = True
            calls['_DuplicateCurrentThread'].return_value = 77
            calls['_CancelSynchronousIo'].return_value = True
            calls['_CloseHandle'].return_value = True
            reader = handle_reader.HandleReader(7, _InlineLoop(), _FakeQueue())
            try:
                yield reader, calls
            finally:
                await reader.aclose()

    async def test_notifications_are_consumed_without_reading_and_keys_keep_the_byte_path(self):
        self.notifications.side_effect = [True, True, False]

        def read(handle, buffer, size, count, overlapped):
            ctypes.memmove(buffer, b'key', 3)
            count._obj.value = 3
            return True

        async with self.source() as (reader, calls):
            calls['_WaitForMultipleObjects'].side_effect = [
                handle_reader.WAIT_OBJECT_0 + 1, handle_reader.WAIT_OBJECT_0 + 1,
                handle_reader.WAIT_OBJECT_0 + 1, handle_reader.WAIT_OBJECT_0]
            calls['_ReadFile'].side_effect = read
            reader.start()
            reader.thread.join()
            self.assertEqual(reader.queue.items, [b'key'])
            self.assertEqual(calls['_ReadFile'].call_count, 1)
            self.assertEqual(self.notifications.call_count, 3)

    async def test_console_query_and_byte_read_errors_wake_the_consumer(self):
        for outcome in ['notification', 'read']:
            async with self.source() as (reader, calls):
                self.notifications.side_effect = (handle_reader.windows_api.WindowsApiError('query failed', status=5)
                                                  if outcome == 'notification' else None)
                calls['_WaitForMultipleObjects'].return_value = handle_reader.WAIT_OBJECT_0 + 1
                reader.start()
                reader.thread.join()
                self.assertEqual(len(reader.queue.items), 1)
                self.assertIsInstance(reader.queue.items[0], OSError)
                self.assertEqual(reader.queue.items[0].errno, 5)
                self.assertEqual(calls['_ReadFile'].call_count, 0 if outcome == 'notification' else 1)

    async def test_acquisition_failure_prevents_read_and_propagates_from_start(self):
        failure = OSError('cannot acquire private thread')
        async with self.source() as (reader, calls):
            calls['_DuplicateCurrentThread'].side_effect = failure
            with self.assertRaises(OSError) as caught:
                reader.start()
            self.assertIs(caught.exception, failure)
            await reader.wait_closed()
            calls['_ReadFile'].assert_not_called()
            calls['_CancelSynchronousIo'].assert_not_called()
            self.assertIsNone(reader.thread)
            self.assertIsNone(reader.stop_event)
            self.assertIsNone(reader._cancel_thread)
            self.assertEqual(calls['_CloseHandle'].call_args_list, [mock.call(2)])

    async def test_failed_stop_retains_the_same_cancellation_capability_for_retry(self):
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def read(*args):
            loop.call_soon_threadsafe(entered.set)
            release.wait()
            return False

        async with self.source() as (reader, calls):
            calls['_WaitForMultipleObjects'].return_value = handle_reader.WAIT_OBJECT_0 + 1
            calls['_ReadFile'].side_effect = read
            reader.start()
            try:
                await entered.wait()
                calls['_CancelSynchronousIo'].return_value = False
                with self.assertRaises(OSError):
                    await reader.aclose()
                self.assertTrue(reader.thread.is_alive())
                self.assertIsNotNone(reader.stop_event)
                self.assertEqual(reader._cancel_thread.result(), 77)
                calls['_CloseHandle'].assert_not_called()
                reader._post(b'late')
                self.assertEqual(reader.queue.items, [])
            finally:
                release.set()
                calls['_CancelSynchronousIo'].return_value = True
                await reader.aclose()
            calls['_DuplicateCurrentThread'].assert_called_once_with()
            self.assertEqual(calls['_CloseHandle'].call_args_list, [mock.call(77), mock.call(2)])

    async def test_close_failure_consumes_handles_without_retry_or_thread_reopen(self):
        async with self.source() as (reader, calls):
            reader.start()
            reader.thread.join()
            calls['_CloseHandle'].side_effect = [False, True]
            with self.assertRaises(OSError):
                await reader.aclose()
            self.assertIsNone(reader.thread)
            self.assertIsNone(reader.stop_event)
            self.assertIsNone(reader._cancel_thread)
            self.assertTrue(reader.closed)
            await reader.aclose()
            calls['_DuplicateCurrentThread'].assert_called_once_with()
            self.assertEqual(calls['_CloseHandle'].call_args_list, [mock.call(77), mock.call(2)])

    async def test_interrupted_start_retains_a_handle_published_after_failed_rollback(self):
        acquiring, release = threading.Event(), threading.Event()
        thread_class = threading.Thread

        def duplicate():
            acquiring.set()
            release.wait()
            return 77

        def make_thread(*args, **kwargs):
            thread = thread_class(*args, **kwargs)
            start = thread.start

            def interrupted():
                start()
                acquiring.wait()
                raise KeyboardInterrupt('interrupted native startup')
            thread.start = interrupted
            return thread

        async with self.source() as (reader, calls):
            calls['_DuplicateCurrentThread'].side_effect = duplicate
            closing = None
            try:
                with mock.patch.object(handle_reader.threading, 'Thread', side_effect=make_thread):
                    with self.assertRaisesRegex(KeyboardInterrupt, 'interrupted native startup'):
                        reader.start()
                self.assertFalse(reader._cancel_thread.done())
                self.assertIsNotNone(reader.stop_event)
                closing = asyncio.create_task(reader.aclose())
                await asyncio.sleep(0)
                self.assertFalse(closing.done())
                calls['_CloseHandle'].assert_not_called()
            finally:
                release.set()
                if closing is not None:
                    await closing
                else:
                    await reader.aclose()
            calls['_ReadFile'].assert_not_called()
            self.assertEqual(calls['_CloseHandle'].call_args_list, [mock.call(77), mock.call(2)])

    async def test_a_signalled_read_is_posted_then_the_stop_event_ends_the_thread(self):
        queue = _FakeQueue()
        waits = iter([handle_reader.WAIT_OBJECT_0 + 1, handle_reader.WAIT_OBJECT_0])

        def wait_for_objects(count, handles, wait_all, timeout):
            return next(waits)

        def read_file(handle, buffer, size, read, overlapped):
            ctypes.memmove(buffer, b"abc", 3)
            ctypes.cast(read, ctypes.POINTER(wintypes.DWORD)).contents.value = 3
            return True

        with mock.patch.object(handle_reader.sys, 'platform', 'win32'), \
                mock.patch.object(handle_reader, '_GetFileType', return_value=2), \
                mock.patch.object(handle_reader, "_handle",
                                  return_value=1), \
                mock.patch.object(handle_reader, "_CreateEventW",
                                  return_value=2), \
                mock.patch.object(handle_reader,
                                  "_WaitForMultipleObjects",
                                  side_effect=wait_for_objects), \
                mock.patch.object(handle_reader, "_ReadFile",
                                  side_effect=read_file), \
                mock.patch.object(handle_reader, "_SetEvent"), \
                mock.patch.object(handle_reader, "_DuplicateCurrentThread", return_value=77), \
                mock.patch.object(handle_reader, "_CloseHandle"):
            reader = handle_reader.HandleReader(7, _InlineLoop(), queue)
            reader.start()
            reader.thread.join()
            self.assertFalse(reader.thread.is_alive())
            await reader.aclose()

        self.assertEqual(queue.items, [b"abc"])

    async def test_stop_does_not_close_the_event_under_a_live_thread(self):
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def read(*args):
            loop.call_soon_threadsafe(entered.set)
            release.wait()
            return False

        async with self.source() as (reader, calls):
            calls['_WaitForMultipleObjects'].return_value = handle_reader.WAIT_OBJECT_0 + 1
            calls['_ReadFile'].side_effect = read
            reader.start()
            closing = []
            try:
                await entered.wait()
                closing = [asyncio.create_task(reader.aclose()) for _ in range(2)]
                responsive = loop.create_future()
                loop.call_soon(responsive.set_result, None)
                await responsive
                self.assertTrue(reader.thread.is_alive())
                self.assertTrue(all(not task.done() for task in closing))
                calls['_CloseHandle'].assert_not_called()
            finally:
                release.set()
                await asyncio.gather(*closing)
                await reader.aclose()
            self.assertIsNone(reader.stop_event)
            self.assertTrue(reader.closed)
            self.assertEqual(calls['_CloseHandle'].call_args_list, [mock.call(77), mock.call(2)])

    async def test_stop_retries_cancellation_when_the_read_enters_after_first_cancel(self):
        announced = asyncio.Event()
        loop = asyncio.get_running_loop()
        allow_read, pending, cancelled = [threading.Event() for _ in range(3)]
        calls = []
        acquired_on = []

        def duplicate_thread():
            acquired_on.append(threading.get_native_id())
            return 77

        def read_file(*arguments):
            loop.call_soon_threadsafe(announced.set)
            allow_read.wait()
            pending.set()
            cancelled.wait()
            return False

        def cancel_io(handle):
            calls.append(handle)
            if len(calls) == 1:
                # The first request precedes native read entry: NOT_FOUND is
                # not a promise that the reader cannot block immediately next.
                allow_read.set()
                return False
            if not pending.is_set():
                return False
            cancelled.set()
            return True

        with (
            mock.patch.object(handle_reader.sys, 'platform', 'win32'),
            mock.patch.object(handle_reader, '_handle', return_value=1),
            mock.patch.object(handle_reader, '_GetFileType', return_value=2),
            mock.patch.object(handle_reader, '_CreateEventW', return_value=2),
            mock.patch.object(handle_reader, '_WaitForMultipleObjects', return_value=handle_reader.WAIT_OBJECT_0 + 1),
            mock.patch.object(handle_reader, '_ReadFile', side_effect=read_file),
            mock.patch.object(handle_reader, '_SetEvent', return_value=True),
            mock.patch.object(handle_reader, '_DuplicateCurrentThread', side_effect=duplicate_thread) as duplicate,
            mock.patch.object(handle_reader, '_CancelSynchronousIo', side_effect=cancel_io),
            mock.patch.object(handle_reader.ctypes, 'get_last_error',
                              return_value=handle_reader.windows_api.ERROR_NOT_FOUND, create=True),
            mock.patch.object(handle_reader, '_CloseHandle', return_value=True) as close,
        ):
            reader = handle_reader.HandleReader(7, _InlineLoop(), _FakeQueue())
            reader.start()
            try:
                await announced.wait()
                native_id = reader.thread.native_id
                await reader.aclose()
                self.assertIsNone(reader.thread)
                self.assertIsNone(reader.stop_event)
                self.assertIsNone(reader._cancel_thread)
                self.assertGreaterEqual(len(calls), 2)
                duplicate.assert_called_once_with()
                self.assertEqual(acquired_on, [native_id])
                self.assertNotEqual(native_id, threading.get_native_id())
                self.assertEqual(close.call_args_list, [mock.call(77), mock.call(2)])
            finally:
                allow_read.set()
                cancelled.set()
                await reader.aclose()


class DuplicateThreadHandleTests(unittest.TestCase):
    def test_duplicates_the_worker_pseudo_handle_with_only_cancellation_rights(self):
        seen = []

        def duplicate(source, thread, target, output, access, inherit, flags):
            seen.append([source, thread, target, access, inherit, flags])
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 77
            return True

        def bind(library, name, *signature):
            return {'GetCurrentThread': lambda: -2, 'DuplicateHandle': duplicate}[name]

        with mock.patch.object(handle_reader.windows_api, 'current_process_handle', return_value=10), \
                mock.patch.object(handle_reader.windows_api, 'bind', side_effect=bind):
            self.assertEqual(handle_reader._DuplicateCurrentThread(), 77)
        self.assertEqual(seen, [[10, -2, 10, handle_reader.windows_api.ThreadAccess.THREAD_TERMINATE, False, 0]])

    def test_failed_duplication_does_not_publish_an_unusable_handle(self):
        def bind(library, name, *signature):
            return {'GetCurrentThread': lambda: -2, 'DuplicateHandle': lambda *args: False}[name]

        with mock.patch.object(handle_reader.windows_api, 'current_process_handle', return_value=10), \
                mock.patch.object(handle_reader.windows_api, 'bind', side_effect=bind), \
                mock.patch.object(handle_reader.ctypes, 'get_last_error', return_value=5, create=True):
            with self.assertRaises(OSError):
                handle_reader._DuplicateCurrentThread()


if __name__ == "__main__":
    unittest.main()
