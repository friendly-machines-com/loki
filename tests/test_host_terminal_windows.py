"""Tests for the Windows console backend.

Console mode checks require Windows. The import-safe handle reader's native
calls are replaced in portable tests so shutdown races are exercised on Linux
as well as by the real Windows PTY tests.
"""

import contextlib
import ctypes
import sys
import threading
import unittest
from ctypes import wintypes
from unittest import mock

from loki_agent import handle_reader

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
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


class _InlineLoop:
    """Run the posted callback where it is posted from."""

    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


class ReaderThreadTests(unittest.TestCase):
    @contextlib.contextmanager
    def source(self):
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
                reader.stop()

    def test_acquisition_failure_prevents_read_and_propagates_from_start(self):
        failure = OSError('cannot acquire private thread')
        with self.source() as (reader, calls):
            calls['_DuplicateCurrentThread'].side_effect = failure
            with self.assertRaises(OSError) as caught:
                reader.start()
            self.assertIs(caught.exception, failure)
            calls['_ReadFile'].assert_not_called()
            calls['_CancelSynchronousIo'].assert_not_called()
            self.assertIsNone(reader.thread)
            self.assertIsNone(reader.stop_event)
            self.assertIsNone(reader._cancel_thread)
            self.assertEqual(calls['_CloseHandle'].call_args_list, [mock.call(2)])

    def test_failed_stop_retains_the_same_cancellation_capability_for_retry(self):
        entered, release = threading.Event(), threading.Event()

        def read(*args):
            entered.set()
            release.wait()
            return False

        with self.source() as (reader, calls):
            calls['_WaitForMultipleObjects'].return_value = handle_reader.WAIT_OBJECT_0 + 1
            calls['_ReadFile'].side_effect = read
            reader.start()
            try:
                entered.wait()
                calls['_CancelSynchronousIo'].return_value = False
                with self.assertRaises(OSError):
                    reader.stop()
                self.assertTrue(reader.thread.is_alive())
                self.assertIsNotNone(reader.stop_event)
                self.assertEqual(reader._cancel_thread.result(), 77)
                calls['_CloseHandle'].assert_not_called()
                reader._post(b'late')
                self.assertEqual(reader.queue.items, [])
            finally:
                release.set()
                calls['_CancelSynchronousIo'].return_value = True
                reader.stop()
            calls['_DuplicateCurrentThread'].assert_called_once_with()
            self.assertEqual(calls['_CloseHandle'].call_args_list, [mock.call(77), mock.call(2)])

    def test_close_failure_keeps_handle_ownership_without_reopening_the_thread(self):
        with self.source() as (reader, calls):
            reader.start()
            reader.thread.join()
            calls['_CloseHandle'].side_effect = [False, True, True]
            with self.assertRaises(OSError):
                reader.stop()
            self.assertIsNone(reader.thread)
            self.assertIsNone(reader.stop_event)
            self.assertEqual(reader._cancel_thread.result(), 77)
            reader.stop()
            self.assertIsNone(reader._cancel_thread)
            calls['_DuplicateCurrentThread'].assert_called_once_with()
            self.assertEqual(calls['_CloseHandle'].call_args_list,
                             [mock.call(77), mock.call(2), mock.call(77)])

    def test_interrupted_start_retains_a_handle_published_after_failed_rollback(self):
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

        with self.source() as (reader, calls):
            calls['_DuplicateCurrentThread'].side_effect = duplicate
            try:
                with mock.patch.object(handle_reader.threading, 'Thread', side_effect=make_thread), \
                        mock.patch.object(handle_reader.time, 'monotonic', side_effect=[0, 3]):
                    with self.assertRaisesRegex(RuntimeError, 'did not stop') as caught:
                        reader.start()
                self.assertIsInstance(caught.exception.__context__, KeyboardInterrupt)
                self.assertFalse(reader._cancel_thread.done())
                self.assertIsNotNone(reader.stop_event)
                calls['_CloseHandle'].assert_not_called()
            finally:
                release.set()
                if reader.thread is not None:
                    reader.thread.join()
                reader.stop()
            calls['_ReadFile'].assert_not_called()
            self.assertEqual(calls['_CloseHandle'].call_args_list, [mock.call(77), mock.call(2)])

    def test_a_signalled_read_is_posted_then_the_stop_event_ends_the_thread(self):
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
            reader.stop()

        self.assertEqual(queue.items, [b"abc"])

    def test_stop_does_not_close_the_event_under_a_live_thread(self):
        waits = iter([handle_reader.WAIT_OBJECT_0])

        def wait_for_objects(count, handles, wait_all, timeout):
            return next(waits)

        with mock.patch.object(handle_reader.sys, 'platform', 'win32'), \
                mock.patch.object(handle_reader, '_GetFileType', return_value=2), \
                mock.patch.object(handle_reader, "_handle",
                                  return_value=1), \
                mock.patch.object(handle_reader, "_CreateEventW",
                                  return_value=2), \
                mock.patch.object(handle_reader,
                                  "_WaitForMultipleObjects",
                                  side_effect=wait_for_objects), \
                mock.patch.object(handle_reader, "_SetEvent"), \
                mock.patch.object(handle_reader, "_DuplicateCurrentThread", return_value=77), \
                mock.patch.object(handle_reader, "_CloseHandle"):
            reader = handle_reader.HandleReader(
                7, _InlineLoop(), _FakeQueue())
            reader.start()
            reader.thread.join()
            reader.stop()

        self.assertIsNone(reader.stop_event)

    def test_stop_retries_cancellation_when_the_read_enters_after_first_cancel(self):
        announced, allow_read, pending, cancelled = [threading.Event() for _ in range(4)]
        calls = []
        acquired_on = []

        def duplicate_thread():
            acquired_on.append(threading.get_native_id())
            return 77

        def read_file(*arguments):
            announced.set()
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
            pending.wait()
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
                announced.wait()
                native_id = reader.thread.native_id
                reader.stop()
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
                reader.stop()


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
                mock.patch.object(handle_reader, '_call', side_effect=bind):
            self.assertEqual(handle_reader._DuplicateCurrentThread(), 77)
        self.assertEqual(seen, [[10, -2, 10, handle_reader.windows_api.ThreadAccess.THREAD_TERMINATE, False, 0]])

    def test_failed_duplication_does_not_publish_an_unusable_handle(self):
        def bind(library, name, *signature):
            return {'GetCurrentThread': lambda: -2, 'DuplicateHandle': lambda *args: False}[name]

        with mock.patch.object(handle_reader.windows_api, 'current_process_handle', return_value=10), \
                mock.patch.object(handle_reader, '_call', side_effect=bind), \
                mock.patch.object(handle_reader.ctypes, 'get_last_error', return_value=5, create=True):
            with self.assertRaises(OSError):
                handle_reader._DuplicateCurrentThread()


if __name__ == "__main__":
    unittest.main()
