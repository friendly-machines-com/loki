"""Tests for the Windows console backend.

Console mode checks require Windows. The import-safe handle reader's native
calls are replaced in portable tests so shutdown races are exercised on Linux
as well as by the real Windows PTY tests.
"""

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
            mock.patch.object(handle_reader, '_OpenThread', return_value=77) as open_thread,
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
                open_thread.assert_called_once_with(
                    handle_reader.windows_api.ThreadAccess.THREAD_TERMINATE, False, native_id)
                self.assertEqual(close.call_args_list, [mock.call(77), mock.call(2)])
            finally:
                allow_read.set()
                cancelled.set()
                reader.stop()


if __name__ == "__main__":
    unittest.main()
