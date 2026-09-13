"""Tests for the Windows console backend.

Skipped anywhere but Windows: the module binds kernel32 at import time, so it
cannot even be imported elsewhere.  These have never run.
"""

import ctypes
import sys
import unittest
from ctypes import wintypes
from unittest import mock

if sys.platform == "win32":
    from loki_agent import handle_reader
    from loki_agent import host_terminal_windows


@unittest.skipUnless(sys.platform == "win32", "Windows console APIs")
class RawInputModeTests(unittest.TestCase):
    def test_clears_line_echo_and_processed_and_sets_vt_input(self):
        original = (host_terminal_windows.ENABLE_LINE_INPUT
                    | host_terminal_windows.ENABLE_ECHO_INPUT
                    | host_terminal_windows.ENABLE_PROCESSED_INPUT
                    | 0x0080)

        mode = host_terminal_windows.raw_input_mode(original)

        self.assertEqual(
            mode, 0x0080 | host_terminal_windows.ENABLE_VIRTUAL_TERMINAL_INPUT)

    def test_other_bits_are_left_alone(self):
        mode = host_terminal_windows.raw_input_mode(0x0000)
        self.assertEqual(
            mode, host_terminal_windows.ENABLE_VIRTUAL_TERMINAL_INPUT)


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


@unittest.skipUnless(sys.platform == "win32", "Windows console APIs")
class ReaderThreadTests(unittest.TestCase):
    def test_a_signalled_read_is_posted_then_the_stop_event_ends_the_thread(self):
        queue = _FakeQueue()
        waits = iter([handle_reader.WAIT_OBJECT_0, 1])

        def wait_for_objects(count, handles, wait_all, timeout):
            return next(waits)

        def read_file(handle, buffer, size, read, overlapped):
            ctypes.memmove(buffer, b"abc", 3)
            ctypes.cast(read, ctypes.POINTER(wintypes.DWORD)).contents.value = 3
            return True

        with mock.patch.object(host_terminal_windows, "_handle",
                               return_value=1), \
                mock.patch.object(host_terminal_windows, "_CreateEventW",
                                  return_value=2), \
                mock.patch.object(host_terminal_windows,
                                  "_WaitForMultipleObjects",
                                  side_effect=wait_for_objects), \
                mock.patch.object(host_terminal_windows, "_ReadFile",
                                  side_effect=read_file), \
                mock.patch.object(host_terminal_windows, "_SetEvent"), \
                mock.patch.object(host_terminal_windows, "_CloseHandle"):
            reader = handle_reader.HandleReader(7, _InlineLoop(), queue)
            reader.start()
            reader.thread.join(timeout=5)
            self.assertFalse(reader.thread.is_alive())
            reader.stop()

        self.assertEqual(queue.items, [b"abc"])

    def test_stop_does_not_close_the_event_under_a_live_thread(self):
        waits = iter([1])

        def wait_for_objects(count, handles, wait_all, timeout):
            return next(waits)

        with mock.patch.object(host_terminal_windows, "_handle",
                               return_value=1), \
                mock.patch.object(host_terminal_windows, "_CreateEventW",
                                  return_value=2), \
                mock.patch.object(host_terminal_windows,
                                  "_WaitForMultipleObjects",
                                  side_effect=wait_for_objects), \
                mock.patch.object(host_terminal_windows, "_SetEvent"), \
                mock.patch.object(host_terminal_windows, "_CloseHandle"):
            reader = handle_reader.HandleReader(
                7, _InlineLoop(), _FakeQueue())
            reader.start()
            reader.thread.join(timeout=5)
            reader.stop()

        self.assertIsNone(reader.stop_event)


if __name__ == "__main__":
    unittest.main()
