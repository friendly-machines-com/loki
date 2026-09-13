"""Portable tests for the advisory lock protocol.

The POSIX branch runs here; the Windows branch is exercised with the native
calls mocked, because the values that matter are the ones Windows reports
(``ERROR_LOCK_VIOLATION``) and the translation into the ``BlockingIOError`` the
callers' retry loops expect.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import file_locks
from loki_agent import windows_api


class ExclusiveLockTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "POSIX flock semantics")
    def test_posix_contention_is_blocking_io_error(self):
        with tempfile.NamedTemporaryFile() as handle:
            # Two separate opens, so the two locks are independent; a dup would
            # share one open-file description and never contend.
            first = os.open(handle.name, os.O_RDWR)
            second = os.open(handle.name, os.O_RDWR)
            self.addCleanup(os.close, first)
            self.addCleanup(os.close, second)

            file_locks.try_lock_exclusive(first)
            try:
                with self.assertRaises(BlockingIOError):
                    file_locks.try_lock_exclusive(second)
            finally:
                file_locks.unlock(first)

    def test_windows_contention_is_reported_as_blocking_io_error(self):
        with mock.patch.object(file_locks.sys, "platform", "win32"), \
                mock.patch.object(
                    windows_api, "lock_file",
                    side_effect=windows_api.WindowsApiError(
                        "busy", status=windows_api.ERROR_LOCK_VIOLATION)):
            with self.assertRaises(BlockingIOError):
                file_locks.try_lock_exclusive(0x1234)

    def test_windows_errors_other_than_contention_propagate(self):
        with mock.patch.object(file_locks.sys, "platform", "win32"), \
                mock.patch.object(
                    windows_api, "lock_file",
                    side_effect=windows_api.WindowsApiError(
                        "nope", status=windows_api.ERROR_ACCESS_DENIED)):
            with self.assertRaises(OSError) as caught:
                file_locks.try_lock_exclusive(0x1234)

        self.assertNotIsInstance(caught.exception, BlockingIOError)

    def test_windows_lock_and_unlock_use_the_one_byte_range_at_zero(self):
        with mock.patch.object(file_locks.sys, "platform", "win32"), \
                mock.patch.object(windows_api, "lock_file") as lock, \
                mock.patch.object(windows_api, "unlock_file") as unlock:
            file_locks.try_lock_exclusive(0x1234)
            file_locks.unlock(0x1234)

        lock.assert_called_once_with(
            0x1234,
            windows_api.LOCKFILE_EXCLUSIVE_LOCK
            | windows_api.LOCKFILE_FAIL_IMMEDIATELY)
        unlock.assert_called_once_with(0x1234)


if __name__ == "__main__":
    unittest.main()
