"""Portable tests for the advisory lock protocol.

The POSIX branch runs here; the Windows branch is exercised with ``msvcrt``
mocked, because the values that matter are the ones Windows reports and the
translation into the ``BlockingIOError`` the callers' retry loops expect.
"""

import errno
import os
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import file_locks


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

    def test_windows_contention_is_reported_as_eacces(self):
        with tempfile.TemporaryFile() as handle:
            with mock.patch.object(file_locks, "msvcrt",
                                   create=True) as msvcrt, \
                    mock.patch.object(file_locks.sys, "platform", "win32"):
                msvcrt.LK_NBLCK = 2
                msvcrt.locking.side_effect = OSError(errno.EACCES, "busy")

                with self.assertRaises(BlockingIOError):
                    file_locks.try_lock_exclusive(handle.fileno())

    def test_windows_errors_other_than_eacces_propagate(self):
        with tempfile.TemporaryFile() as handle:
            with mock.patch.object(file_locks, "msvcrt",
                                   create=True) as msvcrt, \
                    mock.patch.object(file_locks.sys, "platform", "win32"):
                msvcrt.LK_NBLCK = 2
                msvcrt.locking.side_effect = OSError(errno.EINVAL, "nope")

                with self.assertRaises(OSError) as caught:
                    file_locks.try_lock_exclusive(handle.fileno())

        self.assertEqual(caught.exception.errno, errno.EINVAL)

    def test_windows_lock_and_unlock_use_the_one_byte_range_at_zero(self):
        with tempfile.TemporaryFile() as handle:
            fd = handle.fileno()
            with mock.patch.object(file_locks, "msvcrt",
                                   create=True) as msvcrt, \
                    mock.patch.object(file_locks.sys, "platform", "win32"):
                msvcrt.LK_NBLCK = 2
                msvcrt.LK_UNLCK = 0

                file_locks.try_lock_exclusive(fd)
                file_locks.unlock(fd)

        self.assertEqual(msvcrt.locking.call_args_list, [
            mock.call(fd, 2, 1), mock.call(fd, 0, 1)])

    def test_windows_lock_seeks_to_the_range_start(self):
        with tempfile.TemporaryFile() as handle:
            fd = handle.fileno()
            with mock.patch.object(file_locks, "msvcrt",
                                   create=True), \
                    mock.patch.object(file_locks.sys, "platform", "win32"), \
                    mock.patch.object(file_locks.os, "lseek") as seek:
                file_locks.try_lock_exclusive(fd)

        seek.assert_called_once_with(fd, 0, os.SEEK_SET)


if __name__ == "__main__":
    unittest.main()
