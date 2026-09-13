"""Advisory locks for a whole read-modify-write operation on one file.

``fcntl.flock`` has no Windows equivalent, and ``msvcrt.locking`` reports
contention differently, so callers go through here instead of repeating a
platform branch.  The protocol is an exclusive lock on the one-byte range at
offset zero of an already-open lock file; the caller opens the file, keeps it
open for the operation, and closing it releases the lock.
"""

from __future__ import annotations

import errno
import os
import sys

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def try_lock_exclusive(fd) -> None:
    """Take the exclusive lock, raising BlockingIOError while it is held."""
    if sys.platform == "win32":
        # msvcrt locks from the current position, so the one-byte range at
        # offset zero has to be selected explicitly.
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            # Windows reports contention as EACCES, not BlockingIOError.
            if error.errno == errno.EACCES:
                raise BlockingIOError from error
            raise
    else:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock(fd) -> None:
    """Release the lock taken by :func:`try_lock_exclusive`."""
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
