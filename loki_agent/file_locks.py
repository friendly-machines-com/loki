"""Advisory locks for a whole read-modify-write operation on one file.

``fcntl.flock`` has no Windows equivalent and the two primitives report
contention differently, so callers go through here instead of repeating a
platform branch.  The protocol is an exclusive lock on the one-byte range at
offset zero of an already-open lock file; the caller opens the file, keeps it
open for the operation, and closing it releases the lock.

The token is the platform's: a POSIX descriptor for ``flock``, or a Windows
``HANDLE`` for ``LockFileEx``.  A handle is not a CRT descriptor, so
``msvcrt.locking`` cannot serve it; the Windows branch locks the handle.
"""

from __future__ import annotations

import sys

from . import windows_api

if sys.platform != "win32":
    import fcntl


def try_lock_exclusive(token) -> None:
    """Take the exclusive lock, raising BlockingIOError while it is held."""
    if sys.platform == "win32":
        try:
            windows_api.lock_file(
                token,
                windows_api.LockFlags.LOCKFILE_EXCLUSIVE_LOCK
                | windows_api.LockFlags.LOCKFILE_FAIL_IMMEDIATELY)
        except windows_api.WindowsApiError as error:
            # LockFileEx reports contention as ERROR_LOCK_VIOLATION, not as
            # EACCES and not as a blocking call.
            if error.status == windows_api.ERROR_LOCK_VIOLATION:
                raise BlockingIOError from error
            raise OSError(str(error)) from error
    else:
        fcntl.flock(token, fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock(token) -> None:
    """Release the lock taken by :func:`try_lock_exclusive`."""
    if sys.platform == "win32":
        try:
            windows_api.unlock_file(token)
        except windows_api.WindowsApiError as error:
            raise OSError(str(error)) from error
    else:
        fcntl.flock(token, fcntl.LOCK_UN)
