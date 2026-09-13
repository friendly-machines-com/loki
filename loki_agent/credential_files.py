"""Credential-file primitives, selected once per platform.

``JsonCredentialStorage`` is one algorithm over a directory and a few files.
The parts that differ per platform are opening those files without following a
symlink, creating/replacing/removing them relative to the directory, and
checking that the owner is the current user; those are chosen here once, so the
storage code has no platform branch and there is no second copy of the
algorithm.  Classification and the byte-level calls are portable and live
here; each implementation supplies only the calls that are not.

``CredentialStorageError`` is defined here rather than in the storage module so
a platform implementation can raise the same error the storage raises; the
storage re-exports it, so importers are unaffected.
"""

from __future__ import annotations

import os
import stat
import sys


__all__ = [
    "CredentialStorageError",
    "close",
    "create_exclusive_at",
    "fstat",
    "grants_group_or_other",
    "is_directory",
    "is_regular",
    "open_directory",
    "open_lock_file_at",
    "open_read_at",
    "owner_is_current_user",
    "read",
    "replace_at",
    "unlink_at",
    "write",
]


class CredentialStorageError(RuntimeError):
    pass


def is_directory(result) -> bool:
    return stat.S_ISDIR(result.st_mode)


def is_regular(result) -> bool:
    return stat.S_ISREG(result.st_mode)


def grants_group_or_other(result) -> bool:
    return bool(stat.S_IMODE(result.st_mode) & 0o077)


def read(fd, size):
    return os.read(fd, size)


def write(fd, data):
    return os.write(fd, data)


def fsync(fd) -> None:
    os.fsync(fd)


def close(fd) -> None:
    os.close(fd)


def fstat(fd):
    return os.fstat(fd)


if sys.platform == "win32":
    from ._credential_files_windows import (
        create_exclusive_at,
        open_directory,
        open_lock_file_at,
        open_read_at,
        owner_is_current_user,
        replace_at,
        unlink_at,
    )
else:
    from ._credential_files_posix import (
        create_exclusive_at,
        open_directory,
        open_lock_file_at,
        open_read_at,
        owner_is_current_user,
        replace_at,
        unlink_at,
    )
