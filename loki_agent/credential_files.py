"""Credential-file primitives, selected once per platform.

``JsonCredentialStorage`` is one algorithm over a directory and a few files.
The parts that differ per platform are opening those files without following a
symlink, reading/writing/flushing/closing the open object, creating/replacing/
removing them relative to the directory, and describing an open object (is it a
regular file, who owns it, is it private); those are chosen here once, so the
storage code has no platform branch and there is no second copy of the
algorithm.

Read, write, fsync and close live in the platform module rather than here
because the token they operate on is the platform's: a POSIX descriptor is an
``int`` that ``os.*`` understands, a Windows handle is not, so a single
``os.read`` for both would be wrong on one of them.

The description is a :class:`FileFacts` record rather than a ``stat_result``:
a POSIX descriptor has mode bits and a uid, while a Windows handle has
attributes, an owner SID and a DACL, so the store must ask for facts, not for
the platform's shape of them.  ``describe`` takes an open object; ``describe_path``
takes a path, for the directory before it is opened.

``CredentialStorageError`` is defined here rather than in the storage module so
a platform implementation can raise the same error the storage raises; the
storage re-exports it, so importers are unaffected.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class FileFacts:
    """What the store needs to know about an object, per platform."""

    regular: bool
    directory: bool
    reparse_point: bool
    size: int
    owned_by_current_user: bool
    group_or_other_access: bool


class CredentialStorageError(RuntimeError):
    pass


if sys.platform == "win32":
    from ._credential_files_windows import (
        close,
        create_exclusive_at,
        describe,
        describe_path,
        fsync,
        open_directory,
        open_lock_file_at,
        open_read_at,
        read,
        replace_at,
        unlink_at,
        write,
    )
else:
    from ._credential_files_posix import (
        close,
        create_exclusive_at,
        describe,
        describe_path,
        fsync,
        open_directory,
        open_lock_file_at,
        open_read_at,
        read,
        replace_at,
        unlink_at,
        write,
    )


__all__ = [
    "CredentialStorageError",
    "FileFacts",
    "close",
    "create_exclusive_at",
    "describe",
    "describe_path",
    "fsync",
    "open_directory",
    "open_lock_file_at",
    "open_read_at",
    "read",
    "replace_at",
    "unlink_at",
    "write",
]
