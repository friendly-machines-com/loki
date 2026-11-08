"""Native file primitives for retained directories, selected per platform.

Credential, settings/header, job and editor publishers share native acquisition
and relative publication mechanisms, not permission or durability policies.
Those remain with each owner rather than a second copy of the platform calls.

Read, write, fsync and close live in the platform module rather than here
because the token they operate on is the platform's: a POSIX descriptor is an
``int`` that ``os.*`` understands, a Windows handle is not, so a single
``os.read`` for both would be wrong on one of them.

``binary_file`` is the explicit consuming boundary for Python file consumers:
job stdout/stderr need subprocess-compatible descriptors, while editor and
ledger writers use standard buffered I/O. It wraps the already-acquired object
rather than reopening a pathname, and transfers close ownership even on
construction failure. Native-token owners retain the raw operations when the
same token still serves their validation, synchronization and cleanup protocol.

``FileFacts`` and ``CredentialStorageError`` live in their own leaf modules,
below this module and the platform modules, so a platform module can import
them without importing the module that imports it.  They are re-exported here
so importers are unaffected.

``describe`` takes an open object; ``describe_path`` takes a path, for the
directory before it is opened.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import sys

from .credential_errors import CredentialStorageError
from .file_facts import FileFacts, FileInfo


if sys.platform == "win32":
    from ._private_files_windows import (
        binary_file,
        close,
        create_directory_at,
        create_exclusive_at,
        create_private_at,
        describe,
        describe_path,
        fsync,
        information,
        make_private_directory,
        open_append_at,
        open_directory,
        open_directory_at,
        open_lock_file_at,
        open_read_at,
        read,
        replace_at,
        seek,
        unlink_at,
        write,
    )
else:
    from ._private_files_posix import (
        binary_file,
        close,
        create_directory_at,
        create_exclusive_at,
        create_private_at,
        describe,
        describe_path,
        fsync,
        information,
        make_private_directory,
        open_append_at,
        open_directory,
        open_directory_at,
        open_lock_file_at,
        open_read_at,
        read,
        replace_at,
        seek,
        unlink_at,
        write,
    )


def write_all(handle, data):
    """Complete delivery on a borrowed native token, without taking ownership.

    Credential/state publication and job diagnostics still use that token for
    their native validation, synchronization or close protocol. binary_file
    consumes ownership instead, so it is not a drop-in write on these borrowed
    tokens. Permission, durability and cleanup policy remain with the caller.
    """
    view = memoryview(data)
    while view:
        written = write(handle, view)
        if written <= 0:
            raise OSError('could not write complete file contents')
        view = view[written:]


def publish_synced_bytes_at(directory, name, data):
    """Publish inherited state bytes; the caller retains its directory and lock.

    POSIX creation uses mode0600; Windows keeps the directory's inherited DACL.
    This is not editor permission preservation or private credential validation.
    Directory synchronization has the existing platform semantics.
    """
    temporary = None
    try:
        candidate = f".{name}.{os.getpid()}.{secrets.token_hex(12)}"
        handle = create_exclusive_at(directory, candidate, 0o600)
        # Failed exclusive creation never acquires another publication's entry.
        temporary = candidate
        try:
            write_all(handle, data)
            fsync(handle)
        finally:
            close(handle)
        replace_at(directory, temporary, name)
        temporary = None
        fsync(directory)
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                unlink_at(directory, temporary)


def open_directory_tree(path, mode=0o700):
    """Acquire a state directory, creating missing children from native opens.

    Do not stat each ancestor with os.makedirs: a contained Windows process
    may traverse an existing path without permission to inspect the drive root.
    Only the initial acquisition uses names; callers retain the returned handle.
    """
    handle = None
    try:
        try:
            handle = open_directory(path)
        except FileNotFoundError:
            parent, name = os.path.split(path)
            if not name or name in ['.', '..']:
                raise
            directory = open_directory_tree(parent or '.', mode)
            try:
                create_directory_at(directory, name, mode)
                handle = open_directory_at(directory, name)
            finally:
                close(directory)
        info = information(handle)
        if not info.directory or info.reparse_point:
            raise NotADirectoryError("state path is not a plain directory")
        return handle
    except BaseException:
        if handle is not None:
            close(handle)
        raise


__all__ = [
    "CredentialStorageError",
    "FileFacts",
    "FileInfo",
    "binary_file",
    "close",
    "create_directory_at",
    "create_exclusive_at",
    "create_private_at",
    "describe",
    "describe_path",
    "fsync",
    "information",
    "make_private_directory",
    "open_append_at",
    "open_directory",
    "open_directory_at",
    "open_directory_tree",
    "open_lock_file_at",
    "open_read_at",
    "publish_synced_bytes_at",
    "read",
    "replace_at",
    "seek",
    "unlink_at",
    "write",
    "write_all",
]
