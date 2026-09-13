"""POSIX credential-file primitives: descriptor-relative, symlink-refusing.

Every open refuses a symlink in its final component (``O_NOFOLLOW``) and is
made relative to the already-open directory descriptor, so the object acted on
is the one checked rather than a pathname resolved a second time.
"""

from __future__ import annotations

import os
import stat

from .file_facts import FileFacts


def _facts(result) -> FileFacts:
    return FileFacts(
        regular=stat.S_ISREG(result.st_mode),
        directory=stat.S_ISDIR(result.st_mode),
        # O_NOFOLLOW refused a link at open; a link seen by describe_path is
        # neither a regular file nor a directory, which the store refuses.
        reparse_point=False,
        size=result.st_size,
        owned_by_current_user=result.st_uid == os.geteuid(),
        group_or_other_access=bool(stat.S_IMODE(result.st_mode) & 0o077),
    )


def describe(fd) -> FileFacts:
    return _facts(os.fstat(fd))


def describe_path(path) -> FileFacts:
    return _facts(os.lstat(path))


def _no_follow(flags: int) -> int:
    flags |= getattr(os, "O_CLOEXEC", 0)
    return flags | getattr(os, "O_NOFOLLOW", 0)


def open_directory(path: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    return os.open(path, _no_follow(flags))


def open_read_at(directory_fd: int, name: str) -> int:
    return os.open(name, _no_follow(os.O_RDONLY), dir_fd=directory_fd)


def create_exclusive_at(directory_fd: int, name: str, mode: int) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    return os.open(name, _no_follow(flags), mode, dir_fd=directory_fd)


def open_lock_file_at(directory_fd: int, name: str, mode: int) -> int:
    flags = os.O_RDWR | os.O_CREAT
    return os.open(name, _no_follow(flags), mode, dir_fd=directory_fd)


def replace_at(directory_fd: int, temporary: str, name: str) -> None:
    os.replace(temporary, name,
               src_dir_fd=directory_fd, dst_dir_fd=directory_fd)


def unlink_at(directory_fd: int, name: str) -> None:
    os.unlink(name, dir_fd=directory_fd)


# -- descriptor operations ------------------------------------------------
# The token on this platform is an integer descriptor, so these are ``os.*``.
# They are functions rather than aliases so a test can still intercept the
# call the storage makes (``credential_storages`` swaps one descriptor for an
# invalid one to prove it was closed).

def read(fd, size):
    return os.read(fd, size)


def write(fd, data):
    return os.write(fd, data)


def fsync(fd) -> None:
    os.fsync(fd)


def close(fd) -> None:
    os.close(fd)
