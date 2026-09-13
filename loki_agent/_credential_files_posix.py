"""POSIX credential-file primitives: descriptor-relative, symlink-refusing.

Every open refuses a symlink in its final component (``O_NOFOLLOW``) and is
made relative to the already-open directory descriptor, so the object acted on
is the one checked rather than a pathname resolved a second time.
"""

from __future__ import annotations

import os


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


def owner_is_current_user(result) -> bool:
    return result.st_uid == os.geteuid()
