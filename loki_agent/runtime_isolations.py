"""Early, dependency-free isolation for credential-consuming runtimes.

Loki has two kinds of processes:

* credential supervisors capture startup secrets and own the root credential
  broker; and
* runtimes execute model sessions and tools using a restricted credential
  capability.

Supervisors keep their normal filesystem view so they can eventually load and
atomically update persistent credentials.  Every runtime enters a private
Linux user/mount namespace before importing the agent core, then covers Loki's
dedicated credential directory with an empty, read-only tmpfs.  All tools and
nested subagents inherit that view, while the supervisor remains able to
persist refreshed credentials.

The namespace is deliberately established in the newly execed runtime, while
it is still single-threaded.  Doing this in ``preexec_fn`` from an async
supervisor would run Python after ``fork()`` in a potentially threaded process
and can deadlock on inherited interpreter or libc locks.

Python itself, including any interpreter startup hooks, necessarily runs
before this Python entrypoint can call ``unshare()``.  The interpreter
installation and its startup module paths are therefore part of Loki's
trusted launcher boundary and must not be writable by model-run tools.  Moving
the boundary earlier would require a native launcher, contrary to this
project's dependency-free Python design.

This is pathname isolation, not a general-purpose sandbox.  Process
non-dumpability, descriptor closure, owner lifetimes, and credential
capabilities remain separate required layers.
"""

from __future__ import annotations

import ctypes
import errno
import os
import sys

from .paths import credential_directory


class RuntimeIsolationError(RuntimeError):
    pass


_CLONE_NEWNS = 0x00020000
_CLONE_NEWUSER = 0x10000000

_MS_RDONLY = 1
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_NOEXEC = 8
_MS_REC = 16384
_MS_PRIVATE = 1 << 18

_PR_CAPBSET_DROP = 24
_PR_SET_NO_NEW_PRIVS = 38
_LINUX_CAPABILITY_VERSION_3 = 0x20080522


class _CapabilityHeader(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("pid", ctypes.c_int),
    ]


class _CapabilityData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def _raise_errno(operation: str) -> None:
    error_number = ctypes.get_errno()
    raise RuntimeIsolationError(
        f"{operation} failed: {os.strerror(error_number)}")


def _unshare_user_and_mount_namespaces(libc) -> None:
    flags = (
        getattr(os, "CLONE_NEWUSER", _CLONE_NEWUSER)
        | getattr(os, "CLONE_NEWNS", _CLONE_NEWNS)
    )
    unshare = getattr(os, "unshare", None)
    if unshare is not None:
        # os.unshare() was added in Python 3.12. Prefer the standard-library
        # wrapper whenever it exists; Loki's Python 3.10 minimum is the sole
        # reason for retaining the terrible direct-libc fallback below.
        try:
            unshare(flags)
        except OSError as error:
            raise RuntimeIsolationError(
                f"unshare(CLONE_NEWUSER | CLONE_NEWNS) failed: {error}"
            ) from error
        return

    try:
        libc_unshare = libc.unshare
    except AttributeError as error:
        raise RuntimeIsolationError(
            "Python has no os.unshare() and libc has no unshare()"
        ) from error
    libc_unshare.argtypes = (ctypes.c_int,)
    libc_unshare.restype = ctypes.c_int
    if libc_unshare(flags) != 0:
        _raise_errno("unshare(CLONE_NEWUSER | CLONE_NEWNS)")


def _write_namespace_map(path: str, value: str) -> None:
    try:
        with open(path, "w", encoding="ascii") as stream:
            stream.write(value)
    except OSError as error:
        raise RuntimeIsolationError(
            f"could not configure {path}: {error}") from error


def _map_current_identity(uid: int, gid: int) -> None:
    """Map only Loki's current identity into the new user namespace."""
    try:
        _write_namespace_map("/proc/self/setgroups", "deny")
    except RuntimeIsolationError as error:
        if not isinstance(error.__cause__, FileNotFoundError):
            raise
    _write_namespace_map("/proc/self/uid_map", f"{uid} {uid} 1")
    _write_namespace_map("/proc/self/gid_map", f"{gid} {gid} 1")


def _mount_private_credential_view(libc, directory: str) -> None:
    try:
        mount = libc.mount
    except AttributeError as error:
        raise RuntimeIsolationError("libc has no mount()") from error
    mount.argtypes = (
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    )
    mount.restype = ctypes.c_int

    # Prevent either the credential cover mount or later runtime mounts from
    # propagating back into the supervisor's filesystem view.
    if mount(
            None, b"/", None,
            _MS_REC | _MS_PRIVATE, None) != 0:
        _raise_errno("mount(MS_REC | MS_PRIVATE)")

    encoded = os.fsencode(directory)
    flags = _MS_RDONLY | _MS_NOSUID | _MS_NODEV | _MS_NOEXEC
    options = b"size=4096,nr_inodes=1,mode=000"
    if mount(b"tmpfs", encoded, b"tmpfs", flags, options) != 0:
        _raise_errno("credential-directory tmpfs mount")


def _last_capability() -> int:
    try:
        with open(
                "/proc/sys/kernel/cap_last_cap",
                "r", encoding="ascii") as stream:
            value = int(stream.read().strip())
    except (OSError, ValueError):
        # Linux capabilities currently fit well below this conservative
        # bound. Unknown higher values return EINVAL and terminate the loop.
        return 63
    return max(0, min(value, 1024))


def _drop_namespace_capabilities(libc) -> None:
    try:
        prctl = libc.prctl
        capset = libc.capset
    except AttributeError as error:
        raise RuntimeIsolationError(
            "libc lacks capability-dropping interfaces") from error
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    capset.argtypes = (
        ctypes.POINTER(_CapabilityHeader),
        ctypes.POINTER(_CapabilityData),
    )
    capset.restype = ctypes.c_int

    # Drop the bounding set while CAP_SETPCAP is still effective, then clear
    # every effective/permitted/inheritable bit. NO_NEW_PRIVS prevents a later
    # exec from recovering privilege through set-ID or file capabilities.
    for capability in range(_last_capability() + 1):
        result = prctl(
            ctypes.c_int(_PR_CAPBSET_DROP),
            ctypes.c_ulong(capability),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EINVAL:
                # Capability numbers are contiguous. EINVAL therefore means
                # this and every later number is unknown to the running kernel.
                break
            raise RuntimeIsolationError(
                "PR_CAPBSET_DROP failed for capability "
                f"{capability}: {os.strerror(error_number)}")

    header = _CapabilityHeader(
        version=_LINUX_CAPABILITY_VERSION_3, pid=0)
    data = (_CapabilityData * 2)()
    if capset(ctypes.byref(header), data) != 0:
        _raise_errno("capset")
    if prctl(
            ctypes.c_int(_PR_SET_NO_NEW_PRIVS),
            ctypes.c_ulong(1),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
    ) != 0:
        _raise_errno("PR_SET_NO_NEW_PRIVS")


def isolate_credential_directory(directory: str | None = None) -> bool:
    """Hide the credential directory in this runtime and its descendants.

    The supervisor creates the dedicated directory before spawning a runtime
    once persistent credentials are enabled.  Until then, an absent directory
    means there is no persistent credential pathname to hide.
    """
    if not sys.platform.startswith("linux"):
        return False
    target = credential_directory() if directory is None else directory
    if not os.path.isdir(target):
        return False

    libc = ctypes.CDLL(None, use_errno=True)
    # Once the new user namespace exists, an unmapped caller is reported as
    # the overflow identity. Preserve the real IDs before unshare so the map
    # describes the supervisor-visible user rather than that placeholder.
    uid = os.getuid()
    gid = os.getgid()
    _unshare_user_and_mount_namespaces(libc)
    _map_current_identity(uid, gid)
    _mount_private_credential_view(libc, target)
    _drop_namespace_capabilities(libc)
    return True
