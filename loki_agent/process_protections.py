"""Native protections for processes which can obtain request credentials."""

import ctypes
import os
import sys


class ProcessProtectionError(RuntimeError):
    pass


_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4


def private_object_dacl(owner: str) -> str:
    """Windows credential-owner kernel objects, not workspace/file grants."""
    return f'D:P(A;;GA;;;{owner})(A;;GA;;;SY)(A;;GA;;;BA)'


def install_private_default(token, owner: str) -> None:
    """Make future objects private; the token object's own DACL is separate."""
    from . import windows_api

    windows_api.set_token_default_dacl(token, private_object_dacl(owner))


def _protect_windows_process() -> bool:
    """Protect current objects and establish private future-thread creation.

    Owned children receive initial-object security and this default from their
    creator before first resume. A directly invoked public process can tighten
    its current objects here, but cannot revoke handles obtained before startup;
    this function does not pretend otherwise or scan/repair threads periodically.
    """
    from . import windows_api

    owner = windows_api.current_user_sid()
    sddl = private_object_dacl(owner)
    try:
        windows_api.set_handle_dacl(
            windows_api.current_process_handle(), sddl,
            object_type=windows_api.SE_KERNEL_OBJECT)
        token = windows_api.open_process_token(
            windows_api.current_process_handle(),
            windows_api.TokenAccess.TOKEN_QUERY | windows_api.TokenAccess.TOKEN_ADJUST_DEFAULT)
        try:
            install_private_default(token, owner)
        finally:
            windows_api.close_handle(token)
        current_thread = windows_api.bind('kernel32', 'GetCurrentThread', ctypes.c_void_p)
        windows_api.set_handle_dacl(
            current_thread(), sddl, object_type=windows_api.SE_KERNEL_OBJECT)
    except windows_api.WindowsApiError as error:
        raise ProcessProtectionError(f'could not protect credentials: {error}') from error
    return True


def protect_credential_process() -> bool:
    """Apply Windows current/future-object or Linux non-dumpability protections.

    ``close_fds`` prevents descriptor inheritance, but it is not by itself a
    hostile-child boundary: on permissive Linux configurations a tool process
    may otherwise inspect its Loki parent through ptrace-governed ``/proc``
    entries.  Non-dumpability also intentionally disables credential-bearing
    core dumps.

    Linux requires verified non-dumpability; failure is fatal there.
    Windows protects current process/thread objects and future creation as above.
    Other POSIX systems return False without claiming either native protection.
    """
    if sys.platform == "win32":
        return _protect_windows_process()
    if not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        prctl = libc.prctl
    except AttributeError as error:
        raise ProcessProtectionError(
            "libc does not provide prctl()") from error
    prctl.restype = ctypes.c_int
    result = prctl(
        ctypes.c_int(_PR_SET_DUMPABLE),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise ProcessProtectionError(
            f"PR_SET_DUMPABLE failed: {os.strerror(error_number)}")
    state = prctl(
        ctypes.c_int(_PR_GET_DUMPABLE),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if state != 0:
        raise ProcessProtectionError(
            f"PR_GET_DUMPABLE returned unexpected state {state}")
    return True
