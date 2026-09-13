"""Native protections for processes which can obtain request credentials."""

import ctypes
import os
import sys


class ProcessProtectionError(RuntimeError):
    pass


_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4


def _protect_windows_process() -> bool:
    """Restrict the process object to its owner, SYSTEM and Administrators.

    Windows has no ``PR_SET_DUMPABLE``.  Another process reaches this one's
    memory through the process object, so the protection is that object's
    DACL: this grants full control to the current user, SYSTEM and the
    Administrators and removes everything else, including the inherited
    ``Everyone`` and ``Users`` grants.  The AppContainer probes measured that a
    contained worker is then refused ``PROCESS_VM_READ``, which is the property
    this establishes.  A same-user owner still holds ``WRITE_DAC`` and can
    re-grant access, so this is "other principals are refused", not a
    non-dumpable guarantee.
    """
    from . import windows_api

    owner = windows_api.current_user_sid()
    sddl = f"D:P(A;;FA;;;{owner})(A;;FA;;;SY)(A;;FA;;;BA)"
    windows_api.set_handle_dacl(
        windows_api.current_process_handle(), sddl,
        object_type=windows_api.SE_KERNEL_OBJECT)
    return True


def protect_credential_process() -> bool:
    """Block same-UID ptrace and /proc descriptor inspection on Linux.

    ``close_fds`` prevents descriptor inheritance, but it is not by itself a
    hostile-child boundary: on permissive Linux configurations a tool process
    may otherwise inspect its Loki parent through ptrace-governed ``/proc``
    entries.  Non-dumpability also intentionally disables credential-bearing
    core dumps.

    Linux is Loki's declared supported platform, so failure is fatal there.
    Other systems currently get descriptor isolation without a claimed
    same-UID introspection guarantee until they have a tested native backend.
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
