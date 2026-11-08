"""Setup-specific Windows profile lifecycle and pathname DACL mutation.

``windows_setup`` owns grant decisions. Opened-handle DACL updates use the
shared ``windows_api.set_handle_dacl`` primitive, also needed by editor
publication and process protection. The pathname setter stays here because
setup's protected-DACL propagation differs from the shared named setter's
DACL-only operation. Neither module placement nor these primitives grant
permission: the process token and the object's DACL enforce that boundary.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from .windows_api import (
    ERROR_SUCCESS,
    _converted_dacl,
    SE_FILE_OBJECT,
    SecurityInformation,
    WindowsApiError,
    WindowsResourceError,
    bind,
    sid_text,
)


# 0x800700B7: ERROR_ALREADY_EXISTS.  Measured on all three investigation
# interpreters when CreateAppContainerProfile is called for an existing name.
PROFILE_ALREADY_EXISTS = 0x800700B7


def set_dacl_sddl(path: str, sddl: str) -> None:
    """Set a DACL without unconditionally disabling inheritance.

    ``D:P`` explicitly protects Loki's private trees. For an unprotected
    descriptor, leave the object's inheritance setting unchanged.
    """
    set_named = bind("advapi32", "SetNamedSecurityInfoW", wintypes.DWORD,
                     wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                     ctypes.c_void_p)
    with _converted_dacl(sddl) as dacl:
        flags = SecurityInformation.DACL_SECURITY_INFORMATION
        if 'P' in sddl.split('(', 1)[0][2:]:
            flags |= SecurityInformation.PROTECTED_DACL_SECURITY_INFORMATION
        status = set_named(
            path, SE_FILE_OBJECT, flags, None, None, dacl, None)
        if status != ERROR_SUCCESS:
            raise WindowsApiError(
                f"SetNamedSecurityInfoW({path!r}) failed: {status}", status=status)


def create_app_container_profile(name: str) -> str:
    """Create the profile and return its package SID."""
    create = bind("userenv", "CreateAppContainerProfile", ctypes.c_long,
                  wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
                  ctypes.c_void_p, wintypes.DWORD,
                  ctypes.POINTER(ctypes.c_void_p))
    free_sid = bind("advapi32", "FreeSid", ctypes.c_void_p, ctypes.c_void_p)

    sid = ctypes.c_void_p()
    status = create(name, name, name, None, 0, ctypes.byref(sid))
    if status < 0:
        raise WindowsApiError(
            f"CreateAppContainerProfile({name!r}) failed: "
            f"0x{status & 0xffffffff:08x}",
            status=status & 0xffffffff)
    try:
        return sid_text(sid)
    finally:
        if free_sid(sid):
            raise WindowsResourceError('FreeSid failed')


def delete_app_container_profile(name: str) -> None:
    delete = bind("userenv", "DeleteAppContainerProfile", ctypes.c_long,
                  wintypes.LPCWSTR)
    status = delete(name)
    if status < 0:
        raise WindowsApiError(
            f"DeleteAppContainerProfile({name!r}) failed: "
            f"0x{status & 0xffffffff:08x}")
