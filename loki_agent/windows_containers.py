"""OS mutation for the Windows container: profile creation and DACLs.

It lives apart from the shared API so the runtime's own code has no built-in
way to create a profile or rewrite a DACL.  ``windows_setup`` explains what
that buys and what it does not.

Nothing here decides *what* to grant -- that is the editor's model, the path
rules and the user's decision.  These are the calls that carry the decision
out.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from .windows_api import (
    ERROR_SUCCESS,
    SDDL_REVISION_1,
    SE_FILE_OBJECT,
    SecurityInformation,
    WindowsApiError,
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
    convert = bind(
        "advapi32", "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        wintypes.BOOL, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    get_dacl = bind("advapi32", "GetSecurityDescriptorDacl", wintypes.BOOL,
                    ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(wintypes.BOOL))
    set_named = bind("advapi32", "SetNamedSecurityInfoW", wintypes.DWORD,
                     wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                     ctypes.c_void_p)
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)

    descriptor, dacl = ctypes.c_void_p(), ctypes.c_void_p()
    if not convert(sddl, SDDL_REVISION_1, ctypes.byref(descriptor), None):
        raise WindowsApiError(f"invalid security descriptor: {sddl!r}")
    try:
        present = wintypes.BOOL()
        defaulted = wintypes.BOOL()
        if not get_dacl(descriptor, ctypes.byref(present),
                        ctypes.byref(dacl), ctypes.byref(defaulted)):
            raise WindowsApiError("GetSecurityDescriptorDacl failed")
        if not present.value or not dacl.value:
            raise WindowsApiError("security descriptor carries no explicit DACL")
        flags = SecurityInformation.DACL_SECURITY_INFORMATION
        if 'P' in sddl.split('(', 1)[0][2:]:
            flags |= SecurityInformation.PROTECTED_DACL_SECURITY_INFORMATION
        status = set_named(
            path, SE_FILE_OBJECT, flags, None, None, dacl, None)
        if status != ERROR_SUCCESS:
            raise WindowsApiError(
                f"SetNamedSecurityInfoW({path!r}) failed: {status}")
    finally:
        local_free(descriptor)


def set_handle_dacl_sddl(handle, sddl: str) -> None:
    """Set the DACL of the object ``handle`` names, as ``set_dacl_sddl`` would.

    The handle-relative counterpart of ``set_dacl_sddl``: the same descriptor
    conversion and the same protected-flag rule, applied to the object the
    handle names.  Keeping both here means a grant that goes through a handle
    cannot drift from one that goes through a pathname.
    """
    convert = bind(
        "advapi32", "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        wintypes.BOOL, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    get_dacl = bind("advapi32", "GetSecurityDescriptorDacl", wintypes.BOOL,
                    ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(wintypes.BOOL))
    set_info = bind("advapi32", "SetSecurityInfo", wintypes.DWORD,
                    ctypes.c_void_p, ctypes.c_int, wintypes.DWORD,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p)
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)

    descriptor, dacl = ctypes.c_void_p(), ctypes.c_void_p()
    if not convert(sddl, SDDL_REVISION_1, ctypes.byref(descriptor), None):
        raise WindowsApiError(f"invalid security descriptor: {sddl!r}")
    try:
        present = wintypes.BOOL()
        defaulted = wintypes.BOOL()
        if not get_dacl(descriptor, ctypes.byref(present),
                        ctypes.byref(dacl), ctypes.byref(defaulted)):
            raise WindowsApiError("GetSecurityDescriptorDacl failed")
        if not present.value or not dacl.value:
            raise WindowsApiError("security descriptor carries no explicit DACL")
        flags = SecurityInformation.DACL_SECURITY_INFORMATION
        if 'P' in sddl.split('(', 1)[0][2:]:
            flags |= SecurityInformation.PROTECTED_DACL_SECURITY_INFORMATION
        status = set_info(handle, SE_FILE_OBJECT, flags, None, None, dacl, None)
        if status != ERROR_SUCCESS:
            raise WindowsApiError(f"SetSecurityInfo(dacl) failed: {status}")
    finally:
        local_free(descriptor)


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
        free_sid(sid)


def delete_app_container_profile(name: str) -> None:
    delete = bind("userenv", "DeleteAppContainerProfile", ctypes.c_long,
                  wintypes.LPCWSTR)
    status = delete(name)
    if status < 0:
        raise WindowsApiError(
            f"DeleteAppContainerProfile({name!r}) failed: "
            f"0x{status & 0xffffffff:08x}")
