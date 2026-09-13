"""Windows API declarations shared by the rest of Loki.

Import-safe on every platform: nothing here binds a DLL or touches a
Windows-only process, and the *declarations* (GUID layout, flag values,
structure layout) are exercised by portable tests on Linux.  That matters
because these values are invisible on the machine that runs them: a GUID whose
first three groups were byte-swapped, or a structure missing a member, produces
a call that fails somewhere else entirely.

Two rules keep this module safe to share rather than copied around:

* **One binding per (library, symbol).**  ``ctypes`` caches the function object
  per symbol, so a second declaration with a different signature would silently
  replace the first and every caller would get the wrong marshalling.  ``bind``
  refuses that instead -- the project has already lost a debugging round to a
  symbol that could not carry two signatures.
* **Declarations only.**  Nothing here decides what a caller should protect,
  grant or refuse; those are security decisions that belong to the caller and
  to the review the design document requires.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes


class WindowsApiError(RuntimeError):
    """A Windows call was attempted and returned failure.

    ``status`` carries the HRESULT when the failure came from one, so callers
    can distinguish a specific documented result (an already-existing profile,
    for instance) from a genuine failure instead of pattern-matching messages.
    """

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class WindowsUnavailableError(WindowsApiError):
    """A Windows call was attempted on a platform that cannot provide it."""


class Guids(ctypes.Structure):
    """GUID, CLSID, IID and KNOWNFOLDERID in their documented layout.

    The first three groups are little-endian *numbers*; the little-endianness
    is a property of the memory layout, not of the textual form.  Parse the
    text groups as integers -- never as raw bytes -- or the resulting object
    equals a different identifier, which is the failure the COM fixture hit.
    """

    _fields_ = [
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    ]


def guid_from_text(text: str) -> Guids:
    """Build a :class:`Guids` from the usual ``{....-....-....-....}`` form."""
    groups = text.strip().strip("{}").split("-")
    if [len(group) for group in groups] != [8, 4, 4, 4, 12]:
        raise ValueError(f"malformed GUID: {text!r}")
    try:
        numbers = [int(group, 16) for group in groups[:3]]
        tail = bytes.fromhex(groups[3] + groups[4])
    except ValueError as error:
        raise ValueError(f"malformed GUID: {text!r}") from error
    return Guids(
        numbers[0], numbers[1], numbers[2], (ctypes.c_ubyte * 8)(*tail))


# FOLDERID_LocalAppData, from
# https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid
# Only the identifiers Loki actually uses are declared; the configuration
# locations it does not use stay out so the dead-code gate stays meaningful.
FOLDERID_LOCAL_APP_DATA = "{F1B32785-6FBA-4FCF-9D55-7B8E7F157091}"

# KF_FLAG values, from
# https://learn.microsoft.com/en-us/windows/win32/api/shlobj_core/ne-shlobj_core-known_folder_flag
KF_FLAG_DEFAULT = 0x00000000
KF_FLAG_NO_PACKAGE_REDIRECTION = 0x00010000

_libraries = {}
_signatures = {}


def _library(name: str):
    if sys.platform != "win32" or not hasattr(ctypes, "WinDLL"):
        raise WindowsUnavailableError(f"{name} is available on Windows only")
    if name not in _libraries:
        _libraries[name] = ctypes.WinDLL(name, use_last_error=True)
    return _libraries[name]


def _record_signature(library: str, symbol: str, restype, argtypes) -> None:
    """Remember a symbol's signature, refusing a conflicting redeclaration."""
    signature = (restype, *argtypes)
    key = (library, symbol)
    recorded = _signatures.get(key)
    if recorded is None:
        _signatures[key] = signature
    elif recorded != signature:
        raise WindowsApiError(
            f"{library}.{symbol} is already declared with a different "
            "signature; ctypes keeps one function object per symbol, so a "
            "second declaration would silently replace the first")


def bind(library: str, symbol: str, restype, *argtypes):
    """Declare ``library.symbol`` with its documented signature."""
    call = getattr(_library(library), symbol)
    _record_signature(library, symbol, restype, argtypes)
    call.restype = restype
    call.argtypes = list(argtypes)
    return call


def co_task_mem_free(pointer) -> None:
    """Release memory the shell handed back through an out parameter."""
    bind("ole32", "CoTaskMemFree", None, ctypes.c_void_p)(pointer)


def known_folder(folder_id: str, flags: int = KF_FLAG_DEFAULT) -> str:
    """Resolve a KNOWNFOLDERID to its current path for the calling user.

    ``hToken`` is NULL, which is the documented way to ask for the current
    user's folder.  The buffer is released with ``CoTaskMemFree`` whether or not
    the call succeeded, as the reference requires.
    """
    query = bind(
        "shell32", "SHGetKnownFolderPath", ctypes.c_long,
        ctypes.POINTER(Guids), wintypes.DWORD, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p))
    buffer = ctypes.c_void_p()
    status = query(ctypes.byref(guid_from_text(folder_id)), flags, None,
                   ctypes.byref(buffer))
    try:
        if status != 0:
            raise WindowsApiError(
                f"SHGetKnownFolderPath({folder_id}) failed: "
                f"0x{status & 0xffffffff:08x}")
        return ctypes.wstring_at(buffer.value)
    finally:
        co_task_mem_free(buffer)


# -- named security descriptors ------------------------------------------
# DACL_SECURITY_INFORMATION asks for (or sets) the DACL only.
# PROTECTED_DACL_SECURITY_INFORMATION clears inheritance, which is what makes
# "nothing grants the package SID" an invariant we control rather than one the
# parent directory can change under us.
# 0x800700B7: ERROR_ALREADY_EXISTS.  Measured on all three investigation
# interpreters when CreateAppContainerProfile is called for an existing name.
PROFILE_ALREADY_EXISTS = 0x800700B7

DACL_SECURITY_INFORMATION = 0x00000004
PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
SE_FILE_OBJECT = 1
SDDL_REVISION_1 = 1
ERROR_SUCCESS = 0
TOKEN_QUERY = 0x0008
TOKEN_USER_CLASS = 1


def current_user_sid() -> str:
    """Return the calling process's user SID as a string.

    The SID is what the SDDL grants name, so it has to come from the token
    rather than from anything the environment claims.
    """
    open_token = bind("advapi32", "OpenProcessToken", wintypes.BOOL,
                      ctypes.c_void_p, wintypes.DWORD,
                      ctypes.POINTER(ctypes.c_void_p))
    get_current_process = bind("kernel32", "GetCurrentProcess",
                               ctypes.c_void_p)
    get_token_information = bind(
        "advapi32", "GetTokenInformation", wintypes.BOOL, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD))
    convert_sid = bind("advapi32", "ConvertSidToStringSidW", wintypes.BOOL,
                       ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)
    close = bind("kernel32", "CloseHandle", wintypes.BOOL, ctypes.c_void_p)

    token = ctypes.c_void_p()
    if not open_token(get_current_process(), TOKEN_QUERY,
                      ctypes.byref(token)):
        raise WindowsApiError("OpenProcessToken failed")
    try:
        size = wintypes.DWORD()
        get_token_information(token, TOKEN_USER_CLASS, None, 0,
                              ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not get_token_information(token, TOKEN_USER_CLASS, buffer,
                                     len(buffer), ctypes.byref(size)):
            raise WindowsApiError("GetTokenInformation(TokenUser) failed")
        sid = ctypes.c_void_p.from_buffer(buffer).value  # SID_AND_ATTRIBUTES
        text = ctypes.c_void_p()
        if not convert_sid(sid, ctypes.byref(text)):
            raise WindowsApiError("ConvertSidToStringSidW failed")
        try:
            return ctypes.wstring_at(text)
        finally:
            local_free(text)
    finally:
        close(token)


def set_dacl_sddl(path: str, sddl: str) -> None:
    """Replace ``path``'s DACL with ``sddl`` and clear inheritance."""
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
        if not present.value:
            raise WindowsApiError("security descriptor carries no DACL")
        status = set_named(
            path, SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            None, None, dacl, None)
        if status != ERROR_SUCCESS:
            raise WindowsApiError(
                f"SetNamedSecurityInfoW({path!r}) failed: {status}")
    finally:
        local_free(descriptor)


def dacl_sddl(path: str) -> str:
    """Return ``path``'s DACL as SDDL, for verification and for diffs."""
    get_named = bind(
        "advapi32", "GetNamedSecurityInfoW", wintypes.DWORD,
        wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p))
    convert = bind(
        "advapi32", "ConvertSecurityDescriptorToStringSecurityDescriptorW",
        wintypes.BOOL, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)

    descriptor = ctypes.c_void_p()
    status = get_named(path, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
                       None, None, None, None, ctypes.byref(descriptor))
    if status != ERROR_SUCCESS:
        raise WindowsApiError(f"GetNamedSecurityInfoW({path!r}) failed: {status}")
    try:
        text = ctypes.c_void_p()
        if not convert(descriptor, SDDL_REVISION_1, DACL_SECURITY_INFORMATION,
                       ctypes.byref(text), None):
            raise WindowsApiError("ConvertSecurityDescriptorToString failed")
        try:
            return ctypes.wstring_at(text)
        finally:
            local_free(text)
    finally:
        local_free(descriptor)


def app_container_profile_name_is_usable(name: str) -> bool:
    """Whether ``name`` matches the documented profile-name character set."""
    allowed = set("abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "0123456789.-_")
    return bool(name) and len(name) <= 64 and set(name) <= allowed


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


def derive_app_container_sid(name: str) -> str:
    """Derive the package SID for ``name`` without creating anything."""
    derive = bind("userenv", "DeriveAppContainerSidFromAppContainerName",
                  ctypes.c_long, wintypes.LPCWSTR,
                  ctypes.POINTER(ctypes.c_void_p))
    free_sid = bind("advapi32", "FreeSid", ctypes.c_void_p, ctypes.c_void_p)

    sid = ctypes.c_void_p()
    status = derive(name, ctypes.byref(sid))
    if status < 0:
        raise WindowsApiError(
            f"DeriveAppContainerSidFromAppContainerName({name!r}) failed: "
            f"0x{status & 0xffffffff:08x}")
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


def sid_text(sid) -> str:
    """Convert a SID pointer to its string form."""
    convert = bind("advapi32", "ConvertSidToStringSidW", wintypes.BOOL,
                   ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)

    text = ctypes.c_void_p()
    if not convert(sid, ctypes.byref(text)):
        raise WindowsApiError("ConvertSidToStringSidW failed")
    try:
        return ctypes.wstring_at(text)
    finally:
        local_free(text)
