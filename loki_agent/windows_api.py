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
import enum
import subprocess
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
class KnownFolderFlags(enum.IntFlag):
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


def known_folder(folder_id: str,
                 flags: int = KnownFolderFlags.KF_FLAG_DEFAULT) -> str:
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


# SECURITY_INFORMATION (winnt.h): which parts of a security descriptor a call
# asks for (or sets).
class SecurityInformation(enum.IntFlag):
    DACL_SECURITY_INFORMATION = 0x00000004
    # LABEL_SECURITY_INFORMATION asks for the mandatory integrity label.  The
    # label, not the DACL, is what a low-integrity AppContainer meets first: an
    # object with no label is treated as medium, and no-write-up then refuses a
    # write the DACL grants.
    LABEL_SECURITY_INFORMATION = 0x00000010
    PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
    OWNER_SECURITY_INFORMATION = 0x00000001


SE_FILE_OBJECT = 1
# SE_OBJECT_TYPE for kernel objects (processes, threads, jobs, ...), used when
# a process object's DACL is replaced rather than a file's.
SE_KERNEL_OBJECT = 6
SDDL_REVISION_1 = 1
ERROR_SUCCESS = 0


# TOKEN_ACCESS_RIGHTS (winnt.h): the rights a caller requests on a token handle.
class TokenAccess(enum.IntFlag):
    TOKEN_ASSIGN_PRIMARY = 0x0001
    TOKEN_DUPLICATE = 0x0002
    TOKEN_IMPERSONATE = 0x0004
    TOKEN_QUERY = 0x0008
    TOKEN_ADJUST_PRIVILEGES = 0x0020
    TOKEN_ADJUST_DEFAULT = 0x0080


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
    if not open_token(get_current_process(), TokenAccess.TOKEN_QUERY,
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


def _sddl_from_descriptor(descriptor, information) -> str:
    """Render a security descriptor's requested sections as SDDL text."""
    convert = bind(
        "advapi32", "ConvertSecurityDescriptorToStringSecurityDescriptorW",
        wintypes.BOOL, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)
    text = ctypes.c_void_p()
    if not convert(descriptor, SDDL_REVISION_1, information,
                   ctypes.byref(text), None):
        raise WindowsApiError(
            "ConvertSecurityDescriptorToStringSecurityDescriptorW failed")
    try:
        return ctypes.wstring_at(text)
    finally:
        local_free(text)


def _get_security_info():
    return bind(
        "advapi32", "GetSecurityInfo", wintypes.DWORD, ctypes.c_void_p,
        ctypes.c_int, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p))


def dacl_sddl(path: str) -> str:
    """Return ``path``'s DACL as SDDL, for verification and for diffs."""
    get_named = bind(
        "advapi32", "GetNamedSecurityInfoW", wintypes.DWORD,
        wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p))
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)

    descriptor = ctypes.c_void_p()
    status = get_named(path, SE_FILE_OBJECT, SecurityInformation.DACL_SECURITY_INFORMATION,
                       None, None, None, None, ctypes.byref(descriptor))
    if status != ERROR_SUCCESS:
        raise WindowsApiError(f"GetNamedSecurityInfoW({path!r}) failed: {status}",
                              status=status)
    try:
        return _sddl_from_descriptor(descriptor, SecurityInformation.DACL_SECURITY_INFORMATION)
    finally:
        local_free(descriptor)


VOLUME_NAME_DOS = 0x00000000


def open_directory_handle(path: str):
    """Open ``path`` to query its identity: attribute read, no lock held.

    ``FILE_READ_ATTRIBUTES`` is the whole ask, and it is required:
    ``GetFinalPathNameByHandleW`` fails with ``ERROR_ACCESS_DENIED`` on a
    handle opened with no access, which killed the containment gate at startup
    with "GetFinalPathNameByHandleW sizing failed".  It is not ``GENERIC_READ``
    and not ``READ_CONTROL``: the workspace grant is Modify (``0x1301BF``),
    which allows attribute reads but not ``READ_CONTROL``, so the contained
    runtime can still open its own workspace.  The share mode lets other
    processes write, delete or replace the object while the handle lives, so
    this is still not a lock.  ``FILE_FLAG_BACKUP_SEMANTICS`` is what opens a
    directory at all.
    """
    create_file = bind("kernel32", "CreateFileW", ctypes.c_void_p,
                       wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p)
    handle = create_file(path, AccessMask.FILE_READ_ATTRIBUTES,
                         FileShareMode.FILE_SHARE_ALL, None,
                         FileCreateDisposition.OPEN_EXISTING,
                         FileFlags.FILE_FLAG_BACKUP_SEMANTICS, None)
    if handle is None or handle == INVALID_HANDLE_VALUE:
        raise WindowsApiError(f"CreateFileW({path!r}) failed",
                              status=ctypes.get_last_error())
    return handle


def open_directory_for_acl(path: str):
    """Open ``path`` to judge and then re-ACL the *same* object.

    Unlike ``open_directory_handle`` this requests ``WRITE_DAC`` and
    ``READ_CONTROL`` -- the rights needed to read and replace the object's
    DACL through the handle -- and shares ``READ | WRITE`` without ``DELETE``.
    Withholding DELETE does not pin the name: the directory can still be
    renamed while this handle is open (measured on Windows Server 2025), so the
    binding that matters is the handle itself.  Whatever the caller judges
    through the handle is the object whose DACL it goes on to write, and a
    junction swapped into ``path`` afterwards cannot move the operation to a
    different object.
    """
    create_file = bind("kernel32", "CreateFileW", ctypes.c_void_p,
                       wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p)
    handle = create_file(
        path, (AccessMask.WRITE_DAC | AccessMask.READ_CONTROL
               | AccessMask.FILE_READ_ATTRIBUTES),
        FileShareMode.FILE_SHARE_READ | FileShareMode.FILE_SHARE_WRITE,
        None, FileCreateDisposition.OPEN_EXISTING,
        FileFlags.FILE_FLAG_BACKUP_SEMANTICS, None)
    if handle is None or handle == INVALID_HANDLE_VALUE:
        raise WindowsApiError(f"CreateFileW({path!r}) for ACL failed",
                              status=ctypes.get_last_error())
    return handle


def final_path_from_handle(handle) -> str:
    """The canonical DOS path of the object behind ``handle``.

    The answer comes from the object the handle names, so no name is resolved
    and no ancestor is walked, and a rename between two calls cannot change it.
    ``VOLUME_NAME_DOS`` is requested so the answer is in the same namespace a
    user-supplied path uses, and the extended-length prefix is stripped because
    the paths compared here never carry it.
    """
    get_final = bind("kernel32", "GetFinalPathNameByHandleW", wintypes.DWORD,
                     ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD,
                     wintypes.DWORD)
    size = get_final(handle, None, 0, VOLUME_NAME_DOS)
    if not size:
        status = ctypes.get_last_error()
        raise WindowsApiError(
            f"GetFinalPathNameByHandleW sizing failed: {status}",
            status=status)
    buffer = ctypes.create_unicode_buffer(size + 1)
    written = get_final(handle, buffer, size + 1, VOLUME_NAME_DOS)
    if not written:
        status = ctypes.get_last_error()
        raise WindowsApiError(
            f"GetFinalPathNameByHandleW failed: {status}",
            status=status)
    return _strip_extended_prefix(buffer.value)


def _strip_extended_prefix(path: str) -> str:
    """Map a ``VOLUME_NAME_DOS`` handle path to the plain DOS namespace.

    ``GetFinalPathNameByHandleW`` returns an extended-length path (a UNC one
    for a share); the names Loki compares never carry that prefix, so it is
    removed before any comparison.  Getting this wrong is invisible off
    Windows, so the cases are pinned on every host in ``test_windows_api``.
    """
    if path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + path[len("\\\\?\\UNC\\"):]
    if path.startswith("\\\\?\\"):
        return path[len("\\\\?\\"):]
    return path


def label_sddl(path: str) -> str:
    """Return ``path``'s mandatory integrity label as SDDL text.

    An empty string means the object carries no label.  That an unlabeled
    object is what refuses a low-integrity AppContainer a write is *not*
    established: the AppContainer fixture's workspace is unlabeled as well
    (``private_dacl`` writes no ``S:`` section) and a contained child creates
    ``.loki`` there.  So the label is recorded beside the DACL, not treated as
    the explanation for a denial until a case shows it.
    """
    get_named = bind(
        "advapi32", "GetNamedSecurityInfoW", wintypes.DWORD,
        wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p))
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)

    descriptor = ctypes.c_void_p()
    status = get_named(path, SE_FILE_OBJECT, SecurityInformation.LABEL_SECURITY_INFORMATION,
                       None, None, None, None, ctypes.byref(descriptor))
    if status != ERROR_SUCCESS:
        raise WindowsApiError(
            f"GetNamedSecurityInfoW({path!r}) failed: {status}", status=status)
    try:
        return _sddl_from_descriptor(descriptor, SecurityInformation.LABEL_SECURITY_INFORMATION)
    finally:
        local_free(descriptor)


def handle_owner_sid(handle) -> str:
    """Return the SID that owns the object ``handle`` refers to.

    Reading from the handle, not a pathname, is the point: a rename after the
    open cannot redirect the owner query to a different object.
    """
    get = _get_security_info()
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)
    owner = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    status = get(handle, SE_FILE_OBJECT, SecurityInformation.OWNER_SECURITY_INFORMATION,
                 ctypes.byref(owner), None, None, None,
                 ctypes.byref(descriptor))
    if status != ERROR_SUCCESS:
        raise WindowsApiError(f"GetSecurityInfo(owner) failed: {status}",
                              status=status)
    try:
        return sid_text(owner)
    finally:
        local_free(descriptor)


def handle_dacl_sddl(handle, object_type: int = SE_FILE_OBJECT):
    """Return the DACL of the object ``handle`` refers to, as SDDL, or ``None``.

    ``None`` means the object has no DACL, which grants everyone full access:
    a caller must read that as "not private", never as "no grant".
    ``object_type`` selects the kind of object -- a file by default, or
    ``SE_KERNEL_OBJECT`` for a process.
    """
    get = _get_security_info()
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)
    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    status = get(handle, object_type, SecurityInformation.DACL_SECURITY_INFORMATION,
                 None, None, ctypes.byref(dacl), None,
                 ctypes.byref(descriptor))
    if status != ERROR_SUCCESS:
        raise WindowsApiError(f"GetSecurityInfo(dacl) failed: {status}",
                              status=status)
    try:
        if not dacl.value:
            return None
        return _sddl_from_descriptor(descriptor, SecurityInformation.DACL_SECURITY_INFORMATION)
    finally:
        local_free(descriptor)


def set_handle_dacl(handle, sddl: str, object_type: int = SE_FILE_OBJECT) -> None:
    """Replace the DACL of the object ``handle`` refers to.

    Handle-based on purpose: a temporary file whose *name* is replaced between
    creation and this call cannot redirect the change to a different object,
    which is what the POSIX code achieves by setting the mode through the open
    descriptor rather than by pathname.  ``object_type`` selects the kind of
    object -- a file by default, or ``SE_KERNEL_OBJECT`` for a process or
    thread.
    """
    convert = bind(
        "advapi32", "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        wintypes.BOOL, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    get_dacl = bind(
        "advapi32", "GetSecurityDescriptorDacl", wintypes.BOOL,
        ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL))
    set_info = bind(
        "advapi32", "SetSecurityInfo", wintypes.DWORD, ctypes.c_void_p,
        ctypes.c_int, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p)
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)

    descriptor = ctypes.c_void_p()
    if not convert(sddl, SDDL_REVISION_1, ctypes.byref(descriptor), None):
        raise WindowsApiError(
            "ConvertStringSecurityDescriptorToSecurityDescriptorW failed")
    try:
        present = wintypes.BOOL()
        dacl = ctypes.c_void_p()
        defaulted = wintypes.BOOL()
        if not get_dacl(descriptor, ctypes.byref(present),
                        ctypes.byref(dacl), ctypes.byref(defaulted)):
            raise WindowsApiError("GetSecurityDescriptorDacl failed")
        if not present.value or not dacl.value:
            raise WindowsApiError("the descriptor has no DACL to apply")
        status = set_info(handle, object_type, SecurityInformation.DACL_SECURITY_INFORMATION,
                          None, None, dacl, None)
        if status != ERROR_SUCCESS:
            raise WindowsApiError(f"SetSecurityInfo(dacl) failed: {status}",
                                  status=status)
    finally:
        local_free(descriptor)


def set_named_dacl(path: str, sddl: str) -> None:
    """Replace ``path``'s DACL from SDDL.

    A pathname fallback for handles that were opened without ``WRITE_DAC``
    (``SetSecurityInfo`` on such a handle is refused).  Like the POSIX
    ``os.chmod`` fallback, it lacks the descriptor branch's protection against a
    replaced temporary entry, so the caller must name that gap.
    """
    convert = bind(
        "advapi32", "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        wintypes.BOOL, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    get_dacl = bind(
        "advapi32", "GetSecurityDescriptorDacl", wintypes.BOOL,
        ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL))
    set_named = bind(
        "advapi32", "SetNamedSecurityInfoW", wintypes.DWORD, wintypes.LPWSTR,
        ctypes.c_int, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p)
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)

    descriptor = ctypes.c_void_p()
    if not convert(sddl, SDDL_REVISION_1, ctypes.byref(descriptor), None):
        raise WindowsApiError(
            "ConvertStringSecurityDescriptorToSecurityDescriptorW failed")
    try:
        present = wintypes.BOOL()
        dacl = ctypes.c_void_p()
        defaulted = wintypes.BOOL()
        if not get_dacl(descriptor, ctypes.byref(present),
                        ctypes.byref(dacl), ctypes.byref(defaulted)):
            raise WindowsApiError("GetSecurityDescriptorDacl failed")
        if not present.value or not dacl.value:
            raise WindowsApiError("the descriptor has no DACL to apply")
        status = set_named(path, SE_FILE_OBJECT, SecurityInformation.DACL_SECURITY_INFORMATION,
                           None, None, dacl, None)
        if status != ERROR_SUCCESS:
            raise WindowsApiError(
                f"SetNamedSecurityInfoW({path!r}) failed: {status}",
                status=status)
    finally:
        local_free(descriptor)


def app_container_profile_name_is_usable(name: str) -> bool:
    """Whether ``name`` matches the documented profile-name character set."""
    allowed = set("abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "0123456789.-_")
    return bool(name) and len(name) <= 64 and set(name) <= allowed


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


# Well-known SIDs that the kernel spells as SDDL aliases when it renders a
# DACL (SID Strings reference).  On Windows the API is authoritative and this
# table is not consulted; it is the off-Windows path that keeps the privacy
# predicate runnable, and testable, where the API is not.  A wrong entry could
# therefore only weaken a test, never a running system; the native test checks
# each one against the API's own answer.  An alias not listed here still
# refuses on a non-Windows host rather than passing.
_WELL_KNOWN_SIDS = {
    "SY": "S-1-5-18",      # LOCAL SYSTEM
    "BA": "S-1-5-32-544",  # BUILTIN\Administrators
    "OW": "S-1-3-4",       # OWNER RIGHTS
    "CO": "S-1-3-0",       # CREATOR OWNER
    "WD": "S-1-1-0",       # Everyone
    "AU": "S-1-5-11",      # Authenticated Users
    "BU": "S-1-5-32-545",  # BUILTIN\Users
    "IU": "S-1-5-4",       # INTERACTIVE
    "AN": "S-1-5-7",       # ANONYMOUS LOGON
}


def canonical_sid(text: str) -> str:
    """Return the full ``S-1-...`` spelling of a SID or an SDDL alias.

    A DACL read back from the kernel spells well-known trustees as SDDL
    aliases (``SY``, ``BA``, ``OW``) and everyone else in full form, so one
    trustee can arrive in two spellings and a comparison needs one.  A string
    already in full form is returned as is.
    """
    if text.startswith("S-"):
        return text
    if sys.platform != "win32":
        known = _WELL_KNOWN_SIDS.get(text)
        if known is None:
            raise WindowsUnavailableError(
                f"cannot resolve the SID alias {text!r} without Windows")
        return known
    convert = bind("advapi32", "ConvertStringSidToSidW", wintypes.BOOL,
                   wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p))
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)
    sid = ctypes.c_void_p()
    if not convert(text, ctypes.byref(sid)):
        raise WindowsApiError(f"ConvertStringSidToSidW({text!r}) failed",
                              status=ctypes.get_last_error())
    try:
        return sid_text(sid)
    finally:
        local_free(sid)


# -- container identity and launch ---------------------------------------
# An AppContainer token is supplied when the process is created -- through the
# security-capabilities attribute on CreateProcess -- rather than adopted by a
# running process.  These declarations inspect a token and create a child with
# one, so a launcher can verify a child while it is still suspended and the
# runtime can check its own token.  Nothing here creates or modifies a profile
# or a DACL.
#
# TOKEN_INFORMATION_CLASS (winnt.h).  The Learn page names the enumerators but
# gives a value only for TokenUser (1), so these two numbers are from the
# header and are not confirmed by that page.
TOKEN_IS_APP_CONTAINER_CLASS = 29
TOKEN_APP_CONTAINER_SID_CLASS = 31


# PROCESS_ACCESS_RIGHTS, from
# https://learn.microsoft.com/en-us/windows/win32/procthread/process-security-and-access-rights
class ProcessAccess(enum.IntFlag):
    PROCESS_TERMINATE = 0x0001
    PROCESS_VM_OPERATION = 0x0008
    PROCESS_VM_READ = 0x0010
    PROCESS_VM_WRITE = 0x0020
    PROCESS_DUP_HANDLE = 0x0040
    PROCESS_CREATE_PROCESS = 0x0080
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


# Process creation flags, from
# https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags
class CreateProcessFlags(enum.IntFlag):
    CREATE_SUSPENDED = 0x00000004
    CREATE_UNICODE_ENVIRONMENT = 0x00000400
    EXTENDED_STARTUPINFO_PRESENT = 0x00080000


class StartupInfoFlags(enum.IntFlag):
    STARTF_USESTDHANDLES = 0x00000100


# SetInformationJobObject's JOB_OBJECT_LIMIT_* (winnt.h): which limits the
# JOBOBJECT_EXTENDED_LIMIT_INFORMATION struct is setting.
class JobObjectLimits(enum.IntFlag):
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


# PROC_THREAD_ATTRIBUTE_* (winbase.h): one attribute id per
# UpdateProcThreadAttribute call, never combined.  The names map to these
# values, not to the bare enumerators.
class ProcThreadAttribute(enum.IntEnum):
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x20002
    PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x20009
    # The attribute value is the HPCON itself, and the pseudoconsole then
    # supplies the child's standard handles.
    PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x00020016

# File access for the containment probe.  The probe *asks* for rights it must be
# denied and treats the refusal as the pass; it never reads or writes.
#
# ACCESS_MASK values (winnt.h): the rights a security descriptor grants, and the
# rights a CreateFileW open asks for.  Declared once because read-only modules
# compare them: ``windows_verify`` requests them from inside the container, and
# ``windows_state`` evaluates DACL rights against them.


class AccessMask(enum.IntFlag):
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    FILE_GENERIC_READ = 0x00120089
    FILE_GENERIC_WRITE = 0x00120116
    FILE_GENERIC_EXECUTE = 0x001200A0
    FILE_ALL_ACCESS = 0x001F01FF
    DELETE = 0x00010000
    READ_CONTROL = 0x00020000
    WRITE_DAC = 0x00040000
    WRITE_OWNER = 0x00080000
    SYNCHRONIZE = 0x00100000
    ACCESS_SYSTEM_SECURITY = 0x01000000
    # A directory's FILE_WRITE_DATA is FILE_ADD_FILE and its FILE_APPEND_DATA is
    # FILE_ADD_SUBDIRECTORY; the probe requests the former to test create
    # denial.
    FILE_WRITE_DATA = 0x00000002
    FILE_APPEND_DATA = 0x00000004
    FILE_READ_ATTRIBUTES = 0x00000080


# FILE_SHARE_* (winnt.h): CreateFileW's dwShareMode -- what other opens may do
# with the object while this handle is held.
class FileShareMode(enum.IntFlag):
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_ALL = 0x00000007


# CreateFileW's dwCreationDisposition (winnt.h): one of these, not a set.
class FileCreateDisposition(enum.IntEnum):
    CREATE_NEW = 1
    CREATE_ALWAYS = 2
    OPEN_EXISTING = 3
    OPEN_ALWAYS = 4
    TRUNCATE_EXISTING = 5


# FILE_FLAG_* (winnt.h).  CreateFileW's dwFlagsAndAttributes takes these *and*
# the FILE_ATTRIBUTE_* values in one DWORD -- two disjoint families in a single
# parameter, so the same field is written here with a FILE_FLAG_* member and
# read back later as a FILE_ATTRIBUTE_* one.  The function page names both in
# its dwFlagsAndAttributes description:
# https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew
#
# FILE_FLAG_BACKUP_SEMANTICS lets CreateFileW open a directory, which the
# credential tree and the workspace both are.
class FileFlags(enum.IntFlag):
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    FILE_FLAG_OVERLAPPED = 0x40000000
    FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000


ERROR_FILE_NOT_FOUND = 2
ERROR_PATH_NOT_FOUND = 3
ERROR_ACCESS_DENIED = 5
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class SecurityCapabilities(ctypes.Structure):
    """SECURITY_CAPABILITIES: the package SID a child is created in."""

    # Fixed-width types, not ``wintypes.DWORD``: the latter is ``c_ulong``,
    # which is 8 bytes on LP64 non-Windows hosts and would give this structure
    # the wrong layout in the portable tests that pin it.
    _fields_ = [
        ("AppContainerSid", ctypes.c_void_p),
        ("Capabilities", ctypes.c_void_p),
        ("CapabilityCount", ctypes.c_uint32),
        ("Reserved", ctypes.c_uint32),
    ]


class StartupInfo(ctypes.Structure):
    """STARTUPINFOW.

    ``ctypes.wintypes`` does not define STARTUPINFO -- it carries scalar and
    handle aliases and a few unrelated structures -- so the layout is declared
    here.  The test pins its size and field order: a missing or misplaced member
    shifts every field after it.
    """

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", ctypes.c_uint32),
        ("dwY", ctypes.c_uint32),
        ("dwXSize", ctypes.c_uint32),
        ("dwYSize", ctypes.c_uint32),
        ("dwXCountChars", ctypes.c_uint32),
        ("dwYCountChars", ctypes.c_uint32),
        ("dwFillAttribute", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("wShowWindow", ctypes.c_uint16),
        ("cbReserved2", ctypes.c_uint16),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", ctypes.c_void_p),
        ("hStdOutput", ctypes.c_void_p),
        ("hStdError", ctypes.c_void_p),
    ]


class StartupInfoEx(ctypes.Structure):
    """STARTUPINFOEXW: a STARTUPINFOW plus the attribute list."""

    _fields_ = [
        ("StartupInfo", StartupInfo),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class ProcessInformation(ctypes.Structure):
    """PROCESS_INFORMATION: the handles and IDs CreateProcessW returns."""

    _fields_ = [
        ("hProcess", ctypes.c_void_p),
        ("hThread", ctypes.c_void_p),
        ("dwProcessId", ctypes.c_uint32),
        ("dwThreadId", ctypes.c_uint32),
    ]


def _get_token_information():
    return bind("advapi32", "GetTokenInformation", wintypes.BOOL,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))


def close_handle(handle) -> None:
    """Release a kernel handle."""
    bind("kernel32", "CloseHandle", wintypes.BOOL, ctypes.c_void_p)(handle)


def current_process_handle():
    """The pseudo-handle for the calling process; it need not be closed."""
    return bind("kernel32", "GetCurrentProcess", ctypes.c_void_p)()


def open_with_access(path, desired_access,
                     creation=FileCreateDisposition.OPEN_EXISTING,
                     flags=FileFlags.FILE_FLAG_BACKUP_SEMANTICS,
                     share_mode=FileShareMode.FILE_SHARE_ALL):
    """Open ``path`` requesting ``desired_access``, returning the handle.

    The containment probe uses this to request rights it must not have, so a
    refusal is the expected result and its ``status`` carries the Win32 error.
    If the call unexpectedly succeeds the caller must close the handle without
    using it.  ``FILE_FLAG_BACKUP_SEMANTICS`` is the default so that a
    directory can be opened.
    """
    create_file = bind("kernel32", "CreateFileW", ctypes.c_void_p,
                       wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p)
    handle = create_file(path, desired_access, share_mode, None, creation,
                         flags, None)
    if handle is None or handle == INVALID_HANDLE_VALUE:
        raise WindowsApiError(f"CreateFileW({path!r}) failed",
                              status=ctypes.get_last_error())
    return handle


def open_process_token(process, rights: int = TokenAccess.TOKEN_QUERY):
    """Open ``process``'s token for the requested rights.

    ``process`` is a handle the caller already holds; opening a child created
    suspended is how the launcher inspects it before it runs.
    """
    open_token = bind("advapi32", "OpenProcessToken", wintypes.BOOL,
                      ctypes.c_void_p, wintypes.DWORD,
                      ctypes.POINTER(ctypes.c_void_p))
    token = ctypes.c_void_p()
    if not open_token(process, rights, ctypes.byref(token)):
        raise WindowsApiError("OpenProcessToken failed")
    return token


def token_is_app_container(token) -> bool:
    """Whether ``token`` is an AppContainer (lowbox) token.

    This is a property of the token, so the same check covers a child created
    suspended and the runtime's own token.
    """
    get_info = _get_token_information()
    value = wintypes.BOOL()
    returned = wintypes.DWORD()
    if not get_info(token, TOKEN_IS_APP_CONTAINER_CLASS,
                    ctypes.byref(value), ctypes.sizeof(value),
                    ctypes.byref(returned)):
        raise WindowsApiError(
            "GetTokenInformation(TokenIsAppContainer) failed")
    return bool(value.value)


TOKEN_INTEGRITY_LEVEL = 25


def token_integrity_level(token) -> str:
    """The integrity level SID of ``token``, as text.

    The process's own level is half of what the mandatory policy decides on;
    the other half is the object's label.  A low-integrity process writing to
    an object labelled (or defaulted) higher is refused whatever its DACL
    grants, so a report of what a contained process met has to name both.
    """
    get_info = _get_token_information()
    size = wintypes.DWORD()
    get_info(token, TOKEN_INTEGRITY_LEVEL, None, 0, ctypes.byref(size))
    if not size.value:
        raise WindowsApiError(
            "GetTokenInformation(TokenIntegrityLevel) sizing failed")
    buffer = ctypes.create_string_buffer(size.value)
    if not get_info(token, TOKEN_INTEGRITY_LEVEL, buffer, size.value,
                    ctypes.byref(size)):
        raise WindowsApiError(
            "GetTokenInformation(TokenIntegrityLevel) failed")
    return sid_text(ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0])


def token_app_container_sid(token) -> str:
    """Return ``token``'s AppContainer package SID as a string.

    ``TokenAppContainerSid`` yields a ``TOKEN_APPCONTAINER_INFORMATION`` whose
    single member is the SID pointer, so the buffer is read as one pointer.
    """
    get_info = _get_token_information()
    size = wintypes.DWORD()
    get_info(token, TOKEN_APP_CONTAINER_SID_CLASS, None, 0,
             ctypes.byref(size))
    if not size.value:
        raise WindowsApiError(
            "GetTokenInformation(TokenAppContainerSid) reported no size")
    buffer = ctypes.create_string_buffer(size.value)
    if not get_info(token, TOKEN_APP_CONTAINER_SID_CLASS, buffer,
                    len(buffer), ctypes.byref(size)):
        raise WindowsApiError(
            "GetTokenInformation(TokenAppContainerSid) failed")
    sid = ctypes.c_void_p.from_buffer(buffer).value
    if not sid:
        raise WindowsApiError(
            "GetTokenInformation(TokenAppContainerSid) returned a null SID")
    return sid_text(sid)


def resume_thread(thread) -> None:
    """Resume a thread created suspended; the child starts running here."""
    resumed = bind("kernel32", "ResumeThread", wintypes.DWORD, ctypes.c_void_p)
    if resumed(thread) == 0xFFFFFFFF:
        raise WindowsApiError("ResumeThread failed")


def terminate_process(process, exit_code: int = 1) -> None:
    """Terminate a process, used to drop a child that failed verification."""
    terminate = bind("kernel32", "TerminateProcess", wintypes.BOOL,
                     ctypes.c_void_p, ctypes.c_uint)
    if not terminate(process, exit_code):
        raise WindowsApiError("TerminateProcess failed")


class Coord(ctypes.Structure):
    """COORD: a console cell coordinate (x is columns, y is rows)."""

    _fields_ = [("x", ctypes.c_int16), ("y", ctypes.c_int16)]


def pseudoconsole_create(cols, rows, input_handle, output_handle):
    """Create a pseudoconsole; return its ``HPCON`` handle.

    ``input_handle`` is the read end of the host's input pipe (host writes
    child input into the write end); ``output_handle`` is the write end of the
    host's output pipe (host reads the rendered child output from the read
    end).  The pseudoconsole keeps its own copies, so the caller must close
    the ends it passed after the child is created.
    """
    create = bind("kernel32", "CreatePseudoConsole", ctypes.c_long,
                  Coord, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                  ctypes.POINTER(ctypes.c_void_p))
    hpc = ctypes.c_void_p()
    status = create(Coord(cols, rows), input_handle, output_handle, 0,
                    ctypes.byref(hpc))
    if status < 0:
        raise WindowsApiError(
            "CreatePseudoConsole failed", status=status & 0xffffffff)
    return hpc


def pseudoconsole_resize(hpc, cols, rows) -> None:
    """Resize the pseudoconsole; the child's console observes the new size."""
    resize = bind("kernel32", "ResizePseudoConsole", ctypes.c_long,
                  ctypes.c_void_p, Coord)
    status = resize(hpc, Coord(cols, rows))
    if status < 0:
        raise WindowsApiError(
            "ResizePseudoConsole failed", status=status & 0xffffffff)


def pseudoconsole_close(hpc) -> None:
    """Close the pseudoconsole.  The attached client must have exited first."""
    bind("kernel32", "ClosePseudoConsole", None, ctypes.c_void_p)(hpc)


def peek_named_pipe(handle) -> int:
    """Bytes available to read on ``handle`` without consuming them, or 0."""
    peek = bind("kernel32", "PeekNamedPipe", wintypes.BOOL,
                ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(wintypes.DWORD))
    available = wintypes.DWORD()
    if not peek(handle, None, 0, None, ctypes.byref(available), None):
        return 0
    return available.value


def wait_for_single_object(handle, milliseconds: int) -> int:
    """``WaitForSingleObject``: returns 0 when signalled, 258 on timeout."""
    wait = bind("kernel32", "WaitForSingleObject", wintypes.DWORD,
                ctypes.c_void_p, wintypes.DWORD)
    return wait(handle, milliseconds)


def get_exit_code_process(handle) -> int:
    """``GetExitCodeProcess``; ``259`` means the process is still running."""
    get = bind("kernel32", "GetExitCodeProcess", wintypes.BOOL,
               ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))
    code = wintypes.DWORD()
    if not get(handle, ctypes.byref(code)):
        raise WindowsApiError("GetExitCodeProcess failed")
    return code.value


def create_process_with_pseudoconsole(executable, arguments, hpc, *,
                                      environment=None,
                                      current_directory=None):
    """Create ``executable`` attached to the pseudoconsole ``hpc``.

    A plain console child (no AppContainer): the pseudoconsole supplies the
    child's standard handles.  ``STARTF_USESTDHANDLES`` with null handles is
    set so ``CreateProcessW`` does not duplicate this process's redirected
    handles into the child (microsoft/terminal discussion 15814).
    """
    initialize = bind("kernel32", "InitializeProcThreadAttributeList",
                      wintypes.BOOL, ctypes.c_void_p, wintypes.DWORD,
                      wintypes.DWORD, ctypes.POINTER(ctypes.c_size_t))
    update = bind("kernel32", "UpdateProcThreadAttribute", wintypes.BOOL,
                  ctypes.c_void_p, wintypes.DWORD, ctypes.c_size_t,
                  ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                  ctypes.c_void_p)
    delete = bind("kernel32", "DeleteProcThreadAttributeList", None,
                  ctypes.c_void_p)
    create = bind("kernel32", "CreateProcessW", wintypes.BOOL,
                  wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p,
                  ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD,
                  ctypes.c_void_p, wintypes.LPCWSTR,
                  ctypes.POINTER(StartupInfoEx),
                  ctypes.POINTER(ProcessInformation))

    size = ctypes.c_size_t()
    # The sizing call is documented to fail with ERROR_INSUFFICIENT_BUFFER
    # while filling in the required size; only a wrong error is a defect.
    initialize(None, 1, 0, ctypes.byref(size))
    if ctypes.get_last_error() != 122 or not size.value:
        raise WindowsApiError("InitializeProcThreadAttributeList sizing failed")
    storage = ctypes.create_string_buffer(size.value)
    attributes = ctypes.cast(storage, ctypes.c_void_p)
    initialized = False
    try:
        if not initialize(attributes, 1, 0, ctypes.byref(size)):
            raise WindowsApiError("InitializeProcThreadAttributeList failed")
        initialized = True
        # The value is the HPCON itself, not its address (the documented call
        # passes the handle, not a pointer to it).  hpc is already a c_void_p
        # holding that handle, so it is passed as-is; re-wrapping it in
        # c_void_p would try to convert a c_void_p into a pointer and fail.
        if not update(attributes, 0,
                      ProcThreadAttribute.PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                      hpc, ctypes.sizeof(ctypes.c_void_p),
                      None, None):
            raise WindowsApiError(
                "UpdateProcThreadAttribute(PSEUDOCONSOLE) failed")

        startup = StartupInfoEx()
        startup.StartupInfo.cb = ctypes.sizeof(StartupInfoEx)
        startup.lpAttributeList = attributes
        startup.StartupInfo.dwFlags = StartupInfoFlags.STARTF_USESTDHANDLES
        # hStdInput/hStdOutput/hStdError stay NULL; the pseudoconsole supplies
        # them, and null handles keep the compatibility path from duplicating
        # the parent's redirected handles into the child.
        information = ProcessInformation()
        command_line = ctypes.create_unicode_buffer(
            subprocess.list2cmdline([executable, *arguments]))
        flags = CreateProcessFlags.EXTENDED_STARTUPINFO_PRESENT
        block = environment_block(environment, current_directory)
        if block is not None:
            flags |= CreateProcessFlags.CREATE_UNICODE_ENVIRONMENT
        if not create(executable, command_line, None, None, False, flags,
                      block, current_directory, ctypes.byref(startup),
                      ctypes.byref(information)):
            raise WindowsApiError(
                "CreateProcessW failed: "
                f"{ctypes.get_last_error()}")
        return information
    finally:
        if initialized:
            delete(attributes)


def _drive_of(path: str) -> str:
    """The ``X:`` drive of ``path``, or ``""``.

    Stated here rather than with ``os.path.splitdrive`` so the value does not
    depend on which platform's path rules the calling host happens to use.
    """
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        return path[:2].upper()
    return ""


def environment_block(environment: dict, current_directory=None):
    """Encode ``environment`` as the Unicode block ``CreateProcessW`` accepts.

    The supplied block replaces the inherited one, and Windows does not
    propagate the per-drive current-directory entries into it; a block that
    omits the entry for the drive holding the child's current directory is
    rejected with ``ERROR_ENVVAR_NOT_FOUND`` (203).  Carry over any the parent
    already holds, set the child's own directory for its drive, and sort the
    whole block (the system expects a sorted environment; '=' sorts before
    letters, so the drive entries come first on their own).

    Returns a ``ctypes`` unicode buffer ready for ``lpEnvironment``, or
    ``None`` if ``environment`` is ``None``.
    """
    if environment is None:
        return None
    entries = {}
    for entry in drive_environment_entries():
        name = entry.split("=", 2)[1]
        entries[name] = entry
    if current_directory:
        drive = _drive_of(current_directory)
        if drive:
            entries[drive] = f"={drive}={current_directory}"
    for key, value in environment.items():
        if not key or '=' in key or '\0' in key or '\0' in value:
            raise ValueError("invalid child environment entry")
        entries[key] = f"{key}={value}"
    # Sort the full entries, not the keys: the per-drive entries begin
    # with '=' (0x3d), which sorts before every letter, so they come
    # first on their own.  Sorting the keys would place "C:" among the
    # ordinary "C..." names and misorder the block.
    ordered = sorted(entries.values(), key=str.upper)
    return ctypes.create_unicode_buffer('\0'.join(ordered) + '\0\0')


def drive_environment_entries() -> list:
    """The ``=X:=...`` per-drive current-directory entries of this process.

    Windows keeps a hidden environment entry per drive recording that drive's
    current directory.  A supplied environment block that omits the entry for
    the drive holding the child's current directory is rejected by
    ``CreateProcessW`` with ``ERROR_ENVVAR_NOT_FOUND`` (203), so they are read
    from this process's block and placed at the front of the child's.
    """
    get = bind("kernel32", "GetEnvironmentStringsW", ctypes.c_void_p)
    free = bind("kernel32", "FreeEnvironmentStringsW", wintypes.BOOL,
                ctypes.c_void_p)
    pointer = get()
    if not pointer:
        return []
    entries = []
    try:
        address = pointer
        while True:
            text = ctypes.wstring_at(address)
            if not text:
                break
            if text.startswith("="):
                entries.append(text)
            address += (len(text) + 1) * ctypes.sizeof(ctypes.c_wchar)
    finally:
        free(pointer)
    return entries


def create_process_in_app_container(executable, arguments, package_sid,
                                    current_directory=None,
                                    inherited_handles=None, environment=None,
                                    standard_handles=None, pseudoconsole=None):
    """Create ``executable`` suspended inside the AppContainer ``package_sid``.

    ``package_sid`` is the package SID in string form, as
    :func:`derive_app_container_sid` returns; the profile must already exist,
    since this does not create a profile or change a DACL. ``inherited_handles`` are
    the handles the child must receive, passed explicitly through
    ``PROC_THREAD_ATTRIBUTE_HANDLE_LIST`` with inheritance otherwise off.
    ``pseudoconsole`` attaches the child to an existing ``HPCON`` (ConPTY), which
    then supplies its standard handles in place of the caller's console.

    The child is left suspended, so the caller can inspect its token and then
    resume or terminate it.  The returned :class:`ProcessInformation` owns the
    process and thread handles; the caller closes them.
    """
    convert_sid = bind("advapi32", "ConvertStringSidToSidW", wintypes.BOOL,
                       wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p))
    local_free = bind("kernel32", "LocalFree", ctypes.c_void_p,
                      ctypes.c_void_p)
    initialize = bind("kernel32", "InitializeProcThreadAttributeList",
                      wintypes.BOOL, ctypes.c_void_p, wintypes.DWORD,
                      wintypes.DWORD, ctypes.POINTER(ctypes.c_size_t))
    update = bind("kernel32", "UpdateProcThreadAttribute", wintypes.BOOL,
                  ctypes.c_void_p, wintypes.DWORD, ctypes.c_size_t,
                  ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                  ctypes.c_void_p)
    delete = bind("kernel32", "DeleteProcThreadAttributeList", None,
                  ctypes.c_void_p)
    create = bind("kernel32", "CreateProcessW", wintypes.BOOL,
                  wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p,
                  ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD,
                  ctypes.c_void_p, wintypes.LPCWSTR,
                  ctypes.POINTER(StartupInfoEx),
                  ctypes.POINTER(ProcessInformation))

    sid = ctypes.c_void_p()
    if not convert_sid(package_sid, ctypes.byref(sid)):
        raise WindowsApiError(f"ConvertStringSidToSidW({package_sid!r}) failed")
    # The list must be sized for every attribute it will receive: one for the
    # package SID, plus one when handles are inherited, plus one when the child
    # is attached to a pseudoconsole.  Sizing it for fewer and then adding more
    # makes the extra UpdateProcThreadAttribute fail.
    attribute_count = 1 + (1 if inherited_handles else 0) + (
        1 if pseudoconsole is not None else 0)
    size = ctypes.c_size_t()
    attribute_list = None
    initialized = False
    handle_array = None
    try:
        initialize(None, attribute_count, 0, ctypes.byref(size))
        if not size.value:
            raise WindowsApiError(
                "InitializeProcThreadAttributeList reported no size")
        attribute_storage = ctypes.create_string_buffer(size.value)
        attribute_list = ctypes.cast(attribute_storage, ctypes.c_void_p)
        if not initialize(attribute_list, attribute_count, 0,
                          ctypes.byref(size)):
            raise WindowsApiError(
                "InitializeProcThreadAttributeList failed")
        initialized = True
        capabilities = SecurityCapabilities(sid, None, 0, 0)
        if not update(attribute_list, 0,
                      ProcThreadAttribute.PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
                      ctypes.byref(capabilities),
                      ctypes.sizeof(capabilities), None, None):
            raise WindowsApiError(
                "UpdateProcThreadAttribute(SECURITY_CAPABILITIES) failed")
        if inherited_handles:
            handle_array = (ctypes.c_void_p * len(inherited_handles))(
                *inherited_handles)
            if not update(attribute_list, 0,
                          ProcThreadAttribute.PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                          ctypes.cast(handle_array, ctypes.c_void_p),
                          ctypes.sizeof(handle_array), None, None):
                raise WindowsApiError(
                    "UpdateProcThreadAttribute(HANDLE_LIST) failed")
        if pseudoconsole is not None:
            # lpValue is the HPCON value itself, not its address: that is the
            # form the documented ConPTY sample passes.  The generic
            # UpdateProcThreadAttribute wording ("a pointer to the attribute
            # value") reads the other way; the sample for this attribute is the
            # contract.  Note also that a pseudoconsole does not reliably
            # supply a child's standard handles when the parent's are
            # redirected -- the parent's are duplicated into a console child
            # unless STARTF_USESTDHANDLES with null handles suppresses it
            # (microsoft/terminal discussion 15814).  A caller that needs that
            # must pass standard_handles as nulls, not leave them unset.
            hpcon = ctypes.c_void_p(pseudoconsole)
            if not update(attribute_list, 0,
                          ProcThreadAttribute.PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                          hpcon, ctypes.sizeof(hpcon),
                          None, None):
                raise WindowsApiError(
                    "UpdateProcThreadAttribute(PSEUDOCONSOLE) failed")

        startup = StartupInfoEx()
        startup.StartupInfo.cb = ctypes.sizeof(StartupInfoEx)
        startup.lpAttributeList = attribute_list
        if standard_handles is not None:
            startup.StartupInfo.dwFlags = StartupInfoFlags.STARTF_USESTDHANDLES
            (startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput,
             startup.StartupInfo.hStdError) = standard_handles
        information = ProcessInformation()
        command_line = ctypes.create_unicode_buffer(
            subprocess.list2cmdline([executable, *arguments]))
        flags = (CreateProcessFlags.CREATE_SUSPENDED
                 | CreateProcessFlags.EXTENDED_STARTUPINFO_PRESENT)
        block = environment_block(environment, current_directory)
        if block is not None:
            flags |= CreateProcessFlags.CREATE_UNICODE_ENVIRONMENT
        if not create(executable, command_line, None, None,
                      bool(inherited_handles), flags, block,
                      current_directory, ctypes.byref(startup),
                      ctypes.byref(information)):
            raise WindowsApiError(
                "CreateProcessW failed: "
                f"{ctypes.get_last_error()}")
        return information
    finally:
        if initialized:
            delete(attribute_list)
        local_free(sid)


# -- handle-relative file operations -------------------------------------
# ``CreateFileW`` resolves a pathname, so it cannot express POSIX ``dir_fd``:
# a child opened against a *retained directory handle* is bound to the directory
# object that was verified, not a pathname resolved a second time.  That is
# ``NtCreateFile`` with ``RootDirectory``, which the Windows storage
# investigation measured working (including after the directory is renamed and
# its old pathname recreated).
#
# ``FILE_OPEN_REPARSE_POINT`` opens a reparse point itself rather than following
# it, so the object inspected is the link, never its target.  Neither it nor
# ``RootDirectory`` stops traversal through an *intermediate* reparse point, so
# a caller must confirm the root handle is a real directory before using it.
#
# ``SetFileInformationByHandle`` cannot rename to a relative name -- the tested
# Win32 form returned ERROR_INVALID_PARAMETER -- so rename and delete use the
# native ``NtSetInformationFile`` / ``SetFileInformationByHandle`` as measured.

# OBJECT_ATTRIBUTES.Attributes (ntdef.h): the attribute bits the NT object calls
# take in that struct.  Named Nt* to keep it clear of the struct below.
class NtObjectAttributes(enum.IntFlag):
    OBJ_CASE_INSENSITIVE = 0x00000040


# NtCreateFile's CreateOptions (ntifs.h).  Distinct from FILE_FLAG_*, which goes
# to CreateFileW's dwFlagsAndAttributes: these are the NT call's own options.
class NtCreateOptions(enum.IntFlag):
    FILE_NON_DIRECTORY_FILE = 0x00000040
    FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
    FILE_OPEN_REPARSE_POINT = 0x00200000


# NtCreateFile's CreateDisposition (ntifs.h): one of these, not a set.  The
# Win32 counterpart is FileCreateDisposition.
class NtCreateDisposition(enum.IntEnum):
    FILE_SUPERSEDE = 0
    FILE_OPEN = 1
    FILE_CREATE = 2
    FILE_OPEN_IF = 3
    FILE_OVERWRITE = 4
    FILE_OVERWRITE_IF = 5


# FILE_ATTRIBUTE_* (winnt.h): the attribute bits of a file.  Read back from
# BY_HANDLE_FILE_INFORMATION.dwFileAttributes, and given at creation in
# CreateFileW's dwFlagsAndAttributes -- the same DWORD that carries FILE_FLAG_*,
# two disjoint families in one parameter.
class FileAttribute(enum.IntFlag):
    FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400


FILE_RENAME_INFORMATION = 10
FILE_DISPOSITION_INFO = 4
# NTSTATUS values.  ``NtCreateFile`` returns these directly; the caller
# translates the ones it can name to the OSError the storage protocol expects.
STATUS_OBJECT_NAME_NOT_FOUND = 0xC0000034
STATUS_OBJECT_PATH_NOT_FOUND = 0xC000003A
STATUS_OBJECT_NAME_COLLISION = 0xC0000035
STATUS_ACCESS_DENIED = 0xC0000022
STATUS_NOT_A_DIRECTORY = 0xC0000103


class FileTime(ctypes.Structure):
    """FILETIME: two DWORDs, not a 64-bit integer, so the layout is exact.

    Fixed-width ``c_uint32`` rather than ``wintypes.DWORD``: the latter is
    ``c_ulong``, which is 8 bytes on LP64 hosts and would give this structure
    the wrong layout in the portable tests that pin it.
    """

    _fields_ = [("dwLowDateTime", ctypes.c_uint32),
                ("dwHighDateTime", ctypes.c_uint32)]


class UnicodeString(ctypes.Structure):
    """UNICODE_STRING: lengths in *bytes*, buffer separately owned."""

    _fields_ = [("Length", ctypes.c_uint16),
                ("MaximumLength", ctypes.c_uint16),
                ("Buffer", ctypes.c_void_p)]


class ObjectAttributes(ctypes.Structure):
    """OBJECT_ATTRIBUTES: ``RootDirectory`` is what makes an open relative."""

    _fields_ = [("Length", ctypes.c_uint32),
                ("RootDirectory", ctypes.c_void_p),
                ("ObjectName", ctypes.POINTER(UnicodeString)),
                ("Attributes", ctypes.c_uint32),
                ("SecurityDescriptor", ctypes.c_void_p),
                ("SecurityQualityOfService", ctypes.c_void_p)]


class IoStatusBlock(ctypes.Structure):
    """IO_STATUS_BLOCK: the first member is a union of NTSTATUS and a pointer."""

    _fields_ = [("Status", ctypes.c_size_t),
                ("Information", ctypes.c_size_t)]


class FileRenameInformation(ctypes.Structure):
    """FILE_RENAME_INFORMATION with ``RootDirectory`` for a relative rename."""

    _fields_ = [("ReplaceIfExists", ctypes.c_ubyte),
                ("RootDirectory", ctypes.c_void_p),
                ("FileNameLength", ctypes.c_uint32),
                ("FileName", ctypes.c_uint16 * 1)]


class ByHandleFileInformation(ctypes.Structure):
    """BY_HANDLE_FILE_INFORMATION: attributes and size, no security descriptor."""

    _fields_ = [("dwFileAttributes", ctypes.c_uint32),
                ("ftCreationTime", FileTime),
                ("ftLastAccessTime", FileTime),
                ("ftLastWriteTime", FileTime),
                ("dwVolumeSerialNumber", ctypes.c_uint32),
                ("nFileSizeHigh", ctypes.c_uint32),
                ("nFileSizeLow", ctypes.c_uint32),
                ("nNumberOfLinks", ctypes.c_uint32),
                ("nFileIndexHigh", ctypes.c_uint32),
                ("nFileIndexLow", ctypes.c_uint32)]


def nt_create_file(directory, name, desired_access, disposition, options,
                   attributes=0):
    """Open or create ``name`` relative to the directory handle ``directory``.

    ``name`` is one path component; the caller checks that.  ``attributes`` is
    the FileAttribute to give a created file and is ignored for an open.
    Returns the handle.  Raises :class:`WindowsApiError` with the NTSTATUS in
    ``status`` so the caller can translate it.
    """
    create = bind(
        "ntdll", "NtCreateFile", ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD,
        ctypes.POINTER(ObjectAttributes), ctypes.POINTER(IoStatusBlock),
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD)
    encoded = name.encode("utf-16-le")
    text = ctypes.create_string_buffer(encoded + b"\0\0")
    string = UnicodeString(len(encoded), len(encoded) + 2,
                           ctypes.cast(text, ctypes.c_void_p))
    attributes_block = ObjectAttributes(
        ctypes.sizeof(ObjectAttributes), directory, ctypes.pointer(string),
        NtObjectAttributes.OBJ_CASE_INSENSITIVE, None, None)
    io = IoStatusBlock()
    handle = ctypes.c_void_p()
    status = create(ctypes.byref(handle), desired_access,
                    ctypes.byref(attributes_block), ctypes.byref(io),
                    None, attributes, FileShareMode.FILE_SHARE_ALL,
                    disposition, options, None, 0)
    if status < 0:
        raise WindowsApiError(
            f"NtCreateFile({name!r}) failed: 0x{status & 0xffffffff:08x}",
            status=status & 0xffffffff)
    return handle.value


def nt_rename(source, directory, name):
    """Rename the object ``source`` to ``name`` relative to ``directory``.

    ``source`` is an open handle, so the rename acts on the object that was
    opened even if its old name has been replaced meanwhile.
    """
    set_information = bind(
        "ntdll", "NtSetInformationFile", ctypes.c_int32, ctypes.c_void_p,
        ctypes.POINTER(IoStatusBlock), ctypes.c_void_p, wintypes.DWORD,
        ctypes.c_int)
    encoded = name.encode("utf-16-le")
    buffer = ctypes.create_string_buffer(
        ctypes.sizeof(FileRenameInformation) + len(encoded) + 2)
    info = FileRenameInformation.from_buffer(buffer)
    info.ReplaceIfExists = 1
    info.RootDirectory = directory
    info.FileNameLength = len(encoded)
    ctypes.memmove(
        ctypes.addressof(buffer) + FileRenameInformation.FileName.offset,
        encoded, len(encoded))
    io = IoStatusBlock()
    status = set_information(source, ctypes.byref(io), buffer,
                             ctypes.sizeof(buffer), FILE_RENAME_INFORMATION)
    if status < 0:
        raise WindowsApiError(
            "NtSetInformationFile(rename) failed: "
            f"0x{status & 0xffffffff:08x}",
            status=status & 0xffffffff)


def set_delete_disposition(handle) -> None:
    """Mark ``handle``'s object for deletion when the last handle closes."""
    set_information = bind(
        "kernel32", "SetFileInformationByHandle", wintypes.BOOL,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    disposition = ctypes.c_uint8(1)
    if not set_information(handle, FILE_DISPOSITION_INFO,
                           ctypes.byref(disposition), 1):
        raise WindowsApiError("SetFileInformationByHandle(disposition) failed",
                              status=ctypes.get_last_error())


def by_handle_file_information(handle):
    """Return BY_HANDLE_FILE_INFORMATION for an open handle."""
    query = bind("kernel32", "GetFileInformationByHandle", wintypes.BOOL,
                 ctypes.c_void_p, ctypes.POINTER(ByHandleFileInformation))
    information = ByHandleFileInformation()
    if not query(handle, ctypes.byref(information)):
        raise WindowsApiError("GetFileInformationByHandle failed",
                              status=ctypes.get_last_error())
    return information


def read_file(handle, size):
    """Read up to ``size`` bytes; returns what was read (may be short)."""
    read = bind("kernel32", "ReadFile", wintypes.BOOL, ctypes.c_void_p,
                ctypes.c_void_p, wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p)
    buffer = ctypes.create_string_buffer(size)
    count = wintypes.DWORD()
    if not read(handle, ctypes.cast(buffer, ctypes.c_void_p), size,
                ctypes.byref(count), None):
        raise WindowsApiError("ReadFile failed", status=ctypes.get_last_error())
    return buffer.raw[:count.value]


def write_file(handle, data):
    """Write ``data``; returns the number of bytes written."""
    write = bind("kernel32", "WriteFile", wintypes.BOOL, ctypes.c_void_p,
                 ctypes.c_void_p, wintypes.DWORD,
                 ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p)
    payload = bytes(data)
    buffer = ctypes.create_string_buffer(payload)
    count = wintypes.DWORD()
    if not write(handle, ctypes.cast(buffer, ctypes.c_void_p), len(payload),
                 ctypes.byref(count), None):
        raise WindowsApiError("WriteFile failed", status=ctypes.get_last_error())
    return count.value


def flush_file(handle) -> None:
    """Flush a file handle's buffered data to the volume."""
    flush = bind("kernel32", "FlushFileBuffers", wintypes.BOOL,
                 ctypes.c_void_p)
    if not flush(handle):
        raise WindowsApiError("FlushFileBuffers failed",
                              status=ctypes.get_last_error())


# -- range locks ---------------------------------------------------------
# ``LockFileEx`` is the Windows equivalent of ``flock``: an advisory lock on a
# byte range of an open handle.  It locks from the OVERLAPPED offset, so the
# one-byte range at zero has to be stated explicitly.  Contention is reported
# as ERROR_LOCK_VIOLATION, not as a blocking wait.

# LOCKFILE_* (winnt.h): how LockFileEx is to behave while another holds a range.
class LockFlags(enum.IntFlag):
    LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
    LOCKFILE_EXCLUSIVE_LOCK = 0x00000002


ERROR_LOCK_VIOLATION = 33


class Overlapped(ctypes.Structure):
    """OVERLAPPED: the offset a locked range starts at, and an event slot."""

    # Fixed-width ``c_uint32`` for the offset fields: ``wintypes.DWORD`` is
    # ``c_ulong`` (8 bytes on LP64) and would give this the wrong layout.
    _fields_ = [("Internal", ctypes.c_size_t),
                ("InternalHigh", ctypes.c_size_t),
                ("Offset", ctypes.c_uint32),
                ("OffsetHigh", ctypes.c_uint32),
                ("hEvent", ctypes.c_void_p)]


def lock_file(handle, flags) -> None:
    """Lock the one-byte range at offset zero of ``handle``."""
    call = bind("kernel32", "LockFileEx", wintypes.BOOL, ctypes.c_void_p,
                wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                wintypes.DWORD, ctypes.POINTER(Overlapped))
    overlapped = Overlapped()
    if not call(handle, flags, 0, 1, 0, ctypes.byref(overlapped)):
        raise WindowsApiError("LockFileEx failed",
                              status=ctypes.get_last_error())


def unlock_file(handle) -> None:
    """Release the range :func:`lock_file` took."""
    call = bind("kernel32", "UnlockFileEx", wintypes.BOOL, ctypes.c_void_p,
                wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                ctypes.POINTER(Overlapped))
    overlapped = Overlapped()
    if not call(handle, 0, 1, 0, ctypes.byref(overlapped)):
        raise WindowsApiError("UnlockFileEx failed",
                              status=ctypes.get_last_error())


# -- anonymous pipes -----------------------------------------------------
# The Windows credential IPC is two anonymous pipes, not an AF_UNIX socket:
# CPython exposes no AF_UNIX on any interpreter Loki runs, because
# ``Modules/socketmodule.h`` undefines it whenever ``HAVE_SYS_UN_H`` is absent
# and no Windows toolchain -- MSVC or mingw-w64 -- provides ``sys/un.h``.  A
# pipe has no name, so reachability is possession of the inherited handle (the
# authorization ``host_ipc`` already documents) and there is no bind/connect
# race to confirm.
#
# SECURITY_ATTRIBUTES is declared here because ``ctypes.wintypes`` does not
# define it.  ``bInheritHandle`` is a fixed-width ``c_uint32`` rather than
# ``wintypes.BOOL``: the latter is ``c_long``, which is 8 bytes on LP64 hosts,
# and would give the structure the wrong layout in the portable tests that pin
# it -- the same trap the other structures above avoid.

# HANDLE_FLAG_* (winbase.h): what SetHandleInformation may change about a handle.
class HandleFlags(enum.IntFlag):
    HANDLE_FLAG_INHERIT = 0x00000001


class SecurityAttributes(ctypes.Structure):
    """SECURITY_ATTRIBUTES: an optional descriptor plus the inherit flag."""

    _fields_ = [
        ("nLength", ctypes.c_uint32),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", ctypes.c_uint32),
    ]


def create_pipe(inherit: bool = True, size: int = 0):
    """Create an anonymous pipe; return ``(read_handle, write_handle)``.

    ``inherit=True`` asks for both ends to be inheritable, which is what makes
    it possible to name one end in a child's explicit handle list.  The caller
    clears ``HANDLE_FLAG_INHERIT`` on the end it keeps, so only the child's end
    remains eligible to be inherited.  ``size`` of 0 lets the system choose the
    buffer size.
    """
    create = bind("kernel32", "CreatePipe", wintypes.BOOL,
                  ctypes.POINTER(ctypes.c_void_p),
                  ctypes.POINTER(ctypes.c_void_p),
                  ctypes.POINTER(SecurityAttributes), ctypes.c_uint32)
    attributes = SecurityAttributes(
        ctypes.sizeof(SecurityAttributes), None, 1 if inherit else 0)
    read_handle = ctypes.c_void_p()
    write_handle = ctypes.c_void_p()
    if not create(ctypes.byref(read_handle), ctypes.byref(write_handle),
                  ctypes.byref(attributes), size):
        raise WindowsApiError("CreatePipe failed",
                              status=ctypes.get_last_error())
    return read_handle.value, write_handle.value


def set_handle_information(handle, mask: int, flags: int) -> None:
    """Set ``handle``'s flag bits for ``mask``.

    Loki declares it for the inheritance flag alone; the reference defines no
    other callers for this API, so no other mask is passed.
    """
    call = bind("kernel32", "SetHandleInformation", wintypes.BOOL,
                ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD)
    if not call(handle, mask, flags):
        raise WindowsApiError("SetHandleInformation failed",
                              status=ctypes.get_last_error())


def clear_handle_inheritance(handle) -> None:
    """Clear ``HANDLE_FLAG_INHERIT`` so the end cannot leak into a child."""
    set_handle_information(handle, HandleFlags.HANDLE_FLAG_INHERIT, 0)
