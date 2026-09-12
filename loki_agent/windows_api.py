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
    """A Windows call was attempted and returned failure."""


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
