"""Windows credential-file primitives.

Not implemented yet.  The POSIX bodies refuse a symlink in the final component
and act relative to the open directory descriptor; the Windows equivalents are
a retained directory handle with ``NtCreateFile(RootDirectory)`` and
``FILE_FLAG_OPEN_REPARSE_POINT``, with the owner taken from the token's user
SID via ``GetSecurityInfo``.  Until those exist, every entry point raises here
so a Windows caller gets one clear error instead of a POSIX ``AttributeError``
from whichever call site happened to run first.
"""

from __future__ import annotations

from .credential_files import CredentialStorageError


_NOT_IMPLEMENTED = (
    "Windows credential-file access is not implemented yet; Loki cannot read "
    "or write credentials on this platform")


def _unsupported(*args, **kwargs):
    raise CredentialStorageError(_NOT_IMPLEMENTED)


open_directory = _unsupported
open_read_at = _unsupported
create_exclusive_at = _unsupported
open_lock_file_at = _unsupported
replace_at = _unsupported
unlink_at = _unsupported
owner_is_current_user = _unsupported
