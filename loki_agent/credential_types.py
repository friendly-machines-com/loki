"""Types shared by the credential storage and its per-platform primitives.

``FileFacts`` and ``CredentialStorageError`` live here, below both
``credential_files`` and the platform modules, so the platform modules do not
import the module that imports them.  ``credential_files`` selects the platform
set at import time, so a platform module importing ``credential_files`` back is
a cycle: on Windows it resolved only when ``credential_files`` happened to be
imported first, and raised ``ImportError`` when a module imported the platform
module directly.

The description is a :class:`FileFacts` record rather than a ``stat_result``:
a POSIX descriptor carries mode bits and a uid, while a Windows handle carries
attributes, an owner SID and a DACL, so the store asks for facts, not for the
platform's shape of them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileFacts:
    """What the store needs to know about an object, per platform."""

    regular: bool
    directory: bool
    reparse_point: bool
    size: int
    owned_by_current_user: bool
    group_or_other_access: bool


class CredentialStorageError(RuntimeError):
    pass
