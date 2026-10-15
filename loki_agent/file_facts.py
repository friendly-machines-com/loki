"""What a caller needs to know about an open object, per platform.

A POSIX descriptor carries mode bits and a uid; a Windows handle carries
attributes, an owner SID and a DACL.  The platform file primitives describe
either as this record, so callers ask for facts rather than for the platform's
shape of them.  It lives here, below the primitives and their callers, so the
primitives do not import the module that selects them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileInfo:
    """Handle-bound type and size, without interpreting credential privacy."""

    regular: bool
    directory: bool
    reparse_point: bool
    size: int


@dataclass(frozen=True)
class FileFacts(FileInfo):
    """File information plus the strict credential-store privacy checks."""

    owned_by_current_user: bool
    group_or_other_access: bool
