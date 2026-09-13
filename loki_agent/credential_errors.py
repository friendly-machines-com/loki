"""The error the credential storage raises, below the platform selection.

It lives here so a platform primitive and the storage can raise the same error
without either importing the other.
"""

from __future__ import annotations


class CredentialStorageError(RuntimeError):
    pass
