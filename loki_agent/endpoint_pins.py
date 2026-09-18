"""Approved catalog endpoints, recorded when a user confirms one.

A provider's ``api`` field in the downloaded models.dev catalog decides where
that provider's static credential is sent, and the catalog is untrusted input.
The endpoint is therefore not configuration to be believed: it is approved
once, explicitly, and any later change is shown and decided on rather than
silently accepted.  The declared credential is part of the approval too, so a
mutated catalog cannot keep an approved endpoint and swap which stored secret
is sent to it.

This record is deliberately separate from the models.dev catalog cache: that
cache is overwritten by every fetch, while an approval has to survive it.

Callers consult :func:`status` before attaching a credential:

* ``pinned``  - matches what was approved; proceed.
* ``new``     - never approved; require confirmation that shows the endpoint.
* ``changed`` - differs from the approved value; require confirmation that
  shows both.

The store is a small JSON document in Loki's state directory, written
atomically with owner-only permissions.  A missing or unreadable store means
"nothing approved yet" rather than an error.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets

from . import paths, private_files


PINNED = "pinned"
NEW = "new"
CHANGED = "changed"

_FILE_NAME = "provider-endpoints.json"
_MAX_BYTES = 1024 * 1024


def _path() -> str:
    return os.path.join(paths.loki_state_dir(), _FILE_NAME)


def _entry(api_url, credential):
    if (not isinstance(api_url, str) or not api_url
            or not isinstance(credential, str) or not credential):
        return None
    return {"api": api_url, "credential": credential}


def load() -> dict:
    """Return the approved mapping; unreadable or invalid means empty."""
    try:
        with open(_path(), "rb") as stream:
            data = stream.read(_MAX_BYTES + 1)
    except OSError:
        return {}
    if len(data) > _MAX_BYTES:
        return {}
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    approved = {}
    for name, entry in value.items():
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(entry, dict):
            continue
        normalized = _entry(entry.get("api"), entry.get("credential"))
        if normalized is not None:
            approved[name] = normalized
    return approved


def status(provider_id: str, api_url: str, credential: str):
    """Return ``(state, approved_entry)`` for one provider's endpoint+secret."""
    requested = _entry(api_url, credential)
    if requested is None:
        raise ValueError("endpoint and credential are required")
    approved = load().get(provider_id)
    if approved is None:
        return NEW, None
    if approved == requested:
        return PINNED, approved
    return CHANGED, approved


def record(provider_id: str, api_url: str, credential: str) -> None:
    """Approve ``api_url`` with ``credential`` for ``provider_id``, atomically."""
    if not isinstance(provider_id, str) or not provider_id:
        raise ValueError("provider id is required")
    entry = _entry(api_url, credential)
    if entry is None:
        raise ValueError("endpoint and credential are required")
    directory = os.path.dirname(_path())
    os.makedirs(directory, mode=0o700, exist_ok=True)
    directory_fd = private_files.open_directory(directory)
    temporary_name = None
    try:
        entries = load()
        entries[provider_id] = entry
        data = (
            json.dumps(entries, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        temporary_name = (
            f".{_FILE_NAME}.{os.getpid()}.{secrets.token_hex(12)}")
        fd = private_files.create_exclusive_at(
            directory_fd, temporary_name, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = private_files.write(fd, view)
                if written <= 0:
                    raise OSError("short provider endpoint pin write")
                view = view[written:]
            private_files.fsync(fd)
        finally:
            private_files.close(fd)
        private_files.replace_at(directory_fd, temporary_name, _FILE_NAME)
        private_files.fsync(directory_fd)
        temporary_name = None
    finally:
        if temporary_name is not None:
            with contextlib.suppress(OSError):
                private_files.unlink_at(directory_fd, temporary_name)
        private_files.close(directory_fd)
