"""Supervisor-owned persistent credential storage.

Credential-consuming runtimes must never import or construct this layer.
Their filesystem view hides the complete credential directory, and they
receive only short-lived access-token leases through an anonymous capability.

The JSON replacement protocol is also part of OAuth correctness, not merely
file hygiene.  OpenAI refresh tokens may rotate.  Before sending one, Loki
durably removes it from the active JSON record and records a refresh attempt.
If the process crashes or delivery becomes ambiguous, a later process sees
that incomplete attempt and requires login instead of replaying the token.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Awaitable, Callable

from . import authentications, file_locks, paths, private_files
from .credential_errors import CredentialStorageError


FORMAT_VERSION = 1
MAX_CREDENTIAL_FILE_BYTES = 1024 * 1024
LOCK_RETRY_DELAY_S = 0.05
LOCK_TIMEOUT_S = 60
OPENAI_RECORD_TYPE = "openai-chatgpt-oauth"
OPENAI_CREDENTIAL_KEY = (
    authentications.CredentialRef.openai_subscription().encode())


@dataclass(frozen=True)
class StoredOpenAICredential:
    state: str
    revision: int
    tokens: authentications.OpenAITokenSet | None = None


def _no_duplicate_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise CredentialStorageError(
                f"credential JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def _integer(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CredentialStorageError(
            f"credential {label} must be a non-negative integer")
    return value


def _optional_number(value, label):
    if value is None:
        return None
    if (not isinstance(value, (int, float))
            or isinstance(value, bool)):
        raise CredentialStorageError(
            f"credential {label} must be numeric or null")
    return float(value)


def _optional_string(value, label):
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CredentialStorageError(
            f"credential {label} must be a non-empty string or null")
    return value


def _required_string(value, label):
    if not isinstance(value, str) or not value:
        raise CredentialStorageError(
            f"credential {label} must be a non-empty string")
    return value


def _empty_document():
    return {
        "version": FORMAT_VERSION,
        "revision": 0,
        "credentials": {},
    }


def _validate_document(value):
    if not isinstance(value, dict):
        raise CredentialStorageError(
            "credential JSON must contain an object")
    if value.get("version") != FORMAT_VERSION:
        raise CredentialStorageError(
            "unsupported credential JSON format version")
    _integer(value.get("revision"), "document revision")
    records = value.get("credentials")
    if not isinstance(records, dict):
        raise CredentialStorageError(
            "credential JSON credentials must be an object")
    for key, record in records.items():
        if not isinstance(key, str) or not key:
            raise CredentialStorageError(
                "credential record names must be non-empty strings")
        if not isinstance(record, dict):
            raise CredentialStorageError(
                f"credential record {key!r} must be an object")
    return value


def _tokens_from_record(record):
    try:
        return authentications.OpenAITokenSet(
            access_token=_required_string(
                record.get("access_token"), "access token"),
            refresh_token=_required_string(
                record.get("refresh_token"), "refresh token"),
            id_token=_optional_string(
                record.get("id_token"), "ID token"),
            account_id=_optional_string(
                record.get("account_id"), "account ID"),
            fedramp=record.get("fedramp", False),
            expires_at=_optional_number(
                record.get("expires_at"), "expiry"),
            last_refresh=_optional_number(
                record.get("last_refresh"), "last refresh"),
        ).normalized()
    except ValueError as error:
        raise CredentialStorageError(str(error)) from error


def _openai_record(document):
    record = document["credentials"].get(OPENAI_CREDENTIAL_KEY)
    if record is None:
        return None
    if record.get("type") != OPENAI_RECORD_TYPE:
        raise CredentialStorageError(
            "OpenAI subscription record has an unsupported type")
    state = record.get("state")
    if state not in {"active", "refreshing", "reauth-required"}:
        raise CredentialStorageError(
            "OpenAI subscription record has an invalid state")
    revision = _integer(
        record.get("revision"), "OpenAI record revision")
    tokens = _tokens_from_record(record) if state == "active" else None
    return StoredOpenAICredential(state, revision, tokens)


def _active_record(tokens, revision):
    tokens = tokens.normalized()
    return {
        "type": OPENAI_RECORD_TYPE,
        "state": "active",
        "revision": revision,
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "id_token": tokens.id_token,
        "account_id": tokens.account_id,
        "fedramp": tokens.fedramp,
        "expires_at": tokens.expires_at,
        "last_refresh": tokens.last_refresh,
    }


def _inactive_record(state, revision, attempt_id=None):
    record = {
        "type": OPENAI_RECORD_TYPE,
        "state": state,
        "revision": revision,
    }
    if attempt_id is not None:
        record["attempt_id"] = attempt_id
    return record


class JsonCredentialStorage:
    """Small, locked, atomically replaced JSON credential database."""

    def __init__(self, directory=None):
        self.directory = (
            paths.credential_directory()
            if directory is None else os.path.abspath(directory))

    @property
    def file_path(self):
        return os.path.join(
            self.directory, paths.CREDENTIAL_FILE_NAME)

    @property
    def lock_path(self):
        return os.path.join(
            self.directory, paths.CREDENTIAL_LOCK_FILE_NAME)

    def ensure_directory(self):
        # Refuse before creating anything: on Windows versions that ignore
        # os.mkdir(mode) the directory would be created with inherited access,
        # leaving credentials readable by other users.
        unsupported = paths.private_directory_support_error()
        if unsupported is not None:
            raise CredentialStorageError(unsupported)
        parent = os.path.dirname(self.directory)
        os.makedirs(parent, exist_ok=True)
        try:
            os.mkdir(self.directory, 0o700)
        except FileExistsError:
            pass
        facts = private_files.describe_path(self.directory)
        if not facts.directory:
            raise CredentialStorageError(
                f"credential path is not a directory: {self.directory}")
        if facts.reparse_point:
            raise CredentialStorageError(
                f"credential directory is a reparse point: {self.directory}")
        if not facts.owned_by_current_user:
            raise CredentialStorageError(
                f"credential directory is not owned by this user: "
                f"{self.directory}")
        if facts.group_or_other_access:
            raise CredentialStorageError(
                f"credential directory has group or other permission bits "
                f"set: {self.directory!r}; adjust its permissions before "
                "retrying (Loki will not change them)")

    def _open_directory(self):
        self.ensure_directory()
        try:
            directory_fd = private_files.open_directory(self.directory)
        except OSError as error:
            raise CredentialStorageError(
                f"could not open credential directory: {error}") from error
        # Checked on the retained handle, not the pathname: between
        # ensure_directory's describe_path and this open the directory could
        # have been swapped for a link, and every relative open traverses a
        # handle to one.
        try:
            facts = private_files.describe(directory_fd)
        except BaseException:
            private_files.close(directory_fd)
            raise
        if not facts.directory:
            private_files.close(directory_fd)
            raise CredentialStorageError(
                f"credential path is not a directory: {self.directory}")
        if facts.reparse_point:
            private_files.close(directory_fd)
            raise CredentialStorageError(
                f"credential directory is a reparse point: {self.directory}")
        return directory_fd

    @staticmethod
    def _validate_secret_file(facts, label):
        if not facts.regular:
            raise CredentialStorageError(
                f"credential {label} is not a regular file")
        if facts.reparse_point:
            raise CredentialStorageError(
                f"credential {label} is a reparse point")
        if not facts.owned_by_current_user:
            raise CredentialStorageError(
                f"credential {label} is not owned by this user")
        if facts.group_or_other_access:
            raise CredentialStorageError(
                f"credential {label} permissions must not grant "
                "group or other access")

    def _read_document_at(self, directory_fd):
        try:
            fd = private_files.open_read_at(
                directory_fd, paths.CREDENTIAL_FILE_NAME)
        except FileNotFoundError:
            return _empty_document()
        except OSError as error:
            raise CredentialStorageError(
                f"could not open credential JSON: {error}") from error
        try:
            facts = private_files.describe(fd)
            self._validate_secret_file(facts, "JSON file")
            if facts.size > MAX_CREDENTIAL_FILE_BYTES:
                raise CredentialStorageError(
                    "credential JSON exceeds its size limit")
            chunks = []
            remaining = MAX_CREDENTIAL_FILE_BYTES + 1
            while remaining:
                chunk = private_files.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > MAX_CREDENTIAL_FILE_BYTES:
                raise CredentialStorageError(
                    "credential JSON exceeds its size limit")
        finally:
            private_files.close(fd)
        try:
            text = data.decode("utf-8")
            value = json.loads(
                text, object_pairs_hook=_no_duplicate_object)
        except (UnicodeDecodeError, ValueError) as error:
            raise CredentialStorageError(
                f"credential JSON is invalid: {error}") from error
        return _validate_document(value)

    def load_document(self):
        directory_fd = self._open_directory()
        try:
            return self._read_document_at(directory_fd)
        finally:
            private_files.close(directory_fd)

    def load_openai_subscription(self):
        return _openai_record(self.load_document())

    def _write_document_at(self, directory_fd, document):
        document = _validate_document(document)
        data = (
            json.dumps(
                document,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ) + "\n"
        ).encode("utf-8")
        if len(data) > MAX_CREDENTIAL_FILE_BYTES:
            raise CredentialStorageError(
                "credential JSON exceeds its size limit")
        temporary_name = (
            f".{paths.CREDENTIAL_FILE_NAME}."
            f"{os.getpid()}.{secrets.token_hex(12)}")
        fd = None
        try:
            fd = private_files.create_exclusive_at(
                directory_fd, temporary_name, 0o600)
            # Describe the file this call just created, before a byte is
            # written: it takes its access from the directory's DACL, and a
            # directory widened between ensure_directory's check and this create
            # would otherwise hand the file out.
            facts = private_files.describe(fd)
            if not (facts.regular and facts.owned_by_current_user
                    and not facts.reparse_point
                    and not facts.group_or_other_access):
                raise CredentialStorageError(
                    f"new credential file is not private: {temporary_name}")
            view = memoryview(data)
            while view:
                written = private_files.write(fd, view)
                if written <= 0:
                    raise OSError("short credential JSON write")
                view = view[written:]
            private_files.fsync(fd)
            private_files.close(fd)
            fd = None
            private_files.replace_at(
                directory_fd, temporary_name, paths.CREDENTIAL_FILE_NAME)
            private_files.fsync(directory_fd)
        except OSError as error:
            raise CredentialStorageError(
                f"could not persist credential JSON: {error}") from error
        finally:
            if fd is not None:
                private_files.close(fd)
            with contextlib.suppress(
                    FileNotFoundError, CredentialStorageError):
                private_files.unlink_at(directory_fd, temporary_name)

    def _open_lock_at(self, directory_fd):
        try:
            fd = private_files.open_lock_file_at(
                directory_fd, paths.CREDENTIAL_LOCK_FILE_NAME, 0o600)
        except OSError as error:
            raise CredentialStorageError(
                f"could not open credential lock: {error}") from error
        try:
            self._validate_secret_file(
                private_files.describe(fd), "lock file")
        except BaseException:
            private_files.close(fd)
            raise
        return fd

    @asynccontextmanager
    async def _locked_document(self):
        directory_fd = self._open_directory()
        try:
            lock_fd = self._open_lock_at(directory_fd)
            try:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + LOCK_TIMEOUT_S
                acquired = False
                try:
                    while True:
                        try:
                            file_locks.try_lock_exclusive(lock_fd)
                            acquired = True
                            break
                        except BlockingIOError:
                            if loop.time() >= deadline:
                                raise CredentialStorageError(
                                    "timed out waiting for credential lock")
                            await asyncio.sleep(LOCK_RETRY_DELAY_S)
                    yield directory_fd, self._read_document_at(directory_fd)
                finally:
                    if acquired:
                        with contextlib.suppress(OSError):
                            file_locks.unlock(lock_fd)
            finally:
                private_files.close(lock_fd)
        finally:
            private_files.close(directory_fd)

    @staticmethod
    def _next_revision(document):
        revision = _integer(
            document["revision"], "document revision") + 1
        document["revision"] = revision
        return revision

    async def store_openai_login(self, tokens):
        normalized = tokens.normalized()
        async with self._locked_document() as (
                directory_fd, document):
            revision = self._next_revision(document)
            document["credentials"][OPENAI_CREDENTIAL_KEY] = (
                _active_record(normalized, revision))
            self._write_document_at(directory_fd, document)
        return StoredOpenAICredential(
            "active", revision, normalized)

    async def remove_openai_subscription(self):
        async with self._locked_document() as (
                directory_fd, document):
            existed = (
                OPENAI_CREDENTIAL_KEY in document["credentials"])
            if existed:
                self._next_revision(document)
                del document["credentials"][OPENAI_CREDENTIAL_KEY]
                self._write_document_at(directory_fd, document)
            return existed

    async def rotate_openai_subscription(
            self,
            current: authentications.OpenAITokenSet,
            refresh: Callable[
                [str],
                Awaitable[authentications.RefreshResult],
            ] = authentications.request_openai_token_refresh,
            clock=None):
        """Refresh once under a cross-process, crash-safe transaction."""
        async with self._locked_document() as (
                directory_fd, document):
            now = (clock or time.time)()
            stored = _openai_record(document)
            if stored is None:
                raise authentications.RefreshPermanentError(
                    "OpenAI subscription was logged out")
            if stored.state != "active" or stored.tokens is None:
                raise authentications.RefreshPermanentError(
                    "an interrupted OpenAI refresh requires login")
            if stored.tokens != current:
                # Another supervisor completed a login or refresh first.
                # Its durable state wins without using our stale token.
                return stored.tokens

            attempt_id = secrets.token_urlsafe(24)
            refreshing_revision = self._next_revision(document)
            document["credentials"][OPENAI_CREDENTIAL_KEY] = (
                _inactive_record(
                    "refreshing",
                    refreshing_revision,
                    attempt_id,
                ))
            self._write_document_at(directory_fd, document)

            try:
                result = await refresh(current.refresh_token)
                refreshed = authentications.refreshed_openai_tokens(
                    current, result, now)
            except asyncio.CancelledError:
                self._record_reauth_required(
                    directory_fd, document)
                raise
            except authentications.RefreshTransientError as error:
                if not error.request_may_have_been_sent:
                    self._restore_after_unsent_failure(
                        directory_fd, document, current)
                else:
                    self._record_reauth_required(
                        directory_fd, document)
                raise
            except authentications.RefreshPermanentError:
                self._record_reauth_required(
                    directory_fd, document)
                raise
            except BaseException as error:
                self._record_reauth_required(
                    directory_fd, document)
                raise authentications.RefreshIndeterminateError(
                    "OpenAI refresh failed after its durable attempt "
                    "record was created") from error

            try:
                revision = self._next_revision(document)
                document["credentials"][OPENAI_CREDENTIAL_KEY] = (
                    _active_record(refreshed, revision))
                self._write_document_at(directory_fd, document)
            except CredentialStorageError as error:
                # The already-durable refreshing record omits the old refresh
                # token. Even if the final write fails, another process will
                # fail closed instead of replaying it.
                raise authentications.RefreshIndeterminateError(
                    "OpenAI returned refreshed tokens but Loki could not "
                    "persist them; login is required") from error
            return refreshed

    def _restore_after_unsent_failure(
            self, directory_fd, document, tokens):
        try:
            revision = self._next_revision(document)
            document["credentials"][OPENAI_CREDENTIAL_KEY] = (
                _active_record(tokens, revision))
            self._write_document_at(directory_fd, document)
        except CredentialStorageError as error:
            raise authentications.RefreshIndeterminateError(
                "OpenAI refresh was not sent, but Loki could not restore "
                "the active credential record") from error

    def _record_reauth_required(self, directory_fd, document):
        try:
            revision = self._next_revision(document)
            document["credentials"][OPENAI_CREDENTIAL_KEY] = (
                _inactive_record("reauth-required", revision))
            self._write_document_at(directory_fd, document)
        except CredentialStorageError:
            # The previously fsynced refreshing record is already fail-closed.
            pass
