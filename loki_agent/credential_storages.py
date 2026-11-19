"""Supervisor-owned persistent credential storage.

Credential-consuming runtimes must never import or construct this layer.
Their filesystem view hides the complete credential directory, and they
receive only short-lived access-token leases through an anonymous capability.

The JSON replacement protocol is also part of OAuth correctness, not merely
file hygiene. OAuth refresh tokens may rotate. Before sending one, Loki
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
from dataclasses import dataclass, replace
from typing import Awaitable, Callable

from . import authentications, file_locks, paths, private_files
from .credential_errors import CredentialStorageError


FORMAT_VERSION = 1
MAX_CREDENTIAL_FILE_BYTES = 1024 * 1024
LOCK_RETRY_DELAY_S = 0.05
OPENAI_RECORD_TYPE = "openai-chatgpt-oauth"
OPENAI_CREDENTIAL_KEY = (
    authentications.CredentialRef.openai_subscription().encode())
ANTHROPIC_RECORD_TYPE = "anthropic-subscription-oauth"
ANTHROPIC_REFRESH_LOCK = "anthropic-refresh.lock"
ANTHROPIC_CREDENTIAL_KEY = (
    authentications.CredentialRef.anthropic_subscription().encode())


@dataclass(frozen=True)
class StoredOpenAICredential:
    state: str
    revision: int
    tokens: authentications.OpenAITokenSet | None = None


@dataclass(frozen=True)
class StoredAnthropicCredential:
    """The supervisor and auth status read this provider's durable grant.

    The OpenAI record stores different token fields. Both need a revision
    because a refresh must not replace a later login or undo a logout.
    """

    state: str
    revision: int
    tokens: authentications.AnthropicTokenSet | None = None


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
    if (not isinstance(value.get("version"), int)
            or isinstance(value.get("version"), bool)
            or value.get("version") != FORMAT_VERSION):
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
    approvals = value.get("endpoint_approvals", {})
    if not isinstance(approvals, dict):
        raise CredentialStorageError("endpoint approvals must be an object")
    for encoded, destinations in approvals.items():
        try:
            authentications.CredentialRef.decode(encoded)
            if not isinstance(destinations, list):
                raise ValueError("destinations must be a list")
            for destination in destinations:
                authentications.validate_destination_url(destination)
            if len(set(destinations)) != len(destinations):
                raise ValueError("duplicate destinations")
        except (ValueError, authentications.CredentialError) as error:
            raise CredentialStorageError("endpoint approvals are invalid") from error
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


def _anthropic_tokens_from_record(record):
    try:
        account = None
        uuid = _optional_string(
            record.get("account_uuid"), "Anthropic account UUID")
        if uuid is not None:
            account = authentications.AnthropicAccount(
                uuid=uuid,
                email_address=_optional_string(
                    record.get("account_email"), "Anthropic email address"),
                organization_uuid=_optional_string(
                    record.get("organization_uuid"),
                    "Anthropic organization UUID"),
            )
        scopes = record.get("scopes")
        if not isinstance(scopes, list) or any(not isinstance(scope, str) for scope in scopes):
            raise CredentialStorageError("Anthropic subscription record has invalid scopes")
        last_refresh = record.get("last_refresh")
        if last_refresh is None:
            # Refresh age and token equality must survive reopening the store.
            raise CredentialStorageError("Anthropic subscription record has no refresh timestamp")
        return authentications.AnthropicTokenSet(
            access_token=record.get("access_token"),
            refresh_token=record.get("refresh_token"),
            client_id=record.get("client_id"),
            scopes=frozenset(scopes),
            account=account,
            expires_at=record.get("expires_at"),
            last_refresh=last_refresh,
            grant_id=_required_string(record.get("grant_id"), "Anthropic grant ID"),
        ).normalized()
    except ValueError as error:
        raise CredentialStorageError(str(error)) from error


def _anthropic_record(document):
    record = document["credentials"].get(ANTHROPIC_CREDENTIAL_KEY)
    if record is None:
        return None
    if record.get("type") != ANTHROPIC_RECORD_TYPE:
        raise CredentialStorageError(
            "Anthropic subscription record has an unsupported type")
    state = record.get("state")
    if state not in ["active", "refreshing", "reauth-required"]:
        raise CredentialStorageError("Anthropic subscription record has an invalid state")
    revision = _integer(
        record.get("revision"), "Anthropic record revision")
    tokens = (
        _anthropic_tokens_from_record(record) if state == "active" else None)
    return StoredAnthropicCredential(state, revision, tokens)


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


def _active_anthropic_record(tokens, revision):
    tokens = tokens.normalized()
    return {
        "type": ANTHROPIC_RECORD_TYPE,
        "state": "active",
        "revision": revision,
        "grant_id": tokens.grant_id,
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "client_id": tokens.client_id,
        "scopes": sorted(tokens.scopes),
        "account_uuid": (
            tokens.account.uuid if tokens.account is not None else None),
        "account_email": (
            tokens.account.email_address
            if tokens.account is not None else None),
        "organization_uuid": (
            tokens.account.organization_uuid
            if tokens.account is not None else None),
        "expires_at": tokens.expires_at,
        "last_refresh": tokens.last_refresh,
    }


def _inactive_anthropic_record(state, revision):
    return {"type": ANTHROPIC_RECORD_TYPE, "state": state, "revision": revision}


class JsonCredentialStorage:
    """Small, locked, atomically replaced JSON credential database."""

    def __init__(self, directory=None):
        self.directory = (
            paths.credential_directory() if directory is None else directory)
        self._directory_fd = None
        self._directory_fd = self._acquire_directory()

    def close(self):
        if self._directory_fd is not None:
            fd, self._directory_fd = self._directory_fd, None
            private_files.close(fd)

    def __del__(self):
        if getattr(self, '_directory_fd', None) is not None:
            fd, self._directory_fd = self._directory_fd, None
            with contextlib.suppress(OSError):
                private_files.close(fd)

    @property
    def directory_handle(self):
        """Borrow the retained object for trusted runtime startup.

        The store keeps ownership. A spawned process receives its own inherited
        descriptor copy and must close that copy before running model code.
        """
        if self._directory_fd is None:
            raise CredentialStorageError("credential storage is closed")
        return self._directory_fd

    @property
    def file_path(self):
        return os.path.join(
            self.directory, paths.CREDENTIAL_FILE_NAME)

    @property
    def lock_path(self):
        return os.path.join(
            self.directory, paths.CREDENTIAL_LOCK_FILE_NAME)

    def ensure_directory(self):
        """Recheck the held directory's privacy without reopening its name."""
        if self._directory_fd is None:
            raise CredentialStorageError("credential storage is closed")
        self._validate_directory(private_files.describe(self._directory_fd))

    def _validate_directory(self, facts):
        if not facts.directory:
            raise CredentialStorageError(
                f"credential path is not a directory: {self.directory}")
        if facts.reparse_point:
            raise CredentialStorageError(
                f"credential directory is a reparse point: {self.directory}")
        if not facts.owned_by_current_user:
            raise CredentialStorageError(
                f"credential directory is not owned by this user: {self.directory}")
        if facts.group_or_other_access:
            raise CredentialStorageError(
                f"credential directory has group or other permission bits "
                f"set: {self.directory!r}; adjust its permissions before "
                "retrying (Loki will not change them)")

    def _acquire_directory(self):
        # Check the declared private-filesystem support floor before creation.
        # Only construction resolves this configured pathname.
        unsupported = paths.private_directory_support_error()
        if unsupported is not None:
            raise CredentialStorageError(unsupported)
        try:
            try:
                private_files.make_private_directory(self.directory)
            except FileExistsError:
                # The open below must distinguish a non-directory or reparse
                # entry from an existing private directory, on that object.
                pass
            directory_fd = private_files.open_directory(self.directory)
        except NotADirectoryError as error:
            raise CredentialStorageError(
                f"credential path is not a directory: {self.directory}") from error
        except OSError as error:
            raise CredentialStorageError(
                f"could not open credential directory: {error}") from error
        try:
            self._validate_directory(private_files.describe(directory_fd))
        except BaseException:
            private_files.close(directory_fd)
            raise
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
        except (UnicodeDecodeError, ValueError, RecursionError) as error:
            raise CredentialStorageError("credential JSON is invalid") from error
        return _validate_document(value)

    def load_document(self):
        self.ensure_directory()
        return self._read_document_at(self._directory_fd)

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
        candidate = (
            f".{paths.CREDENTIAL_FILE_NAME}."
            f"{os.getpid()}.{secrets.token_hex(12)}")
        temporary_name = None
        fd = None
        try:
            fd = private_files.create_exclusive_at(directory_fd, candidate, 0o600, user_owned=True)
            # A collision is another publication's entry, not our cleanup work.
            temporary_name = candidate
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
            private_files.write_all(fd, data)
            private_files.fsync(fd)
            # Close once before Windows replacement; a failed close must not
            # cause cleanup to retry a potentially released/reused descriptor.
            closing, fd = fd, None
            private_files.close(closing)
            private_files.replace_at(
                directory_fd, temporary_name, paths.CREDENTIAL_FILE_NAME)
            temporary_name = None
            private_files.fsync(directory_fd)
        except OSError as error:
            raise CredentialStorageError(
                f"could not persist credential JSON: {error}") from error
        finally:
            try:
                if fd is not None:
                    private_files.close(fd)
            finally:
                if temporary_name is not None:
                    with contextlib.suppress(FileNotFoundError, CredentialStorageError):
                        private_files.unlink_at(directory_fd, temporary_name)

    def _open_lock_at(self, directory_fd, filename=paths.CREDENTIAL_LOCK_FILE_NAME):
        try:
            fd = private_files.open_lock_file_at(
                directory_fd, filename, 0o600, user_owned=True)
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
        async with self._locked_file(paths.CREDENTIAL_LOCK_FILE_NAME) as directory_fd:
            yield directory_fd, self._read_document_at(directory_fd)

    @asynccontextmanager
    async def _locked_file(self, filename, *, wait_while=None):
        self.ensure_directory()
        directory_fd = self._directory_fd
        lock_fd = self._open_lock_at(directory_fd, filename)
        try:
            acquired = False
            try:
                while True:
                    try:
                        file_locks.try_lock_exclusive(lock_fd)
                        acquired = True
                        break
                    except BlockingIOError:
                        # Only the read-only startup waiter can stop early:
                        # logout or replacement may settle its read while the
                        # refresh owner is still waiting for the HTTP response.
                        if wait_while is not None and not wait_while():
                            break
                        await asyncio.sleep(LOCK_RETRY_DELAY_S)
                    except OSError as error:
                        raise CredentialStorageError(f"could not lock credential file: {error}") from error
                yield directory_fd if acquired else None
            finally:
                if acquired:
                    with contextlib.suppress(OSError):
                        file_locks.unlock(lock_fd)
        finally:
            private_files.close(lock_fd)

    @staticmethod
    def _next_revision(document):
        revision = _integer(
            document["revision"], "document revision") + 1
        document["revision"] = revision
        return revision

    def approved_destinations(self, credential):
        """Fresh permissions only; the protected document is not cached.

        The existing version-1 credential format permits extra top-level
        fields. Documents predating this section mean no approvals. Token
        updates preserve this section, and approval updates preserve tokens.
        """
        document = self.load_document()
        return frozenset(document.get("endpoint_approvals", {}).get(
            credential.encode(), []))

    async def approve_destinations(
            self, credential, destinations, *, before_commit=None):
        for destination in destinations:
            authentications.validate_destination_url(destination)
        async with self._locked_document() as [directory_fd, document]:
            approvals = document.setdefault("endpoint_approvals", {})
            existing = set(approvals.get(credential.encode(), []))
            combined = existing | set(destinations)
            if before_commit is not None:
                before_commit()
            # The final lifetime check and bounded file publication do not
            # yield. Waiting for the shared lock above is async/cancellable.
            if combined != existing:
                approvals[credential.encode()] = sorted(combined)
                self._next_revision(document)
                self._write_document_at(directory_fd, document)

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

    def load_anthropic_subscription(self, *, strict=True):
        document = self.load_document()
        try:
            return _anthropic_record(document)
        except CredentialStorageError:
            # A bad provider record need not disable other credentials. File
            # and document integrity checks above still fail closed.
            if strict:
                raise
            return None

    async def wait_for_anthropic_refresh(self):
        """Settle a refreshing record before the startup broker takes ownership.

        The plain loader also serves status and cannot wait. Only the refresh
        owner's lock distinguishes a live exchange from an orphaned marker.
        Startup borrows that lock to re-read the store, never to send a token
        or recover an orphan. Cancellation just closes the startup lock handle.
        """
        stored = None

        def still_refreshing():
            nonlocal stored
            stored = self.load_anthropic_subscription(strict=False)
            return stored is not None and stored.state == "refreshing"

        async with self._locked_file(ANTHROPIC_REFRESH_LOCK, wait_while=still_refreshing) as directory_fd:
            if directory_fd is not None:
                # A live owner may have committed while we waited. A marker
                # still present under the lock has no live owner; leave it be.
                stored = self.load_anthropic_subscription(strict=False)
        return stored

    async def store_anthropic_login(self, tokens):
        normalized = replace(tokens, grant_id=secrets.token_urlsafe(24)).normalized()
        async with self._locked_document() as (directory_fd, document):
            revision = self._next_revision(document)
            document["credentials"][ANTHROPIC_CREDENTIAL_KEY] = _active_anthropic_record(normalized, revision)
            self._write_document_at(directory_fd, document)
        return StoredAnthropicCredential("active", revision, normalized)

    async def remove_anthropic_subscription(self):
        async with self._locked_document() as (directory_fd, document):
            if ANTHROPIC_CREDENTIAL_KEY not in document["credentials"]:
                return False
            self._next_revision(document)
            del document["credentials"][ANTHROPIC_CREDENTIAL_KEY]
            self._write_document_at(directory_fd, document)
            return True

    async def rotate_anthropic_subscription(
            self, current: authentications.AnthropicTokenSet,
            refresh: Callable[[authentications.AnthropicTokenSet],
                              Awaitable[authentications.AnthropicRefreshResult]] = (
                                  authentications.request_anthropic_token_refresh),
            clock=None):
        """Rotate the durable grant without holding the document lock over HTTP."""
        attempt = None
        try:
            # Serializing just this credential prevents replay without
            # blocking unrelated writes while the endpoint responds.
            async with self._locked_file(ANTHROPIC_REFRESH_LOCK):
                async with self._locked_document() as (directory_fd, document):
                    stored = _anthropic_record(document)
                    if stored is None:
                        raise authentications.RefreshPermanentError("Anthropic subscription was logged out")
                    if stored.state != "active" or stored.tokens is None:
                        raise authentications.RefreshPermanentError("An interrupted Anthropic refresh requires login")
                    if stored.tokens.grant_id != current.grant_id:
                        raise authentications.RefreshPermanentError("Anthropic login changed; restart Loki")
                    if stored.tokens != current:
                        # Another supervisor already rotated the same grant.
                        return stored.tokens
                    # Fsync a token-free record before delivery. A crash after
                    # consuming the old token cannot make it replayable.
                    attempt = self._next_revision(document)
                    document["credentials"][ANTHROPIC_CREDENTIAL_KEY] = _inactive_anthropic_record(
                        "refreshing", attempt)
                    self._write_document_at(directory_fd, document)

                refreshed = await self._exchange_anthropic_refresh(attempt, current, refresh, clock or time.time)
                await self._finish_anthropic_attempt(attempt, refreshed)
                return refreshed
        except asyncio.CancelledError as error:
            if attempt is None:
                error.request_may_have_been_sent = False
            raise
        except (CredentialStorageError, OSError) as error:
            if attempt is None:
                raise authentications.RefreshTransientError("Could not prepare Anthropic token refresh") from error
            # A failed final write must not resurrect the old refresh token.
            raise authentications.RefreshIndeterminateError(
                "Could not finalize Anthropic token refresh; log in again") from error

    async def _exchange_anthropic_refresh(self, attempt, current, refresh, clock):
        try:
            result = await refresh(current)
            return authentications.refreshed_anthropic_tokens(current, result, clock())
        except asyncio.CancelledError as error:
            unsent = not getattr(error, "request_may_have_been_sent", True)
            try:
                await self._finish_anthropic_attempt(attempt, current if unsent else None)
            except (CredentialStorageError, OSError, authentications.RefreshPermanentError):
                # The broker may reuse the old grant only when we could
                # durably restore it as well as proving it was not sent.
                error.request_may_have_been_sent = True
            raise
        except authentications.RefreshTransientError as error:
            if not error.request_may_have_been_sent:
                await self._finish_anthropic_attempt(attempt, current)
            else:
                await self._invalidate_anthropic_attempt(attempt)
            raise
        except authentications.RefreshPermanentError:
            await self._invalidate_anthropic_attempt(attempt)
            raise
        except Exception as error:
            await self._invalidate_anthropic_attempt(attempt)
            raise authentications.RefreshIndeterminateError(
                "Anthropic refresh did not complete; log in again") from error

    async def _invalidate_anthropic_attempt(self, attempt):
        with contextlib.suppress(CredentialStorageError, OSError, authentications.RefreshPermanentError):
            # The durable refreshing marker already fails closed if we
            # cannot record the final failure or a new login has replaced it.
            await self._finish_anthropic_attempt(attempt, None)

    async def _finish_anthropic_attempt(self, attempt, tokens):
        async with self._locked_document() as (directory_fd, document):
            stored = _anthropic_record(document)
            if stored is None or stored.revision != attempt or stored.state != "refreshing":
                # Login or logout was allowed to proceed during the exchange.
                # The late response cannot restore an obsolete grant.
                raise authentications.RefreshPermanentError("Anthropic login changed; restart Loki")
            revision = self._next_revision(document)
            document["credentials"][ANTHROPIC_CREDENTIAL_KEY] = (
                _inactive_anthropic_record("reauth-required", revision)
                if tokens is None else _active_anthropic_record(tokens, revision)
            )
            self._write_document_at(directory_fd, document)
