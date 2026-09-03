"""Request-time credentials, refresh serialization, and authorization policy.

Only the top-level Loki process owns long-lived credentials.  Workers and
subagents receive a :class:`CredentialAuthority` capability which can issue
the request credential they are allowed to use, but never a refresh token.

The generation carried by every lease is part of the correctness protocol:
when several requests reject generation N, only the first rejection may
refresh it.  Later rejections observe generation N+1 instead of replaying a
possibly single-use refresh token.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

from . import http_client


OPENAI_REFRESH_URL = "https://auth.openai.com/oauth/token"
# OAuth client identifiers are public protocol identifiers, not client
# secrets.  This is the native Codex client ID accepted by OpenAI's ChatGPT
# Codex authorization flow; refresh tokens remain the actual secret.
OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_ORIGINATOR = "loki"
OPENAI_ACCESS_TOKEN_REFRESH_WINDOW_S = 5 * 60
OPENAI_FALLBACK_REFRESH_INTERVAL_S = 8 * 24 * 60 * 60
OPENAI_REFRESH_TIMEOUT_S = 30
OPENAI_REFRESH_MAX_BYTES = 256 * 1024
OPENAI_CHATGPT_RESPONSES_URL = (
    "https://chatgpt.com/backend-api/codex/responses")
OPENAI_CHATGPT_MODELS_URL = (
    "https://chatgpt.com/backend-api/codex/models")
# The private endpoint interprets client_version in Codex's compatibility
# namespace: sending Loki's package version returns an empty catalog. Pin the
# newest Codex request contract Loki actually implements. Codex 0.144.0 added
# Responses-Lite/code-mode models, whose different tool framing Loki does not
# implement; 0.124.0 exposes only the regular stateless Responses contract.
# Because the complete URL is authorization-allowlisted below, advancing this
# boundary requires an explicit protocol review.
OPENAI_CHATGPT_CODEX_COMPATIBILITY_VERSION = "0.124.0"
OPENAI_CHATGPT_MODELS_REQUEST_URL = (
    f"{OPENAI_CHATGPT_MODELS_URL}?client_version="
    f"{OPENAI_CHATGPT_CODEX_COMPATIBILITY_VERSION}")
OPENAI_CHATGPT_AUTHORIZED_URLS = frozenset({
    OPENAI_CHATGPT_RESPONSES_URL,
    OPENAI_CHATGPT_MODELS_REQUEST_URL,
})


@dataclass(frozen=True, order=True)
class CredentialRef:
    """Stable, non-secret identity of one brokered credential."""

    kind: str
    name: str

    def __post_init__(self):
        if not self.kind or not self.name:
            raise ValueError("credential reference requires kind and name")
        if ":" in self.kind:
            raise ValueError("credential reference kind cannot contain ':'")

    @classmethod
    def environment(cls, name: str) -> "CredentialRef":
        return cls("env", name)

    @classmethod
    def openai_subscription(cls) -> "CredentialRef":
        return cls("openai-subscription", "openai")

    def encode(self) -> str:
        return f"{self.kind}:{self.name}"

    @classmethod
    def decode(cls, value: str) -> "CredentialRef":
        if not isinstance(value, str):
            raise ValueError("credential reference must be a string")
        kind, separator, name = value.partition(":")
        if not separator:
            raise ValueError("credential reference must contain ':'")
        return cls(kind, name)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name}

    @classmethod
    def from_dict(cls, value) -> "CredentialRef":
        if not isinstance(value, dict):
            raise ValueError("credential reference must be an object")
        kind = value.get("kind")
        name = value.get("name")
        if not isinstance(kind, str) or not isinstance(name, str):
            raise ValueError(
                "credential reference kind and name must be strings")
        return cls(kind, name)


@dataclass(frozen=True)
class AuthSpec:
    """Non-secret rule for turning a credential lease into HTTP headers."""

    credential: CredentialRef | None
    scheme: str = "bearer"
    header_name: str | None = None


@dataclass(frozen=True)
class CredentialLease:
    """A short-lived snapshot used by one HTTP attempt."""

    credential: CredentialRef
    value: str = field(repr=False)
    generation: int = 0
    expires_at: float | None = None
    refreshable: bool = False
    account_id: str | None = None
    fedramp: bool = False

    def to_wire(self) -> dict:
        return {
            "credential": self.credential.encode(),
            "value": self.value,
            "generation": self.generation,
            "expires_at": self.expires_at,
            "refreshable": self.refreshable,
            "account_id": self.account_id,
            "fedramp": self.fedramp,
        }

    @classmethod
    def from_wire(cls, value) -> "CredentialLease":
        if not isinstance(value, dict):
            raise ValueError("credential lease must be an object")
        credential = CredentialRef.decode(value.get("credential"))
        secret = value.get("value")
        generation = value.get("generation")
        expires_at = value.get("expires_at")
        refreshable = value.get("refreshable")
        account_id = value.get("account_id")
        fedramp = value.get("fedramp")
        if not isinstance(secret, str):
            raise ValueError("credential lease value must be a string")
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise ValueError(
                "credential lease generation must be an integer")
        if (expires_at is not None
                and not isinstance(expires_at, (int, float))):
            raise ValueError(
                "credential lease expiry must be numeric or null")
        if not isinstance(refreshable, bool) or not isinstance(fedramp, bool):
            raise ValueError("credential lease flags must be booleans")
        if account_id is not None and not isinstance(account_id, str):
            raise ValueError(
                "credential lease account id must be a string or null")
        return cls(
            credential=credential,
            value=secret,
            generation=generation,
            expires_at=(
                float(expires_at) if expires_at is not None else None),
            refreshable=refreshable,
            account_id=account_id,
            fedramp=fedramp,
        )


@dataclass(frozen=True)
class OpenAITokenSet:
    """In-memory ChatGPT OAuth state supplied by the future storage layer."""

    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    id_token: str | None = field(default=None, repr=False)
    account_id: str | None = None
    fedramp: bool = False
    expires_at: float | None = None
    last_refresh: float | None = None

    def normalized(self, now: float | None = None) -> "OpenAITokenSet":
        if (not isinstance(self.access_token, str)
                or not self.access_token
                or not isinstance(self.refresh_token, str)
                or not self.refresh_token):
            raise ValueError(
                "OpenAI subscription requires access and refresh tokens")
        if (self.id_token is not None
                and (not isinstance(self.id_token, str)
                     or not self.id_token)):
            raise ValueError("OpenAI id token must be a string or null")
        if (self.account_id is not None
                and (not isinstance(self.account_id, str)
                     or not self.account_id)):
            raise ValueError("OpenAI account id must be a string or null")
        if not isinstance(self.fedramp, bool):
            raise ValueError("OpenAI FedRAMP flag must be a boolean")
        for label, value in (
                ("expiry", self.expires_at),
                ("last refresh", self.last_refresh)):
            if (value is not None
                    and (not isinstance(value, (int, float))
                         or isinstance(value, bool)
                         or not math.isfinite(value))):
                raise ValueError(
                    f"OpenAI token {label} must be finite or null")
        current_time = time.time() if now is None else now
        expires_at = self.expires_at
        if expires_at is None:
            expires_at = jwt_expiration(self.access_token)
        account_id = self.account_id or token_account_id(
            self.id_token or self.access_token)
        fedramp = self.fedramp or token_fedramp(
            self.id_token or self.access_token)
        return OpenAITokenSet(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            id_token=self.id_token,
            account_id=account_id,
            fedramp=fedramp,
            expires_at=expires_at,
            last_refresh=(
                self.last_refresh
                if self.last_refresh is not None else current_time),
        )


@dataclass(frozen=True)
class RefreshResult:
    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    id_token: str | None = field(default=None, repr=False)


class CredentialError(RuntimeError):
    pass


class CredentialUnavailable(CredentialError):
    pass


class RefreshTransientError(CredentialError):
    def __init__(self, message: str, *, request_may_have_been_sent=False):
        super().__init__(message)
        self.request_may_have_been_sent = request_may_have_been_sent


class RefreshPermanentError(CredentialError):
    pass


class RefreshIndeterminateError(RefreshPermanentError):
    pass


class CredentialAuthority(Protocol):
    async def lease(
            self, credential: CredentialRef,
            rejected_generation: int | None = None) -> CredentialLease:
        ...

    def available(self) -> frozenset[CredentialRef]:
        ...


class StaticCredential:
    def __init__(self, credential: CredentialRef, value: str):
        self.credential = credential
        self._value = value

    async def lease(
            self, rejected_generation: int | None = None) -> CredentialLease:
        return CredentialLease(
            credential=self.credential,
            value=self._value,
        )


RotationFunction = Callable[
    [OpenAITokenSet],
    Awaitable[OpenAITokenSet],
]


class OpenAIChatGPTCredential:
    """One rotating ChatGPT token set with single-flight refresh."""

    def __init__(
            self, tokens: OpenAITokenSet, *,
            rotate: RotationFunction | None = None,
            clock: Callable[[], float] = time.time,
            generation: int = 0):
        self.credential = CredentialRef.openai_subscription()
        self._clock = clock
        self._tokens = tokens.normalized(clock())
        self._rotate = rotate or rotate_openai_tokens
        self._generation = generation
        self._refresh_lock = asyncio.Lock()
        self._permanent_failure: tuple[int, CredentialError] | None = None

    @property
    def tokens(self) -> OpenAITokenSet:
        return self._tokens

    @property
    def generation(self) -> int:
        return self._generation

    def replace(self, tokens: OpenAITokenSet):
        self._tokens = tokens.normalized(self._clock())
        self._generation += 1
        self._permanent_failure = None

    def _lease(self) -> CredentialLease:
        return CredentialLease(
            credential=self.credential,
            value=self._tokens.access_token,
            generation=self._generation,
            expires_at=self._tokens.expires_at,
            refreshable=True,
            account_id=self._tokens.account_id,
            fedramp=self._tokens.fedramp,
        )

    def _expired(self, now: float) -> bool:
        return (
            self._tokens.expires_at is not None
            and self._tokens.expires_at <= now)

    def _needs_proactive_refresh(self, now: float) -> bool:
        if self._tokens.expires_at is not None:
            return (
                self._tokens.expires_at
                <= now + OPENAI_ACCESS_TOKEN_REFRESH_WINDOW_S)
        return (
            self._tokens.last_refresh is not None
            and self._tokens.last_refresh
            < now - OPENAI_FALLBACK_REFRESH_INTERVAL_S)

    async def lease(
            self, rejected_generation: int | None = None) -> CredentialLease:
        now = self._clock()
        rejected_current = (
            rejected_generation is not None
            and rejected_generation == self._generation)
        if (not rejected_current
                and rejected_generation is not None):
            return self._lease()
        if (not rejected_current
                and not self._needs_proactive_refresh(now)):
            return self._lease()

        async with self._refresh_lock:
            now = self._clock()
            if (rejected_generation is not None
                    and rejected_generation != self._generation):
                return self._lease()
            if (rejected_generation is None
                    and not self._needs_proactive_refresh(now)):
                return self._lease()
            if (self._permanent_failure is not None
                    and self._permanent_failure[0] == self._generation):
                raise self._permanent_failure[1]

            try:
                refresh_generation = self._generation
                tokens = await self._rotate(self._tokens)
                if refresh_generation != self._generation:
                    # An explicit credential replacement won the race while
                    # the exchange was in flight. Its newer state is
                    # authoritative; never overwrite it with the old
                    # generation's late response.
                    return self._lease()
                self._install_rotation_result(tokens)
            except asyncio.CancelledError as error:
                if (refresh_generation == self._generation
                        and getattr(
                            error, "request_may_have_been_sent", True)):
                    # Capability revocation cancels in-flight broker work.
                    # Once request bytes may have reached the token endpoint,
                    # cancellation cannot make replaying that rotating token
                    # safe; remember the indeterminate generation first.
                    self._permanent_failure = (
                        self._generation,
                        RefreshIndeterminateError(
                            "OpenAI token refresh was cancelled after it "
                            "may have been sent; the refresh token will not "
                            "be replayed"),
                    )
                raise
            except RefreshPermanentError as error:
                self._permanent_failure = (self._generation, error)
                raise
            except RefreshTransientError as error:
                if error.request_may_have_been_sent:
                    permanent = RefreshIndeterminateError(
                        "OpenAI token refresh outcome is unknown; the "
                        "refresh token will not be replayed")
                    self._permanent_failure = (
                        self._generation, permanent)
                    raise permanent from error
                if rejected_generation is not None or self._expired(now):
                    raise
                # A proactive refresh is best-effort while the current token
                # is still valid.  A later request can try again.
                return self._lease()
            return self._lease()

    def _install_rotation_result(
            self, tokens: OpenAITokenSet) -> None:
        if not isinstance(tokens, OpenAITokenSet):
            raise RefreshIndeterminateError(
                "OpenAI token rotation returned an invalid result")
        try:
            self._tokens = tokens.normalized(self._clock())
        except ValueError as error:
            raise RefreshIndeterminateError(
                f"OpenAI token rotation returned invalid tokens: "
                f"{error}") from error
        self._generation += 1
        self._permanent_failure = None


class CredentialBroker:
    """Root-process credential registry and local authority."""

    def __init__(self):
        self._records: dict[CredentialRef, object] = {}

    def available(self) -> frozenset[CredentialRef]:
        return frozenset(self._records)

    def install_static(self, credential: CredentialRef, value: str) -> None:
        if not value:
            raise ValueError("static credential value must not be empty")
        self._records[credential] = StaticCredential(credential, value)

    def install_openai_subscription(
            self, tokens: OpenAITokenSet, *,
            rotate: RotationFunction | None = None,
            clock: Callable[[], float] = time.time) -> None:
        credential = CredentialRef.openai_subscription()
        current = self._records.get(credential)
        if isinstance(current, OpenAIChatGPTCredential):
            current.replace(tokens)
            return
        self._records[credential] = OpenAIChatGPTCredential(
            tokens, rotate=rotate, clock=clock)

    def openai_tokens(self) -> OpenAITokenSet | None:
        record = self._records.get(CredentialRef.openai_subscription())
        if isinstance(record, OpenAIChatGPTCredential):
            return record.tokens
        return None

    async def lease(
            self, credential: CredentialRef,
            rejected_generation: int | None = None) -> CredentialLease:
        record = self._records.get(credential)
        if record is None:
            raise CredentialUnavailable(
                f"credential {credential.encode()!r} is unavailable")
        return await record.lease(rejected_generation)


def authorization_headers(
        spec: AuthSpec | None, lease: CredentialLease | None) -> dict:
    if spec is None or spec.credential is None:
        return {}
    if lease is None or lease.credential != spec.credential:
        raise CredentialUnavailable(
            f"no lease for {spec.credential.encode()!r}")
    if spec.scheme == "openai-subscription":
        headers = {
            "Authorization": f"Bearer {lease.value}",
            "originator": OPENAI_ORIGINATOR,
        }
        if lease.account_id:
            headers["ChatGPT-Account-ID"] = lease.account_id
        if lease.fedramp:
            headers["X-OpenAI-Fedramp"] = "true"
        return headers
    if spec.scheme == "anthropic":
        return {"x-api-key": lease.value}
    if spec.scheme == "custom":
        if not spec.header_name:
            raise ValueError(
                "custom authentication requires a header name")
        return {spec.header_name: lease.value}
    if spec.scheme == "bearer":
        return {"Authorization": f"Bearer {lease.value}"}
    raise ValueError(f"unknown authentication scheme {spec.scheme!r}")


def validate_authorization_target(
        spec: AuthSpec | None, request_url: str) -> None:
    """Prevent a persisted connection from redirecting a brokered token."""
    if spec is None or spec.scheme != "openai-subscription":
        return
    try:
        parsed = urllib.parse.urlsplit(request_url)
        port = parsed.port
    except (TypeError, ValueError):
        parsed = None
        port = None
    if (parsed is None
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.scheme.lower() != "https"
            or parsed.hostname is None
            or parsed.hostname.lower() != "chatgpt.com"
            or parsed.fragment
            or request_url not in OPENAI_CHATGPT_AUTHORIZED_URLS):
        raise CredentialUnavailable(
            "OpenAI subscription credentials may only be sent to "
            "the canonical ChatGPT Codex endpoints")


def _decode_jwt_payload(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 3 or not parts[1]:
            return None
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        value = json.loads(base64.urlsafe_b64decode(
            payload.encode("ascii")).decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return None


def jwt_expiration(token: str) -> float | None:
    """Read JWT ``exp`` only as a refresh scheduling hint.

    Authorization still relies on TLS and the issuing service; Loki does not
    mistake this unverified payload decode for JWT signature verification.
    """
    payload = _decode_jwt_payload(token)
    expiry = payload.get("exp") if payload is not None else None
    if isinstance(expiry, (int, float)) and not isinstance(expiry, bool):
        return float(expiry)
    return None


def token_fedramp(token: str) -> bool:
    payload = _decode_jwt_payload(token)
    if payload is None:
        return False
    direct = payload.get("chatgpt_account_is_fedramp")
    if isinstance(direct, bool):
        return direct
    auth_claims = payload.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        nested = auth_claims.get("chatgpt_account_is_fedramp")
        if isinstance(nested, bool):
            return nested
    return False


def token_account_id(token: str) -> str | None:
    payload = _decode_jwt_payload(token)
    if payload is None:
        return None
    direct = payload.get("chatgpt_account_id")
    if isinstance(direct, str) and direct:
        return direct
    auth_claims = payload.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        nested = auth_claims.get("chatgpt_account_id")
        if isinstance(nested, str) and nested:
            return nested
    return None


def refreshed_openai_tokens(
        current: OpenAITokenSet,
        result: RefreshResult,
        now: float | None = None) -> OpenAITokenSet:
    """Validate a token-endpoint result and preserve account invariants."""
    if not isinstance(result, RefreshResult):
        raise RefreshIndeterminateError(
            "OpenAI token refresh returned an invalid result")
    if (not isinstance(result.access_token, str)
            or not result.access_token):
        raise RefreshIndeterminateError(
            "OpenAI token refresh succeeded without an access token")
    if (result.refresh_token is not None
            and (not isinstance(result.refresh_token, str)
                 or not result.refresh_token)):
        raise RefreshIndeterminateError(
            "OpenAI token refresh returned an invalid refresh token")
    if (result.id_token is not None
            and (not isinstance(result.id_token, str)
                 or not result.id_token)):
        raise RefreshIndeterminateError(
            "OpenAI token refresh returned an invalid id token")
    current = current.normalized(now)
    identity_token = result.id_token or result.access_token
    account_id = token_account_id(identity_token)
    if (account_id is not None
            and current.account_id is not None
            and account_id != current.account_id):
        raise RefreshPermanentError(
            "OpenAI token refresh changed the selected account")
    return OpenAITokenSet(
        access_token=result.access_token,
        refresh_token=result.refresh_token or current.refresh_token,
        id_token=result.id_token or current.id_token,
        account_id=current.account_id or account_id,
        fedramp=(
            token_fedramp(identity_token)
            if result.id_token is not None
            else current.fedramp),
        expires_at=jwt_expiration(result.access_token),
        last_refresh=time.time() if now is None else now,
    ).normalized(now)


async def rotate_openai_tokens(
        current: OpenAITokenSet) -> OpenAITokenSet:
    result = await request_openai_token_refresh(
        current.refresh_token)
    return refreshed_openai_tokens(current, result)


def _refresh_error_code(data) -> str | None:
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        return code if isinstance(code, str) else None
    if isinstance(error, str):
        return error
    code = data.get("code")
    return code if isinstance(code, str) else None


async def request_openai_token_refresh(
        refresh_token: str) -> RefreshResult:
    """Exchange one ChatGPT refresh token without automatic POST retries."""
    body = json.dumps({
        "client_id": OPENAI_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }, separators=(",", ":")).encode("utf-8")
    try:
        response = await http_client.async_http_request(
            "POST",
            OPENAI_REFRESH_URL,
            body=body,
            headers_in={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "originator": OPENAI_ORIGINATOR,
            },
            timeout=OPENAI_REFRESH_TIMEOUT_S,
            max_bytes=OPENAI_REFRESH_MAX_BYTES,
            retry_max_attempts=1,
        )
    except Exception as error:
        may_have_been_sent = getattr(
            error, "request_may_have_been_sent", True)
        raise RefreshTransientError(
            f"OpenAI token refresh failed: {error}",
            request_may_have_been_sent=may_have_been_sent,
        ) from error

    if response.truncated and 200 <= response.status < 300:
        raise RefreshIndeterminateError(
            "OpenAI token refresh response exceeded the size limit")
    try:
        data = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        if 200 <= response.status < 300:
            raise RefreshIndeterminateError(
                "OpenAI token refresh returned invalid JSON") from error
        raise RefreshTransientError(
            f"OpenAI token refresh returned HTTP {response.status}",
            # Receiving any HTTP response proves that the request crossed the
            # local delivery boundary. An intermediary error cannot prove that
            # an upstream rotating-token exchange did not commit.
            request_may_have_been_sent=True,
        ) from error

    if not 200 <= response.status < 300:
        code = (_refresh_error_code(data) or "").lower()
        if (response.status == 401
                or code in {
                    "invalid_grant",
                    "invalid_token",
                    "refresh_token_expired",
                    "refresh_token_reused",
                    "refresh_token_invalidated",
                }):
            raise RefreshPermanentError(
                "OpenAI refresh token is expired, reused, or revoked")
        raise RefreshTransientError(
            f"OpenAI token refresh returned HTTP {response.status}",
            request_may_have_been_sent=True,
        )

    access_token = data.get("access_token")
    refresh_value = data.get("refresh_token")
    id_token = data.get("id_token")
    if not isinstance(access_token, str) or not access_token:
        raise RefreshIndeterminateError(
            "OpenAI token refresh response omitted access_token")
    if (refresh_value is not None
            and (not isinstance(refresh_value, str)
                 or not refresh_value)):
        raise RefreshIndeterminateError(
            "OpenAI token refresh returned invalid refresh_token")
    if (id_token is not None
            and (not isinstance(id_token, str) or not id_token)):
        raise RefreshIndeterminateError(
            "OpenAI token refresh returned invalid id_token")
    return RefreshResult(
        access_token=access_token,
        refresh_token=refresh_value,
        id_token=id_token,
    )
