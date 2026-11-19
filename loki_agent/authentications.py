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
from dataclasses import dataclass, field, replace
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
# namespace: sending Loki's package version returns an empty catalog. This
# value selects the current account catalog Loki has tested, rather than
# claiming that Loki's unrelated package version is a Codex protocol level.
# Because the complete URL is authorization-allowlisted below, advancing this
# boundary requires an explicit protocol review and a live model request.
OPENAI_CHATGPT_CODEX_COMPATIBILITY_VERSION = "0.169.0"
OPENAI_CHATGPT_MODELS_REQUEST_URL = (
    f"{OPENAI_CHATGPT_MODELS_URL}?client_version="
    f"{OPENAI_CHATGPT_CODEX_COMPATIBILITY_VERSION}")
# The exact Codex endpoints the ChatGPT subscription credential may authorize
# for inference and model discovery.  Account-level endpoints are a separate
# authority domain and are declared by their own control module; they are never
# added here, so a control-path bug cannot widen where the inference credential
# may be sent.
OPENAI_CHATGPT_CODEX_URLS = frozenset({
    OPENAI_CHATGPT_RESPONSES_URL,
    OPENAI_CHATGPT_MODELS_REQUEST_URL,
})

# OAuth client identifiers are public protocol identifiers, not secrets, the
# same as OpenAI's above. A provider-approved registration must be used; the
# login command accepts an override for deployments registered differently.
ANTHROPIC_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
ANTHROPIC_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
# Subscription traffic is served from the beta endpoint; the query string is
# part of the authorized destination, so the allowlist below names it exactly.
ANTHROPIC_BETA_MESSAGES_URL = f"{ANTHROPIC_MESSAGES_URL}?beta=true"
# A subscription token is refused unless the request names the flags the grant
# was issued under; the OAuth flag alone is not enough.
ANTHROPIC_OAUTH_BETA = "claude-code-20250219,oauth-2025-04-20"
# Inference carries the client identity the OAuth grant was issued to. The
# account attributes quota and billing to that identity, so a request
# presenting an unrelated one is not attributed to the subscription.
ANTHROPIC_CLIENT_USER_AGENT = "claude-cli/2.1.290 (external, cli)"
ANTHROPIC_CLIENT_HEADERS = {"x-app": "cli"}
ANTHROPIC_INFERENCE_SCOPE = "user:inference"
ANTHROPIC_USAGE_SCOPE = "user:profile"
# Request usage access because /account reads live quota; it does not need
# upload, remote tool, or session-management permissions.
ANTHROPIC_OAUTH_SCOPES = frozenset({ANTHROPIC_INFERENCE_SCOPE, ANTHROPIC_USAGE_SCOPE})
# Bind bearer credentials to the exact inference URL, not the provider's host.
# Switching away from the beta endpoint means changing this one constant; the
# allowlist, catalog and model-selection checks all derive from it.
ANTHROPIC_SUBSCRIPTION_URLS = frozenset({ANTHROPIC_BETA_MESSAGES_URL})
# Try to refresh before expiry so a token can survive delivery and retries.
# Opaque tokens without an expiry can only be scheduled by age.
ANTHROPIC_ACCESS_TOKEN_REFRESH_WINDOW_S = 5 * 60
ANTHROPIC_FALLBACK_REFRESH_INTERVAL_S = 60 * 60
ANTHROPIC_REFRESH_TIMEOUT_S = 30
ANTHROPIC_REFRESH_MAX_BYTES = 256 * 1024


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

    @classmethod
    def anthropic_subscription(cls) -> "CredentialRef":
        return cls("anthropic-subscription", "anthropic")

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
    # These are the exact destinations of this connection/operation, not
    # persisted user approvals. The supervisor independently checks approval
    # before issuing a value. An empty set permits no authenticated request.
    authorized_urls: frozenset[str] = frozenset()


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


@dataclass(frozen=True)
class AnthropicAccount:
    """Non-secret account identity reported by the Anthropic token endpoint."""

    uuid: str
    email_address: str | None = None
    organization_uuid: str | None = None


def _visible_ascii(value, *, max_size=8192) -> bool:
    # A bearer must be header-safe and fit in a bounded credential IPC frame
    # even after JSON quoting.
    return (isinstance(value, str) and 0 < len(value) <= max_size
            and all("!" <= char <= "~" for char in value))


def validate_anthropic_client_id(value) -> None:
    if not _visible_ascii(value, max_size=512):
        raise ValueError("Anthropic requires a registered OAuth client ID")


def _valid_anthropic_scopes(scopes) -> bool:
    return (isinstance(scopes, frozenset) and bool(scopes)
            and all(_visible_ascii(scope) and '"' not in scope and '\\' not in scope
                    for scope in scopes))


def parse_anthropic_scopes(value) -> frozenset[str]:
    scopes = frozenset(value.split(" ")) if isinstance(value, str) else frozenset()
    if not _valid_anthropic_scopes(scopes):
        raise ValueError("Anthropic returned invalid OAuth scopes")
    return scopes


def anthropic_access_expiration(expires_in, now) -> float | None:
    if expires_in is None:
        return None
    # Bound the duration before adding it to a floating-point timestamp.
    if type(expires_in) is not int or not 0 < expires_in <= 30 * 24 * 60 * 60:
        raise ValueError("Anthropic returned invalid expires_in")
    return now + expires_in


def validate_anthropic_token_type(data) -> None:
    token_type = data.get("token_type", "Bearer")
    if not isinstance(token_type, str) or token_type.lower() != "bearer":
        raise ValueError("Anthropic returned an unsupported token type")


def parse_anthropic_account(data) -> AnthropicAccount | None:
    account = data.get("account")
    if account is None:
        return None
    if not isinstance(account, dict):
        raise ValueError("Anthropic returned an invalid account")
    org = data.get("organization")
    if org is not None and not isinstance(org, dict):
        raise ValueError("Anthropic returned an invalid organization")
    return AnthropicAccount(
        uuid=account.get("uuid"),
        email_address=account.get("email_address"),
        organization_uuid=org.get("uuid") if org is not None else None,
    )


@dataclass(frozen=True)
class AnthropicTokenSet:
    """The supervisor's stored and in-memory Anthropic login.

    Unlike OpenAI's JWT-based token set, this grant has an opaque access token
    and explicit scopes, and refresh must use its original client registration.
    Only the access token and lease metadata cross into a worker; replacing or
    removing the durable grant disallows any further rotation of the old grant.
    """

    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    # Refresh must use the registration to which the token was issued.
    client_id: str
    scopes: frozenset[str]
    account: AnthropicAccount | None = None
    expires_at: float | None = None
    last_refresh: float | None = None
    # A later login may be a different account even if both responses omit
    # identity. Only replacements from the same durable grant are reusable.
    grant_id: str | None = None

    def normalized(self, now: float | None = None) -> "AnthropicTokenSet":
        if not (_visible_ascii(self.access_token) and _visible_ascii(self.refresh_token)):
            raise ValueError("Anthropic subscription requires valid access and refresh tokens")
        validate_anthropic_client_id(self.client_id)
        if not _valid_anthropic_scopes(self.scopes):
            raise ValueError("Anthropic subscription requires valid OAuth scopes")
        if self.grant_id is not None and not _visible_ascii(self.grant_id, max_size=128):
            raise ValueError("Anthropic subscription has an invalid grant ID")
        if self.account is not None and (
                not isinstance(self.account, AnthropicAccount)
                or not _visible_ascii(self.account.uuid, max_size=128)
                or any(field is not None and (not isinstance(field, str) or not field)
                       for field in [self.account.email_address, self.account.organization_uuid])):
            raise ValueError("Anthropic returned an invalid account identity")
        for label, value in [("expiry", self.expires_at), ("last refresh", self.last_refresh)]:
            if value is None:
                continue
            try:
                if type(value) in {int, float} and math.isfinite(value):
                    continue
            except OverflowError:
                pass
            raise ValueError(f"Anthropic token {label} must be finite or null")
        return AnthropicTokenSet(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            client_id=self.client_id,
            scopes=self.scopes,
            account=self.account,
            expires_at=self.expires_at,
            last_refresh=(self.last_refresh if self.last_refresh is not None
                          else (time.time() if now is None else now)),
            grant_id=self.grant_id,
        )


@dataclass(frozen=True)
class AnthropicRefreshResult:
    """The response before rotation merges it with the existing grant.

    A login must supply a refresh token, while a refresh may omit unchanged
    fields. Do not treat a partial response as a new leaseable token set.
    """

    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_in: int | None = None
    scopes: frozenset[str] | None = None
    account: AnthropicAccount | None = None


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
            self, credential: CredentialRef, destination: str,
            rejected_generation: int | None = None) -> CredentialLease:
        ...

    def available(self) -> frozenset[CredentialRef]:
        ...

    async def unapproved_destinations(
            self, credential: CredentialRef, destinations: list[str]) -> list[str]:
        ...

    async def approve_destinations(
            self, credential: CredentialRef, destinations: list[str]) -> None:
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

AnthropicRotationFunction = Callable[
    [AnthropicTokenSet],
    Awaitable[AnthropicTokenSet],
]


class _RotatingSubscriptionCredential:
    """Serialize refresh and never replay a possibly consumed token.

    Share the generation rules across providers so their delivery and
    cancellation handling cannot diverge. Only expiry and lease fields vary.
    """

    _provider_label: str
    _token_set_type: type

    def __init__(
            self, credential: CredentialRef, tokens, *,
            rotate, clock: Callable[[], float], generation: int = 0):
        self.credential = credential
        self._clock = clock
        self._tokens = self._accepted(tokens)
        self._rotate = rotate
        self._generation = generation
        self._refresh_lock = asyncio.Lock()
        self._permanent_failure: CredentialError | None = None

    @property
    def tokens(self):
        return self._tokens

    @property
    def generation(self) -> int:
        return self._generation

    def replace(self, tokens):
        self._tokens = self._accepted(tokens)
        self._generation += 1
        self._permanent_failure = None

    def _accepted(self, tokens):
        if not isinstance(tokens, self._token_set_type):
            raise ValueError(
                f"{self._provider_label} rotation produced an invalid result")
        return tokens.normalized(self._clock())

    def _lease(self) -> CredentialLease:
        raise NotImplementedError

    def _needs_proactive_refresh(self, now: float) -> bool:
        raise NotImplementedError

    def _expired(self, now: float) -> bool:
        return (
            self._tokens.expires_at is not None
            and self._tokens.expires_at <= now)

    def _refresh_required(self, rejected_generation) -> bool:
        if self._permanent_failure is not None:
            raise self._permanent_failure
        now = self._clock()
        if rejected_generation is None:
            return self._needs_proactive_refresh(now)
        # A delayed rejection must not force rotation of a newer token, but
        # the newer token can independently have expired while we waited.
        return rejected_generation == self._generation or self._expired(now)

    async def lease(
            self, rejected_generation: int | None = None, *,
            check_scope: Callable[[], None] | None = None) -> CredentialLease:
        if check_scope is not None:
            check_scope()
        if not self._refresh_required(rejected_generation):
            return self._lease()

        observed_generation = self._generation
        async with self._refresh_lock:
            # A different request may have narrowed this grant while we
            # queued. Only a request still authorized can rotate it again.
            if check_scope is not None:
                check_scope()
            if not self._refresh_required(rejected_generation):
                return self._lease()
            if (observed_generation != self._generation
                    and rejected_generation != self._generation
                    and not self._expired(self._clock())):
                # One refresh already served this burst of waiters. A short
                # expiry on the new token must not turn it into N rotations.
                return self._lease()
            refresh_generation = self._generation
            try:
                tokens = await self._rotate(self._tokens)
                if refresh_generation == self._generation:
                    self._install_rotation_result(tokens)
            except asyncio.CancelledError as error:
                if (refresh_generation == self._generation
                        and getattr(error, "request_may_have_been_sent", True)):
                    # Cancelling the caller says nothing about whether the
                    # issuer consumed a token whose bytes may have reached it.
                    self._permanent_failure = RefreshIndeterminateError(
                        f"{self._provider_label} token refresh was cancelled after it "
                        "may have been sent; the refresh token will not be replayed")
                raise
            except RefreshPermanentError as error:
                # An explicit replacement wins over both late results and
                # late failures. The failed exchange concerned an older token.
                if refresh_generation == self._generation:
                    self._permanent_failure = error
                    raise
            except RefreshTransientError as error:
                if refresh_generation == self._generation:
                    if error.request_may_have_been_sent:
                        self._permanent_failure = RefreshIndeterminateError(
                            f"{self._provider_label} token refresh outcome is unknown; "
                            "the refresh token will not be replayed")
                        raise self._permanent_failure from error
                    # An unsent proactive refresh may fall back only if the
                    # old token survived the time spent waiting for refresh.
                    if rejected_generation is not None or self._expired(self._clock()):
                        raise
            if self._expired(self._clock()):
                raise CredentialUnavailable(f"{self._provider_label} access token is expired")
            return self._lease()

    def _install_rotation_result(self, tokens) -> None:
        try:
            self._tokens = self._accepted(tokens)
        except ValueError as error:
            raise RefreshIndeterminateError(
                f"{self._provider_label} token rotation returned invalid "
                f"tokens: {error}") from error
        self._generation += 1
        self._permanent_failure = None


class OpenAIChatGPTCredential(_RotatingSubscriptionCredential):
    """One rotating ChatGPT token set with single-flight refresh."""

    _provider_label = "OpenAI"
    _token_set_type = OpenAITokenSet

    def __init__(
            self, tokens: OpenAITokenSet, *,
            rotate: RotationFunction | None = None,
            clock: Callable[[], float] = time.time,
            generation: int = 0):
        super().__init__(
            CredentialRef.openai_subscription(),
            tokens,
            rotate=rotate or rotate_openai_tokens,
            clock=clock,
            generation=generation,
        )

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

    def _needs_proactive_refresh(self, now: float) -> bool:
        if self._tokens.expires_at is not None:
            return (
                self._tokens.expires_at
                <= now + OPENAI_ACCESS_TOKEN_REFRESH_WINDOW_S)
        return (
            self._tokens.last_refresh is not None
            and self._tokens.last_refresh
            < now - OPENAI_FALLBACK_REFRESH_INTERVAL_S)


class AnthropicSubscriptionCredential(_RotatingSubscriptionCredential):
    """Lease only the access token; refresh state stays in the supervisor."""

    _provider_label = "Anthropic"
    _token_set_type = AnthropicTokenSet

    def __init__(
            self, tokens: AnthropicTokenSet, *,
            rotate: AnthropicRotationFunction | None = None,
            clock: Callable[[], float] = time.time,
            generation: int = 0):
        super().__init__(
            CredentialRef.anthropic_subscription(),
            tokens,
            rotate=rotate or rotate_anthropic_tokens,
            clock=clock,
            generation=generation,
        )

    def _lease(self) -> CredentialLease:
        return CredentialLease(
            credential=self.credential,
            value=self._tokens.access_token,
            generation=self._generation,
            expires_at=self._tokens.expires_at,
            refreshable=True,
            account_id=(
                self._tokens.account.uuid
                if self._tokens.account is not None else None),
        )

    def require_scope_for(self, destination: str) -> None:
        # The broker already confined this lease to the two exact route sets.
        inference = destination in ANTHROPIC_SUBSCRIPTION_URLS
        scope = ANTHROPIC_INFERENCE_SCOPE if inference else ANTHROPIC_USAGE_SCOPE
        if scope not in self._tokens.scopes:
            purpose = "inference" if inference else "account usage"
            raise CredentialUnavailable(f"Anthropic login did not grant access to {purpose}")

    def _needs_proactive_refresh(self, now: float) -> bool:
        if self._tokens.expires_at is not None:
            # A token issued for less than the usual margin must still be
            # usable without refreshing before every request.
            lifetime = self._tokens.expires_at - self._tokens.last_refresh
            margin = min(ANTHROPIC_ACCESS_TOKEN_REFRESH_WINDOW_S, max(0, lifetime / 10))
            return self._tokens.expires_at <= now + margin
        return self._tokens.last_refresh <= now - ANTHROPIC_FALLBACK_REFRESH_INTERVAL_S


class CredentialBroker:
    """Root-process credential registry and local authority."""

    def __init__(self, storage=None):
        self._records: dict[
            CredentialRef,
            StaticCredential
            | OpenAIChatGPTCredential
            | AnthropicSubscriptionCredential] = {}
        self._storage = storage
        # Launch configuration belongs to this supervisor's lifetime. Only
        # storage-free authorities need in-memory user approvals; production
        # reads the protected document afresh rather than caching permissions.
        self._startup_destinations: dict[CredentialRef, frozenset[str]] = {}
        self._memory_approvals: dict[CredentialRef, set[str]] = {}

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

    def install_anthropic_subscription(
            self, tokens: AnthropicTokenSet, *,
            rotate: AnthropicRotationFunction | None = None,
            clock: Callable[[], float] = time.time) -> None:
        credential = CredentialRef.anthropic_subscription()
        current = self._records.get(credential)
        if isinstance(current, AnthropicSubscriptionCredential):
            current.replace(tokens)
            return
        self._records[credential] = AnthropicSubscriptionCredential(
            tokens, rotate=rotate, clock=clock)

    def _check_destinations(self, credential, destinations):
        if credential not in self._records:
            raise CredentialUnavailable(
                f"credential {credential.encode()!r} is unavailable")
        if not isinstance(destinations, list) or not destinations:
            raise CredentialUnavailable("request destinations are required")
        for destination in destinations:
            validate_destination_url(destination)
        if credential.kind == "openai-subscription":
            # Account controls own a separate exact set. It is referenced here,
            # never widened into a host/origin rule or overridden by disk data.
            from .openai_controls import OPENAI_CHATGPT_ACCOUNT_URLS
            canonical = OPENAI_CHATGPT_CODEX_URLS | OPENAI_CHATGPT_ACCOUNT_URLS
            if any(destination not in canonical for destination in destinations):
                raise CredentialUnavailable(
                    "OpenAI subscription destination is not supported")
        if credential.kind == "anthropic-subscription":
            # The broker can lease for both operations, but each request path
            # has its own exact AuthSpec; usage does not expand inference authority.
            from .anthropic_controls import ANTHROPIC_ACCOUNT_URLS
            canonical = ANTHROPIC_SUBSCRIPTION_URLS | ANTHROPIC_ACCOUNT_URLS
            if any(destination not in canonical for destination in destinations):
                raise CredentialUnavailable(
                    "Anthropic subscription destination is not supported")

    def allow_startup_destinations(self, credential, destinations):
        """Grant only the exact URLs derived from captured launch configuration."""
        self._check_destinations(credential, destinations)
        self._startup_destinations[credential] = frozenset(destinations)

    async def unapproved_destinations(self, credential, destinations):
        self._check_destinations(credential, destinations)
        if credential.kind in {
                "openai-subscription", "anthropic-subscription"}:
            return []
        approved = self._startup_destinations.get(credential, frozenset())
        if all(destination in approved for destination in destinations):
            return []
        if self._storage is not None:
            approved = approved | self._storage.approved_destinations(credential)
        else:
            approved = approved | self._memory_approvals.get(credential, set())
        return [destination for destination in destinations
                if destination not in approved]

    async def approve_destinations(
            self, credential, destinations, *, before_commit=None):
        self._check_destinations(credential, destinations)
        # Built-in subscription routes and launch configuration need no
        # persistent record. A confirmation only adds previously missing URLs.
        missing = await self.unapproved_destinations(credential, destinations)
        if not missing:
            if before_commit is not None:
                before_commit()
            return
        if self._storage is not None:
            await self._storage.approve_destinations(
                credential, missing, before_commit=before_commit)
        else:
            if before_commit is not None:
                before_commit()
            self._memory_approvals.setdefault(credential, set()).update(missing)

    async def _check_approval(self, credential, destination):
        if await self.unapproved_destinations(credential, [destination]):
            raise CredentialUnavailable(
                "approval is required for this credential and destination")

    async def lease(
            self, credential: CredentialRef, destination: str,
            rejected_generation: int | None = None) -> CredentialLease:
        await self._check_approval(credential, destination)
        record = self._records[credential]
        if isinstance(record, AnthropicSubscriptionCredential):
            # Check inside the shared refresh gate too: a check only in this
            # broker would go stale while the request queued for rotation.
            lease = await record.lease(rejected_generation, check_scope=lambda: record.require_scope_for(destination))
        else:
            lease = await record.lease(rejected_generation)
        # Refresh may have awaited network/storage while an approval changed.
        # No await follows the final checks before returning the value to the
        # capability writer, which itself checks channel revocation.
        await self._check_approval(credential, destination)
        if isinstance(record, AnthropicSubscriptionCredential):
            # The issuer may have narrowed the newly rotated grant.
            record.require_scope_for(destination)
        return lease


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
    if spec.scheme == "anthropic-subscription":
        # Select OAuth bearer handling without changing API-key requests. The
        # beta list and the browser-access marker accompany the bearer: the
        # account refuses the subscription token when either is missing.
        return {
            "Authorization": f"Bearer {lease.value}",
            "anthropic-beta": ANTHROPIC_OAUTH_BETA,
            "anthropic-dangerous-direct-browser-access": "true",
        }
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
    """Reject use of a credential outside its configured network targets."""
    if spec is None or spec.credential is None:
        return
    validate_destination_url(request_url)
    if request_url not in spec.authorized_urls:
        raise CredentialUnavailable(
            "credential may only be sent to this operation's configured URLs")


def validate_destination_url(request_url: str) -> None:
    """Check URL syntax without changing the approval's exact string identity."""
    if (not isinstance(request_url, str) or not request_url
            or any(ord(character) <= 32 or ord(character) == 127
                   for character in request_url)
            or "#" in request_url):
        raise CredentialUnavailable("request destination is not a valid HTTP URL")
    try:
        parsed = urllib.parse.urlsplit(request_url)
        port = parsed.port  # Validate port syntax, not just hostname syntax.
    except ValueError as error:
        raise CredentialUnavailable(
            "request destination is not a valid HTTP URL") from error
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or port == 0
            or parsed.username is not None or parsed.password is not None):
        raise CredentialUnavailable("request destination is not a valid HTTP URL")


def authorization_origin(request_url: str) -> str:
    """Return a canonical HTTP origin suitable for authorization policy."""
    try:
        parsed = urllib.parse.urlsplit(request_url)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise CredentialUnavailable(
            "credential target is not a valid HTTP URL") from error
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if (scheme not in {"http", "https"}
            or hostname is None
            or parsed.username is not None
            or parsed.password is not None):
        raise CredentialUnavailable(
            "credential target is not a valid HTTP URL")
    hostname = hostname.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    default_port = (
        (scheme == "http" and port in {None, 80})
        or (scheme == "https" and port in {None, 443}))
    authority = hostname if default_port else f"{hostname}:{port}"
    return f"{scheme}://{authority}"


async def authorized_request_headers(
        authority: CredentialAuthority | None,
        spec: AuthSpec | None,
        request_url: str,
        base_headers: dict | None = None,
        rejected_generation: int | None = None,
) -> tuple[dict, CredentialLease | None]:
    """Authorize one HTTP attempt and return its complete request headers.

    Target validation, credential leasing, and header construction form one
    security operation: separating them lets a caller validate one URL and
    accidentally attach the resulting credential to another.  Keep that
    operation here so inference and model discovery cannot drift apart.
    """
    headers = dict(base_headers or {})
    if spec is None or spec.credential is None:
        return headers, None
    validate_authorization_target(spec, request_url)
    if authority is None:
        raise CredentialUnavailable(
            "no credential authority is installed")
    lease = await authority.lease(
        spec.credential,
        destination=request_url,
        rejected_generation=rejected_generation,
    )
    generated = authorization_headers(spec, lease)
    existing_names = {
        str(name).lower(): name
        for name in headers
    }
    collisions = [
        existing_names[str(name).lower()]
        for name in generated
        if str(name).lower() in existing_names
    ]
    if collisions:
        labels = ", ".join(repr(str(name)) for name in collisions)
        raise CredentialError(
            "request headers contain authentication-owned "
            f"header(s): {labels}")
    headers.update(generated)
    return headers, lease


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
        # Waiting for the token endpoint deliberately has no duration: the
        # response may involve server-side processing, and abandoning a sent
        # rotation on a guessed budget would manufacture the very
        # indeterminate outcome this caller must then refuse to replay.
        # Connection setup and delivery stay bounded; cancellation still
        # applies and is treated as possibly-sent by the broker.
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
            response_timeout=None,
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


async def request_anthropic_token_refresh(
        current: AnthropicTokenSet) -> AnthropicRefreshResult:
    """Exchange a refresh token for the same client and scopes, without retries."""
    body = json.dumps({
        "grant_type": "refresh_token",
        "refresh_token": current.refresh_token,
        "client_id": current.client_id,
        "scope": " ".join(sorted(current.scopes)),
    }, separators=(",", ":")).encode("utf-8")
    try:
        # Timing out a sent exchange would leave us unable to tell whether
        # the provider consumed the token. Bound delivery, not the response.
        response = await http_client.async_http_request(
            "POST",
            ANTHROPIC_TOKEN_URL,
            body=body,
            headers_in={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=ANTHROPIC_REFRESH_TIMEOUT_S,
            response_timeout=None,
            max_bytes=ANTHROPIC_REFRESH_MAX_BYTES,
            retry_max_attempts=1,
        )
    except Exception as error:
        raise RefreshTransientError(
            "Anthropic token refresh transport failed",
            request_may_have_been_sent=getattr(error, "request_may_have_been_sent", True),
        ) from error

    success = 200 <= response.status < 300
    if response.truncated and success:
        raise RefreshIndeterminateError("Anthropic token refresh response exceeded the size limit")
    try:
        data = json.loads(response.body.decode("utf-8")) if not response.truncated else None
    except (ValueError, RecursionError):
        data = None
    if not success:
        code = (_refresh_error_code(data) or "").lower()
        if response.status == 401 or code in {
                "invalid_client", "invalid_grant", "invalid_token", "invalid_request"}:
            raise RefreshPermanentError("Anthropic rejected the token refresh; log in again")
        # A server or intermediary error does not prove the issuer left the
        # refresh token unconsumed, so the broker must not replay it.
        raise RefreshTransientError(
            f"Anthropic token refresh returned HTTP {response.status}",
            request_may_have_been_sent=True,
        )
    if not isinstance(data, dict):
        raise RefreshIndeterminateError("Anthropic token refresh did not return a JSON object")
    try:
        validate_anthropic_token_type(data)
        access_token = data.get("access_token")
        if not _visible_ascii(access_token):
            raise ValueError("Anthropic token refresh returned invalid access_token")
        refresh_token = data.get("refresh_token")
        if "refresh_token" in data and not _visible_ascii(refresh_token):
            raise ValueError("Anthropic token refresh returned invalid refresh_token")
        expires_in = data.get("expires_in")
        if "expires_in" in data and anthropic_access_expiration(expires_in, 0) is None:
            raise ValueError("Anthropic token refresh returned invalid expires_in")
        return AnthropicRefreshResult(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            scopes=parse_anthropic_scopes(data["scope"]) if "scope" in data else None,
            account=parse_anthropic_account(data),
        )
    except ValueError as error:
        raise RefreshIndeterminateError(str(error)) from error


def refreshed_anthropic_tokens(
        current: AnthropicTokenSet,
        result: AnthropicRefreshResult,
        now: float | None = None) -> AnthropicTokenSet:
    """Validate a refresh before replacing the root broker's credential."""
    if not isinstance(result, AnthropicRefreshResult):
        raise RefreshIndeterminateError("Anthropic token refresh returned an invalid result")
    timestamp = time.time() if now is None else now
    try:
        refreshed = AnthropicTokenSet(
            access_token=result.access_token,
            # OAuth permits the server to omit an unchanged refresh token or scope.
            refresh_token=(current.refresh_token if result.refresh_token is None else result.refresh_token),
            client_id=current.client_id,
            scopes=current.scopes if result.scopes is None else result.scopes,
            account=current.account if result.account is None else result.account,
            expires_at=anthropic_access_expiration(result.expires_in, timestamp),
            last_refresh=timestamp,
            grant_id=current.grant_id,
        ).normalized(timestamp)
    except ValueError as error:
        raise RefreshIndeterminateError(str(error)) from error
    if not refreshed.scopes.issubset(current.scopes):
        raise RefreshPermanentError("Anthropic refresh widened the originally granted scopes")
    if current.account is not None:
        account = refreshed.account
        if (account.uuid != current.account.uuid
                or (current.account.organization_uuid is not None
                    and account.organization_uuid is not None
                    and account.organization_uuid != current.account.organization_uuid)):
            raise RefreshPermanentError("Anthropic refresh changed the selected account")
        # An omitted organization is not evidence that the account can move
        # to another organization on a later refresh.
        if account.organization_uuid is None and current.account.organization_uuid is not None:
            refreshed = replace(refreshed, account=replace(account, organization_uuid=current.account.organization_uuid))
    return refreshed


async def rotate_anthropic_tokens(
        current: AnthropicTokenSet) -> AnthropicTokenSet:
    return refreshed_anthropic_tokens(current, await request_anthropic_token_refresh(current))
