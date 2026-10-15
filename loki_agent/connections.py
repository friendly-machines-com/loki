"""Non-secret connection construction, validation, and saved descriptions."""

import json
import os
import urllib.parse
from dataclasses import dataclass, replace

from . import authentications
from . import models
from . import openai_models
from . import protocols
from .usages import ContextCapacity
from .authentications import CredentialRef
from .credentials import CredentialInventory, CredentialStore, is_credential_name


class ConnectionDescriptorError(ValueError):
    pass


# Fields of ConnectionDescriptor that the resume approval renders.  Together
# with UNDISPLAYED_CONNECTION_FIELDS this partitions the dataclass, so the
# coverage test in tests/test_connections.py fails when a field is added
# without deciding whether a user must see it before approving a connection.
DISPLAYED_CONNECTION_FIELDS = frozenset({
    "provider_id",
    "provider_name",
    "model",
    "chat_url",
    "models_url",
    "protocol",
    "credential_ref",
    "auth_header",
    "auth_scheme",
    "stream",
    "prompt_cache",
})

# Fields deliberately kept out of the approval, each with the reason it is
# safe to omit: none of them changes where a request goes, which credential is
# attached, or how that credential is presented.
UNDISPLAYED_CONNECTION_FIELDS = {
    "max_tokens": "request budget, not an authorization fact",
    "anthropic_version": "protocol header detail with a fixed default",
    "model_status": "informational catalog state",
    "openai_request_profile": (
        "subscription request shape, implied by the credential-scoped "
        "ChatGPT connection"),
    "reasoning_effort_profile": "model effort choices, not authorization facts",
    "reasoning_capabilities": (
        "advertised control hints, not authorization facts"),
    "context_capacity": "display-only model capacity metadata",
}


def connection_display_fields(
        descriptor: "ConnectionDescriptor") -> list[tuple[str, str]]:
    """Return the non-secret connection facts shown for approval.

    Every field of ConnectionDescriptor is either returned here or named in
    UNDISPLAYED_CONNECTION_FIELDS; tests/test_connections.py holds the two
    sets to the dataclass so an omitted field cannot go unnoticed.
    """
    provider = (
        descriptor.provider_name or descriptor.provider_id or "custom")
    fields = [
        ("Provider", provider),
        ("Model", descriptor.model),
        ("Chat endpoint", descriptor.chat_url),
    ]
    if descriptor.models_url:
        fields.append(("Models endpoint", descriptor.models_url))
    fields.append(("Protocol", descriptor.protocol))
    if descriptor.credential_ref is None:
        fields.append(("Authentication", "none"))
    else:
        credential = descriptor.credential_ref
        fields.append((
            "Credential",
            credential.name if credential.kind == "env"
            else credential.encode(),
        ))
    if descriptor.auth_scheme:
        fields.append(("Credential scheme", descriptor.auth_scheme))
    if descriptor.auth_header:
        fields.append(("Credential header", descriptor.auth_header))
    fields.append(("Streaming", "yes" if descriptor.stream else "no"))
    if descriptor.protocol == "anthropic_messages":
        fields.append((
            "Anthropic prompt cache",
            "yes" if descriptor.prompt_cache else "no",
        ))
    return fields


def _optional_string(value, field_name):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConnectionDescriptorError(
            f"connection {field_name} must be a string or null")
    return value


def _required_string(value, field_name):
    if not isinstance(value, str) or not value:
        raise ConnectionDescriptorError(
            f"connection {field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ConnectionDescriptor:
    provider_id: str | None
    provider_name: str | None
    model: str
    chat_url: str
    models_url: str | None
    protocol: str
    credential_ref: CredentialRef | None = None
    max_tokens: int = 4096
    anthropic_version: str = "2023-06-01"
    auth_header: str | None = None
    auth_scheme: str | None = None
    model_status: str | None = None
    stream: bool = False
    context_capacity: ContextCapacity | None = None
    prompt_cache: bool = False
    openai_request_profile: (
        openai_models.CodexModelRequestProfile | None) = None
    reasoning_effort_profile: (
        models.ReasoningEffortProfile | None) = None
    # Connection-bound model facts: what the catalog advertised for this
    # exact leaf. Metadata only -- it can never alter the endpoint,
    # credential, or authorization of the connection it travels with.
    reasoning_capabilities: (
        models.ReasoningCapabilities | None) = None

    def __post_init__(self):
        if (self.context_capacity is not None
                and not isinstance(self.context_capacity, ContextCapacity)):
            raise ConnectionDescriptorError(
                "connection context capacity has the wrong type")
        subscription = (
            self.provider_id == "openai-subscription"
            and self.protocol == "openai_responses"
            and self.credential_ref
            == CredentialRef.openai_subscription())
        if (self.openai_request_profile is not None
                and not isinstance(
                    self.openai_request_profile,
                    openai_models.CodexModelRequestProfile)):
            raise ConnectionDescriptorError(
                "connection OpenAI request profile has the wrong type")
        if ((self.openai_request_profile is not None) != subscription):
            raise ConnectionDescriptorError(
                "OpenAI ChatGPT subscription connections require "
                "an authenticated request profile, and other connections must "
                "not contain it")
        if (self.reasoning_effort_profile is not None
                and not isinstance(
                    self.reasoning_effort_profile,
                    models.ReasoningEffortProfile)):
            raise ConnectionDescriptorError(
                "connection reasoning effort profile has the wrong type")
        if (self.reasoning_effort_profile is not None
                and not protocols.reasoning_effort_supported(
                    self.provider_id, self.protocol)):
            raise ConnectionDescriptorError(
                "connection reasoning effort profile is not valid for this "
                "provider protocol")
        if (self.reasoning_capabilities is not None
                and not isinstance(
                    self.reasoning_capabilities,
                    models.ReasoningCapabilities)):
            raise ConnectionDescriptorError(
                "connection reasoning capabilities have the wrong type")

    def to_dict(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "model": self.model,
            "chat_url": self.chat_url,
            "models_url": self.models_url,
            "protocol": self.protocol,
            "credential": (
                self.credential_ref.to_dict()
                if self.credential_ref is not None else None),
            "max_tokens": self.max_tokens,
            "anthropic_version": self.anthropic_version,
            "auth_header": self.auth_header,
            "auth_scheme": self.auth_scheme,
            "model_status": self.model_status,
            "stream": self.stream,
            "context_capacity": (
                self.context_capacity.to_dict()
                if self.context_capacity is not None else None),
            "prompt_cache": self.prompt_cache,
            "openai_request_profile": (
                self.openai_request_profile.to_dict()
                if self.openai_request_profile is not None else None),
            "reasoning_effort_profile": (
                self.reasoning_effort_profile.to_dict()
                if self.reasoning_effort_profile is not None else None),
            "reasoning_capabilities": (
                self.reasoning_capabilities.to_dict()
                if self.reasoning_capabilities is not None else None),
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ConnectionDescriptorError("saved connection must be an object")
        raw_capacity = value.get("context_capacity")
        try:
            context_capacity = (
                ContextCapacity.from_dict(raw_capacity)
                if raw_capacity is not None else None)
        except ValueError as error:
            raise ConnectionDescriptorError(str(error)) from error
        max_tokens = value.get("max_tokens", 4096)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ConnectionDescriptorError(
                "connection max_tokens must be a positive integer")
        stream = value.get("stream", False)
        if not isinstance(stream, bool):
            raise ConnectionDescriptorError(
                "connection stream must be a boolean")
        prompt_cache = value.get("prompt_cache", False)
        if not isinstance(prompt_cache, bool):
            raise ConnectionDescriptorError(
                "connection prompt_cache must be a boolean")
        # credential_env was the original on-disk representation. Read it so
        # existing chat logs remain resumable, but never write it again: a
        # CredentialRef is the sole in-memory and current on-disk identity.
        credential_env = _optional_string(
            value.get("credential_env"), "credential_env")
        if credential_env is not None and not is_credential_name(
                credential_env):
            raise ConnectionDescriptorError(
                "connection credential_env must be null or end in "
                "_KEY, _TOKEN, or _PAT")
        raw_credential = value.get("credential")
        if raw_credential is None:
            credential_ref = (
                CredentialRef.environment(credential_env)
                if credential_env is not None else None)
        else:
            try:
                credential_ref = CredentialRef.from_dict(raw_credential)
            except ValueError as error:
                raise ConnectionDescriptorError(str(error)) from error
        if (credential_env is not None
                and credential_ref
                != CredentialRef.environment(credential_env)):
            raise ConnectionDescriptorError(
                "connection credential and credential_env disagree")
        raw_openai_request_profile = value.get("openai_request_profile")
        if raw_openai_request_profile is None:
            openai_request_profile = None
        else:
            try:
                openai_request_profile = (
                    openai_models.CodexModelRequestProfile.from_dict(
                        raw_openai_request_profile))
            except openai_models.OpenAIModelProfileError as error:
                raise ConnectionDescriptorError(str(error)) from error
        raw_reasoning_effort_profile = value.get(
            "reasoning_effort_profile")
        if raw_reasoning_effort_profile is None:
            reasoning_effort_profile = None
        else:
            try:
                reasoning_effort_profile = (
                    models.ReasoningEffortProfile.from_dict(
                        raw_reasoning_effort_profile))
            except ValueError as error:
                raise ConnectionDescriptorError(str(error)) from error
        raw_capabilities = value.get("reasoning_capabilities")
        if raw_capabilities is None:
            reasoning_capabilities = None
        else:
            try:
                reasoning_capabilities = (
                    models.ReasoningCapabilities.from_dict(
                        raw_capabilities))
            except ValueError as error:
                raise ConnectionDescriptorError(str(error)) from error
        return cls(
            provider_id=_optional_string(value.get("provider_id"), "provider_id"),
            provider_name=_optional_string(value.get("provider_name"), "provider_name"),
            model=_required_string(value.get("model"), "model"),
            chat_url=_required_string(value.get("chat_url"), "chat_url"),
            models_url=_optional_string(value.get("models_url"), "models_url"),
            protocol=_required_string(value.get("protocol"), "protocol"),
            credential_ref=credential_ref,
            max_tokens=max_tokens,
            anthropic_version=_required_string(
                value.get("anthropic_version", "2023-06-01"),
                "anthropic_version"),
            auth_header=_optional_string(value.get("auth_header"), "auth_header"),
            auth_scheme=_optional_string(
                value.get("auth_scheme"), "auth_scheme"),
            model_status=_optional_string(
                value.get("model_status"), "model_status"),
            stream=stream,
            context_capacity=context_capacity,
            prompt_cache=prompt_cache,
            openai_request_profile=openai_request_profile,
            reasoning_effort_profile=reasoning_effort_profile,
            reasoning_capabilities=reasoning_capabilities,
        )


@dataclass
class RuntimeConfig:
    chat_provider: protocols.Provider
    model: str
    # Authentication is intentionally a non-secret request-time recipe.
    # Neither RuntimeConfig nor Provider retains a credential value.
    auth_spec: authentications.AuthSpec | None = None
    model_status: str | None = None
    stream: bool = False
    context_capacity: ContextCapacity | None = None
    reasoning_effort_profile: (
        models.ReasoningEffortProfile | None) = None
    reasoning_capabilities: (
        models.ReasoningCapabilities | None) = None

    def credential_destinations(self) -> list[str]:
        """Concrete request destinations displayed/approved for this connection."""
        if self.auth_spec is None:
            return []
        from . import provider_controls
        destinations = [self.chat_provider.chat_url,
                        *self.chat_provider.model_urls]
        destinations.extend(provider_controls.account_destinations(
            provider_controls.ControlContext(config=self)))
        return list(dict.fromkeys(destinations))


def _int_setting(name, default, credentials):
    value = credentials.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _bool_setting(name, default, credentials):
    value = credentials.get(name)
    if not value:
        return default
    normalized = value.strip().lower()
    if normalized in ["1", "true", "yes", "on"]:
        return True
    if normalized in ["0", "false", "no", "off"]:
        return False
    raise ValueError(
        f"{name} must be one of: 1, 0, true, false, yes, no, on, off")


def _auth_spec(provider_kind, credential_ref, request_urls,
               auth_header=None, auth_scheme=None):
    if credential_ref is None:
        if auth_scheme is not None:
            raise ValueError(
                "authentication scheme requires a credential reference")
        return None
    if auth_scheme is not None:
        scheme = auth_scheme
    elif credential_ref.kind == "openai-subscription":
        scheme = "openai-subscription"
    elif auth_header:
        scheme = "custom"
    elif provider_kind == protocols.ANTHROPIC_MESSAGES:
        scheme = "anthropic"
    else:
        scheme = "bearer"
    if scheme not in {
            "anthropic", "bearer", "custom", "openai-subscription"}:
        raise ValueError(f"unknown authentication scheme {scheme!r}")
    if ((credential_ref.kind == "openai-subscription")
            != (scheme == "openai-subscription")):
        raise ValueError(
            "OpenAI subscription credentials require the "
            "openai-subscription authentication scheme")
    if scheme == "custom" and not auth_header:
        raise ValueError(
            "custom authentication requires an authentication header")
    return authentications.AuthSpec(
        credential=credential_ref,
        scheme=scheme,
        header_name=auth_header if scheme == "custom" else None,
        # The subscription scheme is bound to the canonical Codex URLs here
        # rather than to whatever endpoints the provider happens to expose, so
        # a provider-configuration change cannot widen credential authority.
        authorized_urls=(
            authentications.OPENAI_CHATGPT_CODEX_URLS
            if scheme == "openai-subscription"
            else frozenset(url for url in request_urls if url is not None)
        ),
    )


def make_runtime_config(
        url, provider_kind, *, model="", models_url=None,
        max_tokens=4096, anthropic_version="2023-06-01",
        auth_header=None, auth_scheme=None, provider_id=None,
        provider_name=None, credential_ref=None, model_status=None,
        stream=False, prompt_cache=False, openai_request_profile=None,
        reasoning_effort_profile=None, context_capacity=None,
        reasoning_capabilities=None):
    """Build a RuntimeConfig (and its Provider) from explicit parameters.

    The single place a production Provider is constructed. Startup reads the
    environment into these parameters; a mid-session /model change calls
    reinstall_provider() which reuses this to rebuild the Provider from the
    current config plus per-model overrides.
    """
    if (context_capacity is not None
            and not isinstance(context_capacity, ContextCapacity)):
        raise ValueError("context_capacity must be ContextCapacity or null")
    if (reasoning_effort_profile is not None
            and not isinstance(
                reasoning_effort_profile,
                models.ReasoningEffortProfile)):
        raise ValueError(
            "reasoning_effort_profile must be ReasoningEffortProfile or null")
    if (reasoning_capabilities is not None
            and not isinstance(
                reasoning_capabilities,
                models.ReasoningCapabilities)):
        raise ValueError(
            "reasoning_capabilities must be ReasoningCapabilities or null")
    chat_provider = protocols.make_provider(
        url,
        provider=provider_kind,
        models_url=models_url,
        max_tokens=max_tokens,
        anthropic_version=anthropic_version,
        provider_id=provider_id,
        provider_name=provider_name,
        prompt_cache=prompt_cache,
        openai_request_profile=openai_request_profile,
        reasoning_field=(
            reasoning_capabilities.interleaved_field
            if reasoning_capabilities is not None else None),
    )
    if (reasoning_effort_profile is not None
            and not protocols.reasoning_effort_supported(
                provider_id, chat_provider.kind)):
        raise ValueError(
            "reasoning effort profile is not valid for this provider "
            "protocol")
    auth_spec = _auth_spec(
        provider_kind,
        credential_ref,
        [chat_provider.chat_url, *chat_provider.model_urls],
        auth_header,
        auth_scheme,
    )
    if (auth_spec is not None
            and auth_spec.scheme == "openai-subscription"
            and provider_id
            != models.OPENAI_SUBSCRIPTION_PROVIDER_ID):
        raise ValueError(
            "OpenAI subscription authentication requires the "
            "openai-subscription provider identity")
    authentications.validate_authorization_target(
        auth_spec, chat_provider.chat_url)
    return RuntimeConfig(
        chat_provider=chat_provider,
        model=model,
        auth_spec=auth_spec,
        model_status=model_status,
        stream=stream,
        reasoning_effort_profile=reasoning_effort_profile,
        context_capacity=context_capacity,
        reasoning_capabilities=reasoning_capabilities,
    )


def _context_capacity_setting(credentials, fallback=None):
    value = credentials.get("LOKI_CONTEXT_WINDOW")
    if not value:
        return fallback
    try:
        return ContextCapacity(int(value), "configured")
    except ValueError as error:
        raise ValueError(
            "LOKI_CONTEXT_WINDOW must be a positive integer") from error


def _available_environment_credential(credentials, names):
    name = credentials.first_available_name(names)
    return (
        authentications.CredentialRef.environment(name)
        if name is not None else None)


def _explicit_credential_ref(credentials):
    encoded = credentials.get("LOKI_CREDENTIAL_REF")
    if not encoded:
        return _available_environment_credential(
            credentials, ["LOKI_API_KEY"])
    try:
        credential = authentications.CredentialRef.decode(encoded)
    except ValueError as error:
        raise ValueError(f"invalid LOKI_CREDENTIAL_REF: {error}") from error
    if not credentials.has_ref(credential):
        raise ValueError(
            "LOKI_CREDENTIAL_REF names an unavailable credential: "
            f"{credential.encode()}")
    return credential


def _openai_request_profile_setting(credentials, subscription):
    encoded = credentials.get("LOKI_OPENAI_REQUEST_PROFILE")
    if not encoded:
        if subscription:
            raise ValueError(
                "OpenAI subscription configuration requires "
                "LOKI_OPENAI_REQUEST_PROFILE; select the model with /model")
        return None
    if not subscription:
        raise ValueError(
            "LOKI_OPENAI_REQUEST_PROFILE requires an OpenAI subscription "
            "credential")
    try:
        value = json.loads(encoded)
        return openai_models.CodexModelRequestProfile.from_dict(value)
    except ValueError as error:
        raise ValueError(
            f"invalid LOKI_OPENAI_REQUEST_PROFILE: {error}") from error


def _reasoning_effort_profile_setting(credentials):
    encoded = credentials.get("LOKI_REASONING_EFFORT_PROFILE")
    if not encoded:
        return None
    try:
        value = json.loads(encoded)
        return models.ReasoningEffortProfile.from_dict(value)
    except ValueError as error:
        raise ValueError(
            f"invalid LOKI_REASONING_EFFORT_PROFILE: {error}") from error


def build_config_from_env(
        environ=os.environ,
        credentials: CredentialStore | CredentialInventory | None = None):
    """Build an explicit/default runtime config from captured startup values."""
    if credentials is None:
        credentials = CredentialStore.capture(environ)

    # Only Loki's namespace configures Loki. Generic SDK variables such as
    # OPENAI_API_BASE and ANTHROPIC_BASE_URL may belong to unrelated programs
    # sharing this environment and must not silently select Loki's endpoint.
    config_url = credentials.get("LOKI_API_BASE")
    if not config_url:
        raise ValueError(
            "API endpoint missing; set LOKI_API_BASE or select one with /model")
    provider_override = credentials.get("LOKI_PROVIDER")
    config_provider_kind = protocols.resolve_protocol(config_url, provider_override or "auto")

    if config_provider_kind == protocols.DUMMY:
        # No-op provider for testing: no network, no API key, fake URL.
        config_model = credentials.get("LOKI_MODEL") or "dummy"
        return make_runtime_config(
            config_url,
            config_provider_kind,
            model=config_model,
            max_tokens=_int_setting("LOKI_MAX_TOKENS", 4096, credentials),
            anthropic_version=credentials.get(
                "LOKI_ANTHROPIC_VERSION", "2023-06-01"),
            provider_name="Explicit LOKI_* connection",
            stream=_bool_setting("LOKI_STREAM", False, credentials),
            prompt_cache=False,
            context_capacity=_context_capacity_setting(credentials),
        )

    # Never infer credential ownership from the wire protocol. An explicitly
    # configured endpoint uses LOKI_API_KEY when present and otherwise sends
    # no authentication header.
    credential_ref = _explicit_credential_ref(credentials)
    subscription = (
        credential_ref is not None
        and credential_ref.kind == "openai-subscription")
    openai_request_profile = _openai_request_profile_setting(
        credentials, subscription)
    reasoning_effort_profile = _reasoning_effort_profile_setting(
        credentials)
    delegated_provider_id = credentials.get("LOKI_PROVIDER_ID") or None
    config_provider_id = (
        models.OPENAI_SUBSCRIPTION_PROVIDER_ID
        if subscription else delegated_provider_id)

    config_model = credentials.get("LOKI_MODEL") or ""
    return make_runtime_config(
        config_url,
        config_provider_kind,
        model=config_model,
        models_url=credentials.get("LOKI_MODELS_URL") or None,
        max_tokens=_int_setting("LOKI_MAX_TOKENS", 4096, credentials),
        anthropic_version=credentials.get(
            "LOKI_ANTHROPIC_VERSION", "2023-06-01"),
        auth_header=credentials.get("LOKI_AUTH_HEADER") or None,
        auth_scheme=credentials.get("LOKI_AUTH_SCHEME") or None,
        provider_id=config_provider_id,
        provider_name=(
            "OpenAI ChatGPT subscription"
            if subscription else "Explicit LOKI_* connection"),
        credential_ref=credential_ref,
        stream=_bool_setting("LOKI_STREAM", False, credentials),
        prompt_cache=_bool_setting(
            "LOKI_PROMPT_CACHE",
            urllib.parse.urlparse(config_url).hostname
            == "api.anthropic.com",
            credentials,
        ),
        openai_request_profile=openai_request_profile,
        reasoning_effort_profile=reasoning_effort_profile,
        context_capacity=_context_capacity_setting(credentials),
    )


def explicit_api_base_configured(credentials) -> bool:
    return bool(credentials.get("LOKI_API_BASE"))


def explicit_connection_option(
        credentials
) -> models.ExplicitConnectionOption | None:
    """Return a selectable captured LOKI_* connection when it is complete."""
    try:
        config = build_config_from_env(credentials=credentials)
    except (protocols.ProtocolError, ValueError):
        return None
    if not config.model:
        return None
    return models.ExplicitConnectionOption(
        model=config.model,
        api_url=config.chat_provider.input_url,
        protocol=config.chat_provider.kind,
    )


def config_from_connection_descriptor(
        descriptor: ConnectionDescriptor,
        credentials) -> RuntimeConfig:
    credential_ref = descriptor.credential_ref
    if (credential_ref is not None
            and not credentials.has_ref(credential_ref)):
        raise ValueError(
            "saved connection requires missing "
            f"{credential_ref.encode()}")

    provider_kind = credentials.get("LOKI_PROVIDER") or descriptor.protocol
    resolved_provider_kind = protocols.resolve_protocol(
        descriptor.chat_url, provider_kind)
    configured_model = credentials.get("LOKI_MODEL")
    config_model = configured_model or descriptor.model
    model_status = (
        descriptor.model_status
        if not configured_model or configured_model == descriptor.model
        else None)
    openai_request_profile = (
        descriptor.openai_request_profile
        if not configured_model or configured_model == descriptor.model
        else None)
    # Connection-bound metadata travels only with the same connection:
    # a protocol override builds a different connection, whose facts the
    # saved descriptor cannot vouch for.
    reasoning_effort_profile = (
        descriptor.reasoning_effort_profile
        if ((not configured_model or configured_model == descriptor.model)
            and resolved_provider_kind == descriptor.protocol)
        else None)
    max_tokens = (
        _int_setting("LOKI_MAX_TOKENS", descriptor.max_tokens, credentials)
        if credentials.get("LOKI_MAX_TOKENS") else descriptor.max_tokens)
    anthropic_version = (
        credentials.get("LOKI_ANTHROPIC_VERSION")
        or descriptor.anthropic_version)
    auth_header = (
        credentials.get("LOKI_AUTH_HEADER")
        if credentials.get("LOKI_AUTH_HEADER")
        else descriptor.auth_header)
    configured_auth_scheme = credentials.get("LOKI_AUTH_SCHEME") or None
    auth_scheme = (
        configured_auth_scheme
        if configured_auth_scheme is not None
        else (None
              if credentials.get("LOKI_AUTH_HEADER")
              else descriptor.auth_scheme))
    stream = (
        _bool_setting("LOKI_STREAM", descriptor.stream, credentials)
        if credentials.get("LOKI_STREAM") else descriptor.stream)
    prompt_cache = (
        _bool_setting(
            "LOKI_PROMPT_CACHE", descriptor.prompt_cache, credentials)
        if credentials.get("LOKI_PROMPT_CACHE")
        else descriptor.prompt_cache)
    return make_runtime_config(
        # Restore the concrete endpoint that was actually used, rather than
        # deriving it again from a catalog base URL.
        descriptor.chat_url,
        provider_kind,
        model=config_model,
        models_url=descriptor.models_url,
        max_tokens=max_tokens,
        anthropic_version=anthropic_version,
        auth_header=auth_header,
        auth_scheme=auth_scheme,
        provider_id=descriptor.provider_id,
        provider_name=descriptor.provider_name,
        credential_ref=credential_ref,
        model_status=model_status,
        stream=stream,
        prompt_cache=prompt_cache,
        openai_request_profile=openai_request_profile,
        reasoning_effort_profile=reasoning_effort_profile,
        context_capacity=_context_capacity_setting(
            credentials,
            descriptor.context_capacity
            if config_model == descriptor.model
            and resolved_provider_kind == descriptor.protocol else None),
    )


def _is_openai_subscription_connection(descriptor):
    return (
        descriptor.provider_id
        == models.OPENAI_SUBSCRIPTION_PROVIDER_ID
        and descriptor.protocol == protocols.OPENAI_RESPONSES
        and descriptor.credential_ref
        == authentications.CredentialRef.openai_subscription()
    )


def reconcile_connection_descriptor(descriptor, catalog):
    """Replace saved subscription model data with fresh exact-slug data."""
    if not _is_openai_subscription_connection(descriptor):
        return descriptor
    profile = models.reconcile_openai_subscription_profile(
        descriptor.model,
        descriptor.openai_request_profile,
        catalog,
    )
    reasoning_effort_profile = (
        models.reconcile_openai_subscription_reasoning_effort_profile(
            descriptor.model,
            descriptor.reasoning_effort_profile,
            catalog,
        ))
    capacity = descriptor.context_capacity
    provider = catalog.get(models.OPENAI_SUBSCRIPTION_PROVIDER_ID)
    if (provider is not None
            and (capacity is None or capacity.source != "configured")):
        capacity = models.context_capacity(
            provider, provider["models"][descriptor.model])
    return replace(
        descriptor,
        openai_request_profile=profile,
        reasoning_effort_profile=reasoning_effort_profile,
        context_capacity=capacity,
    )


def config_from_modelsdev_selection(
        provider_id: str,
        provider_entry: dict,
        model_entry: dict,
        credentials) -> RuntimeConfig:
    provider_entry = models.effective_provider(provider_entry, model_entry)
    access = models.provider_access(provider_entry, credentials)
    if access is None:
        raise ValueError(
            f"provider {provider_id!r} is not available from startup credentials")
    # Construction is pure. Approval belongs to the supervisor's lease path,
    # not a tools-writable file consulted while building an otherwise valid
    # configuration. Controllers confirm the concrete destinations before use.
    selected_model = model_entry.get("id") or model_entry.get("name")
    if not selected_model:
        raise ValueError(f"provider {provider_id!r} returned a model without an id")
    model_status = model_entry.get("status")
    if not isinstance(model_status, str):
        model_status = None
    return make_runtime_config(
        access.api_url,
        access.protocol,
        model=selected_model,
        models_url=access.models_url,
        max_tokens=_int_setting("LOKI_MAX_TOKENS", 4096, credentials),
        anthropic_version=credentials.get(
            "LOKI_ANTHROPIC_VERSION", "2023-06-01"),
        auth_header=credentials.get("LOKI_AUTH_HEADER") or None,
        auth_scheme=credentials.get("LOKI_AUTH_SCHEME") or None,
        provider_id=provider_id,
        provider_name=models.provider_display_name(
            provider_id, provider_entry),
        credential_ref=access.credential_ref,
        model_status=model_status,
        stream=_bool_setting(
            "LOKI_STREAM",
            provider_id == models.OPENAI_SUBSCRIPTION_PROVIDER_ID,
            credentials,
        ),
        prompt_cache=_bool_setting(
            "LOKI_PROMPT_CACHE",
            provider_id == "anthropic",
            credentials,
        ),
        openai_request_profile=models.openai_request_profile(
            provider_entry, model_entry),
        reasoning_effort_profile=models.reasoning_effort_profile(
            provider_id, provider_entry, model_entry),
        context_capacity=_context_capacity_setting(
            credentials, models.context_capacity(provider_entry, model_entry)),
    )


def connection_descriptor_from_config(
        config: RuntimeConfig) -> ConnectionDescriptor | None:
    if config.chat_provider.kind == protocols.DUMMY:
        return None
    auth_spec = config.auth_spec
    credential_ref = (
        auth_spec.credential if auth_spec is not None else None)
    provider = config.chat_provider
    return ConnectionDescriptor(
        provider_id=provider.provider_id,
        provider_name=provider.provider_name,
        model=config.model,
        chat_url=provider.chat_url,
        models_url=provider.models_url,
        protocol=provider.kind,
        credential_ref=credential_ref,
        max_tokens=provider.max_tokens,
        anthropic_version=provider.headers.get(
            "anthropic-version", "2023-06-01"),
        auth_header=(
            auth_spec.header_name
            if auth_spec is not None
            and auth_spec.scheme == "custom" else None),
        auth_scheme=auth_spec.scheme if auth_spec is not None else None,
        model_status=config.model_status,
        stream=config.stream,
        prompt_cache=provider.prompt_cache,
        openai_request_profile=provider.openai_request_profile,
        reasoning_effort_profile=config.reasoning_effort_profile,
        context_capacity=config.context_capacity,
    )
