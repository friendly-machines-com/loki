"""Serializable, non-secret provider connection descriptions."""

from dataclasses import dataclass

from . import openai_models
from . import reasonings
from .authentications import CredentialRef
from .credentials import is_credential_name


class ConnectionDescriptorError(ValueError):
    pass


def connection_display_fields(
        descriptor: "ConnectionDescriptor") -> list[tuple[str, str]]:
    """Return the complete non-secret connection facts shown for approval."""
    provider = (
        descriptor.provider_name or descriptor.provider_id or "custom")
    fields = [
        ("Provider", provider),
        ("Model", descriptor.model),
        ("Chat endpoint", descriptor.chat_url),
    ]
    if descriptor.models_url:
        fields.append(("Models endpoint", descriptor.models_url))
    if descriptor.credential_ref is None:
        fields.append(("Authentication", "none"))
    else:
        credential = descriptor.credential_ref
        fields.append((
            "Credential",
            credential.name if credential.kind == "env"
            else credential.encode(),
        ))
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
    prompt_cache: bool = False
    openai_request_profile: (
        openai_models.CodexModelRequestProfile | None) = None
    reasoning_effort_profile: (
        reasonings.ReasoningEffortProfile | None) = None

    def __post_init__(self):
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
                    reasonings.ReasoningEffortProfile)):
            raise ConnectionDescriptorError(
                "connection reasoning effort profile has the wrong type")
        if (self.reasoning_effort_profile is not None
                and not reasonings.wire_protocol_supported(
                    self.provider_id, self.protocol)):
            raise ConnectionDescriptorError(
                "connection reasoning effort profile is not valid for this "
                "provider protocol")
        if subscription:
            try:
                reasonings.validate_codex_effort_profile(
                    self.reasoning_effort_profile,
                    supports_reasoning=(
                        self.openai_request_profile
                        .supports_reasoning_summaries),
                    request_default=(
                        self.openai_request_profile
                        .default_reasoning_level),
                )
            except reasonings.ReasoningEffortError as error:
                raise ConnectionDescriptorError(str(error)) from error

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
            "prompt_cache": self.prompt_cache,
            "openai_request_profile": (
                self.openai_request_profile.to_dict()
                if self.openai_request_profile is not None else None),
            "reasoning_effort_profile": (
                self.reasoning_effort_profile.to_dict()
                if self.reasoning_effort_profile is not None else None),
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ConnectionDescriptorError("saved connection must be an object")
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
                    reasonings.ReasoningEffortProfile.from_dict(
                        raw_reasoning_effort_profile))
            except reasonings.ReasoningEffortError as error:
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
            prompt_cache=prompt_cache,
            openai_request_profile=openai_request_profile,
            reasoning_effort_profile=reasoning_effort_profile,
        )
