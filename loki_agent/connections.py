"""Serializable, non-secret provider connection descriptions."""

from dataclasses import dataclass

from .credentials import is_credential_name


class ConnectionDescriptorError(ValueError):
    pass


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
    api_url: str
    chat_url: str
    models_url: str | None
    protocol: str
    credential_env: str
    max_tokens: int = 4096
    anthropic_version: str = "2023-06-01"
    auth_header: str | None = None

    def to_dict(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "model": self.model,
            "api_url": self.api_url,
            "chat_url": self.chat_url,
            "models_url": self.models_url,
            "protocol": self.protocol,
            "credential_env": self.credential_env,
            "max_tokens": self.max_tokens,
            "anthropic_version": self.anthropic_version,
            "auth_header": self.auth_header,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ConnectionDescriptorError("saved connection must be an object")
        max_tokens = value.get("max_tokens", 4096)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ConnectionDescriptorError(
                "connection max_tokens must be a positive integer")
        credential_env = _required_string(
            value.get("credential_env"), "credential_env")
        if not is_credential_name(credential_env):
            raise ConnectionDescriptorError(
                "connection credential_env must end in _KEY, _TOKEN, or _PAT")
        return cls(
            provider_id=_optional_string(value.get("provider_id"), "provider_id"),
            provider_name=_optional_string(value.get("provider_name"), "provider_name"),
            model=_required_string(value.get("model"), "model"),
            api_url=_required_string(value.get("api_url"), "api_url"),
            chat_url=_required_string(value.get("chat_url"), "chat_url"),
            models_url=_optional_string(value.get("models_url"), "models_url"),
            protocol=_required_string(value.get("protocol"), "protocol"),
            credential_env=credential_env,
            max_tokens=max_tokens,
            anthropic_version=_required_string(
                value.get("anthropic_version", "2023-06-01"),
                "anthropic_version"),
            auth_header=_optional_string(value.get("auth_header"), "auth_header"),
        )
