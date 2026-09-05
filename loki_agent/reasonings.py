"""Model-specific reasoning-effort capabilities and session preferences.

An effort profile describes one exact provider/model leaf.  It is inert data:
it may constrain a request parameter, but it never selects an endpoint,
credential, header, or wire protocol.  The user's preference has a different
lifetime and is stored on the conversation's Session.
"""

from __future__ import annotations

from dataclasses import dataclass


class ReasoningEffortError(ValueError):
    pass


_DISPLAY_NAMES = {
    "none": "None",
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "Extra high",
    "max": "Maximum",
    "ultra": "Ultra",
}
_CODEX_APPLICATION_EFFORTS = frozenset({"ultra"})

# Catalog values do not carry their raw HTTP field path. Register only
# provider/protocol surfaces whose effort request contract Loki implements.
# This is also used while decoding saved connections so an untrusted chat log
# cannot manufacture support for an arbitrary compatible endpoint.
_WIRE_PROTOCOLS = {
    "anthropic": "anthropic_messages",
    "openai": "openai_responses",
    "openai-subscription": "openai_responses",
    "openrouter": "openai_chat",
    "zai": "openai_chat",
    "zai-coding-plan": "openai_chat",
    "zhipuai": "openai_chat",
    "zhipuai-coding-plan": "openai_chat",
}
_ZAI_PROVIDER_IDS = frozenset({
    "zai",
    "zai-coding-plan",
    "zhipuai",
    "zhipuai-coding-plan",
})


def _effort_value(value, field_name):
    if not isinstance(value, str) or not value:
        raise ReasoningEffortError(
            f"{field_name} must be a non-empty string")
    return value


def preference_from_value(value):
    """Validate a persistent or delegated session preference."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ReasoningEffortError(
            "reasoning effort preference must be a string or null")
    return _effort_value(value, "reasoning effort preference")


def display_name(value: str) -> str:
    value = _effort_value(value, "reasoning effort")
    return _DISPLAY_NAMES.get(value, value)


def wire_protocol_supported(provider_id, protocol) -> bool:
    return _WIRE_PROTOCOLS.get(provider_id) == protocol


def is_zai_provider(provider_id) -> bool:
    """Whether effort uses Z.AI's paired thinking/effort parameters."""
    return provider_id in _ZAI_PROVIDER_IDS


@dataclass(frozen=True)
class ReasoningEffortOption:
    value: str
    description: str | None = None

    def __post_init__(self):
        _effort_value(self.value, "reasoning effort option")
        if self.description is not None:
            if not isinstance(self.description, str) or not self.description:
                raise ReasoningEffortError(
                    "reasoning effort description must be a non-empty "
                    "string or null")

    def to_dict(self):
        return {
            "value": self.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ReasoningEffortError(
                "reasoning effort option must be an object")
        if set(value) != {"value", "description"}:
            raise ReasoningEffortError(
                "reasoning effort option has unexpected or missing fields")
        return cls(
            value=value["value"],
            description=value["description"],
        )


@dataclass(frozen=True)
class ReasoningEffortProfile:
    """Allowed effort choices and optional default for one model leaf."""

    options: tuple[ReasoningEffortOption, ...]
    default_value: str | None = None

    def __post_init__(self):
        if not isinstance(self.options, tuple) or not self.options:
            raise ReasoningEffortError(
                "reasoning effort profile requires at least one option")
        if not all(
                isinstance(option, ReasoningEffortOption)
                for option in self.options
        ):
            raise ReasoningEffortError(
                "reasoning effort profile options have the wrong type")
        values = tuple(option.value for option in self.options)
        if len(set(values)) != len(values):
            raise ReasoningEffortError(
                "reasoning effort profile contains duplicate values")
        if self.default_value is not None:
            _effort_value(
                self.default_value, "reasoning effort profile default")

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(option.value for option in self.options)

    def supports(self, preference: str | None) -> bool:
        return preference is not None and preference in self.values

    def option(self, value: str) -> ReasoningEffortOption | None:
        return next(
            (option for option in self.options if option.value == value),
            None,
        )

    def to_dict(self):
        return {
            "options": [option.to_dict() for option in self.options],
            "default_value": self.default_value,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ReasoningEffortError(
                "reasoning effort profile must be an object")
        if set(value) != {"options", "default_value"}:
            raise ReasoningEffortError(
                "reasoning effort profile has unexpected or missing fields")
        raw_options = value["options"]
        if not isinstance(raw_options, list):
            raise ReasoningEffortError(
                "reasoning effort profile options must be an array")
        return cls(
            options=tuple(
                ReasoningEffortOption.from_dict(option)
                for option in raw_options
            ),
            default_value=value["default_value"],
        )


def from_modelsdev_model(model_entry) -> ReasoningEffortProfile | None:
    """Extract an effort profile from one models.dev provider/model leaf."""
    if not isinstance(model_entry, dict):
        raise ReasoningEffortError("models.dev model must be an object")
    raw_options = model_entry.get("reasoning_options")
    if raw_options is None:
        return None
    if not isinstance(raw_options, list):
        raise ReasoningEffortError(
            "models.dev reasoning_options must be an array")
    effort_entries = [
        option for option in raw_options
        if isinstance(option, dict) and option.get("type") == "effort"
    ]
    if not effort_entries:
        return None
    if len(effort_entries) != 1:
        raise ReasoningEffortError(
            "models.dev model has multiple reasoning effort entries")
    values = effort_entries[0].get("values")
    if not isinstance(values, list) or not values:
        raise ReasoningEffortError(
            "models.dev reasoning effort values must be a non-empty array")
    return ReasoningEffortProfile(
        options=tuple(
            ReasoningEffortOption(
                _effort_value(value, "models.dev reasoning effort value"))
            for value in values
        ),
    )


def from_codex_catalog_model(value) -> ReasoningEffortProfile | None:
    """Extract choices from one authenticated ChatGPT Codex catalog model."""
    if not isinstance(value, dict):
        raise ReasoningEffortError(
            "OpenAI Codex model must be an object")
    if not value.get("supports_reasoning_summaries", False):
        # Loki's pinned Codex request contract gates the complete reasoning
        # object on this capability.
        return None
    levels = value.get("supported_reasoning_levels", [])
    if not isinstance(levels, list):
        raise ReasoningEffortError(
            "OpenAI Codex supported_reasoning_levels must be an array")
    if not levels:
        return None
    options = []
    for level in levels:
        if not isinstance(level, dict):
            raise ReasoningEffortError(
                "OpenAI Codex reasoning level must be an object")
        raw_effort = level.get("effort")
        if not isinstance(raw_effort, str):
            raise ReasoningEffortError(
                "OpenAI Codex reasoning effort must be a string")
        effort = _effort_value(
            raw_effort, "OpenAI Codex reasoning effort")
        # Codex's ``ultra`` is an application mode: it selects proactive
        # multi-agent behavior in addition to sending ``max`` to inference.
        # Loki does not implement that mode, so it must not advertise
        # ``ultra`` as an ordinary LLM effort or silently rename it.
        if effort in _CODEX_APPLICATION_EFFORTS:
            continue
        description = level.get("description")
        if description is not None and (
                not isinstance(description, str) or not description):
            description = None
        options.append(ReasoningEffortOption(effort, description))

    raw_default = value.get("default_reasoning_level")
    default = (
        None if raw_default is None else _effort_value(
            raw_default, "OpenAI Codex default reasoning level",
        )
    )
    if not options:
        return None
    return ReasoningEffortProfile(tuple(options), default)


def validate_codex_effort_profile(
        profile: ReasoningEffortProfile | None, *,
        supports_reasoning: bool,
        request_default: str | None,
) -> None:
    """Validate a saved/delegated selector against its request contract."""
    if profile is None:
        return
    if not supports_reasoning:
        raise ReasoningEffortError(
            "OpenAI Codex reasoning effort profile requires reasoning "
            "support")
    if profile.default_value != request_default:
        raise ReasoningEffortError(
            "OpenAI Codex reasoning effort and request profile defaults "
            "disagree")
    if any(
            option.value in _CODEX_APPLICATION_EFFORTS
            for option in profile.options
    ):
        raise ReasoningEffortError(
            "OpenAI Codex reasoning effort profile contains an unsupported "
            "application mode")


def effective_effort(
        profile: ReasoningEffortProfile | None,
        preference: str | None,
) -> str | None:
    """Return the explicit wire effort, or null for model/provider default."""
    preference = preference_from_value(preference)
    if profile is None or not profile.supports(preference):
        return None
    return preference


def default_option_name(
        profile: ReasoningEffortProfile,
        preference: str | None,
) -> str:
    name = "Model default"
    if profile.default_value is not None:
        name += f" ({display_name(profile.default_value)})"
    if preference is not None and not profile.supports(preference):
        name += f" - preferred {display_name(preference)} is unavailable"
    return name
