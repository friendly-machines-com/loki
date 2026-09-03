"""Request-relevant fields from the authenticated ChatGPT model catalog.

The complete catalog record belongs to the Codex application, not to Loki.
Persisting it would copy foreign prompts and unrelated application policy into
chat logs and child-process configuration.  ``CodexModelRequestProfile`` is
therefore deliberately smaller than Codex's ``ModelInfo``: it contains only
the fields Loki currently consumes while constructing a Responses request.

The field inventory below is from ``ModelInfo`` in
``codex-rs/protocol/src/openai_models.rs`` at ``rust-v0.144.1``:

* discovery-only in Loki: ``slug``, ``display_name``, ``visibility``,
  ``supported_reasoning_levels``, and ``input_modalities``;
* retained in the request profile: ``use_responses_lite``,
  ``supports_parallel_tool_calls``, ``supports_reasoning_summaries``,
  ``default_reasoning_level``, ``default_reasoning_summary``,
  ``support_verbosity``, and ``default_verbosity``;
* known but unused: ``description``, ``shell_type``, ``supported_in_api``,
  ``priority``, ``additional_speed_tiers``, ``service_tiers``,
  ``default_service_tier``, ``availability_nux``, ``upgrade``,
  ``base_instructions``, ``model_messages``,
  ``include_skills_usage_instructions``, ``apply_patch_tool_type``,
  ``web_search_tool_type``, ``truncation_policy``,
  ``supports_image_detail_original``, ``context_window``,
  ``max_context_window``, ``auto_compact_token_limit``, ``comp_hash``,
  ``effective_context_window_percent``, ``experimental_supported_tools``,
  ``supports_search_tool``, ``auto_review_model_override``, ``tool_mode``,
  and ``multi_agent_version``. ``used_fallback_model_metadata`` is an
  internal Codex field and is not received from the service.

The unused fields describe Codex UI, prompts, compaction, or Codex's own tool
implementations.  In particular, ``tool_mode`` is not a Responses parameter:
Loki always exposes its own direct function tools.  If Loki implements a
corresponding feature later, the same change must add the field's parser,
consumer, persistence semantics, and request tests.  Merely receiving a field
from the catalog is not a reason to retain it.
"""

from dataclasses import dataclass


class OpenAIModelProfileError(ValueError):
    pass


_REASONING_SUMMARIES = {"auto", "concise", "detailed", "none"}
_VERBOSITIES = {"low", "medium", "high"}


def _boolean(value, field_name, *, default=None):
    field = value.get(field_name, default)
    if not isinstance(field, bool):
        raise OpenAIModelProfileError(
            f"OpenAI Codex model {field_name} must be a boolean")
    return field


def _optional_string(field, field_name):
    if field is None:
        return None
    if not isinstance(field, str) or not field:
        raise OpenAIModelProfileError(
            f"OpenAI Codex model {field_name} must be a non-empty string "
            "or null")
    return field


def _choice(field, field_name, choices):
    if field not in choices:
        expected = ", ".join(sorted(choices))
        raise OpenAIModelProfileError(
            f"OpenAI Codex model {field_name} must be one of {expected}")
    return field


@dataclass(frozen=True)
class CodexModelRequestProfile:
    """Authenticated per-model inputs to Loki's Responses request builder."""

    use_responses_lite: bool
    supports_parallel_tool_calls: bool
    supports_reasoning_summaries: bool
    default_reasoning_level: str | None
    default_reasoning_summary: str | None
    supports_verbosity: bool
    default_verbosity: str | None

    @classmethod
    def from_catalog_model(cls, value):
        """Extract only request fields from one authenticated catalog model."""
        if not isinstance(value, dict):
            raise OpenAIModelProfileError(
                "OpenAI Codex model must be an object")

        supports_reasoning = _boolean(
            value, "supports_reasoning_summaries", default=False)
        if supports_reasoning:
            reasoning_level = _optional_string(
                value.get("default_reasoning_level"),
                "default_reasoning_level",
            )
            reasoning_summary = _choice(
                value.get("default_reasoning_summary", "auto"),
                "default_reasoning_summary",
                _REASONING_SUMMARIES,
            )
        else:
            # The value cannot affect a request while summaries are disabled.
            # Do not let an irrelevant malformed catalog value reject a model.
            reasoning_level = None
            reasoning_summary = None

        supports_verbosity = _boolean(
            value, "support_verbosity", default=False)
        if supports_verbosity:
            verbosity = _optional_string(
                value.get("default_verbosity"), "default_verbosity")
            if verbosity is not None:
                verbosity = _choice(
                    verbosity, "default_verbosity", _VERBOSITIES)
        else:
            # As above, unsupported configuration is deliberately ignored.
            verbosity = None

        return cls(
            use_responses_lite=_boolean(
                value, "use_responses_lite", default=False),
            supports_parallel_tool_calls=_boolean(
                value, "supports_parallel_tool_calls", default=False),
            supports_reasoning_summaries=supports_reasoning,
            default_reasoning_level=reasoning_level,
            default_reasoning_summary=reasoning_summary,
            supports_verbosity=supports_verbosity,
            default_verbosity=verbosity,
        )

    @classmethod
    def from_dict(cls, value):
        """Decode the complete compact profile stored by Loki itself."""
        if not isinstance(value, dict):
            raise OpenAIModelProfileError(
                "OpenAI Codex request profile must be an object")

        required = {
            "use_responses_lite",
            "supports_parallel_tool_calls",
            "supports_reasoning_summaries",
            "default_reasoning_level",
            "default_reasoning_summary",
            "supports_verbosity",
            "default_verbosity",
        }
        if set(value) != required:
            raise OpenAIModelProfileError(
                "OpenAI Codex request profile has unexpected or missing "
                "fields")

        profile = cls.from_catalog_model({
            "use_responses_lite": value["use_responses_lite"],
            "supports_parallel_tool_calls":
                value["supports_parallel_tool_calls"],
            "supports_reasoning_summaries":
                value["supports_reasoning_summaries"],
            "default_reasoning_level": value["default_reasoning_level"],
            "default_reasoning_summary":
                value["default_reasoning_summary"],
            "support_verbosity": value["supports_verbosity"],
            "default_verbosity": value["default_verbosity"],
        })
        if (not profile.supports_reasoning_summaries
                and (value["default_reasoning_level"] is not None
                     or value["default_reasoning_summary"] is not None)):
            raise OpenAIModelProfileError(
                "OpenAI Codex request profile has disabled reasoning values")
        if (not profile.supports_verbosity
                and value["default_verbosity"] is not None):
            raise OpenAIModelProfileError(
                "OpenAI Codex request profile has disabled verbosity values")
        if (profile.supports_reasoning_summaries
                and profile.default_reasoning_summary is None):
            raise OpenAIModelProfileError(
                "OpenAI Codex request profile requires a reasoning summary "
                "default when reasoning summaries are supported")
        return profile

    def to_dict(self):
        return {
            "use_responses_lite": self.use_responses_lite,
            "supports_parallel_tool_calls":
                self.supports_parallel_tool_calls,
            "supports_reasoning_summaries":
                self.supports_reasoning_summaries,
            "default_reasoning_level": self.default_reasoning_level,
            "default_reasoning_summary":
                self.default_reasoning_summary,
            "supports_verbosity": self.supports_verbosity,
            "default_verbosity": self.default_verbosity,
        }
