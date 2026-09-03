"""models.dev-backed model/provider picker for /model.

Self-contained feature module: it owns the models.dev data model, the
"conflated" model-name grouping, and the two-level picker UI (model ->
provider). loki.py calls in; nothing here imports loki.py.

models.dev data is provider-keyed: {provider_id: {id, env, npm, name, api,
doc, models: {model_id: entry}}}. Model entries carry ``id``, ``name``, an
optional ``family``, and feature flags (reasoning, tool_call, ...).

Providers of the same real model often disagree on features and on wire
protocol, so:

  * the first menu lists conflated model identities keyed by *name*, showing
    in parentheses the *minimal* features guaranteed across that model's
    providers (the intersection over providers); and
  * the second menu lists that model's providers, showing each provider's own
    features and its protocol (inferred from the provider's ``api`` base URL
    via protocols.detect_protocol_from_url, falling back to the ``npm`` SDK).

Menus mirror the existing numbered-list look ("N. <text>", "User choice: ")
and additionally accept "filter WORDS" to narrow a large list and empty to
cancel.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from . import authentications
from . import http_client
from . import protocols
from .credentials import (
    CredentialInventory,
    CredentialStore,
    is_credential_name,
)

MODELS_DEV_URL = "https://models.dev/api.json"
USER_AGENT = "loki-models.dev-picker/1.0"
MODELS_DEV_TIMEOUT_S = 30
MODELS_DEV_MAX_BYTES = 20 * 1024 * 1024
MODELS_DEV_RETRY_MAX_ATTEMPTS = 3

# models.dev omits endpoints that its native JavaScript SDK packages know
# internally.  Keep the downloaded catalog and cache faithful to models.dev,
# then apply narrowly matched, explicitly annotated repairs in memory.
OPENAI_PLATFORM_API_BASE = "https://api.openai.com/v1"
_LOKI_API_SOURCE_KEY = "_loki_api_source"
_LOKI_API_REJECTION_KEY = "_loki_api_rejection"
_OPENAI_PLATFORM_API_SOURCE = "built-in OpenAI Platform default"
_OPENAI_SUBSCRIPTION_API_SOURCE = (
    "built-in OpenAI ChatGPT subscription endpoint")
_LOKI_CREDENTIAL_REF_KEY = "_loki_credential_ref"
_LOKI_SYNTHETIC_KEY = "_loki_synthetic"
_OPENAI_SUBSCRIPTION_SENTINEL = object()
OPENAI_SUBSCRIPTION_PROVIDER_ID = "openai-subscription"

# Feature flags shown in the pickers, in a stable order. A provider may set
# each one differently for the same model; menus show the ones that are on.
FEATURE_KEYS = (
    "reasoning",
    "tool_call",
    "structured_output",
    "temperature",
    "attachment",
    "open_weights",
)

FEATURE_LABELS = {
    "reasoning": "reasoning",
    "tool_call": "tools",
    "structured_output": "struct",
    "temperature": "temp",
    "attachment": "attach",
    "open_weights": "open",
}

_index_cache = None  # (data, groups) once fetched this session
_API_TEMPLATE_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class ProviderAccess:
    credential_ref: authentications.CredentialRef
    api_url: str
    protocol: str


@dataclass(frozen=True)
class ExplicitConnectionOption:
    """A captured LOKI_* connection offered alongside catalog providers."""

    model: str
    api_url: str
    protocol: str


# --------------------------------------------------------------------------
# Data loading and indexing
# --------------------------------------------------------------------------

def _read_catalog_cache(cache_path):
    with open(cache_path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_catalog_cache(cache_path, data):
    with open(cache_path, "w", encoding="utf-8") as stream:
        json.dump(data, stream)


def _validate_catalog(data):
    if not isinstance(data, dict):
        raise ValueError("models.dev catalog must be a JSON object")
    return data


def _openai_platform_signature(provider_entry):
    """Whether this is exactly the native OpenAI provider models.dev emits."""
    return (
        provider_entry.get("id") == "openai"
        and provider_entry.get("npm") == "@ai-sdk/openai"
        and provider_entry.get("env") == ["OPENAI_API_KEY"]
    )


def _canonical_openai_platform_api(api):
    if not isinstance(api, str):
        return False
    return api.rstrip("/") in {
        OPENAI_PLATFORM_API_BASE,
        OPENAI_PLATFORM_API_BASE + "/responses",
    }


def normalize_catalog(data):
    """Return a catalog with fail-closed, explicitly marked Loki repairs.

    The provider-map key, provider id, native SDK package, and credential
    declaration must all identify OpenAI exactly.  The SDK package alone is
    not unique in models.dev.  A changed signature or noncanonical endpoint
    is rejected rather than risking disclosure of ``OPENAI_API_KEY`` to a
    different service.

    This function never mutates the raw downloaded or cached catalog.
    """
    provider_entry = data.get("openai")
    if provider_entry is None:
        return data
    if not isinstance(provider_entry, dict):
        return data

    normalized = dict(data)
    normalized_provider = dict(provider_entry)
    normalized["openai"] = normalized_provider

    if not _openai_platform_signature(provider_entry):
        normalized_provider[_LOKI_API_REJECTION_KEY] = (
            "canonical OpenAI provider signature changed")
        return normalized

    api = provider_entry.get("api")
    if not api:
        normalized_provider["api"] = OPENAI_PLATFORM_API_BASE
        normalized_provider[_LOKI_API_SOURCE_KEY] = (
            _OPENAI_PLATFORM_API_SOURCE)
    elif not _canonical_openai_platform_api(api):
        normalized_provider[_LOKI_API_REJECTION_KEY] = (
            "canonical OpenAI provider declared a non-OpenAI endpoint")
        return normalized

    subscription = dict(normalized_provider)
    subscription["id"] = OPENAI_SUBSCRIPTION_PROVIDER_ID
    subscription["name"] = "OpenAI ChatGPT subscription"
    subscription["api"] = (
        authentications.OPENAI_CHATGPT_RESPONSES_URL)
    subscription["env"] = []
    subscription[_LOKI_API_SOURCE_KEY] = (
        _OPENAI_SUBSCRIPTION_API_SOURCE)
    subscription[_LOKI_CREDENTIAL_REF_KEY] = (
        authentications.CredentialRef.openai_subscription().encode())
    # A non-JSON sentinel ensures downloaded catalog fields that happen to
    # use Loki's private names can never manufacture a broker credential ref.
    subscription[_LOKI_SYNTHETIC_KEY] = (
        _OPENAI_SUBSCRIPTION_SENTINEL)
    subscription["models"] = {
        model_id: {
            key: value for key, value in model.items()
            if key != "cost"
        }
        for model_id, model in (
            normalized_provider.get("models") or {}).items()
        if isinstance(model, dict)
    }
    normalized[OPENAI_SUBSCRIPTION_PROVIDER_ID] = subscription
    return normalized


async def fetch_models_dev(cache_path=None, url=MODELS_DEV_URL):
    """Fetch and parse models.dev/api.json through Loki's HTTP transport."""
    if cache_path:
        try:
            return _validate_catalog(_read_catalog_cache(cache_path))
        except OSError:
            pass
    response = await http_client.async_http_request(
        "GET",
        url,
        headers_in={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        timeout=MODELS_DEV_TIMEOUT_S,
        max_bytes=MODELS_DEV_MAX_BYTES,
        retry_max_attempts=MODELS_DEV_RETRY_MAX_ATTEMPTS,
    )
    if response.status >= 400:
        raise OSError(
            f"models.dev returned HTTP {response.status} {response.reason}")
    if response.truncated:
        raise OSError(
            f"models.dev catalog exceeds {MODELS_DEV_MAX_BYTES} bytes")
    data = _validate_catalog(json.loads(
        response.body.decode("utf-8-sig")))
    if cache_path:
        _write_catalog_cache(cache_path, data)
    return data


def build_groups(data):
    """Group (provider_id, provider_entry, model_entry) by model name.

    Name is the "conflated" key: it is the human label providers share, where
    per-provider model *ids* diverge (zai-org/GLM-5.2 vs glm-5.2 vs glm-5-2)
    and the ``family`` key is missing/inconsistent across providers.
    """
    groups = defaultdict(list)
    for pid, prov in (data or {}).items():
        for mid, m in (prov.get("models") or {}).items():
            if not isinstance(m, dict):
                continue
            key = m.get("name") or mid
            groups[key].append((pid, prov, m))
    return dict(groups)


async def ensure_index(cache_path=None):
    """Fetch (once per session) and return (data, groups)."""
    global _index_cache
    if _index_cache is None:
        data = normalize_catalog(
            await fetch_models_dev(cache_path=cache_path))
        _index_cache = (data, build_groups(data))
    return _index_cache


# --------------------------------------------------------------------------
# Feature helpers
# --------------------------------------------------------------------------

def feature_bits(model_entry):
    return tuple(bool(model_entry.get(k)) for k in FEATURE_KEYS)


def feature_names(bits):
    labels = [FEATURE_LABELS[k] for k in FEATURE_KEYS]
    return ", ".join(label for label, on in zip(labels, bits) if on)


def minimal_feature_bits(members):
    """Intersection of features over a model's providers (AND)."""
    bits = feature_bits(members[0][2])
    for _, _, m in members[1:]:
        bits = tuple(a and b for a, b in zip(bits, feature_bits(m)))
    return bits


def union_feature_bits(members):
    """Union of features over a model's providers (OR)."""
    bits = feature_bits(members[0][2])
    for _, _, m in members[1:]:
        bits = tuple(a or b for a, b in zip(bits, feature_bits(m)))
    return bits


# --------------------------------------------------------------------------
# Cost helpers (USD per 1M tokens, from each model entry's "cost" object)
# --------------------------------------------------------------------------

def cost_pair(model_entry):
    """(input, output) USD per 1M tokens, or None if the entry has no cost."""
    c = model_entry.get("cost")
    if not isinstance(c, dict):
        return None
    i, o = c.get("input"), c.get("output")
    if i is None and o is None:
        return None
    return (i if isinstance(i, (int, float)) else 0,
            o if isinstance(o, (int, float)) else 0)


def _fmt_usd(value):
    # Compact dollar rendering: $0.6, $1.4, $30, $0.00015.
    return f"${value:g}"


def cost_range_text(members):
    """'cost: in $minIn-$maxIn per 1M tokens, out $minOut-$maxOut per 1M
    tokens' over providers, or ''. Units are USD per 1M tokens (models.dev
    convention); the unit is repeated on each figure.
    """
    pairs = [p for p in (cost_pair(m) for _, _, m in members) if p]
    if not pairs:
        return ""
    ins = [p[0] for p in pairs]
    outs = [p[1] for p in pairs]
    return (f" cost: in {_fmt_usd(min(ins))}-{_fmt_usd(max(ins))} per 1M tokens, "
            f"out {_fmt_usd(min(outs))}-{_fmt_usd(max(outs))} per 1M tokens")


def cost_text(model_entry):
    """'cost: in $in per 1M tokens, out $out per 1M tokens' for one provider,
    or ''."""
    pair = cost_pair(model_entry)
    if pair is None:
        return ""
    return (f" cost: in {_fmt_usd(pair[0])} per 1M tokens, "
            f"out {_fmt_usd(pair[1])} per 1M tokens")


def protocol_label(provider_entry):
    """Short wire-protocol-ish label for a provider entry."""
    detected = provider_protocol(provider_entry)
    if detected:
        return detected.replace("_", "-")
    return (provider_entry.get("npm") or "no-api")


def provider_protocol(provider_entry):
    """Implemented wire protocol for a catalog provider, or None."""
    api = provider_entry.get("api") or ""
    detected = protocols.detect_protocol_from_url(api)
    if detected in protocols.SUPPORTED_PROTOCOLS:
        return detected
    detected = protocols.detect_protocol_from_npm(provider_entry.get("npm"))
    if detected in protocols.SUPPORTED_PROTOCOLS:
        return detected
    if api.rstrip("/").endswith("/v1"):
        return protocols.OPENAI_CHAT
    return None


def provider_supported(provider_entry):
    """True if Loki can actually use this provider.

    A provider needs a concrete API URL and one of:
      1. Its ``api`` URL names one of the supported wire protocols via its
         endpoint path (.../chat/completions, .../messages, .../responses).
      2. Its ``npm`` package is one of the three that names a protocol
         directly (@ai-sdk/openai-compatible -> chat, @ai-sdk/anthropic ->
         messages, @ai-sdk/openai -> responses).
      3. Its ``api`` URL follows the OpenAI-compatible bare ``/v1`` base
         convention, which reinstall_provider treats as openai_chat.

    Providers with no API URL or usable protocol signal are dropped.
    """
    if provider_entry.get(_LOKI_API_REJECTION_KEY):
        return False
    return bool(provider_entry.get("api")) and provider_protocol(provider_entry) is not None


def provider_display_name(provider_id, provider_entry):
    """Provider name including the provenance of Loki-supplied defaults."""
    name = provider_entry.get("name") or provider_id
    if provider_entry.get(_LOKI_API_SOURCE_KEY) == (
            _OPENAI_PLATFORM_API_SOURCE):
        return f"{name} Platform API [endpoint supplied by Loki]"
    if provider_entry.get(_LOKI_API_SOURCE_KEY) == (
            _OPENAI_SUBSCRIPTION_API_SOURCE):
        return f"{name} [endpoint supplied by Loki]"
    return name


def provider_description(provider_entry):
    """ACP/provider description with any endpoint repair made explicit."""
    description = provider_entry.get("api") or ""
    if provider_entry.get(_LOKI_API_SOURCE_KEY) == (
            _OPENAI_PLATFORM_API_SOURCE):
        return (
            f"{description}; endpoint supplied by Loki because models.dev "
            "omits the native SDK default")
    if provider_entry.get(_LOKI_API_SOURCE_KEY) == (
            _OPENAI_SUBSCRIPTION_API_SOURCE):
        return (
            f"{description}; ChatGPT subscription provider synthesized "
            "by Loki from the canonical OpenAI catalog entry")
    return description


def provider_access(
        provider_entry,
        credentials: CredentialStore | CredentialInventory):
    """Resolve a catalog provider against the captured startup environment.

    Returns the credential name, expanded API URL, and implemented protocol.
    Credential candidates preserve models.dev declaration order. API-template
    variables may use any startup variable, not only credential-shaped ones.
    """
    if not provider_supported(provider_entry):
        return None
    encoded_ref = provider_entry.get(_LOKI_CREDENTIAL_REF_KEY)
    if encoded_ref is not None:
        if provider_entry.get(_LOKI_SYNTHETIC_KEY) is not (
                _OPENAI_SUBSCRIPTION_SENTINEL):
            return None
        try:
            credential_ref = authentications.CredentialRef.decode(
                encoded_ref)
        except ValueError:
            return None
        if not credentials.has_ref(credential_ref):
            return None
    else:
        credential_names = [
            name for name in (provider_entry.get("env") or [])
            if is_credential_name(name)
        ]
        credential_env = credentials.first_available_name(
            credential_names)
        if not credential_env:
            return None
        credential_ref = authentications.CredentialRef.environment(
            credential_env)

    api_template = provider_entry.get("api")
    template_names = _API_TEMPLATE_VAR_RE.findall(api_template)
    # Persisted and displayed endpoint URLs are non-secret configuration.
    # Never expand a credential itself into a URL.
    if any(is_credential_name(name) for name in template_names):
        return None
    missing = [
        name for name in template_names if not credentials.has(name)]
    if missing:
        return None
    api_url = _API_TEMPLATE_VAR_RE.sub(
        lambda match: credentials.get(match.group(1)), api_template)
    return ProviderAccess(
        credential_ref=credential_ref,
        api_url=api_url,
        protocol=provider_protocol(provider_entry),
    )


def is_deprecated(model_entry):
    return model_entry.get("status") == "deprecated"


def filter_supported_groups(
        groups,
        credentials: CredentialStore | CredentialInventory | None = None):
    """Drop unusable provider members and then empty model groups.

    With a credential store, usable also means that a declared credential and
    every API-template variable are present. The optional protocol-only mode is
    useful to callers that are inspecting catalog support independently of one
    process environment.
    """
    out = {}
    for name, members in groups.items():
        kept = [m for m in members
                if provider_supported(m[1])
                and (credentials is None
                     or provider_access(m[1], credentials) is not None)]
        if kept:
            out[name] = kept
    return out


# --------------------------------------------------------------------------
# Picker UI
# --------------------------------------------------------------------------

def _row_matches(row, query):
    """Match a menu row against "filter WORDS" (words in any order).

    A row is (value, display) or (value, display, search_blob); the search
    blob, when present, extends what the filter can match beyond the visible
    line.
    """
    if len(row) >= 3:
        blob = row[2]
    else:
        blob = row[1]
    words = query.lower().split()
    blob = blob.lower()
    return all(w in blob for w in words)


async def _numbered_menu_async(
        rows, prompt, input_fn, *, text_writer, header=None):
    """Numbered menu over rows=[(value, display_text)] (optionally a third
    search_text element for filtering beyond the visible line).

    Bare int selects that row; "filter WORDS" narrows (words in any order);
    empty cancels (returns None). Mirrors the session-picker gestures.
    """
    query = ""
    while True:
        shown = [r for r in rows if _row_matches(r, query)] if query else list(rows)
        if header:
            print()
            print(header)
        for i, row in enumerate(shown, 1):
            print(f"{i}. ", end="")
            text_writer(row[1])
            print()
        choice = (await input_fn(prompt) or "").strip()
        if choice == "filter" or choice.startswith("filter "):
            query = choice[len("filter"):].strip()
            continue
        try:
            n = int(choice)
        except ValueError:
            n = None
        if n is not None:
            if 1 <= n <= len(shown):
                return shown[n - 1][0]
            continue
        if not choice:
            return None
        continue


def _provider_list(members):
    """All provider ids for a grouped model, in catalog order."""
    return ", ".join(
        "explicit LOKI_*" if isinstance(member, ExplicitConnectionOption)
        else member[0]
        for member in members)


def _catalog_members(members):
    return [
        member for member in members
        if not isinstance(member, ExplicitConnectionOption)]


def _add_explicit_connection(groups, explicit_connection):
    if explicit_connection is None:
        return {name: list(members) for name, members in groups.items()}

    result = {name: list(members) for name, members in groups.items()}
    matching_name = next(
        (name for name in result
         if name.casefold() == explicit_connection.model.casefold()),
        None,
    )
    if matching_name is None:
        matching_name = next(
            (name for name, members in result.items()
             if any(
                 (model_entry.get("id") or "") == explicit_connection.model
                 for _, _, model_entry in members)),
            explicit_connection.model,
        )
    result.setdefault(matching_name, []).append(explicit_connection)
    return result


def _model_rows(groups):
    rows = []
    for name, members in groups.items():
        catalog_members = _catalog_members(members)
        explicit_members = [
            member for member in members
            if isinstance(member, ExplicitConnectionOption)]
        has_explicit = bool(explicit_members)
        status = " (deprecated)" if (
            catalog_members
            and not has_explicit
            and all(is_deprecated(model)
                    for _, _, model in catalog_members)) else ""
        feat = (
            feature_names(minimal_feature_bits(catalog_members))
            if catalog_members else "")
        more = " [and more]" if (
            catalog_members
            and union_feature_bits(catalog_members)
            != minimal_feature_bits(catalog_members)) else ""
        cost = cost_range_text(catalog_members)
        count = len(members)
        label = f"{name}{status} ({feat}){more}{cost} [{count} providers: {_provider_list(members)}]" if feat \
            else f"{name}{status}{more}{cost} [{count} providers: {_provider_list(members)}]"
        # Search blob also includes provider display names, which do not
        # necessarily appear in the visible list of provider ids.
        search = " ".join(
            [name]
            + [pid for pid, _, _ in catalog_members]
            + [
                provider_display_name(pid, provider)
                for pid, provider, _ in catalog_members
            ]
            + [m.get("status") or "" for _, _, m in catalog_members]
            + [
                value
                for explicit in explicit_members
                for value in (
                    "explicit LOKI connection",
                    explicit.api_url,
                    explicit.protocol,
                )
            ])
        rows.append((members, label, search))
    rows.sort(key=lambda r: r[1].lower())
    return rows


def _provider_rows(members):
    rows = []
    for member in members:
        if isinstance(member, ExplicitConnectionOption):
            label = (
                f"Explicit LOKI_* connection id={member.model} "
                f"[{member.protocol.replace('_', '-')}] "
                f"api={member.api_url}")
            rows.append((member, label))
            continue
        pid, prov, m = member
        parts = [provider_display_name(pid, prov)]
        model_id = m.get("id")
        if model_id:
            parts.append(f"id={model_id}")
        if is_deprecated(m):
            parts.append("(deprecated)")
        feat = feature_names(feature_bits(m))
        if feat:
            parts.append(f"({feat})")
        parts.append(f"[{protocol_label(prov)}]")
        if prov.get("api"):
            parts.append(f"api={prov['api']}")
        cost = cost_text(m).strip()
        if cost:
            parts.append(cost)
        rows.append(((pid, prov, m), " ".join(parts)))
    rows.sort(key=lambda r: r[1].lower())
    return rows


async def run_flat_model_picker_async(
        input_fn, model_ids,
        explicit_connection: ExplicitConnectionOption | None = None,
        *,
        text_writer):
    """Single-level menu over a flat list of model ids (outage fallback).

    Used when models.dev is unreachable: loki.py fetches the current
    provider's own /models list via the existing load_models_async() and feeds
    it here, so /model keeps the same menu UI and gestures instead of falling
    back to a different, older menu. Returns the chosen model id, or None.
    """
    ids = [m for m in (model_ids or []) if m]
    if not ids and explicit_connection is None:
        print()
        print("Usable models:")
        print("No models available from the current provider.")
        return None
    current_suffix = (
        " [current provider]" if explicit_connection is not None else "")
    rows = [(m, m + current_suffix) for m in ids]
    if explicit_connection is not None:
        rows.append((
            explicit_connection,
            f"{explicit_connection.model} "
            f"[Explicit LOKI_* connection; "
            f"{explicit_connection.protocol.replace('_', '-')}; "
            f"api={explicit_connection.api_url}]",
        ))
    rows.sort(key=lambda r: r[1].lower())
    return await _numbered_menu_async(
        rows,
        'Model choice (number selects, "filter WORDS" narrows, empty cancels): ',
        input_fn,
        text_writer=text_writer,
        header="Usable models:")


def flattened_config_option_choices(
        credentials, explicit_connection=None, groups=None):
    """Return ``(ACP option, selectable leaf)`` pairs.

    ``groups`` lets asynchronous front-ends supply catalog discovery after
    loading it through ``ensure_index``. Passing ``None`` is deliberately
    offline-only: synchronous option formatting must never perform network
    I/O. An explicit LOKI_* connection therefore remains available without
    depending on the catalog service.
    """
    if groups is None:
        groups = {}
    groups = _add_explicit_connection(
        filter_supported_groups(groups, credentials),
        explicit_connection,
    )
    choices = []
    for row in _model_rows(groups):
        for member in row[0]:
            if isinstance(member, ExplicitConnectionOption):
                choices.append(({
                    "value": "loki-explicit",
                    "name": f"{member.model} [LOKI_* connection]",
                    "description": (
                        f"explicit LOKI_* env connection; "
                        f"{member.protocol}; api={member.api_url}"),
                }, member))
                continue
            provider_id, provider_entry, model_entry = member
            model_id = model_entry.get("id") or model_entry.get("name")
            label = provider_display_name(provider_id, provider_entry)
            description = provider_description(provider_entry)
            choices.append(({
                "value": f"{provider_id}/{model_id}",
                "name": f"{model_id} ({label})",
                "description": description,
            }, member))
    return choices


def flattened_config_options(credentials, explicit_connection=None,
                             groups=None):
    """ACP select entries, preserving explicit config while offline."""
    return [
        option for option, _leaf in flattened_config_option_choices(
            credentials,
            explicit_connection=explicit_connection,
            groups=groups,
        )
    ]


async def run_model_picker_async(
        input_fn,
        credentials: CredentialStore | CredentialInventory,
        cache_path=None,
        explicit_connection: ExplicitConnectionOption | None = None,
        *,
        text_writer):
    """Two-level picker: conflated model -> provider.

    Returns (provider_id, provider_entry, model_entry) for the chosen
    catalog provider, an ExplicitConnectionOption for the captured LOKI_*
    connection, or None if the user cancelled (at either menu). If models.dev
    cannot be fetched, the network exception from fetch_models_dev propagates
    naturally; the caller catches it and uses the outage picker.
    """
    _, groups = await ensure_index(cache_path=cache_path)
    # Keep only models served by at least one provider whose wire protocol
    # Loki can speak, so the menu is not flooded with the long tail.
    groups = _add_explicit_connection(
        filter_supported_groups(groups, credentials),
        explicit_connection,
    )

    model_rows = _model_rows(groups)
    if not model_rows:
        print()
        print("Usable models:")
        print("No models available for configured provider credentials.")
        return None
    members = await _numbered_menu_async(
        model_rows,
        'Model choice (number selects, "filter WORDS" narrows, empty cancels): ',
        input_fn,
        text_writer=text_writer,
        header="Usable models:")
    if members is None:
        return None

    provider_rows = _provider_rows(members)
    picked = await _numbered_menu_async(
        provider_rows,
        'Provider choice (number selects, "filter WORDS" narrows, empty cancels): ',
        input_fn,
        text_writer=text_writer,
        header="Usable providers:")
    if picked is None:
        return None
    return picked
