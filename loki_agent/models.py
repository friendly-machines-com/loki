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
import urllib.request
from collections import defaultdict
from dataclasses import dataclass

from . import protocols
from .credentials import CredentialStore, is_credential_name

MODELS_DEV_URL = "https://models.dev/api.json"
USER_AGENT = "loki-models.dev-picker/1.0"

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
    credential_env: str
    api_url: str
    protocol: str


# --------------------------------------------------------------------------
# Data loading and indexing
# --------------------------------------------------------------------------

def fetch_models_dev(cache_path=None, url=MODELS_DEV_URL):
    """Fetch and parse models.dev/api.json (stdlib only; sets a real UA)."""
    if cache_path:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except OSError:
            pass
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
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


def ensure_index(cache_path=None):
    """Fetch (once per session) and return (data, groups)."""
    global _index_cache
    if _index_cache is None:
        data = fetch_models_dev(cache_path=cache_path)
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
    return bool(provider_entry.get("api")) and provider_protocol(provider_entry) is not None


def provider_access(provider_entry, credentials: CredentialStore):
    """Resolve a catalog provider against the captured startup environment.

    Returns the credential name, expanded API URL, and implemented protocol.
    Credential candidates preserve models.dev declaration order. API-template
    variables may use any startup variable, not only credential-shaped ones.
    """
    if not provider_supported(provider_entry):
        return None
    credential_names = [
        name for name in (provider_entry.get("env") or [])
        if is_credential_name(name)
    ]
    credential_env, credential_value = credentials.first_available(
        credential_names)
    if not credential_env or not credential_value:
        return None

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
        credential_env=credential_env,
        api_url=api_url,
        protocol=provider_protocol(provider_entry),
    )


def is_deprecated(model_entry):
    return model_entry.get("status") == "deprecated"


def filter_supported_groups(groups, credentials: CredentialStore | None = None):
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
    line -- e.g. the provider ids/names behind a truncated provider snippet.
    """
    if len(row) >= 3:
        blob = row[2]
    else:
        blob = row[1]
    words = query.lower().split()
    blob = blob.lower()
    return all(w in blob for w in words)


async def _numbered_menu_async(rows, prompt, input_fn, header=None):
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
            print(f"{i}. {row[1]}")
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


def _provider_snippet(members, max_shown=5):
    """'fireworks-ai, openrouter, ...' -- up to max_shown provider ids."""
    ids = [pid for pid, _, _ in members]
    shown = ", ".join(ids[:max_shown])
    if len(ids) > max_shown:
        shown += ", ..."
    return shown


def _model_rows(groups):
    rows = []
    for name, members in groups.items():
        status = " (deprecated)" if all(
            is_deprecated(model) for _, _, model in members) else ""
        feat = feature_names(minimal_feature_bits(members))
        more = " [and more]" if union_feature_bits(members) != minimal_feature_bits(members) else ""
        cost = cost_range_text(members)
        count = len(members)
        label = f"{name}{status} ({feat}){more}{cost} [{count} providers: {_provider_snippet(members)}]" if feat \
            else f"{name}{status}{more}{cost} [{count} providers: {_provider_snippet(members)}]"
        # Search blob: model name plus every provider id/name, so
        # "filter opencode" (or any provider) finds the model even when the
        # visible snippet truncates the provider list.
        search = " ".join([name] +
                          [pid for pid, _, _ in members] +
                          [p.get("name") or "" for _, p, _ in members] +
                          [m.get("status") or "" for _, _, m in members])
        rows.append((members, label, search))
    rows.sort(key=lambda r: r[1].lower())
    return rows


def _provider_rows(members):
    rows = []
    for pid, prov, m in members:
        parts = [prov.get("name") or pid]
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


async def run_flat_model_picker_async(input_fn, model_ids):
    """Single-level menu over a flat list of model ids (outage fallback).

    Used when models.dev is unreachable: loki.py fetches the current
    provider's own /models list via the existing load_models_async() and feeds
    it here, so /model keeps the same menu UI and gestures instead of falling
    back to a different, older menu. Returns the chosen model id, or None.
    """
    ids = [m for m in (model_ids or []) if m]
    if not ids:
        print()
        print("Usable models:")
        print("No models available from the current provider.")
        return None
    rows = [(m, m) for m in ids]
    rows.sort(key=lambda r: r[1].lower())
    return await _numbered_menu_async(
        rows,
        'Model choice (number selects, "filter WORDS" narrows, empty cancels): ',
        input_fn,
        header="Usable models:")


async def run_model_picker_async(input_fn, credentials: CredentialStore,
                                 cache_path=None):
    """Two-level picker: conflated model -> provider.

    Returns (provider_id, provider_entry, model_entry) for the chosen
    provider of the chosen model, or None if the user cancelled (at either
    menu). If models.dev cannot be fetched, the network exception from
    fetch_models_dev propagates naturally; the caller catches it and falls
    back to the current provider's own model list.
    """
    _, groups = ensure_index(cache_path=cache_path)
    # Keep only models served by at least one provider whose wire protocol
    # Loki can speak, so the menu is not flooded with the long tail.
    groups = filter_supported_groups(groups, credentials)

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
        header="Usable models:")
    if members is None:
        return None

    provider_rows = _provider_rows(members)
    picked = await _numbered_menu_async(
        provider_rows,
        'Provider choice (number selects, "filter WORDS" narrows, empty cancels): ',
        input_fn,
        header="Usable providers:")
    if picked is None:
        return None
    return picked
