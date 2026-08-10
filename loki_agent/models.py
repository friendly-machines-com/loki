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
import os
import sys
import urllib.request
from collections import defaultdict

from . import protocols

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


def protocol_label(provider_entry):
    """Short wire-protocol-ish label for a provider entry."""
    api = provider_entry.get("api")
    if api:
        detected = protocols.detect_protocol_from_url(api)
        if detected:
            return detected.replace("_", "-")
    return (provider_entry.get("npm") or "no-api")


def api_key_for(provider_entry, fallback=""):
    """First API key env var the provider declares, else the fallback key."""
    for var in provider_entry.get("env") or []:
        value = os.environ.get(var)
        if value:
            return value
    return fallback


# --------------------------------------------------------------------------
# Picker UI
# --------------------------------------------------------------------------

def _row_matches(text, query):
    words = query.lower().split()
    blob = text.lower()
    return all(w in blob for w in words)


async def _numbered_menu_async(rows, prompt, input_fn):
    """Numbered menu over rows=[(value, display_text)].

    Bare int selects that row; "filter WORDS" narrows (words in any order);
    empty cancels (returns None). Mirrors the session-picker gestures.
    """
    query = ""
    while True:
        shown = [r for r in rows if _row_matches(r[1], query)] if query else list(rows)
        for i, (_, text) in enumerate(shown, 1):
            print(f"{i}. {text}")
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


def _model_rows(groups):
    rows = []
    for name, members in groups.items():
        feat = feature_names(minimal_feature_bits(members))
        more = " [and more]" if union_feature_bits(members) != minimal_feature_bits(members) else ""
        label = f"{name} ({feat}){more} [{len(members)} providers]" if feat \
            else f"{name}{more} [{len(members)} providers]"
        rows.append((members, label))
    rows.sort(key=lambda r: r[1].lower())
    return rows


def _provider_rows(members):
    rows = []
    for pid, prov, m in members:
        parts = [prov.get("name") or pid]
        model_id = m.get("id")
        if model_id:
            parts.append(f"id={model_id}")
        feat = feature_names(feature_bits(m))
        if feat:
            parts.append(f"({feat})")
        parts.append(f"[{protocol_label(prov)}]")
        rows.append(((pid, prov, m), " ".join(parts)))
    rows.sort(key=lambda r: r[1].lower())
    return rows


async def run_model_picker_async(input_fn, cache_path=None):
    """Two-level picker: conflated model -> provider.

    Returns (provider_id, provider_entry, model_entry) for the chosen
    provider of the chosen model, or None if the user cancelled or models.dev
    could not be fetched.
    """
    try:
        _, groups = ensure_index(cache_path=cache_path)
    except Exception as e:
        print(f"models.dev unavailable: {e}", file=sys.stderr)
        return None

    model_rows = _model_rows(groups)
    if not model_rows:
        print("models.dev returned no models.", file=sys.stderr)
        return None
    members = await _numbered_menu_async(model_rows, "Model choice: ", input_fn)
    if members is None:
        return None

    provider_rows = _provider_rows(members)
    picked = await _numbered_menu_async(provider_rows, "Provider choice: ", input_fn)
    if picked is None:
        return None
    return picked
