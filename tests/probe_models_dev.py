#!/usr/bin/env python3
"""Standalone probe of https://models.dev/api.json for model grouping.

Goal: find the right grouping key to identify "the same model" across
providers, so a future /model picker can first pick a model and then pick
among its providers -- which may differ in features and in wire protocol.

The models.dev data is provider-keyed: top level is {provider_id: entry},
and each entry has a provider-level ``npm`` (AI SDK package) and ``api``
(base URL), plus a ``models`` dict of {model_id: model_entry}. Model entries
carry ``id``, ``name``, and an optional ``family``, plus feature flags such
as ``reasoning``/``tool_call``/``temperature``/``attachment``.

This probe answers, empirically:
  1. What ``npm`` values are in use currently (and by which providers)?
  2. How many providers/models does each model group have when keyed by
     ``family``, by model ``id``, and by model ``name``?
  3. For multi-provider model groups, how do features and protocol
     (npm + api base URL) differ across the providers of one model?

Run:  python3 tests/probe_models_dev.py [--cache /tmp/models_dev_api.json]
Uses stdlib only.
"""

import argparse
import json
import sys
import urllib.request
from collections import Counter, defaultdict

API_URL = "https://models.dev/api.json"

# Feature flags a group's providers may disagree on. "interleaved" is a dict,
# so its presence/absence is compared via a marker instead of raw equality.
FEATURE_KEYS = [
    "reasoning",
    "tool_call",
    "temperature",
    "attachment",
    "structured_output",
    "open_weights",
]

PROTOCOL_KINDS = {
    "anthropic": "anthropic_messages",
    "openai": "openai_responses",
    "azure": "openai_chat/azure",
    "google": "gemini",
}


def fetch(url, cache_path=None):
    if cache_path:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except OSError:
            pass
    print(f"fetching {url} ...", file=sys.stderr)
    req = urllib.request.Request(url, headers={
        "User-Agent": "loki-models.dev-probe/1.0 (model grouping analysis)",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def protocol_label(provider_entry):
    """Compact protocol-ish descriptor from provider-level fields."""
    npm = provider_entry.get("npm") or "?"
    api = provider_entry.get("api") or "?"
    return f"{npm} @ {api}"


def group_models(data, keyfn, skip_missing=True):
    """Group (provider_id, model_entry) pairs by keyfn(model_entry, model_id)."""
    groups = defaultdict(list)
    missing = 0
    for pid, prov in data.items():
        for mid, m in (prov.get("models") or {}).items():
            key = keyfn(m, mid)
            if key is None:
                missing += 1
                continue
            groups[key].append((pid, m))
    return groups, missing


def key_family(m, mid):
    return m.get("family")


def key_id(m, mid):
    return m.get("id") or mid


def key_name(m, mid):
    return m.get("name")


def feature_row(m):
    out = []
    for k in FEATURE_KEYS:
        v = m.get(k)
        if k == "interleaved":
            v = bool(m.get("interleaved"))
        out.append("1" if v is True else ("0" if v is False else ("~" if v is None else str(v))))
    return "".join(out)


def summarize_grouping(data, label, keyfn):
    groups, missing = group_models(data, keyfn)
    sizes = Counter(len(v) for v in groups.values())
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"\n=== Grouping by {label} ===")
    print(f"  distinct groups: {len(groups)}   (models missing key: {missing})")
    print(f"  group size distribution (size: count): {dict(sorted(sizes.items()))}")
    print(f"  multi-provider groups: {len(multi)}")
    if multi:
        biggest = sorted(multi.items(), key=lambda kv: len(kv[1]), reverse=True)[:12]
        for key, members in biggest:
            print(f"\n  group {key!r} ({len(members)} providers):")
            for pid, m in members:
                prov = data[pid]
                print(f"    {pid:28s} npm={prov.get('npm', '?'):32s} "
                      f"feat=[{feature_row(m)}] id={m.get('id')!r} name={m.get('name')!r}")
    return groups, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None,
                    help="JSON file to reuse/write instead of refetching")
    ap.add_argument("--url", default=API_URL)
    args = ap.parse_args(argv)

    data = fetch(args.url, args.cache)
    print(f"providers: {len(data)}")

    # --- 1. npm values in use, and which providers use each ---
    npm_by_provider = {pid: p.get("npm") for pid, p in data.items()}
    npm_counter = Counter(npm_by_provider.values())
    print(f"\ndistinct npm values: {len(npm_counter)}")
    for value, count in npm_counter.most_common():
        providers = sorted(pid for pid, v in npm_by_provider.items() if v == value)
        print(f"  {count:4d}  {value:36s} e.g. {providers[:6]}")
    print("\nproviders with NO api base URL:",
          sorted(pid for pid, p in data.items() if not p.get("api")))

    # --- 2. grouping comparisons ---
    for label, keyfn in [
        ("family", key_family),
        ("model id", key_id),
        ("model name", key_name),
    ]:
        summarize_grouping(data, label, keyfn)

    # --- 3. per-model protocol divergence across providers ---
    print("\n=== Protocol divergence inside multi-provider model-id groups ===")
    groups, _ = group_models(data, key_id)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    diverged = 0
    feat_diverged = 0
    npm_diverged = 0
    for key, members in sorted(multi.items(), key=lambda kv: len(kv[1]), reverse=True):
        labels = {protocol_label(data[pid]) for pid, _ in members}
        nf = {feature_row(m) for _, m in members}
        nn = {data[pid].get("npm") for pid, _ in members}
        if len(labels) > 1:
            diverged += 1
        if len(nf) > 1:
            feat_diverged += 1
        if len(nn) > 1:
            npm_diverged += 1
        if len(labels) > 1:
            print(f"\n  {key!r} ({len(members)} providers) -- protocols differ:")
            for pid, m in members:
                print(f"    {pid:28s} {protocol_label(data[pid])}")
    print(f"\n  model-id groups where providers disagree on protocol: {diverged} / {len(multi)}")
    print(f"  model-id groups where providers disagree on npm SDK:     {npm_diverged} / {len(multi)}")
    print(f"  model-id groups where providers disagree on features:    {feat_diverged} / {len(multi)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
