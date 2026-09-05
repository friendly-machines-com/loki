"""Best-effort inference response observations, never request authority.

Each inference runtime owns a Store. Only explicit saves and orderly runtime
shutdown write its dirty observations. Independent runtimes merge under a lock;
observation time, not shutdown order, decides which value survives.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import math
import os
import re
import stat
import sys
import tempfile
import time
import urllib.parse
from datetime import datetime, timezone


MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
_ANY_CREDENTIAL = object()
# Response headers can carry credentials too. Keep names for discovery, but
# never copy these values into diagnostics. Generic "token" substring matching
# would wrongly discard useful token-budget and rate-limit headers.
SENSITIVE_HEADERS = frozenset({
    "cookie", "cookie2", "set-cookie", "set-cookie2",
    "authorization", "proxy-authorization", "authentication-info",
    "proxy-authentication-info", "api-key", "x-api-key", "x-goog-api-key",
    "x-auth-token", "x-access-token", "x-refresh-token", "x-session-token",
    "x-csrf-token", "x-xsrf-token", "x-codex-turn-state",
})


def _safe_value(name, value):
    return "[redacted]" if name.lower() in SENSITIVE_HEADERS else value


def snapshot_path():
    home = os.environ.get("XDG_STATE_HOME") or "~/.local/state"
    return os.path.join(os.path.expanduser(home), "loki", "response-headers.json")


def sanitized_endpoint(url):
    """Keep endpoint paths distinct; never persist URL credentials or queries."""
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host or parts.scheme.lower() not in {"http", "https"}:
        raise ValueError("invalid inference endpoint")
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    scheme = parts.scheme.lower()
    if port is not None and port != (443 if scheme == "https" else 80):
        host += f":{port}"
    return urllib.parse.urlunsplit((scheme, host, parts.path or "/", "", ""))


def _key(entry):
    return entry["endpoint"], entry["credential"]


def _newer(left, right):
    # A deterministic tie-break also makes merges independent of writer order
    # when two processes happen to timestamp observations identically.
    return (left["observed_at_ns"], json.dumps(left, sort_keys=True)) > (
        right["observed_at_ns"], json.dumps(right, sort_keys=True))


def _merge(target, entries):
    for entry in entries:
        key = _key(entry)
        if key not in target:
            target[key] = {**entry, "headers": dict(entry["headers"])}
            continue
        existing = target[key]
        if _newer(entry["latest"], existing["latest"]):
            existing["latest"] = entry["latest"]
        for name, observation in entry["headers"].items():
            previous = existing["headers"].get(name)
            if previous is None or _newer(observation, previous):
                existing["headers"][name] = observation


def _valid_observation(value):
    return (
        isinstance(value, dict)
        and type(value.get("observed_at_ns")) is int
        and type(value.get("status")) is int
        and isinstance(value.get("model"), str)
    )


def _read(path):
    try:
        with open(path, "r", encoding="utf-8") as source:
            text = source.read(MAX_SNAPSHOT_BYTES + 1)
    except FileNotFoundError:
        return {}
    if len(text) > MAX_SNAPSHOT_BYTES:
        raise ValueError("response status snapshot is too large")
    data = json.loads(text)
    if (not isinstance(data, dict) or data.get("version") != 1
            or not isinstance(data.get("endpoints"), list)):
        raise ValueError("invalid response status snapshot")
    for entry in data["endpoints"]:
        if (not isinstance(entry, dict)
                or not isinstance(entry.get("endpoint"), str)
                or not (entry.get("credential") is None
                        or isinstance(entry.get("credential"), str))
                or not isinstance(entry.get("headers"), dict)
                or not _valid_observation(entry.get("latest"))):
            raise ValueError("invalid response status entry")
        latest = entry["latest"]
        if (not isinstance(latest.get("header_names"), list)
                or not all(isinstance(name, str)
                           for name in latest["header_names"])
                or not all(latest.get(key) is None
                           or isinstance(latest.get(key), str)
                           for key in ("provider_id", "provider_name"))):
            raise ValueError("invalid latest response status")
        for name, observation in entry["headers"].items():
            if (not name or name != name.lower()
                    or not _valid_observation(observation)
                    or not isinstance(observation.get("value"), str)):
                raise ValueError("invalid response header observation")
            observation["value"] = _safe_value(name, observation["value"])
    result = {}
    _merge(result, data["endpoints"])
    return result


def _document(entries):
    return {"version": 1, "endpoints": sorted(
        entries.values(), key=lambda entry: (
            entry["endpoint"], entry["credential"] or ""))}


class Store:
    def __init__(self, path=None):
        self.path = path or snapshot_path()
        self.observations = {}
        self.dirty = False

    def observer(self, endpoint, credential, model, provider_id=None,
                 provider_name=None):
        """Freeze non-secret request context before the first network await."""
        endpoint = sanitized_endpoint(endpoint)

        def observe(status, headers):
            now = time.time_ns()
            context = {"observed_at_ns": now, "status": status, "model": model}
            values = {
                name.lower(): {**context, "value": _safe_value(name, value)}
                for name, value in headers.items()
            }
            _merge(self.observations, [{
                "endpoint": endpoint,
                "credential": credential,
                "latest": {
                    **context,
                    "provider_id": provider_id,
                    "provider_name": provider_name,
                    "header_names": sorted(values),
                },
                "headers": values,
            }])
            self.dirty = True

        return observe

    def snapshot(self, endpoint=None, *, credential=_ANY_CREDENTIAL):
        entries = _read(self.path)
        _merge(entries, self.observations.values())
        if endpoint is not None:
            endpoint = sanitized_endpoint(endpoint)
            entries = {key: value for key, value in entries.items()
                       if key[0] == endpoint}
        if credential is not _ANY_CREDENTIAL:
            entries = {key: value for key, value in entries.items()
                       if key[1] == credential}
        return _document(entries)

    async def save(self):
        if not self.dirty:
            return
        directory = os.path.dirname(self.path)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        lock_fd = os.open(self.path + ".lock", os.O_CREAT | os.O_RDWR
                          | os.O_NOFOLLOW, 0o600)
        temporary = None
        try:
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise OSError("response status lock is not a regular file")
            deadline = asyncio.get_running_loop().time() + 0.25
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if asyncio.get_running_loop().time() >= deadline:
                        raise OSError("response status snapshot is busy")
                    await asyncio.sleep(0.01)
            # No await between the merge and publish: observations cannot race
            # this runtime's dirty reset. Small local filesystem operations are
            # synchronous; lock contention never blocks the asyncio loop.
            entries = _read(self.path)
            _merge(entries, self.observations.values())
            content = json.dumps(_document(entries), indent=2, ensure_ascii=True)
            if len(content) > MAX_SNAPSHOT_BYTES:
                raise ValueError("response status snapshot is too large")
            fd, temporary = tempfile.mkstemp(prefix=".response-headers-",
                                             dir=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(content + "\n")
            os.replace(temporary, self.path)
            temporary = None
            self.dirty = False
        finally:
            if temporary is not None:
                os.unlink(temporary)
            os.close(lock_fd)

    async def save_on_exit(self):
        try:
            await self.save()
        except (OSError, ValueError) as error:
            print(f"Could not save response status: {ascii(str(error))}",
                  file=sys.stderr)


def _codex_quota_summary(entry):
    # These header meanings belong to this endpoint, not a models.dev label.
    if entry["endpoint"] != "https://chatgpt.com/backend-api/codex/responses":
        return []
    headers = entry["headers"]
    latest = entry["latest"]
    rows = []
    for name, used_observation in headers.items():
        match = re.fullmatch(
            r"x-codex(?:-([a-z0-9-]+))?-(primary|secondary)-used-percent", name)
        if match is None:
            continue
        bucket, window = match.groups()
        prefix = "x-codex" + (f"-{bucket}" if bucket else "")
        window_name = f"{prefix}-{window}-window-minutes"
        window_observation = headers.get(window_name)
        # Destructive per-key updates can retain a window from a different
        # response. Do not synthesize a quota from that mixed observation.
        if (window_observation is None
                or window_observation["observed_at_ns"]
                != used_observation["observed_at_ns"]):
            continue
        try:
            used = float(used_observation["value"])
            minutes = int(window_observation["value"])
        except (ValueError, OverflowError):
            continue
        if not math.isfinite(used) or not 0 <= used <= 100 or minutes <= 0:
            continue
        label = "Main subscription bucket"
        if bucket:
            label = headers.get(f"{prefix}-limit-name", {}).get("value") or bucket
            # Provider-supplied labels must not become terminal control text.
            label = ascii(label)[1:-1]
        if minutes % 1440 == 0:
            amount, unit = minutes // 1440, "day"
        elif minutes % 60 == 0:
            amount, unit = minutes // 60, "hour"
        else:
            amount, unit = minutes, "minute"
        duration = f"{amount} {unit}{'' if amount == 1 else 's'}"
        retained = (
            used_observation["observed_at_ns"] != latest["observed_at_ns"]
            or name not in latest["header_names"]
            or window_name not in latest["header_names"])
        text = (f"  {label}: {used:g}% used, {100 - used:g}% remaining"
                f" - {duration}" + (" [retained observation]" if retained else ""))
        rows.append((bucket or "", window, text))
    return ["Subscription quota (last observed):"] + [
        text for _, _, text in sorted(rows)] if rows else []


def render(document):
    if not document["endpoints"]:
        return "No HTTP chat response headers observed yet."
    lines = []
    for entry in document["endpoints"]:
        latest = entry["latest"]
        lines.append(f"Endpoint: {ascii(entry['endpoint'])}")
        lines.append(f"Credential: {ascii(entry['credential'])}")
        lines.append(f"Provider (informational): {ascii(latest['provider_id'])}")
        lines.append(f"Latest response: HTTP {latest['status']}; "
                     f"model {ascii(latest['model'])}")
        lines.append("Response headers:")
        for name, observation in sorted(entry["headers"].items()):
            stamp = datetime.fromtimestamp(
                observation["observed_at_ns"] / 1e9, timezone.utc).isoformat()
            retained = " retained" if name not in latest["header_names"] else ""
            lines.append(
                f"  {ascii(name)}: {ascii(observation['value'])} "
                f"[{stamp}; HTTP {observation['status']}; "
                f"model {ascii(observation['model'])}{retained}]")
        lines.append("")
        summary = _codex_quota_summary(entry)
        if summary:
            lines.extend(summary)
        lines.append("")
    return "\n".join(lines).rstrip()


def main(args):
    parser = argparse.ArgumentParser(description=(
        "Inspect saved HTTP chat response headers (not live worker memory)."))
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--endpoint", help="filter by inference endpoint URL")
    options = parser.parse_args(args)
    try:
        document = Store().snapshot(options.endpoint)
        print(json.dumps(document, indent=2, ensure_ascii=True)
              if options.json else (
                  "Saved observations only (not live balances); other runtimes' "
                  "unsaved observations are not visible.\n" + render(document)))
    except (OSError, ValueError, OverflowError) as error:
        print(f"Could not read response status: {ascii(str(error))}",
              file=sys.stderr)
        return 1
    return 0
