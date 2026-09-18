"""Z.ai (Zhipu) Coding Plan account controls: quotas and reset cards.

The quota endpoint is the one the z.ai console itself polls; the reset-card
endpoints come from the console's own frontend bundle.  All sit behind the
same gateway (unauthenticated requests answer ``code 1001``).  Live checks
(2026-09-17) confirmed the card-list endpoint accepts the Coding Plan API
key both raw and under ``Bearer``; the consume endpoint has not been
exercised, because doing so spends a card.  Requests send the key raw first
and fall back to ``Bearer`` only on an authorization refusal, so a refusal
still gets an explanatory error rather than a false unknown outcome, and a
mutation is never re-sent after the server may have acted on it.
"""

from __future__ import annotations

import datetime
import json
import uuid

from . import authentications, provider_controls


ZAI_ORIGINS = frozenset({
    "https://api.z.ai",        # global
    "https://open.bigmodel.cn",  # China
})
_QUOTA_PATH = "/api/monitor/usage/quota/limit"
# Cards reset one quota to 100%.  The console sends targetType=PERSONAL on
# personal plans and TEAM on team plans; a team response to PERSONAL reports
# no cards rather than an error.
_RESET_LIST_PATH = "/api/biz/customer-package-reset/list?targetType=PERSONAL"
_RESET_USE_PATH = "/api/biz/customer-package-reset/use"
_TARGET_TYPE = "PERSONAL"

_PROVIDER_LABEL = "Z.ai Coding Plan"

# (type, unit, number) discriminators from the quota response; the reset-card
# groups address the same quotas by unit alone (3 five-hour, 6 weekly).
_LIMIT_LABELS = {
    ("TOKENS_LIMIT", 3, 5): "5-hour prompt pool",
    ("TOKENS_LIMIT", 6, 1): "Weekly quota",
    ("TIME_LIMIT", 5, 1): "Tool calls",
}
_CARD_GROUPS = (
    ("fiveHourResets", "FIVE_HOUR", "5-hour", 3),
    ("weekResets", "WEEK", "Weekly", 6),
)
_TOOL_CALLS_KEY = ("TIME_LIMIT", 5, 1)
_RESET_LABELS = dict(
    (reset_type, label) for _source, reset_type, label, _unit in _CARD_GROUPS)


class _AuthRefused(OSError):
    """Every supported authorization style was rejected before acting."""


def _text(value) -> str:
    return ascii(str(value))[1:-1]


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _integer(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _relative(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60
    if days:
        return f"{days} day{'' if days == 1 else 's'} {hours} hour{'' if hours == 1 else 's'}"
    if hours:
        return f"{hours} hour{'' if hours == 1 else 's'} {minutes} minute{'' if minutes == 1 else 's'}"
    return f"{minutes} minute{'' if minutes == 1 else 's'}"


def _parse_time(value):
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return moment.astimezone(datetime.timezone.utc)


def _epoch_text(value) -> str | None:
    """Render ``nextResetTime`` epoch milliseconds."""
    milliseconds = _integer(value)
    if milliseconds is None or milliseconds <= 0:
        return None
    moment = datetime.datetime.fromtimestamp(
        milliseconds / 1000, tz=datetime.timezone.utc)
    seconds = (moment - datetime.datetime.now(
        datetime.timezone.utc)).total_seconds()
    if seconds < 0:
        return f"resets {moment:%Y-%m-%d %H:%M} UTC ({_relative(-seconds)} ago)"
    return (
        f"resets {moment:%Y-%m-%d %H:%M} UTC (in {_relative(seconds)})")


def _expiry_text(value) -> str:
    moment = _parse_time(value)
    if moment is None:
        return "no expiry" if value in (None, "") else f"expires {_text(value)}"
    now = datetime.datetime.now(datetime.timezone.utc)
    seconds = (moment - now).total_seconds()
    if seconds <= 0:
        return f"expired {_relative(-seconds)} ago"
    return (
        f"expires {moment.date().isoformat()} (in {_relative(seconds)})")


def _credential(context):
    spec = getattr(context.config, "auth_spec", None)
    if spec is None or spec.credential is None:
        raise authentications.CredentialUnavailable(
            "no Z.ai credential is selected")
    return spec.credential


def _applies(context) -> bool:
    spec = getattr(context.config, "auth_spec", None)
    if spec is None or spec.credential is None:
        return False
    return provider_controls.connection_origin(context) in ZAI_ORIGINS


def _url(context, path: str) -> str:
    return provider_controls.connection_origin(context) + path


def _auth_specs(context):
    """The styles one request may try, in order, scoped to this origin."""
    origin = frozenset({provider_controls.connection_origin(context)})
    credential = _credential(context)
    return (
        ("key", authentications.AuthSpec(
            credential, "custom", header_name="Authorization",
            authorized_origins=origin)),
        ("bearer", authentications.AuthSpec(
            credential, "bearer", authorized_origins=origin)),
    )


def _refused_payload(payload) -> bool:
    return isinstance(payload, dict) and payload.get("code") in (1001, 1003)


async def _authorized_json(context, method, url, *, body=None,
                           content_type=None, retry=True):
    """Send one request, trying each supported authorization style.

    Returns the checked payload and the style that succeeded.  A refusal
    under one style falls through to the next; any other error, and any
    non-refused response, is final.
    """
    attempts = provider_controls.READ_RETRY_MAX_ATTEMPTS if retry else 1
    for style, spec in _auth_specs(context):
        response = await provider_controls.authorized_request(
            context, spec, method, url,
            body=body, content_type=content_type,
            retry_max_attempts=attempts)
        if response.status in (401, 403):
            # Refused at the HTTP layer; json_document would treat this as
            # a hard error, but another style may still be accepted.
            continue
        payload = provider_controls.json_document(response)
        if _refused_payload(payload):
            continue
        if not isinstance(payload, dict):
            raise OSError("z.ai response was not a JSON object")
        if payload.get("success") is not True and payload.get("code") != 200:
            code = payload.get("code")
            message = payload.get("msg") or payload.get("message")
            raise OSError(f"z.ai API error {code}: {_text(message)}")
        return payload, style
    raise _AuthRefused(
        "z.ai did not accept the Coding Plan API key for this endpoint "
        "(tried the raw key and Bearer); the console reaches these "
        "endpoints with a signed-in session token, which is not available")


def _limit_key(limit):
    return (
        limit.get("type"),
        _integer(limit.get("unit")),
        _integer(limit.get("number")),
    )


def _limit_line(limit) -> str | None:
    if not isinstance(limit, dict):
        return None
    percentage = _number(limit.get("percentage"))
    if percentage is None:
        return None
    key = _limit_key(limit)
    label = _LIMIT_LABELS.get(key)
    if label is None:
        label = (
            f"{_text(key[0] or 'limit')} "
            f"(unit={key[1]} number={key[2]})")
    line = (
        f"  {label}: {percentage:g}% used, "
        f"{max(0.0, 100 - percentage):g}% remaining")
    if key == _TOOL_CALLS_KEY:
        current = _integer(limit.get("currentValue"))
        total = _integer(limit.get("usage"))
        if current is not None and total is not None:
            line += f", {current} of {total} used"
    reset = _epoch_text(limit.get("nextResetTime"))
    if reset is not None:
        line += f"; {reset}"
    return line


def _usage_lines(data) -> list[str]:
    lines = [f"{_PROVIDER_LABEL} - live"]
    level = data.get("level")
    if isinstance(level, str) and level:
        lines.append(f"Plan: {_text(level)}")
    limits = data.get("limits") if isinstance(data, dict) else None
    rendered = [
        line for line in (
            _limit_line(limit) for limit in (
                limits if isinstance(limits, list) else []))
        if line is not None]
    if rendered:
        lines.extend(rendered)
    else:
        lines.append("No usage limits were reported.")
    return lines


def _cards(data, source_key):
    entries = data.get(source_key) if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return []
    cards = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        record = entry.get("recordId")
        if _integer(record) is None and not isinstance(record, str):
            continue
        cards.append(entry)
    return cards


def _usable_cards(data, source_key):
    cards = _cards(data, source_key)
    return sorted(
        (card for card in cards if card.get("available") is True),
        key=lambda card: str(card.get("expireTime") or ""))


def _reset_lines(data) -> list[str]:
    lines = [f"{_PROVIDER_LABEL} - live"]
    available = 0
    expired = 0
    reported = False
    for source_key, _reset_type, label, _unit in _CARD_GROUPS:
        cards = _cards(data, source_key)
        usable = _usable_cards(data, source_key)
        reported = reported or bool(cards)
        available += len(usable)
        expired += len(cards) - len(usable)
        lines.append(f"  {label} reset cards: {len(usable)} available")
        for index, card in enumerate(usable, start=1):
            lines.append(
                f"    {index}. {_expiry_text(card.get('expireTime'))}")
    if not reported:
        lines.append("  (no reset cards were reported)")
    lines.append(f"  Total: {available} available, {expired} expired")
    return lines


def _use_action(context, reset_type, label, card):
    record_id = card["recordId"]
    expiry = _expiry_text(card.get("expireTime"))
    return provider_controls.ControlAction(
        id=f"use:{record_id}",
        title=f"Use {label} reset card ({expiry})",
        confirm=f"Use the {label} reset card ({expiry})?",
        run=lambda: _use(context, reset_type, record_id),
    )


async def _read_usage(context) -> provider_controls.ControlResult:
    url = _url(context, _QUOTA_PATH)
    payload, style = await _authorized_json(context, "GET", url)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    return provider_controls.ControlResult(
        lines=tuple(_usage_lines(data)),
        document={
            "endpoint": url,
            "auth": style,
            "usage": payload,
        },
    )


async def _read_resets(context) -> provider_controls.ControlResult:
    url = _url(context, _RESET_LIST_PATH)
    payload, style = await _authorized_json(context, "GET", url)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    actions = []
    documents = []
    for source_key, reset_type, label, _unit in _CARD_GROUPS:
        for card in _cards(data, source_key):
            documents.append({
                "resetType": reset_type,
                "recordId": card.get("recordId"),
                "expireTime": card.get("expireTime"),
                "available": card.get("available") is True,
            })
            if card.get("available") is True:
                actions.append(
                    _use_action(context, reset_type, label, card))
    return provider_controls.ControlResult(
        lines=tuple(_reset_lines(data)),
        document={
            "endpoint": url,
            "auth": style,
            "cards": documents,
        },
        actions=tuple(actions),
    )


async def _use(context, reset_type: str,
               record_id) -> provider_controls.ControlResult:
    label = _RESET_LABELS[reset_type]
    # The request id is generated before the request and reported on an
    # ambiguous failure, mirroring the console, which reuses one id across
    # retries of a single click so the server can deduplicate them.
    request_id = str(uuid.uuid4())
    body = json.dumps({
        "targetType": _TARGET_TYPE,
        "resetType": reset_type,
        "recordId": record_id,
        "requestId": request_id,
    }).encode("utf-8")
    try:
        payload, _style = await _authorized_json(
            context, "POST", _url(context, _RESET_USE_PATH),
            body=body, content_type="application/json", retry=False)
    except _AuthRefused as error:
        return provider_controls.ControlResult(lines=(
            f"Reset card not used: {_text(error)}",))
    except Exception as error:  # noqa: BLE001 - reported, never retried
        return provider_controls.ControlResult(lines=(
            f"Could not confirm the reset (request {request_id}).",
            f"Check the current status before retrying: {_text(error)}",))
    return provider_controls.ControlResult(
        lines=(
            f"Reset card used: {label} quota is back to 100% "
            f"(request {request_id}).",),
        document={
            "endpoint": _url(context, _RESET_USE_PATH),
            "request_id": request_id,
            "reset_type": reset_type,
            "record_id": record_id,
            "result": payload,
        },
    )


USAGE = provider_controls.ControlSpec(
    id="usage",
    title="Usage",
    description="current quota windows and reset times",
    applies=_applies,
    read=_read_usage,
)

RESETS = provider_controls.ControlSpec(
    id="resets",
    title="Reset cards",
    description="quota reset cards, with redemption",
    applies=_applies,
    read=_read_resets,
)

CONTROLS = (USAGE, RESETS)
