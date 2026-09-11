"""OpenAI ChatGPT subscription account controls.

These controls read usage windows and banked rate-limit resets, and redeem a
reset.  Their endpoints are declared here and authorized through their own
``AuthSpec`` URL set, separate from the Codex inference endpoints the same
credential uses, so a defect in an account control cannot widen where the
inference credential may be sent.
"""

from __future__ import annotations

import datetime
import json
import uuid

from . import authentications, http_client, provider_controls


OPENAI_CHATGPT_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
OPENAI_CHATGPT_RESET_CREDITS_URL = (
    "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits")
OPENAI_CHATGPT_CONSUME_RESET_URL = (
    OPENAI_CHATGPT_RESET_CREDITS_URL + "/consume")
OPENAI_CHATGPT_ACCOUNT_URLS = frozenset({
    OPENAI_CHATGPT_USAGE_URL,
    OPENAI_CHATGPT_RESET_CREDITS_URL,
    OPENAI_CHATGPT_CONSUME_RESET_URL,
})

_ACCOUNT_TIMEOUT_S = 30
_ACCOUNT_MAX_BYTES = 1024 * 1024
_PROVIDER_LABEL = "OpenAI ChatGPT subscription"


def _applies(context) -> bool:
    spec = getattr(context.config, "auth_spec", None)
    return getattr(spec, "scheme", None) == "openai-subscription"


def _text(value) -> str:
    """Escape a provider-supplied string for terminal output."""
    return ascii(str(value))[1:-1]


def _parse_time(value):
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return moment.astimezone(datetime.timezone.utc)


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


def _duration(minutes: int) -> str:
    if minutes % 1440 == 0:
        amount, unit = minutes // 1440, "day"
    elif minutes % 60 == 0:
        amount, unit = minutes // 60, "hour"
    else:
        amount, unit = minutes, "minute"
    return f"{amount} {unit}{'' if amount == 1 else 's'}"


def _expiry_text(value) -> str:
    moment = _parse_time(value)
    if moment is None:
        return "no expiry" if value in (None, "") else f"expires {_text(value)}"
    now = datetime.datetime.now(datetime.timezone.utc)
    seconds = (moment - now).total_seconds()
    if seconds <= 0:
        return f"expired {_relative(-seconds)} ago"
    return (
        f"expires {moment.date().isoformat()} "
        f"(in {_relative(seconds)})")


def _window_minutes(window) -> int | None:
    for key, scale in (
            ("window_minutes", 1),
            ("limit_window_seconds", 60),
            ("window_seconds", 60)):
        value = window.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > 0:
                return max(1, int(value // scale))
    return None


def _window_line(label: str, window) -> str | None:
    used = window.get("used_percent")
    if not isinstance(used, (int, float)) or isinstance(used, bool):
        return None
    used = float(used)
    if not 0 <= used <= 100:
        return None
    line = f"  {label}: {used:g}% used, {100 - used:g}% remaining"
    minutes = _window_minutes(window)
    if minutes is not None:
        line += f" - {_duration(minutes)}"
    reset = window.get("reset_after_seconds")
    if not isinstance(reset, (int, float)) or isinstance(reset, bool):
        reset = window.get("resets_in_seconds")
    if isinstance(reset, (int, float)) and not isinstance(reset, bool):
        line += f"; resets in {_relative(reset)}"
    return line


def _window_lines(payload) -> list[str]:
    rate_limit = payload.get("rate_limit") if isinstance(payload, dict) else None
    if not isinstance(rate_limit, dict):
        return []
    lines = []
    for key, label in (
            ("primary_window", "Primary window"),
            ("secondary_window", "Secondary window")):
        window = rate_limit.get(key)
        if not isinstance(window, dict):
            continue
        line = _window_line(label, window)
        if line is not None:
            lines.append(line)
    return lines


def _usage_lines(payload) -> list[str]:
    lines = [f"{_PROVIDER_LABEL} - live"]
    plan = payload.get("plan_type") if isinstance(payload, dict) else None
    if isinstance(plan, str) and plan:
        lines.append(f"Plan: {_text(plan)}")
    windows = _window_lines(payload)
    if windows:
        lines.extend(windows)
    else:
        lines.append("No usage windows were reported.")
    credits = (payload.get("rate_limit_reset_credits")
               if isinstance(payload, dict) else None)
    if isinstance(credits, dict):
        count = credits.get("available_count")
        if isinstance(count, int) and not isinstance(count, bool):
            lines.append(f"Banked limit resets: {count} available")
    return lines


def _is_available(credit) -> bool:
    return isinstance(credit, dict) and credit.get("status") == "available"


def _reset_lines(payload, credits) -> list[str]:
    lines = [f"{_PROVIDER_LABEL} - live"]
    available = (payload.get("available_count")
                 if isinstance(payload, dict) else None)
    if isinstance(available, int) and not isinstance(available, bool):
        lines.append(f"Banked limit resets: {available} available")
    else:
        lines.append("Banked limit resets:")
    if not credits:
        lines.append("  (none reported)")
        return lines
    for index, credit in enumerate(credits, start=1):
        if not isinstance(credit, dict):
            continue
        title = _text(
            credit.get("title") or credit.get("id") or "reset credit")
        status = credit.get("status")
        suffix = (
            ""
            if not isinstance(status, str) or status == "available"
            else f" [{_text(status)}]")
        lines.append(
            f"  {index}. {title}{suffix} - "
            f"{_expiry_text(credit.get('expires_at'))}")
    return lines


def _credit_document(credit) -> dict:
    if not isinstance(credit, dict):
        return {}
    return {
        "id": credit.get("id"),
        "title": credit.get("title"),
        "status": credit.get("status"),
        "granted_at": credit.get("granted_at"),
        "expires_at": credit.get("expires_at"),
    }


def _auth_spec(credential):
    return authentications.AuthSpec(
        credential,
        "openai-subscription",
        authorized_urls=OPENAI_CHATGPT_ACCOUNT_URLS,
    )


async def _request(context, method, url, *, body=None, content_type=None):
    spec = getattr(context.config, "auth_spec", None)
    if spec is None or spec.credential is None:
        raise authentications.CredentialUnavailable(
            "no OpenAI subscription credential is selected")
    account_spec = _auth_spec(spec.credential)
    request = context.request or http_client.async_http_request
    base = {"Accept": "application/json"}
    if content_type is not None:
        base["Content-Type"] = content_type
    rejected_generation = None
    recovered = False
    while True:
        headers, lease = await authentications.authorized_request_headers(
            context.credential_authority,
            account_spec,
            url,
            base,
            rejected_generation,
        )
        kwargs = {
            "headers_in": headers,
            "timeout": _ACCOUNT_TIMEOUT_S,
            "max_bytes": _ACCOUNT_MAX_BYTES,
            # Account operations are not automatically retried: the credential
            # refresh below is the only replay, and a mutation supplies its own
            # idempotency key instead.
            "retry_max_attempts": 1,
        }
        if body is not None:
            kwargs["body"] = body
        response = await request(method, url, **kwargs)
        if (response.status == 401
                and lease is not None
                and lease.refreshable
                and not recovered):
            # Match inference recovery: a rejected generation refreshes once.
            rejected_generation = lease.generation
            recovered = True
            continue
        break
    return response


def _json_body(response):
    if response.status >= 400:
        raise OSError(
            f"account API returned HTTP {response.status} {response.reason}")
    if response.truncated:
        raise OSError("account response exceeds its size limit")
    try:
        return json.loads(response.body.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as error:
        raise OSError(f"account response is invalid: {error}") from error


async def _read_usage(context) -> provider_controls.ControlResult:
    response = await _request(context, "GET", OPENAI_CHATGPT_USAGE_URL)
    payload = _json_body(response)
    return provider_controls.ControlResult(
        lines=tuple(_usage_lines(payload)),
        document={
            "endpoint": OPENAI_CHATGPT_USAGE_URL,
            "usage": payload,
        },
    )


async def _read_resets(context) -> provider_controls.ControlResult:
    response = await _request(context, "GET", OPENAI_CHATGPT_RESET_CREDITS_URL)
    payload = _json_body(response)
    credits = payload.get("credits") if isinstance(payload, dict) else None
    credits = credits if isinstance(credits, list) else []
    actions = tuple(
        _redeem_action(context, credit)
        for credit in credits
        if _is_available(credit) and isinstance(credit.get("id"), str))
    return provider_controls.ControlResult(
        lines=tuple(_reset_lines(payload, credits)),
        document={
            "endpoint": OPENAI_CHATGPT_RESET_CREDITS_URL,
            "available_count": (
                payload.get("available_count")
                if isinstance(payload, dict) else None),
            "credits": [_credit_document(credit) for credit in credits],
        },
        actions=actions,
    )


def _redeem_action(context, credit) -> provider_controls.ControlAction:
    credit_id = credit["id"]
    title = _text(credit.get("title") or credit_id)
    return provider_controls.ControlAction(
        id=f"redeem:{credit_id}",
        title=f"Redeem {title}",
        confirm=(
            f"Redeem {title}? This clears the exhausted usage window(s) now "
            "and cannot be undone."),
        run=lambda credit_id=credit_id: _redeem(context, credit_id),
    )


def _ineligible_detail(response) -> str | None:
    try:
        payload = json.loads(response.body.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        return None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        code = detail.get("code")
        return code if isinstance(code, str) else None
    return None


async def _redeem(context, credit_id: str) -> provider_controls.ControlResult:
    # The request id is generated before the request and reported on an
    # ambiguous failure so a retry can be checked rather than guessed.
    request_id = str(uuid.uuid4())
    body = json.dumps({
        "credit_id": credit_id,
        "redeem_request_id": request_id,
    }).encode("utf-8")
    try:
        response = await _request(
            context, "POST", OPENAI_CHATGPT_CONSUME_RESET_URL,
            body=body, content_type="application/json")
    except Exception as error:  # noqa: BLE001 - reported, never retried
        return provider_controls.ControlResult(lines=(
            f"Could not confirm the reset (request {request_id}).",
            f"Check the current status before retrying: {_text(error)}",
        ))
    if response.status == 403:
        detail = _ineligible_detail(response)
        if detail == "rate_limit_reset_ineligible":
            return provider_controls.ControlResult(lines=(
                "The account is not currently eligible for a reset; "
                "no window is exhausted.",))
        return provider_controls.ControlResult(lines=(
            f"Reset refused: HTTP 403 {_text(response.reason)}",))
    if not 200 <= response.status < 300:
        # An error status leaves the outcome ambiguous, so report the request
        # id instead of implying the reset did or did not happen.
        return provider_controls.ControlResult(lines=(
            f"Could not confirm the reset (request {request_id}).",
            f"Provider returned HTTP {response.status} "
            f"{_text(response.reason)}.",
        ))
    payload = _json_body(response)
    code = payload.get("code") if isinstance(payload, dict) else None
    windows = payload.get("windows_reset") if isinstance(payload, dict) else None
    if code == "reset":
        count = windows if isinstance(windows, int) else 0
        line = f"Reset redeemed: {count} window(s) reset."
    elif code == "already_redeemed":
        line = "Already redeemed (no change)."
    elif code == "nothing_to_reset":
        line = "Nothing to reset; the usage windows were already reset."
    elif code == "no_credit":
        line = "No available reset credit."
    else:
        line = f"Reset result: {_text(code)}"
    return provider_controls.ControlResult(
        lines=(line,),
        document={
            "redeem_request_id": request_id,
            "code": code,
            "windows_reset": windows,
        },
    )


USAGE = provider_controls.ControlSpec(
    id="usage",
    title="Usage",
    description="current usage windows and banked reset count",
    applies=_applies,
    read=_read_usage,
)

RESETS = provider_controls.ControlSpec(
    id="resets",
    title="Limit resets",
    description="banked reset credits, with redemption",
    applies=_applies,
    read=_read_resets,
)

CONTROLS = (USAGE, RESETS)
