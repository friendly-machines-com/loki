"""Read live Anthropic usage for the active subscription's /account command.

The existing account dispatcher owns authentication, error handling and 401
recovery. It cannot infer the provider's quota or units from a Messages
response, so this control owns a separate read-only route and renderer. It
borrows a scoped access-token lease for one request and never stores credentials.
"""

from __future__ import annotations

import datetime
import math

from . import authentications, provider_controls


ANTHROPIC_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
ANTHROPIC_ACCOUNT_URLS = frozenset({ANTHROPIC_USAGE_URL})

_WINDOW_TITLES = {
    "five_hour": "Session window (5 hour)",
    "seven_day": "Weekly window (all models)",
    "seven_day_sonnet": "Weekly window (Sonnet)",
    "seven_day_opus": "Weekly window (Opus)",
}


def _applies(context) -> bool:
    spec = getattr(context.config, "auth_spec", None)
    return (spec is not None and spec.scheme == "anthropic-subscription"
            and spec.credential == authentications.CredentialRef.anthropic_subscription())


def _reset_text(value) -> str:
    moment = provider_controls.account_time_utc(value)
    if moment is None:
        return "reset time unknown"
    stamp = f"{moment:%Y-%m-%d %H:%M:%S} UTC"
    seconds = (moment - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    if seconds <= 0:
        # The endpoint may still report an elapsed window. It is not a new quota.
        return f"reported reset at {stamp} has passed"
    return f"resets {stamp} (in {provider_controls.relative_duration(seconds)})"


def _window_line(title, window) -> str | None:
    if not isinstance(window, dict):
        return None
    # The usage document is in percent; rate-limit headers use fractions.
    # Scaling values below one would misreport small but valid utilizations.
    used = window.get("utilization")
    if type(used) not in {int, float} or not 0 <= used <= 100:
        return f"  {title}: reported an unrecognized utilization"
    return (f"  {title}: {used:g}% used, {100 - used:g}% remaining; "
            f"{_reset_text(window.get('resets_at'))}")


def _usage_lines(payload) -> list[str]:
    lines = ["Anthropic subscription - live"]
    for key, title in _WINDOW_TITLES.items():
        line = _window_line(title, payload.get(key))
        if line is not None:
            lines.append(line)
    if len(lines) == 1:
        lines.append("No usage windows were reported.")
    lines.extend(_extra_usage_lines(payload.get("extra_usage")))
    return lines


def _amount(value) -> float | None:
    if type(value) not in {int, float} or value < 0:
        return None
    try:
        amount = float(value)
    except OverflowError:
        return None
    return amount if math.isfinite(amount) else None


def _extra_usage_lines(extra) -> list[str]:
    if not isinstance(extra, dict) or extra.get("is_enabled") is not True:
        return []
    # A missing field is not evidence of an unlimited plan.
    if "monthly_limit" in extra and extra["monthly_limit"] is None:
        return ["  Extra usage: Unlimited"]
    limit = _amount(extra.get("monthly_limit"))
    used = _amount(extra.get("used_credits"))
    if limit is None or limit == 0 or used is None:
        return ["  Extra usage: enabled; spending unavailable"]
    # Spending is in cents. Derive the percentage from amounts in the same
    # units rather than guessing how an optional utilization field is scaled.
    line = f"  Extra usage: ${used / 100:.2f} / ${limit / 100:.2f} spent"
    ratio = used / limit * 100
    if math.isfinite(ratio):
        line += f"; {ratio:g}% used"
    return [line]


async def _read_usage(context) -> provider_controls.ControlResult:
    if not _applies(context):
        raise authentications.CredentialUnavailable("Select an Anthropic subscription for account usage")
    spec = provider_controls.credential_spec(
        context.config.auth_spec.credential,
        scheme="anthropic-subscription", authorized_urls=ANTHROPIC_ACCOUNT_URLS,
    )
    response = await provider_controls.authorized_request(
        context, spec, "GET", ANTHROPIC_USAGE_URL,
        retry_max_attempts=provider_controls.READ_RETRY_MAX_ATTEMPTS,
    )
    payload = provider_controls.json_document(response)
    if not isinstance(payload, dict):
        raise OSError("Anthropic did not return a usage object")
    return provider_controls.ControlResult(
        lines=_usage_lines(payload),
        document={"endpoint": ANTHROPIC_USAGE_URL, "usage": payload},
    )


USAGE = provider_controls.ControlSpec(
    id="usage",
    title="Usage",
    description="current session and weekly limits with reset times",
    applies=_applies,
    read=_read_usage,
    destinations=lambda context: ANTHROPIC_ACCOUNT_URLS,
)
