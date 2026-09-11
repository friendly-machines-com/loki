"""OpenRouter account controls: per-key spend limit and account credits.

OpenRouter's ``/key`` endpoint is readable with the ordinary API key a chat
connection already uses, so it answers "will my next call be rejected" without
any extra credential.  The account-wide ``/credits`` totals additionally
require a management key; they are read only when the key reports itself as
one, so a normal key never issues a request that is guaranteed to be refused.
"""

from __future__ import annotations

from . import authentications, provider_controls


OPENROUTER_API_ORIGIN = "https://openrouter.ai"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"

_PROVIDER_LABEL = "OpenRouter"


def _text(value) -> str:
    return ascii(str(value))[1:-1]


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _usd(value: float) -> str:
    return f"${value:,.2f}"


def _credential(context):
    spec = getattr(context.config, "auth_spec", None)
    if spec is None or spec.credential is None:
        raise authentications.CredentialUnavailable(
            "no OpenRouter credential is selected")
    return spec.credential


def _applies(context) -> bool:
    spec = getattr(context.config, "auth_spec", None)
    if spec is None or spec.credential is None:
        return False
    return (
        provider_controls.connection_origin(context) == OPENROUTER_API_ORIGIN)


def _spec(credential):
    return provider_controls.credential_spec(
        credential,
        scheme="bearer",
        authorized_origins=frozenset({OPENROUTER_API_ORIGIN}),
    )


def _key_lines(data) -> list[str]:
    lines = [f"{_PROVIDER_LABEL} - live"]
    limit = _number(data.get("limit"))
    remaining = _number(data.get("limit_remaining"))
    if limit is None:
        lines.append("  Spend limit: none set for this key")
    else:
        lines.append(f"  Spend limit: {_usd(limit)}")
        if remaining is not None:
            lines.append(f"  Remaining: {_usd(remaining)}")
    for key, label in (
            ("usage", "Usage (all time)"),
            ("usage_monthly", "Usage (this month)"),
            ("usage_weekly", "Usage (this week)"),
            ("usage_daily", "Usage (today)")):
        value = _number(data.get(key))
        if value is not None:
            lines.append(f"  {label}: {_usd(value)}")
    if isinstance(data.get("limit_reset"), str) and data["limit_reset"]:
        lines.append(f"  Limit reset: {_text(data['limit_reset'])}")
    if data.get("is_free_tier") is True:
        lines.append("  Free-tier key")
    return lines


def _credits_lines(payload) -> list[str]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return []
    total = _number(data.get("total_credits"))
    used = _number(data.get("total_usage"))
    lines = []
    if total is not None:
        lines.append(f"  Account credits: {_usd(total)}")
    if used is not None:
        lines.append(f"  Account usage: {_usd(used)}")
    if total is not None and used is not None:
        lines.append(f"  Account remaining: {_usd(total - used)}")
    return lines


async def _read_balance(context):
    credential = _credential(context)
    spec = _spec(credential)
    response = await provider_controls.authorized_request(
        context, spec, "GET", OPENROUTER_KEY_URL,
        retry_max_attempts=provider_controls.READ_RETRY_MAX_ATTEMPTS)
    payload = provider_controls.json_document(response)
    data = payload.get("data") if isinstance(payload, dict) else None
    data = data if isinstance(data, dict) else {}
    lines = _key_lines(data)
    document = {
        "endpoint": OPENROUTER_KEY_URL,
        "key": data,
    }
    if data.get("is_management_key") is True:
        try:
            credits_response = await provider_controls.authorized_request(
                context, spec, "GET", OPENROUTER_CREDITS_URL,
                retry_max_attempts=provider_controls.READ_RETRY_MAX_ATTEMPTS)
            credits = provider_controls.json_document(credits_response)
        except (OSError, authentications.CredentialError) as error:
            lines.append(f"  Account credits: unavailable ({_text(error)})")
        else:
            lines.extend(_credits_lines(credits))
            document["credits"] = (
                credits.get("data") if isinstance(credits, dict) else None)
    return provider_controls.ControlResult(
        lines=tuple(lines), document=document)


BALANCE = provider_controls.ControlSpec(
    id="balance",
    title="Balance",
    description="prepaid credit and per-key spend limit",
    applies=_applies,
    read=_read_balance,
)

CONTROLS = (BALANCE,)
