"""DeepSeek account control: prepaid balance.

Answers whether the account can still pay for the next call, using the same
API key the chat connection already uses.
"""

from __future__ import annotations

from . import authentications, provider_controls


DEEPSEEK_API_ORIGIN = "https://api.deepseek.com"
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"

_PROVIDER_LABEL = "DeepSeek"


def _text(value) -> str:
    return ascii(str(value))[1:-1]


def _credential(context):
    spec = getattr(context.config, "auth_spec", None)
    if spec is None or spec.credential is None:
        raise authentications.CredentialUnavailable(
            "no DeepSeek credential is selected")
    return spec.credential


def _applies(context) -> bool:
    spec = getattr(context.config, "auth_spec", None)
    if spec is None or spec.credential is None:
        return False
    return (
        provider_controls.connection_origin(context) == DEEPSEEK_API_ORIGIN)


def _spec(credential):
    return provider_controls.credential_spec(
        credential,
        scheme="bearer",
        authorized_origins=frozenset({DEEPSEEK_API_ORIGIN}),
    )


def _balance_lines(payload) -> list[str]:
    lines = [f"{_PROVIDER_LABEL} - live"]
    available = payload.get("is_available") if isinstance(payload, dict) else None
    if isinstance(available, bool):
        lines.append(
            "  Balance sufficient for API calls: "
            + ("yes" if available else "no"))
    balance_infos = payload.get("balance_infos") if isinstance(payload, dict) else None
    if not isinstance(balance_infos, list) or not balance_infos:
        lines.append("  (no balance reported)")
        return lines
    for info in balance_infos:
        if not isinstance(info, dict):
            continue
        currency = _text(info.get("currency") or "?")
        total = _text(info.get("total_balance") or "?")
        granted = _text(info.get("granted_balance") or "?")
        topped_up = _text(info.get("topped_up_balance") or "?")
        lines.append(
            f"  {currency}: {total} total "
            f"({granted} granted, {topped_up} topped up)")
    return lines


async def _read_balance(context):
    credential = _credential(context)
    response = await provider_controls.authorized_request(
        context, _spec(credential), "GET", DEEPSEEK_BALANCE_URL,
        retry_max_attempts=provider_controls.READ_RETRY_MAX_ATTEMPTS)
    payload = provider_controls.json_document(response)
    if not isinstance(payload, dict):
        payload = {}
    return provider_controls.ControlResult(
        lines=tuple(_balance_lines(payload)),
        document={
            "endpoint": DEEPSEEK_BALANCE_URL,
            "balance": payload,
        },
    )


BALANCE = provider_controls.ControlSpec(
    id="balance",
    title="Balance",
    description="prepaid account balance",
    applies=_applies,
    read=_read_balance,
)

CONTROLS = (BALANCE,)
