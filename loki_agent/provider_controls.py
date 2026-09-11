"""Provider-dependent account controls, gated by the active connection.

Controls are declared locally in code.  A provider response only supplies the
values a known control renders; it can never add, name, or parameterize a
control.  That keeps a changed or hostile backend from turning itself into new
executable authority.

This module resolves which controls apply to the current connection, owns the
authenticated request path the controls share, and defines the data exchanged
with the front-end.  All interaction (menus, confirmation, output) belongs to
the front-end, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from . import authentications, http_client

# The single in-session entry point.  Controls are addressed as tokens under
# it, so registering a control never adds a global command.
ENTRY = "/account"

_REQUEST_TIMEOUT_S = 30
_REQUEST_MAX_BYTES = 1024 * 1024
# Idempotent reads may be retried on a transient transport failure.  A
# mutation must not be: it supplies its own idempotency instead.
READ_RETRY_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class ControlAction:
    """A locally-defined operation offered after a control's read.

    ``id`` is the token a user can type after the control id; ``confirm`` is
    the text of the mandatory confirmation prompt.
    """

    id: str
    title: str
    confirm: str
    run: Callable[[], Awaitable["ControlResult"]]


@dataclass(frozen=True)
class ControlResult:
    """Rendered lines plus an optional machine-readable document."""

    lines: tuple[str, ...]
    document: dict | None = None
    actions: tuple[ControlAction, ...] = ()


@dataclass(frozen=True)
class ControlContext:
    """What a control may use.  Carries no credential value itself."""

    config: object | None
    credential_authority: object | None = None
    request: object | None = None


@dataclass(frozen=True)
class ControlSpec:
    id: str
    title: str
    description: str
    applies: Callable[[ControlContext], bool]
    read: Callable[[ControlContext], Awaitable[ControlResult]]


def _registry() -> tuple[ControlSpec, ...]:
    # Imported lazily so provider modules can import these types.
    from . import deepseek_controls, openai_controls, openrouter_controls
    return (
        *openai_controls.CONTROLS,
        *openrouter_controls.CONTROLS,
        *deepseek_controls.CONTROLS,
    )


def available_controls(context: ControlContext) -> list[ControlSpec]:
    return [spec for spec in _registry() if spec.applies(context)]


def control_names(spec: ControlSpec) -> set[str]:
    """Tokens that select one control: its id and a slug of its title.

    The menu shows titles, so a user may type a visible word ("limit") rather
    than the id ("resets"); both must select the same control.
    """
    slug = "-".join(
        "".join(character for character in word.lower()
                if character.isalnum())
        for word in spec.title.split())
    return {spec.id, slug} - {""}


def find_control(context: ControlContext, control_id: str) -> ControlSpec | None:
    token = control_id.strip().lower()
    if not token:
        return None
    available = available_controls(context)
    for spec in available:
        if token in control_names(spec):
            return spec
    prefixed = [
        spec for spec in available
        if any(name.startswith(token) for name in control_names(spec))
    ]
    return prefixed[0] if len({id(spec) for spec in prefixed}) == 1 else None


def live_hint(context: ControlContext) -> str | None:
    """A static /status pointer.  Performs no request."""
    specs = available_controls(context)
    if not specs:
        return None
    return "Live account data: " + ", ".join(
        f"{ENTRY} {spec.id}" for spec in specs)


def connection_origin(context: ControlContext) -> str | None:
    """The active chat endpoint's origin, or None if there is not one."""
    provider = getattr(context.config, "chat_provider", None)
    url = getattr(provider, "chat_url", None)
    if not isinstance(url, str) or not url:
        return None
    try:
        return authentications.authorization_origin(url)
    except authentications.CredentialUnavailable:
        return None


async def authorized_request(context: ControlContext, spec, method, url, *,
                             body=None, content_type=None,
                             retry_max_attempts=1):
    """Lease, send one authenticated request, recover at most one 401."""
    request = context.request or http_client.async_http_request
    base = {"Accept": "application/json"}
    if content_type is not None:
        base["Content-Type"] = content_type
    rejected_generation = None
    recovered = False
    while True:
        headers, lease = await authentications.authorized_request_headers(
            context.credential_authority,
            spec,
            url,
            base,
            rejected_generation,
        )
        kwargs = {
            "headers_in": headers,
            "timeout": _REQUEST_TIMEOUT_S,
            "max_bytes": _REQUEST_MAX_BYTES,
            "retry_max_attempts": retry_max_attempts,
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


def json_document(response) -> object:
    """Parse a successful account response, or raise a user-facing error."""
    if response.status >= 400:
        raise OSError(
            f"provider API returned HTTP {response.status} {response.reason}")
    if response.truncated:
        raise OSError("provider response exceeds its size limit")
    try:
        return json.loads(response.body.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as error:
        raise OSError(f"provider response is invalid: {error}") from error


def credential_spec(credential, *, scheme, authorized_origins=frozenset(),
                    authorized_urls=frozenset()):
    """An AuthSpec for one account control's own endpoints."""
    return authentications.AuthSpec(
        credential,
        scheme,
        authorized_origins=authorized_origins,
        authorized_urls=authorized_urls,
    )
