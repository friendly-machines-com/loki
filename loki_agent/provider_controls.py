"""Provider-dependent account controls, gated by the active connection.

Controls are declared locally in code.  A provider response only supplies the
values a known control renders; it can never add, name, or parameterize a
control.  That keeps a changed or hostile backend from turning itself into new
executable authority.

This module resolves which controls apply to the current connection and owns
the data exchanged with the front-end.  All interaction (menus, confirmation,
output) belongs to the front-end, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

# The single in-session entry point.  Controls are addressed as tokens under
# it, so registering a control never adds a global command.
ENTRY = "/account"


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
    from . import openai_controls
    return openai_controls.CONTROLS


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
