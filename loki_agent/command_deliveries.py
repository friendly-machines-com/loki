"""Which slash commands bypass the prompt queue: the hypervisor monitor plane.

Immediate commands answer from monitor-plane state only. They never enter the
transcript, the model context, or a turn, and their output is rendered by the
frontend alone (invariant H: monitor output never becomes conversation).
Command text is classified here and nowhere else: frontends consult
``terminal_immediate``/``classify`` instead of comparing command names at
their admission points.

This table owns terminal delivery only. ACP commands use ordinary prompt
admission in ``acp_worker._prepare_prompt``, which rejects overlap rather
than queueing. There is deliberately no unused ACP delivery flag here:
``agent_message_chunk`` has no message boundary, so ACP immediacy needs an
out-of-band output channel before that gate can be changed (decision D2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


PROMPT = "prompt"
IMMEDIATE = "immediate"


@dataclass(frozen=True)
class Delivery:
    """Terminal command delivery; ordinary prompts enter the input FIFO."""

    terminal: str = PROMPT


@dataclass(frozen=True)
class ParsedCommand:
    """One recognized command line, before any frontend handles it."""

    name: str
    argument: str
    delivery: Delivery


def _exact_forms(name: str, forms: frozenset[str]) -> Callable[[str], str | None]:
    """Recognize ``/name`` plus an argument from FORMS (whitespace-normalized).

    Unknown arguments return None: they keep whatever behavior the frontend
    already gives unrecognized input, instead of silently becoming monitor
    output.
    """
    prefix = "/" + name

    def recognize(text: str) -> str | None:
        if text == prefix:
            argument = ""
        elif text.startswith(prefix + " "):
            argument = text[len(prefix):].strip()
        else:
            return None
        return argument if argument in forms else None

    return recognize


def _any_argument(name: str) -> Callable[[str], str | None]:
    """Recognize ``/name`` with any (or no) argument."""
    prefix = "/" + name

    def recognize(text: str) -> str | None:
        if text == prefix:
            return ""
        if text.startswith(prefix + " "):
            return text[len(prefix):].strip()
        return None

    return recognize


def _account_read_argument(name: str) -> Callable[[str], str | None]:
    """Recognize only the read-only control form of ``/account``.

    Exactly one control token (``--json`` flags aside) is immediate: it reads
    and prints. The bare listing and ``CONTROL ACTION`` forms stay queued --
    they interact (numbered choices, confirmations) and must not take the
    reader from a running turn.
    """
    prefix = "/" + name

    def recognize(text: str) -> str | None:
        if not text.startswith(prefix + " "):
            return None
        tokens = [token for token in text[len(prefix):].split()
                  if token != "--json"]
        if len(tokens) != 1:
            return None
        return tokens[0]

    return recognize


@dataclass(frozen=True)
class _CommandSpec:
    delivery: Delivery
    recognize: Callable[[str], str | None]


_COMMANDS: dict[str, _CommandSpec] = {
    # /ps is job control: sync, await-free, immediate in any form.
    "ps": _CommandSpec(
        Delivery(terminal=IMMEDIATE), _any_argument("ps")),
    # The six inspected/saved forms; other arguments stay unrecognized input.
    "status": _CommandSpec(
        Delivery(terminal=IMMEDIATE),
        _exact_forms("status", frozenset({
            "", "--json", "all", "all --json", "--json all", "save"}))),
    # Only the direct read-only control form.
    "account": _CommandSpec(
        Delivery(terminal=IMMEDIATE), _account_read_argument("account")),
    # /queue inspects the FIFO and staged images: a snapshot while any number
    # of prompts waits behind a running turn.
    "queue": _CommandSpec(
        Delivery(terminal=IMMEDIATE), _any_argument("queue")),
}


def classify(text: str) -> ParsedCommand | None:
    """Classify a submitted line; None when it is not a declared command.

    The slash prefix is required and matching is case-sensitive and
    word-boundary safe: ``/statusx`` and ``/ps-file`` never match.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    name = stripped[1:].partition(" ")[0]
    spec = _COMMANDS.get(name)
    if spec is None:
        return None
    return ParsedCommand(name=name, argument=stripped[len(name) + 1:].strip(),
                         delivery=spec.delivery)


def terminal_immediate(text: str) -> ParsedCommand | None:
    """The command when TEXT bypasses the terminal prompt queue, else None."""
    parsed = classify(text)
    if parsed is None or parsed.delivery.terminal != IMMEDIATE:
        return None
    if _COMMANDS[parsed.name].recognize(text.strip()) is None:
        return None
    return parsed
