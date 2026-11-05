"""One user-turn recovery decision shared by request replay and continuation.

Request executors already own prepared bodies and idempotency keys, but cannot
continue from an incomplete canonical response. The model/tool loop owns that
history, but must not stack another policy/budget on those request executors.
Terminal and ACP create one TurnRecovery and lend it to both execution points.
The existing command-hook runner supplies policy; this module neither spawns
processes nor selects filenames, credentials, endpoints, or process cwd.
"""

import asyncio
from dataclasses import dataclass
import math
import sys

from . import formats


# model_requests.failure_event classifies request exceptions and the decoded
# responses preserved by the tool loop. TurnRecovery offers them to policy; final
# turn notification also uses this set, adding its local transcript-error case.
# Membership means "offer recovery", not "this failure is transient or safe".
MODEL_FAILURE_TYPES = frozenset({
    "api_error", "network_error", "stream_error", "provider_error",
    "response_failed", "response_incomplete",
})


class Cancelled(Exception):
    """Cooperative cancellation of inference, hook policy, or its backoff.

    Request executors and TurnRecovery raise this; the model/tool loop reports
    the existing response_cancelled outcome. asyncio.CancelledError remains
    distinct: task/shutdown cancellation must propagate after resource cleanup.
    """


@dataclass(frozen=True)
class RetryDecision:
    """Validated hook result, not another independently mutable action/delay pair.

    ToolHookPipeline constructs this from JSON; TurnRecovery consumes it.
    delay_seconds=None finishes the original outcome, whereas zero explicitly
    authorizes an immediate retry (still subject to cancellation). Finite,
    nonnegative numbers are the only waits the active turn may accept.
    """

    delay_seconds: float | None

    def __post_init__(self):
        delay = self.delay_seconds
        if delay is None:
            return
        try:
            valid = (not isinstance(delay, bool)
                     and isinstance(delay, int | float)
                     and delay >= 0 and math.isfinite(delay))
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("delay_seconds must be a finite nonnegative number")

    @classmethod
    def from_output(cls, value):
        """Empty output defers; an explicit decision stops the hook chain."""
        if not value:
            return None
        if value == {"action": "finish"}:
            return cls(None)
        if (set(value) != {"action", "delay_seconds"}
                or value["action"] != "retry"):
            raise ValueError("expected finish or retry with delay_seconds")
        return cls(value["delay_seconds"])


def failure_fields(failure):
    """Normalize existing canonical failure data for policy and final observers.

    Request executors supply type/error, retaining the real exception for the
    terminal and ACP's existing error reporting. The tool loop supplies decoded
    status/protocol_data/stop_reason after preserving a real response in history.
    Neither producer's data becomes executable identity or request authority.

    Both turn_error policy and turn_end notification consume error_type and the
    nullable error_message. incomplete_reason preserves a native token-limit
    reason, never guessed from assistant text. retry_after_seconds is an existing
    provider minimum, not a newly parsed header or a policy-generated backoff;
    TurnRecovery enforces it, and policy may inspect it before choosing to stop.
    """
    cause = failure.get("error")
    detail = cause
    protocol_data = failure.get("protocol_data")
    native = (protocol_data.get(formats.OPENAI_RESPONSES)
              if isinstance(protocol_data, dict) else None)
    incomplete_reason = failure.get("stop_reason")
    if isinstance(native, dict):
        incomplete = native.get("incomplete_details")
        if isinstance(incomplete, dict):
            incomplete_reason = incomplete.get("reason")
        if detail is None:
            details = native.get("error") or incomplete
            if isinstance(details, dict):
                detail = details.get("message") or details.get("reason")
    if failure["type"] == "response_incomplete" and detail is None:
        detail = incomplete_reason
    fields = {
        "error_type": failure["type"],
        "error_message": str(detail) if detail is not None else None,
    }
    if failure["type"] == "response_incomplete":
        fields["incomplete_reason"] = (
            incomplete_reason if isinstance(incomplete_reason, str) else None)
    hint = getattr(cause, "retry_after", None)
    if isinstance(hint, int | float) and not isinstance(hint, bool):
        try:
            if hint >= 0 and math.isfinite(hint):
                fields["retry_after_seconds"] = hint
        except OverflowError:
            pass
    return fields


class TurnRecovery:
    """Account for one user turn without introducing another lifetime owner.

    Terminal/ACP keep this object local to their admitted turn. model_requests
    borrows it for prepared-request replay, and run_tool_loop_async borrows the
    SAME object for history continuation. Their existing awaited call chain is
    sequential; frontend task ownership already controls admission and lifetime.
    A per-request counter would reset between these two execution points.

    _policy is the hook pipeline selected once at admission, not a global lookup
    on each retry. _cancel_event borrows the frontend's existing cancellation
    state, also queried by requests/tools; it is never cleared or serialized here.
    _retry_count counts completed retry waits across BOTH borrowers, including
    failures separated by successful intermediate work, not the 401 refresh.
    _finished latches a decline so its original exception can propagate from a
    request to the tool loop without invoking policy again. No activation or
    concurrency state is needed, and this object owns no child/task resources.
    Each admitted turn gets a fresh object; none of this state is persisted.
    """

    def __init__(self, policy, cancel_event: asyncio.Event):
        if not isinstance(cancel_event, asyncio.Event):
            raise TypeError("turn recovery requires its owning asyncio.Event")
        self._policy = policy
        self._cancel_event = cancel_event
        self._retry_count = 0
        self._finished = False

    @property
    def cancel_event(self):
        return self._cancel_event

    def _check_cancelled(self):
        if self._cancel_event.is_set():
            self._finished = True
            raise Cancelled()

    async def retry(self, failure):
        self._check_cancelled()
        if self._finished:
            return False
        if failure["type"] not in MODEL_FAILURE_TYPES:
            raise ValueError("recovery requires a model failure")
        try:
            fields = failure_fields(failure)
            decision = await self._policy({
                "event": "turn_error", "retry_count": self._retry_count,
                **fields,
            }, cancel_event=self._cancel_event)
            self._check_cancelled()
            if not isinstance(decision, RetryDecision):
                raise TypeError("recovery policy must return RetryDecision")
            if decision.delay_seconds is None:
                self._finished = True
                return False
            # A policy's generated-delay cap must not shorten a provider's
            # existing minimum. One owner applies this rule to both borrowers.
            delay = max(decision.delay_seconds, fields.get("retry_after_seconds", 0))
            print(f"Retrying in {delay:g} seconds.", file=sys.stderr)
            try:
                await asyncio.wait_for(self._cancel_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            self._check_cancelled()
            self._retry_count += 1
            return True
        except BaseException:
            self._finished = True
            raise
