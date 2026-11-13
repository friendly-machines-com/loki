"""Native checkpoint requests and explicit retained-window policy, not Session I/O.

Terminal and ACP v1 explicit requests borrow their existing idle/turn execution
owner. An ordinary tool-loop response would execute calls and create task history;
a V2 request must only produce a validated candidate. Borrow an explicit RequestContext and
its existing credential leases, exact-URL authorization, cancellation and bounded
prepared-request transport retries. There is no second client/retry controller,
ambient credential lookup, summary fallback, endpoint or profile switch here.
"""

import asyncio
import copy
import uuid

from . import authentications, formats, model_requests, protocols, recoveries
from .tool_runtime import HookExecutionError


class CompactionError(ValueError):
    """A refused operation or unusable candidate, not permission for a fallback."""


class Requests:
    """One frontend-owned explicit request, shared through pending and execution.

    TUI receipt and ACP control replies need to coalesce before FIFO delivery;
    ordinary prompt admission cannot do that while a turn is running. The existing
    turn/idle task executes run(), never this intake object or a second Task.
    One Future carries a terminal DATA outcome, so abandoned observers cannot leave
    unobserved exception Futures. ACP's on_result flushes its existing request IDs
    synchronously; neither callback nor duplicate creates another model turn.
    Owners finish unserviced intent before their turn-finalization awaits. Later
    receipt then belongs to their next-idle tail, not a loop which has returned.
    """

    def __init__(self, on_result=None):
        self.pending = None
        self.on_result = on_result

    def request(self):
        if self.pending is None:
            self.pending = asyncio.get_running_loop().create_future()
        return self.pending

    def _finish(self, future, outcome):
        if self.pending is future:
            self.pending = None
        if not future.done():
            future.set_result(outcome)
        if self.on_result is not None:
            self.on_result(outcome)
        return outcome

    def finish_pending(self, *, cancelled=False):
        if self.pending is None:
            return
        if cancelled:
            outcome = {"type": "compaction_cancelled", "text": "Context compaction cancelled."}
        else:
            outcome = {"type": "compaction_failed", "error": CompactionError(
                "The turn ended before context compaction could run.")}
        self._finish(self.pending, outcome)

    async def run(self, operation, on_event=None):
        if self.pending is None:
            return None
        future = self.pending
        emit = on_event if on_event is not None else lambda event: None
        compaction_id = None
        outcome = None

        def started():
            nonlocal compaction_id
            compaction_id = str(uuid.uuid4())
            emit({"type": "compaction_started", "compaction_id": compaction_id})

        try:
            text = await operation(started)
            outcome = {"type": "compaction_completed", "text": text}
        except recoveries.Cancelled:
            outcome = {"type": "compaction_cancelled", "text": "Context compaction cancelled."}
        except (CompactionError, formats.TranscriptFormatError, protocols.ProtocolError,
                model_requests.ApiError, authentications.CredentialError, OSError, HookExecutionError) as error:
            outcome = {"type": "compaction_failed", "error": error}
        except BaseException as error:
            # Task cancellation and programming failures retain their semantics,
            # but every accepted control still acquires one terminal outcome.
            outcome = ({"type": "compaction_cancelled", "text": "Context compaction cancelled."}
                       if isinstance(error, asyncio.CancelledError)
                       else {"type": "compaction_failed", "error": error})
            raise
        finally:
            if outcome is not None:
                if compaction_id is not None:
                    outcome["compaction_id"] = compaction_id
                try:
                    emit(outcome)
                finally:
                    self._finish(future, outcome)
        return outcome


def retained_messages(active_events):
    """The V2 retention rule also owns its empty-operation predicate.

    Startup already has instructions, so a literal empty-list check is wrong.
    Keeping this selection shared prevents a no-op from silently evicting data a
    changed window policy would retain. Checkpoints are responses, not empty input.
    """
    return [event for event in active_events if event.get("type") == "message"
            and event.get("role") in ["system", "developer", "user"]]


def format_error(error):
    """Command fronts need HTTP details, not ordinary display-streaming advice.

    StreamingApiError's normal formatter suggests LOKI_STREAM=0 for incompatible
    servers. That cannot change this dedicated always-streamed checkpoint request;
    retain its HTTP status/body without inviting an ineffective retry or profile
    switch. Keep the original typed exception intact for its owning caller.
    """
    if isinstance(error, model_requests.ApiError):
        return model_requests.ApiError.formatted(error)
    return str(error)


def require_v2(context):
    """Admit the native subscription contract, not a model-name capability guess.

    The authenticated catalog owns the request profile (including Responses-Lite).
    The tested contract uses the already-admitted Responses URL, never a sibling
    compact URL. This admits an operation, not universal model/deployment support:
    a provider/context-limit rejection is reported without trimming or fallback.
    """
    config = context.config
    if config is None:
        raise CompactionError("Select a model before compacting context.")
    provider = config.chat_provider
    if (provider.kind != formats.OPENAI_RESPONSES
            or provider.provider_id != "openai-subscription"
            or provider.chat_url != authentications.OPENAI_CHATGPT_RESPONSES_URL
            or provider.openai_request_profile is None
            or config.auth_spec is None or config.auth_spec.scheme != "openai-subscription"
            or config.auth_spec.credential != authentications.CredentialRef.openai_subscription()):
        raise CompactionError("Context compaction is not supported by the selected connection.")
    if not config.model:
        raise CompactionError("Select a model before compacting context.")


async def request_v2(context, active_events, tools, thinking, cancel_check, *, codex_turn_state):
    """Produce one checkpoint, never a continuation or live transcript mutation.

    Build the complete active history, saved instructions, ordinary offered tools
    and captured thinking before appending the FINAL trigger after profile framing.
    Always stream: the subscription contract requires it even when display streaming
    is off. Routing is borrowed from the actual owner: a running turn retains its
    existing state and an idle operation supplies a fresh one. Conversation/cache
    identity remain unchanged. Completion, item count and encrypted-state
    checks are independent of SSE transport success. Ignore all other returned
    items/instructions; only the whole native checkpoint is decoded and copied into
    its origin-bound response. No returned tool can execute through this path.
    """
    require_v2(context)
    formats.validate_replacement(active_events)
    if cancel_check():
        raise recoveries.Cancelled()
    config = context.config
    provider = config.chat_provider
    payload = provider.streaming_chat_payload(
        active_events, tools, config.model, prompt_cache_key=context.conversation_id,
        reasoning_effort=thinking.effort, thinking_mode=thinking.mode,
        thinking_budget=thinking.budget, reasoning_retention=thinking.retention,
        reasoning_traces=thinking.traces)
    payload["input"].append({"type": "compaction_trigger"})
    response = await model_requests.async_chat_stream_request(
        provider.chat_url, payload, request_headers=provider.headers, context=context,
        codex_turn_state=codex_turn_state, cancel_check=cancel_check)
    if cancel_check():
        raise recoveries.Cancelled()
    data = response.payload
    if (not isinstance(data, dict) or data.get("object") != "response"
            or data.get("status") != "completed" or not isinstance(data.get("output"), list)):
        raise CompactionError("Context compaction did not complete.")
    checkpoints = [item for item in data["output"] if isinstance(item, dict) and item.get("type") == "compaction"]
    if len(checkpoints) != 1:
        raise CompactionError("Context compaction did not return exactly one checkpoint.")
    encrypted = checkpoints[0].get("encrypted_content")
    if not isinstance(encrypted, str) or not encrypted:
        raise CompactionError("Context compaction returned an unusable checkpoint.")
    # Do not decode/execute unrelated output or promote returned instructions.
    selected = {"object": "response", "status": "completed", "output": checkpoints,
                "usage": data.get("usage")}
    turn = model_requests.decode_chat_response(selected, context=context, effective_model=response.effective_model)
    return turn.to_event()


def build_v2_window(active_events, checkpoint_response):
    """Retain local instruction/user messages, then the NEW checkpoint response.

    Provider output cannot universally choose Loki's retained window. This first
    V2 policy is explicit and deliberately lossy: older assistant/tool history and
    previous checkpoints stop being active, although archived originals remain.
    request_v2 received their full current active tool results BEFORE this omission;
    lack of assistant commentary never establishes that a result is unimportant or
    reproducible. An archive is not continued model knowledge/retrieval. Keeping
    all user messages may still leave a large prompt; no classifier/budget manager
    is invented here. Copy candidates so awaits/commit cannot mutate old snapshots;
    do not reread SYSTEM.md or turn provider messages into local instructions.
    """
    formats.validate_replacement(active_events)
    if (checkpoint_response.get("type") != "model_response"
            or checkpoint_response.get("status") != "completed"
            or len(checkpoint_response.get("items", [])) != 1
            or checkpoint_response["items"][0].get("type") != "compaction"):
        raise CompactionError("Context compaction returned an invalid response.")
    events = retained_messages(active_events)
    window = copy.deepcopy(events + [checkpoint_response])
    formats.validate_replacement(window)
    return window
