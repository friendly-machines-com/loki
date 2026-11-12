"""Native checkpoint requests and explicit retained-window policy, not Session I/O.

The manual terminal/ACP operation is the current caller. An ordinary tool-loop
turn would execute returned calls and create assistant/task history; a V2 request
must only produce a validated candidate. Borrow an explicit RequestContext and
its existing credential leases, exact-URL authorization, cancellation and bounded
prepared-request transport retries. There is no second client/retry controller,
ambient credential lookup, summary fallback, endpoint or profile switch here.
"""

import copy

from . import authentications, formats, model_requests, recoveries


class CompactionError(ValueError):
    """A refused operation or unusable candidate; old Session context stays active."""


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


async def request_v2(context, active_events, tools, thinking, cancel_check):
    """Produce one checkpoint, never a continuation or live transcript mutation.

    Build the complete active history, saved instructions, ordinary offered tools
    and captured thinking before appending the FINAL trigger after profile framing.
    Always stream: the subscription contract requires it even when display streaming
    is off. A fresh routing owner belongs to this manual operation; conversation
    and cache identity remain unchanged. Completion, item count and encrypted-state
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
        codex_turn_state=model_requests.CodexTurnState(), cancel_check=cancel_check)
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
    events = [event for event in active_events if event.get("type") == "message"
              and event.get("role") in ["system", "developer", "user"]]
    window = copy.deepcopy(events + [checkpoint_response])
    formats.validate_replacement(window)
    return window
