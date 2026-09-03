"""Map Loki turn events to ACP session/update notifications.

Pure: an event in, an update dict out (or None when the event has no
client-facing meaning).  The worker applies the sessionId and sends.
"""

from __future__ import annotations

TOOL_KINDS = {
    "Read": "read",
    "Glob": "search",
    "Grep": "search",
    "Edit": "edit",
    "Write": "edit",
    "Bash": "execute",
    "Jobs": "execute",
    "JobStatus": "execute",
    "JobStop": "execute",
    "WebFetch": "fetch",
    "WebSearch": "fetch",
    "Skill": "other",
    "Agent": "other",
    "TodoRead": "other",
    "TodoWrite": "other",
}


def _content(text: str) -> list:
    return [{
        "type": "content",
        "content": {"type": "text", "text": text},
    }]


def agent_message_chunk(session_id: str, text: str) -> dict:
    return {
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        },
    }


def tool_call(session_id: str, tool_call_id: str, title: str, kind: str,
              status: str = "in_progress") -> dict:
    return {
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "tool_call",
            "toolCallId": tool_call_id,
            "title": title,
            "kind": kind,
            "status": status,
        },
    }


def tool_call_update(session_id: str, tool_call_id: str, content,
                     status: str = "completed") -> dict:
    update = {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tool_call_id,
    }
    if status != "completed":
        update["status"] = status
    if content is not None:
        update["content"] = content
    return {"sessionId": session_id, "update": update}


def map_event(session_id: str, event: dict, state: dict) -> list:
    """Translate one Loki on_event into zero or more update params.

    ``state`` carries turn-local bookkeeping. Provider call ids are preserved
    end to end; inventing positional ids makes concurrent/rejected calls
    attach updates to the wrong tool.
    """
    kind = event.get("type")
    if kind == "assistant_start":
        state["message_id"] = state.get("message_counter", 0) + 1
        state["message_counter"] = state["message_id"]
        return []
    if kind == "assistant_delta":
        return [agent_message_chunk(session_id, event.get("content", ""))]
    if kind == "assistant_message":
        return [agent_message_chunk(session_id, event.get("content", ""))]
    if kind == "tool_call":
        name = event.get("name") or "tool"
        call_id = _event_call_id(event, state, begin=True)
        state.setdefault("announced_calls", set()).add(call_id)
        args = event.get("args") or {}
        title = name
        command = args.get("command") if isinstance(args, dict) else None
        if command:
            title = f"{name}: {command}"
        return [tool_call(
            session_id, call_id, title,
            TOOL_KINDS.get(name, "other"))]
    if kind == "tool_result":
        call_id = _event_call_id(event, state)
        text = event.get("content")
        if not isinstance(text, str):
            text = str(text)
        status = "failed" if event.get("is_error") else "completed"
        return [tool_call_update(
            session_id, call_id, _content(text), status=status)]
    if kind == "tool_error":
        # A tool_result event follows with the same real call id and complete
        # content. Sending both produces duplicate terminal updates.
        return []
    if kind == "tool_rejected":
        call_id = _event_call_id(event, state, begin=True)
        announced = state.setdefault("announced_calls", set())
        if call_id in announced:
            return []
        announced.add(call_id)
        name = event.get("name") or "tool"
        return [tool_call(
            session_id,
            call_id,
            f"{name}: not executed",
            TOOL_KINDS.get(name, "other"),
            status="failed",
        )]
    if kind == "response_cancelled":
        return [agent_message_chunk(
            session_id, "[turn cancelled by user]")]
    # provider_notice has no ACP session/update representation; presenting it
    # as agent_message_chunk would falsely make provider metadata assistant
    # speech. assistant_end, response_timing, max_loops, stream_error,
    # transcript_error, response_incomplete/failed: no per-event update;
    # the stopReason or an error response carries them.
    return []


def _fallback_call_id(state: dict) -> str:
    state["tool_counter"] = state.get("tool_counter", 0) + 1
    return f"loki-call-{state['tool_counter']}"


def _event_call_id(event: dict, state: dict, *, begin=False) -> str:
    supplied = event.get("call_id")
    if supplied is not None:
        call_id = str(supplied)
    elif not begin and state.get("last_call_id"):
        call_id = state["last_call_id"]
    else:
        call_id = _fallback_call_id(state)
    state["last_call_id"] = call_id
    return call_id
