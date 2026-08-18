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
    return [{"type": "text", "text": text}]


def agent_message_chunk(session_id: str, text: str) -> dict:
    return {
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        },
    }


def tool_call(session_id: str, tool_call_id: str, title: str, kind: str) -> dict:
    return {
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "tool_call",
            "toolCallId": tool_call_id,
            "title": title,
            "kind": kind,
            "status": "in_progress",
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

    ``state`` carries turn-local bookkeeping between calls: the current
    tool call id (tool_call events carry no id, so ids are assigned in
    order and results attach to the most recent one).
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
        state["tool_counter"] = state.get("tool_counter", 0) + 1
        call_id = f"call-{state['tool_counter']}"
        state["pending_call_id"] = call_id
        args = event.get("args") or {}
        title = name
        command = args.get("command") if isinstance(args, dict) else None
        if command:
            title = f"{name}: {command}"
        return [tool_call(
            session_id, call_id, title,
            TOOL_KINDS.get(name, "other"))]
    if kind == "tool_result":
        call_id = state.get("pending_call_id") or "call-1"
        text = event.get("content")
        if not isinstance(text, str):
            text = str(text)
        status = "failed" if event.get("is_error") else "completed"
        return [tool_call_update(
            session_id, call_id, _content(text), status=status)]
    if kind == "tool_error":
        call_id = state.get("pending_call_id") or "call-1"
        text = event.get("result") or "tool error"
        return [tool_call_update(
            session_id, call_id, _content(str(text)), status="failed")]
    if kind == "tool_rejected":
        call_id = state.get("pending_call_id") or "call-1"
        return [tool_call_update(
            session_id, call_id,
            _content("not executed"), status="failed")]
    if kind == "response_cancelled":
        return [agent_message_chunk(
            session_id, "[turn cancelled by user]")]
    # assistant_end, response_timing, max_loops, stream_error,
    # transcript_error, response_incomplete/failed: no per-event update;
    # the stopReason or an error response carries them.
    return []
