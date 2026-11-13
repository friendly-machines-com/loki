"""Map Loki turn events to ACP session/update notifications.

Pure: an event in, an update dict out (or None when the event has no
client-facing meaning).  The worker applies the sessionId and sends.
"""

from __future__ import annotations

import copy
import uuid

from . import process_outputs

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


def plan(session_id: str, todos: list) -> dict:
    # Saved logs validate the list, not its entries. Project only valid ACP
    # fields so malformed restored data cannot break the client's plan view.
    entries = []
    for todo in todos:
        if not isinstance(todo, dict):
            continue
        content = todo.get("content")
        status = todo.get("status")
        priority = todo.get("priority")
        if (not isinstance(content, str) or not content.strip()
                or status not in ["pending", "in_progress", "completed"]
                or priority not in ["high", "medium", "low"]):
            continue
        entries.append({"content": content, "status": status, "priority": priority})
    return {
        "sessionId": session_id,
        "update": {"sessionUpdate": "plan", "entries": entries},
    }


def context_usage(session_id, snapshot):
    """ACP carries counts; presentation and integer rounding belong to clients.

    Unknown is not zero. ACP usage_update has no unknown/reset representation,
    so clients retain their last report until another usable observation.
    """
    if snapshot.usage is None or snapshot.capacity is None:
        return None
    return {
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "usage_update",
            "used": snapshot.usage.used,
            "size": snapshot.capacity.tokens,
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


def map_event(session_id: str, event: dict, state: dict, *, compaction=False, protocol_version=1) -> list:
    """Translate one Loki on_event into zero or more update params.

    ``state`` carries turn-local bookkeeping. Provider call ids are preserved
    end to end; inventing positional ids makes concurrent/rejected calls
    attach updates to the wrong tool.
    """
    kind = event.get("type")
    if kind in ["compaction_started", "compaction_completed", "compaction_failed", "compaction_cancelled"]:
        # Only the settled execution owner emits these events, never busy
        # receipt. Untyped v1 text therefore cannot splice into live deltas.
        # Lifecycle is live presentation: saved replacements retain their
        # existing ordinary replay notice, not an invented ID/final outcome.
        if compaction and event.get("compaction_id") is not None:
            update = {"sessionUpdate": "compaction_update",
                      "compactionId": event["compaction_id"],
                      "status": {"compaction_started": "in_progress", "compaction_completed": "completed",
                                 "compaction_failed": "failed", "compaction_cancelled": "cancelled"}[kind]}
            if kind == "compaction_failed":
                update["error"] = event["text"]
            return [{"sessionId": session_id, "update": update}]
        return [agent_message_chunk(session_id, event.get("text", "Compacting context."))]
    if kind == "reasoning_start":
        if protocol_version == 2:
            state["thought_id"] = str(uuid.uuid4())
            return []
        if state.get("last_content_kind") == "thought":
            return [{"sessionId": session_id, "update": {
                "sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "\n\n"}}}]
        return []
    if kind == "reasoning_delta":
        state["last_content_kind"] = "thought"
        update = {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": event.get("text", "")}}
        if protocol_version == 2:
            update["messageId"] = state.get("thought_id") or state.setdefault("thought_id", str(uuid.uuid4()))
        return [{"sessionId": session_id, "update": update}]
    if kind in ["assistant_start", "assistant_delta", "assistant_message", "tool_call", "tool_result", "tool_rejected"]:
        state["last_content_kind"] = "other"
    if kind == "assistant_start":
        state["message_id"] = state.get("message_counter", 0) + 1
        state["message_counter"] = state["message_id"]
        if protocol_version == 2:
            state["agent_message_id"] = str(uuid.uuid4())
        return []
    if kind == "assistant_end":
        state.pop("agent_message_id", None)
        return []
    if kind == "assistant_delta":
        packet = agent_message_chunk(session_id, event.get("content", ""))
        if protocol_version == 2:
            packet["update"]["messageId"] = state.get("agent_message_id") or state.setdefault("agent_message_id", str(uuid.uuid4()))
        return [packet]
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
        if event.get("cwd") is not None:
            title += f" (cwd: {event['cwd']})"
        return [tool_call(
            session_id, call_id, title,
            TOOL_KINDS.get(name, "other"))]
    if kind == "todos_updated":
        return [plan(session_id, event["todos"])]
    if kind == "tool_result":
        call_id = _event_call_id(event, state)
        # Model/history keep the full image. ACP needs only a readable notice;
        # inline base64 here would also exceed its frame cap for large images.
        text = process_outputs.presentation_text(event.get("content"))
        status = "failed" if event.get("is_error") else "completed"
        packet = tool_call_update(session_id, call_id, _content(text), status=status)
        if protocol_version == 2:
            # V1 defaults an omitted result status to completed; v2 omission
            # preserves in_progress from the first tool update instead.
            packet["update"]["status"] = status
        return [packet]
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
        title = f"{name}: not executed"
        if event.get("cwd") is not None:
            title += f" (cwd: {event['cwd']})"
        return [tool_call(
            session_id,
            call_id,
            title,
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


def config_options_v2(options):
    """Project owned configuration facts; never change selection/approval policy."""
    result = copy.deepcopy(options)
    for option in result:
        if "id" in option:
            option["configId"] = option.pop("id")
        for entry in option.get("options", []):
            if "group" in entry:
                entry["groupId"] = entry.pop("group")
    return result


def commands_v2(commands):
    result = copy.deepcopy(commands)
    for command in result:
        command.setdefault("type", "command")
        if isinstance(command.get("input"), dict):
            command["input"].setdefault("type", "text")
    return result


def update_v2(update):
    """One dialect projection for live, local-command and replay producers.

    Runtime execution stays shared with v1. The worker owns stream IDs at actual
    event boundaries; a standalone readable status gets its own ID here. These
    are presentation identities, never tool execution or credential authority.
    """
    result = copy.deepcopy(update)
    kind = result.get("sessionUpdate")
    if kind == "tool_call":
        result["sessionUpdate"] = "tool_call_update"
    elif kind == "plan":
        # Existing todos are one conversation-owned plan, not a plan registry.
        result = {"sessionUpdate": "plan_update", "plan": {
            "type": "items", "planId": "loki-plan", "entries": result.get("entries", [])}}
    elif kind == "config_option_update":
        result["configOptions"] = config_options_v2(result.get("configOptions", []))
    elif kind == "available_commands_update":
        result["availableCommands"] = commands_v2(result.get("availableCommands", []))
    elif kind in ["user_message_chunk", "agent_message_chunk", "agent_thought_chunk"] and "messageId" not in result:
        result["messageId"] = str(uuid.uuid4())
    return result


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
