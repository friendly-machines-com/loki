"""Transcript replay classification for ACP session/load.

The ACP front-end re-emits each historical transcript event as
``session/update`` notifications (user_message_chunk, agent_message_chunk,
tool-call updates) before answering ``session/load``.  This module is a
pure classifier: it maps each event to zero or more tuples:

* ("user", text)            -- user message text
* ("agent", text, key)      -- assistant text; key identifies the message
* ("tool", title, call_id)  -- a tool call, with its result if known

Events with no user-visible meaning (system instructions, provider-internal
reasoning) classify to nothing.  Unknown event types become agent text,
never dropped.
"""

from __future__ import annotations

from . import formats


def _message_text(item: dict) -> str:
    return formats.item_text(item).strip()


def classify_message(item: dict):
    """Classify a transcript `message` event.

    Returns a list of tuples; empty for system/developer messages.
    """
    role = item.get("role")
    if role in ("system", "developer"):
        return []
    text = _message_text(item)
    if not text and not any(
            isinstance(c, dict) and c.get("type") in
            ("image", "file", "document", "audio")
            for c in item.get("content", [])):
        return []
    blocks = []
    if text:
        blocks.append(("user" if role == "user" else "agent", text,
                       ("message", role)))
    for content in item.get("content", []):
        if not isinstance(content, dict):
            continue
        ctype = content.get("type")
        if ctype in ("image", "file", "document", "audio"):
            label = {
                "image": "[Image content]",
                "file": "[File content]",
                "document": "[File content]",
                "audio": "[Audio content]",
            }[ctype]
            blocks.append(("agent" if role != "user" else "user", label,
                           ("message", role)))
    return blocks


def classify_response_item(item: dict, response: dict):
    """Classify one item inside a `model_response` event."""
    item_type = item.get("type")
    if item_type == "message":
        return classify_message(item)
    if item_type == "function_call":
        name = formats.tool_call_name(item) or "<unknown>"
        return [("tool", name, item.get("call_id"))]
    # Reasoning items and provider-internal outputs carry no
    # user-visible conversation content.
    return []


def classify_response(event: dict):
    """Classify a `model_response` event into replay tuples."""
    blocks = []
    # Provider notices are persisted for terminal replay but ACP has no
    # metadata update for them. Do not fabricate assistant speech.
    status = event.get("status", "completed")
    if status != "completed":
        blocks.append((
            "agent", f"[Model response {status}]",
            ("response", event.get("model") or "")))
    for item in event.get("items", []):
        blocks.extend(classify_response_item(item, event))
    return blocks


def classify_tool_result(item: dict):
    """Classify a top-level `tool_result` event."""
    name = item.get("name") or item.get("call_id") or "<unknown>"
    label = "Tool error" if item.get("is_error") else "Tool result"
    text = _message_text(item)
    return [("tool", f"{label}: {name}" + (f"\n{text}" if text else ""),
             item.get("call_id"))]


def classify_event(event: dict):
    """Classify one canonical transcript event into replay tuples."""
    event_type = event.get("type")
    if event_type == "message":
        return classify_message(event)
    if event_type == "model_response":
        return classify_response(event)
    if event_type == "tool_result":
        return classify_tool_result(event)
    # Unknown event types become visible text rather than being dropped.
    from pprint import pformat
    return [("agent",
             f"[Session event: {event_type or 'unknown'}]\n"
             f"{pformat(event, width=100)}",
             ("session_event", str(event_type)))]


def classify_transcript(events: list):
    """Classify a full transcript into a flat list of replay tuples."""
    blocks = []
    for event in events:
        blocks.extend(classify_event(event))
    return blocks
