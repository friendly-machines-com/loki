import copy
import json


TRANSCRIPT_SCHEMA = "day-agent.transcript.v2"
LEGACY_TRANSCRIPT_SCHEMAS = {"day-agent.transcript.v1"}

# The persisted format is a transcript item stream, not a provider message
# list. OpenAI Chat and Anthropic Messages are rendered as projections from
# this stream; OpenAI Responses already exposes message, reasoning, function
# call, and tool-result records as top-level items, so v2 keeps those concepts
# top-level on disk.


class TranscriptFormatError(ValueError):
    pass


def _copy(value):
    return copy.deepcopy(value)


def _put_optional(item, key, value):
    if value is not None:
        item[key] = _copy(value)


def text_block(text, annotations=None, logprobs=None, provider_raw=None):
    block = {"type": "text", "text": str(text)}
    _put_optional(block, "annotations", annotations)
    _put_optional(block, "logprobs", logprobs)
    _put_optional(block, "provider_raw", provider_raw)
    return block


def media_block(kind, value, provider=None, provider_raw=None):
    block = {"type": kind, "value": _copy(value)}
    _put_optional(block, "provider", provider)
    _put_optional(block, "provider_raw", provider_raw)
    return block


def provider_content_block(provider, value):
    return {
        "type": "provider_content",
        "provider": provider,
        "value": _copy(value),
    }


def content_blocks(content):
    if content is None:
        return []
    if isinstance(content, str):
        if content == "":
            return []
        return [text_block(content)]
    if isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, dict):
                blocks.append(_copy(block))
            else:
                blocks.append(text_block(block))
        return blocks
    return [text_block(content)]


def instruction_item(content, authority="system", provider_ids=None, provider_raw=None):
    item = {
        "type": "instruction",
        "authority": authority or "system",
        "content": content_blocks(content),
    }
    _put_optional(item, "provider_ids", provider_ids)
    _put_optional(item, "provider_raw", provider_raw)
    return item


def message_item(role, content=None, status=None, provider_ids=None, provider_raw=None):
    item = {
        "type": "message",
        "role": role,
        "content": content_blocks(content),
    }
    _put_optional(item, "status", status)
    _put_optional(item, "provider_ids", provider_ids)
    _put_optional(item, "provider_raw", provider_raw)
    return item


def tool_call_item(call_id, name, input_value, raw_arguments=None, provider_ids=None,
                   parse_error=None, tool_kind="function", status=None,
                   provider_raw=None):
    item = {
        "type": "tool_call",
        "id": call_id,
        "call_id": call_id,
        "tool_kind": tool_kind or "function",
        "name": name,
        "input": _copy(input_value),
    }
    _put_optional(item, "raw_arguments", raw_arguments)
    _put_optional(item, "provider_ids", provider_ids)
    _put_optional(item, "parse_error", parse_error)
    _put_optional(item, "status", status)
    _put_optional(item, "provider_raw", provider_raw)
    return item


def tool_result_item(tool_call_id, content, name=None, is_error=False,
                     tool_kind="function", provider_ids=None, provider_raw=None):
    item = {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "tool_kind": tool_kind or "function",
        "content": content_blocks(content),
        "is_error": bool(is_error),
    }
    _put_optional(item, "name", name)
    _put_optional(item, "provider_ids", provider_ids)
    _put_optional(item, "provider_raw", provider_raw)
    return item


def reasoning_item(content=None, summary=None, encrypted_content=None,
                   status=None, provider_ids=None, provider_raw=None):
    item = {"type": "reasoning"}
    _put_optional(item, "content", content)
    _put_optional(item, "summary", summary)
    _put_optional(item, "encrypted_content", encrypted_content)
    _put_optional(item, "status", status)
    _put_optional(item, "provider_ids", provider_ids)
    _put_optional(item, "provider_raw", provider_raw)
    return item


def response_metadata_item(provider, protocol, response=None):
    item = {
        "type": "response_metadata",
        "provider": provider,
        "protocol": protocol,
    }
    if isinstance(response, dict):
        for key in [
            "id",
            "model",
            "status",
            "created_at",
            "completed_at",
            "previous_response_id",
            "conversation",
            "store",
            "parallel_tool_calls",
            "usage",
            "error",
            "incomplete_details",
        ]:
            _put_optional(item, key, response.get(key))
        item["provider_raw"] = _copy(response)
    return item


def provider_item(provider, value):
    return {
        "type": "provider_item",
        "provider": provider,
        "value": _copy(value),
    }


def blocks_text(blocks):
    parts = []
    for block in blocks or []:
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)


def item_text(item):
    return blocks_text(item.get("content", []))


def _tool_call_from_legacy_block(block):
    return tool_call_item(
        block.get("call_id") or block.get("id"),
        block.get("name"),
        block.get("input", {}),
        raw_arguments=block.get("raw_arguments"),
        provider_ids=block.get("provider_ids"),
        parse_error=block.get("parse_error"),
        tool_kind=block.get("tool_kind", "function"),
        status=block.get("status"),
        provider_raw=block.get("provider_raw"),
    )


def normalize_items_to_v2(items):
    # v1 stored assistant tool calls inside message content because it followed
    # OpenAI Chat's envelope. Split those into top-level tool_call items so old
    # logs migrate into the v2 stream shape when loaded or saved.
    normalized = []
    for item in items or []:
        if not isinstance(item, dict):
            normalized.append(provider_item("unknown", item))
            continue

        item_type = item.get("type")
        if item_type == "message":
            message = _copy(item)
            content = []
            tool_calls = []
            for block in message.get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "tool_call":
                    tool_calls.append(_tool_call_from_legacy_block(block))
                else:
                    content.append(_copy(block))
            message["content"] = content
            normalized.append(message)
            normalized.extend(tool_calls)
            continue

        if item_type == "tool_call":
            call = _copy(item)
            call_id = call.get("call_id") or call.get("id")
            call["id"] = call_id
            call["call_id"] = call_id
            call.setdefault("tool_kind", "function")
            normalized.append(call)
            continue

        normalized.append(_copy(item))
    return normalized


def item_tool_calls(item):
    if item.get("type") == "tool_call":
        return [item]
    if item.get("type") != "message" or item.get("role") != "assistant":
        return []
    return [block for block in item.get("content", []) if block.get("type") == "tool_call"]


def item_has_tool_calls(item):
    return bool(item_tool_calls(item))


def is_app_tool_call(item):
    return item.get("type") == "tool_call" and item.get("tool_kind", "function") == "function"


def response_tool_calls(items):
    calls = []
    for item in items or []:
        for call in item_tool_calls(item):
            if is_app_tool_call(call):
                calls.append(call)
    return calls


def user_prompt_history(items):
    history = []
    for item in items:
        if item.get("type") == "message" and item.get("role") == "user":
            text = item_text(item)
            if text:
                history.append(text)
    return history


def new_log_blob(items, session_todos):
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "items": normalize_items_to_v2(items),
        "session_todos": session_todos,
    }


def load_log_blob(blob):
    if isinstance(blob, dict) and blob.get("schema") == TRANSCRIPT_SCHEMA:
        return normalize_items_to_v2(blob.get("items", [])), _copy(blob.get("session_todos", []))
    if isinstance(blob, dict) and blob.get("schema") in LEGACY_TRANSCRIPT_SCHEMAS:
        return normalize_items_to_v2(blob.get("items", [])), _copy(blob.get("session_todos", []))
    if isinstance(blob, dict) and "items" in blob:
        return normalize_items_to_v2(blob.get("items", [])), _copy(blob.get("session_todos", []))
    if isinstance(blob, dict) and "messages" in blob:
        return openai_chat_messages_to_items(blob["messages"]), _copy(blob.get("session_todos", []))
    if isinstance(blob, list):
        return openai_chat_messages_to_items(blob), []
    raise TranscriptFormatError("unrecognized chat log format")


def openai_content_to_blocks(content):
    if content is None:
        return []
    if isinstance(content, str):
        return content_blocks(content)
    if isinstance(content, list):
        blocks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                blocks.append(text_block(part.get("text", "")))
            else:
                blocks.append(provider_content_block("openai_chat", part))
        return blocks
    return [text_block(content)]


def _parse_json_arguments(raw_arguments):
    if not isinstance(raw_arguments, str):
        return raw_arguments, None
    try:
        return json.loads(raw_arguments), None
    except json.JSONDecodeError as e:
        return {}, str(e)


def _openai_tool_call_to_item(tool_call):
    call_id = tool_call.get("id")
    function = tool_call.get("function") or {}
    raw_arguments = function.get("arguments", "{}")
    input_value, parse_error = _parse_json_arguments(raw_arguments)
    return tool_call_item(
        call_id,
        function.get("name"),
        input_value,
        raw_arguments=raw_arguments,
        provider_ids={"openai_chat": call_id},
        parse_error=parse_error,
        provider_raw=tool_call,
    )


def openai_chat_message_to_items(message):
    role = message.get("role")
    if role in ["system", "developer"]:
        return [instruction_item(message.get("content", ""), authority=role, provider_raw=message)]
    if role in ["user", "assistant"]:
        items = [message_item(role, openai_content_to_blocks(message.get("content")),
                              provider_raw=message)]
        if role == "assistant":
            items.extend(_openai_tool_call_to_item(tc)
                         for tc in message.get("tool_calls", []) or [])
        return items
    if role == "tool":
        return [tool_result_item(
            message.get("tool_call_id"),
            message.get("content", ""),
            name=message.get("name"),
            is_error=message.get("is_error", False),
            provider_raw=message,
        )]
    if role == "function":
        return [tool_result_item(
            message.get("name"),
            message.get("content", ""),
            name=message.get("name"),
            is_error=message.get("is_error", False),
            provider_raw=message,
        )]
    return [provider_item("openai_chat", message)]


def openai_chat_message_to_item(message):
    return openai_chat_message_to_items(message)[0]


def openai_chat_messages_to_items(messages):
    items = []
    for message in messages:
        items.extend(openai_chat_message_to_items(message))
    return items


def _tool_call_arguments(call):
    raw_arguments = call.get("raw_arguments")
    if isinstance(raw_arguments, str):
        return raw_arguments
    return json.dumps(call.get("input", {}), separators=(",", ":"))


def _json_text(value, max_chars=4000):
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated: {len(text)} chars total]"
    return text


def _portable_content_block_text(block, target_protocol):
    block_type = block.get("type")
    if block_type == "text":
        return str(block.get("text", ""))
    if block_type in ["image", "file", "audio", "refusal"]:
        provider = block.get("provider") or "unknown"
        return (
            f"[Transcript content not native to {target_protocol}: "
            f"{block_type} block from {provider}]\n"
            f"{_json_text(block.get('provider_raw') or block.get('value') or block)}"
        )
    if block_type == "provider_content":
        provider = block.get("provider") or "unknown"
        return (
            f"[Transcript content not native to {target_protocol}: "
            f"provider_content from {provider}]\n"
            f"{_json_text(block.get('value'))}"
        )
    return (
        f"[Transcript content not native to {target_protocol}: {block_type or 'unknown'} block]\n"
        f"{_json_text(block)}"
    )


def _portable_item_text(item, target_protocol):
    item_type = item.get("type")
    if item_type == "response_metadata":
        fields = {
            key: item.get(key)
            for key in [
                "provider",
                "protocol",
                "id",
                "model",
                "status",
                "previous_response_id",
                "conversation",
                "error",
                "incomplete_details",
            ]
            if item.get(key) is not None
        }
        return (
            f"[Transcript metadata preserved while rendering to {target_protocol}]\n"
            f"{_json_text(fields)}"
        )
    if item_type == "reasoning":
        fields = {
            "status": item.get("status"),
            "summary": item.get("summary"),
            "content": item.get("content"),
            "provider_ids": item.get("provider_ids"),
            "encrypted_content_present": item.get("encrypted_content") is not None,
        }
        return (
            f"[Previous assistant reasoning item not native to {target_protocol}]\n"
            f"{_json_text({k: v for k, v in fields.items() if v not in [None, False]})}"
        )
    if item_type == "tool_call":
        fields = {
            "tool_kind": item.get("tool_kind"),
            "call_id": item.get("call_id") or item.get("id"),
            "name": item.get("name"),
            "input": item.get("input"),
            "raw_arguments": item.get("raw_arguments"),
            "status": item.get("status"),
            "provider_ids": item.get("provider_ids"),
        }
        return (
            f"[Previous assistant tool call not native to {target_protocol}]\n"
            f"{_json_text({k: v for k, v in fields.items() if v is not None})}"
        )
    if item_type == "tool_result":
        fields = {
            "tool_kind": item.get("tool_kind"),
            "tool_call_id": item.get("tool_call_id"),
            "name": item.get("name"),
            "is_error": item.get("is_error"),
            "content": item_text(item),
            "provider_ids": item.get("provider_ids"),
        }
        return (
            f"[Previous tool result not native to {target_protocol}]\n"
            f"{_json_text({k: v for k, v in fields.items() if v is not None})}"
        )
    if item_type == "provider_item":
        provider = item.get("provider") or "unknown"
        return (
            f"[Provider-specific transcript item from {provider} rendered as text for {target_protocol}]\n"
            f"{_json_text(item.get('value'))}"
        )
    return (
        f"[Transcript item not native to {target_protocol}: {item_type or 'unknown'}]\n"
        f"{_json_text(item)}"
    )


def _portable_chat_message(item, target_protocol, role="user"):
    return {"role": role, "content": _portable_item_text(item, target_protocol)}


def _openai_chat_content_from_blocks(blocks):
    provider_parts = []
    text_parts = []
    for block in blocks or []:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "provider_content" and block.get("provider") == "openai_chat":
            provider_parts.append(_copy(block.get("value", {})))
        elif block_type == "image" and block.get("provider") == "openai_chat":
            provider_parts.append(_copy(block.get("value", {})))
        else:
            text_parts.append(_portable_content_block_text(block, "openai_chat"))
    text = "\n".join(part for part in text_parts if part)
    if not provider_parts:
        return text
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.extend(provider_parts)
    return parts


def _openai_chat_tool_call(call):
    return {
        "id": call.get("call_id") or call.get("id"),
        "type": "function",
        "function": {
            "name": call.get("name"),
            "arguments": _tool_call_arguments(call),
        },
    }


def items_to_openai_chat_messages(items):
    messages = []
    i = 0
    while i < len(items):
        item = items[i]
        item_type = item.get("type")
        if item_type == "instruction":
            role = item.get("authority") or "system"
            if role not in ["system", "developer"]:
                role = "system"
            messages.append({"role": role, "content": item_text(item)})
            i += 1
            continue
        if item_type == "message":
            role = item.get("role")
            if role not in ["system", "developer", "user", "assistant"]:
                messages.append(_portable_chat_message(item, "openai_chat"))
                i += 1
                continue
            msg = {"role": role}
            content = _openai_chat_content_from_blocks(item.get("content", []))
            tool_calls = []
            portable_notes = []
            i += 1
            if role == "assistant":
                while i < len(items) and items[i].get("type") == "tool_call":
                    call = items[i]
                    if is_app_tool_call(call):
                        tool_calls.append(_openai_chat_tool_call(call))
                    else:
                        portable_notes.append(_portable_item_text(call, "openai_chat"))
                    i += 1
            if portable_notes:
                note_text = "\n\n".join(portable_notes)
                content = f"{content}\n\n{note_text}" if content else note_text
            if content or role != "assistant" or not tool_calls:
                msg["content"] = content
            if role == "assistant" and tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        if item_type == "tool_call":
            if is_app_tool_call(item):
                messages.append({
                    "role": "assistant",
                    "tool_calls": [_openai_chat_tool_call(item)],
                })
            else:
                messages.append(_portable_chat_message(item, "openai_chat", role="assistant"))
            i += 1
            continue
        if item_type == "tool_result":
            if item.get("tool_kind", "function") == "function":
                msg = {
                    "role": "tool",
                    "tool_call_id": item.get("tool_call_id"),
                    "content": item_text(item),
                }
                if item.get("name"):
                    msg["name"] = item["name"]
                messages.append(msg)
            else:
                messages.append(_portable_chat_message(item, "openai_chat"))
            i += 1
            continue
        if item_type == "provider_item" and item.get("provider") == "openai_chat":
            messages.append(_copy(item.get("value", {})))
            i += 1
            continue
        if item_type == "provider_item":
            messages.append(_portable_chat_message(item, "openai_chat"))
            i += 1
            continue
        if item_type in ["response_metadata", "reasoning"]:
            messages.append(_portable_chat_message(item, "openai_chat"))
            i += 1
            continue
        messages.append(_portable_chat_message(item, "openai_chat"))
        i += 1
    return messages


def openai_chat_response_to_items(response):
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise TranscriptFormatError(f"OpenAI Chat response missing choices[0].message: {e}")
    return [response_metadata_item("openai", "openai_chat", response)] + openai_chat_message_to_items(message)


def openai_tools_to_anthropic_tools(tools):
    if tools is None:
        return None
    anthropic_tools = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function = tool.get("function") or {}
        anthropic_tools.append({
            "name": function.get("name"),
            "description": function.get("description", ""),
            "input_schema": _copy(function.get("parameters", {"type": "object", "properties": {}})),
        })
    return anthropic_tools


def _anthropic_text_blocks(blocks):
    out = []
    for block in blocks or []:
        block_type = block.get("type")
        if block_type == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "provider_content" and block.get("provider") == "anthropic":
            out.append(_copy(block.get("value", {})))
        else:
            out.append({"type": "text", "text": _portable_content_block_text(block, "anthropic_messages")})
    return out


def _anthropic_content_from_message(item):
    content = []
    for block in item.get("content", []) or []:
        block_type = block.get("type")
        if block_type == "text":
            content.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "provider_content" and block.get("provider") == "anthropic":
            content.append(_copy(block.get("value", {})))
        else:
            content.append({"type": "text", "text": _portable_content_block_text(block, "anthropic_messages")})
    return content


def _anthropic_tool_use(call):
    return {
        "type": "tool_use",
        "id": call.get("call_id") or call.get("id"),
        "name": call.get("name"),
        "input": _copy(call.get("input", {})),
    }


def items_to_anthropic_parts(items):
    system_parts = []
    messages = []
    seen_conversation = False
    i = 0
    while i < len(items):
        item = items[i]
        item_type = item.get("type")
        if item_type == "instruction":
            content_text = item_text(item)
            if not seen_conversation:
                if content_text:
                    system_parts.append(content_text)
            else:
                messages.append({"role": "system", "content": content_text})
            i += 1
            continue
        seen_conversation = True
        if item_type == "message":
            role = item.get("role")
            if role not in ["user", "assistant"]:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": _portable_item_text(item, "anthropic_messages")}],
                })
                i += 1
                continue
            content = _anthropic_content_from_message(item)
            i += 1
            if role == "assistant":
                while i < len(items) and items[i].get("type") == "tool_call":
                    call = items[i]
                    if is_app_tool_call(call):
                        content.append(_anthropic_tool_use(call))
                    else:
                        content.append({
                            "type": "text",
                            "text": _portable_item_text(call, "anthropic_messages"),
                        })
                    i += 1
            messages.append({"role": role, "content": content or ""})
            continue
        if item_type == "tool_call":
            if is_app_tool_call(item):
                messages.append({"role": "assistant", "content": [_anthropic_tool_use(item)]})
            else:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": _portable_item_text(item, "anthropic_messages")}],
                })
            i += 1
            continue
        if item_type == "tool_result":
            content = []
            while i < len(items) and items[i].get("type") == "tool_result":
                result = items[i]
                if result.get("tool_kind", "function") == "function":
                    block = {
                        "type": "tool_result",
                        "tool_use_id": result.get("tool_call_id"),
                        "content": _anthropic_text_blocks(result.get("content", [])) or item_text(result),
                    }
                    if result.get("is_error"):
                        block["is_error"] = True
                    content.append(block)
                else:
                    content.append({
                        "type": "text",
                        "text": _portable_item_text(result, "anthropic_messages"),
                    })
                i += 1
            messages.append({"role": "user", "content": content})
            continue
        if item_type == "provider_item" and item.get("provider") == "anthropic":
            messages.append(_copy(item.get("value", {})))
            i += 1
            continue
        if item_type == "provider_item":
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": _portable_item_text(item, "anthropic_messages")}],
            })
            i += 1
            continue
        if item_type in ["response_metadata", "reasoning"]:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": _portable_item_text(item, "anthropic_messages")}],
            })
            i += 1
            continue
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": _portable_item_text(item, "anthropic_messages")}],
        })
        i += 1
    return "\n\n".join(system_parts), messages


def anthropic_response_to_items(response):
    if response.get("type") != "message" or response.get("role") != "assistant":
        raise TranscriptFormatError("Anthropic response is not an assistant message")
    items = [response_metadata_item("anthropic", "anthropic_messages", response)]
    blocks = []
    tool_calls = []
    for block in response.get("content", []) or []:
        block_type = block.get("type")
        if block_type == "text":
            blocks.append(text_block(block.get("text", "")))
        elif block_type == "tool_use":
            call_id = block.get("id")
            tool_calls.append(tool_call_item(
                call_id,
                block.get("name"),
                _copy(block.get("input", {})),
                provider_ids={"anthropic": call_id},
                provider_raw=block,
            ))
        else:
            blocks.append(provider_content_block("anthropic", block))
    items.append(message_item(
        "assistant",
        blocks,
        status=response.get("stop_reason"),
        provider_ids={"anthropic": response.get("id")},
        provider_raw=response,
    ))
    items.extend(tool_calls)
    return items


def _responses_content_to_blocks(content):
    blocks = []
    for part in content or []:
        if not isinstance(part, dict):
            blocks.append(text_block(part))
            continue
        part_type = part.get("type")
        if part_type in ["input_text", "output_text", "text"]:
            blocks.append(text_block(
                part.get("text", ""),
                annotations=part.get("annotations"),
                logprobs=part.get("logprobs"),
                provider_raw=part,
            ))
        elif part_type in ["input_image", "output_image", "image"]:
            blocks.append(media_block("image", part, provider="openai_responses", provider_raw=part))
        elif part_type in ["input_file", "output_file", "file"]:
            blocks.append(media_block("file", part, provider="openai_responses", provider_raw=part))
        elif part_type in ["input_audio", "output_audio", "audio"]:
            blocks.append(media_block("audio", part, provider="openai_responses", provider_raw=part))
        elif part_type == "refusal":
            blocks.append(media_block("refusal", part.get("refusal", part), provider="openai_responses", provider_raw=part))
        else:
            blocks.append(provider_content_block("openai_responses", part))
    return blocks


def _responses_item_provider_ids(item):
    ids = {}
    if item.get("id") is not None:
        ids["id"] = item.get("id")
    if item.get("call_id") is not None:
        ids["call_id"] = item.get("call_id")
    return {"openai_responses": ids} if ids else None


def openai_responses_item_to_items(item):
    item_type = item.get("type")
    if item_type == "message":
        return [message_item(
            item.get("role"),
            _responses_content_to_blocks(item.get("content", [])),
            status=item.get("status"),
            provider_ids=_responses_item_provider_ids(item),
            provider_raw=item,
        )]
    if item_type == "function_call":
        raw_arguments = item.get("arguments", "{}")
        input_value, parse_error = _parse_json_arguments(raw_arguments)
        return [tool_call_item(
            item.get("call_id"),
            item.get("name"),
            input_value,
            raw_arguments=raw_arguments,
            provider_ids=_responses_item_provider_ids(item),
            parse_error=parse_error,
            status=item.get("status"),
            provider_raw=item,
        )]
    if item_type == "custom_tool_call":
        return [tool_call_item(
            item.get("call_id"),
            item.get("name"),
            item.get("input", ""),
            raw_arguments=item.get("input"),
            provider_ids=_responses_item_provider_ids(item),
            tool_kind="custom",
            status=item.get("status"),
            provider_raw=item,
        )]
    if item_type in ["function_call_output", "custom_tool_call_output"]:
        return [tool_result_item(
            item.get("call_id"),
            item.get("output", ""),
            tool_kind="custom" if item_type.startswith("custom_") else "function",
            provider_ids=_responses_item_provider_ids(item),
            provider_raw=item,
        )]
    if item_type == "reasoning":
        return [reasoning_item(
            content=item.get("content"),
            summary=item.get("summary"),
            encrypted_content=item.get("encrypted_content"),
            status=item.get("status"),
            provider_ids=_responses_item_provider_ids(item),
            provider_raw=item,
        )]
    if item_type and item_type.endswith("_call"):
        return [tool_call_item(
            item.get("call_id") or item.get("id"),
            item.get("name") or item_type,
            item,
            provider_ids=_responses_item_provider_ids(item),
            tool_kind=item_type,
            status=item.get("status"),
            provider_raw=item,
        )]
    return [provider_item("openai_responses", item)]


def openai_responses_response_to_items(response):
    if response.get("object") != "response" and not isinstance(response.get("output"), list):
        raise TranscriptFormatError("OpenAI Responses response is not a response object")
    items = [response_metadata_item("openai", "openai_responses", response)]
    for output_item in response.get("output", []) or []:
        items.extend(openai_responses_item_to_items(output_item))
    return items


def _responses_content_from_blocks(blocks, role):
    out = []
    text_type = "output_text" if role == "assistant" else "input_text"
    for block in blocks or []:
        block_type = block.get("type")
        if block_type == "text":
            part = {"type": text_type, "text": block.get("text", "")}
            _put_optional(part, "annotations", block.get("annotations"))
            _put_optional(part, "logprobs", block.get("logprobs"))
            out.append(part)
        elif block_type in ["image", "file", "audio", "refusal"] and block.get("provider") == "openai_responses":
            out.append(_copy(block.get("provider_raw") or block.get("value", {})))
        elif block_type == "provider_content" and block.get("provider") == "openai_responses":
            out.append(_copy(block.get("value", {})))
        else:
            out.append({
                "type": text_type,
                "text": _portable_content_block_text(block, "openai_responses"),
            })
    return out


def _responses_note_item(item, role="user"):
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": _portable_item_text(item, "openai_responses")}],
    }


def _responses_tool_call_item(call):
    if call.get("tool_kind", "function") == "function":
        return {
            "type": "function_call",
            "call_id": call.get("call_id") or call.get("id"),
            "name": call.get("name"),
            "arguments": _tool_call_arguments(call),
        }
    if call.get("tool_kind") == "custom":
        return {
            "type": "custom_tool_call",
            "call_id": call.get("call_id") or call.get("id"),
            "name": call.get("name"),
            "input": call.get("raw_arguments", call.get("input", "")),
        }
    if call.get("provider_raw"):
        return _copy(call["provider_raw"])
    return _responses_note_item(call, role="assistant")


def items_to_openai_responses_parts(items):
    # Responses accepts prior output items, including reasoning and function
    # calls, as future input. Keep those as top-level transcript items so a
    # future Responses adapter can round-trip them without inventing a chat
    # message wrapper.
    instructions = []
    input_items = []
    seen_conversation = False
    for item in items:
        item_type = item.get("type")
        if item_type == "instruction":
            text = item_text(item)
            if not seen_conversation:
                if text:
                    instructions.append(text)
            else:
                input_items.append({
                    "type": "message",
                    "role": item.get("authority") or "system",
                    "content": [{"type": "input_text", "text": text}],
                })
            continue
        seen_conversation = True
        if item_type == "response_metadata":
            input_items.append(_responses_note_item(item))
        elif item_type == "message":
            if item.get("role") not in ["system", "developer", "user", "assistant"]:
                input_items.append(_responses_note_item(item))
                continue
            msg = {
                "type": "message",
                "role": item.get("role"),
                "content": _responses_content_from_blocks(item.get("content", []), item.get("role")),
            }
            _put_optional(msg, "status", item.get("status"))
            input_items.append(msg)
        elif item_type == "tool_call":
            input_items.append(_responses_tool_call_item(item))
        elif item_type == "tool_result":
            output = item_text(item)
            if item.get("tool_kind") == "custom":
                input_items.append({
                    "type": "custom_tool_call_output",
                    "call_id": item.get("tool_call_id"),
                    "output": output,
                })
            elif item.get("tool_kind", "function") == "function":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": item.get("tool_call_id"),
                    "output": output,
                })
            elif item.get("provider_raw"):
                input_items.append(_copy(item["provider_raw"]))
            else:
                input_items.append(_responses_note_item(item))
        elif item_type == "reasoning":
            if item.get("provider_raw"):
                input_items.append(_copy(item["provider_raw"]))
            else:
                reasoning = {"type": "reasoning"}
                _put_optional(reasoning, "content", item.get("content"))
                _put_optional(reasoning, "summary", item.get("summary"))
                _put_optional(reasoning, "encrypted_content", item.get("encrypted_content"))
                input_items.append(reasoning)
        elif item_type == "provider_item" and item.get("provider") == "openai_responses":
            input_items.append(_copy(item.get("value", {})))
        elif item_type == "provider_item":
            input_items.append(_responses_note_item(item))
        else:
            input_items.append(_responses_note_item(item))
    return "\n\n".join(instructions), input_items


def openai_tools_to_responses_tools(tools):
    if tools is None:
        return None
    out = []
    for tool in tools:
        if tool.get("type") != "function":
            out.append(_copy(tool))
            continue
        function = tool.get("function") or {}
        response_tool = {
            "type": "function",
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parameters": _copy(function.get("parameters", {"type": "object", "properties": {}})),
        }
        _put_optional(response_tool, "strict", function.get("strict"))
        out.append(response_tool)
    return out
