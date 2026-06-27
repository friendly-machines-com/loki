import copy
import json


TRANSCRIPT_SCHEMA = "day-agent.transcript.v1"


class TranscriptFormatError(ValueError):
    pass


def text_block(text):
    return {"type": "text", "text": str(text)}


def instruction_item(content, authority="system"):
    return {
        "type": "instruction",
        "authority": authority or "system",
        "content": content_blocks(content),
    }


def message_item(role, content=None):
    return {
        "type": "message",
        "role": role,
        "content": content_blocks(content),
    }


def tool_call_block(call_id, name, input_value, raw_arguments=None, provider_ids=None,
                    parse_error=None):
    block = {
        "type": "tool_call",
        "id": call_id,
        "name": name,
        "input": input_value,
    }
    if raw_arguments is not None:
        block["raw_arguments"] = raw_arguments
    if provider_ids:
        block["provider_ids"] = provider_ids
    if parse_error:
        block["parse_error"] = parse_error
    return block


def tool_result_item(tool_call_id, content, name=None, is_error=False):
    item = {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "content": content_blocks(content),
        "is_error": bool(is_error),
    }
    if name:
        item["name"] = name
    return item


def provider_content_block(provider, value):
    return {
        "type": "provider_content",
        "provider": provider,
        "value": copy.deepcopy(value),
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
                blocks.append(copy.deepcopy(block))
            else:
                blocks.append(text_block(block))
        return blocks
    return [text_block(content)]


def blocks_text(blocks):
    parts = []
    for block in blocks or []:
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)


def item_text(item):
    return blocks_text(item.get("content", []))


def item_tool_calls(item):
    if item.get("type") != "message" or item.get("role") != "assistant":
        return []
    return [block for block in item.get("content", []) if block.get("type") == "tool_call"]


def item_has_tool_calls(item):
    return bool(item_tool_calls(item))


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
        "items": items,
        "session_todos": session_todos,
    }


def load_log_blob(blob):
    if isinstance(blob, dict) and blob.get("schema") == TRANSCRIPT_SCHEMA:
        return copy.deepcopy(blob.get("items", [])), copy.deepcopy(blob.get("session_todos", []))
    if isinstance(blob, dict) and "items" in blob:
        return copy.deepcopy(blob.get("items", [])), copy.deepcopy(blob.get("session_todos", []))
    if isinstance(blob, dict) and "messages" in blob:
        return openai_chat_messages_to_items(blob["messages"]), copy.deepcopy(blob.get("session_todos", []))
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


def _openai_tool_call_to_block(tool_call):
    call_id = tool_call.get("id")
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw_arguments = function.get("arguments", "{}")
    try:
        input_value = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        parse_error = None
    except Exception as e:
        input_value = {}
        parse_error = str(e)
    return tool_call_block(
        call_id,
        name,
        input_value,
        raw_arguments=raw_arguments,
        provider_ids={"openai_chat": call_id},
        parse_error=parse_error,
    )


def openai_chat_message_to_item(message):
    role = message.get("role")
    if role in ["system", "developer"]:
        return instruction_item(message.get("content", ""), authority=role)
    if role in ["user", "assistant"]:
        blocks = openai_content_to_blocks(message.get("content"))
        if role == "assistant":
            blocks.extend(_openai_tool_call_to_block(tc)
                          for tc in message.get("tool_calls", []) or [])
        return message_item(role, blocks)
    if role == "tool":
        return tool_result_item(
            message.get("tool_call_id"),
            message.get("content", ""),
            name=message.get("name"),
            is_error=message.get("is_error", False),
        )
    if role == "function":
        return tool_result_item(
            message.get("name"),
            message.get("content", ""),
            name=message.get("name"),
            is_error=message.get("is_error", False),
        )
    return {
        "type": "provider_item",
        "provider": "openai_chat",
        "value": copy.deepcopy(message),
    }


def openai_chat_messages_to_items(messages):
    return [openai_chat_message_to_item(message) for message in messages]


def _tool_call_arguments(block):
    raw_arguments = block.get("raw_arguments")
    if isinstance(raw_arguments, str):
        return raw_arguments
    return json.dumps(block.get("input", {}), separators=(",", ":"))


def items_to_openai_chat_messages(items):
    messages = []
    for item in items:
        item_type = item.get("type")
        if item_type == "instruction":
            role = item.get("authority") or "system"
            if role not in ["system", "developer"]:
                role = "system"
            messages.append({"role": role, "content": item_text(item)})
        elif item_type == "message":
            role = item.get("role")
            text = item_text(item)
            tool_calls = item_tool_calls(item)
            msg = {"role": role}
            if text or role != "assistant" or not tool_calls:
                msg["content"] = text
            if role == "assistant" and tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": _tool_call_arguments(block),
                        },
                    }
                    for block in tool_calls
                ]
            messages.append(msg)
        elif item_type == "tool_result":
            msg = {
                "role": "tool",
                "tool_call_id": item.get("tool_call_id"),
                "content": item_text(item),
            }
            if item.get("name"):
                msg["name"] = item["name"]
            messages.append(msg)
        elif item_type == "provider_item" and item.get("provider") == "openai_chat":
            messages.append(copy.deepcopy(item.get("value", {})))
        else:
            raise TranscriptFormatError(f"cannot render {item_type!r} as OpenAI Chat")
    return messages


def openai_chat_response_to_items(response):
    try:
        message = response["choices"][0]["message"]
    except Exception as e:
        raise TranscriptFormatError(f"OpenAI Chat response missing choices[0].message: {e}")
    return [openai_chat_message_to_item(message)]


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
            "input_schema": copy.deepcopy(function.get("parameters", {"type": "object", "properties": {}})),
        })
    return anthropic_tools


def _anthropic_text_blocks(blocks):
    out = []
    for block in blocks or []:
        block_type = block.get("type")
        if block_type == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "provider_content" and block.get("provider") == "anthropic":
            out.append(copy.deepcopy(block.get("value", {})))
        else:
            raise TranscriptFormatError(f"cannot render content block {block_type!r} as Anthropic text content")
    return out


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
                raise TranscriptFormatError(f"cannot render message role {role!r} as Anthropic")
            content = []
            for block in item.get("content", []):
                block_type = block.get("type")
                if block_type == "text":
                    content.append({"type": "text", "text": block.get("text", "")})
                elif block_type == "tool_call" and role == "assistant":
                    content.append({
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": copy.deepcopy(block.get("input", {})),
                    })
                elif block_type == "provider_content" and block.get("provider") == "anthropic":
                    content.append(copy.deepcopy(block.get("value", {})))
                else:
                    raise TranscriptFormatError(f"cannot render content block {block_type!r} as Anthropic")
            messages.append({"role": role, "content": content or ""})
            i += 1
            continue
        if item_type == "tool_result":
            content = []
            while i < len(items) and items[i].get("type") == "tool_result":
                result = items[i]
                block = {
                    "type": "tool_result",
                    "tool_use_id": result.get("tool_call_id"),
                    "content": _anthropic_text_blocks(result.get("content", [])) or item_text(result),
                }
                if result.get("is_error"):
                    block["is_error"] = True
                content.append(block)
                i += 1
            messages.append({"role": "user", "content": content})
            continue
        if item_type == "provider_item" and item.get("provider") == "anthropic":
            messages.append(copy.deepcopy(item.get("value", {})))
            i += 1
            continue
        raise TranscriptFormatError(f"cannot render {item_type!r} as Anthropic")
    return "\n\n".join(system_parts), messages


def anthropic_response_to_items(response):
    if response.get("type") != "message" or response.get("role") != "assistant":
        raise TranscriptFormatError("Anthropic response is not an assistant message")
    blocks = []
    for block in response.get("content", []) or []:
        block_type = block.get("type")
        if block_type == "text":
            blocks.append(text_block(block.get("text", "")))
        elif block_type == "tool_use":
            call_id = block.get("id")
            blocks.append(tool_call_block(
                call_id,
                block.get("name"),
                copy.deepcopy(block.get("input", {})),
                provider_ids={"anthropic": call_id},
            ))
        else:
            blocks.append(provider_content_block("anthropic", block))
    return [message_item("assistant", blocks)]
