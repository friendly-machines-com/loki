import copy
import json
import sys
import uuid
from dataclasses import dataclass, field


TRANSCRIPT_SCHEMA = "day-agent.session.v4"
OPENAI_CHAT = "openai_chat"
ANTHROPIC_MESSAGES = "anthropic_messages"
OPENAI_RESPONSES = "openai_responses"
OPENAI_CHAT_REASONING_FIELDS = (
    "reasoning_content",
    "reasoning",
    "reasoning_details",
)


class TranscriptFormatError(ValueError):
    pass


class ProjectionError(TranscriptFormatError):
    pass


def _copy(value):
    return copy.deepcopy(value)


def _put_optional(target, key, value):
    if value is not None:
        target[key] = _copy(value)


def report_unknown(protocol, context, value):
    rendered = json.dumps(
        value, ensure_ascii=True, sort_keys=True, default=str)
    sys.stdout.flush()
    print(
        f"Unknown {protocol} {context}:\n{rendered}",
        file=sys.stderr,
    )
    sys.stderr.flush()


def _protocol_data(target, protocol, value):
    if value:
        target.setdefault("protocol_data", {})[protocol] = _copy(value)


def _protocol_fields(value, protocol):
    data = value.get("protocol_data", {})
    if not isinstance(data, dict):
        return {}
    fields = data.get(protocol, {})
    return fields if isinstance(fields, dict) else {}


def text_block(text, **fields):
    block = {"type": "text", "text": str(text)}
    for key, value in fields.items():
        _put_optional(block, key, value)
    return block


def media_block(kind, value):
    return {"type": kind, "value": _copy(value)}


def _image_source_from_url(value):
    if not isinstance(value, str):
        return {"type": "url", "url": value}
    if value.startswith("data:"):
        metadata, separator, data = value.partition(",")
        parts = metadata[5:].split(";")
        if (separator and len(parts) >= 2
                and parts[-1].lower() == "base64"
                and parts[0] and data):
            return {
                "type": "base64",
                "media_type": parts[0],
                "data": data,
            }
    return {"type": "url", "url": value}


def content_blocks(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [] if content == "" else [text_block(content)]
    if isinstance(content, list):
        return [
            _copy(block) if isinstance(block, dict) else text_block(block)
            for block in content
        ]
    return [text_block(content)]


def message_item(role, content=None, **fields):
    item = {
        "type": "message",
        "role": role,
        "content": content_blocks(content),
    }
    for key, value in fields.items():
        _put_optional(item, key, value)
    return item


def instruction_item(content, authority="system", **fields):
    return message_item(authority or "system", content, **fields)


def _json_arguments(input_value):
    return json.dumps(
        input_value if input_value is not None else {},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def tool_call_item(call_id, name, input_value=None, raw_arguments=None,
                   parse_error=None, tool_kind="function", status=None,
                   execution="client", protocol_data=None):
    arguments = (
        raw_arguments if isinstance(raw_arguments, str)
        else _json_arguments(input_value)
    )
    item = {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }
    if tool_kind != "function":
        item["tool_kind"] = tool_kind
    if execution != "client":
        item["execution"] = execution
    _put_optional(item, "parse_error", parse_error)
    _put_optional(item, "status", status)
    if protocol_data:
        item["protocol_data"] = _copy(protocol_data)
    return item


def tool_result_item(tool_call_id, content, name=None, is_error=False,
                     tool_kind="function", execution=None):
    item = {
        "type": "tool_result",
        "call_id": tool_call_id,
        "content": content_blocks(content),
        "is_error": bool(is_error),
    }
    _put_optional(item, "name", name)
    if tool_kind != "function":
        item["tool_kind"] = tool_kind
    if execution:
        item["execution"] = _copy(execution)
    return item


def tool_result_for_call(call, content, is_error=False, execution=None):
    return tool_result_item(
        tool_call_id(call),
        content,
        name=tool_call_name(call),
        is_error=is_error,
        tool_kind=call.get("tool_kind", "function"),
        execution=execution,
    )


def provider_operation_item(call_id, name, input_value, output=None, *,
                            status=None, protocol_data=None):
    item = {
        "type": "provider_operation",
        "call_id": call_id,
        "name": name,
        "input": _copy(input_value if input_value is not None else {}),
        "output": _copy(output),
    }
    _put_optional(item, "status", status)
    if protocol_data:
        item["protocol_data"] = _copy(protocol_data)
    return item


def model_response_event(protocol, items, *, provider=None, endpoint=None,
                         model=None,
                         status="completed", stop_reason=None, usage=None,
                         protocol_data=None):
    event = {
        "type": "model_response",
        "protocol": protocol,
        "items": _copy(items),
        "status": status or "completed",
    }
    _put_optional(event, "provider", provider)
    _put_optional(event, "endpoint", endpoint)
    _put_optional(event, "model", model)
    _put_optional(event, "stop_reason", stop_reason)
    _put_optional(event, "usage", usage)
    if protocol_data:
        event["protocol_data"] = _copy(protocol_data)
    return event


@dataclass
class DecodedTurn:
    """One decoded provider response.

    ``items`` are the ordered canonical output items inside one real model
    response.  The response boundary is added to the session exactly once by
    ``to_event``.  Transport failures never construct this object.
    """

    items: list
    metadata: dict = field(default_factory=dict)
    complete: bool = True

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __bool__(self):
        return bool(self.items)

    def to_event(self):
        status = self.metadata.get("status")
        if status is None:
            status = "completed" if self.complete else "incomplete"
        return model_response_event(
            self.metadata.get("protocol", "unknown"),
            self.items,
            provider=(
                self.metadata.get("provider_id")
                or self.metadata.get("provider_name")),
            endpoint=self.metadata.get("endpoint"),
            model=self.metadata.get("model"),
            status=status,
            stop_reason=self.metadata.get("stop_reason"),
            usage=self.metadata.get("usage"),
            protocol_data=self.metadata.get("protocol_data"),
        )


def coerce_decoded_turn(value):
    if isinstance(value, DecodedTurn):
        return value
    if isinstance(value, list):
        return DecodedTurn(value)
    raise TranscriptFormatError(
        "chat adapter must return DecodedTurn or a list of response items")


def blocks_text(blocks):
    parts = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in ["text", "input_text", "output_text"]:
            value = block.get("text", "")
        elif block_type == "refusal":
            value = block.get("text", block.get("refusal", ""))
        else:
            continue
        if isinstance(value, str) and value:
            parts.append(value)
    return "\n".join(parts)


def item_text(item):
    if not isinstance(item, dict):
        return ""
    if item.get("type") == "model_response":
        return "\n".join(
            text for text in (item_text(value)
                              for value in item.get("items", []))
            if text
        )
    return blocks_text(item.get("content", []))


def _parse_json_arguments(raw):
    if not isinstance(raw, str):
        return _copy(raw), None
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as error:
        return {}, str(error)


def tool_call_id(item):
    return item.get("call_id")


def tool_call_name(item):
    return item.get("name")


def tool_call_input(item):
    if "input" in item:
        return _copy(item["input"])
    raw = item.get("arguments", "{}")
    value, error = _parse_json_arguments(raw)
    if error:
        raise TranscriptFormatError(
            f"tool arguments are not valid JSON: {error}")
    return value


def is_app_tool_call(item):
    return (
        isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("execution", "client") == "client"
        and item.get("tool_kind", "function") == "function"
    )


def _event_items(event):
    if isinstance(event, dict) and event.get("type") == "model_response":
        return event.get("items", [])
    return [event]


def projection_target(protocol, *, provider_id=None, provider_name=None,
                      endpoint=None, model=None):
    """Describe the concrete connection receiving a projection.

    Opaque continuation data is safe to replay only to the connection that
    produced it.  Protocol equality alone is insufficient: two unrelated
    providers can implement the same wire format while using incompatible
    signatures, encrypted reasoning payloads, and provider-scoped IDs.
    """
    return {
        "protocol": protocol,
        "provider_id": provider_id,
        "provider_name": provider_name,
        "endpoint": endpoint,
        "model": model,
    }


def _native_replay_allowed(event, protocol, target):
    if event.get("protocol") != protocol:
        return False
    # Format-level round-trip helpers historically had no runtime connection
    # argument.  Keep that useful behavior for isolated decode/replay tests;
    # production Provider payload construction always supplies a target.
    if target is None:
        return True
    if target.get("protocol") != protocol:
        return False

    origin_endpoint = event.get("endpoint")
    target_endpoint = target.get("endpoint")
    if isinstance(origin_endpoint, str) and origin_endpoint:
        if origin_endpoint != target_endpoint:
            return False
        origin_provider = event.get("provider")
        target_provider_id = target.get("provider_id")
        if (isinstance(origin_provider, str)
                and origin_provider
                and isinstance(target_provider_id, str)
                and target_provider_id
                and origin_provider != target_provider_id):
            return False
        origin_model = event.get("model")
        target_model = target.get("model")
        return not (
            isinstance(origin_model, str)
            and origin_model
            and isinstance(target_model, str)
            and target_model
            and origin_model != target_model
        )

    # Older v4 events did not record the endpoint.  A catalog provider ID is a
    # usable conservative fallback.  Provider display names (especially the
    # generic explicit-connection label) are not identities.
    origin_provider = event.get("provider")
    target_provider_id = target.get("provider_id")
    if not bool(
        isinstance(origin_provider, str)
        and origin_provider
        and isinstance(target_provider_id, str)
        and target_provider_id
        and origin_provider == target_provider_id
    ):
        return False
    origin_model = event.get("model")
    target_model = target.get("model")
    return not (
        isinstance(origin_model, str)
        and origin_model
        and isinstance(target_model, str)
        and target_model
        and origin_model != target_model
    )


def response_tool_calls(value):
    if isinstance(value, DecodedTurn):
        values = value.items
    elif isinstance(value, dict):
        values = _event_items(value)
    else:
        values = value or []
    calls = []
    for event in values:
        for item in _event_items(event):
            if is_app_tool_call(item):
                calls.append(item)
    return calls


def pending_tool_calls(events):
    pending = {}
    order = []
    for event in events or []:
        for item in _event_items(event):
            if is_app_tool_call(item):
                call_id = tool_call_id(item)
                if call_id is None:
                    continue
                pending[call_id] = item
                if call_id in order:
                    order.remove(call_id)
                order.append(call_id)
        if isinstance(event, dict) and event.get("type") == "tool_result":
            call_id = event.get("call_id")
            pending.pop(call_id, None)
            if call_id in order:
                order.remove(call_id)
    return [pending[key] for key in order if key in pending]


def user_prompt_history(events):
    history = []
    for event in events or []:
        if (isinstance(event, dict)
                and event.get("type") == "message"
                and event.get("role") == "user"):
            text = item_text(event)
            if text:
                history.append(text)
    return history


def _validate_response(event, index):
    protocol = event.get("protocol")
    if not isinstance(protocol, str) or not protocol:
        raise TranscriptFormatError(
            f"event {index} model_response protocol must be a string")
    items = event.get("items")
    if not isinstance(items, list):
        raise TranscriptFormatError(
            f"event {index} model_response items must be a list")
    if any(not isinstance(item, dict) for item in items):
        raise TranscriptFormatError(
            f"event {index} model_response items must be objects")
    for field_name in ["provider", "endpoint", "model", "status"]:
        value = event.get(field_name)
        if value is not None and not isinstance(value, str):
            raise TranscriptFormatError(
                f"event {index} model_response {field_name} "
                "must be a string or null")
    response_calls = set()
    for item in items:
        if item.get("type") == "provider_operation":
            for field_name in ["call_id", "name"]:
                value = item.get(field_name)
                if not isinstance(value, str) or not value:
                    raise TranscriptFormatError(
                        f"event {index} provider operation requires "
                        f"{field_name}")
        if not is_app_tool_call(item):
            continue
        call_id = tool_call_id(item)
        if not isinstance(call_id, str) or not call_id:
            raise TranscriptFormatError(
                f"event {index} function call requires call_id")
        if call_id in response_calls:
            raise TranscriptFormatError(
                f"event {index} contains duplicate function call "
                f"{call_id!r}")
        response_calls.add(call_id)


def validate_events(events):
    if not isinstance(events, list):
        raise TranscriptFormatError("chat log events must be a list")
    pending = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise TranscriptFormatError(
                f"event {index} must be an object")
        event_type = event.get("type")
        if event_type == "model_response":
            _validate_response(event, index)
            for call in response_tool_calls(event):
                call_id = tool_call_id(call)
                pending.setdefault(call_id, []).append(index)
        elif event_type == "tool_result":
            call_id = event.get("call_id")
            execution = event.get("execution")
            if execution is not None and not isinstance(execution, dict):
                raise TranscriptFormatError(
                    f"event {index} tool result execution metadata "
                    "must be an object")
            matches = pending.get(call_id, [])
            if not matches:
                raise TranscriptFormatError(
                    f"event {index} tool result has no preceding unresolved "
                    f"call {call_id!r}")
            matches.pop()
            if not matches:
                pending.pop(call_id, None)
        elif event_type != "message":
            raise TranscriptFormatError(
                f"event {index} has unknown type {event_type!r}")
    return events


def new_log_blob(events, session_todos, toolsets=None):
    validate_events(events)
    if not isinstance(session_todos, list):
        raise TranscriptFormatError("session_todos must be a list")
    if toolsets is None:
        toolsets = []
    if (not isinstance(toolsets, list)
            or any(not isinstance(toolset, list) for toolset in toolsets)):
        raise TranscriptFormatError("toolsets must be a list of lists")
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "events": _copy(events),
        "toolsets": _copy(toolsets),
        "session_todos": _copy(session_todos),
    }


def load_log_blob(blob):
    if not isinstance(blob, dict) or blob.get("schema") != TRANSCRIPT_SCHEMA:
        raise TranscriptFormatError(
            f"expected chat log schema {TRANSCRIPT_SCHEMA!r}")
    events = blob.get("events")
    validate_events(events)
    todos = blob.get("session_todos", [])
    if not isinstance(todos, list):
        raise TranscriptFormatError("chat log session_todos must be a list")
    toolsets = blob.get("toolsets", [])
    if (not isinstance(toolsets, list)
            or any(not isinstance(toolset, list) for toolset in toolsets)):
        raise TranscriptFormatError("chat log toolsets must be a list of lists")
    return _copy(events), _copy(todos)


def log_toolsets(blob):
    return _copy(blob.get("toolsets", []))


def _native_content(protocol, value):
    return {
        "type": "native_content",
        "protocol": protocol,
        "value": _copy(value),
    }


def _native_output(protocol, value):
    return {
        "type": "native_output",
        "protocol": protocol,
        "value": _copy(value),
    }


def _chat_content_block(part):
    if not isinstance(part, dict):
        report_unknown(OPENAI_CHAT, "message content block", part)
        return _native_content(OPENAI_CHAT, part)
    part_type = part.get("type")
    if part_type == "text":
        block = text_block(
            part.get("text", ""),
            annotations=part.get("annotations"),
            citations=part.get("citations"),
            logprobs=part.get("logprobs"),
        )
        extras = {
            key: value for key, value in part.items()
            if key not in [
                "type", "text", "annotations", "citations", "logprobs"]
        }
        if extras:
            report_unknown(OPENAI_CHAT, "text content fields", extras)
            _protocol_data(block, OPENAI_CHAT, extras)
        return block
    if part_type == "refusal":
        block = {
            "type": "refusal",
            "text": part.get("refusal", part.get("text", "")),
        }
        extras = {
            key: value for key, value in part.items()
            if key not in ["type", "refusal", "text"]
        }
        if extras:
            _protocol_data(block, OPENAI_CHAT, extras)
        return block
    if part_type == "image_url":
        image = part.get("image_url", {})
        block = {
            "type": "image",
            "source": _image_source_from_url(
                image.get("url") if isinstance(image, dict)
                else image),
        }
        if isinstance(image, dict):
            _put_optional(block, "detail", image.get("detail"))
        native = {
            key: value for key, value in part.items()
            if key not in ["type", "image_url"]
        }
        if isinstance(image, dict):
            image_fields = {
                key: value for key, value in image.items()
                if key not in ["url", "detail"]
            }
            if image_fields:
                native["image_fields"] = image_fields
        if native:
            _protocol_data(block, OPENAI_CHAT, native)
        return block
    if part_type in ["input_audio", "audio"]:
        audio = part.get("input_audio", part.get("audio", {}))
        if not isinstance(audio, dict):
            audio = {}
        block = {
            "type": "audio",
            "source": {
                "type": "base64",
                "data": audio.get("data"),
                "format": audio.get("format"),
            },
        }
        native = {
            key: value for key, value in part.items()
            if key not in ["type", "input_audio", "audio"]
        }
        audio_fields = {
            key: value for key, value in audio.items()
            if key not in ["data", "format"]
        }
        if audio_fields:
            native["audio_fields"] = audio_fields
        if native:
            _protocol_data(block, OPENAI_CHAT, native)
        return block
    if part_type == "file":
        file_value = part.get("file", {})
        if not isinstance(file_value, dict):
            file_value = {}
        source = {"type": "file_id", "file_id": file_value.get("file_id")}
        if file_value.get("file_data") is not None:
            source = {
                "type": "base64",
                "data": file_value.get("file_data"),
            }
        block = {
            "type": "file",
            "source": source,
        }
        _put_optional(block, "filename", file_value.get("filename"))
        native = {
            key: value for key, value in part.items()
            if key not in ["type", "file"]
        }
        file_fields = {
            key: value for key, value in file_value.items()
            if key not in ["file_id", "file_data", "filename"]
        }
        if file_fields:
            native["file_fields"] = file_fields
        if native:
            _protocol_data(block, OPENAI_CHAT, native)
        return block
    report_unknown(OPENAI_CHAT, "message content block", part)
    return _native_content(OPENAI_CHAT, part)


def _chat_content_to_blocks(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [] if content == "" else [text_block(content)]
    if isinstance(content, list):
        return [_chat_content_block(part) for part in content]
    report_unknown(OPENAI_CHAT, "message content", content)
    return [_native_content(OPENAI_CHAT, content)]


def _chat_tool_call(tool_call, legacy=False):
    function = (
        tool_call if legacy else tool_call.get("function", {})
    )
    if not isinstance(function, dict):
        function = {}
    call_id = None if legacy else tool_call.get("id")
    if not call_id:
        call_id = "loki_legacy_" + uuid.uuid4().hex
    raw_arguments = function.get("arguments", "{}")
    _, parse_error = _parse_json_arguments(raw_arguments)
    protocol_fields = {
        "id": None if legacy else tool_call.get("id"),
        "wire_type": (
            "function" if legacy else tool_call.get("type", "function")),
        "legacy": bool(legacy),
    }
    extras = {
        key: value for key, value in tool_call.items()
        if key not in (
            ["name", "arguments"] if legacy
            else ["id", "type", "function"])
    }
    function_extras = {
        key: value for key, value in function.items()
        if key not in ["name", "arguments"]
    }
    if extras:
        protocol_fields["fields"] = extras
    if function_extras:
        protocol_fields["function_fields"] = function_extras
    return tool_call_item(
        call_id,
        function.get("name"),
        raw_arguments=(
            raw_arguments if isinstance(raw_arguments, str)
            else _json_arguments(raw_arguments)),
        parse_error=parse_error,
        tool_kind=(
            "function" if legacy
            else tool_call.get("type", "function")),
        protocol_data={OPENAI_CHAT: protocol_fields},
    )


def openai_chat_message_to_items(message):
    if not isinstance(message, dict):
        report_unknown(OPENAI_CHAT, "message", message)
        return [_native_output(OPENAI_CHAT, message)]
    role = message.get("role")
    if role in ["system", "developer", "user", "assistant"]:
        item = message_item(
            role, _chat_content_to_blocks(message.get("content")))
        native = {
            "content_present": "content" in message,
            "content_form": (
                "null" if message.get("content") is None
                else "array" if isinstance(message.get("content"), list)
                else "string"),
        }
        if isinstance(message.get("refusal"), str):
            refusal_block = {
                "type": "refusal",
                "text": message["refusal"],
            }
            _protocol_data(
                refusal_block,
                OPENAI_CHAT,
                {"top_level_refusal": True},
            )
            item["content"].append(refusal_block)
        known = {
            "role", "content", "tool_calls", "function_call",
            "refusal", "name", "audio", "phase",
        } | set(OPENAI_CHAT_REASONING_FIELDS)
        for key in ["name", "audio", "phase"]:
            _put_optional(native, key, message.get(key))
        native_fields = {
            key: _copy(message[key])
            for key in OPENAI_CHAT_REASONING_FIELDS
            if key in message
        }
        extras = {
            key: value for key, value in message.items()
            if key not in known
        }
        if extras:
            report_unknown(OPENAI_CHAT, "message fields", extras)
            native_fields.update(extras)
        if native_fields:
            native["fields"] = native_fields
        _protocol_data(item, OPENAI_CHAT, native)
        items = [item]
        if role == "assistant":
            calls = message.get("tool_calls", [])
            if isinstance(calls, list):
                items.extend(_chat_tool_call(call) for call in calls)
            legacy = message.get("function_call")
            if isinstance(legacy, dict):
                items.append(_chat_tool_call(legacy, legacy=True))
        return items
    if role == "tool":
        return [tool_result_item(
            message.get("tool_call_id"),
            _chat_content_to_blocks(message.get("content")),
            name=message.get("name"),
            is_error=message.get("is_error", False),
        )]
    if role == "function":
        item = tool_result_item(
            message.get("name"),
            _chat_content_to_blocks(message.get("content")),
            name=message.get("name"),
            is_error=message.get("is_error", False),
        )
        _protocol_data(item, OPENAI_CHAT, {"legacy": True})
        return [item]
    report_unknown(OPENAI_CHAT, "message", message)
    return [_native_output(OPENAI_CHAT, message)]


def openai_chat_messages_to_items(messages):
    items = []
    for message in messages or []:
        items.extend(openai_chat_message_to_items(message))
    return items


def openai_chat_response_to_items(response):
    if not isinstance(response, dict):
        raise TranscriptFormatError("OpenAI Chat response must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TranscriptFormatError(
            "OpenAI Chat response has no choices")
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(
            first.get("message"), dict):
        raise TranscriptFormatError(
            "OpenAI Chat first choice has no message")
    unknown_envelope = {
        key: value for key, value in response.items()
        if key not in [
            "id", "object", "created", "model", "choices", "usage",
            "service_tier", "system_fingerprint", "cost",
            "_loki_stream_extensions"]
    }
    if unknown_envelope:
        report_unknown(OPENAI_CHAT, "response fields", unknown_envelope)
    unknown_choice = {
        key: value for key, value in first.items()
        if key not in ["index", "message", "finish_reason", "logprobs"]
    }
    if unknown_choice:
        report_unknown(OPENAI_CHAT, "choice fields", unknown_choice)
    if len(choices) > 1:
        report_unknown(OPENAI_CHAT, "additional choices", choices[1:])
    protocol_data = {}
    _put_optional(protocol_data, "choice_index", first.get("index"))
    _put_optional(protocol_data, "logprobs", first.get("logprobs"))
    _put_optional(protocol_data, "cost", response.get("cost"))
    return DecodedTurn(
        openai_chat_message_to_items(first["message"]),
        {
            "protocol": OPENAI_CHAT,
            "status": "completed",
            "stop_reason": first.get("finish_reason"),
            "usage": _copy(response.get("usage")),
            "protocol_data": (
                {OPENAI_CHAT: protocol_data} if protocol_data else None),
        },
    )


def _anthropic_text_block(block):
    result = text_block(
        block.get("text", ""),
        citations=block.get("citations"),
    )
    extras = {
        key: value for key, value in block.items()
        if key not in ["type", "text", "citations"]
    }
    if extras:
        _protocol_data(result, ANTHROPIC_MESSAGES, extras)
    return result


def _anthropic_call_item(block, execution="client"):
    call_id = block.get("id")
    input_value = block.get("input", {})
    raw = _json_arguments(input_value)
    native = {
        key: value for key, value in block.items()
        if key not in ["type", "id", "name", "input"]
    }
    native["native_type"] = block.get("type")
    native["id"] = call_id
    return tool_call_item(
        call_id,
        block.get("name"),
        raw_arguments=raw,
        execution=execution,
        protocol_data={ANTHROPIC_MESSAGES: native},
    )


def _flush_message(items, blocks):
    if blocks:
        items.append(message_item("assistant", list(blocks)))
        blocks.clear()


def anthropic_response_to_items(response):
    if (not isinstance(response, dict)
            or response.get("type") != "message"
            or response.get("role") != "assistant"):
        raise TranscriptFormatError(
            "Anthropic response is not an assistant message")
    content = response.get("content")
    if not isinstance(content, list):
        raise TranscriptFormatError(
            "Anthropic response content must be a list")
    items = []
    blocks = []
    result_types = {
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "tool_search_tool_result",
        "mcp_tool_result",
    }
    for block in content:
        if not isinstance(block, dict):
            report_unknown(ANTHROPIC_MESSAGES, "content block", block)
            blocks.append(_native_content(ANTHROPIC_MESSAGES, block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            blocks.append(_anthropic_text_block(block))
        elif block_type == "thinking":
            _flush_message(items, blocks)
            item = {
                "type": "anthropic_thinking",
                "thinking": block.get("thinking", ""),
                "signature": block.get("signature", ""),
            }
            extras = {
                key: value for key, value in block.items()
                if key not in ["type", "thinking", "signature"]
            }
            if extras:
                report_unknown(
                    ANTHROPIC_MESSAGES, "thinking fields", extras)
                _protocol_data(item, ANTHROPIC_MESSAGES, extras)
            items.append(item)
        elif block_type == "redacted_thinking":
            _flush_message(items, blocks)
            item = {
                "type": "anthropic_redacted_thinking",
                "data": _copy(block.get("data")),
            }
            extras = {
                key: value for key, value in block.items()
                if key not in ["type", "data"]
            }
            if extras:
                report_unknown(
                    ANTHROPIC_MESSAGES,
                    "redacted-thinking fields",
                    extras,
                )
                _protocol_data(item, ANTHROPIC_MESSAGES, extras)
            items.append(item)
        elif block_type == "tool_use":
            _flush_message(items, blocks)
            items.append(_anthropic_call_item(block))
        elif block_type in ["server_tool_use", "mcp_tool_use"]:
            _flush_message(items, blocks)
            items.append(_anthropic_call_item(
                block, execution="provider"))
        elif block_type in result_types:
            _flush_message(items, blocks)
            native = {
                key: value for key, value in block.items()
                if key not in ["type", "tool_use_id", "content"]
            }
            native["native_type"] = block_type
            items.append({
                "type": "provider_tool_result",
                "call_id": block.get("tool_use_id"),
                "content": _copy(block.get("content")),
                "protocol_data": {
                    ANTHROPIC_MESSAGES: native,
                },
            })
        elif block_type in [
                "image", "document", "search_result", "container_upload"]:
            if block_type in ["image", "document"]:
                source = block.get("source", {})
                if not isinstance(source, dict):
                    source = {}
                canonical = {
                    "type": block_type,
                    "source": _copy(source),
                }
                native = {
                    key: value for key, value in block.items()
                    if key not in ["type", "source"]
                }
                if native:
                    _protocol_data(
                        canonical, ANTHROPIC_MESSAGES, native)
            else:
                canonical = media_block(block_type, block)
            blocks.append(canonical)
        else:
            report_unknown(ANTHROPIC_MESSAGES, "content block", block)
            blocks.append(_native_content(ANTHROPIC_MESSAGES, block))
    _flush_message(items, blocks)
    unknown_envelope = {
        key: value for key, value in response.items()
        if key not in [
            "id", "type", "role", "content", "model", "stop_reason",
            "stop_sequence", "usage", "container", "context_management",
            "_loki_stream_extensions"]
    }
    if unknown_envelope:
        report_unknown(
            ANTHROPIC_MESSAGES, "response fields", unknown_envelope)
    protocol_data = {}
    _put_optional(protocol_data, "stop_sequence", response.get("stop_sequence"))
    return DecodedTurn(
        items,
        {
            "protocol": ANTHROPIC_MESSAGES,
            "status": "completed",
            "stop_reason": response.get("stop_reason"),
            "usage": _copy(response.get("usage")),
            "protocol_data": (
                {ANTHROPIC_MESSAGES: protocol_data}
                if protocol_data else None),
        },
    )


def _responses_content_block(block):
    if not isinstance(block, dict):
        report_unknown(OPENAI_RESPONSES, "message content block", block)
        return _native_content(OPENAI_RESPONSES, block)
    block_type = block.get("type")
    if block_type in ["input_text", "output_text", "text"]:
        result = text_block(
            block.get("text", ""),
            annotations=block.get("annotations"),
            citations=block.get("citations"),
            logprobs=block.get("logprobs"),
        )
        native = {
            key: value for key, value in block.items()
            if key not in [
                "type", "text", "annotations", "citations", "logprobs"]
        }
        native["wire_type"] = block_type
        _protocol_data(result, OPENAI_RESPONSES, native)
        return result
    if block_type == "refusal":
        result = {
            "type": "refusal",
            "text": block.get("refusal", block.get("text", "")),
        }
        native = {
            key: value for key, value in block.items()
            if key not in ["type", "refusal", "text"]
        }
        if native:
            _protocol_data(result, OPENAI_RESPONSES, native)
        return result
    if block_type in [
            "input_image", "output_image", "image",
            "input_file", "output_file", "file",
            "input_audio", "output_audio", "audio"]:
        kind = (
            "image" if "image" in block_type
            else "file" if "file" in block_type
            else "audio")
        if kind == "image":
            if block.get("file_id"):
                source = {
                    "type": "file_id",
                    "file_id": block.get("file_id"),
                }
            else:
                source = _image_source_from_url(
                    block.get("image_url", block.get("url")))
            result = {"type": "image", "source": source}
            _put_optional(result, "detail", block.get("detail"))
            known = [
                "type", "file_id", "image_url", "url", "detail"]
        elif kind == "file":
            if block.get("file_id"):
                source = {
                    "type": "file_id",
                    "file_id": block.get("file_id"),
                }
            elif block.get("file_url"):
                source = {
                    "type": "url",
                    "url": block.get("file_url"),
                }
            else:
                source = {
                    "type": "base64",
                    "data": block.get("file_data"),
                }
            result = {"type": "file", "source": source}
            _put_optional(result, "filename", block.get("filename"))
            known = [
                "type", "file_id", "file_url", "file_data", "filename"]
        else:
            audio = block.get("input_audio", block.get("audio", {}))
            if not isinstance(audio, dict):
                audio = {}
            result = {
                "type": "audio",
                "source": {
                    "type": "base64",
                    "data": audio.get("data"),
                    "format": audio.get("format"),
                },
            }
            known = ["type", "input_audio", "audio"]
        native = {
            key: value for key, value in block.items()
            if key not in known
        }
        native["wire_type"] = block_type
        if native:
            _protocol_data(result, OPENAI_RESPONSES, native)
        return result
    report_unknown(OPENAI_RESPONSES, "message content block", block)
    return _native_content(OPENAI_RESPONSES, block)


def _responses_message_item(item):
    result = message_item(
        item.get("role"),
        [_responses_content_block(block)
         for block in item.get("content", [])])
    native = {
        key: value for key, value in item.items()
        if key not in ["type", "role", "content"]
    }
    _protocol_data(result, OPENAI_RESPONSES, native)
    return result


def _responses_call(item):
    raw = item.get("arguments", "{}")
    _, parse_error = _parse_json_arguments(raw)
    native = {
        key: value for key, value in item.items()
        if key not in [
            "type", "call_id", "name", "arguments", "status"]
    }
    native["native_type"] = item.get("type")
    return tool_call_item(
        item.get("call_id"),
        item.get("name"),
        raw_arguments=(
            raw if isinstance(raw, str) else _json_arguments(raw)),
        parse_error=parse_error,
        status=item.get("status"),
        tool_kind=(
            "custom" if item.get("type") == "custom_tool_call"
            else "function"),
        protocol_data={OPENAI_RESPONSES: native},
    )


_RESPONSES_PROVIDER_TYPES = {
    "computer_call", "computer_call_output",
    "web_search_call", "file_search_call",
    "image_generation_call", "code_interpreter_call",
    "local_shell_call", "local_shell_call_output",
    "shell_call", "shell_call_output",
    "mcp_call", "mcp_list_tools", "mcp_approval_request",
    "mcp_approval_response", "apply_patch_call",
    "apply_patch_call_output", "tool_search_call",
    "tool_search_output", "program", "program_output",
}


def _responses_provider_operation(item):
    item_type = item.get("type")
    mappings = {
        "code_interpreter_call": (
            "code_interpreter", "code", "outputs"),
        "file_search_call": (
            "file_search", "queries", "results"),
        "image_generation_call": (
            "image_generation", None, "result"),
        "web_search_call": (
            "web_search", "action", None),
    }
    mapping = mappings.get(item_type)
    if mapping is None:
        return None
    name, input_field, output_field = mapping
    input_value = (
        {input_field: _copy(item.get(input_field))}
        if input_field is not None and input_field in item else {})
    output_value = (
        _copy(item.get(output_field))
        if output_field is not None else None)
    known = {"type", "id", "status"}
    if input_field is not None:
        known.add(input_field)
    if output_field is not None:
        known.add(output_field)
    native = {
        "native_type": item_type,
        "input_field": input_field,
        "output_field": output_field,
        "input_present": (
            input_field is not None and input_field in item),
        "output_present": (
            output_field is not None and output_field in item),
        "fields": {
            key: _copy(value) for key, value in item.items()
            if key not in known
        },
    }
    return provider_operation_item(
        item.get("id"),
        name,
        input_value,
        output_value,
        status=item.get("status"),
        protocol_data={OPENAI_RESPONSES: native},
    )


def openai_responses_response_to_items(response):
    if (not isinstance(response, dict)
            or response.get("object") != "response"
            or not isinstance(response.get("output"), list)):
        raise TranscriptFormatError(
            "OpenAI Responses response is not a response object")
    items = []
    for output in response["output"]:
        if not isinstance(output, dict):
            report_unknown(OPENAI_RESPONSES, "output item", output)
            items.append(_native_output(OPENAI_RESPONSES, output))
            continue
        item_type = output.get("type")
        if item_type == "message":
            items.append(_responses_message_item(output))
        elif item_type in ["function_call", "custom_tool_call"]:
            items.append(_responses_call(output))
        elif item_type == "reasoning":
            items.append({
                "type": "openai_reasoning",
                "value": _copy(output),
            })
        elif item_type in [
                "function_call_output", "custom_tool_call_output"]:
            items.append(_native_output(OPENAI_RESPONSES, output))
        elif item_type in _RESPONSES_PROVIDER_TYPES:
            operation = _responses_provider_operation(output)
            if operation is not None:
                items.append(operation)
            else:
                report_unknown(
                    OPENAI_RESPONSES,
                    "unsupported provider output",
                    output,
                )
                items.append({
                    "type": "provider_output",
                    "protocol": OPENAI_RESPONSES,
                    "value": _copy(output),
                })
        else:
            report_unknown(OPENAI_RESPONSES, "output item", output)
            items.append(_native_output(OPENAI_RESPONSES, output))
    unknown_envelope = {
        key: value for key, value in response.items()
        if key not in {
            "id", "object", "created_at", "completed_at", "status",
            "error", "incomplete_details", "instructions",
            "max_output_tokens", "max_tool_calls", "model", "output",
            "parallel_tool_calls", "previous_response_id", "prompt",
            "prompt_cache_key", "prompt_cache_retention",
            "prompt_cache_options", "reasoning", "safety_identifier",
            "service_tier", "store", "temperature", "text",
            "tool_choice", "tools", "top_logprobs", "top_p",
            "truncation", "usage", "user", "metadata", "conversation",
            "background", "_loki_stream_extensions",
        }
    }
    if unknown_envelope:
        report_unknown(
            OPENAI_RESPONSES, "response fields", unknown_envelope)
    status = response.get("status", "completed")
    protocol_data = {}
    for key in ["incomplete_details", "error"]:
        _put_optional(protocol_data, key, response.get(key))
    return DecodedTurn(
        items,
        {
            "protocol": OPENAI_RESPONSES,
            "status": status,
            "usage": _copy(response.get("usage")),
            "protocol_data": (
                {OPENAI_RESPONSES: protocol_data}
                if protocol_data else None),
        },
        complete=status == "completed",
    )


def _plain_content(blocks, target):
    parts = []
    for block in blocks or []:
        if not isinstance(block, dict):
            raise ProjectionError(f"{target} content block must be an object")
        block_type = block.get("type")
        if block_type in ["text", "input_text", "output_text"]:
            parts.append(str(block.get("text", "")))
        elif block_type == "refusal":
            parts.append(str(block.get("text", "")))
        elif block_type in [
                "anthropic_thinking", "anthropic_redacted_thinking",
                "native_content"]:
            continue
        else:
            raise ProjectionError(
                f"cannot project content block {block_type!r} to {target}")
    return "\n".join(part for part in parts if part)


def _portable_provider_value(value, protocol):
    opaque_fields = {
        ANTHROPIC_MESSAGES: {
            "encrypted_content", "encrypted_index", "signature",
        },
        OPENAI_RESPONSES: {
            "encrypted_content",
        },
    }.get(protocol, set())
    if isinstance(value, dict):
        return {
            key: _portable_provider_value(item, protocol)
            for key, item in value.items()
            if key not in opaque_fields
        }
    if isinstance(value, list):
        return [
            _portable_provider_value(item, protocol)
            for item in value
        ]
    return _copy(value)


def _item_source_protocol(item):
    protocol_data = item.get("protocol_data", {})
    if isinstance(protocol_data, dict):
        for protocol in [
                ANTHROPIC_MESSAGES, OPENAI_RESPONSES, OPENAI_CHAT]:
            if protocol in protocol_data:
                return protocol
    return None


def _provider_result_output(item):
    source_protocol = _item_source_protocol(item)
    content = _portable_provider_value(
        item.get("content", ""), source_protocol)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        try:
            return _plain_content(content, "provider tool result")
        except ProjectionError:
            pass
    return json.dumps(
        content, ensure_ascii=False, separators=(",", ":"),
        default=str)


def _provider_operation_output(item):
    output = _portable_provider_value(
        item.get("output"), _item_source_protocol(item))
    if isinstance(output, str):
        return output
    return json.dumps(
        output, ensure_ascii=False, separators=(",", ":"),
        default=str)


def _provider_operation_call(item):
    return tool_call_item(
        item.get("call_id"),
        item.get("name"),
        input_value=_portable_provider_value(
            item.get("input", {}), _item_source_protocol(item)),
        execution="provider",
    )


def _chat_block(block, same_protocol):
    block_type = block.get("type")
    native = _protocol_fields(block, OPENAI_CHAT)
    if same_protocol and isinstance(native.get("wire"), dict):
        return _copy(native["wire"])
    if block_type in ["text", "input_text", "output_text"]:
        part = {"type": "text", "text": block.get("text", "")}
        if same_protocol:
            for key in ["annotations", "citations", "logprobs"]:
                _put_optional(part, key, block.get(key))
            part.update({
                key: _copy(value) for key, value in native.items()
                if key != "wire"
            })
        return part
    if block_type == "refusal":
        result = {
            "type": "refusal",
            "refusal": block.get("text", ""),
        }
        if same_protocol:
            result.update(_copy(native))
        return result
    if block_type == "image":
        source = block.get("source", {})
        if source.get("type") == "url" and source.get("url"):
            url = source["url"]
        elif (source.get("type") == "base64"
              and source.get("media_type")
              and source.get("data")):
            url = (
                f"data:{source['media_type']};base64,"
                f"{source['data']}")
        else:
            raise ProjectionError(
                "OpenAI Chat cannot encode this image source")
        image = {"url": url}
        _put_optional(image, "detail", block.get("detail"))
        if same_protocol:
            image.update(_copy(native.get("image_fields", {})))
        result = {"type": "image_url", "image_url": image}
        if same_protocol:
            result.update({
                key: _copy(value) for key, value in native.items()
                if key != "image_fields"
            })
        return result
    if block_type == "audio":
        source = block.get("source", {})
        if (source.get("type") != "base64"
                or not source.get("data")
                or not source.get("format")):
            raise ProjectionError(
                "OpenAI Chat cannot encode this audio source")
        audio = {
            "data": source["data"],
            "format": source["format"],
        }
        if same_protocol:
            audio.update(_copy(native.get("audio_fields", {})))
        result = {"type": "input_audio", "input_audio": audio}
        if same_protocol:
            result.update({
                key: _copy(value) for key, value in native.items()
                if key != "audio_fields"
            })
        return result
    if block_type in ["file", "document"]:
        source = block.get("source", {})
        file_value = {}
        if source.get("type") == "file_id" and source.get("file_id"):
            file_value["file_id"] = source["file_id"]
        elif source.get("type") == "base64" and source.get("data"):
            file_value["file_data"] = source["data"]
        else:
            raise ProjectionError(
                "OpenAI Chat cannot encode this file source")
        _put_optional(file_value, "filename", block.get("filename"))
        if same_protocol:
            file_value.update(_copy(native.get("file_fields", {})))
        result = {"type": "file", "file": file_value}
        if same_protocol:
            result.update({
                key: _copy(value) for key, value in native.items()
                if key != "file_fields"
            })
        return result
    if (block_type == "native_content"
            and block.get("protocol") == OPENAI_CHAT
            and same_protocol):
        return _copy(block.get("value"))
    return None


def _chat_message_content(item, same_protocol):
    rendered = []
    for block in item.get("content", []):
        part = _chat_block(block, same_protocol)
        if part is not None:
            rendered.append(part)
    native = _protocol_fields(item, OPENAI_CHAT)
    form = native.get("content_form") if same_protocol else None
    if not rendered:
        if same_protocol and native.get("content_present") is False:
            return None, False
        return None if form == "null" else "", True
    if all(part.get("type") == "text" for part in rendered):
        text = "\n".join(part.get("text", "") for part in rendered)
        if form != "array":
            return text, True
    return rendered, True


def _chat_call(call, same_protocol):
    native = _protocol_fields(call, OPENAI_CHAT)
    function = {
        "name": call.get("name"),
        "arguments": call.get("arguments", "{}"),
    }
    if same_protocol:
        function.update(_copy(native.get("function_fields", {})))
    wire = {
        "id": (
            native.get("id") if same_protocol and native.get("id")
            else call.get("call_id")),
        "type": (
            native.get("wire_type", "function")
            if same_protocol else "function"),
        "function": function,
    }
    if same_protocol:
        wire.update(_copy(native.get("fields", {})))
    return wire


def _chat_foreign_provider_messages(event, call_info):
    messages = []
    text = []
    parts = []
    calls = []

    def flush_assistant():
        if not text and not parts and not calls:
            return
        if parts:
            content = (
                [{"type": "text", "text": value}
                 for value in text if value]
                + list(parts))
        else:
            content = "\n".join(text) if text else None
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = list(calls)
        messages.append(message)
        text.clear()
        parts.clear()
        calls.clear()

    for item in event.get("items", []):
        item_type = item.get("type")
        if item_type == "message":
            for block in item.get("content", []):
                if block.get("type") == "refusal":
                    text.append(str(block.get("text", "")))
                    continue
                rendered = _chat_block(block, False)
                if rendered is None:
                    continue
                if (rendered.get("type") == "text"
                        and set(rendered) <= {"type", "text"}):
                    text.append(rendered.get("text", ""))
                else:
                    parts.append(rendered)
        elif (item_type == "function_call"
              and item.get("tool_kind", "function") == "function"):
            call_info[item.get("call_id")] = item
            calls.append(_chat_call(item, False))
        elif item_type == "provider_tool_result":
            flush_assistant()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": _provider_result_output(item),
            })
        elif item_type == "provider_operation":
            flush_assistant()
            call = _provider_operation_call(item)
            call_info[item.get("call_id")] = call
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [_chat_call(call, False)],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": _provider_operation_output(item),
            })
    flush_assistant()
    return messages


def _chat_response_messages(event, call_info, target=None):
    same_protocol = _native_replay_allowed(
        event, OPENAI_CHAT, target)
    if (not same_protocol
            and any(item.get("type") in [
                        "provider_tool_result", "provider_operation"]
                    for item in event.get("items", []))):
        return _chat_foreign_provider_messages(event, call_info)
    text_parts = []
    content_parts = []
    ordered_array_content = []
    refusals = []
    calls = []
    legacy = None
    raw_messages = []
    provider_results = []
    message_native = {}
    for item in event.get("items", []):
        item_type = item.get("type")
        if item_type == "message":
            if not message_native:
                message_native = _protocol_fields(
                    item, OPENAI_CHAT)
            for block in item.get("content", []):
                if block.get("type") == "refusal":
                    block_native = _protocol_fields(
                        block, OPENAI_CHAT)
                    if (same_protocol
                            and message_native.get("content_form") == "array"
                            and not block_native.get(
                                "top_level_refusal")):
                        ordered_array_content.append(
                            _chat_block(block, True))
                    else:
                        refusals.append(str(block.get("text", "")))
                    continue
                rendered = _chat_block(block, same_protocol)
                if rendered is None:
                    continue
                if (same_protocol
                        and message_native.get("content_form") == "array"):
                    ordered_array_content.append(rendered)
                if (rendered.get("type") == "text"
                        and set(rendered) <= {"type", "text"}):
                    text_parts.append(rendered.get("text", ""))
                else:
                    content_parts.append(rendered)
        elif item_type == "function_call":
            call_info[item.get("call_id")] = item
            native = _protocol_fields(item, OPENAI_CHAT)
            rendered = _chat_call(item, same_protocol)
            if same_protocol and native.get("legacy"):
                legacy = rendered["function"]
            elif (same_protocol
                  or item.get("tool_kind", "function") == "function"):
                calls.append(rendered)
        elif item_type == "provider_tool_result":
            provider_results.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": _provider_result_output(item),
            })
        elif (item_type == "native_output"
              and item.get("protocol") == OPENAI_CHAT
              and same_protocol):
            value = item.get("value")
            if isinstance(value, dict) and value.get("role"):
                raw_messages.append(_copy(value))
    if (not text_parts and not content_parts and not refusals
            and not calls and legacy is None):
        return raw_messages + provider_results
    content = None
    if ordered_array_content:
        content = ordered_array_content
    elif text_parts or content_parts:
        if content_parts:
            content = (
                [{"type": "text", "text": text}
                 for text in text_parts if text]
                + content_parts)
        else:
            content = "\n".join(text_parts)
    if (same_protocol and message_native.get("content_form") == "array"
            and isinstance(content, str)):
        content = [{"type": "text", "text": content}]
    message = {"role": "assistant"}
    if not (same_protocol
            and message_native.get("content_present") is False):
        message["content"] = content
    if refusals:
        if same_protocol:
            message["refusal"] = "\n".join(refusals)
        elif content is None:
            message["content"] = "\n".join(refusals)
        elif isinstance(message.get("content"), str):
            message["content"] += "\n" + "\n".join(refusals)
    if calls:
        message["tool_calls"] = calls
    if legacy is not None:
        message["function_call"] = legacy
    for item in event.get("items", []):
        if item.get("type") != "message":
            continue
        if same_protocol:
            for key in ["name", "audio", "phase"]:
                _put_optional(message, key, message_native.get(key))
            message.update(_copy(message_native.get("fields", {})))
        break
    return [message] + provider_results + raw_messages


def items_to_openai_chat_messages(events, target=None):
    messages = []
    call_info = {}
    index = 0
    while index < len(events):
        event = events[index]
        event_type = event.get("type")
        if event_type == "message":
            role = event.get("role")
            if role not in [
                    "system", "developer", "user", "assistant"]:
                raise ProjectionError(
                    f"cannot project message role {role!r} to OpenAI Chat")
            content, present = _chat_message_content(event, False)
            message = {"role": role}
            if present:
                message["content"] = content
            messages.append(message)
        elif event_type == "model_response":
            messages.extend(_chat_response_messages(
                event, call_info, target=target))
        elif event_type == "tool_result":
            call = call_info.get(event.get("call_id"), {})
            native_call = _protocol_fields(call, OPENAI_CHAT)
            content = _plain_content(event.get("content", []), "OpenAI Chat")
            if native_call.get("legacy"):
                messages.append({
                    "role": "function",
                    "name": event.get("name") or call.get("name"),
                    "content": content,
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": event.get("call_id"),
                    "content": content,
                })
        else:
            raise ProjectionError(
                f"cannot project event {event_type!r} to OpenAI Chat")
        index += 1
    return messages


def _anthropic_block(block, same_protocol):
    block_type = block.get("type")
    native = _protocol_fields(block, ANTHROPIC_MESSAGES)
    if same_protocol and isinstance(native.get("wire"), dict):
        return _copy(native["wire"])
    if block_type in ["text", "input_text", "output_text"]:
        result = {"type": "text", "text": block.get("text", "")}
        if same_protocol:
            _put_optional(result, "citations", block.get("citations"))
            result.update(_copy(native))
        return result
    if block_type == "refusal":
        return {"type": "text", "text": block.get("text", "")}
    if block_type in [
            "image", "document", "search_result", "container_upload"]:
        value = block.get("value")
        if isinstance(value, dict):
            if block_type != "container_upload" or same_protocol:
                return _copy(value)
            return None
        if block_type in ["image", "document"]:
            source = block.get("source")
            if isinstance(source, dict) and source.get("type"):
                result = {
                    "type": block_type,
                    "source": _copy(source),
                }
                if same_protocol:
                    result.update(_copy(native))
                return result
        raise ProjectionError(
            f"Anthropic cannot encode {block_type} content")
    if (block_type == "native_content"
            and block.get("protocol") == ANTHROPIC_MESSAGES
            and same_protocol):
        return _copy(block.get("value"))
    return None


def _anthropic_wire_call(call, same_protocol):
    native = _protocol_fields(call, ANTHROPIC_MESSAGES)
    if (same_protocol
            and call.get("execution") == "provider"
            and native.get("native_type") in [
                "server_tool_use", "mcp_tool_use"]):
        result = {
            "type": native.get("native_type"),
            "id": native.get("id") or call.get("call_id"),
            "name": call.get("name"),
            "input": tool_call_input(call),
        }
        result.update({
            key: _copy(value) for key, value in native.items()
            if key not in ["native_type", "id"]
        })
        return result
    result = {
        "type": "tool_use",
        "id": call.get("call_id"),
        "name": call.get("name"),
        "input": tool_call_input(call),
    }
    if same_protocol:
        result.update({
            key: _copy(value) for key, value in native.items()
            if key not in ["native_type", "id"]
        })
    return result


def _anthropic_response_messages(event, target=None):
    same_protocol = _native_replay_allowed(
        event, ANTHROPIC_MESSAGES, target)
    messages = []
    content = []

    def flush_assistant():
        if content:
            messages.append({
                "role": "assistant",
                "content": list(content),
            })
            content.clear()

    for item in event.get("items", []):
        item_type = item.get("type")
        if item_type == "message":
            for block in item.get("content", []):
                rendered = _anthropic_block(block, same_protocol)
                if rendered is not None:
                    content.append(rendered)
        elif item_type == "function_call":
            if (same_protocol
                    or item.get("tool_kind", "function") == "function"):
                content.append(_anthropic_wire_call(
                    item, same_protocol))
        elif item_type == "anthropic_thinking" and same_protocol:
            block = {
                "type": "thinking",
                "thinking": item.get("thinking", ""),
                "signature": item.get("signature", ""),
            }
            block.update(_copy(
                _protocol_fields(item, ANTHROPIC_MESSAGES)))
            content.append(block)
        elif item_type == "anthropic_redacted_thinking" and same_protocol:
            block = {
                "type": "redacted_thinking",
                "data": _copy(item.get("data")),
            }
            block.update(_copy(
                _protocol_fields(item, ANTHROPIC_MESSAGES)))
            content.append(block)
        elif item_type == "provider_tool_result" and same_protocol:
            native = _protocol_fields(item, ANTHROPIC_MESSAGES)
            result = {
                "type": native.get(
                    "native_type", "tool_result"),
                "tool_use_id": item.get("call_id"),
                "content": _copy(item.get("content")),
            }
            result.update({
                key: _copy(value) for key, value in native.items()
                if key != "native_type"
            })
            content.append(result)
        elif item_type == "provider_tool_result":
            flush_assistant()
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": item.get("call_id"),
                    "content": _provider_result_output(item),
                }],
            })
        elif item_type == "provider_operation":
            flush_assistant()
            call = _provider_operation_call(item)
            messages.append({
                "role": "assistant",
                "content": [_anthropic_wire_call(call, False)],
            })
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": item.get("call_id"),
                    "content": _provider_operation_output(item),
                }],
            })
        elif (item_type == "native_output"
              and item.get("protocol") == ANTHROPIC_MESSAGES
              and same_protocol):
            content.append(_copy(item.get("value")))
    flush_assistant()
    return messages


def _anthropic_direct_content(event):
    content = []
    for block in event.get("content", []):
        rendered = _anthropic_block(block, False)
        if rendered is not None:
            content.append(rendered)
    return content


def items_to_anthropic_parts(events, target=None):
    system = []
    messages = []
    index = 0
    while index < len(events):
        event = events[index]
        event_type = event.get("type")
        if event_type == "message":
            role = event.get("role")
            if role in ["system", "developer"]:
                system.extend(_anthropic_direct_content(event))
            elif role in ["user", "assistant"]:
                messages.append({
                    "role": role,
                    "content": _anthropic_direct_content(event),
                })
            else:
                raise ProjectionError(
                    f"cannot project message role {role!r} to Anthropic")
        elif event_type == "model_response":
            messages.extend(_anthropic_response_messages(
                event, target=target))
        elif event_type == "tool_result":
            results = []
            while (index < len(events)
                   and events[index].get("type") == "tool_result"):
                result = events[index]
                result_content = _anthropic_direct_content({
                    "content": result.get("content", [])
                })
                block = {
                    "type": "tool_result",
                    "tool_use_id": result.get("call_id"),
                    "content": (
                        result_content
                        if any(part.get("type") != "text"
                               for part in result_content)
                        else _plain_content(
                            result.get("content", []), "Anthropic")),
                }
                if result.get("is_error"):
                    block["is_error"] = True
                results.append(block)
                index += 1
            if (index < len(events)
                    and events[index].get("type") == "message"
                    and events[index].get("role") == "user"):
                results.extend(_anthropic_direct_content(events[index]))
            else:
                index -= 1
            messages.append({"role": "user", "content": results})
        else:
            raise ProjectionError(
                f"cannot project event {event_type!r} to Anthropic")
        index += 1
    return system, messages


def _responses_block(block, role, same_protocol):
    block_type = block.get("type")
    native = _protocol_fields(block, OPENAI_RESPONSES)
    if same_protocol and isinstance(native.get("wire"), dict):
        return _copy(native["wire"])
    if block_type in ["text", "input_text", "output_text"]:
        wire_type = (
            native.get("wire_type")
            if same_protocol and native.get("wire_type")
            else "output_text" if role == "assistant" else "input_text")
        result = {"type": wire_type, "text": block.get("text", "")}
        if role == "assistant" and same_protocol:
            _put_optional(result, "annotations", block.get("annotations"))
            _put_optional(result, "logprobs", block.get("logprobs"))
        if same_protocol:
            result.update({
                key: _copy(value) for key, value in native.items()
                if key != "wire_type"
            })
        if wire_type == "output_text":
            result.setdefault("annotations", [])
        return result
    if block_type == "refusal":
        result = {
            "type": "refusal",
            "refusal": block.get("text", ""),
        }
        if same_protocol:
            result.update(_copy(native))
        return result
    if block_type in ["image", "file", "audio"]:
        source = block.get("source", {})
        wire_type = (
            native.get("wire_type") if same_protocol
            else "input_image" if block_type == "image"
            else "input_file" if block_type == "file"
            else "input_audio")
        if block_type == "image":
            result = {"type": wire_type}
            if source.get("type") == "url" and source.get("url"):
                result["image_url"] = source["url"]
                _put_optional(result, "detail", block.get("detail"))
            elif (source.get("type") == "base64"
                  and source.get("media_type")
                  and source.get("data")):
                result["image_url"] = (
                    f"data:{source['media_type']};base64,"
                    f"{source['data']}")
            elif (source.get("type") == "file_id"
                  and source.get("file_id")):
                result["file_id"] = source["file_id"]
            else:
                raise ProjectionError(
                    "OpenAI Responses cannot encode this image source")
        elif block_type == "file":
            result = {"type": wire_type}
            if source.get("type") == "file_id" and source.get("file_id"):
                result["file_id"] = source["file_id"]
            elif source.get("type") == "url" and source.get("url"):
                result["file_url"] = source["url"]
            elif source.get("type") == "base64" and source.get("data"):
                result["file_data"] = source["data"]
            else:
                raise ProjectionError(
                    "OpenAI Responses cannot encode this file source")
            _put_optional(result, "filename", block.get("filename"))
        else:
            if (source.get("type") != "base64"
                    or not source.get("data")
                    or not source.get("format")):
                raise ProjectionError(
                    "OpenAI Responses cannot encode this audio source")
            result = {
                "type": wire_type,
                "input_audio": {
                    "data": source["data"],
                    "format": source["format"],
                },
            }
        if same_protocol:
            result.update({
                key: _copy(value) for key, value in native.items()
                if key != "wire_type"
            })
        return result
    if (block_type == "native_content"
            and block.get("protocol") == OPENAI_RESPONSES
            and same_protocol):
        return _copy(block.get("value"))
    return None


def _responses_wire_message(item, same_protocol):
    role = item.get("role")
    result = {
        "type": "message",
        "role": role,
        "content": [
            rendered for rendered in (
                _responses_block(block, role, same_protocol)
                for block in item.get("content", []))
            if rendered is not None
        ],
    }
    if same_protocol:
        result.update(_copy(_protocol_fields(item, OPENAI_RESPONSES)))
    return result


def _responses_call_wire(item, same_protocol):
    native = _protocol_fields(item, OPENAI_RESPONSES)
    native_type = (
        native.get("native_type") if same_protocol else None)
    result = {
        "type": (
            native_type if native_type in [
                "function_call", "custom_tool_call"]
            else "function_call"),
        "call_id": item.get("call_id"),
        "name": item.get("name"),
    }
    if result["type"] == "custom_tool_call":
        result["input"] = item.get("arguments", "")
    else:
        result["arguments"] = item.get("arguments", "{}")
    if same_protocol:
        _put_optional(result, "status", item.get("status"))
        result.update({
            key: _copy(value) for key, value in native.items()
            if key != "native_type"
        })
    return result


def _responses_provider_operation_wire(item, same_protocol):
    if not same_protocol:
        call = _provider_operation_call(item)
        return [
            _responses_call_wire(call, False),
            {
                "type": "function_call_output",
                "call_id": item.get("call_id"),
                "output": _provider_operation_output(item),
            },
        ]

    native = _protocol_fields(item, OPENAI_RESPONSES)
    result = {
        "type": native.get("native_type"),
        "id": item.get("call_id"),
    }
    _put_optional(result, "status", item.get("status"))
    result.update(_copy(native.get("fields", {})))
    input_field = native.get("input_field")
    if input_field is not None and native.get("input_present"):
        input_value = item.get("input", {})
        result[input_field] = _copy(
            input_value.get(input_field)
            if isinstance(input_value, dict) else input_value)
    output_field = native.get("output_field")
    if output_field is not None and native.get("output_present"):
        result[output_field] = _copy(item.get("output"))
    return [result]


def _responses_response_items(event, target=None):
    same_protocol = _native_replay_allowed(
        event, OPENAI_RESPONSES, target)
    result = []
    for item in event.get("items", []):
        item_type = item.get("type")
        if item_type == "message":
            result.append(_responses_wire_message(
                item, same_protocol))
        elif item_type == "function_call":
            if (same_protocol
                    or item.get("tool_kind", "function") == "function"):
                result.append(_responses_call_wire(item, same_protocol))
        elif item_type == "provider_tool_result":
            result.append({
                "type": "function_call_output",
                "call_id": item.get("call_id"),
                "output": _provider_result_output(item),
            })
        elif item_type == "provider_operation":
            result.extend(_responses_provider_operation_wire(
                item, same_protocol))
        elif item_type == "openai_reasoning" and same_protocol:
            result.append(_copy(item.get("value")))
        elif (item_type in ["provider_output", "native_output"]
              and item.get("protocol") == OPENAI_RESPONSES
              and same_protocol):
            result.append(_copy(item.get("value")))
    return result


def _responses_tool_output(blocks):
    text = []
    rendered = []
    for block in blocks or []:
        if block.get("type") in [
                "text", "input_text", "output_text", "refusal"]:
            text.append(str(
                block.get("text", block.get("refusal", ""))))
            continue
        value = _responses_block(block, "user", False)
        if value is not None:
            rendered.append(value)
    if not rendered:
        return "\n".join(part for part in text if part)
    rendered = (
        [{"type": "input_text", "text": part}
         for part in text if part]
        + rendered)
    return rendered


def items_to_openai_responses_parts(events, target=None):
    instructions = []
    input_items = []
    instruction_prefix_open = True
    for event in events:
        event_type = event.get("type")
        if event_type == "message":
            role = event.get("role")
            if role in ["system", "developer"]:
                if instruction_prefix_open:
                    text = _plain_content(
                        event.get("content", []), "OpenAI Responses")
                    if text:
                        instructions.append(text)
                else:
                    # Responses accepts system/developer message items in
                    # input. Keeping a late operator instruction here
                    # preserves session order and the already-cacheable prefix.
                    input_items.append(_responses_wire_message(
                        event, False))
            elif role in ["user", "assistant"]:
                instruction_prefix_open = False
                input_items.append(_responses_wire_message(
                    event, False))
            else:
                raise ProjectionError(
                    f"cannot project message role {role!r} "
                    "to OpenAI Responses")
        elif event_type == "model_response":
            instruction_prefix_open = False
            input_items.extend(_responses_response_items(
                event, target=target))
        elif event_type == "tool_result":
            instruction_prefix_open = False
            input_items.append({
                "type": (
                    "custom_tool_call_output"
                    if event.get("tool_kind") == "custom"
                    else "function_call_output"),
                "call_id": event.get("call_id"),
                "output": _responses_tool_output(
                    event.get("content", [])),
            })
        else:
            raise ProjectionError(
                f"cannot project event {event_type!r} "
                "to OpenAI Responses")
    return "\n\n".join(instructions), input_items


def openai_tools_to_anthropic_tools(tools):
    if tools is None:
        return None
    result = []
    for tool in tools:
        if tool.get("type") != "function":
            result.append(_copy(tool))
            continue
        function = tool.get("function") or {}
        item = {
            "name": function.get("name"),
            "description": function.get("description", ""),
            "input_schema": _copy(function.get(
                "parameters", {"type": "object", "properties": {}})),
        }
        _put_optional(item, "strict", function.get("strict"))
        result.append(item)
    return result


def openai_tools_to_responses_tools(tools):
    if tools is None:
        return None
    result = []
    for tool in tools:
        if tool.get("type") != "function":
            result.append(_copy(tool))
            continue
        function = tool.get("function") or {}
        item = {
            "type": "function",
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parameters": _copy(function.get(
                "parameters", {"type": "object", "properties": {}})),
        }
        _put_optional(item, "strict", function.get("strict"))
        result.append(item)
    return result


def openai_tools_to_responses_lite_tools(tools):
    """Group ordinary function tools into Responses-Lite's namespace form."""
    responses_tools = openai_tools_to_responses_tools(tools)
    if not responses_tools:
        return responses_tools

    result = []
    functions = []
    functions_index = None
    for tool in responses_tools:
        if tool.get("type") == "function":
            if functions_index is None:
                functions_index = len(result)
            functions.append(tool)
        else:
            result.append(tool)
    if functions:
        # Responses-Lite makes these functions available directly to the
        # model; "functions" is the protocol's conventional default
        # namespace, not part of Loki's internal tool names.
        result.insert(functions_index, {
            "type": "namespace",
            "name": "functions",
            "description": "Tools in the functions namespace.",
            "tools": functions,
        })
    return result
