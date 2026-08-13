import copy
import json
import urllib.parse
from dataclasses import dataclass

from . import formats


OPENAI_CHAT = "openai_chat"
ANTHROPIC_MESSAGES = "anthropic_messages"
OPENAI_RESPONSES = "openai_responses"
DUMMY = "dummy"
AUTO = "auto"

# Wire protocols this client can actually speak. Single source of truth:
# consumers reference this instead of re-listing the constants, so adding a
# protocol here is all that is needed to teach the rest of the code about it.
SUPPORTED_PROTOCOLS = [OPENAI_CHAT, ANTHROPIC_MESSAGES, OPENAI_RESPONSES]


class ProtocolError(ValueError):
    pass


class ProviderDetectionError(ProtocolError):
    pass


class UnsupportedProtocolError(ProtocolError):
    pass


class StreamProtocolError(ProtocolError):
    pass


@dataclass
class Provider:
    kind: str
    input_url: str
    chat_url: str
    models_url: str | None
    model_urls: list[str]
    headers: dict
    max_tokens: int

    def chat_payload(self, items, tools, model):
        if self.kind == OPENAI_CHAT:
            payload = {
                "model": model,
                "messages": formats.items_to_openai_chat_messages(items),
            }
            if tools is not None:
                payload["tools"] = tools
            return payload
        if self.kind == ANTHROPIC_MESSAGES:
            system, messages = formats.items_to_anthropic_parts(items)
            payload = {
                "model": model,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            if system:
                payload["system"] = system
            anthropic_tools = formats.openai_tools_to_anthropic_tools(tools)
            if anthropic_tools:
                payload["tools"] = anthropic_tools
            return payload
        if self.kind == OPENAI_RESPONSES:
            instructions, input_items = formats.items_to_openai_responses_parts(items)
            payload = {
                "model": model,
                "input": input_items,
                "max_output_tokens": self.max_tokens,
            }
            if instructions:
                payload["instructions"] = instructions
            responses_tools = formats.openai_tools_to_responses_tools(tools)
            if responses_tools:
                payload["tools"] = responses_tools
            return payload
        if self.kind == DUMMY:
            # Never sent over the wire; async_chat_completion short-circuits on
            # this kind before any HTTP happens.
            return {}
        raise ProtocolError(f"unknown protocol {self.kind!r}")

    def streaming_chat_payload(self, items, tools, model):
        payload = self.chat_payload(items, tools, model)
        if self.kind != DUMMY:
            payload["stream"] = True
        return payload

    def stream_accumulator(self, on_text_delta=None):
        callback = on_text_delta or (lambda text: None)
        if self.kind == OPENAI_CHAT:
            return OpenAIChatStreamAccumulator(callback)
        if self.kind == ANTHROPIC_MESSAGES:
            return AnthropicMessagesStreamAccumulator(callback)
        if self.kind == OPENAI_RESPONSES:
            return OpenAIResponsesStreamAccumulator(callback)
        raise StreamProtocolError(
            f"streaming is not implemented for protocol {self.kind!r}")

    def parse_chat_response(self, response):
        if self.kind == OPENAI_CHAT:
            return formats.openai_chat_response_to_items(response)
        if self.kind == ANTHROPIC_MESSAGES:
            return formats.anthropic_response_to_items(response)
        if self.kind == OPENAI_RESPONSES:
            return formats.openai_responses_response_to_items(response)
        if self.kind == DUMMY:
            return formats.openai_chat_response_to_items(response)
        raise ProtocolError(f"unknown protocol {self.kind!r}")

    def parse_model_ids(self, response):
        if not isinstance(response, dict):
            return []
        data = response.get("data", [])
        if not isinstance(data, list):
            return []
        return [item["id"] for item in data
                if isinstance(item, dict) and isinstance(item.get("id"), str)]


class OpenAIChatStreamAccumulator:
    def __init__(self, on_text_delta):
        self.on_text_delta = on_text_delta
        self.response = {}
        self.choice = {
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }
        self.tool_calls = {}
        self.complete = False

    def feed(self, event):
        if event.data == "[DONE]":
            self.complete = True
            return
        try:
            chunk = json.loads(event.data)
        except json.JSONDecodeError as e:
            raise StreamProtocolError(
                f"OpenAI Chat stream event is not JSON: {e}")
        if not isinstance(chunk, dict):
            raise StreamProtocolError(
                "OpenAI Chat stream event must be an object")
        if chunk.get("error"):
            raise StreamProtocolError(_stream_error_text(chunk["error"]))
        for key in [
                "id", "object", "created", "model", "system_fingerprint",
                "service_tier", "usage"]:
            if key in chunk and chunk[key] is not None:
                self.response[key] = copy.deepcopy(chunk[key])
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            if choice.get("finish_reason") is not None:
                self.choice["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            role = delta.get("role")
            if isinstance(role, str):
                self.choice["message"]["role"] = role
            content = delta.get("content")
            if isinstance(content, str):
                self.choice["message"]["content"] += content
                if content:
                    self.on_text_delta(content)
            self._accumulate_tool_calls(delta.get("tool_calls"))

    def _accumulate_tool_calls(self, calls):
        if not isinstance(calls, list):
            return
        for position, delta in enumerate(calls):
            if not isinstance(delta, dict):
                continue
            index = delta.get("index", position)
            if not isinstance(index, int) or isinstance(index, bool):
                raise StreamProtocolError(
                    "OpenAI Chat tool call index must be an integer")
            call = self.tool_calls.setdefault(index, {
                "id": None,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if delta.get("id") is not None:
                call["id"] = delta["id"]
            if delta.get("type") is not None:
                call["type"] = delta["type"]
            function = delta.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments")
                if isinstance(name, str):
                    call["function"]["name"] += name
                if isinstance(arguments, str):
                    call["function"]["arguments"] += arguments

    def finish(self):
        if not self.complete:
            raise StreamProtocolError(
                "OpenAI Chat stream ended before data: [DONE]")
        message = self.choice["message"]
        if self.tool_calls:
            message["tool_calls"] = [
                self.tool_calls[index] for index in sorted(self.tool_calls)]
        self.response.setdefault("object", "chat.completion")
        self.response["choices"] = [self.choice]
        return self.response


class AnthropicMessagesStreamAccumulator:
    def __init__(self, on_text_delta):
        self.on_text_delta = on_text_delta
        self.message = None
        self.blocks = {}
        self.tool_json = {}
        self.complete = False

    def feed(self, event):
        try:
            data = json.loads(event.data)
        except json.JSONDecodeError as e:
            raise StreamProtocolError(
                f"Anthropic stream event is not JSON: {e}")
        if not isinstance(data, dict):
            raise StreamProtocolError(
                "Anthropic stream event must be an object")
        event_type = data.get("type") or event.event
        if event_type == "error":
            raise StreamProtocolError(
                _stream_error_text(data.get("error") or data))
        if event_type == "message_start":
            message = data.get("message")
            if not isinstance(message, dict):
                raise StreamProtocolError(
                    "Anthropic message_start is missing message")
            self.message = copy.deepcopy(message)
            self.message["content"] = []
        elif event_type == "content_block_start":
            index = _stream_index(data, "Anthropic content block")
            block = data.get("content_block")
            if not isinstance(block, dict):
                raise StreamProtocolError(
                    "Anthropic content_block_start is missing content_block")
            self.blocks[index] = copy.deepcopy(block)
            if block.get("type") == "tool_use":
                self.tool_json[index] = ""
        elif event_type == "content_block_delta":
            self._feed_delta(data)
        elif event_type == "content_block_stop":
            index = _stream_index(data, "Anthropic content block")
            self._finish_block(index)
        elif event_type == "message_delta":
            if self.message is None:
                raise StreamProtocolError(
                    "Anthropic message_delta preceded message_start")
            delta = data.get("delta")
            if isinstance(delta, dict):
                self.message.update(copy.deepcopy(delta))
            if data.get("usage") is not None:
                usage = self.message.setdefault("usage", {})
                if isinstance(usage, dict) and isinstance(
                        data["usage"], dict):
                    usage.update(copy.deepcopy(data["usage"]))
                else:
                    self.message["usage"] = copy.deepcopy(data["usage"])
        elif event_type == "message_stop":
            self.complete = True
        elif event_type == "ping":
            return
        # Unknown event types are allowed by Anthropic's versioning policy.

    def _feed_delta(self, data):
        index = _stream_index(data, "Anthropic content block")
        block = self.blocks.get(index)
        if block is None:
            raise StreamProtocolError(
                "Anthropic content delta preceded content block start")
        delta = data.get("delta")
        if not isinstance(delta, dict):
            return
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text")
            if isinstance(text, str):
                block["text"] = str(block.get("text", "")) + text
                if text:
                    self.on_text_delta(text)
        elif delta_type == "input_json_delta":
            partial = delta.get("partial_json")
            if isinstance(partial, str):
                self.tool_json[index] = self.tool_json.get(index, "") + partial
        elif delta_type == "thinking_delta":
            thinking = delta.get("thinking")
            if isinstance(thinking, str):
                block["thinking"] = (
                    str(block.get("thinking", "")) + thinking)
        elif delta_type == "signature_delta":
            signature = delta.get("signature")
            if isinstance(signature, str):
                block["signature"] = (
                    str(block.get("signature", "")) + signature)
        elif delta_type == "citations_delta":
            citation = delta.get("citation")
            if citation is not None:
                block.setdefault("citations", []).append(
                    copy.deepcopy(citation))

    def _finish_block(self, index):
        block = self.blocks.get(index)
        if block is None:
            raise StreamProtocolError(
                "Anthropic content block stop preceded start")
        if block.get("type") == "tool_use":
            raw = self.tool_json.get(index, "")
            try:
                block["input"] = json.loads(raw or "{}")
            except json.JSONDecodeError as e:
                raise StreamProtocolError(
                    f"Anthropic tool input is not valid JSON: {e}")

    def finish(self):
        if self.message is None:
            raise StreamProtocolError(
                "Anthropic stream ended before message_start")
        if not self.complete:
            raise StreamProtocolError(
                "Anthropic stream ended before message_stop")
        for index in list(self.blocks):
            if self.blocks[index].get("type") == "tool_use":
                self._finish_block(index)
        self.message["content"] = [
            self.blocks[index] for index in sorted(self.blocks)]
        self.message["type"] = "message"
        self.message["role"] = "assistant"
        return self.message


class OpenAIResponsesStreamAccumulator:
    def __init__(self, on_text_delta):
        self.on_text_delta = on_text_delta
        self.completed_response = None
        self.failed = None

    def feed(self, event):
        try:
            data = json.loads(event.data)
        except json.JSONDecodeError as e:
            raise StreamProtocolError(
                f"OpenAI Responses stream event is not JSON: {e}")
        if not isinstance(data, dict):
            raise StreamProtocolError(
                "OpenAI Responses stream event must be an object")
        event_type = data.get("type") or event.event
        if event_type == "response.output_text.delta":
            delta = data.get("delta")
            if isinstance(delta, str) and delta:
                self.on_text_delta(delta)
        elif event_type == "response.completed":
            response = data.get("response")
            if not isinstance(response, dict):
                raise StreamProtocolError(
                    "response.completed is missing its response")
            self.completed_response = copy.deepcopy(response)
        elif event_type in (
                "response.failed", "response.incomplete", "response.cancelled",
                "error"):
            self.failed = copy.deepcopy(data)

    def finish(self):
        if self.failed is not None:
            raise StreamProtocolError(_stream_error_text(self.failed))
        if self.completed_response is None:
            raise StreamProtocolError(
                "OpenAI Responses stream ended before response.completed")
        return self.completed_response


def _stream_index(data, context):
    index = data.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise StreamProtocolError(f"{context} index must be a nonnegative integer")
    return index


def _stream_error_text(value):
    if isinstance(value, dict):
        nested = value.get("error")
        if nested is not None and nested is not value:
            return _stream_error_text(nested)
        message = value.get("message")
        error_type = value.get("type") or value.get("code")
        if message and error_type:
            return f"{error_type}: {message}"
        if message:
            return str(message)
    return json.dumps(value, ensure_ascii=False, default=str)


def normalize_protocol(value):
    value = (value or AUTO).strip().lower().replace("-", "_")
    aliases = {
        "openai": OPENAI_CHAT,
        "chat": OPENAI_CHAT,
        "openai_chat_completions": OPENAI_CHAT,
        "anthropic": ANTHROPIC_MESSAGES,
        "claude": ANTHROPIC_MESSAGES,
        "messages": ANTHROPIC_MESSAGES,
        "responses": OPENAI_RESPONSES,
        "openai_new": OPENAI_RESPONSES,
    }
    return aliases.get(value, value)


def detect_protocol_from_url(url):
    path = urllib.parse.urlparse(url).path.rstrip("/")
    # Infer protocol only from a configured chat endpoint path. A base URL
    # without a recognized endpoint path needs an explicit provider.
    if path.endswith("/v1/chat/completions") or path.endswith("/chat/completions"):
        return OPENAI_CHAT
    if path.endswith("/v1/messages") or path.endswith("/messages"):
        return ANTHROPIC_MESSAGES
    if path.endswith("/v1/responses") or path.endswith("/responses"):
        return OPENAI_RESPONSES
    return None


# Maps models.dev provider ``npm`` values to the wire protocol they speak.
# Only packages whose name unambiguously names a protocol belong here.
# Vendor-specific SDKs (@ai-sdk/togetherai, @ai-sdk/deepinfra,
# @openrouter/ai-sdk-provider, @aihubmix/ai-sdk-provider, ...) are excluded
# because they wrap endpoints that may speak any protocol; those still need
# URL detection or an explicit override.
NPM_PROTOCOL = {
    "@ai-sdk/openai-compatible": OPENAI_CHAT,
    "@ai-sdk/anthropic":         ANTHROPIC_MESSAGES,
    "@ai-sdk/openai":            OPENAI_RESPONSES,
}


def detect_protocol_from_npm(npm):
    """Infer wire protocol from a models.dev provider's npm package.

    Returns one of SUPPORTED_PROTOCOLS for the three packages whose names
    name a protocol directly, or None for vendor-specific SDKs and unknown
    packages.
    """
    return NPM_PROTOCOL.get(npm) if npm else None


def detect_protocol_from_response(response):
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and isinstance(first.get("message"), dict):
            return OPENAI_CHAT
    if response.get("type") == "message" and response.get("role") == "assistant":
        if isinstance(response.get("content"), list):
            return ANTHROPIC_MESSAGES
    if response.get("object") == "response" or isinstance(response.get("output"), list):
        return OPENAI_RESPONSES
    return None


def resolve_protocol(url, override=AUTO):
    requested = normalize_protocol(override)
    if requested != AUTO:
        if requested not in [OPENAI_CHAT, ANTHROPIC_MESSAGES, OPENAI_RESPONSES, DUMMY]:
            raise ProviderDetectionError(f"unknown provider {override!r}")
        return requested
    detected = detect_protocol_from_url(url)
    if detected:
        return detected
    raise ProviderDetectionError(
        "cannot infer chat protocol from URL; set LOKI_PROVIDER=openai_chat, "
        "LOKI_PROVIDER=anthropic_messages, LOKI_PROVIDER=openai_responses, "
        "or LOKI_PROVIDER=dummy")


def _replace_path(parsed, path):
    return urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        path,
        "",
        parsed.query if path == parsed.path else "",
        "",
    ))


def _append_path(parsed, suffix):
    # Only used after endpoint detection failed and the caller supplied an
    # explicit protocol. At that point the path is treated as a provider base
    # prefix, not as a complete chat endpoint.
    base_path = parsed.path.rstrip("/")
    if not base_path:
        base_path = ""
    return _replace_path(parsed, base_path + suffix)


def _strip_suffix_path(path, suffix):
    clean = path.rstrip("/")
    if clean.endswith(suffix):
        return clean[:-len(suffix)] or "/"
    return None


def _v1_root(parsed):
    path = parsed.path.rstrip("/")
    for suffix in ["/chat/completions", "/messages", "/responses", "/models"]:
        if path.endswith("/v1" + suffix):
            return path[:-len(suffix)]
    if path.endswith("/v1"):
        return path
    return None


def endpoint_urls(input_url, protocol, models_url=None):
    parsed = urllib.parse.urlparse(input_url)
    if not parsed.scheme or not parsed.netloc:
        raise ProtocolError(f"unsupported URL {input_url!r}")
    v1_endpoint_path = {
        OPENAI_CHAT: "/chat/completions",
        ANTHROPIC_MESSAGES: "/messages",
        OPENAI_RESPONSES: "/responses",
    }.get(protocol)
    base_endpoint_path = {
        OPENAI_CHAT: "/chat/completions",
        ANTHROPIC_MESSAGES: "/v1/messages",
        OPENAI_RESPONSES: "/v1/responses",
    }.get(protocol)
    if v1_endpoint_path is None:
        raise ProtocolError(f"unknown protocol {protocol!r}")

    root = _v1_root(parsed)
    if root:
        # A URL ending at /v1, or at a known endpoint under /v1, is normalized
        # around that /v1 root. This keeps full endpoint and /v1 base inputs
        # equivalent for standard OpenAI/Anthropic-compatible layouts.
        chat_url = _replace_path(parsed, root + v1_endpoint_path)
        default_models_url = _replace_path(parsed, root + "/models")
    else:
        detected = detect_protocol_from_url(input_url)
        if detected == protocol:
            # The input already names the concrete chat endpoint. Use it
            # literally; only derive the model-list URL from the endpoint path.
            chat_url = input_url
            clean_path = parsed.path.rstrip("/")
            for suffix in ["/chat/completions", "/messages", "/responses"]:
                root_path = _strip_suffix_path(clean_path, suffix)
                if root_path is not None:
                    default_models_url = _replace_path(parsed, root_path + "/models")
                    break
            else:
                default_models_url = None
        else:
            # With an explicit provider override, a non-endpoint URL is a
            # provider base. Anthropic-compatible bases append /v1/messages;
            # OpenAI Chat bases append /chat/completions.
            chat_url = _append_path(parsed, base_endpoint_path)
            default_models_url = _append_path(parsed, "/v1/models" if protocol != OPENAI_CHAT else "/models")
    return chat_url, models_url or default_models_url


def model_url_candidates(input_url, protocol, primary_models_url=None, explicit_models_url=None):
    if explicit_models_url:
        return [explicit_models_url]

    candidates = []
    if primary_models_url:
        candidates.append(primary_models_url)

    parsed = urllib.parse.urlparse(input_url)
    root = _v1_root(parsed)
    if root:
        candidate = _replace_path(parsed, root + "/models")
        if candidate not in candidates:
            candidates.append(candidate)
        return candidates

    if parsed.path.rstrip("/") and not detect_protocol_from_url(input_url):
        # Some compatibility bases include a path prefix for chat while their
        # model-list endpoint remains at the API root. Try that non-mutating
        # endpoint after the protocol-derived candidate.
        candidate = _replace_path(parsed, "/models")
        if candidate not in candidates:
            candidates.append(candidate)

    return candidates


def build_headers(protocol, api_key, anthropic_version="2023-06-01",
                  auth_header=None, user_agent="TinyAgent/1.0"):
    headers = {
        "Content-Type": "application/json",
        "User-Agent": user_agent,
    }
    if protocol == ANTHROPIC_MESSAGES:
        # This identifies the Anthropic wire format; it is required protocol
        # metadata, not an authentication credential.
        headers["anthropic-version"] = anthropic_version
    if not api_key:
        return headers
    if auth_header:
        # Compatibility gateways sometimes use a nonstandard auth header. When
        # configured, honor it exactly instead of also adding Bearer/x-api-key.
        headers[auth_header] = api_key
        return headers
    if protocol == ANTHROPIC_MESSAGES:
        headers["x-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def make_provider(input_url, provider=AUTO, api_key="", models_url=None,
                  max_tokens=4096, anthropic_version="2023-06-01",
                  auth_header=None):
    protocol = resolve_protocol(input_url, provider)
    if protocol == DUMMY:
        # No-op provider for testing: no real endpoint or URL structure.
        return Provider(
            kind=DUMMY,
            input_url=input_url,
            chat_url=input_url,
            models_url=None,
            model_urls=[],
            headers={},
            max_tokens=max_tokens,
        )
    chat_url, resolved_models_url = endpoint_urls(input_url, protocol, models_url=models_url)
    model_urls = model_url_candidates(input_url, protocol, resolved_models_url,
                                      explicit_models_url=models_url)
    headers = build_headers(protocol, api_key, anthropic_version=anthropic_version,
                            auth_header=auth_header)
    return Provider(
        kind=protocol,
        input_url=input_url,
        chat_url=chat_url,
        models_url=resolved_models_url,
        model_urls=model_urls,
        headers=headers,
        max_tokens=max_tokens,
    )


def json_body(payload):
    if payload is None:
        return b""
    return json.dumps(payload).encode("utf-8")


def copy_headers(headers):
    return copy.deepcopy(headers or {})
