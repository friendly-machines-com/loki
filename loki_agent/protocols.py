import copy
import json
import re
import urllib.parse
from dataclasses import dataclass

from . import formats
from . import openai_models


OPENAI_CHAT = "openai_chat"
ANTHROPIC_MESSAGES = "anthropic_messages"
OPENAI_RESPONSES = "openai_responses"
DUMMY = "dummy"
AUTO = "auto"

# Wire protocols this client can actually speak. Single source of truth:
# consumers reference this instead of re-listing the constants, so adding a
# protocol here is all that is needed to teach the rest of the code about it.
SUPPORTED_PROTOCOLS = [OPENAI_CHAT, ANTHROPIC_MESSAGES, OPENAI_RESPONSES]
RESPONSES_LITE_HEADER = "x-openai-internal-codex-responses-lite"


class ProtocolError(ValueError):
    def __init__(self, message, payload=None):
        self.payload = copy.deepcopy(payload)
        super().__init__(message)


class ProviderDetectionError(ProtocolError):
    pass


class UnsupportedProtocolError(ProtocolError):
    pass


class StreamProtocolError(ProtocolError):
    def __init__(self, message, payload=None):
        super().__init__(message, payload=payload)


class ResponseApiError(ProtocolError):
    """A terminal error reported inside a Responses protocol stream.

    This is distinct from malformed SSE or an interrupted transport: the
    server completed the protocol exchange with ``response.failed``.  Keeping
    the classification on the exception lets the request layer retry only
    failures for which replaying the same idempotent request is appropriate.
    """

    def __init__(
            self, message, *, code=None, category="retryable",
            retryable=False, retry_after=None, payload=None):
        self.code = code
        self.category = category
        self.retryable = retryable
        self.retry_after = retry_after
        super().__init__(message, payload=payload)

    def formatted(self):
        label = "Responses API error"
        if self.code:
            label += f" ({self.code})"
        return f"{label}: {self}"


@dataclass(frozen=True)
class ProviderResponse:
    """Decoded provider payload plus response-scoped observations.

    These observations describe the provider exchange, not conversation
    content. Keeping them beside the payload prevents transport metadata from
    being injected into model-visible transcript items.
    """

    payload: object
    effective_model: str | None = None
    notice_codes: tuple[str, ...] = ()


def _header_string(headers, name):
    if not isinstance(headers, dict):
        return None
    expected = name.lower()
    for header_name, value in headers.items():
        if (isinstance(header_name, str)
                and header_name.lower() == expected
                and isinstance(value, str)
                and value.strip()):
            return value.strip()
    return None


def openai_response_model_header(headers):
    """Read OpenAI's effective-model header case-insensitively."""
    return _header_string(headers, "openai-model")


def _codex_reasoning_parameter(profile):
    if not profile.supports_reasoning_summaries:
        return None
    reasoning = {}
    if profile.default_reasoning_level is not None:
        # Codex accepts "ultra" in model/config metadata but the Responses
        # request contract currently represents it as "max".
        reasoning["effort"] = (
            "max"
            if profile.default_reasoning_level == "ultra"
            else profile.default_reasoning_level)
    if profile.default_reasoning_summary != "none":
        reasoning["summary"] = profile.default_reasoning_summary
    if profile.use_responses_lite:
        reasoning["context"] = "all_turns"
    return reasoning


def _codex_text_parameter(profile):
    if not profile.supports_verbosity or profile.default_verbosity is None:
        return None
    return {"verbosity": profile.default_verbosity}


_RESPONSE_RETRY_AFTER_RE = re.compile(
    r"try again in (\d+(?:\.\d+)?) (s|ms|seconds?)",
    re.IGNORECASE,
)


def _response_retry_after(code, message):
    # The Responses stream has no HTTP Retry-After header at this point.
    # Codex's backend includes the delay in rate-limit messages, so recognize
    # only that documented error class and its seconds/milliseconds forms.
    if code != "rate_limit_exceeded" or not isinstance(message, str):
        return None
    match = _RESPONSE_RETRY_AFTER_RE.search(message)
    if match is None:
        return None
    delay = float(match.group(1))
    if match.group(2).lower() == "ms":
        delay /= 1000
    return delay


def _response_failed_error(event):
    response = event.get("response")
    error = response.get("error") if isinstance(response, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(code, str):
        code = None
    if not isinstance(message, str) or not message:
        message = "the provider returned response.failed"

    categories = {
        "context_length_exceeded": ("context_window", False),
        "insufficient_quota": ("quota", False),
        "usage_not_included": ("subscription", False),
        "cyber_policy": ("cyber_policy", False),
        "invalid_prompt": ("invalid_request", False),
        "bio_policy": ("invalid_request", False),
        "server_is_overloaded": ("overloaded", False),
        "slow_down": ("overloaded", False),
    }
    category, retryable = categories.get(
        code, ("retryable", True))
    return ResponseApiError(
        message,
        code=code,
        category=category,
        retryable=retryable,
        retry_after=_response_retry_after(code, message),
        payload=event,
    )


@dataclass
class Provider:
    kind: str
    input_url: str
    chat_url: str
    models_url: str | None
    model_urls: list[str]
    headers: dict
    max_tokens: int
    provider_id: str | None = None
    provider_name: str | None = None
    prompt_cache: bool = False
    openai_request_profile: (
        openai_models.CodexModelRequestProfile | None) = None

    @property
    def responses_lite(self):
        return (
            self.openai_request_profile is not None
            and self.openai_request_profile.use_responses_lite
        )

    def projection_target(self, model):
        return formats.projection_target(
            self.kind,
            provider_id=self.provider_id,
            provider_name=self.provider_name,
            endpoint=self.chat_url,
            model=model,
        )

    def chat_payload(
            self, items, tools, model, *, prompt_cache_key=None):
        target = self.projection_target(model)
        if self.kind == OPENAI_CHAT:
            payload = {
                "model": model,
                "messages": formats.items_to_openai_chat_messages(
                    items, target=target),
            }
            if tools is not None:
                payload["tools"] = tools
            return payload
        if self.kind == ANTHROPIC_MESSAGES:
            system, messages = formats.items_to_anthropic_parts(
                items, target=target)
            payload = {
                "model": model,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            if self.prompt_cache:
                # Automatic prompt caching advances its breakpoint with the
                # append-only history. Anthropic hashes the logical prefix as
                # tools -> system -> messages regardless of JSON key order.
                payload["cache_control"] = {"type": "ephemeral"}
            if system:
                payload["system"] = system
            anthropic_tools = formats.openai_tools_to_anthropic_tools(tools)
            if anthropic_tools:
                payload["tools"] = anthropic_tools
            return payload
        if self.kind == OPENAI_RESPONSES:
            instructions, input_items = (
                formats.items_to_openai_responses_parts(
                    items, target=target))
            payload = {
                "model": model,
                "input": input_items,
                "max_output_tokens": self.max_tokens,
            }
            if self.provider_id == "openai-subscription":
                profile = self.openai_request_profile
                if profile is None:
                    raise ProtocolError(
                        "OpenAI subscription requests require authenticated "
                        "request profile for the selected model")
                # ChatGPT's Codex backend is not the public Responses API.
                # The authenticated model catalog is part of this request
                # contract. Do not infer capabilities from a framing flag:
                # each selected model explicitly controls parallel tools,
                # reasoning, and verbosity.
                #
                # tool_mode is deliberately absent here. It chooses Codex's
                # client-side direct/code-mode tool plan; it is not a wire
                # parameter. Loki always advertises its own direct function
                # tools, including for models whose Codex catalog recommends
                # code mode.
                payload.pop("max_output_tokens")
                payload.update({
                    "tool_choice": "auto",
                    "parallel_tool_calls": (
                        profile.supports_parallel_tool_calls
                        and not profile.use_responses_lite),
                    "store": False,
                })
                reasoning = _codex_reasoning_parameter(profile)
                if reasoning is not None:
                    payload["reasoning"] = reasoning
                    # Loki sends full history rather than provider-side
                    # response IDs, so encrypted reasoning must be returned
                    # and replayed with later requests.
                    payload["include"] = [
                        "reasoning.encrypted_content"]
                text = _codex_text_parameter(profile)
                if text is not None:
                    payload["text"] = text
                if prompt_cache_key is not None:
                    # This partitions server-side prefix caching by Loki
                    # conversation. It is neither authentication nor the
                    # per-turn sticky-routing token.
                    payload["prompt_cache_key"] = prompt_cache_key
                # Loki has no service-tier setting. Omitting service_tier asks
                # the backend for its ordinary default; Codex likewise does
                # not copy default_service_tier into a request unless a tier
                # was selected by client configuration.
            if self.responses_lite:
                # Responses-Lite is a distinct ChatGPT Codex request
                # contract. Its header requires all-turn reasoning context,
                # and its instructions and client tools are input items
                # rather than top-level request fields.
                prefix = []
                lite_tools = (
                    formats.openai_tools_to_responses_lite_tools(tools))
                if lite_tools:
                    prefix.append({
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": lite_tools,
                    })
                if instructions:
                    prefix.append({
                        "type": "message",
                        "role": "developer",
                        "content": [{
                            "type": "input_text",
                            "text": instructions,
                        }],
                    })
                payload["input"] = prefix + input_items
            else:
                if instructions:
                    payload["instructions"] = instructions
                responses_tools = (
                    formats.openai_tools_to_responses_tools(tools))
                if responses_tools:
                    payload["tools"] = responses_tools
            return payload
        if self.kind == DUMMY:
            # Never sent over the wire; async_chat_completion short-circuits on
            # this kind before any HTTP happens.
            return {}
        raise ProtocolError(f"unknown protocol {self.kind!r}")

    def streaming_chat_payload(
            self, items, tools, model, *, prompt_cache_key=None):
        payload = self.chat_payload(
            items,
            tools,
            model,
            prompt_cache_key=prompt_cache_key,
        )
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
        if self.provider_id == "openai-subscription":
            models = response.get("models", [])
            if not isinstance(models, list):
                return []
            return [
                item["slug"] for item in models
                if (isinstance(item, dict)
                    and isinstance(item.get("slug"), str)
                    and item.get("visibility") == "list")
            ]
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
            "message": {"role": "assistant"},
            "finish_reason": None,
        }
        self.tool_calls = {}
        self.legacy_function_call = None
        self.stream_extensions = []
        self.complete = False

    def _preserve_extension(self, context, value):
        formats.report_unknown(OPENAI_CHAT, context, value)
        self.stream_extensions.append({
            "context": context,
            "value": copy.deepcopy(value),
        })

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
            raise StreamProtocolError(
                _stream_error_text(chunk["error"]), payload=chunk)
        for key in [
                "id", "object", "created", "model", "system_fingerprint",
                "service_tier", "usage"]:
            if key in chunk and chunk[key] is not None:
                self.response[key] = copy.deepcopy(chunk[key])
        unknown_envelope = {
            key: value for key, value in chunk.items()
            if key not in [
                "id", "object", "created", "model",
                "system_fingerprint", "service_tier", "usage",
                "choices", "error"]
        }
        # These are final-response fields repeated on stream chunks, not
        # semantic deltas. Keep the latest value so the buffered decoder can
        # diagnose them once from the assembled response.
        self.response.update(copy.deepcopy(unknown_envelope))
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                self._preserve_extension("stream choice", choice)
                continue
            if choice.get("index", 0) != 0:
                self._preserve_extension(
                    "additional stream choice", choice)
                continue
            if choice.get("finish_reason") is not None:
                self.choice["finish_reason"] = choice["finish_reason"]
            if choice.get("logprobs") is not None:
                self.choice["logprobs"] = self._merge_delta_value(
                    self.choice.get("logprobs"),
                    choice["logprobs"],
                )
            unknown_choice = {
                key: value for key, value in choice.items()
                if key not in [
                    "index", "delta", "finish_reason", "logprobs"]
            }
            self.choice.update(copy.deepcopy(unknown_choice))
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            role = delta.get("role")
            if isinstance(role, str):
                self.choice["message"]["role"] = role
            if "content" in delta:
                content = delta["content"]
                message = self.choice["message"]
                if isinstance(content, str):
                    message["content"] = self._merge_delta_value(
                        message.get("content"), content)
                    if content:
                        self.on_text_delta(content)
                elif content is None:
                    message.setdefault("content", None)
                else:
                    # Compatibility providers may stream structured content.
                    # Preserve and assemble it even though only string deltas
                    # can be rendered incrementally.
                    self._accumulate_message_delta_field(
                        "content", content)
            refusal = delta.get("refusal")
            if isinstance(refusal, str):
                self.choice["message"]["refusal"] = (
                    str(self.choice["message"].get("refusal", ""))
                    + refusal)
            for key in ["name", "phase"]:
                if delta.get(key) is not None:
                    self.choice["message"][key] = copy.deepcopy(delta[key])
            self._accumulate_message_delta_field(
                "audio", delta.get("audio"))
            for key in formats.OPENAI_CHAT_REASONING_FIELDS:
                self._accumulate_message_delta_field(
                    key, delta.get(key))
            self._accumulate_tool_calls(delta.get("tool_calls"))
            self._accumulate_legacy_function_call(
                delta.get("function_call"))
            unknown_delta = {
                key: value for key, value in delta.items()
                if key not in [
                    "role", "content", "refusal", "name", "audio",
                    "phase", "tool_calls", "function_call",
                    *formats.OPENAI_CHAT_REASONING_FIELDS]
            }
            for key, value in unknown_delta.items():
                # A streamed message extension can arrive one token at a
                # time. Assemble it into the authoritative final message so
                # the ordinary Chat decoder diagnoses it once and preserves
                # it, instead of printing one diagnostic per token.
                self._accumulate_message_delta_field(key, value)

    def _accumulate_message_delta_field(self, key, value):
        if value is None:
            return
        message = self.choice["message"]
        message[key] = self._merge_delta_value(
            message.get(key), value)

    @classmethod
    def _merge_delta_value(cls, current, value):
        if current is None:
            return copy.deepcopy(value)
        if isinstance(current, str) and isinstance(value, str):
            return current + value
        if isinstance(current, list) and isinstance(value, list):
            return current + copy.deepcopy(value)
        if isinstance(current, dict) and isinstance(value, dict):
            result = copy.deepcopy(current)
            for key, child in value.items():
                result[key] = cls._merge_delta_value(
                    result.get(key), child)
            return result
        return copy.deepcopy(value)

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
            unknown_call = {
                key: value for key, value in delta.items()
                if key not in ["index", "id", "type", "function"]
            }
            for key, value in unknown_call.items():
                call[key] = self._merge_delta_value(
                    call.get(key), value)
            if isinstance(function, dict):
                unknown_function = {
                    key: value for key, value in function.items()
                    if key not in ["name", "arguments"]
                }
                for key, value in unknown_function.items():
                    call["function"][key] = self._merge_delta_value(
                        call["function"].get(key), value)

    def _accumulate_legacy_function_call(self, delta):
        if not isinstance(delta, dict):
            return
        if self.legacy_function_call is None:
            self.legacy_function_call = {"name": "", "arguments": ""}
        for key in ["name", "arguments"]:
            value = delta.get(key)
            if isinstance(value, str):
                self.legacy_function_call[key] += value
        unknown = {
            key: value for key, value in delta.items()
            if key not in ["name", "arguments"]
        }
        for key, value in unknown.items():
            self.legacy_function_call[key] = self._merge_delta_value(
                self.legacy_function_call.get(key), value)

    def finish(self):
        if not self.complete:
            raise StreamProtocolError(
                "OpenAI Chat stream ended before data: [DONE]")
        message = self.choice["message"]
        if self.tool_calls:
            message["tool_calls"] = [
                self.tool_calls[index] for index in sorted(self.tool_calls)]
        if self.legacy_function_call is not None:
            message["function_call"] = self.legacy_function_call
        # Chat Completion assistant messages use a nullable content field.
        # Tool-only streamed messages normally never send a string delta, so
        # their completed counterpart has content=null, not an invented empty
        # string.
        message.setdefault("content", None)
        # The first streamed chunk identifies itself as
        # ``chat.completion.chunk``.  The accumulator returns a completed
        # response object, not a chunk.
        self.response["object"] = "chat.completion"
        self.response["choices"] = [self.choice]
        if self.stream_extensions:
            self.response["_loki_stream_extensions"] = (
                self.stream_extensions)
        return self.response


class AnthropicMessagesStreamAccumulator:
    def __init__(self, on_text_delta):
        self.on_text_delta = on_text_delta
        self.message = None
        self.blocks = {}
        self.tool_json = {}
        self.stream_extensions = []
        self.complete = False

    @staticmethod
    def _uses_streamed_json(block):
        return block.get("type") in {
            "tool_use",
            "server_tool_use",
            "mcp_tool_use",
        }

    def _preserve_extension(self, context, value):
        formats.report_unknown(ANTHROPIC_MESSAGES, context, value)
        self.stream_extensions.append({
            "context": context,
            "value": copy.deepcopy(value),
        })

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
                _stream_error_text(data.get("error") or data),
                payload=data,
            )
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
            if self._uses_streamed_json(block):
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
        else:
            # Forward-compatible event types must not silently disappear.  The
            # final message remains authoritative model context; the event is
            # retained out of band for diagnostics.
            self._preserve_extension("stream event", data)

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
        else:
            self._preserve_extension(
                "content-block delta", data)

    def _finish_block(self, index):
        block = self.blocks.get(index)
        if block is None:
            raise StreamProtocolError(
                "Anthropic content block stop preceded start")
        if self._uses_streamed_json(block):
            raw = self.tool_json.get(index, "")
            if not raw and "input" in block:
                return
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
            if self._uses_streamed_json(self.blocks[index]):
                self._finish_block(index)
        self.message["content"] = [
            self.blocks[index] for index in sorted(self.blocks)]
        self.message["type"] = "message"
        self.message["role"] = "assistant"
        if self.stream_extensions:
            self.message["_loki_stream_extensions"] = (
                self.stream_extensions)
        return self.message


class OpenAIResponsesStreamAccumulator:
    def __init__(self, on_text_delta):
        self.on_text_delta = on_text_delta
        self.completed_response = None
        self.incomplete_response = None
        self.failed_response = None
        self.stream_error = None
        self.stream_extensions = []
        self.output_items = []
        self._nested_effective_model = None
        self._top_level_effective_model = None
        self.notice_codes = []

    @property
    def effective_model(self):
        return (
            self._nested_effective_model
            or self._top_level_effective_model)

    def _observe_effective_model(self, data):
        # Codex reports the model selected by the private backend in headers,
        # not in the ordinary Responses envelope's requested-model field.
        # Keep the two SSE locations separate so nested response headers have
        # deterministic precedence regardless of event order.
        response = data.get("response")
        nested = (
            openai_response_model_header(response.get("headers"))
            if isinstance(response, dict) else None)
        top_level = openai_response_model_header(data.get("headers"))
        if nested is not None and self._nested_effective_model is None:
            self._nested_effective_model = nested
        if (top_level is not None
                and self._top_level_effective_model is None):
            self._top_level_effective_model = top_level

    def _observe_metadata(self, data):
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            if metadata is not None:
                formats.report_unknown(
                    OPENAI_RESPONSES, "response metadata", metadata)
            return
        # Safety buffering and moderation are known private-backend metadata,
        # but Loki has no buffering policy or moderation UI that can consume
        # them. Recognize and discard them instead of persisting opaque data.
        if metadata.get("type") == "safety_buffering":
            return
        remaining = copy.deepcopy(metadata)
        recommendations = remaining.pop(
            "openai_verification_recommendation", None)
        remaining.pop("openai_chatgpt_moderation_metadata", None)
        if isinstance(recommendations, list):
            unknown_recommendations = []
            for recommendation in recommendations:
                if (recommendation
                        == formats.TRUSTED_ACCESS_FOR_CYBER):
                    if recommendation not in self.notice_codes:
                        self.notice_codes.append(recommendation)
                else:
                    unknown_recommendations.append(recommendation)
            if unknown_recommendations:
                formats.report_unknown(
                    OPENAI_RESPONSES,
                    "verification recommendation",
                    unknown_recommendations,
                )
        elif recommendations is not None:
            formats.report_unknown(
                OPENAI_RESPONSES,
                "verification recommendation",
                recommendations,
            )
        if remaining:
            formats.report_unknown(
                OPENAI_RESPONSES, "response metadata", remaining)

    def _preserve_extension(self, context, value):
        formats.report_unknown(OPENAI_RESPONSES, context, value)
        self.stream_extensions.append({
            "context": context,
            "value": copy.deepcopy(value),
        })

    def feed(self, event):
        try:
            data = json.loads(event.data)
        except json.JSONDecodeError as e:
            raise StreamProtocolError(
                f"OpenAI Responses stream event is not JSON: {e}")
        if not isinstance(data, dict):
            raise StreamProtocolError(
                "OpenAI Responses stream event must be an object")
        self._observe_effective_model(data)
        event_type = data.get("type") or event.event
        if event_type == "response.output_text.delta":
            delta = data.get("delta")
            if isinstance(delta, str) and delta:
                self.on_text_delta(delta)
        elif event_type == "response.output_item.done":
            item = data.get("item")
            if not isinstance(item, dict):
                raise StreamProtocolError(
                    "response.output_item.done is missing its item")
            self.output_items.append(copy.deepcopy(item))
        elif event_type == "response.completed":
            response = data.get("response")
            if not isinstance(response, dict):
                raise StreamProtocolError(
                    "response.completed is missing its response")
            self.completed_response = copy.deepcopy(response)
            self.completed_response.pop("headers", None)
            self.completed_response.setdefault("object", "response")
            self.completed_response.setdefault("status", "completed")
            return True
        elif event_type == "response.incomplete":
            response = data.get("response")
            if not isinstance(response, dict):
                raise StreamProtocolError(
                    "response.incomplete is missing its response")
            self.incomplete_response = copy.deepcopy(response)
            self.incomplete_response.pop("headers", None)
            self.incomplete_response.setdefault("object", "response")
            self.incomplete_response.setdefault("status", "incomplete")
        elif event_type == "response.failed":
            raise _response_failed_error(data)
        elif event_type == "response.cancelled":
            response = data.get("response")
            if isinstance(response, dict):
                self.failed_response = copy.deepcopy(response)
                self.failed_response.pop("headers", None)
                self.failed_response.setdefault("object", "response")
                self.failed_response.setdefault(
                    "status",
                    "cancelled"
                    if event_type == "response.cancelled" else "failed",
                )
            else:
                self.stream_error = copy.deepcopy(data)
        elif event_type == "error":
            self.stream_error = copy.deepcopy(data)
        elif event_type == "response.metadata":
            self._observe_metadata(data)
        elif not _known_responses_stream_event(event_type):
            extension = copy.deepcopy(data)
            # Codex can attach this typed payload to events other than
            # response.metadata. Loki deliberately has no consumer for it.
            extension.pop("safety_buffering", None)
            if extension and set(extension) != {"type"}:
                self._preserve_extension("stream event", extension)
        return False

    def _finish_response(self, response):
        # In the Responses streaming protocol, output_item.done carries each
        # completed conversational item. response.completed is the terminal
        # event and carries response metadata such as id and usage; Codex's
        # own decoder does not read output items from it. Preserve a populated
        # terminal output only as a compatibility fallback for implementations
        # that omit the standard item events.
        if self.output_items:
            response["output"] = copy.deepcopy(self.output_items)
        else:
            response.setdefault("output", [])
        if self.stream_extensions:
            response["_loki_stream_extensions"] = (
                self.stream_extensions)
        return response

    def finish(self):
        if self.stream_error is not None:
            raise StreamProtocolError(
                _stream_error_text(self.stream_error),
                payload=self.stream_error)
        if self.completed_response is None:
            if self.incomplete_response is not None:
                return self._finish_response(self.incomplete_response)
            if self.failed_response is not None:
                return self._finish_response(self.failed_response)
            raise StreamProtocolError(
                "OpenAI Responses stream ended before response.completed "
                "or another terminal response event")
        return self._finish_response(self.completed_response)


def _known_responses_stream_event(event_type):
    if not isinstance(event_type, str):
        return False
    if event_type in [
            "response.created", "response.in_progress", "response.queued"]:
        return True
    return event_type.startswith((
        "response.output_item.",
        "response.content_part.",
        "response.output_text.",
        "response.refusal.",
        "response.reasoning_",
        "response.function_call_arguments.",
        "response.custom_tool_call_input.",
        "response.web_search_call.",
        "response.file_search_call.",
        "response.image_generation_call.",
        "response.code_interpreter_call.",
        "response.mcp_call.",
        "response.audio.",
    ))


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
    if (response.get("object") == "response"
            and isinstance(response.get("output"), list)):
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


def build_headers(protocol, anthropic_version="2023-06-01"):
    """Build non-secret wire-protocol headers."""
    headers = {
        "Content-Type": "application/json",
    }
    if protocol == ANTHROPIC_MESSAGES:
        # This identifies the Anthropic wire format; it is required protocol
        # metadata, not an authentication credential.
        headers["anthropic-version"] = anthropic_version
    return headers


def make_provider(input_url, provider=AUTO, models_url=None,
                  max_tokens=4096, anthropic_version="2023-06-01",
                  provider_id=None, provider_name=None, prompt_cache=False,
                  openai_request_profile=None):
    protocol = resolve_protocol(input_url, provider)
    if (openai_request_profile is not None
            and not isinstance(
                openai_request_profile,
                openai_models.CodexModelRequestProfile)):
        raise ProtocolError(
            "openai_request_profile must be CodexModelRequestProfile or null")
    if openai_request_profile is not None and (
            protocol != OPENAI_RESPONSES
            or provider_id != "openai-subscription"):
        raise ProtocolError(
            "OpenAI Codex request profiles are only valid for the OpenAI "
            "ChatGPT subscription Responses provider")
    if (provider_id == "openai-subscription"
            and openai_request_profile is None):
        raise ProtocolError(
            "OpenAI ChatGPT subscription providers require authenticated "
            "request profiles")
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
            provider_id=provider_id,
            provider_name=provider_name,
            prompt_cache=False,
            openai_request_profile=None,
        )
    chat_url, resolved_models_url = endpoint_urls(input_url, protocol, models_url=models_url)
    model_urls = model_url_candidates(input_url, protocol, resolved_models_url,
                                      explicit_models_url=models_url)
    headers = build_headers(
        protocol, anthropic_version=anthropic_version)
    if (openai_request_profile is not None
            and openai_request_profile.use_responses_lite):
        headers[RESPONSES_LITE_HEADER] = "true"
    return Provider(
        kind=protocol,
        input_url=input_url,
        chat_url=chat_url,
        models_url=resolved_models_url,
        model_urls=model_urls,
        headers=headers,
        max_tokens=max_tokens,
        provider_id=provider_id,
        provider_name=provider_name,
        prompt_cache=prompt_cache,
        openai_request_profile=openai_request_profile,
    )


def json_body(payload):
    if payload is None:
        return b""
    return json.dumps(payload).encode("utf-8")


def copy_headers(headers):
    return copy.deepcopy(headers or {})
