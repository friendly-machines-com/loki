import copy
import json
import urllib.parse
from dataclasses import dataclass

import formats


OPENAI_CHAT = "openai_chat"
ANTHROPIC_MESSAGES = "anthropic_messages"
OPENAI_RESPONSES = "openai_responses"
AUTO = "auto"


class ProtocolError(ValueError):
    pass


class ProviderDetectionError(ProtocolError):
    pass


class UnsupportedProtocolError(ProtocolError):
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
            raise UnsupportedProtocolError("OpenAI Responses is reserved in the transcript format but not implemented")
        raise ProtocolError(f"unknown protocol {self.kind!r}")

    def parse_chat_response(self, response):
        if self.kind == OPENAI_CHAT:
            return formats.openai_chat_response_to_items(response)
        if self.kind == ANTHROPIC_MESSAGES:
            return formats.anthropic_response_to_items(response)
        if self.kind == OPENAI_RESPONSES:
            raise UnsupportedProtocolError("OpenAI Responses is reserved in the transcript format but not implemented")
        raise ProtocolError(f"unknown protocol {self.kind!r}")

    def parse_model_ids(self, response):
        if not isinstance(response, dict):
            return []
        data = response.get("data", [])
        if not isinstance(data, list):
            return []
        return [item["id"] for item in data
                if isinstance(item, dict) and isinstance(item.get("id"), str)]


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
        if requested not in [OPENAI_CHAT, ANTHROPIC_MESSAGES, OPENAI_RESPONSES]:
            raise ProviderDetectionError(f"unknown provider {override!r}")
        return requested
    detected = detect_protocol_from_url(url)
    if detected:
        return detected
    raise ProviderDetectionError(
        "cannot infer chat protocol from URL; set LOKI_PROVIDER=openai_chat, "
        "LOKI_PROVIDER=anthropic_messages, or LOKI_PROVIDER=openai_responses")


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
    # Treat input_url as a provider base URL and append the concrete protocol
    # endpoint. This is separate from _replace_path(), which is for known /v1
    # roots or complete endpoints.
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
    if not api_key:
        return headers
    if auth_header:
        headers[auth_header] = api_key
        return headers
    if protocol == ANTHROPIC_MESSAGES:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = anthropic_version
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def make_provider(input_url, provider=AUTO, api_key="", models_url=None,
                  max_tokens=4096, anthropic_version="2023-06-01",
                  auth_header=None):
    protocol = resolve_protocol(input_url, provider)
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
