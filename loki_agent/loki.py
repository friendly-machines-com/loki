#!/usr/bin/env python3

# TODO: chat (and file) history rewinding
# TODO: Provide command to set effort level
# TODO: /goal
# TODO: paste support ? maybe not; automatic; weird 4096 Byte length limit ?  It's especially good so pasting something doesnt send 237 requests in a row
# TODO: mouse support; but what for?
# TODO: input with readline support (just print the text you have so far--up to the cursor)
# TODO: maybe sixel bitmap support; but what for?
# TODO: background tasks and job control, maybe
# TODO: make this an actual shell; pipeable and so on like always; history search etc

import sys
import os
import asyncio
import collections
import copy
import json
import random
import time
import re
import urllib.parse
import subprocess
import signal
import socket
import uuid
import getopt
import tempfile
import shutil
import threading
import shlex
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pprint import pprint, pformat

from . import formats
from . import http_client
from . import models as modelsdev
from . import protocols
from . import savefiles
from . import sse
from . import tool_runtime
from .connections import ConnectionDescriptor, ConnectionDescriptorError
from .credentials import CredentialStore
from .savefiles import ResumeTranscriptRenderer
from .sessions import Session, default_session
from . import terminals
from .terminals import (
    input_session, restore_output_area_after_input, terminal)


# Conversation state lives in a sessions.Session.  current_session() returns
# the process-wide session used by the terminal and headless front-ends; a
# front-end hosting several conversations passes its own Session instead.

_DEFAULT_SESSION: Session | None = None


def current_session() -> Session:
    global _DEFAULT_SESSION
    if _DEFAULT_SESSION is None:
        _DEFAULT_SESSION = default_session()
    return _DEFAULT_SESSION


def current_cwd() -> str:
    return current_session().shell_cwd


def current_previous_cwd() -> str:
    return current_session().previous_shell_cwd


def current_config():
    return current_session().runtime_config


def current_model() -> str:
    return current_session().model


def current_transcript() -> list:
    return current_session().transcript_items


def current_todos() -> list:
    return current_session().session_todos


def current_toolsets() -> list:
    return current_session().session_toolsets


def current_state() -> dict:
    return current_session().session_state


def current_chat_log_path():
    return current_session().chat_log_path


def current_dirty() -> bool:
    return current_session().chat_log_dirty


def current_agent_mode() -> str:
    return current_session().agent_mode


def current_job_manager():
    session = current_session()
    if session.job_manager is None:
        session.job_manager = JobManager(LOKI_JOB_STATE_DIR)
    return session.job_manager

# --------------------------------------------------------------------------

computer = socket.gethostname()

ERROR_COLOR = 1
TOOL_CALL_COLOR = 5

MAX_LOOP_LIMIT = 50
READ_CHAR_CAP = 10 * 1024 * 1024
READ_PATHS_LIMIT = 1000
READ_DEFAULT_LINES = 2000
READ_MAX_LINES = 2000
BASH_DEFAULT_TIMEOUT_MS = 300000
BASH_MAX_TIMEOUT_MS = 600000
BASH_MAX_OUTPUT_CHARS = 10_000_000
WRITE_MAX_OUTPUT_CHARS = 1_000_000
GLOB_MAX_RESULTS = 100
GREP_DEFAULT_HEAD_LIMIT = 250
SEARCH_TIMEOUT_S = 30
SUBAGENT_TIMEOUT_S = 600
TODO_MAX_TODOS = 100
SKILL_MAX_BYTES = 100_000
LOKI_CONFIG_DIR_NAME = "loki"
XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
LOKI_CONFIG_DIR = os.path.join(os.path.expanduser(XDG_CONFIG_HOME), LOKI_CONFIG_DIR_NAME)
XDG_STATE_HOME = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
LOKI_STATE_DIR = os.path.join(os.path.expanduser(XDG_STATE_HOME), LOKI_CONFIG_DIR_NAME)
LOKI_JOB_STATE_DIR = os.path.join(LOKI_STATE_DIR, "jobs")
STARTUP_CWD = os.getcwd()
_UMASK = os.umask(0)          # probe: sets umask to 0 momentarily, returns the previous value
os.umask(_UMASK)             # restore immediately; no concurrent code runs at startup
LOCAL_LOKI_DIR = os.path.join(STARTUP_CWD, ".loki")
CHAT_LOG_DIR = os.path.join(LOCAL_LOKI_DIR, "chats")
JOB_TAIL_CHARS = 20_000

WEBFETCH_TIMEOUT_S = 30
LLM_STREAM_IDLE_TIMEOUT_S = 300
WEBFETCH_MAX_BYTES = 10_485_760  # 10 MiB
WEBFETCH_MAX_OUTPUT = 100_000   # 100 KB inline result
WEBFETCH_MAX_PROMPT_CHARS = 200_000
WEBFETCH_CACHE_TTL = 15 * 60    # 15 minutes
WEBFETCH_CACHE_MAX_ENTRIES = 128
WEBSEARCH_TIMEOUT_S = 20
WEBSEARCH_MAX_RESPONSE_BYTES = 2_000_000
WEBSEARCH_MAX_RESULTS = 8
DUCKDUCKGO_HTML_SEARCH_URL = 'https://html.duckduckgo.com/html/'
HTTP_MAX_RESPONSE_BYTES = http_client.HTTP_MAX_RESPONSE_BYTES

# Retry policy for transient transport failures (timeouts, connection resets).
# Safe in this codebase because every request opens a fresh TCP connection
# with Connection: close; the kernel drops late packets from the old socket on
# the new connection's source port. Edit these to tune or disable.
HTTP_RETRY_MAX_ATTEMPTS = 3         # for GETs and the read-only search POST
HTTP_RETRY_MAX_ATTEMPTS_LLM = 3     # for chat-completion POSTs (with idempotency key)
HTTP_RETRY_BASE_DELAY_S = 0.5
HTTP_RETRY_MAX_JITTER_S = 0.5
HTTP_RETRY_BACKOFF_FACTOR = 2.0
# Idempotency-key header injected on every chat-completion POST so the provider
# can dedup a retry. Anthropic honors this server-side; OpenAI-compat servers
# may ignore it and bill for each attempt -- set HTTP_RETRY_MAX_ATTEMPTS_LLM=1
# if your provider does not honor the header.
LLM_IDEMPOTENCY_HEADER_ANTHROPIC = "anthropic-idempotency-key"
LLM_IDEMPOTENCY_HEADER_OPENAI = "Idempotency-Key"


class ApiError(Exception):
    def __init__(self, request_url: str, status: int, reason: str, body_text: str):
        self.request_url = request_url
        self.status = status
        self.reason = reason
        self.body_text = body_text
        try:
            self.body = json.loads(body_text) if body_text else None
        except json.JSONDecodeError:
            self.body = None
        super().__init__(self.summary())

    def summary(self) -> str:
        return f"API Error for <{self.request_url}>: HTTP {self.status} {self.reason}"

    def formatted_body(self) -> str:
        if self.body is not None:
            # API error bodies are often JSON dictionaries. Pretty-print them so
            # terminal errors stay readable. Use the real terminal width instead
            # of an arbitrary fixed column so wide terminals are not wrapped early.
            width = shutil.get_terminal_size(fallback=(80, 24)).columns
            return pformat(self.body, width=width)
        return self.body_text

    def formatted(self) -> str:
        body = self.formatted_body()
        if not body:
            return self.summary()
        return f"{self.summary()}:\n{body}"


class StreamingApiError(ApiError):
    def formatted(self) -> str:
        return (
            super().formatted()
            + "\nStreaming was requested. If this server does not support "
              "streaming, set LOKI_STREAM=0.")


class StreamCancelled(Exception):
    pass


@dataclass
class RuntimeConfig:
    url: str
    provider_kind: str
    netloc: str
    api_key: str
    chat_provider: protocols.Provider
    headers: dict
    model: str
    # Retained from startup env so a mid-session Provider reinstall
    # (e.g. /model choosing a model on a different provider) can reproduce
    # the exact same Provider settings instead of falling back to defaults.
    anthropic_version: str = "2023-06-01"
    auth_header: str | None = None
    provider_id: str | None = None
    provider_name: str | None = None
    credential_env: str | None = None
    model_status: str | None = None
    stream: bool = False
    prompt_cache: bool = False


CREDENTIALS: CredentialStore | None = None


def _int_setting(name, default, credentials: CredentialStore):
    value = credentials.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _bool_setting(name, default, credentials: CredentialStore):
    value = credentials.get(name)
    if not value:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{name} must be one of: 1, 0, true, false, yes, no, on, off")


def make_runtime_config(url, provider_kind, api_key, *, model="", models_url=None,
                        max_tokens=4096, anthropic_version="2023-06-01",
                        auth_header=None, provider_id=None, provider_name=None,
                        credential_env=None, model_status=None, stream=False,
                        prompt_cache=False):
    """Build a RuntimeConfig (and its Provider) from explicit parameters.

    The single place a production Provider is constructed. Startup reads the
    environment into these parameters; a mid-session /model change calls
    reinstall_provider() which reuses this to rebuild the Provider from the
    current config plus per-model overrides.
    """
    chat_provider = protocols.make_provider(
        url,
        provider=provider_kind,
        api_key=api_key,
        models_url=models_url,
        max_tokens=max_tokens,
        anthropic_version=anthropic_version,
        auth_header=auth_header,
        provider_id=provider_id,
        provider_name=provider_name,
        prompt_cache=prompt_cache,
    )
    return RuntimeConfig(
        url=url,
        provider_kind=provider_kind,
        netloc=urllib.parse.urlparse(url).netloc,
        api_key=api_key,
        chat_provider=chat_provider,
        headers=chat_provider.headers,
        model=model,
        anthropic_version=anthropic_version,
        auth_header=auth_header,
        provider_id=provider_id,
        provider_name=provider_name,
        credential_env=credential_env,
        model_status=model_status,
        stream=stream,
        prompt_cache=prompt_cache,
    )


def build_config_from_env(environ=os.environ,
                          credentials: CredentialStore | None = None):
    """Build an explicit/default runtime config from captured startup values."""
    if credentials is None:
        credentials = CredentialStore.capture(environ)

    # Only Loki's namespace configures Loki. Generic SDK variables such as
    # OPENAI_API_BASE and ANTHROPIC_BASE_URL may belong to unrelated programs
    # sharing this environment and must not silently select Loki's endpoint.
    config_url = credentials.get("LOKI_API_BASE")
    if not config_url:
        raise ValueError(
            "API endpoint missing; set LOKI_API_BASE or select one with /model")
    provider_override = credentials.get("LOKI_PROVIDER")
    config_provider_kind = protocols.resolve_protocol(config_url, provider_override or "auto")

    if config_provider_kind == protocols.DUMMY:
        # No-op provider for testing: no network, no API key, fake URL.
        config_api_key = "dummy-key"
        config_model = credentials.get("LOKI_MODEL") or "dummy"
        return make_runtime_config(
            config_url,
            config_provider_kind,
            config_api_key,
            model=config_model,
            max_tokens=_int_setting("LOKI_MAX_TOKENS", 4096, credentials),
            anthropic_version=credentials.get(
                "LOKI_ANTHROPIC_VERSION", "2023-06-01"),
            provider_name="Explicit LOKI_* connection",
            stream=_bool_setting("LOKI_STREAM", False, credentials),
            prompt_cache=False,
        )

    # Never infer credential ownership from the wire protocol. An explicitly
    # configured endpoint uses LOKI_API_KEY when present and otherwise sends
    # no authentication header.
    credential_env, config_api_key = credentials.first_available(
        ["LOKI_API_KEY"])

    config_model = credentials.get("LOKI_MODEL") or ""
    return make_runtime_config(
        config_url,
        config_provider_kind,
        config_api_key,
        model=config_model,
        models_url=credentials.get("LOKI_MODELS_URL") or None,
        max_tokens=_int_setting("LOKI_MAX_TOKENS", 4096, credentials),
        anthropic_version=credentials.get(
            "LOKI_ANTHROPIC_VERSION", "2023-06-01"),
        auth_header=credentials.get("LOKI_AUTH_HEADER") or None,
        provider_name="Explicit LOKI_* connection",
        credential_env=credential_env,
        stream=_bool_setting("LOKI_STREAM", False, credentials),
        prompt_cache=_bool_setting(
            "LOKI_PROMPT_CACHE",
            urllib.parse.urlparse(config_url).hostname
            == "api.anthropic.com",
            credentials,
        ),
    )


def explicit_api_base_configured(credentials: CredentialStore) -> bool:
    return bool(credentials.get("LOKI_API_BASE"))


def explicit_connection_option(
        credentials: CredentialStore
) -> modelsdev.ExplicitConnectionOption | None:
    """Return a selectable captured LOKI_* connection when it is complete."""
    try:
        config = build_config_from_env(credentials=credentials)
    except (protocols.ProtocolError, ValueError):
        return None
    if not config.model:
        return None
    return modelsdev.ExplicitConnectionOption(
        model=config.model,
        api_url=config.url,
        protocol=config.provider_kind,
    )


def config_from_connection_descriptor(
        descriptor: ConnectionDescriptor,
        credentials: CredentialStore) -> RuntimeConfig:
    if descriptor.credential_env is None:
        api_key = ""
    else:
        api_key = credentials.get(descriptor.credential_env)
        if not api_key:
            raise ValueError(
                "saved connection requires missing "
                f"{descriptor.credential_env}")

    provider_kind = credentials.get("LOKI_PROVIDER") or descriptor.protocol
    configured_model = credentials.get("LOKI_MODEL")
    config_model = configured_model or descriptor.model
    model_status = (
        descriptor.model_status
        if not configured_model or configured_model == descriptor.model
        else None)
    max_tokens = (
        _int_setting("LOKI_MAX_TOKENS", descriptor.max_tokens, credentials)
        if credentials.get("LOKI_MAX_TOKENS") else descriptor.max_tokens)
    anthropic_version = (
        credentials.get("LOKI_ANTHROPIC_VERSION")
        or descriptor.anthropic_version)
    auth_header = (
        credentials.get("LOKI_AUTH_HEADER")
        if credentials.get("LOKI_AUTH_HEADER")
        else descriptor.auth_header)
    stream = (
        _bool_setting("LOKI_STREAM", descriptor.stream, credentials)
        if credentials.get("LOKI_STREAM") else descriptor.stream)
    prompt_cache = (
        _bool_setting(
            "LOKI_PROMPT_CACHE", descriptor.prompt_cache, credentials)
        if credentials.get("LOKI_PROMPT_CACHE")
        else descriptor.prompt_cache)
    return make_runtime_config(
        # Restore the concrete endpoint that was actually used, rather than
        # deriving it again from a catalog base URL.
        descriptor.chat_url,
        provider_kind,
        api_key,
        model=config_model,
        models_url=descriptor.models_url,
        max_tokens=max_tokens,
        anthropic_version=anthropic_version,
        auth_header=auth_header,
        provider_id=descriptor.provider_id,
        provider_name=descriptor.provider_name,
        credential_env=descriptor.credential_env,
        model_status=model_status,
        stream=stream,
        prompt_cache=prompt_cache,
    )


def config_from_modelsdev_selection(
        provider_id: str,
        provider_entry: dict,
        model_entry: dict,
        credentials: CredentialStore) -> RuntimeConfig:
    access = modelsdev.provider_access(provider_entry, credentials)
    if access is None:
        raise ValueError(
            f"provider {provider_id!r} is not available from startup credentials")
    selected_model = model_entry.get("id") or model_entry.get("name")
    if not selected_model:
        raise ValueError(f"provider {provider_id!r} returned a model without an id")
    model_status = model_entry.get("status")
    if not isinstance(model_status, str):
        model_status = None
    return make_runtime_config(
        access.api_url,
        access.protocol,
        credentials.get(access.credential_env),
        model=selected_model,
        max_tokens=_int_setting("LOKI_MAX_TOKENS", 4096, credentials),
        anthropic_version=credentials.get(
            "LOKI_ANTHROPIC_VERSION", "2023-06-01"),
        auth_header=credentials.get("LOKI_AUTH_HEADER") or None,
        provider_id=provider_id,
        provider_name=provider_entry.get("name"),
        credential_env=access.credential_env,
        model_status=model_status,
        stream=_bool_setting("LOKI_STREAM", False, credentials),
        prompt_cache=_bool_setting(
            "LOKI_PROMPT_CACHE",
            provider_id == "anthropic",
            credentials,
        ),
    )


def active_connection_descriptor() -> ConnectionDescriptor | None:
    if not current_config() or current_config().provider_kind == protocols.DUMMY:
        return None
    credential_env = current_config().credential_env
    provider = current_config().chat_provider
    return ConnectionDescriptor(
        provider_id=current_config().provider_id,
        provider_name=current_config().provider_name,
        model=current_model(),
        chat_url=provider.chat_url,
        models_url=provider.models_url,
        protocol=current_config().provider_kind,
        credential_env=credential_env,
        max_tokens=provider.max_tokens,
        anthropic_version=current_config().anthropic_version,
        auth_header=current_config().auth_header,
        model_status=current_config().model_status,
        stream=current_config().stream,
        prompt_cache=current_config().prompt_cache,
    )


def apply_runtime_config(config: RuntimeConfig):
    # model is derived from runtime_config (Session.model property); nothing
    # else to publish.
    current_session().runtime_config = config


_UNSET = object()


def reinstall_provider(*, model=None, url=None, provider_kind=None, api_key=None,
                       models_url=None, max_tokens=None, anthropic_version=None,
                       auth_header=None, provider_id=None, provider_name=None,
                       credential_env=None, model_status=_UNSET, stream=None,
                       prompt_cache=None):
    """Rebuild and swap RUNTIME_CONFIG (and its Provider) mid-session.

    Overrides default to the current runtime config, so a bare call reinstates
    the Provider from the existing settings
    """
    current = current_config()
    if current is None:
        raise RuntimeError("cannot reinstall provider before startup config is applied")
    new_url = url if url is not None else current.url
    new_kind = provider_kind if provider_kind is not None else current.provider_kind
    new_api_key = api_key if api_key is not None else current.api_key
    new_model = model if model is not None else current.model
    new_anthropic_version = anthropic_version if anthropic_version is not None else current.anthropic_version
    new_auth_header = auth_header if auth_header is not None else current.auth_header
    new_provider_id = (
        provider_id if provider_id is not None else current.provider_id)
    if model_status is _UNSET:
        same_catalog_entry = (
            new_model == current.model
            and new_url == current.url
            and new_kind == current.provider_kind
            and new_provider_id == current.provider_id)
        new_model_status = (
            current.model_status if same_catalog_entry else None)
    else:
        new_model_status = model_status
    apply_runtime_config(make_runtime_config(
        new_url,
        new_kind,
        new_api_key,
        model=new_model,
        # models_url is passed through as-is (None -> derive from new_url);
        # never carry the previous provider's models URL across a provider
        # switch.
        models_url=models_url,
        max_tokens=max_tokens if max_tokens is not None else current.chat_provider.max_tokens,
        anthropic_version=new_anthropic_version,
        auth_header=new_auth_header,
        provider_id=new_provider_id,
        provider_name=(provider_name if provider_name is not None
                       else current.provider_name),
        credential_env=(credential_env if credential_env is not None
                        else current.credential_env),
        model_status=new_model_status,
        stream=stream if stream is not None else current.stream,
        prompt_cache=(
            prompt_cache
            if prompt_cache is not None
            else current.prompt_cache),
    ))


class LruCache(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = collections.OrderedDict()

    def __setitem__(self, key, value):
        self.items[key] = value
        self.items.move_to_end(key)
        while len(self.items) > self.max_size:
            self.items.popitem(last=False)

    def __hasitem__(self, key):
        return key in self.items

    def __contains__(self, key):
        return key in self.items

    def get(self, key, default=None):
        if key not in self.items:
            return default
        self.items.move_to_end(key)
        return self.items[key]

    def __getitem__(self, key):
        self.items.move_to_end(key)
        return self.items[key]

    def clear(self):
        self.items.clear()

    def pop(self, key, default=None):
        return self.items.pop(key, default)


file_state = LruCache(READ_PATHS_LIMIT) # file_path -> last content the agent observed; keys = files Read this session
_webfetch_cache = LruCache(WEBFETCH_CACHE_MAX_ENTRIES)  # url -> (fetched_at_epoch, content_text, content_type, final_url, status)


def _resolve_path(path: str, base_dir: str = None) -> str:
    if not path:
        return path
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        joined = path
    else:
        joined = os.path.join(base_dir or current_cwd(), path)
    # Resolve symlinks so Read/Write/Edit operate on the real target inode
    # (matches open()'s follow behavior, keeps the temp file on the same
    # filesystem as the target for atomic os.replace, and keys file_state
    # consistently across reads and edits). realpath does not raise on
    # dangling symlinks -- it appends the missing tail lexically -- so creating
    # a file through a dangling link works.
    return os.path.realpath(os.path.normpath(joined))


def _path_under(path: str, parent: str) -> bool:
    try:
        return os.path.commonpath([path, parent]) == parent
    except ValueError:
        return False


def display_path(path: str) -> str:
    return os.path.normpath(path)


def change_shell_cwd(path: str = None) -> str:
    session = current_session()
    target = (path or "").strip()
    if not target:
        target = "~"
    elif target == "-":
        target = session.previous_shell_cwd
    resolved = _resolve_path(os.path.expanduser(target))
    if not os.path.isdir(resolved):
        raise FileNotFoundError(resolved)
    old_cwd = session.shell_cwd
    session.shell_cwd = resolved
    session.previous_shell_cwd = old_cwd
    return session.shell_cwd


ToolSchemaError = tool_runtime.ToolSchemaError


class ToolValidationError(ValueError):
    pass


SCHEMA_ANNOTATION_KEYS = tool_runtime.SCHEMA_ANNOTATION_KEYS
SCHEMA_VALIDATION_KEYS = tool_runtime.SCHEMA_VALIDATION_KEYS
SCHEMA_ALLOWED_KEYS = tool_runtime.SCHEMA_ALLOWED_KEYS


def _schema_path(path: str, key) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


def _json_type_name(value) -> str:
    return tool_runtime.json_type_name(value)


def _matches_json_type(value, expected_type: str) -> bool:
    return tool_runtime.matches_json_type(value, expected_type)


def _validate_schema(schema: dict, value, path: str = "$"):
    if path != "$":
        raise ToolSchemaError(
            "_validate_schema compatibility wrapper only accepts root path")
    issues = tool_runtime.validate_schema(schema, value)
    if issues:
        raise ToolValidationError(issues[0].message)


def _close_object_schemas(schema: dict):
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" or "properties" in schema:
        # Closed object schemas make unexpected model-generated arguments fail
        # validation before they reach tool handlers.
        schema.setdefault("additionalProperties", False)
        for subschema in schema.get("properties", {}).values():
            _close_object_schemas(subschema)
    if "items" in schema:
        _close_object_schemas(schema["items"])


def _check_schema_supported(schema: dict, path: str = "$"):
    if not isinstance(schema, dict):
        raise ToolSchemaError(f"{path}: schema must be an object")
    unsupported = sorted(set(schema) - SCHEMA_ALLOWED_KEYS)
    if unsupported:
        raise ToolSchemaError(f"{path}: unsupported schema keys: {', '.join(unsupported)}")
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict):
            raise ToolSchemaError(f"{path}: properties must be an object")
        for key, subschema in properties.items():
            _check_schema_supported(subschema, _schema_path(path, key))
    if "items" in schema:
        _check_schema_supported(schema["items"], f"{path}[]")


def _build_tool_registry(tools: list, handlers: dict) -> dict:
    registry = {}
    seen = set()
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name")
        parameters = function.get("parameters")
        if not name or parameters is None:
            continue
        if name in seen:
            raise ToolSchemaError(f"duplicate tool definition: {name}")
        seen.add(name)
        if name not in handlers:
            raise ToolSchemaError(f"missing handler for tool: {name}")
        _close_object_schemas(parameters)
        _check_schema_supported(parameters)
        handler = handlers[name]
        sync_handler = handler.get("handler")
        async_handler = handler.get("async_handler")
        if sync_handler is None and async_handler is None:
            raise ToolSchemaError(f"tool {name} has neither handler nor async_handler")
        registry[name] = {
            "definition": tool,
            "schema": parameters,
            "semantics": copy.deepcopy(
                handler.get("semantics", {})),
            "handler": sync_handler,
            "async_handler": async_handler,
            "explore": handler.get("explore", False),
        }

    extra_handlers = sorted(set(handlers) - seen)
    if extra_handlers:
        raise ToolSchemaError(f"handler without tool definition: {', '.join(extra_handlers)}")
    return registry


def validate_tool_args(fn_name: str, args) -> str | None:
    spec = TOOL_REGISTRY.get(fn_name)
    if spec is None:
        return f"Error: unknown tool: {fn_name}"
    try:
        _validate_schema(spec["schema"], args)
    except ToolValidationError as e:
        return f"Error: invalid arguments for {fn_name}: {e}"
    except ToolSchemaError as e:
        return f"Error: invalid schema for {fn_name}: {e}"
    return None


def _truncate_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n... [output truncated: {len(s)} chars total, {max_chars} shown]"


def _format_numbered_lines(lines: list[str], first_line_number: int = 1) -> str:
    return "\n".join(f"{i}\t{line}" for i, line in enumerate(lines, start=first_line_number))


def _format_bash_result(stdout: str, stderr: str, exit_code: int | None,
                        status: str = "completed", no_output_expected: bool = False) -> str:
    # Keep stdout and stderr separate in normal mode. The model can still ask
    # for shell-level merging with 2>&1 inside the command when that is desired.
    parts = [f"status: {status}"]
    if exit_code is not None:
        parts.append(f"exit_code: {exit_code}")
    if no_output_expected:
        parts.append("no_output_expected: true")
    parts.extend([
        "[stdout]",
        stdout if stdout else "(empty)",
        "[stderr]",
        stderr if stderr else "(empty)",
    ])
    return _truncate_text("\n".join(parts), BASH_MAX_OUTPUT_CHARS)


def _atomic_write_text(file_path: str, content: str):
    directory = os.path.dirname(file_path) or '.'
    # Capture the desired final mode BEFORE writing so a write/replace failure
    # can never leave the public path's mode corrupted. For an existing file we
    # preserve its current mode (rwx bits, including execute); for a new file
    # we match what plain open(...,'w') would produce, i.e. 0o666 & ~umask.
    # os.stat follows symlinks (callers resolve them via _resolve_path), so this
    # reads the real target inode's mode.
    try:
        target_mode = os.stat(file_path).st_mode & 0o7777
    except FileNotFoundError:
        target_mode = 0o666 & ~_UMASK
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(file_path)}.",
        suffix=".tmp",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            #os.fsync(f.fileno())
        # Apply the target mode to the temp inode BEFORE the atomic publish.
        # This matches what text editors (e.g. vim's buf_write) do: the temp
        # lives in the user's own destination directory (not a shared /tmp),
        # so broadening it here is safe -- the content is fully written and the
        # directory is not adversarially pre-populated. Doing it before replace
        # means the public path never appears with wrong-to-broad perms: the
        # rename is atomic, so it jumps straight from (old inode, old mode) to
        # (new inode, correct mode).
        os.chmod(tmp_path, target_mode)
        os.replace(tmp_path, file_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            # The write failed, but another cleanup path may already have
            # removed the temp file; preserve the original write exception.
            pass
        raise


def _stale_file_error(file_path: str, action: str) -> str | None:
    observed = file_state.get(file_path)
    if observed is None:
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            current = f.read()
    except Exception as e:
        return f"Error checking current file contents before {action}: {e}"
    if observed != current:
        return (f"Error: {file_path} changed on disk since you last read it. "
                f"Read it again before {action}.")
    return None


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _decode_spooled_output(data: bytes) -> str:
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('utf-8', errors='replace')


def _read_spool_tail(path: str, max_chars: int = JOB_TAIL_CHARS) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            # Read extra bytes because UTF-8 may use multiple bytes per
            # character; trim after decoding so the character cap is stable.
            f.seek(max(0, size - max_chars * 4))
            text = _decode_spooled_output(f.read())
    except FileNotFoundError:
        return ""
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


@dataclass
class Job:
    id: str
    command: str
    argv: list[str] | None
    shell: bool
    description: str
    background: bool
    spool_dir: str
    stdout_path: str
    stderr_path: str
    metadata_path: str
    started_at: float
    started_at_iso: str
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    pid: int | None = None
    pgid: int | None = None
    status: str = "starting"
    exit_code: int | None = None
    signal: int | None = None
    finished_at: float | None = None
    finished_at_iso: str | None = None
    timeout_ms: int | None = None


class JobManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.session_id = str(uuid.uuid4())
        self.session_dir = os.path.join(base_dir, self.session_id)
        self.jobs = {}
        self._counter = 0
        self._lock = threading.RLock()

    def _next_job_id(self) -> str:
        with self._lock:
            self._counter += 1
            return str(self._counter)

    def _job_dir(self, job_id: str) -> str:
        return os.path.join(self.session_dir, job_id)

    def _job_metadata(self, job: Job) -> dict:
        return {
            "id": job.id,
            "command": job.command,
            "argv": job.argv,
            "shell": job.shell,
            "description": job.description,
            "background": job.background,
            "pid": job.pid,
            "pgid": job.pgid,
            "status": job.status,
            "exit_code": job.exit_code,
            "signal": job.signal,
            "started_at": job.started_at_iso,
            "finished_at": job.finished_at_iso,
            "timeout_ms": job.timeout_ms,
            "stdout_path": job.stdout_path,
            "stderr_path": job.stderr_path,
        }

    def _write_metadata(self, job: Job):
        _atomic_write_text(job.metadata_path, json.dumps(self._job_metadata(job), indent=2) + "\n")

    def _record_exit(self, job: Job, exit_code: int):
        with self._lock:
            if job.status in ["timed_out", "stopped"]:
                # Preserve the user-visible reason even after wait() reports the
                # eventual process exit code.
                pass
            elif exit_code < 0:
                job.status = "signaled"
                job.signal = -exit_code
            else:
                job.status = "exited"
            job.exit_code = exit_code
            job.finished_at = time.time()
            job.finished_at_iso = _now_iso()
            self._write_metadata(job)

    def _refresh_job(self, job: Job):
        if job.status not in ["running", "stopping"]:
            return
        exit_code = job.process.returncode
        if exit_code is not None:
            self._record_exit(job, exit_code)

    async def _spawn(self, command, display_command: str, description: str,
                     background: bool, timeout_ms: int | None, shell: bool,
                     env: dict | None = None, cwd: str = None) -> Job:
        os.makedirs(self.session_dir, exist_ok=True)
        job_id = self._next_job_id()
        spool_dir = self._job_dir(job_id)
        os.makedirs(spool_dir, exist_ok=True)
        stdout_path = os.path.join(spool_dir, "stdout.log")
        stderr_path = os.path.join(spool_dir, "stderr.log")
        metadata_path = os.path.join(spool_dir, "job.json")
        job = Job(
            id=job_id,
            command=display_command,
            argv=None if shell else list(command),
            shell=shell,
            description=description,
            background=background,
            spool_dir=spool_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata_path=metadata_path,
            started_at=time.time(),
            started_at_iso=_now_iso(),
            timeout_ms=timeout_ms,
        )
        with open(stdout_path, 'wb') as stdout_file, open(stderr_path, 'wb') as stderr_file:
            # Tool output is spooled to separate fd streams. start_new_session
            # gives each job a process group so timeouts and JobStop can signal
            # the whole command tree.
            if shell:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                    env=env,
                    cwd=cwd or current_cwd(),
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                    env=env,
                    cwd=cwd or current_cwd(),
                )
        job.process = proc
        job.pid = proc.pid
        try:
            job.pgid = os.getpgid(proc.pid)
        except OSError:
            job.pgid = proc.pid
        job.status = "running"
        with self._lock:
            self.jobs[job.id] = job
            self._write_metadata(job)
        return job

    async def _wait_for_job(self, job: Job) -> int:
        return await job.process.wait()

    async def _monitor_background_job(self, job: Job):
        try:
            exit_code = await self._wait_for_job(job)
            self._record_exit(job, exit_code)
        except Exception as e:
            # Background monitor failures cannot be returned synchronously, so
            # record them in the same stderr spool the caller already inspects.
            with self._lock:
                job.status = "monitor_error"
                job.finished_at = time.time()
                job.finished_at_iso = _now_iso()
                with open(job.stderr_path, 'ab') as stderr_file:
                    stderr_file.write(f"\n[job monitor error: {type(e).__name__}: {e}]\n".encode('utf-8'))
                self._write_metadata(job)

    async def run_foreground(self, command, display_command: str, timeout_ms: int,
                             description: str = "", shell: bool = False,
                             output_chars: int = BASH_MAX_OUTPUT_CHARS,
                             env: dict | None = None, cwd: str = None,
                             cancel_event: asyncio.Event | None = None):
        job = await self._spawn(command, display_command, description, False, timeout_ms, shell,
                                env=env, cwd=cwd)
        
        # Define the exit_task and nested signal helpers first so they are
        # available throughout the entire function.
        exit_task = asyncio.create_task(self._wait_for_job(job))

        def _signal_job(signum):
            # Signal the job's whole process group so shell children are not
            # left behind. Guarded by the lock and the running status: a
            # job already reaped here may have had its pgid recycled by an
            # unrelated process group, and killpg would hit that instead.
            with self._lock:
                if job.status != "running":
                    return False
            try:
                os.killpg(job.pgid or job.pid, signum)
                return True
            except ProcessLookupError:
                return False

        async def terminate(signum, grace_s: float):
            sent = _signal_job(signum)
            if not sent:
                return await exit_task
            try:
                exit_code = await asyncio.wait_for(
                    asyncio.shield(exit_task), timeout=grace_s)
            except asyncio.TimeoutError:
                _signal_job(signal.SIGKILL)
                exit_code = await exit_task
            return exit_code

        # 1. Pre-wait cancellation check: Was the event set while self._spawn awaited?
        # Bypasses asyncio.wait entirely to avoid watcher/selector registration races.
        if cancel_event is not None and cancel_event.is_set():
            # Send the termination signal while status is still "running" so
            # that _signal_job is not locked out by the guard.
            exit_code = await terminate(signal.SIGINT, 2.0)
            with self._lock:
                job.status = "cancelled"
                self._write_metadata(job)
            self._record_exit(job, exit_code)
            return job, "cancelled", _read_spool_tail(job.stdout_path, output_chars), _read_spool_tail(job.stderr_path, output_chars)

        cancel_task = (asyncio.create_task(cancel_event.wait())
                       if cancel_event is not None else None)
        watchers = {exit_task} if cancel_task is None else {
            exit_task, cancel_task}
        
        try:
            done, _pending = await asyncio.wait(
                watchers, timeout=timeout_ms / 1000,
                return_when=asyncio.FIRST_COMPLETED)
        finally:
            # Clean up pending cancel task if it is still running to prevent memory leaks
            if cancel_task is not None and not cancel_task.done():
                cancel_task.cancel()
                try:
                    await cancel_task
                except asyncio.CancelledError:
                    pass

        # Cancel wins a simultaneous finish: if the user cancelled and the
        # job also exited in the same loop pass, the intent outranks the
        # race. A job that exited on its own (no cancel pending) records
        # completed.
        interrupted = cancel_task is not None and (
            cancel_task in done or cancel_task.done())
        if interrupted and exit_task not in done:
            pass  # Fall through to the kill path below
        elif exit_task in done:
            self._record_exit(job, exit_task.result())
            return job, "completed", _read_spool_tail(job.stdout_path, output_chars), _read_spool_tail(job.stderr_path, output_chars)

        # The job is still running; either the user cancelled or the
        # timeout fired. Send the signal first while status is still "running",
        # then update state after the signal has successfully dispatched.
        exit_code = await terminate(
            signal.SIGINT if interrupted else signal.SIGTERM, 2.0)
        
        with self._lock:
            job.status = "cancelled" if interrupted else "timed_out"
            self._write_metadata(job)
            
        self._record_exit(job, exit_code)
        status = "cancelled" if interrupted else "timed_out"
        return job, status, _read_spool_tail(job.stdout_path, output_chars), _read_spool_tail(job.stderr_path, output_chars)

    async def run_shell(self, command: str, timeout: int = None, description: str = "",
                        run_in_background: bool = False,
                        cancel_event: asyncio.Event | None = None) -> str:
        if command is None:
            return "Error: command is required"
        if command.strip() == "":
            return _format_bash_result("", "", 0, no_output_expected=True)

        timeout_ms = int(timeout) if timeout else BASH_DEFAULT_TIMEOUT_MS
        timeout_ms = min(timeout_ms, BASH_MAX_TIMEOUT_MS)

        if run_in_background:
            job = await self._spawn(command, command, description, True, None, True, cwd=current_cwd())
            try:
                asyncio.get_running_loop().create_task(self._monitor_background_job(job))
            except RuntimeError:
                # If no loop can accept the monitor task, the job still exists
                # with stdout/stderr/metadata paths for manual inspection.
                pass
            return "\n".join([
                f"Started background job {job.id}",
                f"pid: {job.pid}",
                f"pgid: {job.pgid}",
                f"status: {job.status}",
                f"stdout: {job.stdout_path}",
                f"stderr: {job.stderr_path}",
            ])

        job, status, stdout, stderr = await self.run_foreground(
            command, command, timeout_ms, description=description, shell=True,
            cwd=current_cwd(), cancel_event=cancel_event)
        if status == "cancelled":
            return _tool_result(
                False,
                "Tool call interrupted by user (SIGINT to the job's process "
                "group)" + (f"; partial output:\n{stdout}" if stdout else ""))
        if status == "timed_out":
            if stderr:
                stderr += "\n"
            stderr += f"command timed out after {timeout_ms}ms"
            return _format_bash_result(stdout, stderr, job.exit_code, status="timed_out")

        return _format_bash_result(stdout, stderr, job.exit_code)

    async def run_exec(self, argv: list[str], timeout_ms: int, description: str = "",
                       output_chars: int = BASH_MAX_OUTPUT_CHARS,
                       env: dict | None = None, cwd: str = None):
        if not argv:
            raise ValueError("argv must not be empty")
        return await self.run_foreground(argv, " ".join(argv), timeout_ms,
                                         description=description, shell=False,
                                         output_chars=output_chars, env=env, cwd=cwd or current_cwd())

    async def run_background_exec(self, argv: list[str], description: str = "",
                                  env: dict | None = None, cwd: str = None) -> Job:
        if not argv:
            raise ValueError("argv must not be empty")
        job = await self._spawn(argv, " ".join(argv), description, True, None, False,
                                env=env, cwd=cwd or current_cwd())
        asyncio.get_running_loop().create_task(self._monitor_background_job(job))
        return job

    def _get_job(self, job_id: str) -> Job | None:
        with self._lock:
            job = self.jobs.get(str(job_id))
        if job is not None:
            self._refresh_job(job)
        return job

    def list_jobs(self) -> str:
        with self._lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            self._refresh_job(job)
        if not jobs:
            return "No jobs."
        lines = ["Jobs:"]
        for job in jobs:
            lines.append(
                f"{job.id}. status={job.status} pid={job.pid} exit={job.exit_code} "
                f"started={job.started_at_iso} command={job.command!r}"
            )
        return "\n".join(lines)

    def job_status(self, job_id: str, tail_chars: int = JOB_TAIL_CHARS) -> str:
        job = self._get_job(job_id)
        if job is None:
            return f"Error: unknown job id {job_id!r}"
        stdout = _read_spool_tail(job.stdout_path, tail_chars)
        stderr = _read_spool_tail(job.stderr_path, tail_chars)
        return "\n".join([
            f"job_id: {job.id}",
            f"status: {job.status}",
            f"pid: {job.pid}",
            f"pgid: {job.pgid}",
            f"exit_code: {job.exit_code}",
            f"signal: {job.signal}",
            f"started_at: {job.started_at_iso}",
            f"finished_at: {job.finished_at_iso}",
            f"stdout_path: {job.stdout_path}",
            f"stderr_path: {job.stderr_path}",
            "[stdout_tail]",
            stdout if stdout else "(empty)",
            "[stderr_tail]",
            stderr if stderr else "(empty)",
        ])

    def stop_job(self, job_id: str, force: bool = False) -> str:
        job = self._get_job(job_id)
        if job is None:
            return f"Error: unknown job id {job_id!r}"
        if job.status != "running":
            return f"Job {job.id} is not running (status={job.status})."
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(job.pgid or job.pid, sig)
        except ProcessLookupError:
            return f"Job {job.id} is no longer running."
        with self._lock:
            job.status = "stopping"
            self._write_metadata(job)
        return f"Sent {sig.name} to job {job.id} (pgid={job.pgid})."




def run_bash(command: str, timeout: int = None, description: str = "",
              run_in_background: bool = False) -> str:
    return asyncio.run(run_bash_async(command, timeout, description,
                                      run_in_background))


async def run_bash_async(command: str, timeout: int = None, description: str = "",
                         run_in_background: bool = False,
                         cancel_event: asyncio.Event | None = None) -> str:
    return await current_job_manager().run_shell(command, timeout=timeout, description=description,
                                       run_in_background=run_in_background,
                                       cancel_event=cancel_event)


def run_jobs() -> str:
    return current_job_manager().list_jobs()


def run_job_status(job_id: str, tail_chars: int = JOB_TAIL_CHARS) -> str:
    if not job_id:
        return "Error: job_id is required"
    try:
        tail_chars = int(tail_chars)
    except (TypeError, ValueError) as e:
        return f"Error: invalid tail_chars: {e}"
    if tail_chars < 0:
        return "Error: tail_chars must be non-negative"
    return current_job_manager().job_status(job_id, tail_chars=tail_chars)


def run_job_stop(job_id: str, force: bool = False) -> str:
    if not job_id:
        return "Error: job_id is required"
    return current_job_manager().stop_job(job_id, force=bool(force))


def run_read(file_path: str, offset: int = None, limit: int = None) -> str:
    if not file_path:
        return "Error: file_path is required"
    file_path = _resolve_path(file_path)
    if os.path.isdir(file_path):
        return f"Error: {file_path} is a directory, not a file"
    try:
        st = os.stat(file_path)
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except IsADirectoryError:
        return f"Error: {file_path} is a directory"
    except Exception as e:
        return f"Error: {e}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        file_state[file_path] = None
        return f"File {file_path} is an image ({st.st_size} bytes). Visual content rendering not supported in this environment; reading the file is acknowledged."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read one character past the cap so the response can say whether
            # the file was actually truncated.
            content = f.read(READ_CHAR_CAP + 1)
        truncated_chars = len(content) > READ_CHAR_CAP
        if truncated_chars:
            content = content[:READ_CHAR_CAP]
    except UnicodeDecodeError:
        file_state[file_path] = None
        return f"File {file_path} is binary ({st.st_size} bytes); cannot display as text."
    except Exception as e:
        return f"Error reading file: {e}"

    if not content:
        file_state[file_path] = ""
        return f"File {file_path} is empty."

    if truncated_chars:
        content += f"\n\n[... file truncated at {READ_CHAR_CAP} characters; file is {st.st_size} bytes on disk -- pass offset/limit to read further]"

    lines = content.splitlines()
    total_lines = len(lines)
    start = int(offset) if offset is not None else 0
    if start < 0 or start >= total_lines:
        return f"Error: offset {offset} out of range (file has {total_lines} lines)"
    if limit is not None:
        lim = int(limit)
        if lim <= 0:
            return "Error: limit must be positive"
    else:
        lim = READ_DEFAULT_LINES
    sliced = lines[start:start + lim]
    rendered = _format_numbered_lines(sliced, first_line_number=start + 1)
    if start + lim < total_lines:
        rendered += f"\n... ({total_lines - start - lim} more lines not shown)"
    file_state[file_path] = content
    return rendered


def run_write(file_path: str, content: str) -> str:
    if not file_path:
        return "Error: file_path is required"
    if not content:
        return "Error: content is required"
    file_path = _resolve_path(file_path)
    existed = os.path.exists(file_path)
    if existed and file_path not in file_state:
        return (f"Error: You must Read {file_path} before overwriting it. "
                "Read it first, then retry the Write.")
    if existed:
        stale_error = _stale_file_error(file_path, "overwriting it")
        if stale_error:
            return stale_error
    try:
        _atomic_write_text(file_path, content)
        file_state[file_path] = content
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    if not file_path:
        return "Error: file_path is required"
    if old_string == new_string:
        return "Error: new_string must be different from old_string"
    file_path = _resolve_path(file_path)
    if file_path not in file_state:
        return f"Error: You must Read {file_path} before editing it."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error: {e}"

    stale_error = _stale_file_error(file_path, "editing it")
    if stale_error:
        return stale_error

    occurrences = data.count(old_string)
    if occurrences == 0:
        return f"Error: old_string not found in {file_path}."
    if occurrences > 1 and not replace_all:
        return (f"Error: old_string is not unique ({occurrences} occurrences) in {file_path}. "
                "Provide more context to make it unique, or pass replace_all=true.")

    if replace_all:
        new_data = data.replace(old_string, new_string)
        count = occurrences
    else:
        new_data = data.replace(old_string, new_string, 1)
        count = 1

    try:
        _atomic_write_text(file_path, new_data)
        file_state[file_path] = new_data
        return f"Successfully edited {file_path} ({count} replacement{'s' if count != 1 else ''})."
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str, path: str = None) -> str:
    return asyncio.run(run_glob_async(pattern, path))


async def run_glob_async(pattern: str, path: str = None) -> str:
    if not pattern:
        return "Error: pattern is required"
    rg = _find_rg_binary()
    if not rg:
        return "Error: ripgrep binary not found. Install rg."
    root = _resolve_path(path) if path else current_cwd()
    if not os.path.isdir(root):
        return f"Error: {root} is not a directory"
    args = [rg, '--files', '--color=never', '--glob', pattern, root]
    start = time.perf_counter()
    job, status, stdout, stderr = await current_job_manager().run_exec(
        args, SEARCH_TIMEOUT_S * 1000, description=f"Glob {pattern!r}")
    if status == "timed_out":
        return f"Error: ripgrep timed out after {SEARCH_TIMEOUT_S}s"
    duration_ms = int((time.perf_counter() - start) * 1000)
    stderr = stderr.strip()
    if job.exit_code not in [0, 1]:
        return f"Error: ripgrep failed with exit code {job.exit_code}" + (f"\n{stderr}" if stderr else "")
    matches = stdout.splitlines()
    num_files = len(matches)
    truncated = len(matches) > GLOB_MAX_RESULTS

    def mtime_or_zero(file_path: str) -> float:
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return 0

    # Put recently changed files first because coding tasks usually care about
    # active work more than lexicographic directory order.
    matches.sort(key=mtime_or_zero, reverse=True)
    matches = matches[:GLOB_MAX_RESULTS]
    if not matches:
        return f"No files matched pattern {pattern!r} in {root}"
    return "\n".join([
        f"duration_ms: {duration_ms}",
        f"num_files: {num_files}",
        f"truncated: {str(truncated).lower()}",
        "[filenames]",
        *matches,
    ])


def _find_rg_binary() -> str | None:
    return shutil.which('rg')


def _parse_nonnegative_int(value, name: str, default: int = None) -> tuple[int | None, str | None]:
    if value is None:
        return default, None
    try:
        number = int(value)
    except (TypeError, ValueError) as e:
        return None, f"Error: invalid {name}: {e}"
    if number < 0:
        return None, f"Error: {name} must be non-negative"
    return number, None


def _select_limited(lines: list[str], offset: int, head_limit: int) -> tuple[list[str], bool]:
    if head_limit == 0:
        return lines[offset:], False
    return lines[offset:offset + head_limit], len(lines) > offset + head_limit


def run_grep(pattern: str, path: str = None, glob: str = None,
             output_mode: str = "files_with_matches", **kwargs) -> str:
    return asyncio.run(run_grep_async(pattern, path, glob, output_mode, **kwargs))


async def run_grep_async(pattern: str, path: str = None, glob: str = None,
                         output_mode: str = "files_with_matches", **kwargs) -> str:
    if not pattern:
        return "Error: pattern is required"
    if output_mode not in ['content', 'files_with_matches', 'count']:
        return f"Error: invalid output_mode {output_mode!r}"
    rg = _find_rg_binary()
    if not rg:
        return "Error: ripgrep binary not found. Install rg."
    root = _resolve_path(path) if path else current_cwd()
    if not os.path.exists(root):
        return f"Error: {root} does not exist"

    head_limit, err = _parse_nonnegative_int(kwargs.get('head_limit'), 'head_limit', GREP_DEFAULT_HEAD_LIMIT)
    if err:
        return err
    offset_n, err = _parse_nonnegative_int(kwargs.get('offset'), 'offset', 0)
    if err:
        return err

    args = [rg, '--color=never']
    if output_mode == 'files_with_matches':
        args.append('--files-with-matches')
    elif output_mode == 'count':
        args.extend(['--count-matches', '--with-filename'])
    else:
        args.append('--with-filename')
        if kwargs.get('-n', True):
            args.append('--line-number')
        else:
            args.append('--no-line-number')
        if kwargs.get('-o'):
            args.append('--only-matching')
        context_value = kwargs.get('-C')
        if context_value is None:
            context_value = kwargs.get('context')
        if context_value is not None:
            context_value, err = _parse_nonnegative_int(context_value, '-C/context')
            if err:
                return err
            args.extend(['-C', str(context_value)])
        else:
            before, err = _parse_nonnegative_int(kwargs.get('-B'), '-B')
            if err:
                return err
            after, err = _parse_nonnegative_int(kwargs.get('-A'), '-A')
            if err:
                return err
            if before is not None:
                args.extend(['-B', str(before)])
            if after is not None:
                args.extend(['-A', str(after)])

    if kwargs.get('-i'):
        args.append('--ignore-case')
    if glob:
        args.extend(['--glob', glob])
    if kwargs.get('type'):
        args.extend(['--type', str(kwargs['type'])])
    if kwargs.get('multiline'):
        args.extend(['--multiline', '--multiline-dotall'])
    args.extend(['--', pattern, root])

    start = time.perf_counter()
    job, status, stdout, stderr = await current_job_manager().run_exec(
        args, SEARCH_TIMEOUT_S * 1000, description=f"Grep {pattern!r}")
    if status == "timed_out":
        return f"Error: ripgrep timed out after {SEARCH_TIMEOUT_S}s"
    duration_ms = int((time.perf_counter() - start) * 1000)
    stderr = stderr.strip()
    if job.exit_code not in [0, 1]:
        return f"Error: ripgrep failed with exit code {job.exit_code}" + (f"\n{stderr}" if stderr else "")

    lines = stdout.splitlines()
    selected, truncated = _select_limited(lines, offset_n, head_limit)
    if not selected:
        return f"No matches for {pattern!r}"
    result = "\n".join([
        f"mode: {output_mode}",
        f"duration_ms: {duration_ms}",
        f"num_entries: {len(lines)}",
        f"applied_offset: {offset_n}",
        f"applied_limit: {head_limit}",
        f"truncated: {str(truncated).lower()}",
        "[results]",
        *selected,
    ])
    return _truncate_text(result, BASH_MAX_OUTPUT_CHARS)


def run_todoread() -> str:
    if not current_todos():
        return "No todos for this session."
    out = ["Todos:"]
    for i, t in enumerate(current_todos(), start=1):
        out.append(f"  {i}. [{t['status']}] ({t['priority']}) {t['content']}")
    return "\n".join(out)


def run_todowrite(todos: list) -> str:
    if not isinstance(todos, list):
        return "Error: todos must be an array"
    if len(todos) > TODO_MAX_TODOS:
        return f"Error: too many todos (max {TODO_MAX_TODOS})"
    in_progress = sum(1 for t in todos if t.get('status') == 'in_progress')
    if in_progress > 1:
        return "Error: At most one todo can be in_progress at a time"
    cleaned = []
    for t in todos:
        content = (t.get('content') or '').strip()
        status = t.get('status')
        priority = t.get('priority')
        if not content:
            return "Error: each todo requires non-empty content"
        if status not in ['pending', 'in_progress', 'completed']:
            return f"Error: invalid status {status!r}"
        if priority not in ['high', 'medium', 'low']:
            return f"Error: invalid priority {priority!r}"
        cleaned.append({'content': content, 'status': status, 'priority': priority})
    current_session().session_todos = cleaned
    mark_chat_log_dirty()
    summary = {'total': len(cleaned), 'pending': sum(1 for t in cleaned if t['status'] == 'pending'),
               'in_progress': in_progress,
               'completed': sum(1 for t in cleaned if t['status'] == 'completed')}
    return f"Updated todos: {summary}"


def _handle_bash(args: dict) -> str:
    return run_bash(args["command"],
                    timeout=args.get("timeout"),
                    description=args.get("description"),
                    run_in_background=args.get("run_in_background", False))


async def _handle_bash_async(args: dict, extra_context=None) -> str:
    return await run_bash_async(args["command"],
                                timeout=args.get("timeout"),
                                description=args.get("description", ""),
                                run_in_background=args.get("run_in_background", False),
                                cancel_event=(extra_context or {}).get("cancel_event")
                                if isinstance(extra_context, dict) else None)


def _handle_read(args: dict) -> str:
    return run_read(args["file_path"], offset=args.get("offset"), limit=args.get("limit"))


def _handle_write(args: dict) -> str:
    return run_write(args["file_path"], args["content"])


def _handle_edit(args: dict) -> str:
    return run_edit(args["file_path"], args["old_string"],
                    args["new_string"], replace_all=args.get("replace_all", False))


def _handle_glob(args: dict) -> str:
    return run_glob(args["pattern"], args.get("path"))


async def _handle_glob_async(args: dict, extra_context=None) -> str:
    return await run_glob_async(args["pattern"], args.get("path"))


def _handle_grep(args: dict) -> str:
    return run_grep(args["pattern"],
                    path=args.get("path"),
                    glob=args.get("glob"),
                    output_mode=args.get("output_mode", "files_with_matches"),
                    **{k: v for k, v in args.items()
                       if k in ["-B", "-A", "-C", "context", "-n", "-i", "-o",
                                "type", "head_limit", "offset", "multiline"]})


async def _handle_grep_async(args: dict, extra_context=None) -> str:
    extra = {k: v for k, v in args.items()
             if k not in ["pattern", "path", "glob", "output_mode"]}
    return await run_grep_async(args["pattern"],
                                path=args.get("path"),
                                glob=args.get("glob"),
                                output_mode=args.get("output_mode", "files_with_matches"),
                                **extra)


def _handle_jobs(args: dict) -> str:
    return run_jobs()


def _handle_job_status(args: dict) -> str:
    return run_job_status(args["job_id"], args.get("tail_chars", JOB_TAIL_CHARS))


def _handle_job_stop(args: dict) -> str:
    return run_job_stop(args["job_id"], args.get("force", False))


def _handle_todoread(args: dict) -> str:
    return run_todoread()


def _handle_todowrite(args: dict) -> str:
    return run_todowrite(args["todos"])


def _handle_agent(args: dict) -> str:
    return run_agent(args["description"],
                     args["prompt"],
                     run_in_background=args.get("run_in_background", False),
                     subagent_type=args.get("subagent_type", "Explore"))


async def _handle_agent_async(args: dict, extra_context=None) -> str:
    return await run_agent_async(args.get("description", ""),
                                 args["prompt"],
                                 args.get("run_in_background", False),
                                 args.get("subagent_type", "Explore"))


def _handle_skill(args: dict) -> str:
    return run_skill(args["skill"], args.get("args"))


async def _handle_webfetch_async(args: dict, extra_context=None) -> str:
    return await run_webfetch_async(args["url"], args["prompt"])


async def _handle_websearch_async(args: dict, extra_context=None) -> str:
    return await run_websearch_async(args["query"],
                                     allowed_domains=args.get("allowed_domains"),
                                     blocked_domains=args.get("blocked_domains"))


def _tool_result(ok: bool, content) -> dict:
    return {"ok": ok, "content": str(content)}


def _looks_like_tool_error(content: str) -> bool:
    return content.startswith("Error: ") or content.startswith("Failed")


async def with_exception_to_tool_result_async(context: str, thunk) -> dict:
    try:
        content = await thunk()
    except (KeyboardInterrupt, SystemExit):
        raise
    except FileNotFoundError as e:
        return _tool_result(False, f"Error while {context}: file not found: {e}")
    except PermissionError as e:
        return _tool_result(False, f"Error while {context}: permission denied: {e}")
    except TimeoutError as e:
        return _tool_result(False, f"Error while {context}: timed out: {e}")
    except OSError as e:
        return _tool_result(False, f"Error while {context}: OS error: {e}")
    except ValueError as e:
        return _tool_result(False, f"Error while {context}: invalid value: {e}")
    except Exception as e:
        return _tool_result(False, f"Failed while {context}: {type(e).__name__}: {e}")

    text = str(content)
    return _tool_result(not _looks_like_tool_error(text), text)


def _tool_access_error(fn_name: str, allowed=None, extra_context=None):
    inhibit_edits = (extra_context or {}).get('inhibit_edits', False)
    spec = TOOL_REGISTRY.get(fn_name)
    if spec is None:
        return f"Unknown function: {fn_name}"
    if allowed is not None and fn_name not in allowed:
        return (
            f"Tool {fn_name} not available in this subagent "
            f"(allowed: {sorted(allowed)})")
    if inhibit_edits and spec.get('explore') is not True:
        return (
            f"Not allowed to {fn_name} instead of doing the following: "
            f"{inhibit_edits}")
    return None


async def dispatch_tool_async(fn_name: str, args: dict, allowed=None, extra_context=None) -> dict:
    spec = TOOL_REGISTRY.get(fn_name)
    access_error = _tool_access_error(
        fn_name, allowed=allowed, extra_context=extra_context)
    if access_error:
        return _tool_result(False, access_error)

    async def run_handler():
        if spec.get("async_handler") is not None:
            return await spec["async_handler"](
                args, extra_context=extra_context)
        return spec["handler"](args)

    return await with_exception_to_tool_result_async(f"executing {fn_name}", run_handler)


def _raw_tool_arguments(call):
    if "input" in call:
        return copy.deepcopy(call["input"])
    return call.get("arguments", "{}")


def _runtime_provider_label():
    if current_config() is None:
        return None
    return (
        current_config().provider_id
        or current_config().provider_name
        or current_config().netloc
    )


def _repair_event(invocation):
    return {
        "type": "tool_input_repaired",
        "name": invocation.tool_name,
        "call_id": invocation.call_id,
        "model": invocation.model,
        "provider": invocation.provider,
        "repairs": [
            {
                "hook": adjustment.hook,
                "rule": adjustment.rule,
                "path": list(adjustment.path),
                "display_path": tool_runtime.format_path(
                    adjustment.path),
            }
            for adjustment in invocation.adjustments
        ],
    }


def _execution_metadata(invocation, outcome, defaults):
    if not (invocation.adjustments
            or invocation.hook_records
            or defaults
            or invocation.denied_by
            or invocation.changed_paths
            or invocation.invalidate_all_files):
        return None
    result = {
        "executed": outcome.executed,
        "status": outcome.status,
    }
    if invocation.adjustments:
        adjustment_key = (
            "adjustments"
            if outcome.executed else "attempted_adjustments")
        result[adjustment_key] = [
            adjustment.to_dict()
            for adjustment in invocation.adjustments
        ]
    if invocation.hook_records:
        result["hooks"] = [
            record.to_dict() for record in invocation.hook_records
        ]
    if defaults:
        result["defaults"] = copy.deepcopy(defaults)
    if invocation.denied_by:
        result["denied_by"] = invocation.denied_by
    if invocation.changed_paths:
        result["changed_paths"] = list(invocation.changed_paths)
    if invocation.invalidate_all_files:
        result["invalidated_all_file_state"] = True
    return result


def _invalidate_hook_file_state(invocation):
    if invocation.invalidate_all_files:
        file_state.clear()
        return
    for path in invocation.changed_paths:
        file_state.pop(_resolve_path(path, invocation.cwd), None)


def _read_default_notes(invocation):
    if invocation.tool_name != "Read":
        return [], []
    args = invocation.effective_arguments
    if not isinstance(args, dict):
        return [], []
    has_offset = "offset" in args
    has_limit = "limit" in args
    if has_offset == has_limit:
        return [], []
    if has_offset:
        return (
            ["Note: limit was omitted; Read used the default of "
             f"{READ_DEFAULT_LINES} lines."],
            [{"field": "limit", "value": READ_DEFAULT_LINES}],
        )
    return (
        ["Note: offset was omitted; Read started at line offset 0."],
        [{"field": "offset", "value": 0}],
    )


def _prepend_tool_notes(content, notes):
    useful = [str(note).strip() for note in notes if str(note).strip()]
    if not useful:
        return str(content)
    return "\n\n".join(useful + [str(content)])


async def execute_tool_call_async(
        call, allowed=None, extra_context=None, on_event=None,
        hook_pipeline=None):
    if on_event is None:
        on_event = lambda event: None
    if hook_pipeline is None:
        hook_pipeline = TOOL_HOOK_PIPELINE
    fn_name = formats.tool_call_name(call)
    spec = TOOL_REGISTRY.get(fn_name)
    if spec is None:
        result = _tool_result(False, f"Unknown function: {fn_name}")
        on_event({
            "type": "tool_rejected",
            "name": fn_name,
            "args": _raw_tool_arguments(call),
        })
        return result, None

    parse_error = call.get("parse_error")
    try:
        original_args = formats.tool_call_input(call)
    except formats.TranscriptFormatError as error:
        original_args = None
        if parse_error is None:
            parse_error = str(error)
    invocation = tool_runtime.ToolInvocation(
        call_id=formats.tool_call_id(call),
        tool_name=fn_name,
        raw_arguments=_raw_tool_arguments(call),
        original_arguments=copy.deepcopy(original_args),
        effective_arguments=copy.deepcopy(original_args),
        schema=spec["schema"],
        semantics=spec.get("semantics", {}),
        cwd=current_cwd(),
        model=current_model() or None,
        provider=_runtime_provider_label(),
        parse_error=parse_error,
    )

    access_error = _tool_access_error(
        fn_name, allowed=allowed, extra_context=extra_context)
    if access_error:
        invocation.denied_reason = access_error
        invocation.denied_by = "loki.capability-gate"
        outcome = tool_runtime.ToolOutcome(
            "rejected", False, False, access_error)
    else:
        invocation = await hook_pipeline.prepare(invocation)
        if invocation.denied_reason:
            outcome = tool_runtime.ToolOutcome(
                "rejected",
                False,
                False,
                invocation.denied_reason,
            )
        elif invocation.validation_issues:
            on_event({
                "type": "tool_input_invalid",
                "name": fn_name,
                "call_id": invocation.call_id,
                "model": invocation.model,
                "provider": invocation.provider,
                "issues": [
                    {
                        "code": issue.code,
                        "path": list(issue.path),
                        "display_path": tool_runtime.format_path(
                            issue.path),
                    }
                    for issue in invocation.validation_issues
                ],
            })
            outcome = tool_runtime.ToolOutcome(
                "invalid",
                False,
                False,
                tool_runtime.invalid_input_message(
                    fn_name, invocation.validation_issues),
            )
        else:
            if invocation.adjustments:
                on_event(_repair_event(invocation))
            on_event({
                "type": "tool_call",
                "name": fn_name,
                "args": copy.deepcopy(
                    invocation.effective_arguments),
            })
            result = await dispatch_tool_async(
                fn_name,
                invocation.effective_arguments,
                allowed=allowed,
                extra_context=extra_context,
            )
            outcome = tool_runtime.ToolOutcome(
                "success" if result["ok"] else "tool_error",
                True,
                result["ok"],
                result["content"],
            )

    default_notes, defaults = _read_default_notes(invocation)
    if outcome.executed:
        if invocation.adjustments:
            invocation.notes.insert(
                0,
                "Note: Loki repaired or adjusted tool input before "
                "execution:\n"
                + "\n".join(
                    "- " + tool_runtime.adjustment_summary(adjustment)
                    for adjustment in invocation.adjustments),
            )
        invocation.notes.extend(default_notes)
    else:
        defaults = []
    invocation, outcome = await hook_pipeline.finish(
        invocation, outcome)
    _invalidate_hook_file_state(invocation)
    outcome.content = _prepend_tool_notes(
        outcome.content, invocation.notes)
    if not outcome.executed:
        on_event({
            "type": "tool_rejected",
            "name": fn_name,
            "args": copy.deepcopy(
                invocation.effective_arguments),
        })
    result = _tool_result(outcome.ok, outcome.content)
    return result, _execution_metadata(
        invocation, outcome, defaults)


def get_tool_loop_extra_context(transcript_items: list):
    inhibit_edits = False
    # "explore" mode: read-only. Force the explore-only tool gate so the agent
    # can't write/edit while investigating.
    if current_agent_mode() == "explore":
        inhibit_edits = "explore mode"
    if len(transcript_items) > 0 and transcript_items[-1].get("type") == "message" and transcript_items[-1].get("role") == "user":
        content = transcript_items[-1].get('content')
        content = [x for x in content if x.get('type') == 'text'][-1:]
        if len(content) > 0:
            text = content[-1].get("text")
            if (text and (
                    text.strip().endswith("?")
                    or "what?" in text.strip().lower())):
                inhibit_edits = "answering the user's question"

    return {'inhibit_edits': inhibit_edits}


async def run_tool_loop_async(transcript_items: list, allowed=None, max_loops=MAX_LOOP_LIMIT,
                              chat_fn=None, on_event=None, cancel_check=None,
                              stream_chat=False, report_timing=False,
                              on_response=None,
                              hook_pipeline=None,
                              cancel_event: asyncio.Event | None = None) -> str:
    """Run the model/tool loop over canonical session events.

    Mutates ``transcript_items`` by appending one ``model_response`` for each
    actual provider response and one event for each local tool result.
    """
    if chat_fn is None:
        advertised_tools = (
            TOOLS if allowed is None else [
                spec["definition"]
                for name, spec in TOOL_REGISTRY.items()
                if name in allowed
            ])
        chat_fn = lambda items: async_chat_completion(
            items, tools=advertised_tools)
    if on_event is None:
        on_event = lambda event: None
    if cancel_check is None:
        cancel_check = lambda: False
    if on_response is None:
        on_response = lambda turn, event: None
    if max_loops < 1:
        raise ValueError("max_loops must be at least 1")

    tool_loop_extra_context = get_tool_loop_extra_context(transcript_items)
    if cancel_event is not None:
        tool_loop_extra_context["cancel_event"] = cancel_event

    def append_turn(turn):
        response_event = turn.to_event()
        transcript_items.append(response_event)
        on_response(turn, response_event)

    def append_unexecuted_results(tool_calls, reason):
        for call in tool_calls:
            transcript_items.append(formats.tool_result_for_call(
                call, reason, is_error=True))

    loop_count = 0
    while True:
        loop_count += 1
        live_text_started = False
        live_text_parts = []

        def on_text_delta(text):
            nonlocal live_text_started
            if not text:
                return
            live_text_parts.append(text)
            if not live_text_started:
                live_text_started = True
                on_event({"type": "assistant_start"})
            on_event({"type": "assistant_delta", "content": text})

        def finish_live_text(complete, reason=None):
            if live_text_started:
                on_event({
                    "type": "assistant_end",
                    "complete": complete,
                    "reason": reason,
                })

        request_start = time.perf_counter()
        try:
            if stream_chat:
                response_items = await chat_fn(
                    transcript_items, on_text_delta)
            else:
                response_items = await chat_fn(transcript_items)
        except StreamCancelled:
            finish_live_text(False, "cancelled")
            on_event({
                "type": "response_cancelled",
                "partial": live_text_started,
                "saved": False,
            })
            return ""
        except protocols.StreamProtocolError as e:
            finish_live_text(False, "error")
            on_event({"type": "stream_error", "error": e})
            return ""
        except (formats.TranscriptFormatError, protocols.ProtocolError) as e:
            finish_live_text(False, "error")
            on_event({"type": "transcript_error", "error": e})
            return ""
        except ApiError as e:
            finish_live_text(False, "error")
            on_event({"type": "api_error", "error": e})
            return ""
        except OSError as e:
            finish_live_text(False, "error")
            msg = str(e) or f"{type(e).__name__}() (errno={e.errno!r}, no OS message)"
            on_event({"type": "network_error", "error": msg})
            return ""
        turn = formats.coerce_decoded_turn(response_items)
        finish_live_text(turn.complete, None if turn.complete else "incomplete")
        if report_timing:
            on_event({
                "type": "response_timing",
                "elapsed": time.perf_counter() - request_start,
            })
        append_turn(turn)

        tool_calls = formats.response_tool_calls(turn.items)
        assistant_items = [
            item for item in turn.items
            if item.get("type") == "message" and item.get("role") == "assistant"
        ]
        assistant_text = "\n".join(
            text for text in (
                formats.item_text(item) for item in assistant_items)
            if text)
        if assistant_text and not live_text_started:
            on_event({"type": "assistant_message", "content": assistant_text})

        if not turn.complete:
            if tool_calls:
                append_unexecuted_results(
                    tool_calls,
                    "Tool call not executed because the provider response "
                    "was incomplete.",
                )
            event_type = (
                "response_failed"
                if turn.metadata.get("status") == "failed"
                else "response_incomplete")
            on_event({
                "type": event_type,
                "status": turn.metadata.get("status"),
                "protocol_data": copy.deepcopy(
                    turn.metadata.get("protocol_data")),
            })
            return assistant_text

        if (turn.metadata.get("stop_reason") == "pause_turn"
                and not tool_calls):
            if loop_count >= max_loops:
                on_event({"type": "max_loops"})
                return assistant_text
            # Anthropic server tools require the paused assistant response to
            # be sent back unchanged in the next request. No synthetic user or
            # tool-result event belongs between the two provider responses.
            continue

        if not turn.items:
            return ""
        if not assistant_items and not tool_calls:
            return ""

        if not tool_calls:
            return assistant_text

        if cancel_check():
            reason = "Tool call not executed because the user cancelled the turn."
            append_unexecuted_results(tool_calls, reason)
            on_event({
                "type": "response_cancelled",
                "partial": live_text_started,
                "saved": True,
            })
            return assistant_text

        if loop_count >= max_loops:
            reason = (
                f"Tool call not executed because Loki reached its "
                f"{max_loops}-response autonomous loop limit.")
            append_unexecuted_results(tool_calls, reason)
            on_event({"type": "max_loops"})
            return assistant_text

        for call_index, tc in enumerate(tool_calls):
            result, execution = await execute_tool_call_async(
                tc,
                allowed=allowed,
                extra_context=tool_loop_extra_context,
                on_event=on_event,
                hook_pipeline=hook_pipeline,
            )
            if not result["ok"]:
                on_event({"type": "tool_error", "result": result["content"]})
            result_item = formats.tool_result_for_call(
                tc,
                result["content"],
                is_error=not result["ok"],
                execution=execution,
            )
            transcript_items.append(result_item)
            if cancel_check():
                append_unexecuted_results(
                    tool_calls[call_index + 1:],
                    "Tool call not executed because the user cancelled "
                    "the turn.",
                )
                on_event({
                    "type": "response_cancelled",
                    "partial": False,
                    "saved": True,
                })
                return assistant_text


def _print_tool_args(args):
    if not isinstance(args, dict):
        pprint(args)
        return
    for k, v in args.items():
        pprint((k, v))


def _print_terminal_fragments(fragments):
    for fragment in fragments:
        print(fragment, end='', flush=True)


def _terminal_agent_event(event: dict):
    # Error branches reset attributes before emitting their final newline. That
    # prevents terminal scroll-fill from inheriting the red background.
    kind = event.get("type")
    if kind == "max_loops":
        print("\n[!] [Max Loop Limit Reached - Stopping Autonomous Execution]")
    elif kind == "api_error":
        terminal.set_background_color(ERROR_COLOR)
        print(event["error"].formatted(), end='')
        terminal.reset_colors_and_flags()
        print()
    elif kind == "network_error":
        print(f"\n{computer}: NETWORK ERROR: {event['error']}")
    elif kind == "transcript_error":
        terminal.set_background_color(ERROR_COLOR)
        error = event["error"]
        print(f"Transcript render error: {error}", end='')
        terminal.reset_colors_and_flags()
        print()
        sys.stdout.flush()
        payload = getattr(error, "payload", None)
        if payload is not None:
            print(
                "Provider payload:\n"
                + json.dumps(
                    payload, ensure_ascii=False, sort_keys=True,
                    default=str),
                file=sys.stderr,
            )
            sys.stderr.flush()
    elif kind == "assistant_message":
        rendered_content = terminals.render_markdown(event["content"])
        print(f"\n{current_model()}: {rendered_content}")
    elif kind == "assistant_start":
        was_active = terminal.assistant_markdown.active
        stale = terminal.assistant_markdown.start()
        if was_active:
            _print_terminal_fragments(stale)
        print(f"\n{current_model()}: ", end='', flush=True)
    elif kind == "assistant_delta":
        _print_terminal_fragments(
            terminal.assistant_markdown.feed(event["content"]))
    elif kind == "assistant_end":
        _print_terminal_fragments(terminal.assistant_markdown.finish())
        print()
        sys.stdout.flush()
    elif kind == "response_timing":
        sys.stdout.flush()
        print(
            f"\n[T]  [LLM Response Time: {event['elapsed']:.3f}s]",
            file=sys.stderr)
        sys.stderr.flush()
    elif kind == "response_cancelled":
        sys.stdout.flush()
        detail = ""
        if event.get("partial"):
            detail = (
                "; partial response saved"
                if event.get("saved")
                else "; partial transport output was not added to history")
        print(f"[model response cancelled{detail}]", file=sys.stderr)
        sys.stderr.flush()
    elif kind == "response_incomplete":
        sys.stdout.flush()
        detail = event.get("protocol_data")
        suffix = (
            "\n" + json.dumps(
                detail, ensure_ascii=False, sort_keys=True, default=str)
            if detail else "")
        print(
            "[model response incomplete; provider output saved]"
            + suffix,
            file=sys.stderr)
        sys.stderr.flush()
    elif kind == "response_failed":
        sys.stdout.flush()
        detail = event.get("protocol_data")
        suffix = (
            "\n" + json.dumps(
                detail, ensure_ascii=False, sort_keys=True, default=str)
            if detail else "")
        print("[model response failed; provider output saved]" + suffix,
              file=sys.stderr)
        sys.stderr.flush()
    elif kind == "stream_error":
        error = event["error"]
        terminal.set_background_color(ERROR_COLOR)
        print(
            f"Streaming response error: {error}\n"
            "Set LOKI_STREAM=0 to disable streaming for this connection.",
            end='')
        terminal.reset_colors_and_flags()
        print()
        sys.stdout.flush()
        payload = getattr(error, "payload", None)
        if payload is not None:
            print(
                "Provider payload:\n"
                + json.dumps(
                    payload, ensure_ascii=False, sort_keys=True,
                    default=str),
                file=sys.stderr,
            )
            sys.stderr.flush()
    elif kind == "tool_input_repaired":
        terminal.set_foreground_color(TOOL_CALL_COLOR)
        print(f"{computer}: Repaired Tool Input: {event['name']}")
        for repair in event["repairs"]:
            print(
                f"  {repair['display_path']}: "
                f"{repair['rule'].replace('_', ' ')}")
        terminal.reset_colors_and_flags()
    elif kind == "tool_call":
        terminal.set_foreground_color(TOOL_CALL_COLOR)
        print(f"{computer}: Executing Tool: {event['name']} with args:")
        _print_tool_args(event["args"])
        terminal.reset_colors_and_flags()
    elif kind == "tool_rejected":
        terminal.set_foreground_color(TOOL_CALL_COLOR)
        print(f"{computer}: Rejected Tool: {event['name']} with invalid args:")
        _print_tool_args(event["args"])
        terminal.reset_colors_and_flags()
    elif kind == "tool_error":
        terminal.set_background_color(ERROR_COLOR)
        print(event["result"], end='')
        terminal.reset_colors_and_flags()
        print()


async def run_terminal_turn_async(transcript_items: list, cancel_check=None,
                                  cancel_event: asyncio.Event | None = None) -> str:
    async def chat_fn(items, on_text_delta):
        return await async_chat_completion(
            items, TOOLS, True, False,
            on_text_delta=on_text_delta,
            cancel_check=cancel_check)

    return await run_tool_loop_async(
        transcript_items,
        chat_fn=chat_fn,
        on_event=_terminal_agent_event,
        cancel_check=cancel_check,
        stream_chat=True,
        report_timing=True,
        cancel_event=cancel_event,
        on_response=lambda turn, event: _remember_session_toolset(TOOLS),
    )


async def run_toolless_completion_async(transcript_items: list) -> str:
    try:
        response_items = await async_chat_completion(transcript_items, tools=[])
    except (formats.TranscriptFormatError, protocols.ProtocolError) as e:
        return f"Transcript render error: {e}"
    except ApiError as e:
        return e.formatted()
    turn = formats.coerce_decoded_turn(response_items)
    if not turn:
        return ""
    return "\n".join(
        text for text in (
            formats.item_text(item).strip()
            for item in turn.items
            if (item.get("type") == "message"
                and item.get("role") == "assistant"))
        if text)


def _subprocess_stream_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return str(value)


def _format_subagent_result(agent_type: str, description: str, status: str,
                            exit_code: int | None, stdout, stderr) -> str:
    stdout_text = _subprocess_stream_text(stdout).strip()
    stderr_text = _subprocess_stream_text(stderr).strip()
    parts = [
        f"[{agent_type} subagent: {description or 'subagent task'}]",
        f"status: {status}",
    ]
    if exit_code is not None:
        parts.append(f"exit_code: {exit_code}")
    parts.extend([
        "[stdout]",
        stdout_text if stdout_text else "(empty)",
        "[stderr]",
        stderr_text if stderr_text else "(empty)",
    ])
    return "\n".join(parts)


def _subagent_env() -> dict:
    env = os.environ.copy()
    # Parent startup consumes provider-specific key variables. Subagents receive
    # only the normalized runtime provider/url/model/key they should use.
    if current_config():
        env['LOKI_PROVIDER'] = current_config().chat_provider.kind
        env['LOKI_API_BASE'] = current_config().chat_provider.input_url
        env['LOKI_MODEL'] = current_model()
        if current_config().api_key:
            env['LOKI_API_KEY'] = current_config().api_key
        else:
            env.pop('LOKI_API_KEY', None)
        env['LOKI_STREAM'] = '1' if current_config().stream else '0'
        env['LOKI_PROMPT_CACHE'] = (
            '1' if current_config().prompt_cache else '0')
    return env


def _format_started_background_job(job: Job, kind: str = "job") -> str:
    return "\n".join([
        f"Started background {kind} {job.id}",
        f"pid: {job.pid}",
        f"pgid: {job.pgid}",
        f"status: {job.status}",
        f"stdout: {job.stdout_path}",
        f"stderr: {job.stderr_path}",
    ])


def _current_entrypoint_argv() -> list[str]:
    argv0 = sys.argv[0]
    if argv0 and os.path.basename(argv0) != "__main__.py":
        script = shutil.which(argv0) if not os.path.dirname(argv0) else argv0
        script = script or argv0
        if not os.path.isabs(script):
            script = os.path.join(STARTUP_CWD, script)
        return [sys.executable, os.path.abspath(script)]
    # python -m loki_agent sets sys.argv[0] to loki_agent/__main__.py. Running
    # that file directly would lose package context and break relative imports.
    return [sys.executable, "-m", "loki_agent"]


def _subagent_argv(agent_type: str, prompt: str) -> list[str]:
    return _current_entrypoint_argv() + [
        '--subagent',
        agent_type,
        '--prompt',
        prompt,
    ]


def run_agent(description: str, prompt: str, run_in_background: bool = False,
              subagent_type: str = "Explore") -> str:
    return asyncio.run(run_agent_async(description, prompt, run_in_background, subagent_type))


async def run_agent_async(description: str, prompt: str, run_in_background: bool = False,
                          subagent_type: str = "Explore") -> str:
    agent_type = subagent_type or "Explore"
    if not prompt:
        return "Error: prompt is required"
    if agent_type != "Explore":
        return f"Error: unknown subagent_type {agent_type!r} (only 'Explore' is supported)"
    argv = _subagent_argv(agent_type, prompt)
    if run_in_background:
        job = await current_job_manager().run_background_exec(
            argv,
            description=description or "subagent task",
            env=_subagent_env())
        return _format_started_background_job(job, "subagent")

    job, status, stdout, stderr = await current_job_manager().run_exec(
        argv, SUBAGENT_TIMEOUT_S * 1000,
        description=description or "subagent task",
        env=_subagent_env())
    if status == "timed_out":
        result = _format_subagent_result(agent_type, description, "timed_out",
                                         job.exit_code, stdout, stderr)
        return f"Error: subagent timed out after {SUBAGENT_TIMEOUT_S}s for {description or 'task'}\n{result}"
    result = _format_subagent_result(agent_type, description, "completed",
                                     job.exit_code, stdout, stderr)
    if job.exit_code != 0:
        return f"Error: subagent exited with code {job.exit_code}\n{result}"
    return result


def run_skill(skill: str, args: str = None) -> str:
    if not skill:
        return "Error: skill is required"
    skill_root = os.path.join(LOKI_CONFIG_DIR, "skills")
    skill_path = os.path.join(skill_root, skill, "SKILL.md")
    if not os.path.isfile(skill_path):
        return (f"Error: skill {skill!r} not found. Available skills can be discovered by listing "
                f"{skill_root}.")
    try:
        with open(skill_path, 'r', encoding='utf-8') as f:
            content = f.read(SKILL_MAX_BYTES)
    except Exception as e:
        return f"Error loading skill: {e}"
    truncated = len(content) >= SKILL_MAX_BYTES
    base_dir = os.path.dirname(skill_path)
    header = f"<skill_content name=\"{skill}\">\n# Skill: {skill}\n\n"
    body = content
    if args:
        body = f"Args: {args}\n\n{body}"
    footer = ("\n\n[Skill content truncated]" if truncated else "") + \
             f"\n\nBase directory for this skill: {base_dir}\n" \
             "Relative paths in this skill are relative to this base directory.\n</skill_content>"
    return header + body + footer


HTML_TEXT_BLOCK_TAGS = {
    'address', 'article', 'aside', 'blockquote', 'br', 'dd', 'details', 'div',
    'dl', 'dt', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3',
    'h4', 'h5', 'h6', 'header', 'hr', 'li', 'main', 'nav', 'ol', 'p', 'pre',
    'section', 'table', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'ul',
}
HTML_TEXT_SKIP_TAGS = {'script', 'style', 'template', 'noscript', 'svg'}


class HtmlTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0

    def _newline(self):
        if self.parts and self.parts[-1] != '\n':
            self.parts.append('\n')

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in HTML_TEXT_SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in HTML_TEXT_BLOCK_TAGS:
            self._newline()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in HTML_TEXT_SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in HTML_TEXT_BLOCK_TAGS:
            self._newline()

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = ' '.join(data.split())
        if text:
            self.parts.append(text)

    def text(self) -> str:
        lines = []
        current = []
        for part in self.parts:
            if part == '\n':
                if current:
                    lines.append(' '.join(current))
                    current = []
            else:
                current.append(part)
        if current:
            lines.append(' '.join(current))
        return '\n'.join(lines).strip()


def _html_to_text(html: str) -> str:
    parser = HtmlTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def _decode_http_text(raw: bytes, headers, default_charset: str = 'utf-8') -> str:
    candidates = []
    get_content_charset = getattr(headers, 'get_content_charset', None)
    if callable(get_content_charset):
        charset = get_content_charset()
        if charset:
            candidates.append(charset)
    elif isinstance(headers, dict):
        content_type = headers.get('content-type') or headers.get('Content-Type') or ''
        for part in content_type.split(';')[1:]:
            name, sep, value = part.strip().partition('=')
            if sep and name.lower() == 'charset':
                candidates.append(value.strip('"'))
    candidates.append(default_charset)

    seen = set()
    for charset in candidates:
        charset = charset.strip()
        if not charset or charset.lower() in seen:
            continue
        seen.add(charset.lower())
        try:
            return raw.decode(charset)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode(default_charset, errors='replace')


def _content_media_type(content_type: str) -> str:
    return content_type.split(';', 1)[0].strip().lower()


def _is_html_content_type(content_type: str) -> bool:
    return _content_media_type(content_type) in {'text/html', 'application/xhtml+xml'}


def _decode_duckduckgo_result_url(raw_href: str) -> str:
    parsed = urllib.parse.urlparse(raw_href)
    query = urllib.parse.parse_qs(parsed.query)
    return query.get('uddg', [raw_href])[0]


class DuckDuckGoResultParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._href = None
        self._text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'a' or self._href is not None:
            return
        attr_map = {name.lower(): value or '' for name, value in attrs}
        classes = attr_map.get('class', '').split()
        href = attr_map.get('href')
        if href and 'result__a' in classes:
            self._href = href
            self._text_parts = []

    def handle_data(self, data):
        if self._href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() != 'a' or self._href is None:
            return
        title = ' '.join(''.join(self._text_parts).split())
        if title:
            self.results.append({
                'title': title,
                'url': _decode_duckduckgo_result_url(self._href),
            })
        self._href = None
        self._text_parts = []


def _parse_duckduckgo_results(html: str) -> list[dict]:
    parser = DuckDuckGoResultParser()
    parser.feed(html)
    parser.close()
    return parser.results


async def _fetch_url_async(url: str) -> dict:
    """GET a URL with redirect tracking, return dict with content/contentType/status/finalUrl/redirects.
    HTTP is upgraded to HTTPS. Cross-host redirects are surfaced, not followed."""
    if url.startswith('http://'):
        url = 'https://' + url[len('http://'):]
    elif not url.startswith('https://'):
        url = 'https://' + url

    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return {'error': f'invalid URL: {url}'}

    request_headers = {
        'User-Agent': 'loki-WebFetch/0.1 (coding-agent)',
        'Accept': 'text/markdown;q=1.0, text/html;q=0.9, text/plain;q=0.8, application/json;q=0.7, */*;q=0.1',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        response = await http_client.async_http_request_follow_same_host(
            'GET', url, headers_in=request_headers,
            timeout=WEBFETCH_TIMEOUT_S, max_bytes=WEBFETCH_MAX_BYTES,
            retry_max_attempts=HTTP_RETRY_MAX_ATTEMPTS,
            retry_base_delay_s=HTTP_RETRY_BASE_DELAY_S,
            retry_max_jitter_s=HTTP_RETRY_MAX_JITTER_S,
            retry_backoff_factor=HTTP_RETRY_BACKOFF_FACTOR)
        if response.redirect_url:
            return {'redirectUrl': response.redirect_url, 'status': response.status,
                    'finalUrl': response.url, 'error': None}
        content_type = response.header('content-type')
        body = _decode_http_text(response.body, response.headers)
        return {'content': body, 'contentType': content_type, 'status': response.status,
                'finalUrl': response.url, 'truncated': response.truncated, 'error': None}
    except Exception as e:
        return {'error': f'fetch failed: {e}', 'finalUrl': url}


async def run_webfetch_async(url: str, prompt: str) -> str:
    if not url:
        return "Error: url is required"
    if not prompt:
        return "Error: prompt is required"
    now = time.time()
    cached = _webfetch_cache.get(url)
    if cached and now - cached[0] < WEBFETCH_CACHE_TTL:
        content_text, content_type, final_url, status = cached[1], cached[2], cached[3], cached[4]
        cache_hit = True
    else:
        response = await _fetch_url_async(url)
        if response.get('error'):
            return f"Error: {response['error']}"
        if response.get('redirectUrl'):
            return "\n".join([
                f"WebFetch redirect: HTTP {response['status']}",
                f"requested_url: {response['finalUrl']}",
                f"redirect_url: {response['redirectUrl']}",
                "Call WebFetch again with redirect_url if you want to fetch that page.",
            ])
        content_type = response['contentType']
        if _is_html_content_type(content_type):
            content_text = _html_to_text(response['content'])
        else:
            content_text = response['content']
        if response.get('truncated'):
            content_text += "\n[... page truncated at fetch limit]"
        # cap to a sane size before sending to the model
        if len(content_text) > WEBFETCH_MAX_PROMPT_CHARS:
            content_text = content_text[:WEBFETCH_MAX_PROMPT_CHARS] + "\n[... content truncated for prompt processing]"
        final_url = response['finalUrl']
        status = response['status']
        _webfetch_cache[url] = (now, content_text, content_type, final_url, status)
        cache_hit = False

    msgs = [
        formats.instruction_item(
            "You are processing content fetched by the WebFetch tool. "
            "Answer only from the fetched page content. "
            "If the content does not contain the answer, say so plainly. "
            "Keep quotes short and do not reproduce large copyrighted passages."),
        formats.message_item(
            "user",
            f"URL: {final_url}\nContent-Type: {content_type}\nPrompt: {prompt}\n\n--- Page content ---\n{content_text}"),
    ]
    answer = await run_toolless_completion_async(msgs) or "(no answer returned)"
    header = f"[WebFetch status={status} cache_hit={cache_hit} bytes~={len(content_text)} url={final_url}]"
    return f"{header}\n{answer}"


async def run_websearch_async(query: str, allowed_domains: list = None,
                              blocked_domains: list = None) -> str:
    if not query or len(query) < 2:
        return "Error: query must be at least 2 characters"
    if allowed_domains and blocked_domains:
        return "Error: allowed_domains and blocked_domains cannot both be specified"
    if allowed_domains and len(allowed_domains) > 20:
        return "Error: allowed_domains max 20 entries"
    if blocked_domains and len(blocked_domains) > 20:
        return "Error: blocked_domains max 20 entries"

    form = urllib.parse.urlencode({'q': query, 'b': '', 'kl': 'us-en'})
    try:
        response = await http_client.async_http_request(
            'POST',
            DUCKDUCKGO_HTML_SEARCH_URL,
            body=form.encode('utf-8'),
            headers_in={'User-Agent': 'loki-WebSearch/0.1 (coding-agent)',
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.1'},
            timeout=WEBSEARCH_TIMEOUT_S,
            max_bytes=WEBSEARCH_MAX_RESPONSE_BYTES,
            retry_max_attempts=HTTP_RETRY_MAX_ATTEMPTS,
            retry_base_delay_s=HTTP_RETRY_BASE_DELAY_S,
            retry_max_jitter_s=HTTP_RETRY_MAX_JITTER_S,
            retry_backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
        )
        if response.status >= 400:
            return f"Error: web search request failed: HTTP {response.status} {response.reason}"
        html = _decode_http_text(response.body, response.headers)
    except Exception as e:
        return f"Error: web search request failed: {e}"

    results = []
    for result in _parse_duckduckgo_results(html):
        target = result['url']
        host = urllib.parse.urlparse(target).netloc.lower()
        if allowed_domains and not any(host == d.lower() or host.endswith('.' + d.lower())
                                       for d in allowed_domains):
            continue
        if blocked_domains and (any(host == d.lower() or host.endswith('.' + d.lower())
                                    for d in blocked_domains)):
            continue
        results.append(result)
        if len(results) >= WEBSEARCH_MAX_RESULTS:
            break

    if not results:
        return f"No search results for query {query!r}"

    out_lines = [f"WebSearch results for {query!r} ({len(results)} results):"]
    for i, r in enumerate(results, start=1):
        out_lines.append(f"{i}. {r['title']}\n   {r['url']}")

    return "\n".join(out_lines)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "\n".join([
                "Reads a file from the local filesystem.",
                "",
                "- `file_path` may be absolute or relative to the current Loki cwd.",
                "- Reads up to 2000 lines by default.",
                "- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters",
                "- Results are returned using cat -n format, with line numbers starting at 1",
                "- Reads images (PNG, JPG/JPEG, GIF, WEBP) and presents them visually.",
                "- Reading a directory, a missing file, or an empty file returns an error or system reminder rather than content.",
                "- Do NOT re-read a file you just edited to verify -- Edit/Write would have errored if the change failed, and the harness tracks file state for you.",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Plain absolute or relative filesystem path "
                            "passed to file APIs. Do not use Markdown links "
                            "or URLs."),
                    },
                    "offset": {"type": "integer", "minimum": 0,
                               "description": "The line number to start reading from. Only provide if the file is too large to read at once"},
                    "limit": {"type": "integer", "minimum": 1,
                              "description": "The number of lines to read. Only provide if the file is too large to read at once."}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "\n".join([
                "Writes a file to the local filesystem, overwriting if one exists.",
                "",
                "When to use: creating a new file, or fully replacing one you've already Read. Overwriting an existing file you haven't Read will fail. For partial changes, use Edit instead."
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Plain absolute or relative filesystem path "
                            "passed to file APIs. Do not use Markdown links "
                            "or URLs."),
                    },
                    "content": {"type": "string", "description": "The content to write to the file"}
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "\n".join([
                "Performs exact string replacement in a file.",
                "",
                "- You must Read the file in this conversation before editing, or the call will fail.",
                "- `old_string` must match the file exactly, including indentation, and be unique -- the edit fails otherwise. Strip the Read line prefix (line number + tab) before matching.",
                "- `replace_all: true` replaces every occurrence instead."
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Plain absolute or relative filesystem path "
                            "passed to file APIs. Do not use Markdown links "
                            "or URLs."),
                    },
                    "old_string": {"type": "string", "description": "The text to replace"},
                    "new_string": {"type": "string",
                                   "description": "The text to replace it with (must be different from old_string)"},
                    "replace_all": {"type": "boolean", "default": False,
                                    "description": "Replace all occurrences of old_string (default false)"}
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "\n".join([
                "Executes a bash command and returns its output.",
                "",
                "- Commands run in the current Loki cwd. Use /cd DIR to change that cwd; cd inside a Bash command only affects that one command.",
                "- IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool as this will provide a much better experience for the user.",
                f"- `timeout` is in milliseconds: default {BASH_DEFAULT_TIMEOUT_MS}, max {BASH_MAX_TIMEOUT_MS}.",
                "- `run_in_background` starts a detached job with stdout/stderr spooled to files. Use Jobs/JobStatus/JobStop to inspect or control it. No `&` needed.",
                "",
                "# Git",
                "- Interactive flags (`-i`, e.g. `git rebase -i`, `git add -i`) are not supported in this environment.",
                "- Commit or push only when the user asks. If on the default branch, branch first."
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute"},
                    "timeout": {"type": "integer", "minimum": 0, "maximum": BASH_MAX_TIMEOUT_MS,
                                "description": f"Optional timeout in milliseconds (max {BASH_MAX_TIMEOUT_MS})"},
                    "description": {"type": "string", "description": "\n".join([
                        'Clear, concise description of what this command does in active voice. Never use words like "complex" or "risk" in the description - just describe what it does.',
                        "",
                        "For simple commands (git, npm, standard CLI tools), keep it brief (5-10 words):",
                        '- ls -> "List files in current Loki cwd"',
                        '- git status -> "Show working tree status"',
                        '- npm install -> "Install package dependencies"',
                        "",
                        "For commands that are harder to parse at a glance (piped commands, obscure flags, etc.), add enough context to clarify what it does:",
                        '- find . -name "*.tmp" -exec rm {} \\; -> "Find and delete all .tmp files recursively"',
                        '- git reset --hard origin/main -> "Discard all local changes and match remote main"',
                        "- curl -s url | jq '.data[]' -> \"Fetch JSON from URL and extract data array elements\"",
                    ])},
                    "run_in_background": {"type": "boolean",
                                          "description": "Set to true to run this command in the background."},
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Jobs",
            "description": "List background jobs with their status, pid, exit code, and command.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "JobStatus",
            "description": "Inspect one shell job, including status, exit code, spool paths, and stdout/stderr tails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The job id returned by Bash, Agent, or Jobs"},
                    "tail_chars": {"type": "integer", "minimum": 0,
                                   "description": f"Maximum characters to show from each spool file. Defaults to {JOB_TAIL_CHARS}."}
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "JobStop",
            "description": "Stop a running shell job by sending SIGTERM to its process group, or SIGKILL if force is true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The job id returned by Bash, Agent, or Jobs"},
                    "force": {"type": "boolean",
                              "description": "Use SIGKILL instead of SIGTERM. Default false."}
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Fast file pattern matching. Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\". Returns matching file paths sorted by modification time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The glob pattern to match files against"},
                    "path": {
                        "type": "string",
                        "description": (
                            "Plain filesystem directory path. Defaults to "
                            "the current Loki cwd. Omit it for the default; "
                            "do not pass null, undefined, Markdown, or a URL."),
                    },
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "\n".join([
                "Content search built on ripgrep. Prefer this over `grep`/`rg` via Bash -- results integrate with the permission UI and file links.",
                "",
                "- Full regex syntax (e.g. \"log.*Error\", \"function\\s+\\w+\"). Ripgrep, not grep -- escape literal braces (`interface\\{\\}`).",
                "- Filter with `glob` (e.g. \"**/*.tsx\") or `type` (e.g. \"js\", \"py\", \"rust\").",
                "- `output_mode`: \"content\" (matching lines), \"files_with_matches\" (paths only, default), or \"count\".",
                "- `multiline: true` for patterns that span lines.",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string",
                                "description": "The regular expression pattern to search for in file contents"},
                    "path": {
                        "type": "string",
                        "description": (
                            "Plain filesystem file or directory path for "
                            "rg. Defaults to the current Loki cwd; do not "
                            "use Markdown links or URLs."),
                    },
                    "glob": {"type": "string",
                             "description": "Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\") - maps to rg --glob"},
                    "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"],
                                    "description": "Output mode: \"content\" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), \"files_with_matches\" shows file paths (supports head_limit), \"count\" shows match counts (supports head_limit). Defaults to \"files_with_matches\"."},
                    "-B": {"type": "integer",
                           "description": "Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise."},
                    "-A": {"type": "integer",
                           "description": "Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise."},
                    "-C": {"type": "integer", "description": "Alias for context."},
                    "context": {"type": "integer",
                                "description": "Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise."},
                    "-n": {"type": "boolean",
                           "description": "Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise. Defaults to true."},
                    "-i": {"type": "boolean", "description": "Case insensitive search (rg -i)"},
                    "-o": {"type": "boolean",
                           "description": "Print only the matched (non-empty) parts of each matching line, one match per output line (rg -o / --only-matching). Requires output_mode: \"content\", ignored otherwise. Defaults to false."},
                    "type": {"type": "string",
                             "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types."},
                    "head_limit": {"type": "integer",
                                   "description": "Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). Defaults to 250 when unspecified. Pass 0 for unlimited (use sparingly -- large result sets waste context)."},
                    "offset": {"type": "integer",
                               "description": "Skip first N lines/entries before applying head_limit, equivalent to \"| tail -n +N | head -N\". Works across all output modes. Defaults to 0."},
                    "multiline": {"type": "boolean",
                                  "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "TodoRead",
            "description": "Read the current session todo list",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": "\n".join([
                "Create and update a task list for the current session. The list is rendered to the user as your working plan.",
                "",
                "- Each todo has `content`, `status` (\"pending\" | \"in_progress\" | \"completed\"), and `priority` (\"high\" | \"medium\" | \"low\").",
                "- Send the full list each call; it replaces the previous one.",
                "- Keep one item `in_progress` at a time and mark it `completed` when done.",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The complete updated todo list. At most one item may be in_progress at a time.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Brief description of the task"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"],
                                           "description": "Current status of the task"},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"],
                                             "description": "Priority level of the task"}
                            },
                            "required": ["content", "status", "priority"]
                        }
                    }
                },
                "required": ["todos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Agent",
            "description": "\n".join([
                "Launch a new agent to handle complex, multi-step tasks. Each agent type has specific capabilities and tools available to it.",
                "",
                "Available agent types and the tools they have access to:",
                "- Explore: Read-only search agent for broad fan-out searches - when answering means sweeping many files, directories, or naming conventions and you only need the conclusion, not the file dumps. It reads excerpts rather than whole files, so it locates code; it doesn't review or audit it. Specify search breadth: \"medium\" for moderate exploration, \"very thorough\" for multiple locations and naming conventions. (Tools: Glob, Grep, Read, Bash, Jobs, JobStatus, JobStop, WebFetch, WebSearch, TodoWrite)",
                "",
                "When using the Agent tool, specify a subagent_type parameter to select which agent type to use. If omitted, the \"Explore\" agent is used.",
                "",
                "## When to use",
                "",
                "Reach for this when the task matches an available agent type, when you have independent work to run in parallel, or when answering would mean reading across several files - delegate it and you keep the conclusion, not the file dumps. For a single-fact lookup where you already know the file, symbol, or value, search directly. Once you've delegated a search, don't also run it yourself - wait for the result.",
                "",
                "- The agent's final message is returned to you as the tool result; it is not shown to the user - relay what matters.",
                "- A new Agent call starts fresh, so the prompt must be self-contained.",
                "- `run_in_background` starts the subagent as a background job with stdout/stderr spooled to files. Use Jobs/JobStatus/JobStop to inspect or control it.",
                "- When you launch multiple agents for independent work, send them in a single message with multiple tool uses so they run concurrently",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "A short (3-5 word) description of the task"},
                    "prompt": {"type": "string", "description": "The task for the agent to perform"},
                    "run_in_background": {"type": "boolean",
                                          "description": "Set to true to run this agent as a background job. Use Jobs/JobStatus/JobStop to inspect or control it."},
                    "subagent_type": {"type": "string", "enum": ["Explore"],
                                      "description": "The type of specialized agent to use for this task"}
                },
                "required": ["description", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Skill",
            "description": "\n".join([
                "Execute a skill within the main conversation",
                "",
                "When users ask you to perform tasks, check if any of the available skills match. Skills provide specialized capabilities and domain knowledge.",
                "",
                "When users reference a \"slash command\" or \"/<something>\", they are referring to a skill. Use this tool to invoke it.",
                "",
                "How to invoke:",
                "- Set `skill` to the exact name of an available skill (no leading slash). For plugin-namespaced skills use the fully qualified `plugin:skill` form.",
                "- Set `args` to pass optional arguments.",
                "",
                "Important:",
                "- Available skills are listed in system-reminder messages in the conversation",
                "- Only invoke a skill that appears in that list, or one the user explicitly typed as `/<name>` in their message. Never guess or invent a skill name from training data; otherwise do not call this tool",
                "- When a skill matches the user's request, this is a BLOCKING REQUIREMENT: invoke the relevant Skill tool BEFORE generating any other response about the task",
                "- NEVER mention a skill without actually calling this tool",
                "- Do not invoke a skill that is already running",
                "- Do not use this tool for built-in CLI commands (like /help, /clear, etc.)",
                "- If you see a <command-name> tag in the current conversation turn, the skill has ALREADY been loaded - follow the instructions directly instead of calling this tool again",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string",
                              "description": "The name of a skill from the available-skills list. Do not guess names."},
                    "args": {"type": "string", "description": "Optional arguments for the skill"}
                },
                "required": ["skill"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "WebFetch",
            "description": "\n".join([
                "Fetches a URL, converts the page to markdown, and answers `prompt` against it using a small fast model.",
                "",
                "- Fails on authenticated/private URLs -- use an authenticated MCP tool or `gh` for those instead.",
                "- HTTP is upgraded to HTTPS. Cross-host redirects are returned to you rather than followed; call again with the redirect URL.",
                "- Responses are cached for 15 minutes per URL.",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri",
                            "description": "The URL to fetch content from"},
                    "prompt": {"type": "string", "description": "The prompt to run on the fetched content"}
                },
                "required": ["url", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "WebSearch",
            "description": "\n".join([
                "Search the web. Returns result blocks with titles and URLs. US-only.",
                "",
                "- Use the session's current-date instruction when searching for recent information.",
                "- `allowed_domains` / `blocked_domains` filter results.",
                '- After answering from results, end with a "Sources:" list of the URLs you used as markdown links.',
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 2, "description": "The search query to use"},
                    "allowed_domains": {"type": "array", "items": {"type": "string"}, "maxItems": 20,
                                        "description": "Only include search results from these domains"},
                    "blocked_domains": {"type": "array", "items": {"type": "string"}, "maxItems": 20,
                                        "description": "Never include search results from these domains"}
                },
                "required": ["query"]
            }
        }
    },
]

TOOL_HANDLERS = {
    "Read": {
        "handler": _handle_read,
        "explore": True,
        "semantics": {
            ("file_path",): tool_runtime.FILESYSTEM_PATH,
        },
    },
    "Write": {
        "handler": _handle_write,
        "semantics": {
            ("file_path",): tool_runtime.FILESYSTEM_PATH,
        },
    },
    "Edit": {
        "handler": _handle_edit,
        "semantics": {
            ("file_path",): tool_runtime.FILESYSTEM_PATH,
        },
    },
    "Bash": {"handler": _handle_bash, "async_handler": _handle_bash_async, "explore": True},
    "Jobs": {"handler": _handle_jobs, "explore": True},
    "JobStatus": {"handler": _handle_job_status, "explore": True},
    "JobStop": {"handler": _handle_job_stop, "explore": True},
    "Glob": {
        "handler": _handle_glob,
        "async_handler": _handle_glob_async,
        "explore": True,
        "semantics": {
            ("path",): tool_runtime.FILESYSTEM_PATH,
        },
    },
    "Grep": {
        "handler": _handle_grep,
        "async_handler": _handle_grep_async,
        "explore": True,
        "semantics": {
            ("path",): tool_runtime.FILESYSTEM_PATH,
        },
    },
    "TodoRead": {"handler": _handle_todoread, "explore": True},
    "TodoWrite": {"handler": _handle_todowrite, "explore": True},
    "Agent": {"handler": _handle_agent, "async_handler": _handle_agent_async},
    "Skill": {"handler": _handle_skill},
    "WebFetch": {"async_handler": _handle_webfetch_async, "explore": True},
    "WebSearch": {"async_handler": _handle_websearch_async, "explore": True},
}
TOOL_REGISTRY = _build_tool_registry(TOOLS, TOOL_HANDLERS)
TOOLS = [spec["definition"] for spec in TOOL_REGISTRY.values()]
EXPLORE_TOOLS = {name for name, spec in TOOL_REGISTRY.items() if spec["explore"]}
TOOL_HOOK_PIPELINE = tool_runtime.ToolHookPipeline()


def configure_tool_hook_pipeline(environ=os.environ):
    """Load trusted user-selected command hooks; project hooks are never automatic."""
    global TOOL_HOOK_PIPELINE
    configured = environ.get("LOKI_HOOKS")
    if configured is None:
        default_path = os.path.join(LOKI_CONFIG_DIR, "hooks.json")
        configured = default_path if os.path.isfile(default_path) else ""
    if configured.strip().lower() in ["", "0", "none", "off"]:
        TOOL_HOOK_PIPELINE = tool_runtime.ToolHookPipeline()
        return None
    path = os.path.expanduser(configured)
    if not os.path.isabs(path):
        path = os.path.join(STARTUP_CWD, path)
    TOOL_HOOK_PIPELINE = tool_runtime.load_hook_pipeline(path)
    return path

async def async_chat_request(request_url: str, payload, request_headers: dict = None,
                             report_errors: bool = False, show_timing: bool = False) -> dict:
    start = time.perf_counter()
    body = json.dumps(payload).encode('utf-8') if payload is not None else b''
    method = 'POST' if payload is not None else 'GET'

    headers_to_use = request_headers
    if headers_to_use is None:
        headers_to_use = current_config().headers if current_config() else {}
    # Copy so the per-call idempotency key below does not mutate the cached
    # provider headers shared across requests.
    headers_to_use = dict(headers_to_use)

    retry_attempts = HTTP_RETRY_MAX_ATTEMPTS
    if method == 'POST':
        # Best-effort server-side dedup; Anthropic honors this header on
        # /v1/messages. OpenAI-compat servers may ignore it.
        kind = current_config().provider_kind if current_config() else None
        header = (LLM_IDEMPOTENCY_HEADER_ANTHROPIC
                  if kind == protocols.ANTHROPIC_MESSAGES
                  else LLM_IDEMPOTENCY_HEADER_OPENAI)
        headers_to_use[header] = uuid.uuid4().hex
        retry_attempts = HTTP_RETRY_MAX_ATTEMPTS_LLM

    response = await http_client.async_http_request(
        method,
        request_url,
        body=body,
        headers_in=headers_to_use,
        timeout=WEBFETCH_TIMEOUT_S,
        max_bytes=HTTP_MAX_RESPONSE_BYTES,
        retry_max_attempts=retry_attempts,
        retry_base_delay_s=HTTP_RETRY_BASE_DELAY_S,
        retry_max_jitter_s=HTTP_RETRY_MAX_JITTER_S,
        retry_backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
    )
    response_text = _decode_http_text(response.body, response.headers)
    if response.status >= 400:
        raise ApiError(request_url, response.status, response.reason, response_text)
    data = json.loads(response_text)

    elapsed = time.perf_counter() - start
    if show_timing:
        print(f"\n[T]  [LLM Response Time: {elapsed:.3f}s]", file=sys.stderr)
    return data


async def _next_stream_chunk(iterator, cancel_check):
    if cancel_check():
        raise StreamCancelled()
    task = asyncio.create_task(anext(iterator))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.05)
            if done:
                return task.result()
            if cancel_check():
                raise StreamCancelled()
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass


async def _collect_stream_body(iterator, first_chunk, cancel_check):
    chunks = [first_chunk] if first_chunk else []
    while True:
        try:
            chunk = await _next_stream_chunk(iterator, cancel_check)
        except StopAsyncIteration:
            return b"".join(chunks)
        chunks.append(chunk)


def _stream_body_kind(content_type, first_chunk):
    prefix = first_chunk.lstrip()
    if prefix.startswith((b"{", b"[")):
        return "json"
    if prefix.startswith((b"data:", b"event:", b":")):
        return "sse"
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type == "text/event-stream":
        return "sse"
    if media_type in ("application/json", "application/problem+json"):
        return "json"
    return "json"


async def _async_chat_stream_request_once(
        request_url, payload, request_headers, on_text_delta, cancel_check):
    body = json.dumps(payload).encode("utf-8")
    async with http_client.async_http_stream(
            "POST",
            request_url,
            body=body,
            headers_in=request_headers,
            timeout=LLM_STREAM_IDLE_TIMEOUT_S,
            max_bytes=HTTP_MAX_RESPONSE_BYTES,
            cancel_check=cancel_check) as response:
        iterator = response.body.__aiter__()
        try:
            first_chunk = await _next_stream_chunk(iterator, cancel_check)
        except StopAsyncIteration:
            first_chunk = b""

        if response.status >= 400:
            try:
                response_body = await _collect_stream_body(
                    iterator, first_chunk, cancel_check)
            except OSError as e:
                if first_chunk:
                    raise protocols.StreamProtocolError(
                        f"stream interrupted after output began: {e}") from e
                raise
            response_text = _decode_http_text(
                response_body, response.headers)
            raise StreamingApiError(
                request_url, response.status, response.reason, response_text)

        if _stream_body_kind(
                response.header("content-type"), first_chunk) == "json":
            try:
                response_body = await _collect_stream_body(
                    iterator, first_chunk, cancel_check)
            except OSError as e:
                if first_chunk:
                    raise protocols.StreamProtocolError(
                        f"stream interrupted after output began: {e}") from e
                raise
            try:
                return json.loads(_decode_http_text(
                    response_body, response.headers))
            except json.JSONDecodeError as e:
                raise protocols.StreamProtocolError(
                    "streaming response was neither valid SSE nor JSON: "
                    f"{e}") from e

        accumulator = current_config().chat_provider.stream_accumulator(
            on_text_delta)
        decoder = sse.SseDecoder()
        received_body = bool(first_chunk)
        try:
            for event in decoder.feed(first_chunk):
                accumulator.feed(event)
            while True:
                try:
                    chunk = await _next_stream_chunk(
                        iterator, cancel_check)
                except StopAsyncIteration:
                    break
                received_body = received_body or bool(chunk)
                for event in decoder.feed(chunk):
                    accumulator.feed(event)
            for event in decoder.finish():
                accumulator.feed(event)
        except StreamCancelled:
            raise
        except protocols.StreamProtocolError:
            raise
        except ValueError as e:
            raise protocols.StreamProtocolError(
                f"invalid SSE stream: {e}") from e
        except OSError as e:
            if received_body:
                raise protocols.StreamProtocolError(
                    f"stream interrupted after output began: {e}") from e
            raise
        return accumulator.finish()


async def async_chat_stream_request(
        request_url: str, payload, request_headers: dict = None,
        on_text_delta=None, cancel_check=None,
        report_errors: bool = False, show_timing: bool = False) -> dict:
    start = time.perf_counter()
    callback = on_text_delta or (lambda text: None)
    cancel = cancel_check or (lambda: False)
    headers_to_use = dict(
        request_headers if request_headers is not None
        else (current_config().headers if current_config() else {}))
    headers_to_use.setdefault("Accept", "text/event-stream")
    kind = current_config().provider_kind if current_config() else None
    idempotency_header = (
        LLM_IDEMPOTENCY_HEADER_ANTHROPIC
        if kind == protocols.ANTHROPIC_MESSAGES
        else LLM_IDEMPOTENCY_HEADER_OPENAI)
    headers_to_use[idempotency_header] = uuid.uuid4().hex

    attempt = 0
    while True:
        attempt += 1
        try:
            data = await _async_chat_stream_request_once(
                request_url, payload, headers_to_use, callback, cancel)
            break
        except http_client.HttpRequestCancelled:
            raise StreamCancelled()
        except Exception as exc:
            if (attempt >= HTTP_RETRY_MAX_ATTEMPTS_LLM
                    or not http_client.is_transient_error(exc)):
                raise
            delay = (
                HTTP_RETRY_BASE_DELAY_S
                * (HTTP_RETRY_BACKOFF_FACTOR ** (attempt - 1)))
            if HTTP_RETRY_MAX_JITTER_S:
                delay += random.uniform(0, HTTP_RETRY_MAX_JITTER_S)
            await asyncio.sleep(delay)

    elapsed = time.perf_counter() - start
    if show_timing:
        print(f"\n[T]  [LLM Response Time: {elapsed:.3f}s]",
              file=sys.stderr)
    return data


async def async_chat_completion(transcript_items: list, tools=TOOLS, report_errors: bool = False,
                                show_timing: bool = False,
                                on_text_delta=None,
                                cancel_check=None) -> formats.DecodedTurn:
    if not current_config():
        return formats.DecodedTurn([])

    if current_config().chat_provider.kind == protocols.DUMMY:
        # No-op LLM for testing: never touches the network.  The reply is a
        # canned assistant message, so the whole input/turn/render loop runs
        # deterministically. LOKI_DUMMY_STREAM_CHUNKS optionally supplies a
        # JSON array of deltas. A test can make the provider pause after its
        # first delta by setting LOKI_DUMMY_STREAM_GATE to a path that the test
        # creates after observing the corresponding terminal output.
        reply = os.environ.get("LOKI_DUMMY_REPLY", "ok")
        raw_chunks = os.environ.get("LOKI_DUMMY_STREAM_CHUNKS")
        if current_config().stream and on_text_delta and raw_chunks is not None:
            try:
                chunks = json.loads(raw_chunks)
            except json.JSONDecodeError as error:
                raise protocols.ProtocolError(
                    f"LOKI_DUMMY_STREAM_CHUNKS is invalid JSON: {error}"
                ) from error
            if (not isinstance(chunks, list)
                    or any(not isinstance(chunk, str) for chunk in chunks)):
                raise protocols.ProtocolError(
                    "LOKI_DUMMY_STREAM_CHUNKS must be a JSON array "
                    "of strings")
            gate = os.environ.get("LOKI_DUMMY_STREAM_GATE")
            for index, chunk in enumerate(chunks):
                on_text_delta(chunk)
                if index == 0 and gate:
                    deadline = time.monotonic() + 10
                    while not os.path.exists(gate):
                        if cancel_check and cancel_check():
                            raise StreamCancelled()
                        if time.monotonic() >= deadline:
                            raise protocols.ProtocolError(
                                "timed out waiting for "
                                "LOKI_DUMMY_STREAM_GATE")
                        await asyncio.sleep(0.01)
        return formats.DecodedTurn(
            [formats.message_item("assistant", reply)],
            {
                "protocol": "dummy",
                "provider_id": "dummy",
                "model": current_model(),
                "response": {},
            },
        )

    if current_config().stream:
        payload = current_config().chat_provider.streaming_chat_payload(
            transcript_items, tools, current_model())
        data = await async_chat_stream_request(
            current_config().chat_provider.chat_url,
            payload,
            request_headers=current_config().chat_provider.headers,
            on_text_delta=on_text_delta,
            cancel_check=cancel_check,
            report_errors=report_errors,
            show_timing=False,
        )
    else:
        payload = current_config().chat_provider.chat_payload(
            transcript_items, tools, current_model())
        data = await async_chat_request(
            current_config().chat_provider.chat_url,
            payload,
            request_headers=current_config().chat_provider.headers,
            report_errors=report_errors,
            show_timing=show_timing,
        )
    if not isinstance(data, dict):
        raise protocols.ProtocolError(
            "chat response must be a JSON object", payload=data)
    detected = protocols.detect_protocol_from_response(data)
    if detected and detected != current_config().chat_provider.kind:
        # A configured adapter should not parse a response that clearly has
        # another protocol's shape; that usually means endpoint/config mismatch.
        raise protocols.ProtocolError(
            f"configured provider {current_config().chat_provider.kind!r} "
            f"but response looks like {detected!r}",
            payload=data,
        )
    try:
        turn = formats.coerce_decoded_turn(
            current_config().chat_provider.parse_chat_response(data))
    except formats.TranscriptFormatError as e:
        raise protocols.ProtocolError(
            f"invalid {current_config().chat_provider.kind} response: {e}",
            payload=data,
        ) from e
    turn.metadata["provider_id"] = current_config().provider_id
    turn.metadata["provider_name"] = current_config().provider_name
    turn.metadata["endpoint"] = (
        current_config().chat_provider.chat_url)
    turn.metadata["model"] = current_model()
    turn.metadata["protocol"] = current_config().chat_provider.kind
    return turn




def _status_api_base() -> str:
    configured_url = ""
    if current_config():
        configured_url = current_config().chat_provider.input_url if current_config().chat_provider else current_config().url

    if not configured_url:
        return "not configured"
    parsed = urllib.parse.urlparse(configured_url)
    if not parsed.netloc:
        return configured_url
    host = parsed.hostname or parsed.netloc.rsplit("@", 1)[-1]
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    path = parsed.path.rstrip("/")
    for suffix in [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/messages",
        "/messages",
        "/v1/responses",
        "/responses",
        "/v1",
    ]:
        if path.endswith(suffix):
            path = path[:-len(suffix)].rstrip("/")
            break
    return host + path


def status_text() -> str:
    displayed_model = current_model()
    if (current_config() is not None
            and current_config().model_status == "deprecated"):
        displayed_model += " (deprecated)"
    return (
        'Remote: API: {}; Model: {}; /model\n'
        'Local: mode={}; CWD: {}; /pwd, /cd DIR, /ps, !foo, /quit'
    ).format(
        _status_api_base(), displayed_model,
        current_agent_mode(), display_path(current_cwd()))


async def load_models_async():
    """Fetch the active provider's /models listing; [] when unavailable."""
    if not current_config():
        return [current_model()] if current_model() else []

    model_urls = getattr(current_config().chat_provider, "model_urls", None) or ([current_config().chat_provider.models_url]
                                                                if current_config().chat_provider.models_url else [])
    if not model_urls:
        return [current_model()] if current_model() else []
    errors = []
    for models_url in model_urls:
        try:
            data = await async_chat_request(
                models_url,
                None,
                request_headers=current_config().chat_provider.headers,
                report_errors=True,
            )
        except ApiError as e:
            errors.append(e.formatted())
            continue
        except OSError as e:
            errors.append(f"API Error for <{models_url}>: {e}")
            continue
        except json.JSONDecodeError as e:
            # Some providers (e.g. Z.AI's /paas/v4 base) don't expose a /models
            # endpoint at all -- the response is an HTML error page or empty.
            # Treat as "couldn't load" rather than crashing the picker.
            errors.append(f"Model list at <{models_url}> was not JSON: {e.msg}")
            continue
        loaded = current_config().chat_provider.parse_model_ids(data)
        if loaded:
            return loaded

    if errors:
        print("Model list failed:\n" + "\n".join(errors), file=sys.stderr)
    return [current_model()] if current_model() else []

#models = ['hy3-preview', 'glm-5.2', 'glm-5.1', 'kimi-k2.7', 'kimi-k2.6', 'deepseek-v4-pro', 'deepseek-v4-flash', 'mimo-v2.5', 'mimo-v2.5-pro']

terminals.set_status_text_provider(status_text)

# Agent mode, cycled by Shift-Tab: "explore" (read-only), "plan", "edit".
# Takes effect for the next turn; it does not cancel the current turn.
MODE_CYCLE_ORDER = ["normal", "explore", "plan", "edit"]


def cycle_agent_mode() -> str:
    session = current_session()
    i = MODE_CYCLE_ORDER.index(session.agent_mode)
    session.agent_mode = MODE_CYCLE_ORDER[(i + 1) % len(MODE_CYCLE_ORDER)]
    return session.agent_mode


def initial_transcript_items():
    return [formats.instruction_item(
        "You are a helpful system agent running in a terminal. You have these tools: "
        "Read, Write, Edit, Bash, Jobs, JobStatus, JobStop, Glob, Grep, TodoRead, TodoWrite, Agent, Skill, WebFetch, WebSearch. "
        f"Current date: {time.strftime('%Y-%m-%d')}. "
        f"Current Loki cwd: {current_cwd()}. Relative tool paths and Bash commands run from this directory. "
        "Prefer Glob/Grep/Read over Bash equivalents (find/grep/cat). "
        "Always Read a file before editing or overwriting it. "
        "Use TodoWrite to plan multi-step work. Keep responses concise."
    )]


def _remember_session_toolset(tools):
    snapshot = copy.deepcopy(tools or [])
    for existing in current_toolsets():
        if existing == snapshot:
            return
    current_toolsets().append(snapshot)


def user_prompt_history(items):
    return formats.user_prompt_history(items)


def record_shell_cwd_instruction():
    current_transcript().append(formats.instruction_item(
        f"Current Loki cwd changed to: {current_cwd()}. "
        "Relative tool paths and Bash commands now run from this directory."
    ))
    mark_chat_log_dirty()


def print_shell_cwd():
    sys.stdout.flush()
    print(f"cwd: {current_cwd()}", file=sys.stderr)
    sys.stderr.flush()


def change_shell_cwd_from_text(arg_text: str) -> bool:
    try:
        target = _parse_cd_arg_text(arg_text)
        change_shell_cwd(target)
    except (FileNotFoundError, NotADirectoryError) as e:
        sys.stdout.flush()
        print(f"cd: no such directory: {e}", file=sys.stderr)
        sys.stderr.flush()
        return False
    except ValueError as e:
        sys.stdout.flush()
        print(f"cd: {e}", file=sys.stderr)
        sys.stderr.flush()
        return False
    record_shell_cwd_instruction()
    print_shell_cwd()
    return True


def _parse_cd_arg_text(arg_text: str) -> str:
    if not arg_text.strip():
        return ""
    parts = shlex.split(arg_text)
    if len(parts) != 1:
        raise ValueError("expected zero or one directory argument")
    return parts[0]


def chat_log_filename(chat_id: str) -> str:
    return savefiles.chat_log_filename(chat_id)


def ensure_chat_log_dir():
    savefiles.ensure_chat_log_dir(CHAT_LOG_DIR)


def new_chat_log_path() -> str:
    return savefiles.new_chat_log_path(CHAT_LOG_DIR)


def resolve_chat_log_path(resume_arg: str) -> str:
    return savefiles.resolve_chat_log_path(
        resume_arg, STARTUP_CWD, CHAT_LOG_DIR, _resolve_path)


async def run_session_picker_async(session):
    async with session.modal() as modal:
        picked = await savefiles.run_session_picker_async(
            input_fn=modal.prompt,
            terminal=terminal, chat_log_dir=CHAT_LOG_DIR)
        # Finish the picker's output cleanup while the modal still owns the
        # terminal. Only then may the normal input producer resume.
        terminal.goto_position(1, 1)
        terminal.clear_to_end_of_screen()
        terminal.flush()
    return picked


def new_chat_log(filename):
    session = current_session()
    session.transcript_items = initial_transcript_items()
    session.session_todos = []
    session.session_toolsets = []
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    session.chat_log_path = filename
    session.session_state = {"shell_cwd": session.shell_cwd}
    descriptor = active_connection_descriptor()
    if descriptor is not None:
        # Hybrid persistence rule: an explicit connection belongs to a new
        # chat, but does not implicitly replace a resumed chat's connection.
        session.session_state["connection"] = descriptor.to_dict()
    session.chat_log_dirty = True

def save_chat_log():
    session = current_session()

    if session.chat_log_path is None or not session.chat_log_dirty:
        return False

    state = dict(session.session_state)
    saved_connection = state.get("connection")
    if isinstance(saved_connection, dict) and "api_url" in saved_connection:
        saved_connection = dict(saved_connection)
        saved_connection.pop("api_url")
        state["connection"] = saved_connection
    state["shell_cwd"] = session.shell_cwd
    content = savefiles.serialize_chat_log(
        session.transcript_items,
        session.session_todos,
        state,
        toolsets=session.session_toolsets,
    )
    _atomic_write_text(session.chat_log_path, content)
    savefiles.report_chat_log_saved(session.chat_log_path)
    session.session_state = state
    session.chat_log_dirty = False
    return True


def mark_chat_log_dirty():
    session = current_session()
    if session.chat_log_path is not None:
        session.chat_log_dirty = True


def render_resume_transcript(items: list) -> str:
    return savefiles.render_resume_transcript(
        items, current_model() or "Assistant")


def print_resume_transcript(items: list):
    savefiles.print_resume_transcript(
        items, current_model() or "Assistant")


def load_chat_log(filename, loaded=None):
    if loaded is None:
        with open(filename, 'r', encoding="utf-8") as f:
            loaded = savefiles.read_chat_log(f)
    transcript, todos, state, toolsets = loaded
    current_session().replace_transcript(
        transcript, todos, toolsets, state, filename)
    load_session_state(state)
    print_resume_transcript(current_transcript())


def load_session_state(state: dict):
    if not isinstance(state, dict):
        return
    loaded_shell_cwd = state.get("shell_cwd")
    if not isinstance(loaded_shell_cwd, str) or not loaded_shell_cwd:
        return
    try:
        change_shell_cwd(loaded_shell_cwd)
    except FileNotFoundError:
        print(f"Warning: saved cwd no longer exists: {loaded_shell_cwd}", file=sys.stderr)


def connection_from_session_state(state: dict) -> ConnectionDescriptor | None:
    if not isinstance(state, dict) or "connection" not in state:
        return None
    return ConnectionDescriptor.from_dict(state["connection"])


def set_session_connection(descriptor: ConnectionDescriptor):
    current_state()["connection"] = descriptor.to_dict()
    mark_chat_log_dirty()


async def confirm_saved_connection_async(
        descriptor: ConnectionDescriptor, session,
        config: RuntimeConfig | None = None) -> bool:
    provider = descriptor.provider_name or descriptor.provider_id or "custom"
    selected_model = config.model if config is not None else descriptor.model
    endpoint = (config.chat_provider.chat_url
                if config is not None else descriptor.chat_url)
    models_endpoint = (config.chat_provider.models_url
                       if config is not None else descriptor.models_url)

    async with session.modal() as modal:
        print()
        print("Saved connection:")
        print(f"  Provider: {provider}")
        print(f"  Model: {selected_model}")
        print(f"  Chat endpoint: {endpoint}")
        if models_endpoint:
            print(f"  Models endpoint: {models_endpoint}")
        if descriptor.credential_env is None:
            print("  Authentication: none")
        else:
            print(f"  Credential: {descriptor.credential_env}")
        print(f"  Streaming: {'yes' if descriptor.stream else 'no'}")
        if descriptor.protocol == protocols.ANTHROPIC_MESSAGES:
            print(
                "  Anthropic prompt cache: "
                f"{'yes' if descriptor.prompt_cache else 'no'}")
        answer = (await modal.prompt(
            "Use this saved connection? [y/N]: ") or "")
        return answer.strip().lower() in ("y", "yes")

def run_subagent_prompt(subagent_type: str, prompt: str) -> str:
    return asyncio.run(run_subagent_prompt_async(subagent_type, prompt))


async def run_subagent_prompt_async(subagent_type: str, prompt: str) -> str:
    if subagent_type != "Explore":
        return f"Error: unknown subagent_type {subagent_type!r} (only 'Explore' is supported)"
    if not prompt:
        return ""
    msgs = [
        formats.instruction_item(
            "You are a focused Explore subagent. Use Glob/Grep/Read/Bash to investigate, then write a concise final answer."),
        formats.message_item("user", prompt),
    ]
    return await run_tool_loop_async(msgs, allowed=EXPLORE_TOOLS)


def run_subagent_cli(subagent_type: str, prompt: str = None):
    asyncio.run(run_subagent_cli_async(subagent_type, prompt))


async def run_subagent_cli_async(subagent_type: str, prompt: str = None):
    prompt = prompt if prompt is not None else sys.stdin.read().strip()
    result = await run_subagent_prompt_async(subagent_type, prompt)
    if result:
        print(result)


async def async_main(args) -> int:
    # getopt's "resume=" requires a value; normalize a bare `--resume` to
    # `--resume=` so it opens the picker instead of erroring out.
    args = ['--resume=' if a == '--resume' else a for a in args]
    options, args = getopt.getopt(args, 'r:p:', ['resume=', 'prompt=', 'subagent=', 'headless', 'toolset=', 'dangerously-skip-permissions'])
    prompt_arg = None
    subagent_type = None
    headless = False
    toolset = None
    for option_name, option_value in options:
        if option_name in ['--prompt', '-p']:
            prompt_arg = option_value
        elif option_name == '--subagent':
            subagent_type = option_value
        elif option_name == '--headless':
            headless = True
        elif option_name == '--toolset':
            toolset = option_value

    if subagent_type or headless:
        try:
            apply_runtime_config(build_config_from_env(
                credentials=CREDENTIALS))
        except (protocols.ProtocolError, ValueError) as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            return 2
        if not current_model():
            print("Configuration error: model missing; set LOKI_MODEL.",
                  file=sys.stderr)
            return 2
        await run_subagent_cli_async(subagent_type or toolset or "Explore", prompt_arg)
        return 0

    log_filename = None
    for option_name, option_value in options:
        if option_name == '--resume' or option_name == '-r':
            log_filename = option_value

    # The input session owns raw mode, the stdin reader, the producer, and the
    # user_messages queue for the whole session (see terminals.InputSession).
    # loki.py consumes the normal queue; session.modal() is the one exclusive
    # path used by the session picker, saved-connection prompt, and /model.
    async with input_session(on_mode_cycle=lambda: cycle_agent_mode(),
                             history_provider=lambda: user_prompt_history(current_transcript())) as session:
        if args[0:1] == ['resume']:
            if len(args) < 2:
                # Bare "resume" with no id opens the session picker. On cancel
                # (None), leave log_filename as None so the second block (which
                # only triggers on '') doesn't reopen the picker.
                picked = await run_session_picker_async(session=session)
                log_filename = picked
            else:
                log_filename = args[1]

        # An empty --resume value (e.g. "--resume=") also opens the picker.
        if log_filename == '':
            picked = await run_session_picker_async(session=session)
            log_filename = picked if picked is not None else ''

        resolved_log_filename = (
            resolve_chat_log_path(log_filename) if log_filename else None)
        loaded_chat = None
        saved_state = {}
        if resolved_log_filename:
            try:
                with open(resolved_log_filename, "r", encoding="utf-8") as f:
                    loaded_chat = savefiles.read_chat_log(f)
                    _, _, saved_state, _ = loaded_chat
            except (OSError, json.JSONDecodeError,
                    formats.TranscriptFormatError) as e:
                print(f"Could not resume chat: {e}", file=sys.stderr)
                return 1

        try:
            if explicit_api_base_configured(CREDENTIALS):
                config = build_config_from_env(credentials=CREDENTIALS)
            else:
                descriptor = connection_from_session_state(saved_state)
                if descriptor is None:
                    config = None
                else:
                    config = config_from_connection_descriptor(
                        descriptor, CREDENTIALS)
                    confirmed = await confirm_saved_connection_async(
                        descriptor, session, config=config)
                    if not confirmed:
                        print("Resume cancelled.", file=sys.stderr)
                        return 0
        except (ConnectionDescriptorError, protocols.ProtocolError,
                ValueError) as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            print("Starting without a provider; use /model or correct the "
                  "LOKI_* configuration.", file=sys.stderr)
            sys.stderr.flush()
            config = None

        if config is not None:
            apply_runtime_config(config)
            if not current_model():
                print("No model selected; use /model or set LOKI_MODEL.",
                      file=sys.stderr)
                sys.stderr.flush()
        else:
            print("No provider configured; use /model to select one.",
                  file=sys.stderr)
            sys.stderr.flush()

        if resolved_log_filename:
            load_chat_log(resolved_log_filename, loaded_chat)
        else:
            new_chat_log(new_chat_log_path())

        while True:
            user_in = await session.user_messages.get()
            restore_output_area_after_input()

            if user_in is None:  # EOF sentinel from the producer
                break

            if not user_in:
                continue

            terminal.set_background_color(terminals.INPUT_COLOR)
            print('User: ', end='')
            print(user_in, end='')
            terminal.reset_colors_and_flags()
            print()
            command_text = user_in.strip()
            match command_text:
                case '/quit':
                    break
                case '/model':
                    explicit_option = explicit_connection_option(CREDENTIALS)
                    async with session.modal() as modal:
                        try:
                            picked = await modelsdev.run_model_picker_async(
                                input_fn=modal.prompt,
                                credentials=CREDENTIALS,
                                explicit_connection=explicit_option)
                        except (OSError, json.JSONDecodeError) as e:
                            # models.dev unreachable (network errors) or answered
                            # with non-JSON garbage: fall back to the current
                            # provider's own /models list in the same modal.
                            print(f"models.dev unavailable: {e}",
                                  file=sys.stderr)
                            sys.stderr.flush()
                            models_list = await load_models_async()
                            selected_model = (
                                await modelsdev.run_flat_model_picker_async(
                                    modal.prompt, models_list,
                                    explicit_connection=explicit_option))
                            if selected_model:
                                if isinstance(
                                        selected_model,
                                        modelsdev.ExplicitConnectionOption):
                                    apply_runtime_config(
                                        build_config_from_env(
                                            credentials=CREDENTIALS))
                                    selected_label = selected_model.model
                                    selected_via = " via explicit LOKI_*"
                                else:
                                    reinstall_provider(
                                        model=selected_model,
                                        models_url=(
                                            current_config().chat_provider.models_url
                                            if current_config() else None),
                                    )
                                    selected_label = selected_model
                                    selected_via = ""
                                descriptor = active_connection_descriptor()
                                if descriptor is not None:
                                    set_session_connection(descriptor)
                                save_chat_log()
                                print(
                                    f"Selected model: {selected_label}"
                                    f"{selected_via}",
                                      file=sys.stderr)
                                sys.stderr.flush()
                                continue
                            print("Model selection cancelled.",
                                  file=sys.stderr)
                            sys.stderr.flush()
                            continue
                    if picked is None:
                        # User cancelled at either menu; keep the current model.
                        print("Model selection cancelled.", file=sys.stderr)
                        sys.stderr.flush()
                        continue
                    try:
                        if isinstance(
                                picked,
                                modelsdev.ExplicitConnectionOption):
                            apply_runtime_config(build_config_from_env(
                                credentials=CREDENTIALS))
                            via = " via explicit LOKI_*"
                        else:
                            provider_id, provider_entry, model_entry = picked
                            apply_runtime_config(
                                config_from_modelsdev_selection(
                                    provider_id,
                                    provider_entry,
                                    model_entry,
                                    CREDENTIALS,
                                ))
                            via = (
                                f" via {provider_id}" if provider_id else "")
                    except (protocols.ProtocolError, ValueError) as e:
                        print(f"Could not switch model: {e}",
                              file=sys.stderr)
                        sys.stderr.flush()
                        continue
                    descriptor = active_connection_descriptor()
                    if descriptor is not None:
                        set_session_connection(descriptor)
                    save_chat_log()
                    print(f"Selected model: {current_model()}{via}", file=sys.stderr)
                    sys.stderr.flush()
                    continue
                case '/pwd':
                    print_shell_cwd()
                    continue
                case '/ps':
                    print(run_jobs())
                    continue
                case _ if command_text == '/cd' or command_text.startswith('/cd '):
                    change_shell_cwd_from_text(command_text[3:].strip())
                    continue
                case _:
                    if command_text.startswith('!'): # direct command execution
                        cmd = user_in[1:].strip()
                        print(f"{computer}: [Running local command: {cmd}]")
                        cmd_output = await run_bash_async(cmd)
                        print(cmd_output) # Show output to you in the terminal
                        # Morph the user input so the AI sees exactly what you did and the result
                        user_in = f"I ran the local command `{cmd}`.\nOutput:\n```\n{cmd_output}\n```"
                    else:
                        pass

            if current_config() is None:
                sys.stdout.flush()
                print("No provider configured; use /model to select one.",
                      file=sys.stderr)
                sys.stderr.flush()
                continue
            if not current_model():
                sys.stdout.flush()
                print("No model selected; use /model or set LOKI_MODEL.",
                      file=sys.stderr)
                sys.stderr.flush()
                continue

            current_transcript().append(formats.message_item("user", user_in))
            mark_chat_log_dirty()

            try:
                # Ctrl+C is a per-turn request. A Ctrl+C used to cancel an
                # earlier prompt or turn must not poison the next model call.
                session.reader.cancel_requested = False
                session.reader.cancel_event.clear()
                await run_terminal_turn_async(
                    current_transcript(),
                    cancel_check=lambda: session.reader.cancel_requested,
                    cancel_event=session.reader.cancel_event)
            except KeyboardInterrupt:
                terminal.reset_colors_and_flags()
                print("\n\n? [EMERGENCY STOP] Agent execution cancelled by user!")
                # Keep the provider response.  Complete every outstanding call
                # with an explicit local error so the next protocol projection
                # has no dangling call/result pair.
                for call in formats.pending_tool_calls(current_transcript()):
                    current_transcript().append(formats.tool_result_for_call(
                        call,
                        "Tool call not executed because the user interrupted "
                        "the turn.",
                        is_error=True,
                    ))
                mark_chat_log_dirty()
                continue

    return 0


def initialize_terminal_overlay(active_terminal):
    # The input area renders a synthetic reverse-video caret, so the real
    # cursor is hidden for the whole session; restore_terminal_overlay (which
    # clean_up runs on every exit path) shows it again.
    active_terminal.hide_cursor()
    active_terminal.enable_bracketed_paste_mode()
    active_terminal.enable_origin_mode()
    active_terminal.clear_to_end_of_screen()
    active_terminal.reset_colors_and_flags()
    active_terminal.set_clipping_region(*terminals.output_area)
    active_terminal.goto_position(1, 1)
    active_terminal.flush()


def restore_terminal_overlay(active_terminal, run_step=lambda step: step()):
    """Remove Loki's overlay without clearing ordinary terminal contents."""
    terminals.refresh_terminal_layout()
    run_step(active_terminal.disable_bracketed_paste_mode)
    run_step(active_terminal.disable_clipping_regions)
    run_step(active_terminal.disable_origin_mode)
    run_step(active_terminal.reset_colors_and_flags)
    # DECSTBM and DECOM reset the cursor to the terminal home position. Move
    # it to the first row formerly owned by the overlay before erasing, or
    # ED(0) would still erase the entire visible display from home.
    run_step(lambda: active_terminal.goto_position(
        terminals.input_area[0], 1))
    run_step(active_terminal.clear_to_end_of_screen)
    # Reveal the real cursor only once it sits at its final resting position.
    run_step(active_terminal.show_cursor)
    run_step(active_terminal.flush)


def main() -> int:
    global CREDENTIALS
    CREDENTIALS = CredentialStore.capture(os.environ)
    try:
        configure_tool_hook_pipeline()
    except tool_runtime.HookConfigurationError as error:
        print(f"Hook configuration error: {error}", file=sys.stderr)
        sys.stderr.flush()
        return 2
    cleanup_done = False
    cleanup_failed = False

    def clean_up_step(thunk):
        nonlocal cleanup_failed
        try:
            thunk()
        except Exception as e:
            cleanup_failed = True
            # Terminal cleanup is best-effort: one failed restore step should
            # not prevent later steps from disabling modes or resetting colors.
            print(f"Cleanup error: {type(e).__name__}: {e}", file=sys.stderr)
            sys.stderr.flush()

    def clean_up(*args, **kwargs):
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        if current_chat_log_path() is not None:
            clean_up_step(save_chat_log)
        clean_up_step(
            lambda: restore_terminal_overlay(terminal, clean_up_step))

    def clean_up_and_exit(*args, **kwargs):
        clean_up(*args, **kwargs)
        sys.exit(1)

    signal.signal(signal.SIGTERM, clean_up_and_exit)
    signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT,])

    initialize_terminal_overlay(terminal)

    exit_status = 1
    try:
        exit_status = asyncio.run(async_main(sys.argv[1:]))
    finally:
        clean_up()
    if exit_status == 0 and cleanup_failed:
        return 1
    return exit_status


if __name__ == '__main__':
    raise SystemExit(main())
