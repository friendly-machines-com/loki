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
import time
import urllib.parse
import socket
import stat
from dataclasses import replace
from functools import partial

from . import connections
from . import model_requests
from . import files
from . import jobs
from . import web_tools
from . import tool_contracts
from . import formats
from . import models as modelsdev
from . import paths
from . import prompts
from . import protocols
from . import process_outputs
from . import authentications
from . import savefiles
from . import tool_runtime
from .connections import ConnectionDescriptor
from .credentials import (
    CredentialInventory,
    CredentialStore,
    is_credential_name,
)
from .sessions import (
    Session,
    conversation_id_for_path,
    default_session,
)


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


def current_config():
    return current_session().runtime_config


def current_model() -> str:
    return current_session().model


def current_reasoning_effort_preference() -> str | None:
    return current_session().reasoning_effort_preference


def current_reasoning_effort_profile():
    config = current_config()
    return (
        getattr(config, "reasoning_effort_profile", None)
        if config is not None else None
    )


def effective_reasoning_effort() -> str | None:
    """The effort selection that applies to the current connection.

    A saved preference stays dormant when the selected model does not
    advertise it, or while thinking is off on a wire whose effort only
    applies to thinking. Anthropic effort guides non-thinking output
    too and stays active.
    """
    profile = current_reasoning_effort_profile()
    preference = current_reasoning_effort_preference()
    if profile is None or not profile.supports(preference):
        return None
    config = current_config()
    if config is None:
        return None
    spec = config.chat_provider.reasoning_spec
    mode, _budget, _retention = effective_thinking_settings()
    if (mode == "off" and spec is not None
            and spec.wire_format != "anthropic"):
        return None
    return preference


def reasoning_effort_default() -> str | None:
    """The default effort a request without a selection uses: the entry's
    own advertised default, or the authenticated Codex catalog's."""
    profile = current_reasoning_effort_profile()
    if profile is None:
        return None
    if profile.default is not None:
        return profile.default
    provider = current_config().chat_provider
    if (provider.provider_id == modelsdev.OPENAI_SUBSCRIPTION_PROVIDER_ID
            and provider.openai_request_profile is not None):
        return provider.openai_request_profile.default_reasoning_level
    return None


def reasoning_effort_default_text() -> str | None:
    profile = current_reasoning_effort_profile()
    if profile is None:
        return None
    preference = current_reasoning_effort_preference()
    name = "Model default"
    default = reasoning_effort_default()
    if default is not None:
        name += f" ({default})"
    if preference is not None and not profile.supports(preference):
        name += f" - preferred {preference} is unavailable"
    return name


def reasoning_effort_status_text() -> str | None:
    if current_reasoning_effort_profile() is None:
        return None
    return effective_reasoning_effort() or reasoning_effort_default() or "unknown"


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


def current_agent_mode() -> str:
    return current_session().agent_mode


def set_reasoning_effort(value: str | None):
    """Apply one advertised effort value; the /thinking effort control."""
    set_thinking_controls({"effort": value})


# -- thinking controls ------------------------------------------------------


def _thinking_connection():
    config = current_config()
    if config is None:
        return None
    return {
        "model": config.model,
        "protocol": config.chat_provider.kind,
        "endpoint": config.chat_provider.chat_url,
    }


def effective_thinking_settings():
    """Mode, allowance, and retention that apply to the current connection.

    Saved preferences that do not apply here stay dormant (and surface as
    inactive in the status output), never erased: switching back to a
    connection where they apply restores them.
    """
    session = current_session()
    config = current_config()
    if config is None or not config.model:
        return None, None, "default"
    provider = config.chat_provider
    model = config.model
    mode = session.thinking_mode
    if mode is not None and mode not in provider.thinking_modes(model):
        mode = None
    budget = session.thinking_budget
    if budget is not None and not (
            mode == "manual"
            and provider.kind == protocols.ANTHROPIC_MESSAGES):
        budget = None
    retention = session.reasoning_retention
    if (retention == "preserve"
            and provider.reasoning_preservation(model) is None):
        retention = "default"
    return mode, budget, retention


def capture_turn_settings() -> model_requests.TurnThinkingSettings:
    """Project the saved thinking controls into this turn's settings.

    This is the turn-admission projection and the single point that
    validates the joint state: assignment accepted each control
    independently, so an invalid combination surfaces here, once, as an
    error -- never silently, and never as an artifact of the order the
    controls were set in. Dormant-but-applicable handling stays as it
    was: a saved control that does not apply to this connection is
    dropped from the projection, not from the session.
    """
    session = current_session()
    config = current_config()
    if config is None or not config.model:
        return model_requests.TurnThinkingSettings(traces=session.reasoning_traces)
    provider = config.chat_provider
    model = config.model
    settings = model_requests.TurnThinkingSettings(
        effort=effective_reasoning_effort(),
        mode=session.thinking_mode,
        budget=session.thinking_budget,
        retention=session.reasoning_retention,
        traces=session.reasoning_traces,
    )
    if settings.mode == "off":
        # The allowance and (outside Anthropic, whose effort guides
        # non-thinking output) the effort apply to thinking only; with
        # thinking off they are dormant, not erased.
        settings = replace(settings, budget=None)
        spec = provider.reasoning_spec
        if spec is None or spec.wire_format != "anthropic":
            settings = replace(settings, effort=None)
    if settings.mode != "manual" or provider.kind != protocols.ANTHROPIC_MESSAGES:
        # An allowance rides a request only in manual mode on the wire
        # that spells manual; anything else is a dormant saved value.
        settings = replace(settings, budget=None)
    if settings.retention == "preserve":
        if provider.reasoning_preservation(model) is None:
            settings = replace(settings, retention="default")
    # The wire contract sees what a request would carry and owns the
    # joint verdict: an invalid combination is an error here, once,
    # instead of an order-dependent rejection at assignment.
    provider.validate_thinking(
        model, settings.effort, settings.mode,
        settings.budget if settings.mode == "manual" else None,
        settings.retention)
    return settings


def set_thinking_controls(changes, *, persist=True):
    """Assign thinking controls in any order.

    ``changes`` maps a control name ("effort", "mode", "budget",
    "retention") to its requested value; ``None`` clears it. Assignment
    validates field-level syntax only: the joint state becomes valid or
    invalid together, and that verdict belongs to projection
    (``capture_turn_settings``, the status text) -- never to the order
    of assignment. A saved value that does not apply to the current
    connection stays saved and dormant; a value the endpoint rejects
    surfaces as the endpoint's error, in the open.
    """
    session = current_session()
    config = current_config()
    if config is None or not config.model:
        raise ValueError("no model is selected")
    names = {
        "effort": "reasoning_effort_preference",
        "mode": "thinking_mode",
        "budget": "thinking_budget",
        "retention": "reasoning_retention",
    }
    if set(changes) - set(names):
        raise ValueError("unknown thinking control")
    candidate = {
        name: getattr(session, attribute)
        for name, attribute in names.items()}
    candidate.update(changes)
    if candidate["retention"] not in ["default", "preserve"]:
        raise ValueError("expected retention default or preserve")
    if candidate["mode"] is not None and candidate["mode"] not in [
            "on", "off", "manual", "adaptive", "between-tools"]:
        raise ValueError("unknown thinking mode")
    if candidate["budget"] is not None and (
            not isinstance(candidate["budget"], int)
            or isinstance(candidate["budget"], bool)
            or candidate["budget"] < 0):
        raise ValueError("thinking budget must be a nonnegative integer")
    if candidate["effort"] is not None:
        modelsdev.validate_reasoning_effort(candidate["effort"])
        # Advertised choices are model facts, not joint state: an
        # unlisted value is invalid in every order.
        profile = current_reasoning_effort_profile()
        if profile is None or not profile.supports(candidate["effort"]):
            raise ValueError(
                f"reasoning effort {candidate['effort']!r} is not "
                "available for the selected model")
    for name, value in changes.items():
        setattr(session, names[name], value)
    mark_chat_log_dirty()
    if persist:
        try:
            save_chat_log()
        except OSError as error:
            raise OSError(
                "thinking settings are active in this session but could "
                f"not be saved: {error}") from error


def thinking_command(argument):
    """Parse and apply one /thinking command; return its status text."""
    parts = argument.split()
    if len(parts) % 2:
        raise ValueError(
            "usage: /thinking [effort VALUE] [mode VALUE] [budget N] "
            "[retention default|preserve]")
    changes = {}
    for name, value in zip(parts[::2], parts[1::2]):
        if (name not in ["effort", "mode", "budget", "retention"]
                or name in changes):
            raise ValueError("expected distinct thinking field/value pairs")
        if value == "default":
            value = "default" if name == "retention" else None
        elif name == "effort":
            value = modelsdev.validate_reasoning_effort(value)
        elif name == "budget":
            try:
                value = int(value)
            except ValueError as error:
                raise ValueError(
                    "thinking budget must be a positive integer") from error
        changes[name] = value
    if changes:
        set_thinking_controls(changes)
    return thinking_status_text()


def set_reasoning_traces(value, *, persist=True):
    """Show or hide readable thinking output. Observation only: traces
    never change effort, enablement, allowance, or preservation."""
    if value not in ["off", "on"]:
        raise ValueError("expected off or on")
    current_session().reasoning_traces = value
    mark_chat_log_dirty()
    if persist:
        save_chat_log()


def trace_command(argument):
    """Parse and apply one /trace command; return its status text."""
    parts = argument.split()
    if parts not in [
            [], ["thinking"], ["thinking", "on"], ["thinking", "off"]]:
        raise ValueError("usage: /trace thinking [on|off]")
    if len(parts) == 2:
        set_reasoning_traces(parts[1])
    return reasoning_traces_status_text()


def thinking_status_text():
    config = current_config()
    if config is None or not config.model:
        return "Thinking: no model selected"
    provider = config.chat_provider
    model = config.model
    session = current_session()
    mode, budget, retention = effective_thinking_settings()
    modes = provider.thinking_modes(model)
    lines = [f"Thinking controls for {model}"]
    profile = current_reasoning_effort_profile()
    if profile is not None:
        lines.append("Effort choices: default, " + ", ".join(
            value for value in profile.values
            if isinstance(value, str)))
    if modes:
        lines.append(
            "Mode: " + (mode or provider.thinking_default_mode(model)
                        or "not selected"))
        lines.append("Mode choices: default, " + ", ".join(modes))
    if provider.kind == protocols.ANTHROPIC_MESSAGES and "manual" in modes:
        lines.append("Allowance: " + (
            f"{budget} thinking tokens" if budget is not None
            else "default"))
    if provider.reasoning_preservation(model) is not None:
        lines.append(f"Retention: {retention} (default, preserve)")
    for label, saved, active in [
            ("effort", session.reasoning_effort_preference,
             effective_reasoning_effort()),
            ("mode", session.thinking_mode, mode),
            ("budget", session.thinking_budget, budget),
            ("retention",
             None if session.reasoning_retention == "default"
             else session.reasoning_retention,
             None if retention == "default" else retention)]:
        if saved != active:
            lines.append(f"Inactive {label} preference: {saved}")
    if profile is None and not modes:
        lines.append("No verified computation controls for this connection.")
    if profile is not None:
        effective = effective_reasoning_effort()
        if effective is not None:
            lines.append(f"Effort: {effective}")
        else:
            default = reasoning_effort_default()
            lines.append(
                f"Effort: not selected (model default: {default})"
                if default is not None else "Effort: not selected")
    return "\n".join(lines)


def reasoning_traces_status_text():
    session = current_session()
    text = ("Thinking traces: " + session.reasoning_traces
            + " (display only; use /thinking for model controls)")
    config = current_config()
    if config is not None and session.reasoning_traces == "on":
        provider = config.chat_provider
        mode, _budget, _retention = effective_thinking_settings()
        effort = effective_reasoning_effort()
        if ((mode or provider.thinking_default_mode(config.model)) == "off"
                or effort == "none"
                or (effort is None
                    and reasoning_effort_default() == "none")):
            text += "; thinking off"
    return text


def current_job_manager():
    session = current_session()
    if session.job_manager is None:
        session.job_manager = jobs.JobManager(LOKI_JOB_STATE_DIR)
    return session.job_manager


# --------------------------------------------------------------------------

computer = socket.gethostname()

ERROR_COLOR = 1
TOOL_CALL_COLOR = 5

MAX_LOOP_LIMIT = 500
TODO_MAX_TODOS = 100
SKILL_MAX_BYTES = 100_000
LOKI_CONFIG_DIR_NAME = paths.LOKI_CONFIG_DIR_NAME
LOKI_CONFIG_DIR = paths.loki_config_dir()
LOKI_STATE_DIR = paths.loki_state_dir()
LOKI_JOB_STATE_DIR = os.path.join(LOKI_STATE_DIR, "jobs")
STARTUP_CWD = os.getcwd()


CHAT_LOG_DIR = paths.chat_log_dir_for(STARTUP_CWD)


CREDENTIALS: CredentialStore | CredentialInventory | None = None


async def refresh_connection_descriptor_async(
        descriptor, credential_authority, diagnostic_writer=None):
    """Refresh a subscription descriptor, retaining it only on fetch failure."""
    if not connections._is_openai_subscription_connection(descriptor):
        return descriptor
    try:
        response = await modelsdev.fetch_openai_subscription_models(
            credential_authority)
        catalog = modelsdev.add_openai_subscription_catalog(
            {}, response, diagnostic_writer=diagnostic_writer)
    except (OSError, authentications.CredentialError) as error:
        if diagnostic_writer is not None:
            diagnostic_writer(
                "OpenAI subscription model discovery failed; using saved "
                f"request profile: {error}")
        catalog = {}
    return connections.reconcile_connection_descriptor(descriptor, catalog)


def active_connection_descriptor() -> ConnectionDescriptor | None:
    config = current_config()
    if config is None:
        return None
    return connections.connection_descriptor_from_config(config)


def apply_runtime_config(config: connections.RuntimeConfig):
    # model is derived from runtime_config (Session.model property); nothing
    # else to publish.
    current_session().runtime_config = config


_UNSET = object()


def reinstall_provider(*, model=None, url=None, provider_kind=None,
                       models_url=None, max_tokens=None, anthropic_version=None,
                       auth_header=_UNSET, provider_id=None, provider_name=None,
                       credential_ref=_UNSET, auth_scheme=_UNSET,
                       model_status=_UNSET, stream=None, prompt_cache=None,
                       openai_request_profile=_UNSET,
                       reasoning_effort_profile=_UNSET,
                       context_capacity=_UNSET):
    """Rebuild and swap RUNTIME_CONFIG (and its Provider) mid-session.

    Overrides default to the current runtime config, so a bare call reinstates
    the Provider from the existing settings
    """
    current = current_config()
    if current is None:
        raise RuntimeError("cannot reinstall provider before startup config is applied")
    current_provider = current.chat_provider
    current_auth = current.auth_spec
    new_url = (
        url if url is not None else current_provider.input_url)
    new_kind = (
        provider_kind if provider_kind is not None else current_provider.kind)
    new_model = model if model is not None else current.model
    new_anthropic_version = (
        anthropic_version
        if anthropic_version is not None
        else current_provider.headers.get(
            "anthropic-version", "2023-06-01"))
    new_provider_id = (
        provider_id
        if provider_id is not None else current_provider.provider_id)
    provider_connection_changed = (
        (provider_kind is not None
         and provider_kind != current_provider.kind)
        or (url is not None and url != current_provider.input_url)
        or (provider_id is not None
            and provider_id != current_provider.provider_id))
    auth_identity_changed = (
        provider_connection_changed
        or credential_ref is not _UNSET)
    if auth_header is _UNSET:
        new_auth_header = (
            None
            if auth_identity_changed
            else (
                current_auth.header_name
                if current_auth is not None
                and current_auth.scheme == "custom" else None))
    else:
        new_auth_header = auth_header
    new_credential_ref = (
        (None
         if provider_connection_changed
         else (current_auth.credential
               if current_auth is not None else None))
        if credential_ref is _UNSET else credential_ref)
    if auth_scheme is _UNSET:
        new_auth_scheme = (
            None
            if auth_identity_changed or auth_header is not _UNSET
            else (
                current_auth.scheme
                if current_auth is not None else None))
    else:
        new_auth_scheme = auth_scheme
    same_catalog_entry = (
        new_model == current.model
        and new_url == current_provider.input_url
        and new_kind == current_provider.kind
        and new_provider_id == current_provider.provider_id)
    if model_status is _UNSET:
        new_model_status = (
            current.model_status if same_catalog_entry else None)
    else:
        new_model_status = model_status
    if openai_request_profile is _UNSET:
        # The authenticated request profile is one indivisible per-model
        # contract.
        # Preserve it only for the same catalog leaf; carrying even part of it
        # across a model/provider change would manufacture capabilities.
        new_openai_request_profile = (
            current_provider.openai_request_profile
            if same_catalog_entry else None)
    else:
        new_openai_request_profile = openai_request_profile
    if reasoning_effort_profile is _UNSET:
        new_reasoning_effort_profile = (
            current.reasoning_effort_profile
            if same_catalog_entry else None)
    else:
        new_reasoning_effort_profile = reasoning_effort_profile
    apply_runtime_config(connections.make_runtime_config(
        new_url,
        new_kind,
        model=new_model,
        # models_url is passed through as-is (None -> derive from new_url);
        # never carry the previous provider's models URL across a provider
        # switch.
        models_url=models_url,
        max_tokens=(
            max_tokens
            if max_tokens is not None else current_provider.max_tokens),
        anthropic_version=new_anthropic_version,
        auth_header=new_auth_header,
        auth_scheme=new_auth_scheme,
        provider_id=new_provider_id,
        provider_name=(provider_name if provider_name is not None
                       else current_provider.provider_name),
        credential_ref=new_credential_ref,
        model_status=new_model_status,
        stream=stream if stream is not None else current.stream,
        prompt_cache=(
            prompt_cache
            if prompt_cache is not None
            else current_provider.prompt_cache),
        openai_request_profile=new_openai_request_profile,
        reasoning_effort_profile=new_reasoning_effort_profile,
        context_capacity=(
            (current.context_capacity if same_catalog_entry else None)
            if context_capacity is _UNSET else context_capacity),
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


file_state = LruCache(files.READ_PATHS_LIMIT)
_webfetch_cache = LruCache(web_tools.WEBFETCH_CACHE_MAX_ENTRIES)  # url -> (fetched_at_epoch, content_text, content_type, final_url, status)


def _resolve_path(path: str, base_dir: str | None = None) -> str:
    return files.resolve_path(path, base_dir if base_dir is not None else current_cwd())


def display_path(path: str) -> str:
    return path


def change_shell_cwd(path: str | None = None) -> str:
    session = current_session()
    target = (path or "").strip()
    if not target:
        target = "~"
    elif target == "-":
        target = session.previous_shell_cwd
    resolved = _resolve_path(os.path.expanduser(target))
    observed = os.stat(resolved)
    if not stat.S_ISDIR(observed.st_mode):
        raise FileNotFoundError(resolved)
    resolved = files.existing_target(resolved, observed)
    old_cwd = session.shell_cwd
    session.shell_cwd = resolved
    session.previous_shell_cwd = old_cwd
    return session.shell_cwd


ToolSchemaError = tool_runtime.ToolSchemaError


SCHEMA_ANNOTATION_KEYS = tool_runtime.SCHEMA_ANNOTATION_KEYS
SCHEMA_VALIDATION_KEYS = tool_runtime.SCHEMA_VALIDATION_KEYS
SCHEMA_ALLOWED_KEYS = tool_runtime.SCHEMA_ALLOWED_KEYS


def _schema_path(path: str, key) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


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
        tool_kind = tool.get("type")
        if tool_kind == "function":
            function = tool.get("function", {})
            name = function.get("name")
            parameters = function.get("parameters")
        elif tool_kind == "custom":
            name = tool.get("name")
            # A Responses custom tool receives one opaque freeform string.
            # Its optional wire-level grammar constrains model generation,
            # while Loki's runtime boundary only needs to enforce that type.
            parameters = {"type": "string"}
        else:
            continue
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
            "tool_kind": tool_kind,
            "schema": parameters,
            "semantics": copy.deepcopy(
                handler.get("semantics", {})),
            "handler": sync_handler,
            "async_handler": async_handler,
            "explore": handler.get("explore", False),
            "plan": handler.get("plan", False),
        }

    extra_handlers = sorted(set(handlers) - seen)
    if extra_handlers:
        raise ToolSchemaError(f"handler without tool definition: {', '.join(extra_handlers)}")
    return registry


async def run_bash_async(command: str, timeout: int | None = None, description: str = "",
                         run_in_background: bool = False,
                         cancel_event: asyncio.Event | None = None,
                         cwd: str | None = None, *, structured=False):
    return await current_job_manager().run_shell(
        command, timeout=timeout, description=description,
        run_in_background=run_in_background,
        cancel_event=cancel_event,
        cwd=cwd if cwd is not None else current_cwd(), structured=structured)


def run_jobs() -> str:
    return current_job_manager().list_jobs()


def run_ps(argument: str = "") -> str:
    # /ps is immediate user job control, allowed even during a running turn.
    # Keep it synchronous: JobManager transitions cannot interleave, and this
    # must never reset cancellation, mutate the transcript, or start inference.
    parts = argument.split()
    if not parts or parts == ["all"]:
        listing = current_job_manager().list_jobs(show_all=parts == ["all"])
        if not parts:
            listing += "\nfailed = job setup/launch failure, not command exit code."
        return (listing + "\n/ps all - history; /ps ID - tail; "
                "/ps stop ID - stop; /ps kill ID - force termination")
    if len(parts) == 1 and parts[0].isdecimal():
        return run_job_status(parts[0])
    if (len(parts) == 2 and parts[0] in ["stop", "kill"]
            and parts[1].isdecimal()):
        return run_job_stop(parts[1], force=parts[0] == "kill")
    return "usage: /ps [all | ID | stop ID | kill ID]"


def run_job_status(job_id: str, tail_chars: int = tool_contracts.JOB_TAIL_CHARS, *, structured=False):
    if not job_id:
        return "Error: job_id is required"
    try:
        tail_chars = int(tail_chars)
    except (TypeError, ValueError) as e:
        return f"Error: invalid tail_chars: {e}"
    if tail_chars < 0:
        return "Error: tail_chars must be non-negative"
    return current_job_manager().job_status(job_id, tail_chars=tail_chars, structured=structured)


def run_job_stop(job_id: str, force: bool = False) -> str:
    if not job_id:
        return "Error: job_id is required"
    return current_job_manager().stop_job(job_id, force=bool(force))


def run_read(file_path: str, offset: int | None = None, limit: int | None = None,
             *, structured=False, cwd=None):
    return files.run_read(file_path, offset, limit, structured=structured,
                          cwd=cwd if cwd is not None else current_cwd(), observations=file_state)


def run_write(file_path: str, content: str, *, cwd=None) -> str:
    return files.run_write(file_path, content,
                           cwd=cwd if cwd is not None else current_cwd(), observations=file_state)


def run_edit(file_path: str, old_string: str, new_string: str,
             replace_all: bool = False, *, cwd=None) -> str:
    return files.run_edit(file_path, old_string, new_string, replace_all,
                          cwd=cwd if cwd is not None else current_cwd(), observations=file_state)


def run_glob(pattern: str, path: str | None = None, *,
             hidden: bool = False, no_ignore: bool = False) -> str:
    return asyncio.run(run_glob_async(
        pattern, path, hidden=hidden, no_ignore=no_ignore))


async def run_glob_async(pattern: str, path: str | None = None,
                         cancel_event: asyncio.Event | None = None, *,
                         hidden: bool = False, no_ignore: bool = False, cwd=None) -> str:
    cwd = cwd if cwd is not None else current_cwd()
    return await files.run_glob_async(
        pattern, path, cancel_event, hidden=hidden, no_ignore=no_ignore, cwd=cwd,
        run_exec=partial(current_job_manager().run_exec, cwd=cwd))


def run_grep(pattern: str, path: str | None = None, glob: str | None = None,
             output_mode: str = "files_with_matches", **kwargs) -> str:
    return asyncio.run(run_grep_async(pattern, path, glob, output_mode, **kwargs))


async def run_grep_async(pattern: str, path: str | None = None, glob: str | None = None,
                         output_mode: str = "files_with_matches",
                         cancel_event: asyncio.Event | None = None, *, cwd=None, **kwargs) -> str:
    cwd = cwd if cwd is not None else current_cwd()
    return await files.run_grep_async(
        pattern, path, glob, output_mode, cancel_event, cwd=cwd,
        run_exec=partial(current_job_manager().run_exec, cwd=cwd), **kwargs)


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


def _handle_bash(args: dict):
    return asyncio.run(_handle_bash_async(args))


async def _handle_bash_async(args: dict, extra_context=None):
    return await run_bash_async(args["command"],
                                timeout=args.get("timeout"),
                                description=args.get("description", ""),
                                run_in_background=args.get("run_in_background", False),
                                cancel_event=(extra_context or {}).get("cancel_event")
                                if isinstance(extra_context, dict) else None,
                                cwd=(extra_context or {}).get("tool_cwd"),
                                structured=True)


def _handle_read(args: dict):
    return run_read(args["file_path"], offset=args.get("offset"), limit=args.get("limit"),
                    structured=True)


def _handle_write(args: dict) -> str:
    return run_write(args["file_path"], args["content"])


def _handle_edit(args: dict) -> str:
    return run_edit(args["file_path"], args["old_string"],
                    args["new_string"], replace_all=args.get("replace_all", False))


def _handle_glob(args: dict) -> str:
    return run_glob(args["pattern"], args.get("path"),
                    hidden=args.get("hidden", False),
                    no_ignore=args.get("no_ignore", False))


async def _handle_glob_async(args: dict, extra_context=None) -> str:
    return await run_glob_async(
        args["pattern"], args.get("path"),
        cancel_event=(extra_context or {}).get("cancel_event"),
        hidden=args.get("hidden", False),
        no_ignore=args.get("no_ignore", False))


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
                                cancel_event=(extra_context or {}).get(
                                    "cancel_event"),
                                **extra)


def _handle_jobs(args: dict) -> str:
    return run_jobs()


def _handle_job_status(args: dict):
    return run_job_status(args["job_id"], args.get("tail_chars", tool_contracts.JOB_TAIL_CHARS), structured=True)


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
    context = extra_context or {}
    return await run_agent_async(args.get("description", ""),
                                 args["prompt"],
                                 args.get("run_in_background", False),
                                 args.get("subagent_type", "Explore"),
                                 cancel_event=context.get("cancel_event"),
                                 reasoning_effort=context.get(
                                     "reasoning_effort", _UNSET),
                                 thinking=context.get("thinking"))


def _handle_skill(args: dict) -> str:
    return run_skill(args["skill"], args.get("args"))


async def _handle_webfetch_async(args: dict, extra_context=None) -> str:
    context = extra_context or {}
    return await run_webfetch_async(
        args["url"], args["prompt"],
        cancel_event=context.get("cancel_event"),
        reasoning_effort=context.get("reasoning_effort", _UNSET),
        thinking=context.get("thinking"))


async def _handle_websearch_async(args: dict, extra_context=None) -> str:
    return await web_tools.run_websearch_async(
        args["query"], allowed_domains=args.get("allowed_domains"),
        blocked_domains=args.get("blocked_domains"),
        cancel_event=(extra_context or {}).get("cancel_event"))


async def _handle_ask_async(args: dict, extra_context=None) -> str:
    asker = (extra_context or {}).get("ask_user")
    if asker is None:
        return (
            "Ask is not available in this session: it has no interactive "
            "user who could answer.")
    options = args.get("options")
    if not isinstance(options, list) or len(options) < 2:
        return "Ask requires at least two options."
    labels = []
    for option in options:
        if not isinstance(option, dict):
            return "Ask options must be objects with a label."
        label = option.get("label")
        if not isinstance(label, str) or not label.strip():
            return "Ask option labels must be nonempty strings."
        labels.append(label)
    if len(set(labels)) != len(labels):
        return "Ask option labels must be unique so an answer is unambiguous."
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        return "Ask requires a nonempty question."
    outcome = await asker(
        question,
        [(option["label"], option.get("description")) for option in options],
        multi_select=bool(args.get("multi_select", False)),
    )
    if not isinstance(outcome, dict):
        return "The question could not be delivered to the user."
    action = outcome.get("action")
    if action == "declined":
        # A refusal is information, not an error: the turn continues.
        return "The user declined to answer the question."
    if action != "answered":
        return "The user dismissed the question without answering."
    answer = outcome.get("answer")
    if isinstance(answer, list):
        rendered = ", ".join(f'"{item}"' for item in answer)
    else:
        rendered = f'"{answer}"'
    note = outcome.get("custom")
    if note:
        return (f'The user answered "{question}" with {rendered}. '
                f'The user also noted: {note}')
    return f'The user answered "{question}" with {rendered}.'


def _looks_like_tool_error(content: str) -> bool:
    return content.startswith("Error: ") or content.startswith("Failed")


async def with_exception_to_tool_result_async(context: str, thunk) -> dict:
    try:
        content = await thunk()
    except (KeyboardInterrupt, SystemExit):
        raise
    except FileNotFoundError as e:
        return tool_runtime.tool_result(False, f"Error while {context}: file not found: {e}")
    except PermissionError as e:
        return tool_runtime.tool_result(False, f"Error while {context}: permission denied: {e}")
    except TimeoutError as e:
        return tool_runtime.tool_result(False, f"Error while {context}: timed out: {e}")
    except OSError as e:
        return tool_runtime.tool_result(False, f"Error while {context}: OS error: {e}")
    except ValueError as e:
        return tool_runtime.tool_result(False, f"Error while {context}: invalid value: {e}")
    except Exception as e:
        return tool_runtime.tool_result(False, f"Failed while {context}: {type(e).__name__}: {e}")

    if (isinstance(content, dict)
            and {"ok", "content"} <= set(content) <= {"ok", "content", "process_output"}
            and isinstance(content["ok"], bool)):
        return tool_runtime.tool_result(
            content["ok"], content["content"], process_output=content.get("process_output"))
    text = str(content)
    return tool_runtime.tool_result(not _looks_like_tool_error(text), text)


def _refused_tool_hint(fn_name, args, allowed):
    """Best-effort read-only alternative for a refused call. """
    if fn_name != "Bash":
        return None
    try:
        command = args["command"]
        if "grep" in command and "Grep" in TOOL_REGISTRY and (allowed is None or "Grep" in allowed):
            return 'This command looks like a grep; the Grep tool is still available.'
    except Exception:
        return None


def _tool_access_error(fn_name: str, args=None, allowed=None, extra_context=None):
    inhibit_edits = (extra_context or {}).get('inhibit_edits', False)
    spec = TOOL_REGISTRY.get(fn_name)
    if spec is None:
        return f"Unknown function: {fn_name}"
    if allowed is not None and fn_name not in allowed:
        return (
            f"Tool {fn_name} not available in this subagent "
            f"(allowed: {sorted(allowed)})")
    if inhibit_edits and spec.get('explore') is not True:
        plan_only_tool = (
            inhibit_edits == "plan mode" and spec.get('plan') is True)
        if not plan_only_tool:
            message = (
                f"Tool {fn_name} is unavailable while "
                f"{inhibit_edits} is active.")
            hint = _refused_tool_hint(fn_name, args, allowed)
            if hint:
                message = f"{message} {hint}"
            return message
    return None


async def dispatch_tool_async(fn_name: str, args: dict, allowed=None, extra_context=None) -> dict:
    spec = TOOL_REGISTRY.get(fn_name)
    if spec is None:
        return tool_runtime.tool_result(False, f"Unknown function: {fn_name}")
    access_error = _tool_access_error(
        fn_name, args=args, allowed=allowed, extra_context=extra_context)
    if access_error:
        return tool_runtime.tool_result(False, access_error)
    cancel_event = (extra_context or {}).get("cancel_event")
    if (cancel_event is not None
            and isinstance(cancel_event, asyncio.Event)
            and cancel_event.is_set()):
        return tool_runtime.tool_result(
            False, "Tool call not executed because the user cancelled "
                   "the turn.")

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
    provider = current_config().chat_provider
    return (
        provider.provider_id
        or provider.provider_name
        or urllib.parse.urlparse(provider.input_url).netloc
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
        file_state.pop(files.observation_key(_resolve_path(path, invocation.cwd)), None)


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
             f"{tool_contracts.READ_DEFAULT_LINES} lines."],
            [{"field": "limit", "value": tool_contracts.READ_DEFAULT_LINES}],
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
    call_id = formats.tool_call_id(call)
    spec = TOOL_REGISTRY.get(fn_name)
    if spec is None:
        result = tool_runtime.tool_result(False, f"Unknown function: {fn_name}")
        on_event({
            "type": "tool_rejected",
            "name": fn_name,
            "call_id": call_id,
            "args": _raw_tool_arguments(call),
            "cwd": current_cwd(),
        })
        return result, None
    call_kind = call.get("tool_kind", "function")
    if call_kind != spec.get("tool_kind", "function"):
        result = tool_runtime.tool_result(
            False,
            f"Tool {fn_name} was called as {call_kind}, but it is "
            f"registered as {spec.get('tool_kind', 'function')}.",
        )
        on_event({
            "type": "tool_rejected",
            "name": fn_name,
            "call_id": call_id,
            "args": _raw_tool_arguments(call),
            "cwd": current_cwd(),
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
        fn_name, args=original_args, allowed=allowed,
        extra_context=extra_context)
    if access_error:
        invocation.denied_reason = access_error
        invocation.denied_by = "loki.capability-gate"
        outcome = tool_runtime.ToolOutcome(
            "rejected", False, False, access_error)
    elif ((extra_context or {}).get("cancel_event") is not None
            and (extra_context or {})["cancel_event"].is_set()):
        invocation.denied_reason = (
            "Tool call not executed because the user cancelled the turn.")
        invocation.denied_by = "loki.cancellation"
        outcome = tool_runtime.ToolOutcome(
            "rejected", False, False, invocation.denied_reason)
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
                "call_id": invocation.call_id,
                "args": copy.deepcopy(
                    invocation.effective_arguments),
                "cwd": invocation.cwd,
            })
            execution_context = dict(extra_context or {})
            execution_context["tool_cwd"] = invocation.cwd
            result = await dispatch_tool_async(
                fn_name,
                invocation.effective_arguments,
                allowed=allowed,
                extra_context=execution_context,
            )
            outcome = tool_runtime.ToolOutcome(
                "success" if result["ok"] else "tool_error",
                True,
                result["ok"],
                result["content"],
                process_output=result.get("process_output"),
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
    if (fn_name == "Read" or (fn_name == "Bash" and not outcome.ok)) and outcome.process_output is None:
        # Rejections, errors and Read notices contain diagnostics, not stdout.
        # Preserve that distinction for terminal display and transcript replay.
        outcome.process_output = process_outputs.ProcessOutput(
            outcome.content, "", "", shell=fn_name == "Bash", read=fn_name == "Read").to_dict()
    outcome.content = _prepend_tool_notes(
        outcome.content, invocation.notes)
    if not outcome.executed:
        on_event({
            "type": "tool_rejected",
            "name": fn_name,
            "call_id": invocation.call_id,
            "args": copy.deepcopy(
                invocation.effective_arguments),
            "cwd": invocation.cwd,
        })
    if outcome.process_output is not None:
        output = process_outputs.ProcessOutput.from_dict(outcome.process_output)
        outcome.process_output = replace(
            output, preamble=_prepend_tool_notes(output.preamble, invocation.notes)).to_dict()
    result = tool_runtime.tool_result(
        outcome.ok, outcome.content, process_output=outcome.process_output)
    return result, _execution_metadata(
        invocation, outcome, defaults)


def _last_user_question(transcript_items: list) -> bool:
    """True when the pending turn opens with a user question.

    run_tool_loop_async calls this exactly once per turn, before the
    first model response, so the transcript's last item is then the
    user's message. If the turn opens as a question, the whole turn's
    non-explore tool calls are refused; the next user turn starts with
    a fresh decision. Detection is the original ed8c342 semantics --
    do not change it silently; QuestionGuardTests pins it.
    """
    if not transcript_items:
        return False
    item = transcript_items[-1]
    if item.get("type") != "message" or item.get("role") != "user":
        return False
    content = item.get("content")
    if not isinstance(content, list):
        return False
    text_blocks = [
        block for block in content
        if isinstance(block, dict) and block.get("type") == "text"]
    if not text_blocks:
        return False
    text = text_blocks[-1].get("text")
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    return stripped.endswith("?") or "what?" in stripped.lower()


def get_tool_loop_extra_context(transcript_items: list):
    inhibit_edits = False
    if current_agent_mode() in ["explore", "plan"]:
        inhibit_edits = f"{current_agent_mode()} mode"
    if _last_user_question(transcript_items):
        inhibit_edits = "answering the user's question"

    return {'inhibit_edits': inhibit_edits}


def record_agent_mode_instruction():
    session = current_session()
    mode = session.agent_mode
    if session.last_instructed_agent_mode == mode:
        return
    descriptions = {
        "normal": (
            "Normal mode is active. Use available tools as needed to answer "
            "or implement the user's request."),
        "explore": (
            "Explore mode is active for this turn. Investigate and answer, "
            "but do not modify files, todos, jobs, or other state."),
        "plan": (
            "Plan mode is active for this turn. Inspect read-only context and "
            "produce a plan; do not modify files, jobs, or other state. "
            "Tracking the plan with TodoWrite is allowed."),
        "edit": (
            "Edit mode is active. Implement the requested changes and verify "
            "them with the available tools."),
    }
    session.transcript_items.append(formats.instruction_item(
        descriptions.get(mode, descriptions["normal"])))
    session.last_instructed_agent_mode = mode
    mark_chat_log_dirty()


def _turn_is_refusal(turn: formats.DecodedTurn) -> bool:
    stop_reason = str(turn.metadata.get("stop_reason") or "").lower()
    if stop_reason in ["refusal", "content_filter", "safety"]:
        return True
    for item in turn.items:
        if item.get("type") != "message":
            continue
        if any(
                isinstance(block, dict)
                and block.get("type") == "refusal"
                for block in item.get("content", [])):
            return True
    protocol_data = turn.metadata.get("protocol_data")
    if isinstance(protocol_data, dict):
        encoded = json.dumps(protocol_data, default=str).lower()
        if any(marker in encoded for marker in [
                '"reason": "content_filter"',
                '"reason": "refusal"',
                '"reason": "safety"']):
            return True
    return False


async def run_tool_loop_async(
        transcript_items: list, allowed=None, max_loops=MAX_LOOP_LIMIT,
        chat_fn=None, on_event=None, cancel_check=None,
        stream_chat=False, report_timing=False,
        on_response=None, hook_pipeline=None,
        cancel_event: asyncio.Event | None = None,
        ask_user=None,
        reasoning_effort=_UNSET,
        thinking: model_requests.TurnThinkingSettings | None = None) -> str:
    """Run the model/tool loop over canonical session events.

    Mutates ``transcript_items`` by appending one ``model_response`` for each
    actual provider response and one event for each local tool result. The
    callback receives ``codex_turn_state`` explicitly because it changes the
    routing semantics of every provider request in this logical user turn.
    """
    codex_turn_state = model_requests.CodexTurnState()
    # One snapshot for the whole logical turn: tool continuations use the
    # settings captured before the first request, not mid-turn changes.
    if thinking is None:
        thinking = capture_turn_settings()
    if reasoning_effort is _UNSET:
        reasoning_effort = thinking.effort
    elif reasoning_effort is not None:
        reasoning_effort = modelsdev.validate_reasoning_effort(
            reasoning_effort)
    completion_cancel_check = cancel_check
    if chat_fn is None:
        advertised_tools = (
            TOOLS if allowed is None else [
                spec["definition"]
                for name, spec in TOOL_REGISTRY.items()
                if name in allowed
            ])
        if ask_user is None:
            # Without an interactive seam there is nobody to answer, so
            # the model is never offered the question tool at all.
            advertised_tools = tool_contracts.without_ask_tool(advertised_tools)

        def chat_fn(items, on_text_delta=None, *, codex_turn_state, on_reasoning_delta=None):
            kwargs = {
                "tools": advertised_tools,
                "codex_turn_state": codex_turn_state,
                "reasoning_effort": reasoning_effort,
                "thinking": thinking,
            }
            if on_text_delta is not None:
                kwargs["on_text_delta"] = on_text_delta
            if on_reasoning_delta is not None:
                kwargs["on_reasoning_delta"] = on_reasoning_delta
            if completion_cancel_check is not None:
                kwargs["cancel_check"] = completion_cancel_check
            return async_chat_completion(items, **kwargs)
    if on_event is None:
        on_event = lambda event: None
    if cancel_check is None:
        cancel_check = lambda: False
    if on_response is None:
        on_response = lambda turn, event: None
    if max_loops < 1:
        raise ValueError("max_loops must be at least 1")

    tool_loop_extra_context = get_tool_loop_extra_context(transcript_items)
    tool_loop_extra_context["reasoning_effort"] = reasoning_effort
    tool_loop_extra_context["thinking"] = thinking
    if cancel_event is not None:
        tool_loop_extra_context["cancel_event"] = cancel_event
    if ask_user is not None:
        # The interactive-question seam. Only a frontend that can actually
        # put a question to the user supplies one, and the Ask tool is
        # advertised to the model only by such callers.
        tool_loop_extra_context["ask_user"] = ask_user

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
        if cancel_check():
            on_event({
                "type": "response_cancelled",
                "partial": False,
                "saved": False,
            })
            return ""
        loop_count += 1
        live_text_started = False
        live_text_parts = []
        live_reasoning = {}
        open_reasoning = None

        def close_reasoning(complete=True, reason=None):
            nonlocal open_reasoning
            if open_reasoning is not None:
                on_event({"type": "reasoning_end", "key": open_reasoning,
                          "complete": complete, "reason": reason})
                open_reasoning = None

        def on_reasoning_delta(key, kind, text):
            nonlocal open_reasoning
            if thinking.traces != "on" or not text:
                return
            if open_reasoning != key:
                close_reasoning()
                if live_text_started:
                    finish_live_text(True)
                open_reasoning = key
                on_event({"type": "reasoning_start", "key": key, "kind": kind})
            live_reasoning[key] = live_reasoning.get(key, "") + text
            on_event({"type": "reasoning_delta", "key": key, "kind": kind, "text": text})

        def on_text_delta(text):
            nonlocal live_text_started
            if not text:
                return
            close_reasoning()
            live_text_parts.append(text)
            if not live_text_started:
                live_text_started = True
                on_event({"type": "assistant_start"})
            on_event({"type": "assistant_delta", "content": text})

        def finish_live_text(complete, reason=None):
            nonlocal live_text_started
            close_reasoning(complete, reason)
            if live_text_started:
                on_event({
                    "type": "assistant_end",
                    "complete": complete,
                    "reason": reason,
                })
                live_text_started = False

        request_start = time.perf_counter()
        try:
            if stream_chat:
                stream_kwargs = {"codex_turn_state": codex_turn_state}
                if thinking.traces == "on":
                    stream_kwargs["on_reasoning_delta"] = on_reasoning_delta
                response_items = await chat_fn(transcript_items, on_text_delta, **stream_kwargs)
            else:
                response_items = await chat_fn(
                    transcript_items,
                    codex_turn_state=codex_turn_state)
        except model_requests.StreamCancelled:
            finish_live_text(False, "cancelled")
            on_event({
                "type": "response_cancelled",
                "partial": bool(live_text_parts or live_reasoning),
                "saved": False,
            })
            return ""
        except protocols.ResponseApiError as e:
            if cancel_check():
                finish_live_text(False, "cancelled")
                on_event({
                    "type": "response_cancelled",
                    "partial": bool(live_text_parts or live_reasoning),
                    "saved": False,
                })
                return ""
            finish_live_text(False, "error")
            on_event({"type": "api_error", "error": e})
            return ""
        except protocols.StreamProtocolError as e:
            if cancel_check():
                finish_live_text(False, "cancelled")
                on_event({
                    "type": "response_cancelled",
                    "partial": bool(live_text_parts or live_reasoning),
                    "saved": False,
                })
                return ""
            finish_live_text(False, "error")
            on_event({"type": "stream_error", "error": e})
            return ""
        except formats.TranscriptFormatError as e:
            if cancel_check():
                finish_live_text(False, "cancelled")
                on_event({
                    "type": "response_cancelled",
                    "partial": bool(live_text_parts or live_reasoning),
                    "saved": False,
                })
                return ""
            finish_live_text(False, "error")
            on_event({"type": "transcript_error", "error": e})
            return ""
        except protocols.ProtocolError as e:
            if cancel_check():
                finish_live_text(False, "cancelled")
                on_event({
                    "type": "response_cancelled",
                    "partial": bool(live_text_parts or live_reasoning),
                    "saved": False,
                })
                return ""
            finish_live_text(False, "error")
            on_event({"type": "provider_error", "error": e})
            return ""
        except model_requests.ApiError as e:
            if cancel_check():
                finish_live_text(False, "cancelled")
                on_event({
                    "type": "response_cancelled",
                    "partial": bool(live_text_parts or live_reasoning),
                    "saved": False,
                })
                return ""
            finish_live_text(False, "error")
            on_event({"type": "api_error", "error": e})
            return ""
        except OSError as e:
            if cancel_check():
                finish_live_text(False, "cancelled")
                on_event({
                    "type": "response_cancelled",
                    "partial": bool(live_text_parts or live_reasoning),
                    "saved": False,
                })
                return ""
            finish_live_text(False, "error")
            msg = str(e) or f"{type(e).__name__}() (errno={e.errno!r}, no OS message)"
            on_event({"type": "network_error", "error": msg})
            return ""
        except BaseException:
            finish_live_text(False, "cancelled" if cancel_check() else "error")
            raise
        if cancel_check():
            finish_live_text(False, "cancelled")
            on_event({
                "type": "response_cancelled",
                "partial": bool(live_text_parts or live_reasoning),
                "saved": False,
            })
            return ""
        turn = formats.coerce_decoded_turn(response_items)
        buffered_reasoning = thinking.traces == "on" and not (live_reasoning or live_text_parts)
        if thinking.traces == "on" and not buffered_reasoning:
            source = next((key[1] for key in live_reasoning if key[0] == "chat"), formats.reasoning_field(turn))
            for key, kind, text in formats.reasoning_segments(turn.items, source=source):
                delivered = live_reasoning.get(key, "")
                if text.startswith(delivered) and len(text) > len(delivered):
                    on_reasoning_delta(key, kind, text[len(delivered):])
        finish_live_text(turn.complete, None if turn.complete else "incomplete")
        if report_timing:
            on_event({
                "type": "response_timing",
                "elapsed": time.perf_counter() - request_start,
            })
        append_turn(turn)
        for notice_code in formats.provider_notice_codes(turn):
            on_event({
                "type": "provider_notice",
                "code": notice_code,
            })

        buffered_answer_displayed = False
        if buffered_reasoning:
            for index, item in enumerate(turn.items):
                for key, kind, text in formats.reasoning_segments([item], source=formats.reasoning_field(turn)):
                    on_reasoning_delta(("buffered", index, key), kind, text)
                if item.get("type") == "message" and item.get("role") == "assistant":
                    text = formats.item_text(item)
                    if text:
                        close_reasoning(turn.complete)
                        on_event({"type": "assistant_message", "content": text})
                        buffered_answer_displayed = True
            close_reasoning(turn.complete)

        tool_calls = formats.response_tool_calls(turn.items)
        assistant_items = [
            item for item in turn.items
            if item.get("type") == "message" and item.get("role") == "assistant"
        ]
        assistant_text = "\n".join(
            text for text in (
                formats.item_text(item) for item in assistant_items)
            if text)
        if assistant_text and not live_text_parts and not buffered_answer_displayed:
            on_event({"type": "assistant_message", "content": assistant_text})

        if _turn_is_refusal(turn):
            if tool_calls:
                append_unexecuted_results(
                    tool_calls,
                    "Tool call not executed because the provider refused "
                    "the turn.",
                )
            on_event({"type": "response_refusal"})
            return assistant_text

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

        if turn.metadata.get("end_turn") is False and not tool_calls:
            if loop_count >= max_loops:
                on_event({"type": "max_loops"})
                return assistant_text
            # Responses-Lite can explicitly say that a completed response is
            # not the end of the logical turn.  Preserve the real response
            # boundary, then let the provider continue from that history
            # without inventing a user message or tool result.
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
                "partial": bool(live_text_parts or live_reasoning),
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
                on_event({
                    "type": "tool_error",
                    "call_id": formats.tool_call_id(tc),
                    "result": result["content"],
                })
            result_item = formats.tool_result_for_call(
                tc,
                result["content"],
                is_error=not result["ok"],
                execution=execution,
                process_output=result.get("process_output"),
            )
            transcript_items.append(result_item)
            event = {
                "type": "tool_result",
                "name": formats.tool_call_name(tc),
                "call_id": formats.tool_call_id(tc),
                "content": result["content"],
                "is_error": not result["ok"],
            }
            if "process_output" in result:
                event["process_output"] = copy.deepcopy(result["process_output"])
            on_event(event)
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


async def run_toolless_completion_async(
        transcript_items: list, cancel_check=None,
        reasoning_effort=_UNSET,
        thinking: model_requests.TurnThinkingSettings | None = None) -> str:
    # Tool helpers such as WebFetch may run inside an outer user turn, but
    # this completion is a distinct logical turn and therefore owns a
    # distinct routing token passed explicitly to the request path.
    codex_turn_state = model_requests.CodexTurnState()
    if thinking is None:
        thinking = capture_turn_settings()
    thinking = replace(thinking, traces="off")
    if reasoning_effort is _UNSET:
        reasoning_effort = thinking.effort
    try:
        kwargs = {}
        if cancel_check is not None:
            kwargs["cancel_check"] = cancel_check
        response_items = await async_chat_completion(
            transcript_items,
            tools=[],
            codex_turn_state=codex_turn_state,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
            **kwargs,
        )
    except (formats.TranscriptFormatError, protocols.ProtocolError) as e:
        return f"Transcript render error: {e}"
    except model_requests.ApiError as e:
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
    if stdout_text:
        parts.extend(["[stdout]", stdout_text])
    if stderr_text:
        parts.extend(["[stderr]", stderr_text])
    return "\n".join(parts)


def _subagent_env(reasoning_effort=_UNSET, *, environ=None,
                  thinking: model_requests.TurnThinkingSettings | None = None) -> dict:
    source_environ = os.environ if environ is None else environ
    # LOKI_DUMMY_TOOL_CALL makes the DUMMY provider emit a tool call instead of
    # text; a subagent that inherited it would emit the same call and recurse,
    # so it is the parent's test directive, never the subagent's.
    env = {
        name: value for name, value in source_environ.items()
        if not is_credential_name(name) and name != "LOKI_DUMMY_TOOL_CALL"
    }
    env.pop("LOKI_TURN_THINKING", None)
    env.pop("LOKI_REASONING_EFFORT", None)
    env.pop("LOKI_REASONING_TRACES", None)
    # Subagents receive only non-secret provider configuration. The request
    # credential travels through a fresh capability socket, never through
    # argv or the process environment.
    if current_config():
        if thinking is None:
            thinking = capture_turn_settings()
        if reasoning_effort is _UNSET:
            reasoning_effort = thinking.effort
        config = current_config()
        provider = config.chat_provider
        auth_spec = config.auth_spec
        env['LOKI_PROVIDER'] = provider.kind
        env['LOKI_API_BASE'] = provider.input_url
        env['LOKI_MODEL'] = current_model()
        env.pop('LOKI_API_KEY', None)
        env['LOKI_STREAM'] = '1' if config.stream else '0'
        env['LOKI_PROMPT_CACHE'] = (
            '1' if provider.prompt_cache else '0')
        capabilities = config.reasoning_capabilities
        env["LOKI_TURN_THINKING"] = json.dumps({
            "connection": _thinking_connection(),
            "settings": {"effort": reasoning_effort, "mode": thinking.mode,
                         "budget": thinking.budget, "retention": thinking.retention},
            "capabilities": capabilities.to_dict() if capabilities is not None else None,
        }, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        profile = config.reasoning_effort_profile
        if provider.provider_id is not None:
            env['LOKI_PROVIDER_ID'] = provider.provider_id
        else:
            env.pop('LOKI_PROVIDER_ID', None)
        if profile is not None:
            env['LOKI_REASONING_EFFORT_PROFILE'] = json.dumps(
                profile.to_dict(),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        else:
            env.pop('LOKI_REASONING_EFFORT_PROFILE', None)
        if reasoning_effort is not None:
            reasoning_effort = modelsdev.validate_reasoning_effort(
                reasoning_effort)
            if profile is None or not profile.supports(reasoning_effort):
                raise ValueError(
                    "cannot delegate an unsupported reasoning effort")
        if provider.openai_request_profile is not None:
            # This compact, non-secret request profile is process
            # configuration. Credentials still cross only the anonymous
            # capability socket. The full authenticated catalog record,
            # including application prompts, never enters the environment.
            env['LOKI_OPENAI_REQUEST_PROFILE'] = json.dumps(
                provider.openai_request_profile.to_dict(),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        else:
            env.pop('LOKI_OPENAI_REQUEST_PROFILE', None)
        env.pop('LOKI_RESPONSES_LITE', None)
        env['LOKI_MAX_TOKENS'] = str(provider.max_tokens)
        if config.context_capacity is not None:
            env['LOKI_CONTEXT_WINDOW'] = str(config.context_capacity.tokens)
        else:
            env.pop('LOKI_CONTEXT_WINDOW', None)
        if provider.kind == protocols.ANTHROPIC_MESSAGES:
            env['LOKI_ANTHROPIC_VERSION'] = provider.headers[
                "anthropic-version"]
        else:
            env.pop('LOKI_ANTHROPIC_VERSION', None)
        if (auth_spec is not None
                and auth_spec.scheme == "custom"):
            env['LOKI_AUTH_HEADER'] = auth_spec.header_name
        else:
            env.pop('LOKI_AUTH_HEADER', None)
        if auth_spec is not None:
            # This is an identifier and policy, not a token. The child uses
            # it to request precisely this credential from its delegated
            # capability instead of inferring identity from absent env data.
            env['LOKI_CREDENTIAL_REF'] = (
                auth_spec.credential.encode())
            env['LOKI_AUTH_SCHEME'] = auth_spec.scheme
        else:
            env.pop('LOKI_CREDENTIAL_REF', None)
            env.pop('LOKI_AUTH_SCHEME', None)
        if provider.models_url:
            env['LOKI_MODELS_URL'] = provider.models_url
        else:
            env.pop('LOKI_MODELS_URL', None)
    return env


def delegated_turn_settings(credentials) -> model_requests.TurnThinkingSettings:
    """Import non-secret delegation data without importing request authority."""
    encoded = credentials.get("LOKI_TURN_THINKING")
    if not encoded:
        return replace(capture_turn_settings(), traces="off")
    envelope = json.loads(encoded)
    if not isinstance(envelope, dict) or set(envelope) != {"connection", "settings", "capabilities"}:
        raise ValueError("invalid delegated thinking envelope")
    if envelope["connection"] != _thinking_connection():
        raise ValueError("delegated thinking settings do not match the connection")
    values = envelope["settings"]
    if not isinstance(values, dict) or set(values) != {"effort", "mode", "budget", "retention"}:
        raise ValueError("invalid delegated thinking settings")
    effort = values["effort"]
    profile = current_reasoning_effort_profile()
    if effort is not None:
        modelsdev.validate_reasoning_effort(effort)
        if profile is None or not profile.supports(effort):
            raise ValueError("delegated effort is not in the model's profile")
    if values["mode"] is not None and not isinstance(values["mode"], str):
        raise ValueError("delegated mode must be a string or null")
    config = current_config()
    provider = config.chat_provider
    provider.validate_thinking(config.model, effort, values["mode"], values["budget"], values["retention"])
    raw_capabilities = envelope["capabilities"]
    capabilities = (modelsdev.ReasoningCapabilities.from_dict(raw_capabilities)
                    if raw_capabilities is not None else None)
    # The endpoint, authentication recipe, and executable remain owned by the
    # established launch/capability protocol; only model hints are imported.
    apply_runtime_config(replace(
        config, reasoning_capabilities=capabilities,
        chat_provider=replace(provider, reasoning_field=capabilities.interleaved_field if capabilities else None)))
    return model_requests.TurnThinkingSettings(
        effort=effort, mode=values["mode"], budget=values["budget"], retention=values["retention"], traces="off")


def _subagent_credential_refs():
    config = current_config()
    if config is None or config.auth_spec is None:
        return frozenset()
    return frozenset({config.auth_spec.credential})


def _format_started_background_job(job: jobs.Job, kind: str = "job") -> str:
    return "\n".join([
        f"Started background {kind} {job.id}",
        f"cwd: {job.cwd}",
        f"pid: {job.pid}",
        f"pgid: {job.pgid}",
        f"status: {job.status}",
        f"stdout: {job.stdout_path}",
        f"stderr: {job.stderr_path}",
    ])


def _subagent_argv(agent_type: str, prompt: str) -> list[str]:
    depth = current_session().subagent_depth
    if depth >= tool_contracts.MAX_SUBAGENT_DEPTH:
        raise ValueError(
            f"maximum subagent depth reached ({tool_contracts.MAX_SUBAGENT_DEPTH})")
    return [
        sys.argv[0],
        '--subagent',
        agent_type,
        '--subagent-depth',
        str(depth + 1),
        '--root-conversation-id',
        current_session().root_conversation_id,
        '--prompt',
        prompt,
        '--shell-cwd',
        current_cwd(),
    ]


def run_agent(description: str, prompt: str, run_in_background: bool = False,
              subagent_type: str = "Explore",
              reasoning_effort=_UNSET) -> str:
    return asyncio.run(run_agent_async(
        description,
        prompt,
        run_in_background,
        subagent_type,
        reasoning_effort=reasoning_effort,
    ))


async def run_agent_async(description: str, prompt: str, run_in_background: bool = False,
                          subagent_type: str = "Explore",
                          cancel_event: asyncio.Event | None = None,
                          reasoning_effort=_UNSET,
                          thinking: model_requests.TurnThinkingSettings | None = None) -> str:
    agent_type = subagent_type or "Explore"
    if not prompt:
        return "Error: prompt is required"
    if agent_type != "Explore":
        return f"Error: unknown subagent_type {agent_type!r} (only 'Explore' is supported)"
    try:
        argv = _subagent_argv(agent_type, prompt)
    except ValueError as error:
        return f"Error: {error}"
    if thinking is None:
        thinking = capture_turn_settings()
    manager = current_job_manager()
    environment = _subagent_env(reasoning_effort, thinking=thinking)
    references = _subagent_credential_refs()
    authority = current_session().credential_authority
    destinations = None
    if references:
        # The child uses this captured environment, not whatever
        # connection the parent may select after an await. Explore
        # only needs inference, never account/discovery authority.
        chat_url, _models_url = protocols.endpoint_urls(
            environment["LOKI_API_BASE"], environment["LOKI_PROVIDER"])
        destinations = [chat_url]
    if run_in_background:
        try:
            job = await manager.run_background_exec(
                argv,
                description=description or "subagent task",
                env=environment,
                cwd=os.getcwd(),
                session_owned=True,
                credential_refs=references, credential_authority=authority,
                credential_destinations=destinations,
                subagent=True)
        except jobs.SubagentCapacityError as error:
            return f"Error: {error}"
        return _format_started_background_job(job, "subagent")

    try:
        job, status, stdout, stderr = await manager.run_exec(
            argv,
            description=description or "subagent task",
            env=environment, cwd=os.getcwd(),
            cancel_event=cancel_event,
            session_owned=True,
            credential_refs=references, credential_authority=authority,
            credential_destinations=destinations,
            subagent=True)
    except jobs.SubagentCapacityError as error:
        return f"Error: {error}"
    if status == "cancelled":
        return "Error: subagent was cancelled by the user"
    result = _format_subagent_result(agent_type, description, "completed",
                                     job.exit_code, stdout, stderr)
    if job.exit_code != 0:
        return f"Error: subagent exited with code {job.exit_code}\n{result}"
    return result


def run_skill(skill: str, args: str | None = None) -> str:
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
    footer = (
        ("\n\n[Skill content truncated]" if truncated else "")
        + f"\n\nBase directory for this skill: {base_dir}\n"
        + "Relative paths in this skill are relative to this base directory.\n</skill_content>"
    )
    return header + body + footer


async def run_webfetch_async(
        url: str, prompt: str,
        cancel_event: asyncio.Event | None = None,
        reasoning_effort=_UNSET,
        thinking: model_requests.TurnThinkingSettings | None = None) -> str:
    # The web backend owns retrieval/presentation; this adapter supplies only
    # the distinct tool-less inference turn and the existing process cache.
    async def complete(messages, cancel_check=None):
        return await run_toolless_completion_async(
            messages, cancel_check=cancel_check,
            reasoning_effort=reasoning_effort, thinking=thinking)

    return await web_tools.run_webfetch_async(
        url, prompt, cache=_webfetch_cache, complete=complete,
        cancel_event=cancel_event)


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
    "Bash": {"handler": _handle_bash, "async_handler": _handle_bash_async},
    "Jobs": {"handler": _handle_jobs, "explore": True},
    "JobStatus": {"handler": _handle_job_status, "explore": True},
    "JobStop": {"handler": _handle_job_stop},
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
    "TodoWrite": {"handler": _handle_todowrite, "plan": True},
    "Agent": {
        "handler": _handle_agent,
        "async_handler": _handle_agent_async,
        "explore": True,
    },
    "Skill": {"handler": _handle_skill},
    "WebFetch": {"async_handler": _handle_webfetch_async, "explore": True},
    "WebSearch": {"async_handler": _handle_websearch_async, "explore": True},
    # Clarification does not modify workspace state, so it is safe in Explore
    # mode and question-guarded turns. Interactive callers still need a callback.
    tool_contracts.ASK_TOOL_NAME: {"async_handler": _handle_ask_async, "explore": True},
}
TOOL_REGISTRY = _build_tool_registry(tool_contracts.TOOLS, TOOL_HANDLERS)
TOOLS = [spec["definition"] for spec in TOOL_REGISTRY.values()]
EXPLORE_TOOLS = {
    name for name, spec in TOOL_REGISTRY.items() if spec["explore"]}
PLAN_TOOLS = EXPLORE_TOOLS | {
    name for name, spec in TOOL_REGISTRY.items() if spec.get("plan")}
TOOL_HOOK_PIPELINE = tool_runtime.ToolHookPipeline()


async def run_turn_end_hooks_async(events, text="", *, error=None):
    """Notify user-facing turn hooks after the model/tool loop has stopped."""
    pipeline = TOOL_HOOK_PIPELINE
    if not pipeline.turn_end_hooks:
        return
    error_kinds = {
        "api_error", "network_error", "stream_error", "transcript_error",
        "provider_error", "response_failed", "response_incomplete",
    }
    failure = next((event for event in events
                    if event.get("type") in error_kinds), None)
    kinds = {event.get("type") for event in events}
    if error is not None or failure is not None:
        reason = "error"
    elif "response_cancelled" in kinds:
        reason = "cancelled"
    elif "max_loops" in kinds:
        reason = "max_loops"
    else:
        reason = "completed"
    payload = {
        "event": "turn_end",
        "reason": reason,
        "cwd": current_cwd(),
        "text": text,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    elif failure is not None:
        payload["error_type"] = failure["type"]
        detail = failure.get("error")
        if detail is None:
            protocol_data = failure.get("protocol_data")
            native = (protocol_data.get(formats.OPENAI_RESPONSES)
                      if isinstance(protocol_data, dict) else None)
            if isinstance(native, dict):
                details = native.get("error") or native.get("incomplete_details")
                if isinstance(details, dict):
                    detail = details.get("message") or details.get("reason")
        payload["error_message"] = str(detail) if detail is not None else None
    await pipeline.finish_turn(payload)


def configure_tool_hook_pipeline(environ=os.environ, stderr_reporter=None):
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
    TOOL_HOOK_PIPELINE = tool_runtime.load_hook_pipeline(
        path, stderr_reporter=stderr_reporter)
    return path


def _model_request_context():
    session = current_session()
    return model_requests.RequestContext(
        config=session.runtime_config,
        credential_authority=session.credential_authority,
        conversation_id=session.conversation_id,
        root_conversation_id=session.root_conversation_id,
        response_headers=session.response_headers)


async def async_provider_request(
        method: str, request_url: str, payload=None,
        request_headers: dict | None = None, report_errors: bool = False,
        show_timing: bool = False, cancel_check=None,
        codex_turn_state=None, opencode_session_id=None) -> protocols.ProviderResponse:
    return await model_requests.async_provider_request(
        method, request_url, payload, request_headers=request_headers,
        report_errors=report_errors, show_timing=show_timing, cancel_check=cancel_check,
        codex_turn_state=codex_turn_state, opencode_session_id=opencode_session_id,
        context=_model_request_context())


async def async_chat_stream_request(
        request_url: str, payload, request_headers: dict | None = None,
        on_text_delta=None, cancel_check=None,
        report_errors: bool = False, show_timing: bool = False,
        codex_turn_state=None, opencode_session_id=None,
        on_reasoning_delta=None) -> protocols.ProviderResponse:
    return await model_requests.async_chat_stream_request(
        request_url, payload, request_headers=request_headers,
        on_text_delta=on_text_delta, cancel_check=cancel_check,
        report_errors=report_errors, show_timing=show_timing,
        codex_turn_state=codex_turn_state, opencode_session_id=opencode_session_id,
        on_reasoning_delta=on_reasoning_delta, context=_model_request_context())


async def async_chat_completion(
        transcript_items: list, tools=TOOLS, report_errors: bool = False,
        show_timing: bool = False, on_text_delta=None, on_reasoning_delta=None,
        cancel_check=None, codex_turn_state=None, reasoning_effort=_UNSET,
        thinking: model_requests.TurnThinkingSettings | None = None) -> formats.DecodedTurn:
    if cancel_check and cancel_check():
        raise model_requests.StreamCancelled()
    context = _model_request_context()
    if context.config is None:
        return formats.DecodedTurn([])
    # Innermost default: a caller that owns a logical turn passes its
    # captured snapshot; a direct call captures one here.
    if thinking is None:
        thinking = capture_turn_settings()
    if reasoning_effort is _UNSET:
        reasoning_effort = thinking.effort
    elif reasoning_effort is not None:
        reasoning_effort = modelsdev.validate_reasoning_effort(reasoning_effort)
    return await model_requests.async_chat_completion(
        transcript_items, tools, report_errors=report_errors, show_timing=show_timing,
        on_text_delta=on_text_delta, on_reasoning_delta=on_reasoning_delta,
        cancel_check=cancel_check, codex_turn_state=codex_turn_state,
        reasoning_effort=reasoning_effort, thinking=thinking, context=context)


def _status_api_base() -> str:
    configured_url = ""
    if current_config():
        configured_url = current_config().chat_provider.input_url

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


async def load_models_async(diagnostic_writer=None):
    """Fetch the active provider's /models listing; [] when unavailable."""
    if not current_config():
        return [current_model()] if current_model() else []

    model_urls = getattr(
        current_config().chat_provider, "model_urls", None) or (
            [current_config().chat_provider.models_url]
            if current_config().chat_provider.models_url else [])
    if not model_urls:
        return [current_model()] if current_model() else []
    errors = []
    for models_url in model_urls:
        try:
            response = await async_provider_request(
                "GET",
                models_url,
                request_headers=current_config().chat_provider.headers,
                report_errors=True,
            )
        except model_requests.ApiError as e:
            errors.append(e.formatted())
            continue
        except OSError as e:
            errors.append(f"API Error for <{models_url}>: {e}")
            continue
        except (json.JSONDecodeError, protocols.ProtocolError) as e:
            # Some providers (e.g. Z.AI's /paas/v4 base) don't expose a /models
            # endpoint at all -- the response is an HTML error page or empty.
            # Treat as "couldn't load" rather than crashing the picker.
            message = e.msg if isinstance(e, json.JSONDecodeError) else str(e)
            errors.append(
                f"Model list at <{models_url}> was not JSON: {message}")
            continue
        loaded = current_config().chat_provider.parse_model_ids(
            response.payload)
        if loaded:
            return loaded

    if errors and diagnostic_writer is not None:
        diagnostic_writer("\n".join(errors))
    return [current_model()] if current_model() else []

# models = ['hy3-preview', 'glm-5.2', 'glm-5.1', 'kimi-k2.7', 'kimi-k2.6', 'deepseek-v4-pro', 'deepseek-v4-flash', 'mimo-v2.5', 'mimo-v2.5-pro']


# Agent mode, cycled by Shift-Tab: "explore" (read-only), "plan", "edit".
# Takes effect for the next turn; it does not cancel the current turn.
MODE_CYCLE_ORDER = ["normal", "explore", "plan", "edit"]


def cycle_agent_mode() -> str:
    session = current_session()
    i = MODE_CYCLE_ORDER.index(session.agent_mode)
    session.agent_mode = MODE_CYCLE_ORDER[(i + 1) % len(MODE_CYCLE_ORDER)]
    return session.agent_mode


def initial_transcript_items():
    instructions = prompts.load_system_prompt()
    items = [formats.instruction_item(instructions)] if instructions else []
    items.append(formats.instruction_item(
        f"Current date: {time.strftime('%Y-%m-%d')}. "
        f"Current Loki cwd: {current_cwd()}. Relative tool paths and Bash commands run from this directory. "
    ))
    return items


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


def _print_display_line(prefix, text, *, text_writer=None, file=None):
    print(prefix, end="", file=file)
    if text_writer is None:
        print(repr(text), end="", file=file)
    else:
        text_writer(text, file=file)
    print(file=file)


def print_shell_cwd(text_writer=None):
    sys.stdout.flush()
    _print_display_line(
        "cwd: ", current_cwd(),
        text_writer=text_writer, file=sys.stderr)
    sys.stderr.flush()


def change_shell_cwd_from_text(arg_text: str, text_writer=None) -> bool:
    try:
        target = _parse_cd_arg_text(arg_text)
        change_shell_cwd(target)
    except (FileNotFoundError, NotADirectoryError) as e:
        sys.stdout.flush()
        _print_display_line(
            "cd: no such directory: ", str(e),
            text_writer=text_writer, file=sys.stderr)
        sys.stderr.flush()
        return False
    except ValueError as e:
        sys.stdout.flush()
        _print_display_line(
            "cd: ", str(e),
            text_writer=text_writer, file=sys.stderr)
        sys.stderr.flush()
        return False
    record_shell_cwd_instruction()
    print_shell_cwd(text_writer=text_writer)
    return True


def _parse_cd_arg_text(arg_text: str) -> str:
    """The directory operand: the text as given, with one quote layer removed.

    The callers already pass everything after the command name, so this is one
    literal path, not a shell word list.  It used to be ``shlex.split``, which
    tokenises in POSIX mode, where a backslash escapes the next character: it
    ate every separator of a Windows path, so ``/cd D:\\a\\b`` stat'd
    ``D:ab``.  Nothing here may interpret escaping or split on whitespace; the
    only affordance kept is one layer of matching surrounding quotes, for a
    path typed with spaces out of habit.
    """
    target = arg_text.strip()
    if (len(target) >= 2 and target[0] == target[-1]
            and target[0] in ["'", '"']):
        target = target[1:-1]
    return target


def chat_log_filename(chat_id: str) -> str:
    return savefiles.chat_log_filename(chat_id)


def ensure_chat_log_dir():
    savefiles.ensure_chat_log_dir(CHAT_LOG_DIR)


def new_chat_log_path() -> str:
    return savefiles.new_chat_log_path(CHAT_LOG_DIR)


def resolve_chat_log_path(resume_arg: str) -> str:
    return savefiles.resolve_chat_log_path(
        resume_arg, STARTUP_CWD, CHAT_LOG_DIR, _resolve_path)


def new_chat_log(filename):
    # ``filename`` is composed from the runtime's own chat-log root and is used
    # as given: Loki never normalises or resolves a filesystem operand, on any
    # platform, for any reason.
    session = current_session()
    session.conversation_id = conversation_id_for_path(filename)
    session.transcript_items = initial_transcript_items()
    session.session_todos = []
    session.session_toolsets = []
    session.last_instructed_agent_mode = None
    session.reasoning_effort_preference = None
    session.thinking_mode = None
    session.thinking_budget = None
    session.reasoning_retention = "default"
    session.reasoning_traces = "off"
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
    if session.last_instructed_agent_mode is None:
        state.pop("last_instructed_agent_mode", None)
    else:
        state["last_instructed_agent_mode"] = session.last_instructed_agent_mode
    if session.reasoning_effort_preference is None:
        state.pop("reasoning_effort", None)
    else:
        state["reasoning_effort"] = (
            session.reasoning_effort_preference)
    state["thinking_traces"] = session.reasoning_traces
    state["reasoning_retention"] = session.reasoning_retention
    for name in ["thinking_mode", "thinking_budget"]:
        value = getattr(session, name)
        if value is None:
            state.pop(name, None)
        else:
            state[name] = value
    content = savefiles.serialize_chat_log(
        session.transcript_items,
        session.session_todos,
        state,
        toolsets=session.session_toolsets,
    )
    files.atomic_write_text(session.chat_log_path, content)
    savefiles.report_chat_log_saved(session.chat_log_path)
    session.session_state = state
    session.chat_log_dirty = False
    return True


def mark_chat_log_dirty():
    session = current_session()
    if session.chat_log_path is not None:
        session.chat_log_dirty = True


def load_chat_log(filename, loaded=None, *, apply_shell_cwd=True):
    """Load a saved conversation's state; rendering is the caller's job."""
    if loaded is None:
        with open(filename, 'r', encoding="utf-8") as f:
            loaded = savefiles.read_chat_log(f)
    transcript, todos, state, toolsets = loaded
    current_session().replace_transcript(
        transcript, todos, toolsets, state, filename)
    load_session_state(state, apply_shell_cwd=apply_shell_cwd)


def load_session_state(state: dict, *, apply_shell_cwd=True):
    if not isinstance(state, dict):
        return
    session = current_session()
    # This describes an instruction already recorded in the transcript, not
    # the operating mode to activate or a server acknowledgement.
    session.last_instructed_agent_mode = None
    if "last_instructed_agent_mode" in state:
        mode = state["last_instructed_agent_mode"]
        if isinstance(mode, str) and mode in MODE_CYCLE_ORDER:
            session.last_instructed_agent_mode = mode
        else:
            print(
                "Warning: invalid saved last_instructed_agent_mode; "
                "the next user turn will announce the current mode.",
                file=sys.stderr,
            )
    session.reasoning_effort_preference = (
        reasoning_effort_from_session_state(state))
    # Saved chat logs are untrusted input: each thinking control is
    # validated before it can influence a request.
    traces = state.get("thinking_traces", "off")
    if traces not in ["off", "on"]:
        raise ValueError("invalid saved thinking traces")
    retention = state.get("reasoning_retention", "default")
    if retention not in ["default", "preserve"]:
        raise ValueError("invalid saved thinking retention")
    mode = state.get("thinking_mode")
    if mode is not None and mode not in [
            "on", "off", "manual", "adaptive", "between-tools"]:
        raise ValueError("invalid saved thinking mode")
    budget = state.get("thinking_budget")
    if budget is not None and (
            not isinstance(budget, int)
            or isinstance(budget, bool) or budget < 0):
        raise ValueError("invalid saved thinking budget")
    session.reasoning_traces = traces
    session.reasoning_retention = retention
    session.thinking_mode = mode
    session.thinking_budget = budget
    if not apply_shell_cwd:
        # An explicit launch working directory outranks state restored from a
        # file; the caller decides that precedence, and this function must not
        # silently override it.
        return
    loaded_shell_cwd = state.get("shell_cwd")
    if not isinstance(loaded_shell_cwd, str) or not loaded_shell_cwd:
        return
    try:
        change_shell_cwd(loaded_shell_cwd)
    except FileNotFoundError:
        print(
            f"Warning: saved cwd no longer exists: {loaded_shell_cwd!r}",
            file=sys.stderr,
        )


def reasoning_effort_from_session_state(state: dict) -> str | None:
    if not isinstance(state, dict) or "reasoning_effort" not in state:
        return None
    try:
        return modelsdev.validate_reasoning_effort(
            state["reasoning_effort"], "saved reasoning effort")
    except ValueError as error:
        raise ValueError(f"invalid saved reasoning effort: {error}") from error


def connection_from_session_state(state: dict) -> ConnectionDescriptor | None:
    if not isinstance(state, dict) or "connection" not in state:
        return None
    return ConnectionDescriptor.from_dict(state["connection"])


def set_session_connection(descriptor: ConnectionDescriptor):
    current_state()["connection"] = descriptor.to_dict()
    mark_chat_log_dirty()
