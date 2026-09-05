"""Single-session ACP worker behind the front process."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from dataclasses import dataclass

from . import (
    acps,
    acp_events,
    formats,
    loki,
    models as modelsdev,
    reasonings,
    replays,
)
from .connections import ConnectionDescriptor, ConnectionDescriptorError
from .sessions import Session


_DISCONNECTED = object()
_UNCHANGED = object()
_PROMPT_EFFORT_UNSET = object()
_MODEL_DEFAULT_CONFIG_VALUE = "default"
_EFFORT_CONFIG_VALUE_PREFIX = "effort:"


@dataclass(frozen=True)
class PendingSessionOpen:
    """Prepared session state which is inert until the front commits it.

    Saved logs are untrusted input. The worker may parse and reconcile one
    before approval, but network-capable configuration, log rewrites, and
    transcript replay become observable only after the front reports that
    the client accepted the exact effective connection.
    """

    runtime_config: loki.RuntimeConfig | None
    connection_update: object
    open_method: str


class TurnFailure(Exception):
    """A provider/tool-loop failure that must be an ACP request error."""


class Worker:
    def __init__(self, session: Session, write, session_id: str = "worker"):
        self.session = session
        self.write = write
        self.session_id = session_id
        self.cancel_event = asyncio.Event()
        self._prompt_task: asyncio.Task | None = None
        self._model_options: list[dict] = []
        self._option_leaves: dict[str, object] = {}
        self._current_option_value: str | None = None
        self._configuration_error: str | None = None
        self._pending_open: PendingSessionOpen | None = None
        self._active_prompt_reasoning_effort: str | None = None

    async def handle(self, message: dict, concurrent: bool = False):
        method = message.get("method")
        request_id = message.get("id")
        if method is None or request_id is None:
            return
        if concurrent and method == "session/prompt":
            if (self._prompt_task is not None
                    and not self._prompt_task.done()):
                self.write(acps.response(
                    request_id,
                    error={
                        "code": acps.INVALID_PARAMS,
                        "message": (
                            "a prompt is already running for this session"),
                    },
                ))
                return
            # Snapshot before scheduling so a following configuration request
            # cannot race the logical turn's first provider request.
            reasoning_effort = loki.effective_reasoning_effort()
            self._prompt_task = asyncio.create_task(
                self._answer(
                    message,
                    prompt_reasoning_effort=reasoning_effort,
                ),
                name=f"acp-worker-prompt-{request_id}",
            )
            return
        if (method == "session/set_config_option"
                and self._prompt_task is not None
                and not self._prompt_task.done()
                and (message.get("params") or {}).get("configId")
                != "reasoning_effort"):
            self.write(acps.response(
                request_id,
                error={
                    "code": acps.INVALID_PARAMS,
                    "message": (
                        "cannot change session configuration while a "
                        "prompt is running"),
                },
            ))
            return
        await self._answer(message)

    async def close(self):
        try:
            self.cancel_event.set()
            task = self._prompt_task
            if task is not None and not task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task), timeout=3)
                except asyncio.TimeoutError:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        finally:
            manager = self.session.job_manager
            if manager is not None:
                await manager.close_session_owned()

    async def _answer(
            self, message: dict,
            prompt_reasoning_effort=_PROMPT_EFFORT_UNSET):
        request_id = message.get("id")
        try:
            result = await self.dispatch(
                message.get("method"),
                message.get("params") or {},
                prompt_reasoning_effort=prompt_reasoning_effort,
            )
        except acps.TransportError as error:
            self.write(acps.response(
                request_id,
                error={"code": error.code, "message": str(error)},
            ))
            return
        except Exception as error:
            self.write(acps.response(
                request_id,
                error={"code": acps.INTERNAL_ERROR, "message": str(error)},
            ))
            return
        self.write(acps.response(request_id, result=result))

    async def dispatch(
            self, method: str, params: dict,
            prompt_reasoning_effort=_PROMPT_EFFORT_UNSET):
        if method == "session/prompt":
            return await self.prompt(
                params,
                reasoning_effort=prompt_reasoning_effort,
            )
        if method == "session/cancel":
            self.cancel_event.set()
            return {}
        if method == "session/prepare_open":
            return await self.prepare_open(params)
        if method == "session/commit_open":
            return self.commit_open()
        if method == "session/set_config_option":
            return self.set_config_option(params)
        raise acps.TransportError(
            f"worker does not implement {method}",
            code=acps.METHOD_NOT_FOUND,
        )

    # -- session lifecycle and model configuration -----------------------

    def config_options(self) -> list:
        if not self._model_options:
            return []
        values = {option["value"] for option in self._model_options}
        if self._current_option_value not in values:
            raise RuntimeError(
                "ACP model config has no valid current value")
        options = [{
            "id": "model",
            "name": "Model",
            "category": "model",
            "type": "select",
            "currentValue": self._current_option_value,
            "options": copy.deepcopy(self._model_options),
        }]
        reasoning_option = self._reasoning_config_option()
        if reasoning_option is not None:
            options.append(reasoning_option)
        return options

    def _reasoning_config_option(self):
        profile = loki.current_reasoning_effort_profile()
        if profile is None:
            return None
        preference = loki.current_reasoning_effort_preference()
        current_value = (
            _EFFORT_CONFIG_VALUE_PREFIX + preference
            if profile.supports(preference)
            else _MODEL_DEFAULT_CONFIG_VALUE
        )
        values = [{
            "value": _MODEL_DEFAULT_CONFIG_VALUE,
            "name": reasonings.default_option_name(profile, preference),
            "description": (
                "Use the selected model or provider's own reasoning "
                "effort default."),
        }]
        for option in profile.options:
            encoded = {
                "value": _EFFORT_CONFIG_VALUE_PREFIX + option.value,
                "name": reasonings.display_name(option.value),
            }
            if option.description is not None:
                encoded["description"] = option.description
            values.append(encoded)
        return {
            "id": "reasoning_effort",
            "name": "Reasoning effort",
            "description": (
                "Controls the selected model's reasoning depth, latency, "
                "and token use."),
            "category": "thought_level",
            "type": "select",
            "currentValue": current_value,
            "options": values,
        }

    @staticmethod
    def _explicit_choice():
        explicit = loki.explicit_connection_option(loki.CREDENTIALS)
        if explicit is None:
            return None
        option = {
            "value": "loki-explicit",
            "name": f"{explicit.model} [LOKI_* connection]",
            "description": (
                f"explicit LOKI_* env connection; {explicit.protocol}; "
                f"api={explicit.api_url}"),
        }
        return option, explicit

    @staticmethod
    def _saved_choice(descriptor: ConnectionDescriptor):
        provider = descriptor.provider_name or descriptor.provider_id
        label = (
            f"{descriptor.model} ({provider})"
            if provider else descriptor.model)
        return {
            "value": "loki-saved",
            "name": f"{label} [saved connection]",
            "description": (
                f"saved session connection; {descriptor.protocol}; "
                f"api={descriptor.chat_url}"),
        }, descriptor

    def _disconnected_choice(self):
        description = (
            self._configuration_error
            or "No usable connection is configured. Select a model to "
               "connect this session.")
        return {
            "value": "loki-disconnected",
            "name": "No model configured",
            "description": description,
        }, _DISCONNECTED

    def _set_choices(self, choices, current_value: str):
        deduplicated = {}
        leaves = {}
        for option, leaf in choices:
            value = option.get("value")
            if not isinstance(value, str) or not value:
                continue
            if value not in deduplicated:
                deduplicated[value] = dict(option)
                leaves[value] = leaf
        if current_value not in deduplicated:
            raise RuntimeError(
                f"current ACP config value {current_value!r} has no option")
        self._model_options = list(deduplicated.values())
        self._option_leaves = leaves
        self._current_option_value = current_value

    def _install_initial_choices(
            self, saved_descriptor: ConnectionDescriptor | None):
        explicit = self._explicit_choice()
        choices = []

        if explicit is not None and loki.explicit_api_base_configured(
                loki.CREDENTIALS):
            choices.append(explicit)
            self._set_choices(choices, "loki-explicit")
            return

        if saved_descriptor is not None:
            try:
                config = loki.config_from_connection_descriptor(
                    saved_descriptor, loki.CREDENTIALS)
                loki.apply_runtime_config(config)
                choices.append(self._saved_choice(saved_descriptor))
                if explicit is not None:
                    choices.append(explicit)
                self._set_choices(choices, "loki-saved")
                return
            except (ConnectionDescriptorError, loki.protocols.ProtocolError,
                    ValueError) as error:
                self.session.runtime_config = None
                self._configuration_error = (
                    f"Saved connection is unavailable: {error}")

        if explicit is not None:
            # This covers callers that construct a Worker directly after
            # applying a complete explicit config.
            choices.append(explicit)
            self._set_choices(choices, "loki-explicit")
            return

        if (loki.explicit_api_base_configured(loki.CREDENTIALS)
                and not self._configuration_error):
            self._configuration_error = (
                "The explicit LOKI_* connection is incomplete; set "
                "LOKI_MODEL or choose a catalog model.")
        self.session.runtime_config = None
        choices.append(self._disconnected_choice())
        self._set_choices(choices, "loki-disconnected")

    async def prepare_open(self, params: dict) -> dict:
        if self._pending_open is not None:
            raise acps.TransportError(
                "session open is already prepared",
                code=acps.INVALID_PARAMS,
            )
        cwd = params.get("cwd")
        if (not isinstance(cwd, str) or not os.path.isabs(cwd)
                or not os.path.isdir(cwd)):
            raise acps.TransportError(
                "session/open requires an existing absolute cwd",
                code=acps.INVALID_PARAMS,
            )
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise acps.TransportError(
                "session/open requires sessionId",
                code=acps.INVALID_PARAMS,
            )
        self.session_id = session_id

        open_method = params.get("openMethod")
        if open_method not in (
                "session/new", "session/load", "session/resume"):
            raise acps.TransportError(
                f"unsupported session opening method {open_method!r}",
                code=acps.INVALID_PARAMS,
            )
        saved_descriptor = None
        connection_update = _UNCHANGED
        if open_method != "session/new":
            if os.path.basename(session_id) != session_id:
                raise acps.TransportError(
                    "saved session id must be a local chat-log name",
                    code=acps.INVALID_PARAMS,
                )
            path = os.path.join(
                loki.CHAT_LOG_DIR, loki.chat_log_filename(session_id))
            if not os.path.isfile(path):
                raise acps.TransportError(
                    f"no saved session named {session_id!r}",
                    code=acps.INVALID_PARAMS,
                )
            try:
                loki.load_chat_log(path)
                saved_descriptor = loki.connection_from_session_state(
                    self.session.session_state)
            except (OSError, ValueError, formats.TranscriptFormatError,
                    ConnectionDescriptorError) as error:
                raise acps.TransportError(
                    f"could not load saved session {session_id!r}: {error}",
                    code=acps.INVALID_PARAMS,
                ) from error
            saved_cwd = self.session.session_state.get("shell_cwd")
            try:
                same_cwd = (
                    isinstance(saved_cwd, str)
                    and os.path.isabs(saved_cwd)
                    and os.path.samefile(saved_cwd, cwd)
                )
            except (OSError, ValueError):
                same_cwd = False
            if not same_cwd:
                # Loading and resuming continue one conversation; cwd is part
                # of that identity, not permission to relocate it. Directory
                # identity accepts equivalent symlink spellings safely.
                raise acps.TransportError(
                    "restored session cwd does not match its saved cwd",
                    code=acps.INVALID_PARAMS,
                )
            self.session.shell_cwd = cwd
        else:
            self.session.shell_cwd = cwd
            loki.new_chat_log(os.path.join(
                loki.CHAT_LOG_DIR,
                loki.chat_log_filename(session_id),
            ))

        # Await discovery so a resumed subscription uses fresh exact-slug
        # request data before any config choice or prompt becomes available.
        explicit = loki.explicit_connection_option(loki.CREDENTIALS)
        catalog = {}
        discovered = []
        try:
            catalog, groups = await modelsdev.ensure_index(
                credential_authority=self.session.credential_authority,
                diagnostic_writer=lambda message: print(
                    message, file=sys.stderr),
            )
            discovered = modelsdev.flattened_config_option_choices(
                loki.CREDENTIALS,
                explicit_connection=explicit,
                groups=groups,
            )
        except Exception as error:
            print(
                f"models.dev discovery failed: {error!r}",
                file=sys.stderr,
            )

        if saved_descriptor is not None:
            original_descriptor = saved_descriptor
            try:
                saved_descriptor = loki.reconcile_connection_descriptor(
                    saved_descriptor, catalog)
            except ValueError as error:
                self._configuration_error = (
                    f"Saved connection is unavailable: {error}")
                saved_descriptor = None
                connection_update = None
            else:
                if saved_descriptor != original_descriptor:
                    connection_update = saved_descriptor

        self._install_initial_choices(saved_descriptor)
        if discovered:
            self._install_catalog_choices(discovered)
        runtime_config = self.session.runtime_config
        authorization_descriptor = None
        if (self._current_option_value == "loki-saved"
                and runtime_config is not None):
            authorization_descriptor = (
                loki.connection_descriptor_from_config(runtime_config))
        # Preparation must not leave a network-capable session reachable.
        # The front holds the only commit authority after user approval.
        self.session.runtime_config = None
        self._pending_open = PendingSessionOpen(
            runtime_config=runtime_config,
            connection_update=connection_update,
            # Authorization separates prepare from commit, so the exact ACP
            # method must remain explicit until replay can safely occur.
            open_method=open_method,
        )
        result = {}
        if authorization_descriptor is not None:
            result["authorizationConnection"] = (
                authorization_descriptor.to_dict())
        return result

    def commit_open(self) -> dict:
        pending = self._pending_open
        if pending is None:
            raise acps.TransportError(
                "session open was not prepared",
                code=acps.INVALID_PARAMS,
            )
        self._pending_open = None
        self.session.runtime_config = pending.runtime_config
        if pending.connection_update is not _UNCHANGED:
            if pending.connection_update is None:
                self.session.session_state.pop("connection", None)
                loki.mark_chat_log_dirty()
            else:
                loki.set_session_connection(pending.connection_update)
            loki.save_chat_log()
        # ACP load reconstructs the client's view; resume reconstructs only
        # the agent's context and must not emit historical session updates.
        if pending.open_method == "session/load":
            self._replay_transcript()
        return {"configOptions": self.config_options()}

    def _install_catalog_choices(self, discovered):
        if not discovered:
            return
        existing = [
            (option, self._option_leaves[option["value"]])
            for option in self._model_options
        ]
        self._set_choices(
            existing + list(discovered),
            self._current_option_value,
        )

    def set_config_option(self, params: dict) -> dict:
        config_id = params.get("configId")
        if config_id == "reasoning_effort":
            value = params.get("value")
            if value == _MODEL_DEFAULT_CONFIG_VALUE:
                effort = None
            elif (isinstance(value, str)
                    and value.startswith(_EFFORT_CONFIG_VALUE_PREFIX)):
                effort = value[len(_EFFORT_CONFIG_VALUE_PREFIX):]
            else:
                raise acps.TransportError(
                    f"unknown reasoning effort value {value!r}",
                    code=acps.INVALID_PARAMS,
                )
            try:
                loki.set_reasoning_effort(effort)
            except (reasonings.ReasoningEffortError, ValueError) as error:
                raise acps.TransportError(
                    str(error), code=acps.INVALID_PARAMS) from error
            return {"configOptions": self.config_options()}
        if config_id != "model":
            raise acps.TransportError(
                f"unknown config option {config_id!r}",
                code=acps.INVALID_PARAMS,
            )
        value = params.get("value")
        leaf = self._option_leaves.get(value)
        if leaf is None:
            raise acps.TransportError(
                f"unknown model value {value!r}",
                code=acps.INVALID_PARAMS,
            )

        if leaf is _DISCONNECTED:
            self.session.runtime_config = None
            self.session.session_state.pop("connection", None)
            loki.mark_chat_log_dirty()
        elif isinstance(leaf, modelsdev.ExplicitConnectionOption):
            loki.apply_runtime_config(loki.build_config_from_env(
                credentials=loki.CREDENTIALS))
        elif isinstance(leaf, ConnectionDescriptor):
            loki.apply_runtime_config(
                loki.config_from_connection_descriptor(
                    leaf, loki.CREDENTIALS))
        else:
            provider_id, provider_entry, model_entry = leaf
            loki.apply_runtime_config(
                loki.config_from_modelsdev_selection(
                    provider_id, provider_entry, model_entry,
                    loki.CREDENTIALS))

        self._current_option_value = value
        descriptor = loki.active_connection_descriptor()
        if descriptor is not None:
            loki.set_session_connection(descriptor)
        loki.save_chat_log()
        return {"configOptions": self.config_options()}

    def _replay_transcript(self) -> None:
        blocks = replays.classify_transcript(
            self.session.transcript_items)
        replay_call = 0
        for kind, text, key in blocks:
            if kind == "user":
                update = {
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": text},
                }
            elif kind == "tool":
                replay_call += 1
                title, call_id = text, key
                update = {
                    "sessionUpdate": "tool_call",
                    "toolCallId": str(
                        call_id or f"replay-{replay_call}"),
                    "title": title.split("\n")[0][:120],
                    "kind": "other",
                    "status": "completed",
                }
            else:
                update = {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                }
            self.write(acps.notification("session/update", {
                "sessionId": self.session_id,
                "update": update,
            }))

    # -- prompting --------------------------------------------------------

    async def prompt(
            self, params: dict,
            reasoning_effort=_PROMPT_EFFORT_UNSET) -> dict:
        if reasoning_effort is _PROMPT_EFFORT_UNSET:
            reasoning_effort = loki.effective_reasoning_effort()
        blocks = params.get("prompt")
        if not isinstance(blocks, list):
            raise acps.TransportError(
                "prompt must be an array of content blocks",
                code=acps.INVALID_PARAMS,
            )
        supported_types = {"text", "resource_link"}
        unsupported = [
            block.get("type") if isinstance(block, dict) else type(block).__name__
            for block in blocks
            if (not isinstance(block, dict)
                or block.get("type") not in supported_types)
        ]
        if unsupported:
            raise acps.TransportError(
                "unsupported prompt content block types: "
                + ", ".join(map(str, unsupported)),
                code=acps.INVALID_PARAMS,
            )
        parts = []
        for block in blocks:
            if block["type"] == "text":
                if not isinstance(block.get("text"), str):
                    raise acps.TransportError(
                        "text prompt blocks require string text",
                        code=acps.INVALID_PARAMS,
                    )
                if block["text"]:
                    parts.append(block["text"])
                continue
            if (not isinstance(block.get("name"), str)
                    or not isinstance(block.get("uri"), str)):
                raise acps.TransportError(
                    "resource_link prompt blocks require string name and uri",
                    code=acps.INVALID_PARAMS,
                )
            resource = {
                key: block[key]
                for key in (
                    "name", "uri", "title", "description",
                    "mimeType", "size")
                if block.get(key) is not None
            }
            parts.append(
                "[ACP resource link]\n"
                + json.dumps(resource, ensure_ascii=False))
        user_text = "\n".join(parts)
        if not user_text:
            raise acps.TransportError(
                "prompt contains no text", code=acps.INVALID_PARAMS)

        self.cancel_event.clear()
        self.session_id = params.get("sessionId") or self.session_id
        self.session.transcript_items.append(
            formats.message_item("user", user_text))
        loki.mark_chat_log_dirty()
        events = []
        mapper_state: dict = {}

        def on_event(event):
            events.append(event)
            for update in acp_events.map_event(
                    self.session_id, event, mapper_state):
                self.write(acps.notification("session/update", update))

        self._active_prompt_reasoning_effort = reasoning_effort
        try:
            try:
                await self._run_turn(on_event)
            except BaseException:
                if not self.cancel_event.is_set():
                    raise
                if not any(
                        event.get("type") == "response_cancelled"
                        for event in events):
                    on_event({
                        "type": "response_cancelled",
                        "partial": False,
                        "saved": False,
                    })
        finally:
            self._active_prompt_reasoning_effort = None
        failure = self._turn_failure(events)
        if failure:
            raise TurnFailure(failure)
        stop_reason = self._stop_reason(events)
        return {"stopReason": stop_reason}

    async def _run_turn(self, on_event):
        try:
            if not loki.current_model():
                detail = self._configuration_error or (
                    "No model selected; configure LOKI_* or select a model.")
                on_event({
                    "type": "assistant_message",
                    "content": detail,
                })
                return
            cancel_check = self.cancel_event.is_set

            async def chat_fn(
                    items, on_text_delta, *, codex_turn_state):
                kwargs = {
                    "on_text_delta": on_text_delta,
                    "cancel_check": cancel_check,
                    "codex_turn_state": codex_turn_state,
                    "reasoning_effort":
                        self._active_prompt_reasoning_effort,
                }
                return await loki.async_chat_completion(
                    items, loki.TOOLS, True, False, **kwargs)

            await loki.run_tool_loop_async(
                self.session.transcript_items,
                chat_fn=chat_fn,
                on_event=on_event,
                cancel_check=cancel_check,
                cancel_event=self.cancel_event,
                stream_chat=True,
                on_response=lambda turn, event: loki.mark_chat_log_dirty(),
                reasoning_effort=self._active_prompt_reasoning_effort,
            )
        finally:
            loki.save_chat_log()

    @staticmethod
    def _turn_failure(events: list) -> str | None:
        for event in events:
            kind = event.get("type")
            if kind == "api_error":
                error = event.get("error")
                return (
                    error.formatted()
                    if hasattr(error, "formatted") else str(error))
            if kind in ("network_error", "stream_error",
                        "transcript_error", "provider_error"):
                return f"{kind.replace('_', ' ')}: {event.get('error')}"
            if kind == "response_failed":
                return (
                    "provider returned a failed response: "
                    f"{event.get('protocol_data')!r}")
        return None

    @staticmethod
    def _stop_reason(events: list) -> str:
        kinds = {event.get("type") for event in events}
        if "response_cancelled" in kinds:
            return "cancelled"
        if "max_loops" in kinds:
            return "max_turn_requests"
        if "response_incomplete" in kinds:
            return "max_tokens"
        if "response_refusal" in kinds:
            return "refusal"
        return "end_turn"
