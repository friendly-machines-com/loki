"""Single-session ACP worker behind the front process."""

from __future__ import annotations

import asyncio
import collections
import copy
import json
import os
import sys
import uuid
from dataclasses import dataclass

from . import compactions, connections, model_requests, paths, savefiles
from . import (
    acp_commands,
    acps,
    acp_events,
    formats,
    loki,
    models as modelsdev,
    recoveries,
    replays,
)
from .connections import ConnectionDescriptor, ConnectionDescriptorError
from .lifetimes import finish_cleanup
from .sessions import Session


_DISCONNECTED = object()
_UNCHANGED = object()
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

    runtime_config: connections.RuntimeConfig | None
    connection_update: object
    open_method: str


@dataclass(frozen=True)
class SubmittedPrompt:
    # ACP insertion is already acknowledged; execution/model consumption may
    # wait. Keep presentation identity with its input, never another transcript.
    params: dict
    acp_message: dict


class TurnFailure(Exception):
    """A turn or explicit-operation failure requiring an ACP v1 request error."""


class Worker:
    def __init__(self, session: Session, write, session_id: str = "worker"):
        self.session = session
        self._write = write
        self.session_id = session_id
        # Chosen only by the live front's prepare-open envelope, never a log.
        self.protocol_version = 1
        self._v2_pending = collections.deque()
        self._foreground_state = "idle"
        self._v2_cancelled_compaction = None
        # The session reuses one event, cleared only when admitting a new
        # prompt. session/cancel and close set it; requests query is_set, and
        # tools/recovery wait on the same state. Replays must never clear it.
        self.cancel_event = asyncio.Event()
        # Admission/configuration guards and close borrow this owner through
        # _answer's final response write, including hook/backoff/cleanup time.
        self._prompt_task: asyncio.Task | None = None
        # Reverse requests to the front (model questions). The front owns
        # the client channel; the worker only ever asks within its session.
        self._reverse_requests: dict[str, asyncio.Future] = {}
        self._next_reverse_id = 0
        # Set at session open from the front's report of the client's
        # elicitation capability. It decides whether the model is offered
        # the Ask tool at all.
        self.can_ask_user = False
        self.can_report_compaction = False
        # Controls retain their own wire IDs, not another prompt task. Receipt
        # must return immediately so this same input loop can receive cancellation
        # and reverse-question answers while the existing owner does the work.
        self._compaction_replies: list = []
        self.compaction_requests = compactions.Requests(self._settle_compaction_replies)
        self._model_options: list[dict] = []
        self._option_leaves: dict[str, object] = {}
        self._current_option_value: str | None = None
        self._configuration_error: str | None = None
        # An uncommitted open is inert, not merely model-less: local commands
        # and option changes can execute/write logs without leasing credentials.
        # Prompt/config admission therefore checks this existing approval owner.
        self._pending_open: PendingSessionOpen | None = None
        # Images staged by /image, attached to the next real-turn prompt.
        self._pending_images: list = []

    def write(self, message):
        if self.protocol_version == 2 and message.get("method") == "session/update":
            params = message["params"]
            message = {**message, "params": {**params, "update": acp_events.update_v2(params["update"])}}
        self._write(message)

    def _state(self, state, *, stop_reason=None, error=None, snapshot=False):
        if self.protocol_version != 2:
            return
        if self._foreground_state == state and stop_reason is None and not snapshot:
            return
        self._foreground_state = state
        update = {"sessionUpdate": "state_update", "state": state}
        if stop_reason is not None:
            update["stopReason"] = stop_reason
        if error is not None:
            code = getattr(error, "code", None)
            update["error"] = {"code": code if isinstance(code, int) and not isinstance(code, bool) else acps.INTERNAL_ERROR,
                               "message": str(error)}
            if getattr(error, "data", None) is not None:
                update["error"]["data"] = error.data
        self.write(acps.notification("session/update", {"sessionId": self.session_id, "update": update}))

    def _v2_insert(self, message, admitted):
        """Logically insert a live ACP message, not a model prompt or receipt ACK.

        Preflight all envelopes before insertion. The caller-owned record marks
        the actual insertion boundary even if a subsequent writer fails, so an
        already-inserted request can never get a second/error response. Local
        commands remain live-only; ordinary user IDs are retained with their one
        canonical message when the shared interpreter consumes it.
        """
        message_id = str(uuid.uuid4())
        content = copy.deepcopy(message["params"]["prompt"])
        reply = acps.response(message["id"], {"messageId": message_id})
        updates = [acps.notification("session/update", {"sessionId": self.session_id, "update": {
            "sessionUpdate": "user_message_chunk", "messageId": message_id, "content": block}}) for block in content]
        for packet in [reply, *updates]:
            for encoded in acps.encoded_messages(packet):
                del encoded
        admitted.update(messageId=message_id, content=content)
        self.write(reply)
        for packet in updates:
            self.write(packet)

    def _cancel_v2_pending(self):
        # These messages were inserted and ACKed, not held before insertion.
        # Cancellation stops their queued execution; it cannot send a second
        # error response or erase the user's live ACP message.
        self._v2_pending.clear()

    async def _run_v2_prompts(self):
        """The existing foreground owner serializes work, not insertion replies.

        Valid submissions insert immediately; ordinary model consumption queues
        in receipt order, while compaction controls share the same turn-owned
        operation. Queue cancellation covers then-pending execution, not insertion
        or later input. Settings freeze at actual runtime-turn admission.
        This is neither another inference engine nor a task per control.
        """
        try:
            while self.session.active:
                if self.compaction_requests.pending is not None:
                    self.cancel_event.clear()
                    self._state("running")
                    outcome = await self._perform_idle_compaction()
                    if outcome["type"] == "compaction_failed":
                        self._state("idle", stop_reason="error", error=outcome["error"])
                    else:
                        self._state("idle", stop_reason="cancelled" if outcome["type"] == "compaction_cancelled" else "end_turn")
                    continue
                if not self._v2_pending:
                    break
                submitted = self._v2_pending.popleft()
                self.cancel_event.clear()
                self._state("running")
                try:
                    thinking = loki.capture_turn_settings()
                    result = await self._prompt(submitted.params, thinking, acp_message=submitted.acp_message)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._state("idle", stop_reason="error", error=error)
                else:
                    self._state("idle", stop_reason=result["stopReason"])
        finally:
            self._cancel_v2_pending()
            self._v2_cancelled_compaction = None
            if not self.session.active or asyncio.current_task().cancelling():
                self.compaction_requests.finish_pending(cancelled=True)
            if self._foreground_state != "idle":
                self._state("idle", stop_reason="cancelled" if self.cancel_event.is_set() else "error")

    def _handle_v2_prompt(self, message):
        admitted = {}
        first = self._prompt_task is None or self._prompt_task.done()
        try:
            compact = self._is_compact_control(message.get("params") or {})
            if first:
                self.cancel_event.clear()
                self._state("running")
            self._v2_insert(message, admitted)
            if compact:
                self.compaction_requests.request()
            else:
                self._v2_pending.append(SubmittedPrompt(message["params"], admitted))
            if first:
                self._prompt_task = asyncio.create_task(self._run_v2_prompts(), name="acp-worker-v2-foreground")
        except acps.TransportError as error:
            if "messageId" in admitted:
                raise
            self.write(acps.error_response(message["id"], {
                "code": -32800 if not self.session.active else error.code, "message": str(error)}))
            if first:
                self._state("idle", stop_reason="error", error=error)

    def _is_compact_control(self, params):
        if not self.session.active:
            raise acps.TransportError("This conversation has closed", code=acps.INVALID_PARAMS)
        if self._pending_open is not None:
            raise acps.TransportError("session open is awaiting approval", code=acps.INVALID_PARAMS)
        command = acp_commands.parse(self._prompt_text(params))
        return command is not None and command[0] == "compact" and not command[1]

    @staticmethod
    def _compaction_result(outcome):
        if outcome["type"] == "compaction_failed":
            raise TurnFailure("Context compaction failed: " + compactions.format_error(outcome["error"]))
        return {"stopReason": "cancelled" if outcome["type"] == "compaction_cancelled" else "end_turn"}

    def _settle_compaction_replies(self, outcome):
        # _answer owns one ordinary reply; coalesced controls retain independent
        # wire identities. Take the whole batch before writes, attempt every ID
        # even on failure, then expose errors to the existing channel/close owner.
        replies = self._compaction_replies
        self._compaction_replies = []
        errors = []
        for request_id in replies:
            try:
                try:
                    result = self._compaction_result(outcome)
                except TurnFailure as error:
                    self.write(acps.error_response(request_id, {"code": acps.INTERNAL_ERROR, "message": str(error)}))
                else:
                    self.write(acps.response(request_id, result=result))
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Could not deliver compaction replies", errors)

    def _event_updates(self, event, state):
        if event.get("type") == "compaction_failed":
            event = {**event, "text": "Context compaction failed: " + compactions.format_error(event["error"])}
        return acp_events.map_event(self.session_id, event, state,
                                    compaction=self.protocol_version == 2 or self.can_report_compaction,
                                    protocol_version=self.protocol_version)

    async def _perform_idle_compaction(self, thinking=None):
        # The admitting owner captures settings and resets cancellation once. A joining control
        # or a scheduled-before-start cancellation must not reset that event.
        def on_event(event):
            for update in self._event_updates(event, {}):
                self.write(acps.notification("session/update", update))

        async def operation(started):
            # Reuse the registered idle-command adapter inside the actual owner,
            # never from busy receipt or as a second independent execution path.
            admitted = thinking if thinking is not None else loki.capture_turn_settings()
            result = await acp_commands.run(
                "/compact", self.session, cancel_event=self.cancel_event,
                thinking=admitted, allow_ask=self.can_ask_user, on_start=started)
            return result.text

        return await self.compaction_requests.run(operation, on_event)

    async def _compaction_tail(self):
        # _prompt_task also spans final hooks/reply writing. Requests received
        # after loop exit belong to this idle tail, not an already-finalized turn.
        while self.compaction_requests.pending is not None and self.session.active:
            self.cancel_event.clear()
            await self._perform_idle_compaction()
        if not self.session.active:
            self.compaction_requests.finish_pending(cancelled=True)

    async def handle(self, message: dict, concurrent: bool = False):
        method = message.get("method")
        request_id = message.get("id")
        if method is None or request_id is None:
            return
        if self.protocol_version == 2 and method == "session/prompt":
            self._handle_v2_prompt(message)
            return
        if method == "session/prompt":
            try:
                params = message.get("params") or {}
                if (self._prompt_task is not None and not self._prompt_task.done()
                        and self._is_compact_control(params)):
                    self.compaction_requests.request()
                    self._compaction_replies.append(request_id)
                    return
                thinking = self._prepare_prompt(params)
            except acps.TransportError as error:
                self.write(acps.error_response(
                    request_id, {"code": error.code, "message": str(error)}))
                return
            if concurrent:
                self._prompt_task = asyncio.create_task(
                    self._answer(message, thinking=thinking),
                    name=f"acp-worker-prompt-{request_id}")
            else:
                self._prompt_task = asyncio.current_task()
                try:
                    await self._answer(message, thinking=thinking)
                finally:
                    self._prompt_task = None
            return
        await self._answer(message)

    async def close(self):
        # The prompt task already owns hooks and creation through its final
        # response. Join that owner; Session.close_state later retires manager
        # and Store once, rather than creating another retained close lifecycle.
        task = self._prompt_task
        try:
            self.session.begin_close()
        finally:
            self.cancel_event.set()
            self._cancel_v2_pending()
            for future in list(self._reverse_requests.values()):
                if not future.done():
                    future.set_exception(acps.TransportError(
                        "the session closed while a question was open"))
            if task is not None:
                if not task.done() and not task.cancelling():
                    task.cancel()

                async def join_prompt():
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                try:
                    await finish_cleanup(join_prompt(), preserve_failure=sys.exc_info()[0] is not None)
                finally:
                    self.compaction_requests.finish_pending(cancelled=True)
            else:
                self.compaction_requests.finish_pending(cancelled=True)
            # A close can cancel the foreground task before its first byte of
            # execution; that task's finally cannot then publish completion.
            if self._foreground_state != "idle":
                self._state("idle", stop_reason="cancelled")

    async def _answer(
            self, message: dict,
            thinking: model_requests.TurnThinkingSettings | None = None):
        request_id = message.get("id")
        try:
            result = await self.dispatch(
                message.get("method"),
                message.get("params") or {},
                thinking=thinking,
            )
        except acps.TransportError as error:
            self.write(acps.error_response(
                request_id, {"code": error.code, "message": str(error)}))
        except Exception as error:
            self.write(acps.error_response(
                request_id, {"code": acps.INTERNAL_ERROR, "message": str(error)}))
        else:
            try:
                self.write(acps.response(request_id, result=result))
            except acps.TransportError as error:
                # Encoding failed before any frame byte; retain the v1 error path.
                self.write(acps.error_response(
                    request_id, {"code": error.code, "message": str(error)}))
        if asyncio.current_task() is self._prompt_task:
            await self._compaction_tail()

    async def dispatch(
            self, method: str, params: dict,
            thinking: model_requests.TurnThinkingSettings | None = None):
        if not self.session.active:
            raise acps.TransportError(
                "This conversation has closed", code=acps.INVALID_PARAMS)
        if method == "session/prompt":
            return await self.prompt(
                params,
                thinking=thinking,
            )
        if method == "session/cancel":
            if self.protocol_version == 2:
                self._v2_cancelled_compaction = self.compaction_requests.pending
                self.compaction_requests.finish_pending(cancelled=True, expected=self._v2_cancelled_compaction)
                self._cancel_v2_pending()
            self.cancel_event.set()
            return {}
        if method == "session/prepare_open":
            return await self.prepare_open(params)
        if method == "session/commit_open":
            return self.commit_open()
        if (self._pending_open is not None
                and method in ['session/set_config_option', 'session/describe_config_selection']):
            raise acps.TransportError('session open is awaiting approval', code=acps.INVALID_PARAMS)
        # Describe is the front's first step toward a configuration change.
        # Apply the same active-turn guard there, before the front can ask for
        # approval or persist a pin for a change we would subsequently refuse.
        # The worker remains the authority on its active prompt, including the
        # existing exception for next-turn reasoning-effort updates.
        if (method in ["session/set_config_option",
                       "session/describe_config_selection"]
                and self._prompt_task is not None
                and not self._prompt_task.done()
                and params.get("configId") not in [
                    "reasoning_effort", "thinking_mode", "thinking_budget", "reasoning_retention", "reasoning_traces"]):
            raise acps.TransportError(
                "cannot change session configuration while a prompt is running",
                code=acps.INVALID_PARAMS)
        if method == "session/set_config_option":
            return self.set_config_option(params)
        if method == "session/describe_config_selection":
            return self.describe_config_selection(params)
        raise acps.TransportError(
            f"worker does not implement {method}",
            code=acps.METHOD_NOT_FOUND,
        )

    def describe_config_selection(self, params: dict) -> dict:
        """Describe effective request destinations; only the front owns consent."""
        if params.get("configId") != "model":
            return {}
        leaf = self._option_leaves.get(params.get("value"))
        if leaf is None:
            raise acps.TransportError("unknown model value", code=acps.INVALID_PARAMS)
        if leaf is _DISCONNECTED:
            return {}
        if isinstance(leaf, modelsdev.ExplicitConnectionOption):
            config = connections.build_config_from_env(credentials=loki.CREDENTIALS)
        elif isinstance(leaf, ConnectionDescriptor):
            config = connections.config_from_connection_descriptor(leaf, loki.CREDENTIALS)
        else:
            provider_id, provider_entry, model_entry = leaf
            config = connections.config_from_modelsdev_selection(
                provider_id, provider_entry, model_entry, loki.CREDENTIALS)
        if config.auth_spec is None:
            return {}
        return {"credential": config.auth_spec.credential.encode(),
                "destinations": config.credential_destinations()}

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
        config = loki.current_config()
        mode, budget, retention = loki.effective_thinking_settings()
        if config is not None:
            provider = config.chat_provider
            modes = provider.thinking_modes(config.model)
            if modes:
                options.append({
                    "id": "thinking_mode", "name": "Thinking mode", "category": "other", "type": "select",
                    "currentValue": mode or "default",
                    "options": [{"value": "default", "name": "Model default"}]
                    + [{"value": value, "name": value} for value in modes]})
            if "manual" in modes:
                values = [{"value": "default", "name": "Not selected"}]
                if budget is not None:
                    values.append({"value": str(budget), "name": str(budget)})
                options.append({
                    "id": "thinking_budget", "name": "Thinking-token allowance", "category": "other", "type": "select",
                    "currentValue": str(budget) if budget is not None else "default", "options": values,
                    "description": "Set a number with /thinking budget N; enable manual thinking with /thinking mode manual budget N."})
            if provider.reasoning_preservation(config.model) is not None:
                options.append({
                    "id": "reasoning_retention", "name": "Reuse earlier thinking", "category": "other", "type": "select",
                    "currentValue": retention,
                    "options": [{"value": "default", "name": "Default"}, {"value": "preserve", "name": "Preserve"}]})
        options.append({
            "id": "reasoning_traces", "name": "Thinking traces", "category": "other", "type": "select",
            "currentValue": self.session.reasoning_traces,
            "options": [{"value": "off", "name": "Hidden"}, {"value": "on", "name": "Show thinking"}]})
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
            "name": (
                loki.reasoning_effort_default_text()
                or "Model default"),
            "description": (
                "Use the selected model or provider's own reasoning "
                "effort default."),
        }]
        values.extend({
            "value": _EFFORT_CONFIG_VALUE_PREFIX + value,
            "name": value,
        } for value in profile.values if isinstance(value, str))
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
        explicit = connections.explicit_connection_option(loki.CREDENTIALS)
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

        if explicit is not None and connections.explicit_api_base_configured(
                loki.CREDENTIALS):
            choices.append(explicit)
            self._set_choices(choices, "loki-explicit")
            return

        if saved_descriptor is not None:
            try:
                config = connections.config_from_connection_descriptor(
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

        if (connections.explicit_api_base_configured(loki.CREDENTIALS)
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
        if not savefiles.valid_chat_id(session_id):
            raise acps.TransportError(
                "session name is invalid",
                code=acps.INVALID_PARAMS,
            )
        open_method = params.get("openMethod")
        if open_method not in [
                "session/new", "session/load", "session/resume"]:
            raise acps.TransportError(
                f"unsupported session opening method {open_method!r}",
                code=acps.INVALID_PARAMS,
            )
        # Validate both new and restored IDs before state publication or model
        # discovery. A tool-created filename cannot widen the allowed alphabet.
        directory = paths.chat_log_dir_for(cwd)
        try:
            path = (savefiles.new_chat_log_path(directory, session_id)
                    if open_method == 'session/new'
                    else savefiles.saved_chat_log_path(directory, session_id))
        except (OSError, ValueError) as error:
            raise acps.TransportError(
                f'could not select session name {session_id!r}: {error}',
                code=acps.INVALID_PARAMS) from error
        if path is None:
            raise acps.TransportError(
                f'no saved session named {session_id!r}', code=acps.INVALID_PARAMS)
        self.session_id = session_id
        # The front relays whether its client can render questions. Saved
        # conversation data cannot supply this live capability.
        version = params.get('protocolVersion', 1)
        if isinstance(version, bool) or not isinstance(version, int) or version not in acps.SUPPORTED_PROTOCOL_VERSIONS:
            raise acps.TransportError('unsupported worker protocol version', code=acps.INVALID_PARAMS)
        self.protocol_version = version
        self.can_ask_user = params.get('formElicitation') is True
        self.can_report_compaction = version == 2 or params.get('compaction') is True
        saved_descriptor = None
        connection_update = _UNCHANGED
        if open_method != "session/new":
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
            loki.new_chat_log(path)

        # Await discovery so a resumed subscription uses fresh exact-slug
        # request data before any config choice or prompt becomes available.
        explicit = connections.explicit_connection_option(loki.CREDENTIALS)
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
                saved_descriptor = connections.reconcile_connection_descriptor(
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
                connections.connection_descriptor_from_config(runtime_config))
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
            result["authorizationDestinations"] = runtime_config.credential_destinations()
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
        self.session.context_snapshot()
        if pending.open_method == "session/load":
            self._replay_transcript()
            self._publish_context_usage()
        self._state(self._foreground_state, snapshot=True)
        return {
            "configOptions": self.config_options(),
            # Fresh monitor state follows the session reply, including on
            # resume, which must not replay conversation history.
            "lokiUpdates": [
                {"sessionUpdate": "available_commands_update",
                 "availableCommands": acp_commands.advertised_commands()},
                acp_events.plan(self.session_id, self.session.session_todos)["update"],
            ],
        }

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
        if config_id in ["thinking_mode", "thinking_budget", "reasoning_retention", "reasoning_traces"]:
            available = next((option for option in self.config_options() if option["id"] == config_id), None)
            value = params.get("value")
            if available is None or value not in [option["value"] for option in available["options"]]:
                raise acps.TransportError("use /thinking for this update", code=acps.INVALID_PARAMS)
            try:
                if config_id == "reasoning_traces":
                    loki.set_reasoning_traces(value)
                else:
                    name = {"thinking_mode": "mode", "thinking_budget": "budget", "reasoning_retention": "retention"}[config_id]
                    if name != "retention" and value == "default":
                        value = None
                    elif name == "budget":
                        value = int(value)
                    loki.set_thinking_controls({name: value})
            except (ValueError, OSError) as error:
                raise acps.TransportError(str(error), code=acps.INVALID_PARAMS) from error
            return {"configOptions": self.config_options()}
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
            except ValueError as error:
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
            loki.apply_runtime_config(connections.build_config_from_env(
                credentials=loki.CREDENTIALS))
        elif isinstance(leaf, ConnectionDescriptor):
            loki.apply_runtime_config(
                connections.config_from_connection_descriptor(
                    leaf, loki.CREDENTIALS))
        else:
            provider_id, provider_entry, model_entry = leaf
            loki.apply_runtime_config(
                connections.config_from_modelsdev_selection(
                    provider_id, provider_entry, model_entry,
                    loki.CREDENTIALS))

        self._current_option_value = value
        descriptor = loki.active_connection_descriptor()
        if descriptor is not None:
            loki.set_session_connection(descriptor)
        loki.save_chat_log()
        return {"configOptions": self.config_options()}

    def _replay_transcript(self) -> None:
        replay_call = 0
        previous_kind = None
        # Keep each canonical event available while rendering its blocks: the
        # flat text projection loses is_error. Replay must use that existing
        # flag, just like live updates, not infer status from a formatted title.
        user_ids = set()
        for index, event in enumerate(self.session.transcript_items):
            blocks = replays.classify_event(
                event, show_reasoning=self.session.reasoning_traces == 'on')
            if self.protocol_version == 2 and event.get('type') == 'message' and event.get('role') == 'user':
                native = event.get('protocol_data')
                metadata = native.get('loki') if isinstance(native, dict) else None
                metadata = metadata if isinstance(metadata, dict) else {}
                message_id = metadata.get('acp_message_id', f'history-user-{index}')
                if not isinstance(message_id, str) or not message_id or message_id in user_ids:
                    raise formats.TranscriptFormatError('saved ACP user message identity is invalid or duplicated')
                user_ids.add(message_id)
                content = metadata.get('acp_content')
                if content is None:
                    content = ([formats.text_block(block[1]) for block in blocks if block[0] == 'user']
                               if 'acp_message_id' not in metadata else event['content'])
                if not isinstance(content, list) or any(not isinstance(block, dict) for block in content):
                    raise formats.TranscriptFormatError('saved ACP user content is invalid')
                self.write(acps.notification('session/update', {'sessionId': self.session_id, 'update': {
                    'sessionUpdate': 'user_message', 'messageId': message_id, 'content': []}}))
                for block in content:
                    self.write(acps.notification('session/update', {'sessionId': self.session_id, 'update': {
                        'sessionUpdate': 'user_message_chunk', 'messageId': message_id, 'content': block}}))
                previous_kind = 'user'
                continue
            for kind, text, key in blocks:
                if kind == 'thought' and previous_kind == 'thought':
                    text = '\n\n' + text
                previous_kind = kind
                if kind == 'user':
                    update = {
                        'sessionUpdate': 'user_message_chunk',
                        'content': {'type': 'text', 'text': text},
                    }
                elif kind == 'tool':
                    replay_call += 1
                    title, call_id = text, key
                    update = {
                        'sessionUpdate': 'tool_call',
                        'toolCallId': str(call_id or f'replay-{replay_call}'),
                        'title': title.split('\n')[0][:120],
                        'kind': 'other',
                        'status': 'failed' if event.get('type') == 'tool_result' and event.get('is_error') else 'completed',
                    }
                else:
                    update = {
                        'sessionUpdate': 'agent_thought_chunk' if kind == 'thought' else 'agent_message_chunk',
                        'content': {'type': 'text', 'text': text},
                    }
                    if self.protocol_version == 2:
                        update['messageId'] = str(uuid.uuid4())
                        self.write(acps.notification('session/update', {'sessionId': self.session_id, 'update': {
                            'sessionUpdate': 'agent_thought' if kind == 'thought' else 'agent_message',
                            'messageId': update['messageId'], 'content': []}}))
                self.write(acps.notification('session/update', {
                    'sessionId': self.session_id,
                    'update': update,
                }))

    # -- asking the user through the front --------------------------------

    def resolve_reverse(self, request_id, message: dict) -> None:
        """Settle one reverse request with the front's response.

        A late response after its caller was cancelled has no future and
        is discarded, mirroring the front's own reply handling.
        """
        future = self._reverse_requests.pop(str(request_id), None)
        if future is None or future.done():
            return
        if "error" in message:
            error = message.get("error") or {}
            future.set_exception(acps.TransportError(
                str(error.get("message") or "front error"),
                code=error.get("code", acps.INTERNAL_ERROR),
            ))
        else:
            future.set_result(message.get("result"))

    async def request_input(
            self, question: str, options: list, *,
            multi_select: bool = False) -> dict:
        """Ask the front to put one model-authored question to the user.

        Scope and mode are deliberately absent: they belong to the front,
        which binds the ask to this session and refuses url mode.
        """
        self._next_reverse_id += 1
        request_id = f"ask-{self._next_reverse_id}"
        future = asyncio.get_running_loop().create_future()
        self._reverse_requests[request_id] = future
        try:
            self.write(acps.request(
                request_id, "session/request_input", {
                    "question": question,
                    "options": [
                        {"label": label, "description": description}
                        for label, description in options
                    ],
                    "multiSelect": multi_select,
                }))
            return await future
        finally:
            self._reverse_requests.pop(request_id, None)

    async def _ask_user(self, question: str, options: list, *,
                        multi_select: bool = False) -> dict:
        self._state("requires_action")
        try:
            outcome = await self.request_input(question, options, multi_select=multi_select)
        finally:
            if self.session.active and not self.cancel_event.is_set():
                self._state("running")
        if (isinstance(outcome, dict)
                and outcome.get("action") == "answered"
                and self.cancel_event.is_set()):
            # The turn ended while the question was open. The answer must
            # not drive a cancelled turn's continuation.
            return {"action": "cancelled"}
        return outcome

    # -- prompting --------------------------------------------------------

    @staticmethod
    def _prompt_text(params):
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
                for key in [
                    "name", "uri", "title", "description",
                    "mimeType", "size"]
                if block.get(key) is not None
            }
            parts.append(
                "[ACP resource link]\n"
                + json.dumps(resource, ensure_ascii=False))
        user_text = "\n".join(parts)
        if not user_text:
            raise acps.TransportError(
                "prompt contains no text", code=acps.INVALID_PARAMS)

        return user_text

    def _prepare_prompt(self, params, thinking=None):
        # Ordinary prompts and local commands retain overlap rejection. Exact
        # compaction controls are claimed separately by handle/prompt: they only
        # record intent and keep their own v1 replies, without monitor text in
        # the live assistant stream or a second ordinary turn.
        if not self.session.active:
            raise acps.TransportError(
                "This conversation has closed", code=acps.INVALID_PARAMS)
        if self._pending_open is not None:
            raise acps.TransportError('session open is awaiting approval', code=acps.INVALID_PARAMS)
        if self._prompt_task is not None and not self._prompt_task.done():
            raise acps.TransportError(
                "a prompt is already running for this session",
                code=acps.INVALID_PARAMS)
        user_text = self._prompt_text(params)
        command = acp_commands.parse(user_text)
        if command is None or command[0] in ["!", "compact"]:
            # Reset once at admission, before scheduling or a network command
            # awaits. /compact needs the same cancellation snapshot as inference;
            # the other slash-local shortcuts deliberately keep their old signal.
            # Overlap rejection above still owns the whole operation/reply.
            # A following preference update must not change the admitted
            # turn's first request; continuations reuse this same snapshot.
            if thinking is None:
                thinking = loki.capture_turn_settings()
            self.cancel_event.clear()
        return thinking

    async def prompt(
            self, params: dict,
            thinking: model_requests.TurnThinkingSettings | None = None) -> dict:
        task = asyncio.current_task()
        scheduled = self._prompt_task is task
        if (not scheduled and self._prompt_task is not None and not self._prompt_task.done()
                and self._is_compact_control(params)):
            future = self.compaction_requests.request()
            return self._compaction_result(await asyncio.shield(future))
        if not scheduled:
            thinking = self._prepare_prompt(params, thinking)
            self._prompt_task = task
        try:
            return await self._prompt(params, thinking)
        finally:
            # Inline callers own the task only for this prompt. Scheduled
            # callers remain owned through _answer's final response write.
            if not scheduled:
                try:
                    await self._compaction_tail()
                finally:
                    self._prompt_task = None

    async def _prompt(self, params, thinking, acp_message=None):
        user_text = self._prompt_text(params)
        self.session_id = params.get("sessionId") or self.session_id

        # A command is recognized before it reaches the transcript: like the
        # terminal front, local commands answer out of band and never become
        # conversation.  Skills are advertised but not intercepted, so their
        # text falls through to the model's Skill tool.
        outcome = None
        command = acp_commands.parse(user_text)
        if command is not None and command[0] == "compact":
            if command[1]:
                self.write(acps.notification("session/update", acp_events.agent_message_chunk(self.session_id, "Usage: /compact")))
                return {"stopReason": "end_turn"}
            future = self.compaction_requests.request()
            # Admission already captured thinking and reset cancellation once.
            # A pre-start cancel must not be cleared by the idle execution path.
            if self.cancel_event.is_set():
                self.compaction_requests.finish_pending(cancelled=True)
            else:
                await self._perform_idle_compaction(thinking)
            return self._compaction_result(future.result())
        cancelled_bang = (command is not None and command[0] == "!"
                          and self.cancel_event.is_set())
        if not cancelled_bang and user_text.lstrip().startswith(("/", "!")):
            try:
                outcome = await acp_commands.run(
                    user_text, self.session, cancel_event=self.cancel_event)
            except Exception as error:  # noqa: BLE001 - answered, not fatal
                if self.protocol_version == 2:
                    raise
                outcome = acp_commands.Outcome(
                    text=f"Command failed: {error}")
        if outcome is not None:
            if command is not None and command[0] in ["thinking", "trace"]:
                self.write(acps.notification("session/update", {
                    "sessionId": self.session_id,
                    "update": {"sessionUpdate": "config_option_update", "configOptions": self.config_options()}}))
            if outcome.text:
                self.write(acps.notification(
                    "session/update",
                    acp_events.agent_message_chunk(
                        self.session_id, outcome.text)))
            if outcome.image is not None:
                self._pending_images.append(outcome.image)
            if outcome.model_text is None:
                return {"stopReason": "end_turn"}
            user_text = outcome.model_text

        user_content = []
        if user_text:
            user_content.append(formats.text_block(user_text))
        user_content.extend(
            image.content_block() for image in self._pending_images)
        self._pending_images.clear()
        user_event = formats.message_item("user", user_content)
        if acp_message is not None:
            # ACP insertion identity must survive retained-user replay. Keep
            # original resource/command presentation only when runtime expansion
            # differs; none of this metadata becomes provider input or authority.
            metadata = {"acp_message_id": acp_message["messageId"]}
            if acp_message["content"] != user_content:
                metadata["acp_content"] = acp_message["content"]
            user_event["protocol_data"] = {"loki": metadata}
        self.session.transcript_items.append(user_event)
        loki.mark_chat_log_dirty()
        events = []
        mapper_state: dict = {}

        def on_event(event):
            events.append(event)
            for update in self._event_updates(event, mapper_state):
                self.write(acps.notification("session/update", update))

        turn_text = ""
        turn_error = None
        try:
            try:
                turn_text = await self._run_turn(on_event, thinking) or ""
            except BaseException as error:
                if not self.cancel_event.is_set() or any(event.get("type") == "compaction_failed" for event in events):
                    turn_error = error
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
            try:
                loki.save_chat_log()
            finally:
                await loki.run_turn_end_hooks_async(
                    events, turn_text, error=turn_error)
        failure = self._turn_failure(events)
        if failure:
            raise TurnFailure(failure)
        stop_reason = self._stop_reason(events)
        return {"stopReason": stop_reason}

    def _publish_context_usage(self, *, live=False):
        update = acp_events.context_usage(
            self.session_id, self.session.context_snapshot(live=live))
        if update is not None:
            self.write(acps.notification("session/update", update))

    async def _run_turn(self, on_event, thinking):
        if not loki.current_model():
            detail = self._configuration_error or (
                "No model selected; configure LOKI_* or select a model.")
            on_event({
                "type": "assistant_message",
                "content": detail,
            })
            return
        recovery = loki.make_turn_recovery(self.cancel_event)

        def on_response(turn, event):
            loki.mark_chat_log_dirty()
            self._publish_context_usage(live=True)

        # The existing prompt task owns replay/continuation, backoff and hook
        # cleanup through this await. _prompt then saves/notifies once and
        # _answer writes the sole reply; recovery adds no separate lifecycle.
        try:
            return await loki.run_tool_loop_async(
                self.session.transcript_items, on_event=on_event,
                cancel_event=self.cancel_event, recovery=recovery,
                stream_chat=True, on_response=on_response, thinking=thinking,
                ask_user=self._ask_user if self.can_ask_user else None,
                compaction_requests=self.compaction_requests)
        finally:
            # No await before sealing this loop. Later controls are serviced
            # after the original save/hook/reply by _answer's idle tail.
            if self.protocol_version == 2 and self.cancel_event.is_set() and self.session.active:
                # A command inserted AFTER session/cancel is a later operation,
                # not covered by that notification's admission cutoff.
                if self._v2_cancelled_compaction is not None:
                    self.compaction_requests.finish_pending(cancelled=True, expected=self._v2_cancelled_compaction)
            else:
                self.compaction_requests.finish_pending(
                    cancelled=self.cancel_event.is_set() or not self.session.active)

    @staticmethod
    def _turn_failure(events: list) -> str | None:
        for event in events:
            kind = event.get("type")
            if kind == "compaction_failed":
                return "Context compaction failed: " + compactions.format_error(event["error"])
            if kind == "api_error":
                error = event.get("error")
                return (
                    error.formatted()
                    if hasattr(error, "formatted") else str(error))
            if kind == "network_error":
                detail = recoveries.failure_fields(event)["error_message"]
                return f"network error: {detail}"
            if kind in ["stream_error", "transcript_error", "provider_error"]:
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
