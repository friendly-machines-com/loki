"""ACP worker: one single-session Loki process behind the front process.

The worker owns one Session -- exactly like the terminal front-end -- and
speaks JSON-RPC on its stdin/stdout (a socketpair provided by the front
process).  It handles the per-session methods: session/prompt,
session/cancel, session/set_config_option; the front process owns the
session-free methods (initialize, session/new, session/list).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from . import acps, acp_events, formats, loki, models as modelsdev, replays, savefiles
from .sessions import Session


class Worker:
    def __init__(self, session: Session, write, session_id: str = "worker"):
        self.session = session
        self.write = write
        self.session_id = session_id
        self.cancel_event = asyncio.Event()
        self._prompt_task: asyncio.Task | None = None
        self._model_options: list = []
        self._option_leaves: dict = {}

    # -- dispatch ---------------------------------------------------------

    async def handle(self, message: dict, concurrent: bool = False):
        method = message.get("method")
        request_id = message.get("id")
        if method is None or request_id is None:
            return  # notification or malformed; nothing to answer
        if concurrent and method == "session/prompt":
            if (self._prompt_task is not None
                    and not self._prompt_task.done()):
                self.write(acps.response(
                    request_id,
                    error={"code": acps.INVALID_PARAMS,
                           "message":
                               "a prompt is already running for this "
                               "session"}))
                return
            self._prompt_task = asyncio.get_running_loop().create_task(
                self._answer(message))
            return
        await self._answer(message)

    async def _answer(self, message: dict):
        method = message.get("method")
        request_id = message.get("id")
        try:
            result = await self.dispatch(method, message.get("params") or {})
        except Exception as error:  # surface as JSON-RPC error, never crash
            self.write(acps.response(
                request_id,
                error={"code": acps.INTERNAL_ERROR, "message": str(error)}))
            return
        self.write(acps.response(request_id, result=result))

    async def dispatch(self, method: str, params: dict):
        if method == "session/prompt":
            return await self.prompt(params)
        if method == "session/cancel":
            self.cancel_event.set()
            return {}
        if method == "session/open":
            return self.open(params)
        if method == "session/set_config_option":
            return self.set_config_option(params)
        raise acps.TransportError(
            f"worker does not implement {method}")

    def config_options(self) -> list:
        """The model select for the client, with currentValue applied."""
        current = loki.current_model()
        current_value = None
        for option in self._model_options:
            if option["value"] in (
                    current, f"loki-explicit") and (
                    option["value"] != "loki-explicit"
                    or current and loki.current_config()
                    and loki.current_config().provider_kind
                    == loki.protocols.DUMMY):
                current_value = option["value"]
                break
        options = []
        for option in self._model_options:
            entry = dict(option)
            options.append(entry)
        if current_value is None and self._option_leaves:
            # Fall back to matching the active provider/model pair.
            config = loki.current_config()
            if config is not None:
                wanted = f"{config.provider_id}/{current}"
                if any(o["value"] == wanted for o in options):
                    current_value = wanted
        return [{
            "id": "model",
            "name": "Model",
            "category": "model",
            "type": "select",
            "currentValue": current_value or (
                options[0]["value"] if options else ""),
            "options": options,
        }]

    def set_config_option(self, params: dict) -> dict:
        config_id = params.get("configId")
        if config_id != "model":
            raise acps.TransportError(
                f"unknown config option {config_id!r}")
        value = params.get("value")
        leaf = self._option_leaves.get(value)
        if leaf is None:
            raise acps.TransportError(f"unknown model value {value!r}")
        if leaf == "loki-explicit":
            loki.apply_runtime_config(loki.build_config_from_env(
                credentials=loki.CREDENTIALS))
        else:
            provider_id, provider_entry, model_entry = leaf
            loki.apply_runtime_config(
                loki.config_from_modelsdev_selection(
                    provider_id, provider_entry, model_entry,
                    loki.CREDENTIALS))
        descriptor = loki.active_connection_descriptor()
        if descriptor is not None:
            loki.set_session_connection(descriptor)
        loki.save_chat_log()
        return {"configOptions": self.config_options()}

    def open(self, params: dict) -> dict:
        """Prepare the conversation: fresh log, or resume a saved one."""
        resume = params.get("resume")
        if resume:
            path = os.path.join(
                loki.CHAT_LOG_DIR, os.path.basename(str(resume)))
            if os.path.isfile(path):
                loki.load_chat_log(path, quiet=True)
                if params.get("replay"):
                    self._replay_transcript()
            else:
                raise acps.TransportError(
                    f"no saved session named {resume!r}")
        else:
            loki.new_chat_log(os.path.join(
                loki.CHAT_LOG_DIR,
                f"chat-{params.get('sessionId', 'acp')}.json"))
        self._build_model_options()
        return {"configOptions": self.config_options()}

    def _build_model_options(self) -> None:
        """Flatten the catalog into model config options, once."""
        options = modelsdev.flattened_config_options(
            loki.CREDENTIALS,
            explicit_connection=loki.explicit_connection_option(
                loki.CREDENTIALS)
            if hasattr(loki, "explicit_connection_option") else None)
        self._model_options = options
        self._option_leaves = {}
        # Rebuild the leaf map from the same catalog walk so
        # set_config_option can apply a value exactly.
        try:
            _, groups = modelsdev.ensure_index()
        except (OSError, ValueError):
            return
        groups = modelsdev._add_explicit_connection(
            modelsdev.filter_supported_groups(groups, loki.CREDENTIALS),
            loki.explicit_connection_option(loki.CREDENTIALS)
            if hasattr(loki, "explicit_connection_option") else None)
        for row in modelsdev._model_rows(groups):
            for member in row[0]:
                if isinstance(member,
                              modelsdev.ExplicitConnectionOption):
                    self._option_leaves["loki-explicit"] = "loki-explicit"
                    continue
                provider_id, provider_entry, model_entry = member
                model_id = model_entry.get("id") or model_entry.get("name")
                self._option_leaves[f"{provider_id}/{model_id}"] = member

    def _replay_transcript(self) -> None:
        """Emit the loaded history as session/update notifications."""
        blocks = replays.classify_transcript(
            self.session.transcript_items)
        for kind, text, key in blocks:
            if kind == "user":
                self.write(acps.notification("session/update", {
                    "sessionId": self.session_id,
                    "update": {
                        "sessionUpdate": "user_message_chunk",
                        "content": {"type": "text", "text": text},
                    },
                }))
            elif kind == "tool":
                title, call_id = text, key
                self.write(acps.notification("session/update", {
                    "sessionId": self.session_id,
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": str(call_id or "replay"),
                        "title": title.split("\n")[0][:120],
                        "kind": "other",
                        "status": "completed",
                    },
                }))
            else:
                self.write(acps.notification("session/update", {
                    "sessionId": self.session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": text},
                    },
                }))

    # -- session/prompt ----------------------------------------------------

    async def prompt(self, params: dict) -> dict:
        texts = []
        for block in params.get("prompt", []):
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        user_text = "\n".join(text for text in texts if text)
        if not user_text:
            return {"error": "prompt contains no text"}

        self.cancel_event.clear()
        transcript = self.session.transcript_items
        transcript.append(formats.message_item("user", user_text))
        self.session.chat_log_dirty = True
        events = []
        mapper_state: dict = {}
        self.session_id = params.get("sessionId") or self.session_id

        def on_event(event):
            events.append(event)
            for update in acp_events.map_event(
                    self.session_id, event, mapper_state):
                self.write(acps.notification("session/update", update))

        await self._run_turn(on_event)
        return {"stopReason": self._stop_reason(events)}

    async def _run_turn(self, on_event):
        session = self.session
        if not loki.current_model():
            on_event({"type": "assistant_message",
                      "content": "No model selected; configure LOKI_* "
                                 "environment or pick a model."})
            return
        cancel_check = self.cancel_event.is_set

        async def chat_fn(items, on_text_delta):
            return await loki.async_chat_completion(
                items, loki.TOOLS, True, False,
                on_text_delta=on_text_delta,
                cancel_check=cancel_check)

        await loki.run_tool_loop_async(
            session.transcript_items,
            chat_fn=chat_fn,
            on_event=on_event,
            cancel_check=cancel_check,
            cancel_event=self.cancel_event,
            stream_chat=True,
            on_response=lambda turn, event: loki.mark_chat_log_dirty(),
        )
        loki.save_chat_log()

    @staticmethod
    def _stop_reason(events: list) -> str:
        kinds = {event.get("type") for event in events}
        if "response_cancelled" in kinds:
            return "cancelled"
        return "end_turn"
