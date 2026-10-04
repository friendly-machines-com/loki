"""Terminal front-end: the TUI loki.py used to contain.

Owns the ANSI renderer, the status bar, the pickers, the input session,
and process lifecycle (signal handling, overlay setup/teardown).  The
core (loki.py) keeps the tool loop, tools, provider config, and chat
log machinery -- importing it no longer touches the tty.
"""

from __future__ import annotations

import asyncio
import contextlib
import getopt
import json
import logging
import signal
import sys
from dataclasses import dataclass

from . import attachments
from .diagnostics import debug_json
from . import command_deliveries
from . import formats
from .attachments import ImageAttachmentError, load_image_attachment
from . import authentications
from . import credential_capabilities
from . import credential_runtimes
from . import models as modelsdev
from . import protocols
from . import process_outputs
from . import settings
from . import savefiles
from . import subagents
from . import terminals
from . import texts
from . import tool_runtime
from .connections import (
    ConnectionDescriptor,
    ConnectionDescriptorError,
    connection_display_fields,
)
from .loki import RuntimeConfig, computer
from . import loki as _core
from .loki import (
    ERROR_COLOR,
    EXPLORE_TOOLS,
    PLAN_TOOLS,
    TOOLS,
    TOOL_CALL_COLOR,
    _remember_session_toolset,
    _status_api_base,
    active_connection_descriptor,
    apply_runtime_config,
    async_chat_completion,
    build_config_from_env,
    change_shell_cwd_from_text,
    config_from_connection_descriptor,
    config_from_modelsdev_selection,
    connection_descriptor_from_config,
    configure_tool_hook_pipeline,
    connection_from_session_state,
    current_agent_mode,
    current_chat_log_path,
    current_config,
    current_cwd,
    current_model,
    current_session,
    current_transcript,
    cycle_agent_mode,
    display_path,
    explicit_api_base_configured,
    explicit_connection_option,
    load_chat_log,
    load_models_async,
    mark_chat_log_dirty,
    new_chat_log,
    new_chat_log_path,
    print_shell_cwd,
    record_agent_mode_instruction,
    reinstall_provider,
    resolve_chat_log_path,
    run_bash_async,
    run_tool_loop_async,
    save_chat_log,
    set_session_connection,
    user_prompt_history,
)
from .terminals import (
    input_session, restore_output_area_after_input, terminal, terminal_output_mode)


logger = logging.getLogger(__name__)


def _redraw_status():
    try:
        terminals.redraw_status_bar()
    except (AssertionError, OSError):
        # A display failure must not interrupt input, jobs, or the turn loop.
        pass


@dataclass(frozen=True)
class _QueuedImage:
    id: int
    image: attachments.StagedImage


class _QueuedInputs:
    """What /queue reports on and edits: the input FIFO and staged images.

    Both lifetimes belong to the running frontend; async_main registers its
    input session here so the immediate /queue handler can snapshot and edit
    them without consuming from the queue or touching the turn loop.
    """

    def __init__(self):
        self.session = None
        self.staged_images = []
        self._next_image_id = 1

    def reset(self, session):
        """Bind SESSION's queue view and return the fresh staged-images list."""
        self.session = session
        self.staged_images.clear()
        return self.staged_images

    def stage_image(self, image):
        self.staged_images.append(_QueuedImage(self._next_image_id, image))
        self._next_image_id += 1

    def pending_entries(self) -> list:
        if self.session is None:
            return []
        return self.session.user_messages.pending_entries()


_queued_inputs = _QueuedInputs()


def _echo_immediate(text: str):
    # Synchronous, at admission: the submitted line must appear before
    # anything the command produces, exactly like a queued prompt's echo.
    restore_output_area_after_input()
    print()
    terminal.set_background_color(terminals.INPUT_COLOR)
    print("User: ", end="")
    terminal.write_text(text, multiline=True)
    terminal.reset_colors_and_flags()
    print()


def _immediate_ps(argument: str):
    # Sync by contract: JobManager transitions contain no awaits, so they
    # cannot interleave with a running turn.
    try:
        result = _core.run_ps(argument)
    except Exception as error:
        result = f"Could not inspect or control jobs: {error}"
    # Spool contents and command text are untrusted; do not emit their ANSI.
    terminal.write_text(result, multiline=True)
    print()
    sys.stdout.flush()


def _immediate_status(argument: str):
    # The connection context is captured at admission, not when the
    # scheduled task first runs: config, session, and the response-header
    # store are all bound here, synchronously. The handler RETURNS its
    # rendering instead of printing: _run_immediate emits it as one block,
    # never spliced into streaming assistant text.
    config = current_config()
    session = current_session()
    store = session.response_headers

    async def report():
        from . import response_headers
        try:
            if argument == "save":
                await store.save()
                return "Response status saved (this runtime only)."
            tokens = argument.split()
            show_all = "all" in tokens
            as_json = "--json" in tokens
            connected = (config is not None
                         and config.chat_provider.kind != protocols.DUMMY)
            if show_all:
                document = store.snapshot()
            elif connected:
                credential = (config.auth_spec.credential
                              if config.auth_spec else None)
                document = store.snapshot(
                    config.chat_provider.chat_url,
                    credential=credential.encode() if credential else None)
            else:
                document = {"version": 1, "endpoints": []}
            if as_json:
                text = json.dumps(document, indent=2, ensure_ascii=True)
            else:
                scope = ("All known connections" if show_all
                         else "Current endpoint and credential")
                text = (
                    f"{scope} (last observed, not live balances).\n"
                    "Saved observations plus this runtime's memory; other "
                    "runtimes' unsaved observations are not visible.\n")
                if not show_all and not connected:
                    text += ("No active HTTP chat connection. "
                             "Use /status all.")
                elif not show_all and not document["endpoints"]:
                    text += "No observations for the current connection."
                else:
                    text += response_headers.render(document)
            # /status stays offline; this only points at live provider data
            # when the connection supports it.
            from . import provider_controls
            hint = provider_controls.live_hint(
                provider_controls.ControlContext(config=config))
            if hint is not None:
                text += "\n" + hint
            return text
        except (OSError, ValueError, OverflowError) as error:
            return f"Could not read response status: {error}"

    return report()


def _immediate_account_read(argument: str):
    # The connection context is captured at admission, synchronously here --
    # not when the scheduled task first runs: a config change submitted
    # after this line must not redirect a read the user already sent.
    from . import provider_controls
    tokens = [token for token in argument.split() if token != "--json"]
    as_json = "--json" in argument.split()
    context = provider_controls.ControlContext(
        config=current_config(),
        credential_authority=current_session().credential_authority)

    async def read():
        result = await _read_account_control(context, tokens[0])
        text = _render_control_result(result, as_json)
        # List available actions without starting a choice or confirmation
        # in the monitor plane; explicit action commands remain queued.
        hint = _account_actions_hint(tokens[0], result.actions)
        return text + "\n" + hint if hint else text

    return read()


def _queue_usage():
    print("usage: /queue [texts | images] "
          "[delete ID | move ID before OTHER_ID | move ID end | edit ID TEXT]")


def _print_queued_texts():
    entries = _queued_inputs.pending_entries()
    if not entries:
        print("No queued texts.")
        return
    print("Queued texts (send order):")
    # Queued text is untrusted; write_text never emits its ANSI.
    for entry in entries:
        _print_text_line(f"[id {entry.id}] ", entry.text)
    print("/queue texts delete ID - remove; "
          "/queue texts move ID before OTHER_ID | move ID end - reorder; "
          "/queue texts edit ID TEXT - replace")


def _print_staged_images():
    entries = list(_queued_inputs.staged_images)
    if not entries:
        print("No staged images.")
        return
    print("Staged images (sent with the next prompt):")
    for entry in entries:
        image = entry.image
        _print_text_line(
            f"[id {entry.id}] ",
            f"{display_path(image.path)} "
            f"({image.media_type}, {image.byte_size} bytes)")
    print("/queue images delete ID - remove; "
          "/queue images move ID before OTHER_ID | move ID end - reorder")


def _queue_entry(word, entries, label):
    if not word.isdecimal():
        _queue_usage()
        return None
    try:
        entry_id = int(word)
    except ValueError:
        # Oversized decimal input must not kill the immediate input owner.
        _queue_usage()
        return None
    for entry in entries:
        if entry.id == entry_id:
            return entry
    print(f"{label} ID {word} is no longer pending.")
    return None


def _immediate_queue(argument: str):
    # Selectors name submissions, never live positions. Resolution and the
    # entire mutation are synchronous, so a resolved entry cannot disappear
    # between validation and mutation on the event-loop thread.
    words = argument.split()
    if not words:
        print(f"Queued texts: {len(_queued_inputs.pending_entries())}; "
              f"staged images: {len(_queued_inputs.staged_images)}.")
        print("Subcommands: /queue texts, /queue images.")
        return
    if words == ["texts"]:
        _print_queued_texts()
        return
    if words == ["images"]:
        _print_staged_images()
        return
    if words[:2] == ["images", "edit"]:
        print("A staged image cannot be edited; delete it and stage "
              "another with /image PATH.")
        return
    if len(words) < 3 or words[0] not in ["texts", "images"]:
        _queue_usage()
        return
    operation = words[1]
    replacement = None
    before = None
    if operation == "delete" and len(words) == 3:
        pass
    elif operation == "move" and (
            (len(words) == 4 and words[3] == "end")
            or (len(words) == 5 and words[3] == "before")):
        pass
    elif operation == "edit" and words[0] == "texts":
        parts = argument.split(None, 3)
        if len(parts) < 4 or not parts[3].strip():
            _queue_usage()
            return
        replacement = parts[3]
    else:
        _queue_usage()
        return
    is_text = words[0] == "texts"
    entries = (_queued_inputs.pending_entries() if is_text
               else list(_queued_inputs.staged_images))
    label = "Queued text" if is_text else "Staged image"
    entry = _queue_entry(words[2], entries, label)
    if entry is None:
        return
    if operation == "move" and words[3] == "before":
        before = _queue_entry(words[4], entries, label)
        if before is None:
            return
    if is_text:
        queue = _queued_inputs.session.user_messages
        if operation == "delete":
            queue.delete_text(entry.id)
        elif operation == "edit":
            queue.edit_text(entry.id, replacement)
        else:
            queue.move_text(entry.id, before.id if before else None)
        _print_queued_texts()
    else:
        images = _queued_inputs.staged_images
        if operation == "delete":
            images.remove(entry)
        elif entry != before:
            images.remove(entry)
            target = images.index(before) if before else len(images)
            images.insert(target, entry)
        _terminal_activity.set_queued_images(len(images))
        _print_staged_images()


def _emit_immediate_output(command, text, *, file=None):
    """Display a monitor answer now, isolated from either adjacent delta.

    No awaits or turn-running check: even a delayed read answers during a
    running turn. The leading newline separates its result from the live
    streaming cursor; the label identifies which command completed. Do not
    feed or finish assistant_markdown here -- it belongs to the assistant
    stream, including Markdown spans split across chunks.
    """
    if not text:
        return
    file = sys.stdout if file is None else file
    restore_output_area_after_input()
    print(file=file)
    terminal.write_text(command.strip(), file=file)
    print(":", file=file)
    terminal.write_text(text, multiline=True, file=file)
    print(file=file)
    file.flush()


async def _run_immediate(command, outcome):
    # Async handlers return their rendering. Both success and failure are
    # displayed immediately as separate monitor blocks, never conversation.
    try:
        text = await outcome
    except Exception as error:  # noqa: BLE001 - reported, not fatal
        _emit_immediate_output(
            command, f"Command failed: {error}", file=sys.stderr)
        return
    _emit_immediate_output(command, text)


_immediate_tasks: set = set()


async def _cancel_immediate_tasks(session):
    # Close admission before the gather yields: the still-owned input reader
    # must not create another immediate task while shutdown awaits these.
    session.on_submit = lambda text: False
    tasks = list(_immediate_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


_IMMEDIATE_HANDLERS = {
    # Handlers receive the classified argument. Sync handlers run inline in
    # the input owner (atomic against other event-loop work, like /ps) and
    # print at admission. Async handlers capture context synchronously, then
    # RETURN their rendering for immediate, labelled display at completion.
    # They never own the reader, await a turn, or touch transcript state --
    # monitor-plane only, per the contract in command_deliveries.
    "ps": _immediate_ps,
    "status": _immediate_status,
    "account": _immediate_account_read,
    "queue": _immediate_queue,
}


def _dispatch_immediate(parsed: command_deliveries.ParsedCommand):
    """Run one immediate command's handler; return its outcome.

    A None outcome means a sync handler already ran inline. Any other value
    is a coroutine the caller either awaits (queue-fed dispatch) or schedules
    (input-owner dispatch).
    """
    return _IMMEDIATE_HANDLERS[parsed.name](parsed.argument)


def _submit_immediate(text: str) -> bool:
    # Immediate commands are hypervisor monitor-plane interaction, not queued
    # prompts: they run from the sole input owner even while inference, a
    # tool, or a modal is busy, and their output never becomes conversation.
    parsed = command_deliveries.terminal_immediate(text)
    if parsed is None:
        return False
    _echo_immediate(text)
    outcome = _dispatch_immediate(parsed)
    if outcome is not None:
        task = asyncio.create_task(_run_immediate(text, outcome))
        _immediate_tasks.add(task)

        def finished(done):
            _immediate_tasks.discard(done)
            # Cancellation before the wrapper's first tick never enters its
            # body: close the already-created handler coroutine in that case.
            outcome.close()
            if not done.cancelled():
                error = done.exception()
                if error is not None:
                    # A broken display still must not leave an unobserved
                    # task exception or affect the active turn.
                    logger.error("Could not display %s output: %s", text, error)

        task.add_done_callback(finished)
    return True


def _image_command_path(command_text: str) -> str:
    return attachments.image_argument_path(command_text[len("/image"):])


@dataclass
class TerminalActivityStatus:
    turn_running: bool = False
    queued_prompts: int = 0
    queued_images: int = 0

    def _set(self, field_name: str, value):
        if getattr(self, field_name) == value:
            return
        setattr(self, field_name, value)
        _redraw_status()

    def set_turn_running(self, running: bool):
        self._set("turn_running", bool(running))

    def set_queued_prompts(self, count: int):
        self._set("queued_prompts", max(0, int(count)))

    def set_queued_images(self, count: int):
        self._set("queued_images", max(0, int(count)))

    def reset(self):
        self.turn_running = False
        self.queued_prompts = 0
        self.queued_images = 0


_terminal_activity = TerminalActivityStatus()


def _print_tool_args(args):
    terminal.write_text(texts.format_tool_args(args), multiline=True)
    print()


def _print_text_line(prefix, text, *, file=None, multiline=False):
    print(prefix, end="", file=file)
    terminal.write_text(str(text), multiline=multiline, file=file)
    print(file=file)


def _print_repr_line(prefix, value, *, file=None):
    _print_text_line(prefix, repr(value), file=file)


def _report_model_list_errors(text):
    print("Model list failed:", file=sys.stderr)
    terminal.write_text(text, multiline=True, file=sys.stderr)
    print(file=sys.stderr)
    sys.stderr.flush()


def _report_hook_stderr(command, text):
    sys.stdout.flush()
    print(f"Hook {command[0]!r} stderr:", file=sys.stderr)
    terminal.write_text(text, multiline=True, file=sys.stderr)
    print(file=sys.stderr)
    sys.stderr.flush()


class _ResumeTranscriptPresenter:
    """Write a saved transcript without mixing untrusted text with ANSI."""

    def __init__(self, assistant_label, *, ui_settings=None):
        preferences = ui_settings or settings.Settings()
        self.renderer = savefiles.ResumeTranscriptRenderer(
            assistant_label=assistant_label,
            show_reasoning=current_session().reasoning_traces == "on",
            show_bash_stdout=preferences.terminal.show_bash_stdout)

    def write(self, events):
        blocks = self.renderer.presentation(events)
        for block_index, (block_kind, block) in enumerate(blocks):
            if block_index:
                print("\n\n", end="")
            styled = False
            if block_kind == "tool_call":
                terminal.set_foreground_color(TOOL_CALL_COLOR)
                styled = True
            elif block_kind == "tool_error":
                terminal.set_background_color(ERROR_COLOR)
                styled = True
            for kind, text in block:
                if kind == "literal":
                    print(text, end="")
                elif kind == "atom":
                    terminal.write_text(text)
                elif kind == "program_atom":
                    terminal.write_text(repr(text))
                elif kind in ["text", "user_text"]:
                    terminal.write_text(text, multiline=True)
                elif kind == "assistant_markdown":
                    terminal.write_markdown(text)
                else:
                    raise AssertionError(
                        f"unknown transcript presentation kind {kind!r}")
            if styled:
                terminal.reset_colors_and_flags()
        if blocks:
            print()
        print("----")


def _terminal_agent_event(event: dict, *, ui_settings=None):
    # Error branches reset attributes before emitting their final newline. That
    # prevents terminal scroll-fill from inheriting the red background.
    kind = event.get("type")
    if kind == "max_loops":
        print("\n[!] [Max Loop Limit Reached - Stopping Autonomous Execution]")
    elif kind == "api_error":
        terminal.set_background_color(ERROR_COLOR)
        terminal.write_text(
            event["error"].formatted(), multiline=True)
        terminal.reset_colors_and_flags()
        print()
    elif kind == "network_error":
        # Same bracketing as the other error branches: the leading blank line
        # stays uncolored, then the label and body carry the error color, and
        # the reset precedes the final newline so scroll-fill stays neutral.
        print()
        terminal.set_background_color(ERROR_COLOR)
        print(f"{computer}: NETWORK ERROR: ", end="")
        terminal.write_text(event["error"], multiline=True)
        terminal.reset_colors_and_flags()
        print()
    elif kind == "transcript_error":
        terminal.set_background_color(ERROR_COLOR)
        error = event["error"]
        print("Transcript render error: ", end="")
        terminal.write_text(str(error), multiline=True)
        terminal.reset_colors_and_flags()
        print()
        sys.stdout.flush()
        payload = getattr(error, "payload", None)
        if payload is not None:
            debug_json(logger, "Provider payload:", payload)
    elif kind == "provider_error":
        terminal.set_background_color(ERROR_COLOR)
        error = event["error"]
        print("Provider protocol error: ", end="")
        terminal.write_text(str(error), multiline=True)
        terminal.reset_colors_and_flags()
        print()
        sys.stdout.flush()
    elif kind == "reasoning_start":
        print()
        terminal.set_foreground_color(8)
        print("Reasoning summary:" if event.get("kind") == "summary" else "Thinking:")
        terminal.reset_colors_and_flags()
        sys.stdout.flush()
    elif kind == "reasoning_delta":
        terminal.set_foreground_color(8)
        terminal.write_text(event["text"], multiline=True)
        terminal.reset_colors_and_flags()
        sys.stdout.flush()
    elif kind == "reasoning_end":
        print()
        if not event.get("complete", True):
            print("[thinking output incomplete; partial transport output not saved]")
        sys.stdout.flush()
    elif kind == "assistant_message":
        print()
        terminal.write_text(current_model())
        print(": ", end="")
        terminal.write_markdown(event["content"])
        print()
    elif kind == "assistant_start":
        terminal.assistant_markdown.start()
        print()
        terminal.write_text(current_model())
        print(": ", end="", flush=True)
    elif kind == "assistant_delta":
        terminal.assistant_markdown.feed(event["content"])
    elif kind == "assistant_end":
        terminal.assistant_markdown.finish()
        print()
        sys.stdout.flush()
    elif kind == "response_timing":
        sys.stdout.flush()
        logger.debug("LLM Response Time: %.3fs", event["elapsed"])
    elif kind == "provider_notice":
        text = formats.provider_notice_text(event.get("code"))
        if text is not None:
            print("\n\u24d8 ", end="")
            terminal.write_text(text, multiline=True)
            print()
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
    elif kind in ("response_incomplete", "response_failed"):
        sys.stdout.flush()
        status = "incomplete" if kind == "response_incomplete" else "failed"
        print(f"[model response {status}; provider output saved]",
              file=sys.stderr)
        detail = event.get("protocol_data")
        if isinstance(detail, dict):
            response = detail.get(formats.OPENAI_RESPONSES)
            if isinstance(response, dict):
                reason = (response.get("incomplete_details")
                          or response.get("error"))
                if isinstance(reason, dict):
                    text = reason.get("reason") or reason.get("message")
                    if isinstance(text, str) and text:
                        _print_text_line(
                            "Reason: ", text, file=sys.stderr, multiline=True)
        sys.stderr.flush()
        if detail:
            debug_json(logger, "Provider response details:", detail)
    elif kind == "stream_error":
        error = event["error"]
        terminal.set_background_color(ERROR_COLOR)
        print("Streaming response error: ", end="")
        terminal.write_text(str(error), multiline=True)
        print(
            "\nSet LOKI_STREAM=0 to disable streaming for this connection.",
            end="")
        terminal.reset_colors_and_flags()
        print()
        sys.stdout.flush()
        payload = getattr(error, "payload", None)
        if payload is not None:
            debug_json(logger, "Provider payload:", payload)
    elif kind == "tool_input_repaired":
        terminal.set_foreground_color(TOOL_CALL_COLOR)
        _print_repr_line(
            f"{computer}: Repaired Tool Input: ", event["name"])
        for repair in event["repairs"]:
            print("  ", end="")
            terminal.write_text(repr(repair["display_path"]))
            print(": ", end="")
            terminal.write_text(repair["rule"].replace("_", " "))
            print()
        terminal.reset_colors_and_flags()
    elif kind == "tool_call":
        terminal.set_foreground_color(TOOL_CALL_COLOR)
        # Blank line before the block, matching "User:" and "<model>:".
        print()
        print(f"{computer}: Executing Tool: ", end="")
        terminal.write_text(repr(event["name"]))
        print(" with args:")
        _print_tool_args(event["args"])
        if event.get("cwd") is not None:
            _print_repr_line("  Shell CWD: ", event["cwd"])
        terminal.reset_colors_and_flags()
    elif kind == "tool_rejected":
        terminal.set_foreground_color(TOOL_CALL_COLOR)
        print(f"{computer}: Rejected Tool: ", end="")
        terminal.write_text(repr(event["name"]))
        print(" with invalid args:")
        _print_tool_args(event["args"])
        if event.get("cwd") is not None:
            _print_repr_line("  Shell CWD: ", event["cwd"])
        terminal.reset_colors_and_flags()
    elif kind == "tool_result":
        # The canonical result event.  ``tool_error`` carries only the failure
        # content and no name, and ACP drops it for the same reason: printing
        # it as well would duplicate the error now that a result is shown.  It
        # is still emitted, so a mode that hides tool calls can show failures.
        label = "Tool error" if event.get("is_error") else "Tool result"
        if event.get("is_error"):
            terminal.set_background_color(ERROR_COLOR)
        print(f"{computer}: {label}: ", end="")
        terminal.write_text(repr(event["name"]))
        print()
        preferences = ui_settings or settings.Settings()
        text = process_outputs.presentation_text(
            event["content"], event.get("process_output"),
            show_bash_stdout=preferences.terminal.show_bash_stdout)
        terminal.write_text(text, multiline=True)
        if event.get("is_error"):
            # Reset precedes the newline so terminal scroll-fill stays neutral.
            terminal.reset_colors_and_flags()
        print()


async def run_terminal_turn_async(transcript_items: list, cancel_check=None,
                                  cancel_event: asyncio.Event | None = None,
                                  turn_events=None, ui_settings=None) -> str:
    thinking = _core.capture_turn_settings()
    reasoning_effort = thinking.effort
    read_only = current_agent_mode() in ("explore", "plan")
    mode_tools = (
        PLAN_TOOLS if current_agent_mode() == "plan" else EXPLORE_TOOLS)
    active_tools = (
        [
            tool for tool in TOOLS
            if tool.get("function", {}).get("name") in mode_tools
        ]
        if read_only else TOOLS
    )

    async def chat_fn(
            items, on_text_delta, *, codex_turn_state, on_reasoning_delta=None):
        kwargs = {
            "on_text_delta": on_text_delta,
            "cancel_check": cancel_check,
            "codex_turn_state": codex_turn_state,
            "reasoning_effort": reasoning_effort,
            "thinking": thinking,
        }
        if on_reasoning_delta is not None:
            kwargs["on_reasoning_delta"] = on_reasoning_delta
        return await async_chat_completion(
            items, active_tools, True, False, **kwargs)

    def on_response(turn, event):
        _remember_session_toolset(active_tools)
        current_session().context_snapshot(live=True)
        _redraw_status()

    def on_event(event):
        if turn_events is not None:
            turn_events.append(event)
        if event.get("type") in ("tool_result", "response_cancelled", "max_loops"):
            _redraw_status()
        _terminal_agent_event(event, ui_settings=ui_settings)

    # The user/mode input has already been appended: the old report is stale
    # during this request, even before the first response or tool result.
    _redraw_status()
    return await run_tool_loop_async(
        transcript_items,
        allowed=mode_tools if read_only else None,
        chat_fn=chat_fn,
        on_event=on_event,
        cancel_check=cancel_check,
        stream_chat=True,
        report_timing=True,
        cancel_event=cancel_event,
        reasoning_effort=reasoning_effort,
        thinking=thinking,
        on_response=on_response,
    )


def _status_fields(activity):
    activity = activity or _terminal_activity
    displayed_model = current_model()
    if not displayed_model:
        displayed_model = "none"
    elif (current_config() is not None
            and current_config().model_status == "deprecated"):
        displayed_model += " (deprecated)"
    return {
        "api": _status_api_base(),
        "model": displayed_model,
        "effort": _core.reasoning_effort_status_text(),
        "context": current_session().context_snapshot().text,
        "turn": "running" if activity.turn_running else "idle",
        "queued_prompts": activity.queued_prompts,
        "queued_images": activity.queued_images,
        "mode": current_agent_mode(),
        "cwd": display_path(current_cwd()),
    }


def status_text(activity: TerminalActivityStatus | None = None) -> str:
    fields = _status_fields(activity)
    remote = 'Remote: API: {}, Model: {}'.format(
        fields["api"], fields["model"])
    if fields["effort"] is not None:
        remote += ', Effort: {}'.format(fields["effort"])
    remote += ', Context: {}; /model'.format(fields["context"])
    remote += ', /thinking, /trace thinking, /status, /account'
    return (
        remote + '\n'
        'Local: CWD: {}, turn: {}, mode: {}; '
        '/queue(texts: {}, images: {}), '
        '/pwd, /cd DIR, /ps, /image PATH, !foo, /quit'
    ).format(
        fields["cwd"],
        fields["turn"],
        fields["mode"],
        fields["queued_prompts"],
        fields["queued_images"])


def _write_status_text():
    fields = _status_fields(None)
    print("Remote: API: ", end="")
    terminal.write_text(fields["api"])
    print(", Model: ", end="")
    terminal.write_text(fields["model"])
    if fields["effort"] is not None:
        print(", Effort: ", end="")
        terminal.write_text(fields["effort"])
    print(", Context: ", end="")
    terminal.write_text(fields["context"])
    print("; /model", end="")
    print(", /thinking, /trace thinking, /status, /account\nLocal: CWD: ", end="")
    terminal.write_text(fields["cwd"])
    print(", turn: ", end="")
    for label, value, active in [
            ["", fields["turn"], fields["turn"] != "idle"],
            [", mode: ", fields["mode"], False],
            ["; /queue(texts: ", fields["queued_prompts"], fields["queued_prompts"] != 0],
            [", images: ", fields["queued_images"], fields["queued_images"] != 0]]:
        print(label, end="")
        if active:
            print(terminals.BOLD, end="")
        terminal.write_text(str(value))
        if active:
            # End only bold, preserving the status area's background color.
            print(terminals.BOLD_OFF, end="")
    manager = current_session().job_manager
    # The reaper can lag a process exit. Read returncode without refreshing
    # job metadata from the renderer.
    has_running_jobs = manager is not None and any(
        job.status == "running" and job.process.returncode is None
        for job in manager.jobs.values())
    print("), /pwd, /cd DIR, ", end="")
    # /ps is available immediately, including during a turn; this label is
    # only a job-state snapshot, never an instruction to queue the command.
    if has_running_jobs:
        print(terminals.BOLD, end="")
    print("/ps", end="")
    if has_running_jobs:
        print(terminals.BOLD_OFF, end="")
    print(", /image PATH, !foo, /quit", end="")


terminals.set_status_text_provider(_write_status_text)


def _reasoning_effort_rows():
    profile = _core.current_reasoning_effort_profile()
    if profile is None:
        return []
    return [(None, _core.reasoning_effort_default_text() or "Model default")] + [
        (value, value) for value in profile.values if isinstance(value, str)]


async def run_thinking_picker_async(session):
    config = current_config()
    if config is None or not config.model:
        return _core.thinking_status_text()
    provider = config.chat_provider
    fields = []
    effort = _reasoning_effort_rows()
    if effort:
        fields.append(("effort", "Effort", effort))
    modes = provider.thinking_modes(config.model)
    if modes:
        fields.append(("mode", "Thinking mode", [(None, "Model default")] + [(mode, mode) for mode in modes]))
    if "manual" in modes:
        fields.append(("budget", "Thinking-token allowance", None))
    if provider.reasoning_preservation(config.model) is not None:
        fields.append(("retention", "Reuse earlier thinking", [("default", "Default"), ("preserve", "Preserve")]))
    if not fields:
        return _core.thinking_status_text()
    async with session.modal() as modal:
        terminal.write_text(_core.thinking_status_text(), multiline=True)
        print()
        for index, (_name, label, _rows) in enumerate(fields, 1):
            print(f"{index}. {label}")
        choice = await modal.prompt("Control (number selects, empty cancels): ")
        if not choice:
            return "Thinking selection cancelled."
        try:
            index = int(choice)
            if not 1 <= index <= len(fields):
                raise ValueError()
            name, label, rows = fields[index - 1]
        except ValueError:
            return "Invalid thinking control."
        if rows is None:
            choice = await modal.prompt("Thinking tokens (integer or default, empty cancels): ")
            if not choice:
                return "Thinking selection cancelled."
            try:
                value = None if choice == "default" else int(choice)
            except ValueError:
                return "Thinking tokens must be an integer."
        else:
            for index, (_value, text) in enumerate(rows, 1):
                terminal.write_text(f"{index}. {text}", multiline=True)
                print()
            choice = await modal.prompt("Choice (number selects, empty cancels): ")
            if not choice:
                return "Thinking selection cancelled."
            try:
                index = int(choice)
                if not 1 <= index <= len(rows):
                    raise ValueError()
                value = rows[index - 1][0]
            except ValueError:
                return "Invalid thinking choice."
        changes = {name: value}
        if name == "mode" and value == "manual" and current_session().thinking_budget is None:
            choice = await modal.prompt("Manual thinking tokens (empty cancels): ")
            if not choice:
                return "Thinking selection cancelled."
            try:
                changes["budget"] = int(choice)
            except ValueError:
                return "Thinking tokens must be an integer."
        _core.set_thinking_controls(changes)
        return _core.thinking_status_text()


def _report_unavailable_reasoning_preference():
    preference = _core.current_reasoning_effort_preference()
    profile = _core.current_reasoning_effort_profile()
    if preference is None or (
            profile is not None and profile.supports(preference)):
        return
    print("Reasoning effort ", end="", file=sys.stderr)
    terminal.write_text(
        repr(preference), file=sys.stderr)
    print(
        " is unavailable for the selected model; using the model default.",
        file=sys.stderr,
    )


async def run_session_picker_async(session):
    async with session.modal() as modal:
        picked = await savefiles.run_session_picker_async(
            input_fn=modal.prompt,
            chat_log_dir=_core.CHAT_LOG_DIR,
            text_writer=terminal.write_text)
        # Finish the picker's output cleanup while the modal still owns the
        # terminal. Only then may the normal input producer resume.
        terminal.goto_position(1, 1)
        terminal.clear_to_end_of_screen()
        terminal.flush()
    return picked


async def confirm_saved_connection_async(
        descriptor: ConnectionDescriptor, session,
        config: RuntimeConfig | None = None,
        working_directory: str | None = None) -> bool:
    displayed = (
        connection_descriptor_from_config(config)
        if config is not None else descriptor)
    if displayed is None:
        raise ValueError("a dummy provider cannot be resumed")

    fields = connection_display_fields(displayed)
    if working_directory:
        # The directory is where every tool reads and writes, so it is part
        # of what the resume approval covers, not just the connection.
        fields = [*fields, ("Working directory", working_directory)]
    async with session.modal() as modal:
        print()
        print("Saved connection:")
        for label, value in fields:
            if label in {
                    "Authentication", "Streaming",
                    "Anthropic prompt cache"}:
                print(f"  {label}: {value}")
            else:
                _print_repr_line(f"  {label}: ", value)
        answer = (await modal.prompt(
            "Use this saved connection? [y/N]: ") or "")
        return answer.strip().lower() in ("y", "yes")


async def _numbered_choice_async(modal, header, rows, prompt):
    """Minimal numbered choice, matching the reasoning-effort picker.

    Rows are ``(value, label)``.  A bare number selects; empty cancels;
    anything else re-renders.
    """
    print()
    print(header)
    for index, (_value, label) in enumerate(rows, start=1):
        terminal.write_text(f"{index}. {label}", multiline=True)
        print()
    while True:
        choice = (await modal.prompt(prompt) or "").strip()
        if not choice:
            return None
        try:
            index = int(choice)
        except ValueError:
            continue
        if 1 <= index <= len(rows):
            return rows[index - 1][0]


def _render_control_result(result, as_json):
    """The text form of one control result (no trailing newline)."""
    if as_json and result.document is not None:
        return json.dumps(result.document, indent=2, ensure_ascii=True)
    return "\n".join(result.lines)


def _write_control_result(result, as_json):
    terminal.write_text(_render_control_result(result, as_json),
                        multiline=True)
    print()


def _account_actions_hint(control_id, actions):
    """The actions hint text (no trailing newline); empty when none."""
    if not actions:
        return ""
    lines = [f"Actions (run /account {control_id} ACTION to perform one; "
             "confirmation will be requested):"]
    lines.extend(f"  {action.id} - {action.title}" for action in actions)
    return "\n".join(lines)


async def _read_account_control(context, chosen):
    """Read data only; the caller owns rendering, including read errors.

    Returning the existing ControlResult also for an unknown control or a
    failed read keeps delayed errors out of the live streaming cursor.
    """
    from . import provider_controls
    spec = provider_controls.find_control(context, chosen)
    if spec is None:
        lines = [f"No such account control: {chosen}"]
        names = ", ".join(
            item.id for item in provider_controls.available_controls(context))
        if names:
            lines.append(f"Available controls: {names}")
        return provider_controls.ControlResult(lines=lines)
    try:
        return await spec.read(context)
    except (OSError, ValueError, OverflowError,
            authentications.CredentialError) as error:
        return provider_controls.ControlResult(
            lines=[f"Could not read account {chosen}: {error}"])


async def _confirm_account_action(modal, action, as_json):
    answer = (await modal.prompt(f"{action.confirm} [y/N]: ") or "")
    if answer.strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return
    _write_control_result(await action.run(), as_json)


async def run_account_controls_async(command_text, session):
    """Dispatch the provider-dependent account-control entry point.

    The entry lists controls available for the active connection; an optional
    control id runs its read, and an optional action id performs one of the
    actions that read offered.  Nothing here runs implicitly.  The read-only
    control form is normally delivered immediately (see _submit_immediate);
    the interactive forms -- the bare listing and confirmed actions -- are
    queued behind a running turn and take the reader only while a choice or
    confirmation is actually on screen.
    """
    from . import provider_controls
    tokens = command_text.split()
    rest = [token for token in tokens[1:] if token != "--json"]
    as_json = "--json" in tokens[1:]
    if len(rest) > 2:
        print("usage: /account [CONTROL [ACTION]] [--json]")
        return
    context = provider_controls.ControlContext(
        config=current_config(),
        credential_authority=current_session().credential_authority,
    )
    if not rest:
        specs = provider_controls.available_controls(context)
        if not specs:
            print()
            print("No live account controls for the active connection.")
            return
        async with session.modal() as modal:
            chosen = await _numbered_choice_async(
                modal, "Account controls:",
                [(spec.id, f"{spec.title} - {spec.description}")
                 for spec in specs],
                "Control choice (number selects, empty cancels): ")
            if chosen is None:
                return
            result = await _read_account_control(context, chosen)
            _write_control_result(result, as_json)
            if not result.actions:
                return
            action = await _numbered_choice_async(
                modal, "Actions:",
                [(item, item.title) for item in result.actions],
                "Action choice (number selects, empty cancels): ")
            if action is None:
                return
            await _confirm_account_action(modal, action, as_json)
        return
    if len(rest) == 1:
        # The read-only form: modal-free, exactly like its immediate
        # delivery. Reaching here means queue-fed input.
        result = await _read_account_control(context, rest[0])
        _write_control_result(result, as_json)
        hint = _account_actions_hint(rest[0], result.actions)
        if hint:
            terminal.write_text(hint, multiline=True)
            print()
        return
    chosen = rest[0]
    action_id = rest[1]
    result = await _read_account_control(context, chosen)
    _write_control_result(result, as_json)
    action = next(
        (item for item in result.actions if item.id == action_id),
        None)
    if action is None:
        print(f"No such action: {action_id}")
        return
    async with session.modal() as modal:
        await _confirm_account_action(modal, action, as_json)


USAGE = """\
usage: loki [options]

Options:
  -r, --resume LOG        resume a saved chat log (bare --resume: picker)
  -p, --prompt TEXT       one prompt for --headless mode
      --headless          headless single-prompt mode
      --toolset NAME      toolset for headless mode
      --shell-cwd PATH    working directory for tools and Bash
                          (an explicit value outranks a resumed one)
      --dangerously-skip-permissions
                          skip permission prompts
  -h, --help              show this help and exit

Without options, loki starts the interactive TUI.
Use /status for the current connection, /status all for all known connections.
Add --json for JSON; /status save saves this runtime's response observations.
Use loki status [--json] [--endpoint URL] to inspect saved response headers.
Use /account for live provider usage and limit resets, when supported.
Use /queue to inspect queued prompts and staged images while a turn runs.
"""


CLI_SHORT_OPTS = 'r:p:h'
CLI_LONG_OPTS = ['resume=', 'prompt=', 'headless', 'toolset=',
                 'shell-cwd=',
                 'dangerously-skip-permissions', 'help']


def parse_cli_args(args):
    """Normalize and getopt-parse CLI args (raises getopt.GetoptError)."""
    # getopt's "resume=" requires a value; normalize a bare `--resume` to
    # `--resume=` so it opens the picker instead of erroring out.
    args = ['--resume=' if a == '--resume' else a for a in args]
    return getopt.getopt(args, CLI_SHORT_OPTS, CLI_LONG_OPTS)


async def async_main(args) -> int:
    try:
        options, args = parse_cli_args(args)
    except getopt.GetoptError as error:
        # main() handles argument errors before any terminal setup; this
        # fallback only serves direct async_main callers.
        _print_repr_line("loki: ", str(error), file=sys.stderr)
        print(USAGE, end='', file=sys.stderr)
        return 2
    for option_name, _option_value in options:
        if option_name in ('-h', '--help'):
            print(USAGE, end='')
            return 0
    prompt_arg = None
    headless = False
    toolset = None
    shell_cwd = None
    for option_name, option_value in options:
        if option_name in ['--prompt', '-p']:
            prompt_arg = option_value
        elif option_name == '--headless':
            headless = True
        elif option_name == '--toolset':
            toolset = option_value
        elif option_name == '--shell-cwd':
            shell_cwd = option_value

    if shell_cwd is not None:
        try:
            _core.change_shell_cwd(shell_cwd)
        except (FileNotFoundError, NotADirectoryError) as error:
            _print_repr_line(
                "Configuration error: cwd is not a directory: ",
                str(error),
                file=sys.stderr,
            )
            return 2

    if headless:
        try:
            apply_runtime_config(build_config_from_env(
                credentials=_core.CREDENTIALS))
        except (protocols.ProtocolError, ValueError) as e:
            _print_text_line(
                "Configuration error: ", e, file=sys.stderr,
                multiline=True)
            return 2
        if not current_model():
            print("Configuration error: model missing; set LOKI_MODEL.",
                  file=sys.stderr)
            return 2
        await subagents.run_cli_async(
            toolset or "Explore", prompt_arg)
        return 0

    ui_settings = await settings.load_settings(on_error=lambda error: _print_text_line(
        "Could not load settings: ", error, file=sys.stderr, multiline=True))
    log_filename = None
    for option_name, option_value in options:
        if option_name == '--resume' or option_name == '-r':
            log_filename = option_value

    # The input session owns raw mode, the stdin reader, the producer, and the
    # user_messages queue for the whole session (see terminals.InputSession).
    # loki.py consumes the normal queue; session.modal() is the one exclusive
    # path used by the session picker, saved-connection prompt, and /model.
    # Take over the keyboard here, not at terminals import time: importing
    # loki must leave stdin alone (headless and ACP processes read it).
    _terminal_activity.reset()
    terminals.open_terminal_stdin()
    async with input_session(
            on_mode_cycle=lambda: cycle_agent_mode(),
            history_provider=lambda: user_prompt_history(
                current_transcript()),
            on_queue_size_change=(
                _terminal_activity.set_queued_prompts),
            # Immediate commands execute before the input FIFO, regardless of
            # turn state (command_deliveries is the single classifier).
            on_submit=_submit_immediate) as session, contextlib.AsyncExitStack() as status_cleanup:
        # Background exits can occur without input or tool-result events.
        # Detach this display observer before terminal input ownership ends.
        manager = _core.current_job_manager()
        status_cleanup.callback(setattr, manager, "on_change", manager.on_change)
        manager.on_change = _redraw_status
        # Detached monitor-plane work (async immediate commands) must end
        # with the frontend, before terminal teardown races its output.
        # push_async_callback: a plain callback would not await the
        # cancellation gather. The queue-view binding is released the same
        # way so an abnormal exit cannot leave a dead session referenced.
        status_cleanup.callback(_queued_inputs.reset, None)
        status_cleanup.push_async_callback(_cancel_immediate_tasks, session)
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
        refreshed_descriptor = None
        discard_saved_connection = False
        if resolved_log_filename:
            try:
                with open(resolved_log_filename, "r", encoding="utf-8") as f:
                    loaded_chat = savefiles.read_chat_log(f)
                    _, _, saved_state, _ = loaded_chat
                    _core.reasoning_effort_from_session_state(saved_state)
            except (OSError, ValueError, json.JSONDecodeError,
                    formats.TranscriptFormatError) as e:
                _print_text_line(
                    "Could not resume chat: ", e, file=sys.stderr,
                    multiline=True)
                return 1

        try:
            if explicit_api_base_configured(_core.CREDENTIALS):
                config = build_config_from_env(credentials=_core.CREDENTIALS)
            else:
                descriptor = connection_from_session_state(saved_state)
                if descriptor is None:
                    config = None
                else:
                    try:
                        refreshed_descriptor = (
                            await _core.refresh_connection_descriptor_async(
                                descriptor,
                                current_session().credential_authority,
                                diagnostic_writer=_report_model_list_errors,
                            ))
                    except ValueError:
                        # A successful authenticated catalog which no longer
                        # contains this exact slug is authoritative. Keeping
                        # the stale descriptor would repeat the same failure
                        # on every resume.
                        discard_saved_connection = True
                        raise
                    config = config_from_connection_descriptor(
                        refreshed_descriptor, _core.CREDENTIALS)
                    confirmed = await confirm_saved_connection_async(
                        refreshed_descriptor, session, config=config,
                        working_directory=(
                            shell_cwd if shell_cwd is not None
                            else saved_state.get("shell_cwd")))
                    if not confirmed:
                        print("Resume cancelled.", file=sys.stderr)
                        return 0
        except (ConnectionDescriptorError, protocols.ProtocolError,
                ValueError) as e:
            _print_text_line(
                "Configuration error: ", e, file=sys.stderr,
                multiline=True)
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
            # --shell-cwd is an explicit instruction; a value saved in the log
            # must not silently replace it.
            load_chat_log(
                resolved_log_filename, loaded_chat,
                apply_shell_cwd=shell_cwd is None)
            if discard_saved_connection:
                current_session().session_state.pop("connection", None)
                mark_chat_log_dirty()
                save_chat_log()
            elif (refreshed_descriptor is not None
                    and refreshed_descriptor.to_dict()
                    != saved_state.get("connection")):
                set_session_connection(refreshed_descriptor)
                save_chat_log()
            _ResumeTranscriptPresenter(
                current_model() or "Assistant", ui_settings=ui_settings,
            ).write(current_transcript())
        else:
            new_chat_log(new_chat_log_path())

        # The staged-images list is shared with the immediate /queue command
        # (see _QueuedInputs); the loop keeps using this local name.
        pending_images = _queued_inputs.reset(session)
        while True:
            user_in = await session.user_messages.get()
            restore_output_area_after_input()

            if user_in is None:  # EOF sentinel from the producer
                break

            # An empty prompt submits staged images without inventing text.
            if not user_in and not pending_images:
                continue

            print()
            terminal.set_background_color(terminals.INPUT_COLOR)
            print('User: ', end='')
            terminal.write_text(user_in, multiline=True)
            terminal.reset_colors_and_flags()
            print()
            command_text = user_in.strip()
            # Immediate commands never queue: the input owner consumed them
            # before the FIFO (see _submit_immediate). Reaching this point
            # means scripted or otherwise queue-fed input, so run the same
            # handlers once and move on -- they must not start a turn.
            parsed = command_deliveries.terminal_immediate(user_in)
            if parsed is not None:
                outcome = _dispatch_immediate(parsed)
                if outcome is not None:
                    await _run_immediate(user_in, outcome)
                continue
            match command_text:
                case '/quit':
                    break
                case _ if (command_text == '/account'
                           or command_text.startswith('/account ')):
                    # Only the interactive (queued) forms reach here; the
                    # read-only form was claimed by the immediate dispatch
                    # above.
                    await run_account_controls_async(command_text, session)
                    continue
                case '/model':
                    explicit_option = explicit_connection_option(_core.CREDENTIALS)
                    async with session.modal() as modal:
                        try:
                            picked = await modelsdev.run_model_picker_async(
                                input_fn=modal.prompt,
                                credentials=_core.CREDENTIALS,
                                explicit_connection=explicit_option,
                                credential_authority=(
                                    current_session().credential_authority),
                                diagnostic_writer=(
                                    _report_model_list_errors),
                                text_writer=terminal.write_text)
                        except (OSError, json.JSONDecodeError) as e:
                            # models.dev unreachable (network errors) or answered
                            # with non-JSON garbage: fall back to the current
                            # provider's own /models list in the same modal.
                            _print_text_line(
                                "models.dev unavailable: ", e,
                                file=sys.stderr, multiline=True)
                            sys.stderr.flush()
                            models_list = await load_models_async(
                                diagnostic_writer=_report_model_list_errors)
                            selected_model = (
                                await modelsdev.run_flat_model_picker_async(
                                    modal.prompt, models_list,
                                    explicit_connection=explicit_option,
                                    text_writer=terminal.write_text))
                            if selected_model:
                                if isinstance(
                                        selected_model,
                                        modelsdev.ExplicitConnectionOption):
                                    apply_runtime_config(
                                        build_config_from_env(
                                            credentials=_core.CREDENTIALS))
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
                                    "Selected model: ", end="",
                                    file=sys.stderr)
                                terminal.write_text(
                                    repr(selected_label), file=sys.stderr)
                                terminal.write_text(
                                    selected_via, file=sys.stderr)
                                print(file=sys.stderr)
                                _report_unavailable_reasoning_preference()
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
                                credentials=_core.CREDENTIALS))
                            via = " via explicit LOKI_*"
                        else:
                            provider_id, provider_entry, model_entry = picked
                            apply_runtime_config(
                                config_from_modelsdev_selection(
                                    provider_id,
                                    provider_entry,
                                    model_entry,
                                    _core.CREDENTIALS,
                                ))
                            via = (
                                f" via {provider_id!r}"
                                if provider_id else "")
                    except (protocols.ProtocolError, ValueError) as e:
                        _print_text_line(
                            "Could not switch model: ", e,
                            file=sys.stderr, multiline=True)
                        sys.stderr.flush()
                        continue
                    descriptor = active_connection_descriptor()
                    if descriptor is not None:
                        set_session_connection(descriptor)
                    save_chat_log()
                    print("Selected model: ", end="", file=sys.stderr)
                    terminal.write_text(
                        repr(current_model()), file=sys.stderr)
                    terminal.write_text(via, file=sys.stderr)
                    print(file=sys.stderr)
                    _report_unavailable_reasoning_preference()
                    sys.stderr.flush()
                    continue
                case _ if command_text == '/thinking' or command_text.startswith('/thinking '):
                    try:
                        argument = command_text[9:].strip()
                        text = (_core.thinking_command(argument) if argument
                                else await run_thinking_picker_async(session))
                    except (ValueError, OSError) as error:
                        text = str(error)
                    terminal.write_text(text, multiline=True, file=sys.stderr)
                    print(file=sys.stderr)
                    continue
                case _ if command_text == '/trace' or command_text.startswith('/trace '):
                    try:
                        text = _core.trace_command(command_text[6:].strip())
                    except (ValueError, OSError) as error:
                        text = str(error)
                    terminal.write_text(text, multiline=True, file=sys.stderr)
                    print(file=sys.stderr)
                    continue
                case _ if command_text == '/effort' or command_text.startswith('/effort '):
                    print("Use /thinking effort VALUE.", file=sys.stderr)
                    continue
                case '/pwd':
                    print_shell_cwd(
                        text_writer=terminal.write_text)
                    continue
                case _ if command_text == '/cd' or command_text.startswith('/cd '):
                    change_shell_cwd_from_text(
                        command_text[3:].strip(),
                        text_writer=terminal.write_text)
                    continue
                case _ if (command_text == '/image'
                           or (len(command_text) > len('/image')
                               and command_text.startswith('/image')
                               and command_text[len('/image')].isspace())):
                    try:
                        image_path = _image_command_path(command_text)
                        image = load_image_attachment(image_path)
                    except ImageAttachmentError as error:
                        sys.stdout.flush()
                        print("image: ", end="", file=sys.stderr)
                        terminal.write_text(
                            str(error), file=sys.stderr)
                        print(file=sys.stderr)
                        sys.stderr.flush()
                        continue
                    _queued_inputs.stage_image(image)
                    _terminal_activity.set_queued_images(
                        len(pending_images))
                    sys.stdout.flush()
                    print("Attached image for next prompt: ",
                          end="", file=sys.stderr)
                    terminal.write_text(
                        display_path(image.path), file=sys.stderr)
                    print(
                        f" ({image.media_type}, {image.byte_size} bytes)",
                        file=sys.stderr)
                    sys.stderr.flush()
                    continue
                case _:
                    if command_text.startswith('!'):  # direct command execution
                        cmd = user_in[1:].strip()
                        print(
                            f"{computer}: [Running local command: ",
                            end="")
                        terminal.write_text(cmd)
                        print("]")
                        cmd_output = await run_bash_async(cmd)
                        terminal.write_text(cmd_output, multiline=True)
                        print()
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

            record_agent_mode_instruction()
            user_content = []
            if user_in:
                user_content.append(formats.text_block(user_in))
            user_content.extend(
                entry.image.content_block() for entry in pending_images)
            current_transcript().append(
                formats.message_item("user", user_content))
            pending_images.clear()
            _terminal_activity.set_queued_images(0)
            mark_chat_log_dirty()

            turn_events = [] if _core.TOOL_HOOK_PIPELINE.turn_end_hooks else None
            turn_text = ""
            turn_failed = False
            _terminal_activity.set_turn_running(True)
            try:
                # Ctrl+C is a per-turn request. A Ctrl+C used to cancel an
                # earlier prompt or turn must not poison the next model call.
                session.reader.cancel_requested = False
                session.reader.cancel_event.clear()
                turn_text = await run_terminal_turn_async(
                    current_transcript(),
                    cancel_check=lambda: session.reader.cancel_requested,
                    cancel_event=session.reader.cancel_event,
                    turn_events=turn_events, ui_settings=ui_settings)
            except KeyboardInterrupt:
                if turn_events is not None:
                    turn_events.append({"type": "response_cancelled"})
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
            except BaseException:
                turn_failed = True
                raise
            finally:
                try:
                    # Persist the final transcript before notifying external code.
                    save_chat_log()
                    if turn_events is not None:
                        await _core.run_turn_end_hooks_async(
                            turn_events, turn_text, failed=turn_failed)
                finally:
                    _terminal_activity.set_turn_running(False)

        pending_images.clear()
        _terminal_activity.set_queued_images(0)

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
    # Close any open synchronized frame without asserting: at teardown the
    # counter may be anything, and the terminal must not be left frozen.
    run_step(active_terminal.force_end_synchronized_update)
    run_step(active_terminal.flush)


async def _run_frontend(args) -> int:
    """Own terminal setup and teardown inside one already-authorized runtime."""
    # --help and argument errors must leave the user's terminal exactly as
    # it was: handle them before initialize_terminal_overlay touches the
    # screen. No cursor hiding, scroll regions, or teardown output.
    try:
        options, _positional = parse_cli_args(args)
    except getopt.GetoptError as error:
        _print_repr_line("loki: ", str(error), file=sys.stderr)
        print(USAGE, end='', file=sys.stderr)
        return 2
    for option_name, _option_value in options:
        if option_name in ('-h', '--help'):
            print(USAGE, end='')
            return 0
    try:
        configure_tool_hook_pipeline(
            stderr_reporter=_report_hook_stderr)
    except tool_runtime.HookConfigurationError as error:
        _print_text_line(
            "Hook configuration error: ", error,
            file=sys.stderr, multiline=True)
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
            print(
                f"Cleanup error: {type(e).__name__}: ", end="",
                file=sys.stderr)
            terminal.write_text(str(e), file=sys.stderr)
            print(file=sys.stderr)
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
    if hasattr(signal, "pthread_sigmask") and hasattr(signal, "SIG_BLOCK"):
        # POSIX only: block SIGINT so it is delivered to this handler rather
        # than interrupting a turn.  Windows has neither the call nor the
        # signal, and a test may mock one without the other.
        signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT,])

    async def run_with_session_cleanup():
        try:
            return await async_main(args)
        finally:
            manager = current_session().job_manager
            if manager is not None:
                await manager.close_session_owned()

    exit_status = 1
    output_settings = terminal_output_mode()
    with output_settings:
        try:
            # Output mode must outlive InputSession: overlay restoration emits
            # VT sequences too. A partial setup still needs that restoration.
            initialize_terminal_overlay(terminal)
            exit_status = await run_with_session_cleanup()
        finally:
            clean_up()
    if exit_status == 0 and cleanup_failed:
        return 1
    return exit_status


async def _run_credential_runtime(
        args, owner_fd: int, capability_fd: int) -> int:
    runtime = None
    try:
        try:
            runtime = await credential_runtimes.CredentialRuntime.connect(
                owner_fd, capability_fd)
        except (
                credential_capabilities.CapabilityError,
                OSError,
        ) as error:
            _print_text_line(
                "Configuration error: credential capability: ",
                error,
                file=sys.stderr,
                multiline=True,
            )
            return 2
        if runtime is None:
            return 1

        # Terminal and headless runtimes are capability consumers just like
        # ACP workers and subagents. Root broker installation belongs solely
        # to their supervisor; retaining a local fallback here would recreate
        # the asymmetric in-process credential escape hatch.
        _core.CREDENTIALS = runtime.install(current_session())
        completed, result = await runtime.run(_run_frontend(args))
        return result if completed else 1
    finally:
        try:
            await current_session().response_headers.save_on_exit()
        finally:
            if runtime is not None:
                await runtime.close()


def main(args, owner_fd: int, capability_fd: int) -> int:
    """Run the capability-only terminal/headless runtime."""
    return asyncio.run(
        _run_credential_runtime(args, owner_fd, capability_fd))
