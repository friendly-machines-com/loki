"""Terminal front-end: the TUI loki.py used to contain.

Owns the ANSI renderer, the status bar, the pickers, the input session,
and process lifecycle (signal handling, overlay setup/teardown).  The
core (loki.py) keeps the tool loop, tools, provider config, and chat
log machinery -- importing it no longer touches the tty.
"""

from __future__ import annotations

import asyncio
import base64
import getopt
import json
import os
import shlex
import signal
import stat
import sys
from dataclasses import dataclass
from pprint import pprint

from . import formats
from . import models as modelsdev
from . import protocols
from . import savefiles
from . import terminals
from .credentials import CredentialStore
from . import tool_runtime
from .connections import ConnectionDescriptor, ConnectionDescriptorError
from .loki import RuntimeConfig, computer
from . import loki as _core
from .loki import (
    ERROR_COLOR,
    EXPLORE_TOOLS,
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
    run_jobs,
    run_tool_loop_async,
    save_chat_log,
    set_session_connection,
    user_prompt_history,
)
from .terminals import (
    input_session, restore_output_area_after_input, terminal)


IMAGE_ATTACHMENT_MAX_BYTES = 20 * 1024 * 1024


class ImageAttachmentError(ValueError):
    pass


@dataclass(frozen=True)
class StagedImage:
    path: str
    media_type: str
    encoded_data: str
    byte_size: int

    def content_block(self) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.encoded_data,
            },
        }


def _image_media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if (len(data) >= 12
            and data.startswith(b"RIFF")
            and data[8:12] == b"WEBP"):
        return "image/webp"
    return None


def _image_command_path(command_text: str) -> str:
    try:
        parts = shlex.split(command_text[len("/image"):].strip())
    except ValueError as error:
        raise ImageAttachmentError(f"invalid path quoting: {error}") from error
    if len(parts) != 1:
        raise ImageAttachmentError(
            "usage: /image PATH (quote a path containing spaces)")
    return parts[0]


def load_image_attachment(path_text: str, *,
                          base_dir: str | None = None,
                          max_bytes: int | None = None) -> StagedImage:
    """Read one local image snapshot for a later terminal prompt."""
    limit = IMAGE_ATTACHMENT_MAX_BYTES if max_bytes is None else max_bytes
    if limit < 1:
        raise ValueError("image attachment limit must be positive")

    expanded = os.path.expanduser(path_text)
    if not os.path.isabs(expanded):
        expanded = os.path.join(base_dir or current_cwd(), expanded)
    path = os.path.realpath(os.path.normpath(expanded))

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    fd = None
    try:
        fd = os.open(path, flags)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ImageAttachmentError(
                f"not a regular file: {display_path(path)}")
        if file_stat.st_size > limit:
            raise ImageAttachmentError(
                f"image is {file_stat.st_size} bytes; maximum is "
                f"{limit} bytes")
        with os.fdopen(fd, "rb") as image_file:
            fd = None
            data = image_file.read(limit + 1)
    except ImageAttachmentError:
        raise
    except (OSError, ValueError) as error:
        detail = getattr(error, "strerror", None) or str(error)
        raise ImageAttachmentError(
            f"cannot read {display_path(path)}: {detail}") from error
    finally:
        if fd is not None:
            os.close(fd)

    if len(data) > limit:
        raise ImageAttachmentError(
            f"image exceeds the {limit}-byte maximum")
    media_type = _image_media_type(data)
    if media_type is None:
        raise ImageAttachmentError(
            "unsupported image data; expected PNG, JPEG, GIF, or WebP")
    return StagedImage(
        path=path,
        media_type=media_type,
        encoded_data=base64.b64encode(data).decode("ascii"),
        byte_size=len(data),
    )


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
    elif kind == "provider_error":
        terminal.set_background_color(ERROR_COLOR)
        error = event["error"]
        print(f"Provider protocol error: {error}", end='')
        terminal.reset_colors_and_flags()
        print()
        sys.stdout.flush()
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
    read_only = current_agent_mode() in ("explore", "plan")
    active_tools = (
        [
            tool for tool in TOOLS
            if tool.get("function", {}).get("name") in EXPLORE_TOOLS
        ]
        if read_only else TOOLS
    )

    async def chat_fn(items, on_text_delta):
        return await async_chat_completion(
            items, active_tools, True, False,
            on_text_delta=on_text_delta,
            cancel_check=cancel_check)

    return await run_tool_loop_async(
        transcript_items,
        allowed=EXPLORE_TOOLS if read_only else None,
        chat_fn=chat_fn,
        on_event=_terminal_agent_event,
        cancel_check=cancel_check,
        stream_chat=True,
        report_timing=True,
        cancel_event=cancel_event,
        on_response=lambda turn, event: _remember_session_toolset(
            active_tools),
    )


def status_text() -> str:
    displayed_model = current_model()
    if (current_config() is not None
            and current_config().model_status == "deprecated"):
        displayed_model += " (deprecated)"
    return (
        'Remote: API: {}; Model: {}; /model\n'
        'Local: mode={}; CWD: {}; /pwd, /cd DIR, /ps, /image PATH, '
        '!foo, /quit'
    ).format(
        _status_api_base(), displayed_model,
        current_agent_mode(), display_path(current_cwd()))


terminals.set_status_text_provider(status_text)


async def run_session_picker_async(session):
    async with session.modal() as modal:
        picked = await savefiles.run_session_picker_async(
            input_fn=modal.prompt,
            terminal=terminal, chat_log_dir=_core.CHAT_LOG_DIR)
        # Finish the picker's output cleanup while the modal still owns the
        # terminal. Only then may the normal input producer resume.
        terminal.goto_position(1, 1)
        terminal.clear_to_end_of_screen()
        terminal.flush()
    return picked


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
            "You are a focused, read-only Explore subagent. Use "
            "Glob/Grep/Read/WebFetch/WebSearch to investigate, then write a "
            "concise final answer."),
        formats.message_item("user", prompt),
    ]
    current_session().agent_mode = "explore"
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
                credentials=_core.CREDENTIALS))
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
    # Take over the keyboard here, not at terminals import time: importing
    # loki must leave stdin alone (headless and ACP processes read it).
    terminals.open_terminal_stdin()
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
            if explicit_api_base_configured(_core.CREDENTIALS):
                config = build_config_from_env(credentials=_core.CREDENTIALS)
            else:
                descriptor = connection_from_session_state(saved_state)
                if descriptor is None:
                    config = None
                else:
                    config = config_from_connection_descriptor(
                        descriptor, _core.CREDENTIALS)
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
            savefiles.print_resume_transcript(
                current_transcript(), current_model() or "Assistant")
        else:
            new_chat_log(new_chat_log_path())

        pending_images = []
        while True:
            user_in = await session.user_messages.get()
            restore_output_area_after_input()

            if user_in is None:  # EOF sentinel from the producer
                break

            # An empty prompt submits staged images without inventing text.
            if not user_in and not pending_images:
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
                    explicit_option = explicit_connection_option(_core.CREDENTIALS)
                    async with session.modal() as modal:
                        try:
                            picked = await modelsdev.run_model_picker_async(
                                input_fn=modal.prompt,
                                credentials=_core.CREDENTIALS,
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
                case _ if (command_text == '/image'
                           or (len(command_text) > len('/image')
                               and command_text.startswith('/image')
                               and command_text[len('/image')].isspace())):
                    try:
                        image_path = _image_command_path(command_text)
                        image = load_image_attachment(image_path)
                    except ImageAttachmentError as error:
                        sys.stdout.flush()
                        print(f"image: {error}", file=sys.stderr)
                        sys.stderr.flush()
                        continue
                    pending_images.append(image)
                    sys.stdout.flush()
                    print(
                        "Attached image for next prompt: "
                        f"{display_path(image.path)} "
                        f"({image.media_type}, {image.byte_size} bytes)",
                        file=sys.stderr,
                    )
                    sys.stderr.flush()
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

            record_agent_mode_instruction()
            user_content = []
            if user_in:
                user_content.append(formats.text_block(user_in))
            user_content.extend(
                image.content_block() for image in pending_images)
            current_transcript().append(
                formats.message_item("user", user_content))
            pending_images.clear()
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
    # Capture credentials into the core's module state; the rest of this
    # module reads them via _core.CREDENTIALS at call time.
    _core.CREDENTIALS = CredentialStore.capture(os.environ)
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
