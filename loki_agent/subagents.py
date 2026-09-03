"""Headless execution of Loki's internal read-only subagents.

The user-facing executables decide whether ``--subagent`` selects this module.
That keeps executable identity at the process boundary: a terminal-owned
subagent is launched through ``loki.py`` and an ACP-owned subagent through
``loki-acp``.  Neither entry point redirects into the other's frontend.
"""

from __future__ import annotations

import asyncio
import getopt
import os
import sys
from dataclasses import dataclass

from . import credential_capabilities
from . import credential_runtimes
from . import formats
from . import loki as _core
from . import protocols
from . import texts


USAGE_OPTIONS = """\

Options:
  -p, --prompt TEXT       prompt text (default: read stdin)
      --shell-cwd PATH    logical working directory inherited from the owner
      --session-owner-fd FD
                          owner-lifetime descriptor
      --credential-capability-fd FD
                          delegated credential descriptor
  -h, --help              show this help and exit
"""


def usage() -> str:
    return (
        f"usage: {os.path.basename(sys.argv[0])} "
        f"--subagent TYPE [options]\n"
        f"{USAGE_OPTIONS}")


@dataclass(frozen=True)
class SubagentOptions:
    subagent_type: str
    prompt: str | None = None
    shell_cwd: str | None = None
    session_owner_fd: int | None = None
    credential_capability_fd: int | None = None
    help: bool = False


def _print_text_line(prefix, text, *, file=None):
    print(prefix, end="", file=file)
    print(
        texts.escape_terminal_text(str(text), multiline=True),
        end="",
        file=file,
    )
    print(file=file)


def _descriptor(value: str, description: str) -> int:
    try:
        fd = int(value)
        if fd < 3:
            raise ValueError(f"{description} descriptor must be at least 3")
        os.fstat(fd)
        return fd
    except (OSError, ValueError) as error:
        raise ValueError(
            f"invalid {description} descriptor: {error}") from error


def parse_args(args) -> SubagentOptions:
    if not args:
        raise getopt.GetoptError("subagent type is required")
    if args[0] in ("-h", "--help"):
        return SubagentOptions("", help=True)

    subagent_type = args[0]
    options, positional = getopt.getopt(
        args[1:],
        "p:h",
        [
            "prompt=",
            "shell-cwd=",
            "session-owner-fd=",
            "credential-capability-fd=",
            "help",
        ],
    )
    if positional:
        raise getopt.GetoptError(
            f"unexpected positional argument: {positional[0]}")

    values = {
        "prompt": None,
        "shell_cwd": None,
        "session_owner_fd": None,
        "credential_capability_fd": None,
        "help": False,
    }
    for option_name, option_value in options:
        if option_name in ("-p", "--prompt"):
            values["prompt"] = option_value
        elif option_name == "--shell-cwd":
            values["shell_cwd"] = option_value
        elif option_name == "--session-owner-fd":
            values["session_owner_fd"] = _descriptor(
                option_value, "session owner")
        elif option_name == "--credential-capability-fd":
            values["credential_capability_fd"] = _descriptor(
                option_value, "credential capability")
        elif option_name in ("-h", "--help"):
            values["help"] = True

    return SubagentOptions(subagent_type=subagent_type, **values)


async def run_prompt_async(subagent_type: str, prompt: str) -> str:
    if subagent_type != "Explore":
        return (
            f"Error: unknown subagent_type {subagent_type!r} "
            "(only 'Explore' is supported)")
    if not prompt:
        return ""
    messages = [
        formats.instruction_item(
            "You are a focused, read-only Explore subagent. Use "
            "Glob/Grep/Read/WebFetch/WebSearch to investigate. You may use "
            "Agent to delegate another read-only Explore search. Then write "
            "a concise final answer."),
        formats.message_item("user", prompt),
    ]
    _core.current_session().agent_mode = "explore"
    return await _core.run_tool_loop_async(
        messages, allowed=_core.EXPLORE_TOOLS)


async def run_cli_async(
        subagent_type: str, prompt: str | None = None) -> None:
    prompt = prompt if prompt is not None else sys.stdin.read().strip()
    result = await run_prompt_async(subagent_type, prompt)
    if result:
        print(
            texts.escape_terminal_text(result, multiline=True),
            end="",
        )
        print()


def _close_descriptors(options: SubagentOptions) -> None:
    for fd in (
            options.session_owner_fd,
            options.credential_capability_fd):
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


async def async_main(args) -> int:
    try:
        options = parse_args(args)
    except (getopt.GetoptError, ValueError) as error:
        _print_text_line("loki subagent: ", error, file=sys.stderr)
        print(usage(), end="", file=sys.stderr)
        return 2

    if options.help:
        _close_descriptors(options)
        print(usage(), end="")
        return 0

    owner_fd = options.session_owner_fd
    capability_fd = options.credential_capability_fd
    if owner_fd is None or capability_fd is None:
        _close_descriptors(options)
        print(
            "Configuration error: subagent runtimes require both the session "
            "owner and credential capability descriptors.",
            file=sys.stderr,
        )
        return 2

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
            )
            return 2
        if runtime is None:
            return 1
        # A subagent installs exactly the restricted authority it received.
        # There is intentionally no direct/root fallback: such a fallback
        # would let an internal runtime silently bypass its supervisor.
        _core.CREDENTIALS = runtime.install(
            _core.current_session())

        if options.shell_cwd is not None:
            try:
                _core.change_shell_cwd(options.shell_cwd)
            except (FileNotFoundError, NotADirectoryError) as error:
                _print_text_line(
                    "Configuration error: cwd is not a directory: ",
                    error,
                    file=sys.stderr,
                )
                return 2

        try:
            _core.apply_runtime_config(_core.build_config_from_env(
                credentials=_core.CREDENTIALS))
        except (protocols.ProtocolError, ValueError) as error:
            _print_text_line(
                "Configuration error: ", error, file=sys.stderr)
            return 2
        if not _core.current_model():
            print(
                "Configuration error: model missing; set LOKI_MODEL.",
                file=sys.stderr,
            )
            return 2

        operation = run_cli_async(
            options.subagent_type, options.prompt)
        completed, _result = await runtime.run(operation)
        return 0 if completed else 1
    finally:
        if runtime is not None:
            await runtime.close()


def main(args) -> int:
    return asyncio.run(async_main(args))
