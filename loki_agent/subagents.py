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

from . import acps
from . import credential_capabilities
from . import formats
from . import loki as _core
from . import protocols
from . import texts
from .credentials import CredentialInventory, CredentialStore


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


class _SessionOwner:
    """One inherited session-lifetime capability.

    The read end becomes ready only when its parent closes the write end.
    Monitoring begins before credential initialization because ownership
    applies to the delegated process's complete lifetime, not merely its first
    model turn.
    """

    def __init__(self, fd: int):
        os.set_inheritable(fd, False)
        self.fd = fd
        self.closed_task = asyncio.create_task(
            acps.AsyncFdLineReader(fd).readline(),
            name="loki-session-owner",
        )
        self._closed = False

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if not self.closed_task.done():
            self.closed_task.cancel()
        await asyncio.gather(
            self.closed_task, return_exceptions=True)
        try:
            os.close(self.fd)
        except OSError:
            pass


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


def run_prompt(subagent_type: str, prompt: str) -> str:
    return asyncio.run(run_prompt_async(subagent_type, prompt))


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


async def _connect_credential_while_owned(
        capability_fd: int,
        owner: _SessionOwner):
    """Initialize a credential client only while its session owner is live."""
    # CredentialClient.take_fd() takes FD ownership synchronously. That is a
    # security property: if owner revocation wins before the new task receives
    # a timeslice, canceling the task must still close the inherited capability.
    initializer = (
        credential_capabilities.CredentialClient.take_fd(capability_fd))
    client_task = asyncio.create_task(
        initializer.connect(),
        name="loki-credential-initialization",
    )
    try:
        done, _pending = await asyncio.wait(
            {owner.closed_task, client_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        # Cancellation of startup is also revocation. Do not leave either the
        # initialization task or its not-yet-transferred socket behind.
        if not client_task.done():
            client_task.cancel()
        await asyncio.gather(client_task, return_exceptions=True)
        initializer.close_now()
        raise
    if owner.closed_task not in done:
        return await client_task

    # Owner closure wins a simultaneous handshake completion. A credential
    # channel has no independent lifetime: it was delegated by this owner and
    # must not survive the owner's revocation.
    if not client_task.done():
        client_task.cancel()
    result = (await asyncio.gather(
        client_task, return_exceptions=True))[0]
    initializer.close_now()
    if isinstance(
            result, credential_capabilities.CredentialClient):
        await result.close()
    return None


async def _run_until_session_owner_closes(
        coroutine, owner: _SessionOwner,
        credential_client=None) -> bool:
    """Run only while the owning session and credential broker are alive."""
    operation_task = asyncio.create_task(
        coroutine, name="loki-session-owned-operation")
    capability_task = (
        asyncio.create_task(
            credential_client.wait_closed(),
            name="loki-credential-owner",
        )
        if credential_client is not None else None
    )
    monitored = {owner.closed_task, operation_task}
    if capability_task is not None:
        monitored.add(capability_task)
    try:
        done, _pending = await asyncio.wait(
            monitored,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            await operation_task
            return True
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        return False
    finally:
        monitor_tasks = []
        if capability_task is not None:
            if not capability_task.done():
                capability_task.cancel()
            monitor_tasks.append(capability_task)
        if monitor_tasks:
            await asyncio.gather(
                *monitor_tasks, return_exceptions=True)


def _close_descriptors(options: SubagentOptions) -> None:
    for fd in (
            options.session_owner_fd,
            options.credential_capability_fd):
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


async def async_main(
        args, credentials: CredentialStore) -> int:
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
    if (owner_fd is None) != (capability_fd is None):
        _close_descriptors(options)
        print(
            "Configuration error: delegated subagents require both the "
            "session owner and credential capability descriptors.",
            file=sys.stderr,
        )
        return 2

    owner = _SessionOwner(owner_fd) if owner_fd is not None else None
    credential_client = None
    try:
        if capability_fd is not None:
            try:
                credential_client = (
                    await _connect_credential_while_owned(
                        capability_fd, owner))
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
            if credential_client is None:
                return 1
            # A delegated process installs exactly the restricted authority it
            # received. It must not also create a parallel root broker, even
            # though its environment is expected to contain no credentials.
            _core.current_session().credential_authority = credential_client
            _core.CREDENTIALS = CredentialInventory.from_environment(
                os.environ, credential_client.available())
        else:
            # A directly invoked top-level subagent has no delegating parent,
            # so its own captured startup credentials form the root authority.
            _core.install_root_credential_broker(credentials)
            _core.CREDENTIALS = credentials.inventory()

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
        if owner is None:
            await operation
            return 0
        completed = await _run_until_session_owner_closes(
            operation, owner, credential_client)
        return 0 if completed else 1
    finally:
        if credential_client is not None:
            await credential_client.close()
        if owner is not None:
            await owner.close()


def main(args, credentials: CredentialStore) -> int:
    return asyncio.run(async_main(args, credentials))
