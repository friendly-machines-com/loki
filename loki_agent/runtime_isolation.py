"""Per-runtime isolation, chosen once per platform.

The entrypoints need three things before the agent core is imported or a
runtime is started: isolate *this* process, gate a launch on the recorded
workspace, and start the runtime process.  Each is a different mechanism per
platform -- Linux user/mount namespaces and ``create_subprocess_exec``, Windows
AppContainer and its launch API -- so the choice is made here, once.  The
entrypoints contain no platform branch; this module contains no policy about
what to isolate, only how the platform does it.

The seams are deliberately small: a function per capability, each a thin
delegation to the platform module.  Where the platforms do the same thing
(the POSIX no-op for a Windows-only preflight, for instance) the seam says so
locally rather than pushing the condition back into the callers.
"""

from __future__ import annotations

import logging
import sys

from . import host_ipc

logger = logging.getLogger(__name__)

__all__ = [
    "RuntimeIsolationError",
    "close_runtime_process",
    "configured_workspace",
    "isolate_runtime",
    "preflight",
    "start_runtime",
    "verify_contained_runtime",
]


if sys.platform == "win32":
    from . import windows_runtime
    from .runtime_isolations import RuntimeIsolationError

    def isolate_runtime() -> None:
        """Confirm this process is the contained runtime, before it loads."""
        windows_runtime.verify_runtime()

    def verify_contained_runtime() -> None:
        """A subagent inherits containment; re-check the token on Windows."""
        windows_runtime.verify_runtime()

    def configured_workspace(arguments: list[str]) -> str | None:
        return windows_runtime.configured_workspace(arguments)

    def preflight(arguments: list[str]) -> int | None:
        """Windows-only entrypoint checks; ``None`` means continue."""
        import getopt

        from .terminal_frontend import USAGE, parse_cli_args

        try:
            options, positional = parse_cli_args(arguments)
            if positional:
                raise getopt.GetoptError("unexpected positional arguments")
        except getopt.GetoptError as error:
            print(f"loki: {error}\n{USAGE}", file=sys.stderr)
            return 2
        if any(name in ("-h", "--help") for name, _ in options):
            print(USAGE, end="")
            return 0
        windows_runtime.configured_workspace(arguments)
        return None

    async def start_runtime(executable, arguments, workspace, environment,
                            delegation):
        command = [
            executable, "--runtime", *delegation.child_arguments(),
            "--", *arguments,
        ]
        logger.debug("runtime command: %r", command)
        inherited = [*host_ipc.handles(delegation.owner_child),
                     *host_ipc.handles(delegation.credential_child)]
        return windows_runtime.launch(
            executable, command[1:], environment, workspace, inherited)

    def close_runtime_process(process) -> None:
        process.close()

else:
    from . import runtime_isolations
    from .runtime_isolations import RuntimeIsolationError

    def isolate_runtime() -> None:
        runtime_isolations.isolate_credential_directory()

    def verify_contained_runtime() -> None:
        # A subagent inherits the runtime's covered mount; re-unsharing would
        # add a namespace level for no security gain.
        return None

    def configured_workspace(arguments: list[str]) -> str | None:
        return None

    def preflight(arguments: list[str]) -> int | None:
        return None

    async def start_runtime(executable, arguments, workspace, environment,
                            delegation):
        import asyncio

        command = [
            executable, "--runtime", *delegation.child_arguments(),
            "--", *arguments,
        ]
        logger.debug("runtime command: %r", command)
        return await asyncio.create_subprocess_exec(
            *command, close_fds=True, env=environment,
            **delegation.child_spawn_kwargs())

    def close_runtime_process(process) -> None:
        # An asyncio subprocess owns its transport; waiting has closed it.
        return None
