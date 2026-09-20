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
import os
import sys

from . import host_ipc

logger = logging.getLogger(__name__)

__all__ = [
    "RuntimeIsolationError",
    "close_runtime_process",
    "configured_workspace",
    "isolate_runtime",
    "preflight",
    "prepare_runtime_scratch",
    "start_runtime",
    "start_worker",
    "verify_contained_runtime",
    "worker_command",
]


def worker_command() -> list[str]:
    """The command that starts this program again as a worker.

    ``sys.executable`` is the image: the bootloader in a frozen build, and the
    interpreter for a source script, which is then handed its own launcher as
    an absolute path.  ``sys.argv[0]`` is only the name the caller typed --
    possibly relative, possibly a symlink -- and must not be used as an
    executable.

    The contained Windows launch re-derives the same re-entry from
    ``sys.argv[0]`` and ``"--worker"`` (``launch`` applies the frozen/source
    rule itself); the ``--worker`` token is pinned by ``acp_main``'s dispatch.
    """
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--worker"]
    else:
        command = [sys.executable, os.path.abspath(sys.argv[0]), "--worker"]
    logger.debug(
        "worker command: %r (sys.argv[0]=%r sys.executable=%r frozen=%r)",
        command, sys.argv[0], sys.executable, getattr(sys, "frozen", False))
    return command


if sys.platform == "win32":
    from . import windows_runtime
    from .runtime_isolations import RuntimeIsolationError

    def isolate_runtime() -> None:
        """Confirm this process is the contained runtime, before it loads."""
        windows_runtime.verify_runtime()

    def verify_contained_runtime() -> None:
        """A subagent inherits containment; re-check the token on Windows."""
        windows_runtime.verify_runtime()

    def prepare_runtime_scratch() -> None:
        """Give the contained runtime the scratch directory it was pointed at."""
        windows_runtime.ensure_runtime_temp()

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
        # Not the workspace: the ambient cwd is what relative opens, DLL
        # searches and executable lookups resolve against, and the workspace
        # is the one directory model-directed tools can write.
        return windows_runtime.launch(
            executable, command[1:], environment, workspace, inherited,
            current_directory=os.getcwd())

    async def start_worker(cwd, environment, delegation):
        from . import windows_workers
        return await windows_workers.start_worker(cwd, environment, delegation)

    async def close_runtime_process(process) -> None:
        from . import windows_workers
        if isinstance(process, windows_workers.Worker):
            await process.close()
        else:
            # The terminal runtime has no parent-side stdio transports.
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

    def prepare_runtime_scratch() -> None:
        # POSIX scratch is the inherited TMPDIR: /tmp carries the sticky bit,
        # and the surrounding VM is the boundary, so nothing is relocated.
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

    async def start_worker(cwd, environment, delegation):
        """The ACP worker on POSIX: piped stdio, delegated descriptors.

        ``cwd`` is unused here -- the worker learns the session directory
        over the protocol, and POSIX confinement is namespaces established in
        the child, not a per-directory launch gate.
        """
        import asyncio

        command = [*worker_command(), *delegation.child_arguments()]
        logger.debug("worker command: %r", command)
        return await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            close_fds=True,
            env=environment,
            **delegation.child_spawn_kwargs(),
            # ACP uses pipes, not a terminal. A new session prevents
            # an inherited controlling terminal from becoming an
            # escape channel through TIOCSTI or terminal signals.
            start_new_session=True,
        )

    async def close_runtime_process(process) -> None:
        # An asyncio subprocess owns its transport; waiting has closed it.
        return None
