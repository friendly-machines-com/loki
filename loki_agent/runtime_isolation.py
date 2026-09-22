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
    interpreter for a source script, which is then handed its own launcher by
    the name the caller typed.  That name is used as a script argument, never
    as an executable, and it resolves against the process's cwd -- which is
    fixed and inherited by the child, so it names the same file for both.

    The contained Windows launch re-derives the same re-entry from
    ``sys.argv[0]`` and ``"--worker"`` (``launch`` applies the frozen/source
    rule itself); the ``--worker`` token is pinned by ``acp_main``'s dispatch.
    """
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--worker"]
    else:
        command = [sys.executable, sys.argv[0], "--worker"]
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

        from .terminal_frontend import USAGE, _print_repr_line, parse_cli_args

        try:
            options, positional = parse_cli_args(arguments)
            if positional:
                raise getopt.GetoptError("unexpected positional arguments")
        except getopt.GetoptError as error:
            _print_repr_line("loki: ", str(error), file=sys.stderr)
            print(USAGE, end='', file=sys.stderr)
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
        """The ACP worker, contained like the terminal runtime.

        The session cwd is the workspace: its container must already be
        recorded (``loki-setup``), and the launch refuses otherwise, so an
        editor pointing at an unconfigured directory gets a denied session
        rather than an unprotected worker.  The worker's stdin/stdout are two
        pre-connected private pipes whose child ends cross as inherited
        handles -- possession is the authorization, exactly as for the
        credential channel -- and it re-proves containment at its own gate
        before the frontend loads.

        The returned object is an ``asyncio.subprocess.Process``; the
        transport owns the pipes and the native process handles.
        """
        from . import windows_subprocesses

        workspace = windows_runtime.required_workspace(cwd)
        # The worker's actual cwd is the front's at spawn.  It must never be
        # derived from the session cwd: the session directory is
        # protocol-supplied state, and the workspace is the one directory
        # model-directed tools can write, so as the ambient directory it would
        # turn every relative open, DLL search and executable name into
        # model-writable resolution.
        return await windows_subprocesses.create_worker_process(
            workspace=workspace,
            environment=environment,
            arguments=["--worker", *delegation.child_arguments()],
            inherited_handles=[*host_ipc.handles(delegation.owner_child),
                               *host_ipc.handles(delegation.credential_child)],
            current_directory=os.getcwd())

    def close_runtime_process(process) -> None:
        # A contained worker's transport owns its pipes, its native process
        # handles and its job object, and releases them when the worker has
        # exited and its pipes are drained.  Nothing is left for the caller.
        return None

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

    def close_runtime_process(process) -> None:
        # An asyncio subprocess owns its transport; waiting has closed it.
        return None
