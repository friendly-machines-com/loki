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
    from . import windows_api
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
        # Not the workspace: the ambient cwd is what relative opens, DLL
        # searches and executable lookups resolve against, and the workspace
        # is the one directory model-directed tools can write.
        return windows_runtime.launch(
            executable, command[1:], environment, workspace, inherited,
            current_directory=os.getcwd())

    class _ContainedWorker:
        """An asyncio-``Process``-shaped view of one contained ACP worker.

        ``stdin``/``stdout`` are the front's pipe streams, so the worker
        channel code is transport-neutral; ``wait``/``terminate``/``kill``
        and ``returncode`` delegate to the native process and its
        kill-on-close job; ``close`` releases the stdio threads, both front
        pipe handles and the native handles.
        """

        def __init__(self, process, stdio):
            self._process = process
            self._stdio = stdio
            self.stdin = stdio.stdin
            self.stdout = stdio.stdout

        @property
        def returncode(self):
            return self._process.returncode

        async def wait(self):
            return await self._process.wait()

        def terminate(self):
            self._process.terminate()

        kill = terminate

        def close(self):
            self._stdio.close()
            self._process.close()

    async def start_worker(cwd, environment, delegation):
        """The ACP worker, contained like the terminal runtime.

        The session cwd is the workspace: its container must already be
        recorded (``loki-setup``), and the launch refuses otherwise, so an
        editor pointing at an unconfigured directory gets a denied session
        rather than an unprotected worker.  The worker's stdin/stdout are two
        anonymous pipes -- handle possession is the authorization, exactly as
        for the credential channel -- and it re-proves containment at its own
        gate before the frontend loads.
        """
        workspace = windows_runtime.required_workspace(cwd)
        front, child = host_ipc.worker_stdio()
        try:
            inherited = [*host_ipc.handles(delegation.owner_child),
                         *host_ipc.handles(delegation.credential_child)]
            # The worker's actual cwd is the front's at spawn.  It must
            # never be derived from the session cwd: the session directory
            # is protocol-supplied state, and the workspace is the one
            # directory model-directed tools can write, so as the ambient
            # directory it would turn every relative open, DLL search and
            # executable name into model-writable resolution.
            process = windows_runtime.launch(
                sys.argv[0],
                ["--worker", *delegation.child_arguments()],
                environment, workspace, inherited, stdio=child,
                current_directory=os.getcwd())
        except BaseException:
            for handle in (*front, *child):
                windows_api.close_handle(handle)
            raise
        try:
            stdio = host_ipc.WorkerStdio(*front)
        except BaseException:
            process.terminate()
            process.close()
            for handle in (*front, *child):
                windows_api.close_handle(handle)
            raise
        return _ContainedWorker(process, stdio)

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
