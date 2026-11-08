"""Runtime launch and containment for win32 and POSIX.

The entrypoints need three things before the agent core is imported or a
runtime is started: apply the native containment checks, gate a launch on the
recorded workspace where required, and start the runtime process. Windows
uses AppContainer checks and native launch APIs. POSIX uses asyncio
subprocesses; Linux additionally installs a user/mount namespace credential
cover. Other POSIX systems do not install that Linux-specific cover.

The platform choice is made here, not in the entrypoints. Each capability is
a thin delegation to its platform implementation; shared invocation grammar
stays in arguments. This keeps platform launch checks out of the callers.
"""

from __future__ import annotations

import logging
import os
import sys

from . import host_ipc, host_process, streams

logger = logging.getLogger(__name__)

__all__ = [
    "RuntimeIsolationError",
    "close_runtime_process",
    "configured_workspace",
    "internal_command",
    "isolate_runtime",
    "preflight",
    "prepare_runtime_scratch",
    "start_runtime",
    "start_worker",
    "verify_contained_runtime",
    "worker_command",
]


def internal_command(mode: str) -> list[str]:
    """Replay the selected dispatcher with the current image, not tool PATH.

    A hook may intentionally have a minimal environment. A source dispatcher's
    shebang lookup must not silently choose a different interpreter there.
    """
    if getattr(sys, 'frozen', False):
        return [sys.executable, mode]
    return [sys.executable, sys.argv[0], mode]


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
    command = internal_command('--worker')
    logger.debug(
        "worker command: %r (sys.argv[0]=%r sys.executable=%r frozen=%r)",
        command, sys.argv[0], sys.executable, getattr(sys, "frozen", False))
    return command


if sys.platform == "win32":
    from . import windows_runtime
    from .runtime_isolations import RuntimeIsolationError

    def isolate_runtime(credential_directory=None) -> None:
        """Confirm this process is the contained runtime, before it loads."""
        if credential_directory is not None:
            raise RuntimeIsolationError("credential protection is unavailable")
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
        """Validate invocation before the Windows workspace launch gate."""
        from .arguments import preflight as argument_preflight

        status = argument_preflight(arguments, reject_positionals=True)
        if status is not None:
            return status
        windows_runtime.configured_workspace(arguments)
        return None

    async def start_runtime(executable, arguments, workspace, environment,
                            delegation, *, credential_directory=None):
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
        return await windows_runtime.launch(
            executable, command[1:], environment, workspace, inherited,
            current_directory=os.getcwd())

    async def start_worker(cwd, environment, delegation, *, credential_directory=None):
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

else:
    from . import runtime_isolations
    from .runtime_isolations import RuntimeIsolationError

    def isolate_runtime(credential_directory=None) -> None:
        runtime_isolations.isolate_credential_directory(credential_directory)

    def verify_contained_runtime() -> None:
        # On Linux, a subagent inherits the runtime's credential cover;
        # re-unsharing would add a namespace level for no security gain.
        return None

    def prepare_runtime_scratch() -> None:
        # POSIX scratch is the inherited TMPDIR: /tmp carries the sticky bit,
        # and the surrounding VM is the boundary, so nothing is relocated.
        return None

    def configured_workspace(arguments: list[str]) -> str | None:
        return None

    def preflight(arguments: list[str]) -> int | None:
        from .arguments import preflight as argument_preflight

        return argument_preflight(arguments)

    async def _create_posix_process(delegation, *command, stdin=None, stdout=None,
                                    stderr=None, **options):
        # Delegation owns revocation; the shared creator owns unpublished
        # process/transport retirement, including late creation results.
        return await host_process.create_posix_process(
            *command, on_cancel=delegation.revoke_now,
            stdin=stdin, stdout=stdout, stderr=stderr, **options)

    async def start_runtime(executable, arguments, workspace, environment,
                            delegation, *, credential_directory=None):
        if credential_directory is None:
            raise RuntimeIsolationError("credential protection is unavailable")
        directory_fd = host_ipc.duplicate_for_child(credential_directory)
        try:
            command = [
                executable, "--runtime", *delegation.child_arguments(),
                "--credential-directory-fd", str(directory_fd),
                "--", *arguments,
            ]
            logger.debug("runtime command: %r", command)
            return await _create_posix_process(
                delegation, *command, close_fds=True, env=environment,
                **delegation.child_spawn_kwargs(directory_fd))
        finally:
            # Creation and cancellation cleanup have settled before this
            # launch copy closes. Storage still owns its original handle.
            os.close(directory_fd)

    async def start_worker(cwd, environment, delegation, *, credential_directory=None):
        """The ACP worker on POSIX: piped stdio, delegated descriptors.

        ``cwd`` is unused here -- the worker learns the session directory
        over the protocol. POSIX has no per-directory launch gate; on Linux,
        the child additionally establishes the credential-cover namespaces.
        """
        import asyncio

        if credential_directory is None:
            raise RuntimeIsolationError("credential protection is unavailable")
        directory_fd = host_ipc.duplicate_for_child(credential_directory)
        try:
            command = [*worker_command(), *delegation.child_arguments(),
                       "--credential-directory-fd", str(directory_fd)]
            logger.debug("worker command: %r", command)
            return await _create_posix_process(
                delegation, *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,
                close_fds=True,
                env=environment,
                **delegation.child_spawn_kwargs(directory_fd),
                # ACP uses pipes, not a terminal. A new session prevents
                # an inherited controlling terminal from becoming an
                # escape channel through TIOCSTI or terminal signals.
                start_new_session=True,
            )
        finally:
            os.close(directory_fd)


def close_runtime_process(process) -> None:
    # Call only after actual leader exit and reader retirement. Our process
    # surface can close descendant-held pipes without private stdlib access;
    # its Windows transport retains the native observer until exit. Raw native
    # terminal launches instead hand those handles directly to the supervisor.
    # Fixture-supplied stdlib processes retain their own stream owner.
    if (isinstance(process, streams.Process)
            or (sys.platform == 'win32' and isinstance(process, windows_runtime.ContainedProcess))):
        process.close()
