"""Worker entry point: single-session Loki speaking ACP on stdio.

Spawned by loki_agent.acp.Front.  stdin/stdout are a socketpair with the
front process; stderr is inherited (ACP log channel).
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
import getopt
import os
import sys

from . import credential_capabilities, credential_runtimes, host_ipc
from . import runtime_isolation
from .windows_api import WindowsApiError
from .process_protections import (
    ProcessProtectionError,
    protect_credential_process,
)
from .runtime_isolation import RuntimeIsolationError


async def amain(owner_fd: int, capability_fd: int, write) -> int:
    runtime = None
    try:
        runtime = await credential_runtimes.CredentialRuntime.connect(
            owner_fd, capability_fd)
    except (
            credential_capabilities.CapabilityError,
            OSError,
    ) as error:
        print(
            f"Configuration error: credential capability: {error}",
            file=sys.stderr,
        )
        return 2
    if runtime is None:
        return 1

    from . import acps, loki
    from .acp_worker import Worker
    from .sessions import Session

    session = Session(shell_cwd=os.getcwd())
    # Single-session process: make it the process default so loki's
    # current_*() helpers (chat-log bookkeeping, job manager, model) all
    # resolve to this conversation.
    loki._DEFAULT_SESSION = session
    loki.CREDENTIALS = runtime.install(session)
    worker = Worker(session, write)

    async def serve():
        try:
            loki.apply_runtime_config(loki.build_config_from_env(
                credentials=loki.CREDENTIALS))
        except (loki.protocols.ProtocolError, ValueError):
            # No explicit LOKI_* connection. The model is chosen over the
            # wire; a prompt without one gets the terminal's disconnected
            # response.
            pass
        try:
            loki.configure_tool_hook_pipeline()
        except loki.tool_runtime.HookConfigurationError as error:
            print(
                f"Hook configuration error: {error!r}",
                file=sys.stderr,
            )
            return 2

        import json
        try:
            async for raw_line in acps.AsyncFdLineReader(0):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    write(acps.response(
                        None,
                        error={
                            "code": acps.PARSE_ERROR,
                            "message": str(error),
                        }))
                    continue
                if not isinstance(message, dict):
                    write(acps.response(
                        None,
                        error={"code": acps.PARSE_ERROR,
                               "message": "not a JSON object"}))
                    continue
                # Keep reading while a prompt task runs so cancellation does
                # not queue behind the operation it is meant to interrupt.
                await worker.handle(message, concurrent=True)
        finally:
            await worker.close()
        return 0

    try:
        completed, result = await runtime.run(serve())
        return result if completed else 1
    finally:
        try:
            await session.response_headers.save_on_exit()
        finally:
            await runtime.close()


def _descriptor(value: str, description: str):
    try:
        return host_ipc.child_endpoint(value)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"invalid ACP worker {description} descriptor") from error


def _runtime_descriptors(args):
    required = {"--session-owner-fd", "--credential-capability-fd"}
    if sys.platform == "win32":
        required.add("--stdout-null-handle")
    options, positional = getopt.getopt(
        args, "", [name[2:] + "=" for name in sorted(required)])
    if positional:
        raise ValueError("invalid ACP worker arguments")
    values = {}
    for name, value in options:
        if name in values:
            raise ValueError(f"duplicate ACP worker option {name}")
        values[name] = value
    if set(values) != required:
        raise ValueError("ACP workers require " + ", ".join(sorted(required)))
    null_handle = None
    if "--stdout-null-handle" in values:
        null_handle = int(values["--stdout-null-handle"])
        if null_handle <= 0:
            raise ValueError("invalid ACP worker stdout NUL handle")
    return (
        _descriptor(values["--session-owner-fd"], "session owner"),
        _descriptor(
            values["--credential-capability-fd"],
            "credential capability"),
        null_handle,
    )


def _protocol_output(null_fd):
    from . import acps

    saved_stdout = os.dup(1)
    try:
        acps.quarantine_stdout(null_fd)
        return acps.make_writer(saved_stdout)
    finally:
        # make_writer owns a separate duplicate, including on POSIX.
        os.close(saved_stdout)


def main() -> int:
    from .diagnostics import configure_logging

    args = sys.argv[2:] if sys.argv[1:2] == ["--worker"] else sys.argv[1:]
    try:
        owner_fd, capability_fd, null_handle = _runtime_descriptors(args)
        with ExitStack() as cleanup:
            null_fd = None
            if null_handle is not None:
                from . import windows_runtime
                null_fd = cleanup.enter_context(
                    windows_runtime.inherited_stdout_null(null_handle))
            # Adopt the handle before gates/logging can fail; no inherited
            # temporary copy may escape into later tool children.
            runtime_isolation.isolate_runtime()
            runtime_isolation.prepare_runtime_scratch()
            protect_credential_process()
            if not configure_logging():
                return 2
            write = _protocol_output(null_fd)
    except (ValueError, getopt.GetoptError, OSError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except (ProcessProtectionError, RuntimeIsolationError, WindowsApiError) as error:
        print(f"Security initialization error: {error}", file=sys.stderr)
        return 2
    # On Windows only fd 1's non-inheritable NUL reference remains now.
    # Quarantine precedes connection, so its failure owns no live runtime.
    return asyncio.run(amain(owner_fd, capability_fd, write))


if __name__ == "__main__":
    raise SystemExit(main())
