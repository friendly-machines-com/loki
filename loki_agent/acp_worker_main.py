"""Worker entry point: single-session Loki speaking ACP on stdio.

Spawned by loki_agent.acp.Front.  stdin/stdout are a socketpair with the
front process; stderr is inherited (ACP log channel).
"""

from __future__ import annotations

import asyncio
import getopt
import os
import sys

from . import credential_capabilities, credential_runtimes
from .process_protections import (
    ProcessProtectionError,
    protect_credential_process,
)
from .runtime_isolations import (
    RuntimeIsolationError,
    isolate_credential_directory,
)


async def amain(owner_fd: int, capability_fd: int) -> int:
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
    saved_stdout = os.dup(1)
    acps.quarantine_stdout()
    write = acps.make_writer(saved_stdout)
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
        await runtime.close()


def _descriptor(value: str, description: str) -> int:
    try:
        fd = int(value)
        if fd < 3:
            raise ValueError()
        os.fstat(fd)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"invalid ACP worker {description} descriptor") from error
    return fd


def _runtime_descriptors(args) -> tuple[int, int]:
    options, positional = getopt.getopt(
        args, "", [
            "session-owner-fd=",
            "credential-capability-fd=",
        ])
    if positional:
        raise ValueError("invalid ACP worker arguments")
    values = {}
    for name, value in options:
        if name in values:
            raise ValueError(f"duplicate ACP worker option {name}")
        values[name] = value
    if set(values) != {
            "--session-owner-fd",
            "--credential-capability-fd",
    }:
        raise ValueError(
            "ACP workers require session owner and credential capability "
            "descriptors")
    return (
        _descriptor(values["--session-owner-fd"], "session owner"),
        _descriptor(
            values["--credential-capability-fd"],
            "credential capability"),
    )


def main() -> int:
    args = sys.argv[2:] if sys.argv[1:2] == ["--worker"] else sys.argv[1:]
    try:
        owner_fd, capability_fd = _runtime_descriptors(args)
        # This is the worker's earliest trusted startup phase. Establish its
        # filesystem view before importing the agent runtime, then make the
        # final credential-consuming process non-dumpable.
        isolate_credential_directory()
        protect_credential_process()
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except (ProcessProtectionError, RuntimeIsolationError) as error:
        print(f"Security initialization error: {error}", file=sys.stderr)
        return 2
    return asyncio.run(amain(owner_fd, capability_fd))


if __name__ == "__main__":
    raise SystemExit(main())
