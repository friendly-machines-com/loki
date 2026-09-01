"""Worker entry point: single-session Loki speaking ACP on stdio.

Spawned by loki_agent.acp.Front.  stdin/stdout are a socketpair with the
front process; stderr is inherited (ACP log channel).
"""

from __future__ import annotations

import asyncio
import os
import sys

from .credentials import (
    CredentialInventory,
    CredentialStore,
    capture_process_credentials,
)
from .process_protections import (
    ProcessProtectionError,
    protect_credential_process,
)


async def amain(
        credentials: CredentialStore,
        credential_capability_fd: int | None = None) -> int:
    from . import acps, credential_capabilities, loki
    from .acp_worker import Worker
    from .sessions import Session

    session = Session(shell_cwd=os.getcwd())
    # Single-session process: make it the process default so loki's
    # current_*() helpers (chat-log bookkeeping, job manager, model) all
    # resolve to this conversation.
    loki._DEFAULT_SESSION = session
    saved_stdout = os.dup(1)
    acps.quarantine_stdout()
    write = acps.make_writer(saved_stdout)
    worker = Worker(session, write)

    credential_client = None
    if credential_capability_fd is None:
        loki.install_root_credential_broker(credentials, session)
        loki.CREDENTIALS = credentials.inventory()
    else:
        credential_client = (
            await credential_capabilities.CredentialClient.from_fd(
                credential_capability_fd))
        session.credential_authority = credential_client
        loki.CREDENTIALS = CredentialInventory.from_environment(
            os.environ, credential_client.available())
    try:
        loki.apply_runtime_config(loki.build_config_from_env(
            credentials=loki.CREDENTIALS))
    except (loki.protocols.ProtocolError, ValueError):
        # No explicit LOKI_* connection.  The model is chosen over the
        # wire (session/set_config_option); a prompt without one gets the
        # same "No model selected" answer the terminal gives.
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
            # session/prompt runs as a task so the read loop keeps consuming;
            # otherwise a session/cancel arriving mid-turn would queue behind
            # the prompt it is meant to interrupt.
            await worker.handle(message, concurrent=True)
    finally:
        await worker.close()
        if credential_client is not None:
            await credential_client.close()
    return 0


def _credential_capability_fd(args) -> int | None:
    if not args:
        return None
    if len(args) != 2 or args[0] != "--credential-capability-fd":
        raise ValueError("invalid ACP worker arguments")
    try:
        fd = int(args[1])
        if fd < 3:
            raise ValueError()
        os.fstat(fd)
    except (OSError, ValueError) as error:
        raise ValueError(
            "invalid ACP worker credential capability descriptor") from error
    return fd


def main() -> int:
    credentials = capture_process_credentials()
    try:
        protect_credential_process()
    except ProcessProtectionError as error:
        print(f"Security initialization error: {error}", file=sys.stderr)
        return 2
    args = sys.argv[2:] if sys.argv[1:2] == ["--worker"] else sys.argv[1:]
    try:
        credential_fd = _credential_capability_fd(args)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    return asyncio.run(amain(credentials, credential_fd))


if __name__ == "__main__":
    raise SystemExit(main())
