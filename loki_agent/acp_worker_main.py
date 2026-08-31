"""Worker entry point: single-session Loki speaking ACP on stdio.

Spawned by loki_agent.acp.Front.  stdin/stdout are a socketpair with the
front process; stderr is inherited (ACP log channel).
"""

from __future__ import annotations

import asyncio
import os
import sys

from .credentials import CredentialStore, capture_process_credentials


async def amain(credentials: CredentialStore) -> int:
    from . import acps, loki
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

    loki.CREDENTIALS = credentials
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
    async for raw_line in acps.AsyncFdLineReader(0):
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            write(acps.response(
                None, error={"code": acps.PARSE_ERROR, "message": str(error)}))
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
    await worker.close()
    return 0


def main() -> int:
    credentials = capture_process_credentials()
    return asyncio.run(amain(credentials))


if __name__ == "__main__":
    raise SystemExit(main())
