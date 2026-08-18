"""Worker entry point: single-session Loki speaking ACP on stdio.

Spawned by loki_agent.acp.Front.  stdin/stdout are a socketpair with the
front process; stderr is inherited (ACP log channel).
"""

from __future__ import annotations

import asyncio
import os
import sys

from . import acps, loki
from .acp_worker import Worker
from .sessions import Session


async def amain() -> int:
    session = Session(shell_cwd=os.getcwd())
    # Single-session process: make it the process default so loki's
    # current_*() helpers (chat-log bookkeeping, job manager, model) all
    # resolve to this conversation.
    loki._DEFAULT_SESSION = session
    saved_stdout = os.dup(1)
    acps.quarantine_stdout()
    write = acps.make_writer(saved_stdout)
    worker = Worker(session, write)

    loki.CREDENTIALS = loki.CredentialStore.capture(os.environ)
    try:
        loki.apply_runtime_config(loki.build_config_from_env(
            credentials=loki.CREDENTIALS))
    except (loki.protocols.ProtocolError, ValueError) as error:
        # No provider: the session still works; prompts answer with a
        # model-selection notice instead of a hollow turn.
        print(f"ACP worker without provider: {error}", file=sys.stderr)
    try:
        loki.configure_tool_hook_pipeline()
    except loki.tool_runtime.HookConfigurationError as error:
        print(f"Hook configuration error: {error}", file=sys.stderr)
        return 2

    import json
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            write(acps.response(
                None, error={"code": acps.PARSE_ERROR, "message": str(error)}))
            continue
        if not isinstance(message, dict):
            write(acps.response(
                None,
                error={"code": acps.PARSE_ERROR,
                       "message": "not a JSON object"}))
            continue
        await worker.handle(message)
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
