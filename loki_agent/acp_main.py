"""Front process entry point: the process a client (Zed) spawns.

Runs the Front dispatcher on real stdin/stdout.  stderr is inherited and
is ACP's log channel.
"""

from __future__ import annotations

import asyncio
import os
import sys

from .credentials import CredentialStore, capture_process_credentials
from .process_protections import (
    ProcessProtectionError,
    protect_credential_process,
)


async def amain(credentials: CredentialStore) -> int:
    from . import acps
    from .acp import Front

    saved_stdout = os.dup(1)
    acps.quarantine_stdout()
    write = acps.make_writer(saved_stdout)

    async def read():
        async for raw_line in acps.AsyncFdLineReader(0):
            line = raw_line.strip()
            if not line:
                continue
            try:
                import json
                message = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                write(acps.response(
                    None, error={"code": acps.PARSE_ERROR,
                                 "message": str(error)}))
                continue
            if not isinstance(message, dict):
                write(acps.response(
                    None, error={"code": acps.PARSE_ERROR,
                                 "message": "not a JSON object"}))
                continue
            yield message

    front = Front(read, write, credentials)
    await front.run()
    return 0


def main() -> int:
    if sys.argv[1:2] == ["--worker"]:
        from .acp_worker_main import main as worker_main
        return worker_main()

    if sys.argv[1:2] == ["--subagent"]:
        try:
            # ACP subagents inherit the worker's already-covered mount
            # namespace. Re-unsharing would add a namespace level for no
            # security gain and can hit kernel nesting or policy limits.
            protect_credential_process()
        except ProcessProtectionError as error:
            print(
                f"Security initialization error: {error}",
                file=sys.stderr,
            )
            return 2
        from .subagents import main as subagent_main
        return subagent_main(sys.argv[2:])

    credentials = capture_process_credentials()
    try:
        protect_credential_process()
    except ProcessProtectionError as error:
        print(f"Security initialization error: {error}", file=sys.stderr)
        return 2
    return asyncio.run(amain(credentials))


if __name__ == "__main__":
    raise SystemExit(main())
