"""Front process entry point: the process a client (Zed) spawns.

Runs the Front dispatcher on real stdin/stdout.  stderr is inherited and
is ACP's log channel.
"""

from __future__ import annotations

import asyncio
import os

from .credentials import CredentialStore, capture_process_credentials


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
    credentials = capture_process_credentials()
    return asyncio.run(amain(credentials))


if __name__ == "__main__":
    raise SystemExit(main())
