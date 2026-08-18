"""Front process entry point: the process a client (Zed) spawns.

Runs the Front dispatcher on real stdin/stdout.  stderr is inherited and
is ACP's log channel.
"""

from __future__ import annotations

import asyncio
import os
import sys

from . import acps
from .acp import Front


async def amain() -> int:
    saved_stdout = os.dup(1)
    acps.quarantine_stdout()
    write = acps.make_writer(saved_stdout)
    # terminals.py closes sys.stdin at import when stdin is a tty (the
    # terminal UI reopens fd 0 as /dev/tty for its async reader).  The
    # ACP front never uses that UI but inherits the closed object, so
    # read fd 0 directly: it is the client's pipe, or /dev/tty when a
    # human is typing test lines into an interactive run.
    stdin = os.fdopen(0, "r", encoding="utf-8", errors="replace")

    async def read():
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, stdin.readline)
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                import json
                message = json.loads(line)
            except json.JSONDecodeError as error:
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

    front = Front(read, write)
    await front.run()
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
