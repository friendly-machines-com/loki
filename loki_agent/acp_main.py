"""Front process entry point: the process a client (Zed) spawns.

Runs the Front dispatcher on real stdin/stdout.  stderr is inherited and
is ACP's log channel.
"""

from __future__ import annotations

import asyncio
import os
import sys

from . import runtime_isolation
from .credentials import CredentialStore, capture_process_credentials
from .runtime_isolation import RuntimeIsolationError
from .windows_api import WindowsApiError
from .diagnostics import configure_logging
from .lifetimes import finish_cleanup
from .process_protections import (
    ProcessProtectionError,
    protect_credential_process,
)


async def amain(credentials: CredentialStore) -> int:
    from . import acps
    from .acp import Front
    from .credential_storages import JsonCredentialStorage

    saved_stdout = os.dup(1)
    try:
        acps.quarantine_stdout()
        write = acps.make_writer(saved_stdout)
    finally:
        os.close(saved_stdout)
    reader = acps.AsyncFdLineReader(0)

    async def read():
        async for raw_line in reader:
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

    storage = JsonCredentialStorage()
    try:
        front = Front(read, write, credentials, storage)
        await front.run()
        return 0
    finally:
        try:
            await finish_cleanup(reader.aclose(), preserve_failure=sys.exc_info()[0] is not None)
        finally:
            storage.close()


def main() -> int:
    if sys.argv[1:2] == ["--worker"]:
        from .acp_worker_main import main as worker_main
        return worker_main()

    if sys.argv[1:2] == ["--subagent"]:
        try:
            # Process hardening alone does not prove AppContainer containment.
            # The central re-check is a POSIX no-op: Linux inherits the worker's
            # credential cover and must not create another namespace level.
            runtime_isolation.verify_contained_runtime()
            protect_credential_process()
        except (ProcessProtectionError, RuntimeIsolationError, WindowsApiError) as error:
            print(
                f"Security initialization error: {error}",
                file=sys.stderr,
            )
            return 2
        if not configure_logging(contained=True):
            return 2
        from .subagents import main as subagent_main
        return subagent_main(sys.argv[2:])

    credentials = capture_process_credentials()
    try:
        protect_credential_process()
    except ProcessProtectionError as error:
        print(f"Security initialization error: {error}", file=sys.stderr)
        return 2
    if not configure_logging():
        return 2
    from .credential_storages import CredentialStorageError
    try:
        return asyncio.run(amain(credentials))
    except CredentialStorageError as error:
        print(f"Credential storage error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
