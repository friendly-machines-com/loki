"""Subprocess helper for cross-process credential-rotation tests."""

import asyncio
import os
import sys

from loki_agent import authentications
from loki_agent import credential_storages


async def run(directory, calls_path, release_path):
    storage = credential_storages.JsonCredentialStorage(directory)
    current = storage.load_openai_subscription().tokens

    async def refresh(_refresh_token):
        descriptor = os.open(
            calls_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        while not os.path.exists(release_path):
            await asyncio.sleep(0.01)
        return authentications.RefreshResult(
            access_token="access-b",
            refresh_token="refresh-b",
        )

    result = await storage.rotate_openai_subscription(
        current, refresh=refresh, clock=lambda: 200)
    print(result.refresh_token)


if __name__ == "__main__":
    asyncio.run(run(*sys.argv[1:]))
