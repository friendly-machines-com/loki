"""Child-side installation and lifetime of delegated credentials.

A runtime is valid only while both capabilities supplied by its supervisor
remain live.  Owner loss cancels the complete runtime, including credential
initialization; broker loss cancels an active model/tool operation.  This
module centralizes those races so terminal, headless, ACP, and subagent
runtimes cannot accidentally acquire different lifetime semantics.
"""

from __future__ import annotations

import asyncio
import os

from . import credential_capabilities, host_ipc
from .credentials import CredentialInventory
from .lifetimes import finish_cleanup


class SessionOwner:
    """Capture one inherited end before its canonical watcher can first run.

    host_ipc owns endpoint dispatch and close-only observation. This runtime
    owner retains the original end and releases it only after watcher retirement.
    """

    def __init__(self, owner):
        self.endpoint = owner
        self.closed_task = asyncio.create_task(
            host_ipc.watch_closed(owner), name="loki-session-owner")
        self._closed = False

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if not self.closed_task.done():
            self.closed_task.cancel()
        try:
            await asyncio.gather(
                self.closed_task, return_exceptions=True)
        finally:
            # gather joins the watcher even when this closer is cancelled.
            # Cancellation must not skip physical closure: _closed has already
            # disabled a later retry of this inherited endpoint's cleanup.
            host_ipc.close_end(self.endpoint)


async def _connect_while_owned(capability_fd: int, owner: SessionOwner):
    """Initialize a credential client only while its supervisor is live."""
    # take_fd() transfers FD ownership synchronously. If owner revocation wins
    # before the initialization task receives a timeslice, cancellation must
    # still close the inherited capability.
    initializer = (
        credential_capabilities.CredentialClient.take_fd(capability_fd))
    client_task = asyncio.create_task(
        initializer.connect(),
        name="loki-credential-initialization",
    )

    async def retire_client():
        # A completed initialization can own a live client, not just an endpoint.
        # Until publication this scope must retire either result; joining the
        # initialization task alone does not revoke the returned capability.
        try:
            result = (await asyncio.gather(
                client_task, return_exceptions=True))[0]
        finally:
            initializer.close_now()
        if isinstance(result, credential_capabilities.CredentialClient):
            await result.close()

    try:
        done, _pending = await asyncio.wait(
            {owner.closed_task, client_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        client_task.cancel()
        await finish_cleanup(retire_client(), preserve_failure=True)
        raise
    if owner.closed_task not in done:
        return await client_task

    # Owner closure wins a simultaneous handshake completion. The credential
    # channel has no independent lifetime outside the delegating supervisor.
    client_task.cancel()
    await finish_cleanup(retire_client())
    return None


class CredentialRuntime:
    """A capability-only runtime; this class can never create a root broker."""

    def __init__(self, owner: SessionOwner, credential_client):
        self.owner = owner
        self.credential_client = credential_client

    @classmethod
    async def connect(cls, owner_fd: int, capability_fd: int):
        owner = SessionOwner(owner_fd)
        try:
            client = await _connect_while_owned(
                capability_fd, owner)
        except BaseException:
            await owner.close()
            raise
        if client is None:
            await owner.close()
            return None
        return cls(owner, client)

    def install(self, session) -> CredentialInventory:
        """Install delegated authority and return its non-secret inventory."""
        session.credential_authority = self.credential_client
        return CredentialInventory.from_environment(
            os.environ, self.credential_client.available())

    async def run(self, coroutine):
        """Return ``(completed, result)`` while enforcing both lifetimes."""
        if self.owner.closed_task.done() or self.credential_client.closed:
            # Do not start user/model work after an already-observed
            # revocation. Native coroutine objects must be closed explicitly
            # because they will never be wrapped in a task and awaited.
            close = getattr(coroutine, "close", None)
            if close is not None:
                close()
            return False, None
        operation_task = asyncio.create_task(
            coroutine, name="loki-session-owned-operation")
        capability_task = asyncio.create_task(
            self.credential_client.wait_closed(),
            name="loki-credential-owner",
        )
        completed = False
        try:
            done, _pending = await asyncio.wait(
                {
                    self.owner.closed_task,
                    capability_task,
                    operation_task,
                },
                return_when=asyncio.FIRST_COMPLETED,
            )
            if (
                    self.owner.closed_task in done
                    or capability_task in done
            ):
                # Revocation wins if it lands in the same event-loop turn as
                # normal completion. Authority has no useful "last instant"
                # after its owner is already known to be gone.
                self.credential_client.close_now()
                return False, None
            result = await operation_task
            completed = True
            return completed, result
        finally:
            async def retired():
                try:
                    await asyncio.gather(operation_task, return_exceptions=True)
                finally:
                    if not capability_task.done():
                        capability_task.cancel()
                    await asyncio.gather(capability_task, return_exceptions=True)

            try:
                # Revoke before cancellation can be consumed by a lower loop's
                # finally block. Closing hooks cannot start fresh work afterward.
                if not operation_task.done():
                    try:
                        if not self.credential_client.closed:
                            self.credential_client.close_now()
                    finally:
                        operation_task.cancel()
            finally:
                await finish_cleanup(retired(), preserve_failure=not completed)

    async def close(self):
        # Owner closure is mandatory even if transport shutdown reports an
        # error; otherwise the inherited descriptor can keep a delegated
        # process lifetime accidentally alive.
        try:
            await self.credential_client.close()
        finally:
            await self.owner.close()
