"""Child-side installation and lifetime of delegated credentials.

A runtime is valid only while both capabilities supplied by its supervisor
remain live.  Owner loss cancels the complete runtime, including credential
initialization; broker loss cancels an active model/tool operation.  This
module centralizes those races so terminal, headless, ACP, and subagent
runtimes cannot accidentally acquire different lifetime semantics.
"""

from __future__ import annotations

import asyncio
import contextlib
import os

from . import acps, credential_capabilities
from .credentials import CredentialInventory


class SessionOwner:
    """One inherited session-lifetime descriptor."""

    def __init__(self, fd: int):
        os.set_inheritable(fd, False)
        self.fd = fd
        self.closed_task = asyncio.create_task(
            acps.AsyncFdLineReader(fd).readline(),
            name="loki-session-owner",
        )
        self._closed = False

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if not self.closed_task.done():
            self.closed_task.cancel()
        await asyncio.gather(
            self.closed_task, return_exceptions=True)
        with contextlib.suppress(OSError):
            os.close(self.fd)


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
    try:
        done, _pending = await asyncio.wait(
            {owner.closed_task, client_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        if not client_task.done():
            client_task.cancel()
        await asyncio.gather(client_task, return_exceptions=True)
        initializer.close_now()
        raise
    if owner.closed_task not in done:
        return await client_task

    # Owner closure wins a simultaneous handshake completion. The credential
    # channel has no independent lifetime outside the delegating supervisor.
    if not client_task.done():
        client_task.cancel()
    result = (await asyncio.gather(
        client_task, return_exceptions=True))[0]
    initializer.close_now()
    if isinstance(
            result, credential_capabilities.CredentialClient):
        await result.close()
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
        if self.owner.closed_task.done():
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
                if not operation_task.done():
                    operation_task.cancel()
                await asyncio.gather(
                    operation_task, return_exceptions=True)
                return False, None
            return True, await operation_task
        finally:
            # Cancellation of run() itself must not detach the operation from
            # the lifetime monitor. This matters during local shutdown even
            # when neither inherited channel has reached EOF yet.
            if not operation_task.done():
                operation_task.cancel()
            await asyncio.gather(
                operation_task, return_exceptions=True)
            if not capability_task.done():
                capability_task.cancel()
            await asyncio.gather(
                capability_task, return_exceptions=True)

    async def close(self):
        # Owner closure is mandatory even if transport shutdown reports an
        # error; otherwise the inherited descriptor can keep a delegated
        # process lifetime accidentally alive.
        try:
            await self.credential_client.close()
        finally:
            await self.owner.close()
