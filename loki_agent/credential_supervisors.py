"""Root credential ownership and delegation to isolated Loki runtimes.

Every public Loki entrypoint follows the same authority model:

* a supervisor captures environment secrets and owns the root broker;
* each model/tool runtime receives a sanitized environment, an owner-lifetime
  pipe, and one anonymous credential capability; and
* runtimes may relay narrower capabilities to subagents, but can never create
  a root broker or obtain refresh tokens.

The owner pipe and credential socket are intentionally distinct.  Closing the
socket revokes authentication, while closing the owner pipe revokes the
runtime itself even if it is not currently requesting a credential.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass

from . import credential_capabilities
from .authentications import CredentialBroker
from .credentials import CredentialStore


class CredentialSupervisor:
    """Credential-owning state shared by terminal and ACP supervisors."""

    def __init__(self, credentials: CredentialStore):
        self.environment = credentials.sanitized_environment()
        self.broker = CredentialBroker()
        credentials.install_static_credentials(self.broker)
        self.inventory = credentials.inventory()

    async def delegate(self, allowed=None) -> "RuntimeDelegation":
        return await RuntimeDelegation.create(self.broker, allowed)

    async def run_terminal_runtime(
            self, executable: str, arguments: list[str]) -> int:
        """Run one terminal/headless child while serving its credentials."""
        delegation = await self.delegate()
        process = None
        try:
            command = [
                executable,
                "--runtime",
                *delegation.child_arguments(),
                "--",
                *arguments,
            ]
            process = await asyncio.create_subprocess_exec(
                *command,
                close_fds=True,
                pass_fds=delegation.child_fds(),
                env=self.environment,
            )
            delegation.child_spawned()
            return await process.wait()
        finally:
            # Revoke the runtime before waiting for it. In particular, the
            # terminal child observes owner EOF and gets a chance to restore
            # raw tty state itself. Sending SIGTERM immediately would race
            # that cleanup and could leave the caller's terminal damaged.
            delegation.revoke_now()
            if process is not None and process.returncode is None:
                wait_task = asyncio.create_task(
                    process.wait(), name="loki-runtime-shutdown")
                try:
                    await asyncio.wait_for(
                        asyncio.shield(wait_task), timeout=2)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.terminate()
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(wait_task), timeout=2)
                    except asyncio.TimeoutError:
                        with contextlib.suppress(ProcessLookupError):
                            process.kill()
                        await wait_task
            # Awaited capability cleanup serializes refresh cancellation with
            # the supervisor's lifecycle transition.
            await delegation.close()


@dataclass
class RuntimeDelegation:
    """Parent-owned lifetime and credential channels for one runtime."""

    credential_server: object
    owner_read_fd: int | None
    owner_write_fd: int | None
    credential_fd: int | None

    @classmethod
    async def create(cls, authority, allowed=None):
        owner_read_fd, owner_write_fd = os.pipe()
        credential_server = None
        credential_fd = None
        try:
            credential_server, credential_fd = await (
                credential_capabilities.CredentialCapabilityServer.create(
                    authority, allowed))
            for fd in (owner_read_fd, owner_write_fd, credential_fd):
                os.set_inheritable(fd, False)
            return cls(
                credential_server,
                owner_read_fd,
                owner_write_fd,
                credential_fd,
            )
        except BaseException:
            for fd in (owner_read_fd, owner_write_fd, credential_fd):
                if fd is not None:
                    with contextlib.suppress(OSError):
                        os.close(fd)
            if credential_server is not None:
                await credential_server.close()
            raise

    def child_arguments(self) -> list[str]:
        if self.owner_read_fd is None or self.credential_fd is None:
            raise RuntimeError("runtime delegation was already handed off")
        return [
            "--session-owner-fd", str(self.owner_read_fd),
            "--credential-capability-fd", str(self.credential_fd),
        ]

    def child_fds(self) -> tuple[int, int]:
        if self.owner_read_fd is None or self.credential_fd is None:
            raise RuntimeError("runtime delegation was already handed off")
        return self.owner_read_fd, self.credential_fd

    def child_spawned(self) -> None:
        """Close the supervisor's copies of descriptors owned by the child."""
        for attribute in ("owner_read_fd", "credential_fd"):
            fd = getattr(self, attribute)
            setattr(self, attribute, None)
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)

    def revoke_now(self) -> None:
        """Synchronously revoke runtime lifetime and credential authority."""
        owner_write_fd = self.owner_write_fd
        self.owner_write_fd = None
        if owner_write_fd is not None:
            with contextlib.suppress(OSError):
                os.close(owner_write_fd)
        self.credential_server.close_now()

    async def close(self) -> None:
        # Each cleanup action is independently necessary. A transport error
        # must not retain the child-owned descriptor copies or the owner pipe.
        try:
            self.revoke_now()
        finally:
            self.child_spawned()
            await self.credential_server.close()
