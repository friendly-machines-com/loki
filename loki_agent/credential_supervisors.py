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

from . import credential_capabilities, host_ipc
from .authentications import CredentialBroker
from .credentials import CredentialInventory, CredentialStore


class CredentialSupervisor:
    """Credential-owning state shared by terminal and ACP supervisors."""

    def __init__(self, credentials: CredentialStore, storage=None):
        self.environment = credentials.sanitized_environment()
        self.broker = CredentialBroker()
        credentials.install_static_credentials(self.broker)
        self.storage = storage
        if storage is not None:
            # This must happen before a runtime starts.  Even an empty
            # credential directory has to exist when Linux establishes the
            # runtime's cover mount; creating it after that point would expose
            # a later login file in an already-running runtime.
            storage.ensure_directory()
            stored = storage.load_openai_subscription()
            if (stored is not None
                    and stored.state == "active"
                    and stored.tokens is not None):
                self.broker.install_openai_subscription(
                    stored.tokens,
                    rotate=storage.rotate_openai_subscription,
                )
        self.inventory = CredentialInventory(
            self.environment,
            self.broker.available(),
        )

    async def delegate(self, allowed=None) -> "RuntimeDelegation":
        return await RuntimeDelegation.create(self.broker, allowed)

    async def run_terminal_runtime(
            self, executable: str, arguments: list[str]) -> int:
        """Run one terminal/headless child while serving its credentials."""
        workspace = None
        if os.name == "nt":
            from . import windows_runtime
            workspace = windows_runtime.configured_workspace(arguments)
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
            if workspace is not None:
                process = windows_runtime.launch(
                    executable, command[1:], self.environment, workspace,
                    [host_ipc.reference(delegation.owner_child),
                     host_ipc.reference(delegation.credential_child)])
            else:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    close_fds=True,
                    env=self.environment,
                    **delegation.child_spawn_kwargs(),
                )
            delegation.child_spawned()
            return await process.wait()
        finally:
            # Revoke the runtime before waiting for it. In particular, the
            # terminal child observes owner EOF and gets a chance to restore
            # raw tty state itself. Sending SIGTERM immediately would race
            # that cleanup and could leave the caller's terminal damaged.
            try:
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
            finally:
                # Cleanup must also run when waiting or termination fails.
                try:
                    await delegation.close()
                finally:
                    if workspace is not None and process is not None:
                        process.close()


@dataclass
class RuntimeDelegation:
    """Parent-owned lifetime and credential channels for one runtime."""

    credential_server: object
    owner_child: object | None
    owner_parent: object | None
    credential_child: object | None

    @classmethod
    async def create(cls, authority, allowed=None):
        owner_parent, owner_child = host_ipc.owner_channel()
        credential_server = None
        credential_child = None
        try:
            credential_server, credential_child = await (
                credential_capabilities.CredentialCapabilityServer.create(
                    authority, allowed))
            return cls(
                credential_server,
                owner_child,
                owner_parent,
                credential_child,
            )
        except BaseException:
            for end in (owner_parent, owner_child, credential_child):
                if end is not None:
                    with contextlib.suppress(OSError):
                        host_ipc.close_end(end)
            if credential_server is not None:
                await credential_server.close()
            raise

    def child_arguments(self) -> list[str]:
        if self.owner_child is None or self.credential_child is None:
            raise RuntimeError("runtime delegation was already handed off")
        return [
            "--session-owner-fd", str(host_ipc.reference(self.owner_child)),
            "--credential-capability-fd",
            str(host_ipc.reference(self.credential_child)),
        ]

    def child_spawn_kwargs(self) -> dict:
        if self.owner_child is None or self.credential_child is None:
            raise RuntimeError("runtime delegation was already handed off")
        return host_ipc.spawn_kwargs((self.owner_child, self.credential_child))

    def child_spawned(self) -> None:
        """Close the supervisor's copies of the ends the child owns."""
        for attribute in ("owner_child", "credential_child"):
            end = getattr(self, attribute)
            setattr(self, attribute, None)
            if end is not None:
                with contextlib.suppress(OSError):
                    host_ipc.close_end(end)

    def revoke_now(self) -> None:
        """Synchronously revoke runtime lifetime and credential authority."""
        owner_parent = self.owner_parent
        self.owner_parent = None
        if owner_parent is not None:
            with contextlib.suppress(OSError):
                host_ipc.close_end(owner_parent)
        self.credential_server.close_now()

    async def close(self) -> None:
        # Each cleanup action is independently necessary. A transport error
        # must not retain the child-owned descriptor copies or the owner pipe.
        try:
            self.revoke_now()
        finally:
            self.child_spawned()
            await self.credential_server.close()
