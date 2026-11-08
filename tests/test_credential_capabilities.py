import asyncio
import os
import socket
import sys
import tempfile
import unittest
from unittest import mock

from process_lifecycle_fixtures import observe_fixture
from loki_agent import authentications
from loki_agent import credential_capabilities
from loki_agent import credential_runtimes
from loki_agent import credential_supervisors
from loki_agent import host_ipc


DESTINATION = "https://example.test/inference"


CHILD_SCRIPT = """
import asyncio, sys
from loki_agent import authentications, credential_capabilities
from loki_agent import credential_runtimes, host_ipc


def argument(flag):
    return sys.argv[sys.argv.index(flag) + 1]


async def main():
    owner = credential_runtimes.SessionOwner(
        host_ipc.child_endpoint(argument("--session-owner-fd")))
    client = await credential_capabilities.CredentialClient.from_fd(
        host_ipc.child_endpoint(argument("--credential-capability-fd")))
    credential = authentications.CredentialRef.decode(
        argument("--credential"))
    lease = await client.lease(credential, destination="https://example.test/inference")
    print("lease:" + lease.value, flush=True)
    await owner.closed_task
    print("revoked", flush=True)
    await client.close()


asyncio.run(main())
"""


class CredentialCapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.broker = authentications.CredentialBroker()
        self.first = authentications.CredentialRef.environment(
            "FIRST_API_KEY")
        self.second = authentications.CredentialRef.environment(
            "SECOND_API_KEY")
        self.broker.install_static(self.first, "first-secret")
        self.broker.install_static(self.second, "second-secret")
        await self.broker.approve_destinations(self.first, [DESTINATION])
        await self.broker.approve_destinations(self.second, [DESTINATION])
        self.server = None
        self.client = None

    async def asyncTearDown(self):
        if self.client is not None:
            await self.client.close()
        if self.server is not None:
            await self.server.close()

    async def connect(self, allowed):
        self.server, child_fd = (
            await credential_capabilities.CredentialCapabilityServer.create(
                self.broker, allowed))
        # POSIX: the parent must not leave its copy of the child's end
        # inheritable.  Windows: the child end is a pipe endpoint whose handles
        # must be inheritable to cross at all; there the explicit handle list,
        # not this flag, bounds what else may inherit.
        if os.name == "posix":
            self.assertFalse(os.get_inheritable(child_fd))
        else:
            self.assertTrue(host_ipc.is_endpoint(child_fd))
            self.assertTrue(all(
                os.get_handle_inheritable(handle)
                for handle in child_fd.handles()))
        self.client = (
            await credential_capabilities.CredentialClient.from_fd(child_fd))

    async def test_delegates_only_allowed_credentials(self):
        await self.connect({self.first})

        lease = await self.client.lease(self.first, destination=DESTINATION)

        self.assertEqual(lease.value, "first-secret")
        self.assertEqual(self.client.available(), frozenset({self.first}))
        with self.assertRaises(credential_capabilities.CapabilityError):
            await self.client.lease(self.second, destination=DESTINATION)

    async def test_capability_can_delegate_strict_subset(self):
        await self.connect({self.first, self.second})
        relay, child_fd = (
            await credential_capabilities.CredentialCapabilityServer.create(
                self.client, {self.second}))
        grandchild = (
            await credential_capabilities.CredentialClient.from_fd(child_fd))
        try:
            lease = await grandchild.lease(self.second, destination=DESTINATION)
            self.assertEqual(lease.value, "second-secret")
            with self.assertRaises(
                    credential_capabilities.CapabilityError):
                await grandchild.lease(self.first, destination=DESTINATION)
        finally:
            await grandchild.close()
            await relay.close()

    async def test_rejected_generation_refreshes_only_in_root_broker(self):
        calls = []

        async def refresh(refresh_token):
            calls.append(refresh_token)
            generation = len(calls)
            return authentications.RefreshResult(
                access_token=f"access-{generation}",
                refresh_token=f"refresh-{generation}",
            )

        async def rotate(tokens):
            return authentications.refreshed_openai_tokens(
                tokens, await refresh(tokens.refresh_token))

        credential = (
            authentications.CredentialRef.openai_subscription())
        self.broker.install_openai_subscription(
            authentications.OpenAITokenSet(
                access_token="access-old",
                refresh_token="refresh-old",
                expires_at=10**12,
            ),
            rotate=rotate,
        )
        await self.connect({credential})

        first = await self.client.lease(credential, destination=authentications.OPENAI_CHATGPT_RESPONSES_URL)
        second = await self.client.lease(
            credential, destination=authentications.OPENAI_CHATGPT_RESPONSES_URL, rejected_generation=first.generation)

        self.assertEqual(first.value, "access-old")
        self.assertEqual(second.value, "access-1")
        self.assertEqual(first.generation, 0)
        self.assertEqual(second.generation, 1)
        self.assertEqual(calls, ["refresh-old"])
        self.assertFalse(hasattr(second, "refresh_token"))
        third = await self.client.lease(
            credential, destination=authentications.OPENAI_CHATGPT_RESPONSES_URL, rejected_generation=second.generation)
        self.assertEqual(third.value, "access-2")
        self.assertEqual(third.generation, 2)
        self.assertEqual(calls, ["refresh-old", "refresh-1"])
        # A late rejection of generation zero adopts generation two; it must
        # not refresh again or roll the client back to an earlier token.
        stale = await self.client.lease(
            credential, destination=authentications.OPENAI_CHATGPT_RESPONSES_URL, rejected_generation=first.generation)
        self.assertEqual((stale.credential, stale.value, stale.generation),
                         (credential, "access-2", 2))
        self.assertEqual(calls, ["refresh-old", "refresh-1"])
        self.assertFalse(hasattr(third, "refresh_token"))

    async def test_concurrent_requests_are_multiplexed(self):
        await self.connect({self.first, self.second})

        leases = await asyncio.gather(*[
            self.client.lease(
                self.first if index % 2 == 0 else self.second, destination=DESTINATION)
            for index in range(20)
        ])

        self.assertEqual(
            [(lease.credential, lease.value, lease.generation)
             for lease in leases],
            [(self.first, "first-secret", 0) if index % 2 == 0
             else (self.second, "second-secret", 0)
             for index in range(20)])

    async def test_server_serializes_requests_for_bounded_backpressure(self):
        active = 0
        maximum = 0

        class MeasuringAuthority:
            def available(inner_self):
                return frozenset({self.first})

            async def lease(
                    inner_self, credential, destination,
                    rejected_generation=None):
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                try:
                    await asyncio.sleep(0)
                    return authentications.CredentialLease(
                        credential, "secret")
                finally:
                    active -= 1

        self.server, child_fd = (
            await credential_capabilities.CredentialCapabilityServer.create(
                MeasuringAuthority(), {self.first}))
        self.client = (
            await credential_capabilities.CredentialClient.from_fd(child_fd))

        leases = await asyncio.gather(*[
            self.client.lease(self.first, destination=DESTINATION) for _ in range(20)
        ])

        self.assertEqual(maximum, 1)
        self.assertEqual(
            [lease.value for lease in leases], ["secret"] * 20)

    async def test_capability_revocation_fails_live_pending_and_future_requests(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        requests = []

        class BlockingAuthority:
            def available(inner_self):
                return frozenset({self.first})

            async def lease(
                    inner_self, credential, destination,
                    rejected_generation=None):
                requests.append(credential)
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        self.server, child_fd = (
            await credential_capabilities.CredentialCapabilityServer.create(
                BlockingAuthority(), {self.first}))
        self.client = (
            await credential_capabilities.CredentialClient.from_fd(child_fd))
        pending = asyncio.create_task(self.client.lease(self.first, destination=DESTINATION))
        try:
            await observe_fixture(started.wait(), pending, self.server._reader_task)
            self.assertFalse(pending.done())
            await self.server.close()
            with self.assertRaisesRegex(
                    credential_capabilities.CapabilityError, "closed"):
                await pending
            self.assertTrue(cancelled.is_set())
            self.assertTrue(self.server._reader_task.done())
            # Client is still present; denial cannot be supplied by voluntarily
            # shutting it down as the cooperative owner-lifetime child does.
            with self.assertRaisesRegex(
                    credential_capabilities.CapabilityError, "closed"):
                await self.client.lease(self.first, destination=DESTINATION)
            self.assertEqual(requests, [self.first])
            self.assertEqual(self.client._pending, {})
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    async def test_awaited_server_close_waits_for_transport_cleanup(self):
        await self.connect({self.first})
        wait_closed = mock.AsyncMock(
            wraps=self.server._writer.wait_closed)

        with mock.patch.object(
                self.server._writer,
                "wait_closed",
                new=wait_closed):
            await self.server.close()

        wait_closed.assert_awaited()

    async def test_cancelled_runtime_connection_closes_a_completed_client(self):
        self.server, capability = await credential_capabilities.CredentialCapabilityServer.create(
            self.broker, {self.first})
        parent, child = host_ipc.owner_channel()
        owner = credential_runtimes.SessionOwner(child)
        from_streams = credential_capabilities.CredentialClient._from_streams

        async def connected(reader, writer):
            self.client = await from_streams(reader, writer)
            # The real handshake completes, but cancellation wins before its
            # result is transferred to the runtime caller.
            asyncio.current_task().add_done_callback(lambda _task: pending.cancel())
            return self.client

        with mock.patch.object(credential_capabilities.CredentialClient, '_from_streams', new=connected):
            pending = asyncio.create_task(credential_runtimes._connect_while_owned(capability, owner))
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                self.assertIsNotNone(self.client)
                with self.assertRaises(credential_capabilities.CapabilityError):
                    await self.client.lease(self.first, destination=DESTINATION)
                self.assertTrue(self.client.closed)
                self.assertTrue(self.client._reader_task.done())
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                await owner.close()
                host_ipc.close_end(parent)

    def test_wire_messages_are_bounded_and_must_be_objects(self):
        with self.assertRaises(
                credential_capabilities.CapabilityError):
            credential_capabilities._encode({
                "value": "x" * (
                    credential_capabilities.CAPABILITY_MAX_MESSAGE_BYTES),
            })
        with self.assertRaises(
                credential_capabilities.CapabilityError):
            credential_capabilities._decode(b"[]")

    async def test_invalid_delegated_descriptor_is_refused(self):
        read_fd, write_fd = os.pipe()
        try:
            with self.assertRaises(OSError):
                await credential_capabilities.CredentialClient.from_fd(
                    read_fd)
        finally:
            os.close(write_fd)

    async def test_cannot_delegate_more_than_upstream(self):
        await self.connect({self.first})

        with self.assertRaises(ValueError):
            await credential_capabilities.CredentialCapabilityServer.create(
                self.client, {self.second})

    async def test_authority_error_text_cannot_cross_capability(self):
        secret = "refresh-token-that-must-not-cross"

        class FailingAuthority:
            def available(inner_self):
                return frozenset({self.first})

            async def lease(
                    inner_self, credential, destination,
                    rejected_generation=None):
                raise RuntimeError(secret)

        server, child_fd = (
            await credential_capabilities.CredentialCapabilityServer.create(
                FailingAuthority(), {self.first}))
        client = (
            await credential_capabilities.CredentialClient.from_fd(child_fd))
        try:
            with self.assertRaises(
                    credential_capabilities.CapabilityError) as raised:
                await client.lease(self.first, destination=DESTINATION)
            self.assertNotIn(secret, str(raised.exception))
            self.assertEqual(
                str(raised.exception), "credential request failed")
        finally:
            await client.close()
            await server.close()


class SessionOwnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_unstarted_owner_close_reports_native_failure_without_retry(self):
        endpoint = host_ipc.PipeEndpoint(read=101)
        failure = host_ipc.windows_api.WindowsApiError('CloseHandle failed', status=6)
        with mock.patch.object(host_ipc.windows_api, 'read_pipe', new=mock.AsyncMock()) as read, \
                mock.patch.object(host_ipc.windows_api, 'close_handle', side_effect=failure) as close:
            owner = credential_runtimes.SessionOwner(endpoint)
            with self.assertRaises(host_ipc.windows_api.WindowsApiError) as caught:
                await owner.close()
            self.assertIs(caught.exception, failure)
            self.assertTrue(owner.closed_task.cancelled())
            read.assert_not_awaited()
            close.assert_called_once_with(101)
            self.assertEqual(endpoint.handles(), [])
            await owner.close()
            close.assert_called_once_with(101)

    async def test_owner_channel_and_socket_share_close_only_observation(self):
        for create in [host_ipc.owner_channel, socket.socketpair]:
            with self.subTest(create=create.__name__):
                parent, child = create()
                if isinstance(child, int):
                    os.set_inheritable(child, True)
                owner = credential_runtimes.SessionOwner(child)
                try:
                    if isinstance(child, int):
                        # Inheritance is cleared before the watcher gets a step.
                        self.assertFalse(os.get_inheritable(child))
                    host_ipc.close_end(parent)
                    parent = None
                    await owner.closed_task
                    if isinstance(child, int):
                        os.fstat(child)  # The fd watch borrows, rather than consumes, its end.
                finally:
                    await owner.close()
                    if parent is not None:
                        host_ipc.close_end(parent)

    async def test_native_end_release_follows_joined_read_retirement(self):
        entered, retiring, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        closed = []

        async def read_pipe(handle):
            self.assertEqual(handle, 23)
            entered.set()
            try:
                await asyncio.Future()
            finally:
                retiring.set()
                await release.wait()

        endpoint = host_ipc.PipeEndpoint(read=23)
        with mock.patch.object(host_ipc.windows_api, 'read_pipe', new=read_pipe), \
                mock.patch.object(host_ipc.windows_api, 'close_handle', side_effect=closed.append):
            owner = credential_runtimes.SessionOwner(endpoint)
            closing = None
            try:
                await observe_fixture(entered.wait(), owner.closed_task)
                closing = asyncio.create_task(owner.close())
                await observe_fixture(retiring.wait(), closing)
                self.assertFalse(closing.done())
                self.assertEqual(closed, [])
                release.set()
                await closing
                self.assertEqual(closed, [23])
                self.assertEqual(endpoint.handles(), [])
            finally:
                release.set()
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)
                await owner.close()

    async def test_cancelled_close_releases_the_inherited_endpoint(self):
        parent, child = host_ipc.owner_channel()
        owner = credential_runtimes.SessionOwner(child)
        native = host_ipc.is_endpoint(child)
        release = (mock.patch.object(child, 'close', wraps=child.close) if native
                   else mock.patch.object(os, 'close', wraps=os.close))
        with release as released:
            closing = asyncio.create_task(owner.close())
            # Cancel after watcher retirement, before the join resumes its
            # caller. The real endpoint must close before cancellation escapes.
            owner.closed_task.add_done_callback(lambda _task: closing.cancel())
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await closing
                self.assertTrue(owner.closed_task.done())
                if native:
                    from loki_agent import windows_api
                    with self.assertRaises(windows_api.WindowsApiError) as raised:
                        await windows_api.write_pipe(parent.write, b'closed')
                    self.assertIn(raised.exception.status, windows_api.PIPE_CLOSED_ERRORS)
                    released.assert_called_once_with()
                else:
                    with self.assertRaises(BrokenPipeError):
                        os.write(parent, b'closed')
                    released.assert_called_once_with(child)
            finally:
                if not closing.done():
                    closing.cancel()
                await asyncio.gather(closing, return_exceptions=True)
                if not released.called:
                    host_ipc.close_end(child)
                host_ipc.close_end(parent)


class DelegatedRuntimeProcessTests(unittest.IsolatedAsyncioTestCase):
    """A real child receives both ends and the delegation contract holds.

    The child inherits the owner and credential ends through host_ipc -- the
    POSIX descriptor list or the Windows handle list -- leases a credential,
    and is revoked when the owner end closes.  Lifetime revocation is the
    contract the shipped runtime depends on, so it is exercised here as a
    process, not just as two connected sockets.
    """

    async def asyncSetUp(self):
        self.broker = authentications.CredentialBroker()
        self.credential = authentications.CredentialRef.environment(
            "DELEGATED_API_KEY")
        self.broker.install_static(self.credential, "delegated-secret")
        await self.broker.approve_destinations(self.credential, [DESTINATION])
        self.delegation = None

    async def asyncTearDown(self):
        if self.delegation is not None:
            await self.delegation.close()

    async def test_a_delegated_child_leases_then_the_owner_revokes_it(self):
        self.delegation = (
            await credential_supervisors.RuntimeDelegation.create(
                self.broker, {self.credential}))
        arguments = self.delegation.child_arguments()
        spawn = self.delegation.child_spawn_kwargs()
        with tempfile.TemporaryDirectory() as scratch:
            out_path = os.path.join(scratch, "child.out")
            err_path = os.path.join(scratch, "child.err")
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", CHILD_SCRIPT,
                    *arguments, "--credential", self.credential.encode(),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=out, stderr=err, close_fds=True, **spawn)
                # Register the backstop before the first readiness assertion;
                # a failed handshake must not leave a child using the files.
                try:
                    self.delegation.child_spawned()
                    await self._await_marker(out_path, b"lease:", process)
                    self.assertIsNone(process.returncode)
                    self.assertFalse(
                        self.delegation.credential_server._reader_task.done())
                    # This is cooperative lifetime shutdown, not proof that
                    # a still-live credential channel rejects new requests.
                    host_ipc.close_end(self.delegation.owner_parent)
                    self.delegation.owner_parent = None
                    self.assertEqual(
                        await process.wait(), 0)
                finally:
                    if process.returncode is None:
                        process.kill()
                    await process.wait()
            with open(out_path, encoding="utf-8") as stream:
                stdout = stream.read()
            with open(err_path, encoding="utf-8") as stream:
                stderr = stream.read()

        self.assertEqual(stdout, "lease:delegated-secret\nrevoked\n")
        self.assertEqual(stderr, "")

    @staticmethod
    async def _await_marker(path, marker, process):
        # The delegated actor remains live until owner EOF. Its actual exit,
        # not startup speed, invalidates this file observation.
        while True:
            with open(path, "rb") as stream:
                if marker in stream.read():
                    return
            if process.returncode is not None:
                raise AssertionError(
                    "delegated child exited before writing %r" % marker)
            await asyncio.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
