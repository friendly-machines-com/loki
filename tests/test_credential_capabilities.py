import asyncio
import os
import unittest

from loki_agent import authentications
from loki_agent import credential_capabilities


class CredentialCapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.broker = authentications.CredentialBroker()
        self.first = authentications.CredentialRef.environment(
            "FIRST_API_KEY")
        self.second = authentications.CredentialRef.environment(
            "SECOND_API_KEY")
        self.broker.install_static(self.first, "first-secret")
        self.broker.install_static(self.second, "second-secret")
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
        self.assertFalse(os.get_inheritable(child_fd))
        self.client = (
            await credential_capabilities.CredentialClient.from_fd(child_fd))

    async def test_delegates_only_allowed_credentials(self):
        await self.connect({self.first})

        lease = await self.client.lease(self.first)

        self.assertEqual(lease.value, "first-secret")
        self.assertEqual(self.client.available(), frozenset({self.first}))
        with self.assertRaises(credential_capabilities.CapabilityError):
            await self.client.lease(self.second)

    async def test_relay_can_delegate_strict_subset(self):
        await self.connect({self.first, self.second})
        relay, child_fd = (
            await credential_capabilities.CredentialRelay.create(
                self.client, {self.second}))
        grandchild = (
            await credential_capabilities.CredentialClient.from_fd(child_fd))
        try:
            lease = await grandchild.lease(self.second)
            self.assertEqual(lease.value, "second-secret")
            with self.assertRaises(
                    credential_capabilities.CapabilityError):
                await grandchild.lease(self.first)
        finally:
            await grandchild.close()
            await relay.close()

    async def test_rejected_generation_refreshes_only_in_root_broker(self):
        calls = []

        async def refresh(refresh_token):
            calls.append(refresh_token)
            return authentications.RefreshResult(
                access_token="access-new",
                refresh_token="refresh-new",
            )

        credential = (
            authentications.CredentialRef.openai_subscription())
        self.broker.install_openai_subscription(
            authentications.OpenAITokenSet(
                access_token="access-old",
                refresh_token="refresh-old",
                expires_at=10**12,
            ),
            refresh=refresh,
        )
        await self.connect({credential})

        first = await self.client.lease(credential)
        second = await self.client.lease(
            credential, rejected_generation=first.generation)

        self.assertEqual(first.value, "access-old")
        self.assertEqual(second.value, "access-new")
        self.assertEqual(calls, ["refresh-old"])
        self.assertFalse(hasattr(second, "refresh_token"))
        self.assertEqual(
            self.broker.openai_tokens().refresh_token,
            "refresh-new",
        )

    async def test_concurrent_requests_are_multiplexed(self):
        await self.connect({self.first, self.second})

        leases = await asyncio.gather(*[
            self.client.lease(
                self.first if index % 2 == 0 else self.second)
            for index in range(20)
        ])

        self.assertEqual(
            [lease.value for lease in leases].count("first-secret"), 10)
        self.assertEqual(
            [lease.value for lease in leases].count("second-secret"), 10)

    async def test_owner_close_fails_pending_and_future_requests(self):
        started = asyncio.Event()

        class BlockingAuthority:
            def available(inner_self):
                return frozenset({self.first})

            async def lease(
                    inner_self, credential,
                    rejected_generation=None):
                started.set()
                await asyncio.Event().wait()

        self.server, child_fd = (
            await credential_capabilities.CredentialCapabilityServer.create(
                BlockingAuthority(), {self.first}))
        self.client = (
            await credential_capabilities.CredentialClient.from_fd(child_fd))
        pending = asyncio.create_task(self.client.lease(self.first))
        await started.wait()
        await self.server.close()

        with self.assertRaises(
                credential_capabilities.CapabilityError):
            await pending
        with self.assertRaises(credential_capabilities.CapabilityError):
            await self.client.lease(self.first)

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

    async def test_invalid_delegated_descriptor_is_closed(self):
        read_fd, write_fd = os.pipe()
        try:
            with self.assertRaises(OSError):
                await credential_capabilities.CredentialClient.from_fd(
                    read_fd)
            with self.assertRaises(OSError):
                os.fstat(read_fd)
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
                    inner_self, credential,
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
                await client.lease(self.first)
            self.assertNotIn(secret, str(raised.exception))
            self.assertEqual(
                str(raised.exception), "credential request failed")
        finally:
            await client.close()
            await server.close()


if __name__ == "__main__":
    unittest.main()
