"""Exact request permissions across the real authority, channel and store."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from loki_agent import authentications as auth
from loki_agent import credential_capabilities as capabilities
from loki_agent import credential_storages, file_locks, private_files
from loki_agent.credential_errors import CredentialStorageError


URL = "https://example.test/v1/responses?version=1"
OTHER = "https://example.test/v1/responses?version=2"


class EndpointAuthorizationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.storage = credential_storages.JsonCredentialStorage(self.directory.name)
        self.reference = auth.CredentialRef.environment("EXAMPLE_API_KEY")
        self.broker = auth.CredentialBroker(storage=self.storage)
        self.broker.install_static(self.reference, "sample-value")

    async def test_only_exact_recorded_urls_issue_values(self):
        with self.assertRaises(auth.CredentialUnavailable):
            await self.broker.lease(self.reference, URL)
        await self.broker.approve_destinations(self.reference, [URL])
        lease = await self.broker.lease(self.reference, URL)
        self.assertEqual(lease.value, "sample-value")
        for destination in [OTHER, "https://example.test/other",
                            "https://example.test:443/v1/responses?version=1",
                            "http://example.test/v1/responses?version=1",
                            URL + "#part", "https://user@example.test/v1/responses"]:
            with self.subTest(destination=destination), self.assertRaises(auth.CredentialUnavailable):
                await self.broker.lease(self.reference, destination)
        self.assertEqual(self.storage.load_document()["endpoint_approvals"], {
            self.reference.encode(): [URL]})

    async def test_another_credential_does_not_inherit_approval(self):
        other_ref = auth.CredentialRef.environment("OTHER_API_KEY")
        self.broker.install_static(other_ref, "other-value")
        await self.broker.approve_destinations(self.reference, [URL])
        with self.assertRaises(auth.CredentialUnavailable):
            await self.broker.lease(other_ref, URL)

    async def test_old_state_file_is_not_authority(self):
        import json
        from pathlib import Path
        state = Path(self.directory.name, "state", "loki")
        state.mkdir(parents=True)
        (state / "provider-endpoints.json").write_text(json.dumps({
            "example": {"api": URL, "credential": self.reference.encode()}}))
        with mock.patch.dict(os.environ, {"XDG_STATE_HOME": str(state.parent)}):
            with self.assertRaises(auth.CredentialUnavailable):
                await self.broker.lease(self.reference, URL)

    async def test_other_supervisor_changes_are_visible_without_a_cache(self):
        other = auth.CredentialBroker(storage=credential_storages.JsonCredentialStorage(self.directory.name))
        other.install_static(self.reference, "other-owner-value")
        await self.broker.approve_destinations(self.reference, [URL])
        self.assertEqual((await other.lease(self.reference, URL)).value, "other-owner-value")
        async with self.storage._locked_document() as [directory_fd, document]:
            document["endpoint_approvals"][self.reference.encode()] = []
            self.storage._next_revision(document)
            self.storage._write_document_at(directory_fd, document)
        with self.assertRaises(auth.CredentialUnavailable):
            await other.lease(self.reference, URL)

    async def test_permission_is_rechecked_after_waiting_for_a_value(self):
        started = asyncio.Event()
        released = asyncio.Event()

        async def delayed(rejected_generation=None):
            started.set()
            await released.wait()
            return auth.CredentialLease(self.reference, "sample-value")
        await self.broker.approve_destinations(self.reference, [URL])
        with mock.patch.object(self.broker._records[self.reference], "lease", delayed):
            pending = asyncio.create_task(self.broker.lease(self.reference, URL))
            await started.wait()
            async with self.storage._locked_document() as [directory_fd, document]:
                document["endpoint_approvals"][self.reference.encode()] = []
                self.storage._next_revision(document)
                self.storage._write_document_at(directory_fd, document)
            released.set()
            with self.assertRaises(auth.CredentialUnavailable):
                await pending

    async def test_subscription_limit_cannot_be_overridden_by_disk_or_confirmation(self):
        reference = auth.CredentialRef.openai_subscription()
        self.broker.install_openai_subscription(auth.OpenAITokenSet(
            "sample-access", "sample-refresh", expires_at=10**12))
        unrelated = "https://chatgpt.com/not-a-codex-endpoint"
        await self.storage.approve_destinations(reference, [unrelated])
        with self.assertRaises(auth.CredentialUnavailable):
            await self.broker.lease(reference, unrelated)
        with self.assertRaises(auth.CredentialUnavailable):
            await self.broker.approve_destinations(reference, [unrelated])
        lease = await self.broker.lease(reference, auth.OPENAI_CHATGPT_RESPONSES_URL)
        self.assertEqual(lease.value, "sample-access")
        from loki_agent.openai_controls import OPENAI_CHATGPT_USAGE_URL
        self.assertEqual((await self.broker.lease(reference, OPENAI_CHATGPT_USAGE_URL)).value,
                         "sample-access")
        inference = auth.AuthSpec(reference, "openai-subscription",
                                  authorized_urls=auth.OPENAI_CHATGPT_CODEX_URLS)
        with self.assertRaises(auth.CredentialUnavailable):
            await auth.authorized_request_headers(self.broker, inference, OPENAI_CHATGPT_USAGE_URL)

    async def test_token_transactions_preserve_approvals(self):
        tokens = auth.OpenAITokenSet(
            "sample-access", "sample-refresh", expires_at=10**12, last_refresh=1)
        await self.storage.store_openai_login(tokens)
        await self.broker.approve_destinations(self.reference, [URL])

        async def refresh(value):
            self.assertEqual(value, "sample-refresh")
            return auth.RefreshResult("replacement-access", "replacement-refresh")
        refreshed = await self.storage.rotate_openai_subscription(tokens.normalized(), refresh=refresh)
        self.assertEqual(refreshed.refresh_token, "replacement-refresh")
        self.assertEqual(self.storage.approved_destinations(self.reference), frozenset({URL}))
        await self.storage.remove_openai_subscription()
        self.assertEqual(self.storage.approved_destinations(self.reference), frozenset({URL}))

    async def test_older_token_document_without_approvals_is_valid(self):
        await self.storage.store_openai_login(auth.OpenAITokenSet(
            "sample-access", "sample-refresh", expires_at=10**12))
        self.assertNotIn("endpoint_approvals", self.storage.load_document())
        self.assertEqual(self.storage.approved_destinations(self.reference), frozenset())

    async def test_bad_approval_section_is_refused(self):
        for section in [None, [], {"env:KEY": "https://example.test"},
                        {"invalid-reference": [URL]}, {"env:KEY": [URL, URL]},
                        {"env:KEY": ["https://user@example.test/"]}]:
            with self.subTest(section=section), self.assertRaises(CredentialStorageError):
                credential_storages._validate_document({
                    "version": 1, "revision": 0, "credentials": {},
                    "endpoint_approvals": section})

    async def test_closed_owner_cannot_publish_after_waiting_for_the_shared_lock(self):
        held = asyncio.Event()
        release = asyncio.Event()
        waiting = asyncio.Event()
        live = True

        async def holder():
            async with self.storage._locked_document():
                held.set()
                await release.wait()

        def check_owner():
            if not live:
                raise RuntimeError("closed owner")
        lock_holder = asyncio.create_task(holder())
        await held.wait()
        real_lock = file_locks.try_lock_exclusive

        def observe(token):
            try:
                return real_lock(token)
            except BlockingIOError:
                waiting.set()
                raise
        try:
            with mock.patch.object(file_locks, "try_lock_exclusive", observe):
                pending = asyncio.create_task(self.broker.approve_destinations(
                    self.reference, [URL], before_commit=check_owner))
                await waiting.wait()
                live = False
                release.set()
                await lock_holder
                with self.assertRaisesRegex(RuntimeError, "closed owner"):
                    await pending
            self.assertEqual(self.storage.approved_destinations(self.reference), frozenset())
        finally:
            release.set()
            await lock_holder

    async def test_real_channel_record_then_lease_and_relay_limits(self):
        server, endpoint = await capabilities.CredentialCapabilityServer.create(
            self.broker, {self.reference}, manage_approvals=True)
        client = await capabilities.CredentialClient.from_fd(endpoint)
        relay = child = None
        try:
            self.assertEqual(await client.unapproved_destinations(self.reference, [URL]), [URL])
            await client.approve_destinations(self.reference, [URL, OTHER])
            self.assertEqual((await client.lease(self.reference, URL)).value, "sample-value")
            relay, child_endpoint = await capabilities.CredentialCapabilityServer.create(
                client, {self.reference}, destinations=[URL])
            child = await capabilities.CredentialClient.from_fd(child_endpoint)
            self.assertEqual((await child.lease(self.reference, URL)).value, "sample-value")
            with self.assertRaises(capabilities.CapabilityError):
                await child.lease(self.reference, OTHER)
            with self.assertRaises(capabilities.CapabilityError):
                await child.approve_destinations(self.reference, [OTHER])
            with self.assertRaises(capabilities.CapabilityError):
                await child.unapproved_destinations(self.reference, [OTHER])
        finally:
            if child is not None:
                await child.close()
            if relay is not None:
                await relay.close()
            await client.close()
            await server.close()

    async def test_cancelled_channel_record_does_not_publish(self):
        held = asyncio.Event()
        release = asyncio.Event()
        waiting = asyncio.Event()

        async def holder():
            async with self.storage._locked_document():
                held.set()
                await release.wait()
        task = asyncio.create_task(holder())
        await held.wait()
        server, endpoint = await capabilities.CredentialCapabilityServer.create(
            self.broker, {self.reference}, manage_approvals=True)
        client = await capabilities.CredentialClient.from_fd(endpoint)
        real_lock = file_locks.try_lock_exclusive

        def observe(token):
            try:
                return real_lock(token)
            except BlockingIOError:
                waiting.set()
                raise
        try:
            with mock.patch.object(file_locks, "try_lock_exclusive", observe):
                pending = asyncio.create_task(client.approve_destinations(self.reference, [URL]))
                await waiting.wait()
                await server.close()
                with self.assertRaises(capabilities.CapabilityError):
                    await pending
            release.set()
            await task
            self.assertEqual(self.storage.approved_destinations(self.reference), frozenset())
        finally:
            release.set()
            await task
            await client.close()
            await server.close()

    @unittest.skipUnless(os.name == "posix", "POSIX file types")
    async def test_special_document_is_rejected_before_blocking(self):
        self.storage.ensure_directory()
        os.mkfifo(self.storage.file_path, 0o600)
        with self.assertRaises(CredentialStorageError):
            self.storage.approved_destinations(self.reference)

    async def test_catalog_confirmation_uses_effective_full_request_urls(self):
        import contextlib
        import io
        from loki_agent import models, loki
        from loki_agent.credentials import CredentialStore
        credentials = CredentialStore({"EXAMPLE_API_KEY": "sample-value"})
        provider = {"id": "example", "api": "https://api.deepseek.com/anthropic",
                    "npm": "@ai-sdk/anthropic", "env": ["EXAMPLE_API_KEY"]}
        model = {"id": "sample"}
        config = loki.config_from_modelsdev_selection("example", provider, model, credentials)
        targets = config.credential_destinations()
        answer = mock.AsyncMock(return_value="yes")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            accepted = await models._confirm_catalog_endpoint(
                answer, lambda text: print(text, end=""), credentials,
                "example", provider, model, self.broker)
        self.assertTrue(accepted)
        for target in targets:
            self.assertIn(target, output.getvalue())
            self.assertEqual((await self.broker.lease(self.reference, target)).value, "sample-value")
        self.assertIn("https://api.deepseek.com/user/balance", targets)
        self.assertIn("https://api.deepseek.com/models", targets)
        answer.reset_mock()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(await models._confirm_catalog_endpoint(
                answer, lambda text: None, credentials, "renamed-provider", provider,
                model, self.broker))
        answer.assert_not_awaited()

    async def test_declined_catalog_confirmation_creates_no_permission(self):
        import contextlib
        import io
        from loki_agent import models
        from loki_agent.credentials import CredentialStore
        credentials = CredentialStore({"EXAMPLE_API_KEY": "sample-value"})
        provider = {"id": "example", "api": "https://example.test/v1",
                    "npm": "@ai-sdk/openai-compatible", "env": ["EXAMPLE_API_KEY"]}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(await models._confirm_catalog_endpoint(
                mock.AsyncMock(return_value=""), lambda text: None, credentials,
                "example", provider, {"id": "sample"}, self.broker))
        with self.assertRaises(auth.CredentialUnavailable):
            await self.broker.lease(self.reference, "https://example.test/v1/chat/completions")

    async def test_approval_waits_for_refresh_and_preserves_the_completed_tokens(self):
        tokens = auth.OpenAITokenSet(
            "sample-access", "sample-refresh", expires_at=10**12, last_refresh=1)
        await self.storage.store_openai_login(tokens)
        entered = asyncio.Event()
        release = asyncio.Event()
        waiting = asyncio.Event()

        async def refresh(value):
            self.assertEqual(value, "sample-refresh")
            entered.set()
            await release.wait()
            return auth.RefreshResult("next-access", "next-refresh")

        rotation = asyncio.create_task(self.storage.rotate_openai_subscription(
            tokens.normalized(), refresh=refresh))
        await entered.wait()
        real_lock = file_locks.try_lock_exclusive

        def observe(token):
            try:
                return real_lock(token)
            except BlockingIOError:
                waiting.set()
                raise

        try:
            with mock.patch.object(file_locks, "try_lock_exclusive", observe):
                saving = asyncio.create_task(self.broker.approve_destinations(
                    self.reference, [URL]))
                await waiting.wait()
                release.set()
                await asyncio.gather(rotation, saving)
            self.assertEqual(self.storage.load_openai_subscription().tokens.refresh_token,
                             "next-refresh")
            self.assertEqual(self.storage.approved_destinations(self.reference), frozenset({URL}))
        finally:
            release.set()
            await rotation

    async def test_controller_without_approval_permission_cannot_change_the_store(self):
        server, endpoint = await capabilities.CredentialCapabilityServer.create(
            self.broker, {self.reference})
        client = await capabilities.CredentialClient.from_fd(endpoint)
        try:
            with self.assertRaises(capabilities.CapabilityError):
                await client.approve_destinations(self.reference, [URL])
            self.assertEqual(self.storage.approved_destinations(self.reference), frozenset())
            with self.assertRaises(capabilities.CapabilityError):
                await client._request("lease", {"credential": self.reference.encode()})
        finally:
            await client.close()
            await server.close()

    async def test_created_document_uses_platform_private_file_checks(self):
        await self.broker.approve_destinations(self.reference, [URL])
        directory = private_files.open_directory(self.directory.name)
        try:
            from loki_agent import paths
            descriptor = private_files.open_read_at(directory, paths.CREDENTIAL_FILE_NAME)
            try:
                facts = private_files.describe(descriptor)
                self.assertTrue(facts.regular and facts.owned_by_current_user)
                self.assertFalse(facts.group_or_other_access or facts.reparse_point)
            finally:
                private_files.close(descriptor)
        finally:
            private_files.close(directory)
