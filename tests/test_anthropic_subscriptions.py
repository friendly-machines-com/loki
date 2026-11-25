"""Exercise registered-client OAuth and the broker without a live subscription."""

import asyncio
import contextlib
import http.client
import io
import json
import tempfile
import threading
import urllib.parse
import unittest
from unittest import mock

from process_lifecycle_fixtures import observe_fixture, running_fixture_task
from loki_agent import anthropic_controls
from loki_agent import authentication_commands
from loki_agent import authentications
from loki_agent import connections
from loki_agent import credential_storages
from loki_agent import credential_supervisors
from loki_agent import file_locks
from loki_agent import http_client
from loki_agent import model_requests
from loki_agent import models
from loki_agent import oauth_logins
from loki_agent import protocols
from loki_agent.credentials import CredentialInventory, CredentialStore


CLIENT_ID = "registered-app-id"


def response(url, status, body):
    return http_client.HttpResponse(
        url=url,
        status=status,
        reason="",
        headers={},
        body=(
            json.dumps(body).encode("utf-8")
            if isinstance(body, dict) else body),
    )


def anthropic_tokens(**overrides):
    values = dict(
        access_token="anthropic-access-secret",
        refresh_token="anthropic-refresh-secret",
        client_id=CLIENT_ID,
        scopes=authentications.ANTHROPIC_OAUTH_SCOPES,
        account=authentications.AnthropicAccount(
            uuid="account-uuid",
            email_address="user@example.test",
            organization_uuid="organization-uuid"),
        expires_at=10**12,
        last_refresh=100,
    )
    values.update(overrides)
    return authentications.AnthropicTokenSet(**values)


def token_response():
    return {
        "access_token": "anthropic-access-secret",
        "refresh_token": "anthropic-refresh-secret",
        "expires_in": 28800,
        "scope": " ".join(sorted(authentications.ANTHROPIC_OAUTH_SCOPES)),
        "account": {"uuid": "account-uuid",
                    "email_address": "user@example.test"},
        "organization": {"uuid": "organization-uuid"},
    }


class AnthropicAuthorizationUrlTests(unittest.TestCase):
    def test_authorization_url_binds_registered_client_and_pkce(self):
        pkce = oauth_logins.PkceCodes("verifier", "challenge")
        url = oauth_logins.anthropic_authorization_url(
            "http://127.0.0.1:1234/callback", pkce, "state", client_id=CLIENT_ID)
        split = urllib.parse.urlsplit(url)
        self.assertEqual(
            split.scheme + "://" + split.netloc + split.path,
            oauth_logins.ANTHROPIC_AUTHORIZE_URL)
        query = urllib.parse.parse_qs(split.query)
        self.assertEqual(query["code"], ["true"])
        self.assertEqual(
            query["client_id"],
            [CLIENT_ID])
        self.assertEqual(query["response_type"], ["code"])
        self.assertEqual(
            query["redirect_uri"], ["http://127.0.0.1:1234/callback"])
        self.assertEqual(
            query["scope"], [" ".join(sorted(authentications.ANTHROPIC_OAUTH_SCOPES))])
        self.assertEqual(query["code_challenge"], ["challenge"])
        self.assertEqual(query["code_challenge_method"], ["S256"])
        self.assertEqual(query["state"], ["state"])


class AnthropicBrowserLoginTests(unittest.IsolatedAsyncioTestCase):
    async def _redirect(self, login):
        query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(login.authorization_url).query)
        return query, urllib.parse.urlsplit(query["redirect_uri"][0])

    async def _callback(self, login, path, query_text):
        _query, redirect = await self._redirect(login)
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", redirect.port)
        writer.write(
            f"GET {path}?{query_text} HTTP/1.1\r\n\r\n".encode("ascii"))
        await writer.drain()
        answer = await reader.read()
        writer.close()
        await writer.wait_closed()
        return answer

    async def test_callback_acknowledges_code_without_redirecting_or_echoing_it(self):
        login = await oauth_logins.start_anthropic_browser_login(client_id=CLIENT_ID)
        query, redirect = await self._redirect(login)
        self.assertEqual(redirect.path, "/callback")

        answer = await self._callback(
            login, "/callback",
            urllib.parse.urlencode({
                "code": "authorization-secret",
                "state": query["state"][0],
            }))

        self.assertIn(b"200 OK", answer)
        self.assertIn(b"Referrer-Policy: no-referrer", answer)
        self.assertNotIn(b"Location:", answer)
        self.assertNotIn(b"authorization-secret", answer)
        self.assertEqual(await login._result, "authorization-secret")
        login.close()
        await login._server.wait_closed()

    async def test_completed_login_cannot_redeem_callback_again(self):
        requests = []

        async def request(method, url, **kwargs):
            requests.append(json.loads(kwargs["body"]))
            return response(url, 200, token_response())

        login = await oauth_logins.start_anthropic_browser_login(client_id=CLIENT_ID, request=request)
        self.addAsyncCleanup(login.aclose)
        query, redirect = await self._redirect(login)
        state = query["state"][0]
        await self._callback(login, redirect.path, urllib.parse.urlencode({"code": "first", "state": state}))
        await self._callback(login, redirect.path, urllib.parse.urlencode({"code": "second", "state": state}))
        await login.complete()
        with self.assertRaises(oauth_logins.OAuthLoginError):
            await login.complete()
        self.assertEqual([body["code"] for body in requests], ["first"])

    async def test_wrong_state_cannot_finish_login(self):
        login = await oauth_logins.start_anthropic_browser_login(client_id=CLIENT_ID)

        answer = await self._callback(
            login, "/callback",
            urllib.parse.urlencode({"code": "code", "state": "wrong"}))

        self.assertIn(b"400 Bad Request", answer)
        self.assertFalse(login._result.done())
        login.close()
        await login._server.wait_closed()

    async def test_unknown_path_is_ignored_and_matching_denial_finishes_login(self):
        login = await oauth_logins.start_anthropic_browser_login(client_id=CLIENT_ID)
        query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(login.authorization_url).query)
        state = query["state"][0]

        missing = await self._callback(
            login, "/elsewhere",
            urllib.parse.urlencode({"code": "code", "state": state}))
        self.assertIn(b"404 Not Found", missing)

        declined = await self._callback(
            login, "/callback",
            urllib.parse.urlencode({"error": "access_denied",
                                    "state": state}))
        self.assertIn(b"400 Bad Request", declined)
        self.assertTrue(login._result.done())
        with self.assertRaisesRegex(
                oauth_logins.OAuthLoginError, "declined"):
            await login._result
        login.close()
        await login._server.wait_closed()


class AnthropicTokenExchangeTests(unittest.IsolatedAsyncioTestCase):
    async def test_exchange_parses_tokens_account_and_expiry(self):
        now = 1_000_000.0

        async def request(_method, url, **kwargs):
            self.assertEqual(url, authentications.ANTHROPIC_TOKEN_URL)
            body = json.loads(kwargs["body"])
            self.assertEqual(body["grant_type"], "authorization_code")
            self.assertEqual(body["code"], "authorization-secret")
            self.assertEqual(
                body["client_id"], CLIENT_ID)
            self.assertEqual(body["code_verifier"], "verifier")
            self.assertEqual(body["state"], "state")
            return response(url, 200, token_response())

        tokens = await oauth_logins.exchange_anthropic_authorization_code(
            "authorization-secret",
            "http://127.0.0.1:1234/callback",
            oauth_logins.PkceCodes("verifier", "challenge"),
            "state",
            client_id=CLIENT_ID,
            request=request,
            now=now,
        )

        self.assertEqual(tokens.access_token, "anthropic-access-secret")
        self.assertEqual(tokens.refresh_token, "anthropic-refresh-secret")
        self.assertEqual(tokens.scopes, authentications.ANTHROPIC_OAUTH_SCOPES)
        self.assertEqual(tokens.expires_at, now + 28800)
        self.assertEqual(tokens.account.uuid, "account-uuid")
        self.assertEqual(
            tokens.account.email_address, "user@example.test")
        self.assertEqual(
            tokens.account.organization_uuid, "organization-uuid")

    async def test_exchange_rejects_malformed_responses(self):
        for payload in [
            {"access_token": "", "refresh_token": "r",
             "expires_in": 10, "scope": "user:inference"},
            {"access_token": "a", "refresh_token": "",
             "expires_in": 10, "scope": "user:inference"},
            {"access_token": "a", "refresh_token": "r",
             "expires_in": "soon", "scope": "user:inference"},
            {"access_token": "a", "refresh_token": "r",
             "expires_in": 10, "scope": " "},
            {"access_token": "a", "refresh_token": "r",
             "expires_in": 10, "scope": "user:inference",
             "account": {"uuid": ""}},
        ]:
            with self.subTest(payload=payload):
                async def request(_method, url, **_kwargs):
                    return response(url, 200, payload)

                with self.assertRaises(oauth_logins.OAuthLoginError):
                    await oauth_logins.exchange_anthropic_authorization_code(
                        "code", "http://127.0.0.1:1/callback",
                        oauth_logins.PkceCodes("v", "c"), "state",
                        client_id=CLIENT_ID, request=request)

    async def test_token_error_does_not_include_response_body(self):
        secret_body = b'{"error":"invalid_grant","refresh_token":"no"}'

        async def request(_method, url, **_kwargs):
            return response(url, 400, secret_body)

        with self.assertRaises(
                oauth_logins.OAuthLoginError) as raised:
            await oauth_logins.exchange_anthropic_authorization_code(
                "code", "http://127.0.0.1:1/callback",
                oauth_logins.PkceCodes("v", "c"), "state",
                client_id=CLIENT_ID, request=request)

        self.assertNotIn("invalid_grant", str(raised.exception))


class AnthropicRefreshTests(unittest.IsolatedAsyncioTestCase):
    def _refresh_response(self, url, **fields):
        body = {"access_token": "anthropic-access-2", "expires_in": 28800}
        body.update(fields)
        return response(url, 200, body)

    async def _refresh(self, url_response):
        async def transport(method, url, **kwargs):
            self.assertEqual(method, "POST")
            self.assertEqual(url, authentications.ANTHROPIC_TOKEN_URL)
            body = json.loads(kwargs["body"])
            self.assertEqual(kwargs["retry_max_attempts"], 1)
            self.assertEqual(body["grant_type"], "refresh_token")
            self.assertEqual(body["refresh_token"], "anthropic-refresh-secret")
            self.assertEqual(
                body["client_id"], CLIENT_ID)
            self.assertEqual(
                body["scope"],
                " ".join(sorted(authentications.ANTHROPIC_OAUTH_SCOPES)))
            return url_response(url)

        with mock.patch.object(
                http_client, "async_http_request", new=transport):
            return await authentications.request_anthropic_token_refresh(
                anthropic_tokens())

    async def test_refresh_rotates_tokens_and_uses_expiry(self):
        now = 5_000.0
        result = authentications.AnthropicRefreshResult(
            access_token="anthropic-access-2",
            refresh_token="anthropic-refresh-2",
            expires_in=3600,
            scopes=frozenset({"user:inference"}),
        )
        refreshed = authentications.refreshed_anthropic_tokens(
            anthropic_tokens(), result, now)
        self.assertEqual(refreshed.access_token, "anthropic-access-2")
        self.assertEqual(refreshed.refresh_token, "anthropic-refresh-2")
        self.assertEqual(refreshed.expires_at, now + 3600)
        self.assertEqual(
            refreshed.scopes, {"user:inference"})
        self.assertEqual(refreshed.account.uuid, "account-uuid")

    async def test_refresh_keeps_refresh_token_when_rotation_omitted(self):
        result = authentications.AnthropicRefreshResult(
            access_token="anthropic-access-2",
            expires_in=3600,
        )
        refreshed = authentications.refreshed_anthropic_tokens(
            anthropic_tokens(), result, 0.0)
        self.assertEqual(refreshed.refresh_token, "anthropic-refresh-secret")
        self.assertEqual(refreshed.scopes,
                         authentications.ANTHROPIC_OAUTH_SCOPES)

    async def test_http_refresh_error_maps_to_permanent(self):
        for status, body in [
                (401, {}),
                (400, {"error": "invalid_grant"}),
                (400, {"error": {"code": "invalid_request"}}),
        ]:
            with self.subTest(status=status, body=body):
                with self.assertRaises(
                        authentications.RefreshPermanentError):
                    await self._refresh(
                        lambda url, s=status, b=body: response(url, s, b))

    async def test_server_error_maps_to_transient_possibly_sent(self):
        with self.assertRaises(
                authentications.RefreshTransientError) as raised:
            await self._refresh(lambda url: response(url, 503, {}))
        self.assertTrue(raised.exception.request_may_have_been_sent)

    async def test_malformed_success_is_indeterminate(self):
        for document in [b"[]", b"null", b"not json"]:
            with self.subTest(document=document), self.assertRaises(authentications.RefreshIndeterminateError):
                await self._refresh(lambda url: response(url, 200, document))
        for fields in [{"scope": []}, {"scope": None}, {"refresh_token": None}, {"expires_in": True},
                       {"access_token": "line\r\nbreak"}, {"access_token": "x" * 65536}]:
            with self.subTest(fields=list(fields)), self.assertRaises(authentications.RefreshIndeterminateError):
                await self._refresh(lambda url: self._refresh_response(url, **fields))

    async def test_refresh_does_not_widen_scope_or_switch_identity(self):
        for fields in [{"scopes": authentications.ANTHROPIC_OAUTH_SCOPES | {"wider:scope"}},
                       {"account": authentications.AnthropicAccount(uuid="other-account")},
                       {"account": authentications.AnthropicAccount(
                           uuid="account-uuid", organization_uuid="other-org")}]:
            with self.subTest(fields=fields), self.assertRaises(authentications.RefreshPermanentError):
                authentications.refreshed_anthropic_tokens(
                    anthropic_tokens(), authentications.AnthropicRefreshResult("new-access", **fields))

    async def test_late_refresh_failure_does_not_poison_replacement(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def rotate(_current):
            started.set()
            await release.wait()
            raise authentications.RefreshPermanentError("old grant revoked")

        record = authentications.AnthropicSubscriptionCredential(anthropic_tokens(), rotate=rotate)
        task = asyncio.create_task(record.lease(rejected_generation=0))
        await started.wait()
        record.replace(anthropic_tokens(access_token="new-token"))
        release.set()
        self.assertEqual((await task).value, "new-token")
        self.assertEqual((await record.lease()).generation, 1)

    async def test_credential_refresh_is_single_flight_and_rotates(self):
        now = 10_000.0
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def rotate(current):
            calls.append(current.refresh_token)
            started.set()
            await release.wait()
            return anthropic_tokens(
                access_token="anthropic-access-2",
                refresh_token="anthropic-refresh-2",
                expires_at=now + 3600,
            )

        credential = authentications.AnthropicSubscriptionCredential(
            anthropic_tokens(expires_at=now - 1),
            rotate=rotate,
            clock=lambda: now,
        )
        tasks = [asyncio.create_task(credential.lease()) for _ in range(6)]
        await started.wait()
        release.set()
        leases = await asyncio.gather(*tasks)

        self.assertEqual(calls, ["anthropic-refresh-secret"])
        self.assertEqual({lease.generation for lease in leases}, {1})
        self.assertEqual(
            credential.tokens.refresh_token, "anthropic-refresh-2")

    async def test_permanent_failure_is_remembered_for_generation(self):
        now = 10_000.0
        calls = []

        async def rotate(_current):
            calls.append("attempted")
            raise authentications.RefreshPermanentError("revoked")

        credential = authentications.AnthropicSubscriptionCredential(
            anthropic_tokens(expires_at=now - 1),
            rotate=rotate,
            clock=lambda: now,
        )
        with self.assertRaises(authentications.RefreshPermanentError):
            await credential.lease()
        with self.assertRaises(authentications.RefreshPermanentError):
            await credential.lease()
        self.assertEqual(calls, ["attempted"])


class AnthropicBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_subscription_target_is_confined_to_messages_url(self):
        broker = authentications.CredentialBroker()
        broker.install_anthropic_subscription(anthropic_tokens())
        ref = authentications.CredentialRef.anthropic_subscription()
        spec = authentications.AuthSpec(
            ref, scheme="anthropic-subscription",
            authorized_urls=authentications.ANTHROPIC_SUBSCRIPTION_URLS)
        url = authentications.ANTHROPIC_BETA_MESSAGES_URL

        headers, lease = await authentications.authorized_request_headers(
            broker, spec, url)

        self.assertEqual(
            headers["Authorization"], "Bearer anthropic-access-secret")
        self.assertEqual(
            headers["anthropic-beta"], authentications.ANTHROPIC_OAUTH_BETA)
        self.assertEqual(
            headers["anthropic-dangerous-direct-browser-access"], "true")
        self.assertEqual(lease.account_id, "account-uuid")
        # The non-beta path is a different destination and must be refused.
        with self.assertRaises(authentications.CredentialUnavailable):
            await authentications.authorized_request_headers(
                broker, spec, authentications.ANTHROPIC_MESSAGES_URL)
        with self.assertRaises(authentications.CredentialUnavailable):
            await authentications.authorized_request_headers(
                broker, spec, authentications.ANTHROPIC_MESSAGES_URL + "?beta=false")
        with self.assertRaises(authentications.CredentialUnavailable):
            await authentications.authorized_request_headers(
                broker, spec, "https://api.anthropic.com/v1/complete")
        with self.assertRaises(authentications.CredentialUnavailable):
            await broker.lease(
                ref, destination="https://evil.invalid/v1/messages")


class AnthropicStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.storage = credential_storages.JsonCredentialStorage(self.temporary.name)
        self.addCleanup(self.storage.close)

    async def test_store_load_and_remove_roundtrip(self):
        self.assertIsNone(self.storage.load_anthropic_subscription())

        stored = await self.storage.store_anthropic_login(anthropic_tokens())

        self.assertEqual(stored.state, "active")
        loaded = self.storage.load_anthropic_subscription()
        self.assertEqual(loaded.tokens, stored.tokens)
        self.assertEqual(loaded.tokens.client_id, CLIENT_ID)
        self.assertTrue(loaded.tokens.grant_id)
        self.assertTrue(await self.storage.remove_anthropic_subscription())
        self.assertIsNone(self.storage.load_anthropic_subscription())
        self.assertFalse(await self.storage.remove_anthropic_subscription())

    async def test_rotate_persists_rotated_tokens_durable(self):
        stored = await self.storage.store_anthropic_login(anthropic_tokens())
        observed = []

        async def refresh(current):
            observed.append((current.refresh_token, current.scopes, current.client_id))
            self.assertEqual(
                self.storage.load_anthropic_subscription().state,
                "refreshing")
            return authentications.AnthropicRefreshResult(
                access_token="anthropic-access-2",
                refresh_token="anthropic-refresh-2",
                expires_in=3600,
            )

        rotated = await self.storage.rotate_anthropic_subscription(
            stored.tokens, refresh=refresh)

        self.assertEqual(
            observed, [("anthropic-refresh-secret",
                        authentications.ANTHROPIC_OAUTH_SCOPES, CLIENT_ID)])
        self.assertEqual(rotated.refresh_token, "anthropic-refresh-2")
        durable = credential_storages.JsonCredentialStorage(
            self.storage.directory)
        with contextlib.closing(durable):
            loaded = durable.load_anthropic_subscription()
            self.assertEqual(loaded.state, "active")
            self.assertEqual(loaded.tokens.access_token, "anthropic-access-2")
            self.assertEqual(loaded.tokens.refresh_token,
                             "anthropic-refresh-2")
            self.assertEqual(loaded.tokens.expires_at,
                             rotated.expires_at)

    async def test_permanent_refresh_failure_requires_login(self):
        stored = await self.storage.store_anthropic_login(anthropic_tokens())

        async def refresh(_current):
            self.assertEqual(
                self.storage.load_anthropic_subscription().state,
                "refreshing")
            raise authentications.RefreshPermanentError("revoked")

        with self.assertRaises(authentications.RefreshPermanentError):
            await self.storage.rotate_anthropic_subscription(
                stored.tokens, refresh=refresh)
        self.assertEqual(
            self.storage.load_anthropic_subscription().state,
            "reauth-required")

    async def test_rotation_leaves_other_credentials_writable_and_uses_response_time(self):
        stored = await self.storage.store_anthropic_login(anthropic_tokens())
        started, release = asyncio.Event(), asyncio.Event()
        now = 1000
        calls = []
        reopened = credential_storages.JsonCredentialStorage(self.storage.directory)
        self.addCleanup(reopened.close)

        async def refresh(current):
            calls.append(current.refresh_token)
            started.set()
            await release.wait()
            return authentications.AnthropicRefreshResult("new-access", "new-refresh", expires_in=3600)

        contended = asyncio.Event()
        real_lock = file_locks.try_lock_exclusive
        writer = asyncio.current_task()

        def observe_lock(fd):
            try:
                return real_lock(fd)
            except BlockingIOError:
                if asyncio.current_task() is writer:
                    self.fail("Anthropic network I/O held the shared document lock")
                contended.set()
                raise

        async with running_fixture_task(
                self.storage.rotate_anthropic_subscription(stored.tokens, refresh, lambda: now)) as first:
            await observe_fixture(started.wait(), first)
            with mock.patch.object(file_locks, "try_lock_exclusive", observe_lock):
                async with running_fixture_task(
                        reopened.rotate_anthropic_subscription(stored.tokens, refresh, lambda: now)) as second:
                    await observe_fixture(contended.wait(), second, first)
                    pending = self.storage.load_anthropic_subscription()
                    self.assertEqual(pending.state, "refreshing")
                    self.assertIsNone(pending.tokens)
                    await self.storage.store_openai_login(authentications.OpenAITokenSet("a", "r"))
                    now = 2000
                    release.set()
                    first_result, second_result = await asyncio.gather(first, second)

        self.assertEqual(calls, [stored.tokens.refresh_token])
        self.assertEqual(first_result, second_result)
        self.assertEqual(first_result.expires_at, 5600)
        self.assertIsNotNone(self.storage.load_openai_subscription())

    async def test_logout_during_refresh_does_not_resurrect_the_grant(self):
        stored = await self.storage.store_anthropic_login(anthropic_tokens())
        started, release = asyncio.Event(), asyncio.Event()

        async def refresh(_current):
            started.set()
            await release.wait()
            return authentications.AnthropicRefreshResult("new-access", "new-refresh")

        real_lock = file_locks.try_lock_exclusive

        def require_unlocked_document(fd):
            try:
                return real_lock(fd)
            except BlockingIOError:
                self.fail("Logout could not acquire the document lock during refresh")

        async with running_fixture_task(self.storage.rotate_anthropic_subscription(stored.tokens, refresh)) as task:
            await observe_fixture(started.wait(), task)
            with mock.patch.object(file_locks, "try_lock_exclusive", require_unlocked_document):
                self.assertTrue(await self.storage.remove_anthropic_subscription())
            release.set()
            with self.assertRaisesRegex(authentications.RefreshPermanentError, "login changed"):
                await task
        self.assertIsNone(self.storage.load_anthropic_subscription())

    async def test_rotation_cancellation_agrees_with_durable_state(self):
        for sent in [False, True]:
            with self.subTest(sent=sent):
                stored = await self.storage.store_anthropic_login(anthropic_tokens())
                started = asyncio.Event()

                async def refresh(_current):
                    started.set()
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError as error:
                        error.request_may_have_been_sent = sent
                        raise

                async def rotate(current):
                    return await self.storage.rotate_anthropic_subscription(current, refresh)

                record = authentications.AnthropicSubscriptionCredential(stored.tokens, rotate=rotate)
                async with running_fixture_task(record.lease(rejected_generation=0)) as task:
                    await observe_fixture(started.wait(), task)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                durable = self.storage.load_anthropic_subscription()
                if sent:
                    self.assertEqual(durable.state, "reauth-required")
                    with self.assertRaises(authentications.RefreshIndeterminateError):
                        await record.lease()
                else:
                    self.assertEqual(durable.state, "active")
                    self.assertEqual(durable.tokens, stored.tokens)
                    self.assertEqual((await record.lease()).generation, 0)

    async def test_separate_login_is_not_adopted_by_an_existing_session(self):
        stored = await self.storage.store_anthropic_login(anthropic_tokens(account=None))
        replacement = await self.storage.store_anthropic_login(anthropic_tokens(account=None, access_token="other"))
        refresh = mock.AsyncMock()
        with self.assertRaises(authentications.RefreshPermanentError):
            await self.storage.rotate_anthropic_subscription(stored.tokens, refresh)
        refresh.assert_not_awaited()
        self.assertEqual(self.storage.load_anthropic_subscription(), replacement)

    async def test_supervisor_installs_stored_subscription(self):
        await self.storage.store_anthropic_login(anthropic_tokens())
        reopened = credential_storages.JsonCredentialStorage(
            self.storage.directory)
        with contextlib.closing(reopened):
            supervisor = credential_supervisors.CredentialSupervisor(
                CredentialStore({}), reopened)
            ref = authentications.CredentialRef.anthropic_subscription()
            self.assertIn(ref, supervisor.broker.available())
            self.assertTrue(supervisor.inventory.has_ref(ref))

    async def test_startup_delegates_rotated_login(self):
        await self._startup_during_refresh(logout=False)

    async def test_startup_observes_logout_without_waiting_for_obsolete_refresh(self):
        await self._startup_during_refresh(logout=True)

    async def _startup_during_refresh(self, *, logout):
        stored = await self.storage.store_anthropic_login(anthropic_tokens())
        refreshing, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
        reopened = self.enterContext(contextlib.closing(
            credential_storages.JsonCredentialStorage(self.storage.directory)))
        ref = authentications.CredentialRef.anthropic_subscription()

        async def refresh(_current):
            refreshing.set()
            await release.wait()
            return authentications.AnthropicRefreshResult("new-access", "new-refresh", expires_in=3600)

        real_lock = file_locks.try_lock_exclusive

        def observe_lock(fd):
            try:
                return real_lock(fd)
            except BlockingIOError:
                waiting.set()
                raise

        async with running_fixture_task(self.storage.rotate_anthropic_subscription(stored.tokens, refresh)) as rotation:
            await observe_fixture(refreshing.wait(), rotation)
            supervisor = credential_supervisors.CredentialSupervisor(CredentialStore({}), reopened)

            async def delegate():
                delegation = await supervisor.delegate()
                try:
                    return delegation.credential_server.allowed
                finally:
                    await delegation.close()

            with mock.patch.object(file_locks, "try_lock_exclusive", observe_lock):
                async with running_fixture_task(delegate()) as startup:
                    await observe_fixture(waiting.wait(), startup, rotation)
                    self.assertFalse(startup.done())
                    if logout:
                        self.assertTrue(await self.storage.remove_anthropic_subscription())
                        self.assertNotIn(ref, await startup)
                        self.assertFalse(rotation.done())
                        self.assertFalse(supervisor.inventory.has_ref(ref))
                        release.set()
                        with self.assertRaises(authentications.RefreshPermanentError):
                            await rotation
                    else:
                        release.set()
                        await rotation
                        self.assertIn(ref, await startup)
                        self.assertTrue(supervisor.inventory.has_ref(ref))
                        lease = await supervisor.broker.lease(ref, authentications.ANTHROPIC_BETA_MESSAGES_URL)
                        self.assertEqual(lease.value, "new-access")

    async def test_startup_leaves_an_orphaned_refresh_unavailable(self):
        stored = await self.storage.store_anthropic_login(anthropic_tokens())
        refresh = mock.AsyncMock(return_value=authentications.AnthropicRefreshResult("new-access", "new-refresh"))
        with mock.patch.object(self.storage, "_finish_anthropic_attempt",
                               side_effect=credential_storages.CredentialStorageError("could not write")):
            with self.assertRaises(authentications.RefreshIndeterminateError):
                await self.storage.rotate_anthropic_subscription(stored.tokens, refresh)

        supervisor = credential_supervisors.CredentialSupervisor(CredentialStore({}), self.storage)
        await supervisor.prepare_startup()
        self.assertFalse(supervisor.inventory.has_ref(authentications.CredentialRef.anthropic_subscription()))
        self.assertEqual(self.storage.load_anthropic_subscription().state, "refreshing")


def catalog_fixture():
    return {"anthropic": {
        "id": "anthropic", "name": "Anthropic", "npm": "@ai-sdk/anthropic",
        "api": authentications.ANTHROPIC_MESSAGES_URL, "env": ["ANTHROPIC_API_KEY"],
        "models": {"claude-sonnet-4-6": {
            "id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6",
            "provider": {"api": "https://elsewhere.invalid/v1"},
            "cost": {"input": 3}, "limit": {"context": 200000},
        }},
    }}


class AnthropicConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_subscription_entry_requires_login_and_stays_out_of_public_cache(self):
        catalog = catalog_fixture()
        provider_id = models.ANTHROPIC_SUBSCRIPTION_PROVIDER_ID
        catalog[provider_id] = {"name": "pretend subscription", "_loki_synthetic": []}
        self.assertFalse(models.provider_is_synthetic(catalog[provider_id]))
        broker = authentications.CredentialBroker()

        with mock.patch.object(models, "_index_cache", None), mock.patch.object(
                models, "fetch_models_dev", new=mock.AsyncMock(return_value=catalog)):
            before, _ = await models.ensure_index(credential_authority=broker)
            broker.install_anthropic_subscription(anthropic_tokens())
            after, _ = await models.ensure_index(credential_authority=broker)
            anonymous, _ = await models.ensure_index()

        self.assertNotIn(provider_id, before)
        self.assertTrue(models.provider_is_synthetic(after[provider_id]))
        self.assertNotIn(provider_id, anonymous)

    def test_subscription_streaming_default_and_opt_out(self):
        catalog = catalog_fixture()
        model = "claude-sonnet-4-6"
        # Compare the same model on the two first-party Anthropic routes.
        catalog["anthropic"]["models"][model].pop("provider")
        catalog = models.add_anthropic_subscription_catalog(catalog)
        subscription = models.ANTHROPIC_SUBSCRIPTION_PROVIDER_ID
        refs = [
            authentications.CredentialRef.anthropic_subscription(),
            authentications.CredentialRef.environment("ANTHROPIC_API_KEY"),
        ]
        no_override = CredentialInventory({}, refs)
        for provider_id, settings, expected in [
                (subscription, {}, True),
                (subscription, {"LOKI_STREAM": "0"}, False),
                ("anthropic", {}, False)]:
            with self.subTest(provider_id=provider_id, settings=settings):
                provider = catalog[provider_id]
                config = connections.config_from_modelsdev_selection(
                    provider_id, provider, provider["models"][model],
                    CredentialInventory(settings, refs))
                self.assertIs(config.stream, expected)
                descriptor = connections.connection_descriptor_from_config(config)
                restored = connections.config_from_connection_descriptor(
                    descriptor, no_override)
                self.assertIs(restored.stream, expected)

    async def test_selection_and_saved_connection_keep_separate_url_sets(self):
        raw_catalog = catalog_fixture()
        catalog = models.add_anthropic_subscription_catalog(raw_catalog)
        ref = authentications.CredentialRef.anthropic_subscription()
        inventory = CredentialInventory({}, [ref])
        provider_id = models.ANTHROPIC_SUBSCRIPTION_PROVIDER_ID
        provider = catalog[provider_id]
        leaf = provider["models"]["claude-sonnet-4-6"]
        self.assertNotIn("cost", leaf)
        self.assertIn("cost", raw_catalog["anthropic"]["models"]["claude-sonnet-4-6"])
        config = connections.config_from_modelsdev_selection(provider_id, provider, leaf, inventory)
        saved = connections.connection_descriptor_from_config(config).to_dict()
        restored = connections.config_from_connection_descriptor(connections.ConnectionDescriptor.from_dict(saved), inventory)
        for connection in [config, restored]:
            # Admission includes known account reads, not an unauthorized
            # model-list route. Inference owns only the beta Messages URL.
            self.assertEqual(connection.credential_destinations(), [
                authentications.ANTHROPIC_BETA_MESSAGES_URL,
                anthropic_controls.ANTHROPIC_USAGE_URL,
            ])
            self.assertEqual(connection.chat_provider.chat_url,
                             authentications.ANTHROPIC_BETA_MESSAGES_URL)
            self.assertEqual(connection.auth_spec.authorized_urls, authentications.ANTHROPIC_SUBSCRIPTION_URLS)
            self.assertEqual(connection.auth_spec.credential, ref)
            self.assertEqual(connection.context_capacity.source, "models.dev")
            self.assertIsNone(connection.chat_provider.models_url)
            self.assertEqual(connection.chat_provider.model_urls, [])


class AnthropicInferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_inference_401_replays_with_rotated_oauth_in_both_executors(self):
        url = authentications.ANTHROPIC_BETA_MESSAGES_URL
        ref = authentications.CredentialRef.anthropic_subscription()
        config = connections.make_runtime_config(
            url, protocols.ANTHROPIC_MESSAGES, model="claude-sonnet-4-6",
            credential_ref=ref, provider_id=models.ANTHROPIC_SUBSCRIPTION_PROVIDER_ID,
        )
        payload = {"model": config.model, "system": "be helpful", "messages": [{"role": "user", "content": "hi"}]}
        for streaming in [False, True]:
            with self.subTest(streaming=streaming):
                calls = []
                broker = authentications.CredentialBroker()
                rotate = mock.AsyncMock(return_value=anthropic_tokens(access_token="rotated-access"))
                broker.install_anthropic_subscription(anthropic_tokens(), rotate=rotate)
                observations = mock.Mock(observer=mock.Mock(return_value=None))
                context = model_requests.RequestContext(config, broker, "conversation", "root", observations)

                async def request(method, endpoint, **kwargs):
                    self.assertEqual((method, endpoint), ("POST", url))
                    headers = {name.lower(): value for name, value in kwargs["headers_in"].items()}
                    calls.append((headers, kwargs["body"], kwargs.get("user_agent")))
                    return response(endpoint, 401 if len(calls) == 1 else 200, {"content": []})

                @contextlib.asynccontextmanager
                async def stream(method, endpoint, **kwargs):
                    reply = await request(method, endpoint, **kwargs)

                    async def body():
                        if reply.status == 401:
                            yield reply.body
                        else:
                            yield (b'data: {"type":"message_start","message":{"content":[]}}\n\n'
                                   b'data: {"type":"message_stop"}\n\n')

                    yield http_client.HttpStreamResponse(url, reply.status, "", {"content-type": "text/event-stream"}, body())

                with mock.patch.object(http_client, "async_http_request", request), \
                        mock.patch.object(http_client, "async_http_stream", stream):
                    if streaming:
                        await model_requests.async_chat_stream_request(url, payload, context=context)
                    else:
                        await model_requests.async_provider_request("POST", url, payload, context=context)

                rotate.assert_awaited_once()
                self.assertEqual([headers["authorization"] for headers, _, _ in calls], [
                    "Bearer anthropic-access-secret", "Bearer rotated-access",
                ])
                first_headers, first_body, first_agent = calls[0]
                # A retry replays the exact prepared body, including the
                # billing block added for the subscription scheme.
                self.assertEqual(first_body, calls[1][1])
                sent = json.loads(first_body)
                self.assertEqual(sent["messages"], payload["messages"])
                self.assertTrue(
                    sent["system"][0]["text"].startswith(
                        "x-anthropic-billing-header: cc_version=2.1.325."))
                self.assertEqual(
                    sent["system"][1:], [{"type": "text", "text": "be helpful"}])
                header = model_requests.LLM_IDEMPOTENCY_HEADER_ANTHROPIC
                self.assertEqual(first_headers[header], calls[1][0][header])
                self.assertEqual(first_headers["anthropic-beta"], authentications.ANTHROPIC_OAUTH_BETA)
                self.assertEqual(first_headers["anthropic-dangerous-direct-browser-access"], "true")
                self.assertEqual(first_headers["x-app"], "cli")
                self.assertEqual(first_headers["x-claude-code-session-id"], "root")
                # The transport owns User-Agent, so the identity is selected
                # through its argument rather than injected as a header.
                self.assertEqual(
                    first_agent, authentications.ANTHROPIC_CLIENT_USER_AGENT)
                self.assertNotIn("user-agent", first_headers)
                self.assertNotIn("x-api-key", first_headers)

    async def test_api_key_requests_keep_loki_identity_and_no_billing(self):
        url = authentications.ANTHROPIC_MESSAGES_URL
        ref = authentications.CredentialRef.environment("ANTHROPIC_API_KEY")
        broker = authentications.CredentialBroker()
        broker.install_static(ref, "ordinary-api-key")
        broker.allow_startup_destinations(ref, [url])
        config = connections.make_runtime_config(
            url, protocols.ANTHROPIC_MESSAGES, credential_ref=ref)
        context = model_requests.RequestContext(
            config, broker, "conversation", "root",
            mock.Mock(observer=mock.Mock(return_value=None)))
        seen = {}

        async def request(method, endpoint, **kwargs):
            seen.update(kwargs)
            return response(endpoint, 200, {"content": []})

        payload = {"model": "claude", "system": "be helpful", "messages": []}
        with mock.patch.object(http_client, "async_http_request", request):
            await model_requests.async_provider_request(
                "POST", url, payload, context=context)

        self.assertEqual(json.loads(seen["body"]), payload)
        self.assertNotIn("user_agent", seen)
        lowered = {name.lower() for name in seen["headers_in"]}
        self.assertNotIn("x-app", lowered)
        self.assertNotIn("user-agent", lowered)

    async def test_api_key_still_uses_key_header_without_oauth_protocol(self):
        url = authentications.ANTHROPIC_MESSAGES_URL
        ref = authentications.CredentialRef.environment("ANTHROPIC_API_KEY")
        broker = authentications.CredentialBroker()
        broker.install_static(ref, "ordinary-api-key")
        broker.allow_startup_destinations(ref, [url])
        config = connections.make_runtime_config(url, protocols.ANTHROPIC_MESSAGES, credential_ref=ref)
        headers, _ = await authentications.authorized_request_headers(broker, config.auth_spec, url)
        self.assertEqual(headers, {"x-api-key": "ordinary-api-key"})


class AnthropicLoginCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.storage = credential_storages.JsonCredentialStorage(self.temporary.name)
        self.addCleanup(self.storage.close)

    async def test_browser_login_status_and_logout(self):
        output, errors = io.StringIO(), io.StringIO()
        requests = []
        loop_thread = threading.get_ident()

        def browser(url, *, new):
            self.assertNotEqual(threading.get_ident(), loop_thread)
            self.assertEqual(new, 2)
            query = urllib.parse.parse_qs(
                urllib.parse.urlsplit(url).query)
            redirect = urllib.parse.urlsplit(query["redirect_uri"][0])
            self.assertEqual(redirect.path, "/callback")
            target = redirect.path + "?" + urllib.parse.urlencode({
                "code": "authorization-secret",
                "state": query["state"][0]})
            with contextlib.closing(http.client.HTTPConnection(
                    redirect.hostname, redirect.port)) as connection:
                connection.request("GET", target)
                with connection.getresponse() as answer:
                    self.assertEqual(answer.status, 200)
            return True

        async def request(method, url, **kwargs):
            requests.append(url)
            self.assertEqual(method, "POST")
            self.assertEqual(kwargs["retry_max_attempts"], 1)
            self.assertEqual(url, authentications.ANTHROPIC_TOKEN_URL)
            return response(url, 200, token_response())

        with mock.patch.object(http_client, "async_http_request",
                               new=request), \
                mock.patch.object(
                    authentication_commands.webbrowser, "open",
                    side_effect=browser) as launcher, \
                contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(errors):
            self.assertEqual(
                await authentication_commands.run(
                    ["login", "anthropic", "--client-id", CLIENT_ID], storage=self.storage), 0)
            launcher.assert_called_once()
            self.assertEqual(requests, [authentications.ANTHROPIC_TOKEN_URL])
            stored = self.storage.load_anthropic_subscription()
            self.assertEqual(stored.state, "active")
            self.assertEqual(
                stored.tokens.access_token, "anthropic-access-secret")
            self.assertEqual(
                stored.tokens.account.email_address, "user@example.test")
            self.assertEqual(
                await authentication_commands.run(
                    ["status", "anthropic"], storage=self.storage), 0)
            self.assertEqual(
                await authentication_commands.run(
                    ["logout", "anthropic"], storage=self.storage), 0)
            self.assertEqual(
                await authentication_commands.run(
                    ["status", "anthropic"], storage=self.storage), 1)
        self.assertEqual(errors.getvalue(), "")
        self.assertIn("Logged in: Anthropic subscription", output.getvalue())
        for secret in ["anthropic-access-secret", "anthropic-refresh-secret",
                       "authorization-secret"]:
            self.assertNotIn(secret, output.getvalue())

    async def test_login_without_options_uses_the_bundled_client(self):
        errors = io.StringIO()
        start = mock.AsyncMock(side_effect=oauth_logins.OAuthLoginError("stop"))
        with mock.patch.object(
                oauth_logins, "start_anthropic_browser_login", new=start), \
                contextlib.redirect_stderr(errors), \
                self.assertRaises(oauth_logins.OAuthLoginError):
            await authentication_commands.run(
                ["login", "anthropic"], storage=self.storage)
        start.assert_awaited_once_with(client_id=None)

    async def test_client_id_override_is_stored_and_reused_for_refresh(self):
        output, errors = io.StringIO(), io.StringIO()
        requests = []

        def browser(url, *, new):
            query = urllib.parse.parse_qs(
                urllib.parse.urlsplit(url).query)
            self.assertEqual(query["client_id"], ["other-registration"])
            redirect = urllib.parse.urlsplit(query["redirect_uri"][0])
            with contextlib.closing(http.client.HTTPConnection(
                    redirect.hostname, redirect.port)) as connection:
                connection.request("GET", redirect.path + "?" +
                                   urllib.parse.urlencode({
                                       "code": "code",
                                       "state": query["state"][0]}))
                with connection.getresponse() as answer:
                    self.assertEqual(answer.status, 200)
            return True

        async def request(method, url, **kwargs):
            requests.append(json.loads(kwargs["body"]))
            return response(url, 200, token_response())

        with mock.patch.object(http_client, "async_http_request", new=request), \
                mock.patch.object(
                    authentication_commands.webbrowser, "open",
                    side_effect=browser), \
                contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(errors):
            self.assertEqual(
                await authentication_commands.run(
                    ["login", "anthropic", "--client-id", "other-registration"],
                    storage=self.storage), 0)
            stored = self.storage.load_anthropic_subscription()
            refreshed = await authentications.request_anthropic_token_refresh(
                stored.tokens)

        self.assertEqual(stored.tokens.client_id, "other-registration")
        self.assertEqual(requests[0]["client_id"], "other-registration")
        self.assertEqual(refreshed.access_token, "anthropic-access-secret")
        self.assertEqual(requests[1]["client_id"], "other-registration")

    async def test_unqualified_status_succeeds_with_only_openai_logged_in(self):
        await self.storage.store_openai_login(authentications.OpenAITokenSet("a", "r"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(await authentication_commands.run(["status"], storage=self.storage), 0)
        self.assertIn("OpenAI ChatGPT subscription: logged in", output.getvalue())
        self.assertIn("Anthropic subscription: not logged in", output.getvalue())

    async def test_unqualified_status_reports_healthy_peer_and_bad_record(self):
        await self.storage.store_anthropic_login(anthropic_tokens())
        await self.storage.store_openai_login(authentications.OpenAITokenSet("a", "r"))
        original = self.storage.load_document()
        for provider, key, invalid_record, healthy_status in [
                ("anthropic", credential_storages.ANTHROPIC_CREDENTIAL_KEY, None,
                 "OpenAI ChatGPT subscription: logged in"),
                ("openai", credential_storages.OPENAI_CREDENTIAL_KEY,
                 {"type": credential_storages.OPENAI_RECORD_TYPE, "state": []},
                 "Anthropic subscription: logged in")]:
            with self.subTest(provider=provider):
                document = dict(original, credentials=dict(original["credentials"]))
                document["credentials"][key] = invalid_record
                with open(self.storage.file_path, "w", encoding="utf-8") as stream:
                    json.dump(document, stream)
                output, errors = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    self.assertEqual(await authentication_commands.run(["status"], storage=self.storage), 0)
                self.assertIn(healthy_status, output.getvalue())
                self.assertIn(f"{provider}: saved login is invalid", errors.getvalue())

                # With no healthy peer the same bad record remains an error.
                other = next(name for name in document["credentials"] if name != key)
                del document["credentials"][other]
                with open(self.storage.file_path, "w", encoding="utf-8") as stream:
                    json.dump(document, stream)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(await authentication_commands.run(["status"], storage=self.storage), 2)

    async def test_unqualified_status_does_not_ignore_damage_to_shared_document(self):
        await self.storage.store_anthropic_login(anthropic_tokens())
        document = self.storage.load_document()
        document["version"] += 1
        with open(self.storage.file_path, "w", encoding="utf-8") as stream:
            json.dump(document, stream)
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            with self.assertRaisesRegex(credential_storages.CredentialStorageError, "format version"):
                await authentication_commands.run(["status"], storage=self.storage)
        self.assertEqual(output.getvalue(), "")

    async def test_device_code_is_rejected_for_anthropic(self):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(errors):
            self.assertEqual(
                await authentication_commands.run(
                    ["login", "anthropic", "--device-code"],
                    storage=self.storage), 2)
        self.assertIn("device-code", errors.getvalue())
        self.assertIsNone(self.storage.load_anthropic_subscription())
