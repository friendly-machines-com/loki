import asyncio
import base64
import json
import unittest
from unittest import mock

from loki_agent import authentications
from loki_agent import http_client


auth = authentications


def rotation(refresh, now=None):
    async def rotate(tokens):
        result = await refresh(tokens.refresh_token)
        return auth.refreshed_openai_tokens(
            tokens, result, now)

    return rotate


def jwt(payload):
    def encode(value):
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


class CredentialBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_token_shapes_are_validated(self):
        for tokens in [
                auth.OpenAITokenSet(1, "refresh"),
                auth.OpenAITokenSet("access", ""),
                auth.OpenAITokenSet(
                    "access", "refresh", expires_at=float("nan")),
                auth.OpenAITokenSet(
                    "access", "refresh", fedramp="yes"),
        ]:
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):
                    auth.OpenAIChatGPTCredential(tokens)

    async def test_static_credential_never_refreshes(self):
        broker = auth.CredentialBroker()
        ref = auth.CredentialRef.environment("EXAMPLE_API_KEY")
        broker.install_static(ref, "secret")

        lease = await broker.lease(ref, rejected_generation=0)

        self.assertEqual(lease.value, "secret")
        self.assertFalse(lease.refreshable)
        self.assertEqual(broker.available(), frozenset({ref}))

    async def test_subscription_target_is_confined_to_chatgpt_codex(self):
        spec = auth.AuthSpec(
            auth.CredentialRef.openai_subscription(),
            "openai-subscription",
        )

        self.assertEqual(
            auth.OPENAI_CHATGPT_MODELS_REQUEST_URL,
            "https://chatgpt.com/backend-api/codex/models"
            "?client_version=0.144.0",
        )
        for url in auth.OPENAI_CHATGPT_AUTHORIZED_URLS:
            auth.validate_authorization_target(spec, url)
        for url in [
                "http://chatgpt.com/backend-api/codex/responses",
                "https://chatgpt.com:443/backend-api/codex/responses",
                "https://chatgpt.com/backend-api/codex/responses?x=1",
                auth.OPENAI_CHATGPT_MODELS_URL,
                auth.OPENAI_CHATGPT_MODELS_URL + "?client_version=999.0.0",
                "https://chatgpt.com.evil.test/backend-api/codex/responses",
                "https://evil.test/backend-api/codex/responses",
        ]:
            with self.subTest(url=url):
                with self.assertRaises(auth.CredentialUnavailable):
                    auth.validate_authorization_target(spec, url)

    async def test_concurrent_proactive_refresh_is_single_flight(self):
        now = 10_000.0
        calls = []
        gate = asyncio.Event()

        async def refresh(refresh_token):
            calls.append(refresh_token)
            await gate.wait()
            return auth.RefreshResult(
                access_token=jwt({"exp": now + 3600}),
                refresh_token="refresh-b",
            )

        broker = auth.CredentialBroker()
        ref = auth.CredentialRef.openai_subscription()
        broker.install_openai_subscription(
            auth.OpenAITokenSet(
                access_token=jwt({"exp": now + 10}),
                refresh_token="refresh-a",
            ),
            rotate=rotation(refresh, now),
            clock=lambda: now,
        )

        tasks = [asyncio.create_task(broker.lease(ref)) for _ in range(8)]
        await asyncio.sleep(0)
        gate.set()
        leases = await asyncio.gather(*tasks)

        self.assertEqual(calls, ["refresh-a"])
        self.assertEqual({lease.generation for lease in leases}, {1})
        await broker.lease(ref, rejected_generation=1)
        self.assertEqual(calls, ["refresh-a", "refresh-b"])

    async def test_stale_401_does_not_refresh_new_generation(self):
        now = 10_000.0
        calls = []

        async def refresh(refresh_token):
            calls.append(refresh_token)
            return auth.RefreshResult(
                access_token=jwt({"exp": now + 3600}),
                refresh_token="refresh-b",
            )

        broker = auth.CredentialBroker()
        ref = auth.CredentialRef.openai_subscription()
        broker.install_openai_subscription(
            auth.OpenAITokenSet(
                access_token=jwt({"exp": now + 3600}),
                refresh_token="refresh-a",
            ),
            rotate=rotation(refresh, now),
            clock=lambda: now,
        )

        first = await broker.lease(ref)
        refreshed = await broker.lease(
            ref, rejected_generation=first.generation)
        stale = await broker.lease(
            ref, rejected_generation=first.generation)

        self.assertEqual(calls, ["refresh-a"])
        self.assertEqual(refreshed.generation, 1)
        self.assertEqual(stale.generation, 1)

    async def test_proactive_transient_failure_uses_unexpired_token(self):
        now = 10_000.0

        async def refresh(_refresh_token):
            raise auth.RefreshTransientError(
                "offline", request_may_have_been_sent=False)

        record = auth.OpenAIChatGPTCredential(
            auth.OpenAITokenSet(
                access_token=jwt({"exp": now + 30}),
                refresh_token="refresh-a",
            ),
            rotate=rotation(refresh, now),
            clock=lambda: now,
        )

        lease = await record.lease()

        self.assertEqual(lease.generation, 0)

    async def test_ambiguous_refresh_is_never_replayed(self):
        now = 10_000.0
        calls = 0

        async def refresh(_refresh_token):
            nonlocal calls
            calls += 1
            raise auth.RefreshTransientError(
                "connection lost", request_may_have_been_sent=True)

        record = auth.OpenAIChatGPTCredential(
            auth.OpenAITokenSet(
                access_token=jwt({"exp": now - 1}),
                refresh_token="refresh-a",
            ),
            rotate=rotation(refresh, now),
            clock=lambda: now,
        )

        with self.assertRaises(auth.RefreshIndeterminateError):
            await record.lease()
        with self.assertRaises(auth.RefreshIndeterminateError):
            await record.lease()

        self.assertEqual(calls, 1)

    async def test_cancelled_ambiguous_refresh_is_never_replayed(self):
        now = 10_000.0
        calls = 0
        started = asyncio.Event()

        async def refresh(_refresh_token):
            nonlocal calls
            calls += 1
            started.set()
            await asyncio.Event().wait()

        record = auth.OpenAIChatGPTCredential(
            auth.OpenAITokenSet(
                access_token=jwt({"exp": now - 1}),
                refresh_token="refresh-a",
            ),
            rotate=rotation(refresh, now),
            clock=lambda: now,
        )
        pending = asyncio.create_task(record.lease())
        await started.wait()
        pending.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await pending
        with self.assertRaises(auth.RefreshIndeterminateError):
            await record.lease()

        self.assertEqual(calls, 1)

    async def test_refresh_cannot_switch_account(self):
        now = 10_000.0

        async def refresh(_refresh_token):
            return auth.RefreshResult(
                access_token=jwt({
                    "exp": now + 3600,
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "other",
                    },
                }),
                refresh_token="refresh-b",
            )

        record = auth.OpenAIChatGPTCredential(
            auth.OpenAITokenSet(
                access_token=jwt({"exp": now - 1}),
                refresh_token="refresh-a",
                account_id="expected",
            ),
            rotate=rotation(refresh, now),
            clock=lambda: now,
        )

        with self.assertRaises(auth.RefreshPermanentError):
            await record.lease()

    async def test_explicit_replacement_wins_over_in_flight_refresh(self):
        now = 10_000.0
        started = asyncio.Event()
        release = asyncio.Event()

        async def refresh(_refresh_token):
            started.set()
            await release.wait()
            return auth.RefreshResult(
                access_token=jwt({"exp": now + 3600}),
                refresh_token="stale-refresh-result",
            )

        record = auth.OpenAIChatGPTCredential(
            auth.OpenAITokenSet(
                access_token=jwt({"exp": now - 1}),
                refresh_token="old-refresh",
            ),
            rotate=rotation(refresh, now),
            clock=lambda: now,
        )
        pending = asyncio.create_task(record.lease())
        await started.wait()
        record.replace(auth.OpenAITokenSet(
            access_token=jwt({"exp": now + 7200}),
            refresh_token="replacement-refresh",
        ))
        release.set()

        lease = await pending

        self.assertEqual(record.tokens.refresh_token, "replacement-refresh")
        self.assertEqual(lease.value, record.tokens.access_token)

    async def test_initial_tokens_derive_account_id(self):
        record = auth.OpenAIChatGPTCredential(
            auth.OpenAITokenSet(
                access_token=jwt({
                    "exp": 20_000,
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "account",
                    },
                }),
                refresh_token="refresh",
            ),
            clock=lambda: 10_000,
        )

        lease = await record.lease()

        self.assertEqual(lease.account_id, "account")

    async def test_standard_invalid_grant_is_permanent(self):
        response = http_client.HttpResponse(
            auth.OPENAI_REFRESH_URL,
            400,
            "Bad Request",
            {"content-type": "application/json"},
            b'{"error":"invalid_grant"}',
        )
        with mock.patch.object(
                http_client, "async_http_request",
                new=mock.AsyncMock(return_value=response)):
            with self.assertRaises(auth.RefreshPermanentError):
                await auth.request_openai_token_refresh("refresh")

    async def test_http_error_response_makes_refresh_outcome_indeterminate(self):
        for body in (b'{"error":"temporarily_unavailable"}', b"not json"):
            with self.subTest(body=body):
                response = http_client.HttpResponse(
                    auth.OPENAI_REFRESH_URL,
                    503,
                    "Service Unavailable",
                    {"content-type": "application/json"},
                    body,
                )
                with mock.patch.object(
                        http_client, "async_http_request",
                        new=mock.AsyncMock(return_value=response)):
                    with self.assertRaises(
                            auth.RefreshTransientError) as raised:
                        await auth.request_openai_token_refresh("refresh")

                self.assertTrue(
                    raised.exception.request_may_have_been_sent)

    async def test_refresh_exchange_is_one_shot_and_has_no_api_key_header(self):
        response = http_client.HttpResponse(
            auth.OPENAI_REFRESH_URL,
            200,
            "OK",
            {"content-type": "application/json"},
            (
                b'{"access_token":"access-b",'
                b'"refresh_token":"refresh-b"}'
            ),
        )
        request = mock.AsyncMock(return_value=response)

        with mock.patch.object(
                http_client, "async_http_request", new=request):
            result = await auth.request_openai_token_refresh("refresh-a")

        _method, url = request.await_args.args
        kwargs = request.await_args.kwargs
        self.assertEqual(_method, "POST")
        self.assertEqual(url, auth.OPENAI_REFRESH_URL)
        self.assertEqual(kwargs["retry_max_attempts"], 1)
        self.assertEqual(
            json.loads(kwargs["body"]),
            {
                "client_id": auth.OPENAI_OAUTH_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": "refresh-a",
            },
        )
        self.assertNotIn("Authorization", kwargs["headers_in"])
        self.assertEqual(
            kwargs["headers_in"]["originator"],
            auth.OPENAI_ORIGINATOR,
        )
        self.assertEqual(result.refresh_token, "refresh-b")


class AuthorizationHeaderTests(unittest.TestCase):
    def test_openai_subscription_headers(self):
        ref = auth.CredentialRef.openai_subscription()
        headers = auth.authorization_headers(
            auth.AuthSpec(ref, "openai-subscription"),
            auth.CredentialLease(
                ref,
                "access",
                account_id="account",
                fedramp=True,
            ),
        )

        self.assertEqual(headers, {
            "Authorization": "Bearer access",
            "originator": "loki",
            "ChatGPT-Account-ID": "account",
            "X-OpenAI-Fedramp": "true",
        })

    def test_lease_repr_redacts_value(self):
        lease = auth.CredentialLease(
            auth.CredentialRef.environment("EXAMPLE_API_KEY"),
            "do-not-print",
        )

        self.assertNotIn("do-not-print", repr(lease))


class AuthorizedRequestHeaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_unauthenticated_request_copies_base_headers(self):
        base = {"Accept": "application/json"}

        headers, lease = await auth.authorized_request_headers(
            None, None, "https://example.test/data", base)

        self.assertEqual(headers, base)
        self.assertIsNot(headers, base)
        self.assertIsNone(lease)

    async def test_authorizes_subscription_target_as_one_operation(self):
        broker = auth.CredentialBroker()
        ref = auth.CredentialRef.openai_subscription()
        broker.install_openai_subscription(auth.OpenAITokenSet(
            "access", "refresh",
            account_id="account",
            expires_at=10**12,
        ))

        headers, lease = await auth.authorized_request_headers(
            broker,
            auth.AuthSpec(ref, "openai-subscription"),
            auth.OPENAI_CHATGPT_RESPONSES_URL,
            {"Accept": "text/event-stream"},
        )

        self.assertEqual(headers["Accept"], "text/event-stream")
        self.assertEqual(headers["Authorization"], "Bearer access")
        self.assertEqual(headers["ChatGPT-Account-ID"], "account")
        self.assertEqual(lease.credential, ref)

    async def test_rejects_authentication_owned_base_header(self):
        broker = auth.CredentialBroker()
        ref = auth.CredentialRef.environment("EXAMPLE_API_KEY")
        broker.install_static(ref, "secret")

        with self.assertRaisesRegex(
                auth.CredentialError, "authentication-owned"):
            await auth.authorized_request_headers(
                broker,
                auth.AuthSpec(ref, "bearer"),
                "https://example.test/v1/responses",
                {"authorization": "Bearer stale"},
            )

    async def test_validates_target_before_leasing(self):
        authority = mock.Mock()
        authority.lease = mock.AsyncMock()
        ref = auth.CredentialRef.openai_subscription()

        with self.assertRaises(auth.CredentialUnavailable):
            await auth.authorized_request_headers(
                authority,
                auth.AuthSpec(ref, "openai-subscription"),
                "https://example.test/v1/responses",
            )

        authority.lease.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
