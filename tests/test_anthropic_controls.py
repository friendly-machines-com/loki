"""Usage formatting and account authorization without contacting the provider."""

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from loki_agent import anthropic_controls as controls
from loki_agent import authentications as auth
from loki_agent import credential_capabilities as capabilities
from loki_agent import http_client, provider_controls


def response(url, status, payload):
    return http_client.HttpResponse(url, status, "", {}, json.dumps(payload).encode("utf-8"))


def subscription_context(*, broker=None, request=None):
    spec = auth.AuthSpec(
        auth.CredentialRef.anthropic_subscription(), "anthropic-subscription",
        authorized_urls=auth.ANTHROPIC_SUBSCRIPTION_URLS,
    )
    return provider_controls.ControlContext(SimpleNamespace(auth_spec=spec), broker, request)


def tokens(access="anthropic-access", scopes=auth.ANTHROPIC_OAUTH_SCOPES):
    return auth.AnthropicTokenSet(access, "anthropic-refresh", "registered-test-client", scopes, expires_at=10**12)


class UsageRenderingTests(unittest.TestCase):
    def test_windows_are_already_percentages(self):
        lines = controls._usage_lines({
            "five_hour": {"utilization": 42, "resets_at": "2099-01-01T01:00:00+01:00"},
            "seven_day": {"utilization": 0.5, "resets_at": "2099-01-02T00:00:00Z"},
        })
        self.assertEqual(lines[0], "Anthropic subscription - live")
        self.assertIn("Session window (5 hour): 42% used, 58% remaining", lines[1])
        self.assertIn("resets 2099-01-01 00:00:00 UTC", lines[1])
        self.assertIn("Weekly window (all models): 0.5% used, 99.5% remaining", lines[2])

    def test_absent_windows_report_none(self):
        self.assertIn("No usage windows were reported.", controls._usage_lines({}))

    def test_invalid_utilization_is_not_guessed(self):
        for value in [420, True, -1, "42", float("nan"), float("inf"), 10**500]:
            with self.subTest(value=value):
                lines = controls._usage_lines({"five_hour": {"utilization": value}})
                self.assertIn("  Session window (5 hour): reported an unrecognized utilization", lines)

    def test_invalid_and_elapsed_reset_times(self):
        for value in ["later", "0001-01-01T00:00:00+01:00", None]:
            with self.subTest(value=value):
                lines = controls._usage_lines({"five_hour": {"utilization": 50, "resets_at": value}})
                self.assertIn("reset time unknown", lines[1])
        expired = controls._usage_lines({"five_hour": {"utilization": 100, "resets_at": "2000-01-01T00:00:00Z"}})
        self.assertIn("reported reset at 2000-01-01 00:00:00 UTC has passed", expired[1])

    def test_extra_usage_requires_known_spending_data(self):
        enabled = {"is_enabled": True, "monthly_limit": 5000, "used_credits": 1234}
        self.assertIn("  Extra usage: $12.34 / $50.00 spent; 24.68% used",
                      controls._usage_lines({"extra_usage": enabled}))
        self.assertIn("  Extra usage: $12.34 spent (unlimited)",
                      controls._usage_lines({"extra_usage": dict(enabled, monthly_limit=None)}))
        self.assertIn("  Extra usage: Unlimited",
                      controls._usage_lines({"extra_usage": dict(enabled, monthly_limit=None, used_credits=None)}))
        for extra in [{"is_enabled": True}, dict(enabled, monthly_limit=float("inf")),
                      dict(enabled, used_credits=10**500), dict(enabled, used_credits=float("nan"))]:
            self.assertIn("  Extra usage: enabled; spending unavailable",
                          controls._usage_lines({"extra_usage": extra}))
        disabled = controls._usage_lines({"extra_usage": {"is_enabled": False}})
        self.assertFalse(any("Extra usage" in line for line in disabled))


class UsageControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_subscription_connections_get_the_read_only_control(self):
        context = subscription_context()
        self.assertEqual([spec.id for spec in provider_controls.available_controls(context)], ["usage"])
        self.assertEqual(provider_controls.account_destinations(context), [controls.ANTHROPIC_USAGE_URL])
        env_spec = auth.AuthSpec(auth.CredentialRef.environment("ANTHROPIC_API_KEY"), "anthropic")
        request = mock.AsyncMock()
        env_context = provider_controls.ControlContext(SimpleNamespace(auth_spec=env_spec), request=request)
        self.assertEqual(provider_controls.available_controls(env_context), [])
        with self.assertRaises(auth.CredentialUnavailable):
            await controls.USAGE.read(env_context)
        request.assert_not_awaited()

    async def test_usage_request_is_confined_to_the_usage_url(self):
        requests = []
        payload = {"five_hour": {"utilization": 50, "resets_at": "2099-01-01T00:00:00Z"}}

        async def request(method, url, **kwargs):
            requests.append((method, url, kwargs["headers_in"]))
            return response(url, 200, payload)

        broker = auth.CredentialBroker()
        broker.install_anthropic_subscription(tokens())
        result = await controls.USAGE.read(subscription_context(broker=broker, request=request))
        self.assertEqual(requests, [("GET", controls.ANTHROPIC_USAGE_URL, {
            "Accept": "application/json", "Authorization": "Bearer anthropic-access",
            "anthropic-beta": auth.ANTHROPIC_OAUTH_BETA,
            "anthropic-dangerous-direct-browser-access": "true",
        })])
        self.assertIn("50% used, 50% remaining", result.lines[1])
        self.assertEqual(result.document, {"endpoint": controls.ANTHROPIC_USAGE_URL, "usage": payload})
        self.assertFalse(result.actions)

    async def test_rejected_generation_refreshes_once(self):
        rotate = mock.AsyncMock(return_value=tokens("anthropic-new-access"))
        requests = []

        async def request(_method, url, **kwargs):
            requests.append(kwargs["headers_in"]["Authorization"])
            return response(url, 401 if len(requests) == 1 else 200, {})

        broker = auth.CredentialBroker()
        broker.install_anthropic_subscription(tokens(), rotate=rotate)
        await controls.USAGE.read(subscription_context(broker=broker, request=request))
        rotate.assert_awaited_once()
        self.assertEqual(requests, ["Bearer anthropic-access", "Bearer anthropic-new-access"])

    async def test_inference_only_grant_is_denied_without_refreshing(self):
        now = 1000
        scopes = frozenset({auth.ANTHROPIC_INFERENCE_SCOPE})
        expiring = replace(tokens(scopes=scopes), expires_at=now + 1, last_refresh=now - 3600)
        rotate = mock.AsyncMock(return_value=tokens("new-access", scopes))
        request = mock.AsyncMock()
        broker = auth.CredentialBroker()
        broker.install_anthropic_subscription(expiring, rotate=rotate, clock=lambda: now)

        with self.assertRaisesRegex(auth.CredentialUnavailable, "account usage"):
            await controls.USAGE.read(subscription_context(broker=broker, request=request))
        rotate.assert_not_awaited()
        request.assert_not_awaited()

        lease = await broker.lease(auth.CredentialRef.anthropic_subscription(), auth.ANTHROPIC_BETA_MESSAGES_URL)
        self.assertEqual(lease.value, "new-access")
        rotate.assert_awaited_once()

    async def test_narrowed_grant_cannot_retry_usage(self):
        rotate = mock.AsyncMock(return_value=tokens("new-access", frozenset({auth.ANTHROPIC_INFERENCE_SCOPE})))
        reply = response(controls.ANTHROPIC_USAGE_URL, 401, {})
        request = mock.AsyncMock(return_value=reply)
        broker = auth.CredentialBroker()
        broker.install_anthropic_subscription(tokens(), rotate=rotate)
        with self.assertRaisesRegex(auth.CredentialUnavailable, "account usage"):
            await controls.USAGE.read(subscription_context(broker=broker, request=request))
        rotate.assert_awaited_once()
        request.assert_awaited_once()

    async def test_redirect_or_non_object_is_not_reported_as_live_usage(self):
        broker = auth.CredentialBroker()
        broker.install_anthropic_subscription(tokens())
        for status, payload in [(307, {}), (200, [])]:
            with self.subTest(status=status):
                request = mock.AsyncMock(return_value=response(controls.ANTHROPIC_USAGE_URL, status, payload))
                with self.assertRaises(OSError):
                    await controls.USAGE.read(subscription_context(broker=broker, request=request))
                request.assert_awaited_once()

    async def test_operation_specs_confine_both_routes(self):
        broker = auth.CredentialBroker()
        broker.install_anthropic_subscription(tokens())
        inference = subscription_context().config.auth_spec
        usage = auth.AuthSpec(inference.credential, inference.scheme, authorized_urls=controls.ANTHROPIC_ACCOUNT_URLS)
        for spec, other_url in [(inference, controls.ANTHROPIC_USAGE_URL), (usage, auth.ANTHROPIC_MESSAGES_URL)]:
            with self.subTest(other_url=other_url), self.assertRaises(auth.CredentialUnavailable):
                await auth.authorized_request_headers(broker, spec, other_url)

    async def test_inference_only_relay_cannot_use_account_lease(self):
        ref = auth.CredentialRef.anthropic_subscription()
        broker = auth.CredentialBroker()
        broker.install_anthropic_subscription(tokens())
        server, endpoint = await capabilities.CredentialCapabilityServer.create(broker)
        self.addAsyncCleanup(server.close)
        parent = await capabilities.CredentialClient.from_fd(endpoint)
        self.addAsyncCleanup(parent.close)
        self.assertEqual((await parent.lease(ref, controls.ANTHROPIC_USAGE_URL)).value, "anthropic-access")

        relay, child_end = await capabilities.CredentialCapabilityServer.create(
            parent, {ref}, destinations=auth.ANTHROPIC_SUBSCRIPTION_URLS,
        )
        self.addAsyncCleanup(relay.close)
        child = await capabilities.CredentialClient.from_fd(child_end)
        self.addAsyncCleanup(child.close)
        lease = await child.lease(ref, auth.ANTHROPIC_BETA_MESSAGES_URL)
        self.assertEqual(lease.value, "anthropic-access")
        self.assertNotIn("anthropic-refresh", json.dumps(lease.to_wire()))
        with self.assertRaises(capabilities.CapabilityError):
            await child.lease(ref, controls.ANTHROPIC_USAGE_URL)
