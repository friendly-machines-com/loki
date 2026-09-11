import json
import unittest

from loki_agent import authentications as auth
from loki_agent import http_client
from loki_agent import openai_controls
from loki_agent import provider_controls


class _Config:
    def __init__(self, scheme="openai-subscription"):
        self.auth_spec = auth.AuthSpec(
            auth.CredentialRef.openai_subscription(), scheme)


class _Authority:
    async def lease(self, credential, rejected_generation=None):
        return auth.CredentialLease(credential, "access-token")

    def available(self):
        return frozenset({auth.CredentialRef.openai_subscription()})


class _Request:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected request")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def response(payload, status=200, reason="OK"):
    return http_client.HttpResponse(
        url="https://chatgpt.com/",
        status=status,
        reason=reason,
        headers={},
        body=json.dumps(payload).encode("utf-8"),
    )


def context(cfg=None, request=None):
    return provider_controls.ControlContext(
        config=cfg or _Config(),
        credential_authority=_Authority(),
        request=request or _Request(),
    )


CREDS_PAYLOAD = {
    "available_count": 2,
    "total_earned_count": 4,
    "credits": [
        {
            "id": "credit-1",
            "reset_type": "codex_rate_limits",
            "status": "available",
            "granted_at": "2026-06-17T00:00:00Z",
            "expires_at": "2026-07-17T00:00:00Z",
            "title": "Full reset (Weekly + 5 hr)",
            "description": "Ready to redeem",
        },
        {
            "id": "credit-2",
            "reset_type": "codex_rate_limits",
            "status": "expired",
            "granted_at": "2026-06-18T00:00:00Z",
            "expires_at": None,
        },
    ],
}


class AvailabilityTests(unittest.TestCase):
    def test_subscription_connection_exposes_controls(self):
        specs = provider_controls.available_controls(context())
        self.assertEqual(
            [spec.id for spec in specs], ["usage", "resets"])

    def test_other_scheme_exposes_nothing(self):
        specs = provider_controls.available_controls(
            context(_Config(scheme="bearer")))
        self.assertEqual(specs, [])

    def test_discovery_performs_no_request(self):
        request = _Request()

        provider_controls.available_controls(context(request=request))

        self.assertEqual(request.calls, [])

    def test_control_tokens_accept_visible_title_words(self):
        ctx = context()

        self.assertEqual(
            provider_controls.find_control(ctx, "usage").id, "usage")
        self.assertEqual(
            provider_controls.find_control(ctx, "resets").id, "resets")
        # "Limit resets" is the menu title; typing a visible word must work.
        self.assertEqual(
            provider_controls.find_control(ctx, "limit").id, "resets")
        self.assertEqual(
            provider_controls.find_control(ctx, "RES").id, "resets")
        self.assertIsNone(provider_controls.find_control(ctx, "nope"))

    def test_live_hint_is_static_and_provider_gated(self):
        self.assertEqual(
            provider_controls.live_hint(context()),
            "Live account data: /account usage, /account resets")
        self.assertIsNone(
            provider_controls.live_hint(context(_Config(scheme="bearer"))))


class ResetReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_credits_and_offers_only_available_actions(self):
        request = _Request(response(CREDS_PAYLOAD))

        result = await openai_controls._read_resets(context(request=request))

        self.assertIn("Banked limit resets: 2 available", result.lines)
        self.assertTrue(any("Full reset" in line for line in result.lines))
        self.assertTrue(any("[expired]" in line for line in result.lines))
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].id, "redeem:credit-1")
        method, url, _kwargs = request.calls[0]
        self.assertEqual(method, "GET")
        self.assertEqual(url, openai_controls.OPENAI_CHATGPT_RESET_CREDITS_URL)

    async def test_unknown_status_is_shown_not_dropped(self):
        payload = {"credits": [{
            "id": "c", "status": "brand_new", "expires_at": None,
        }]}

        result = await openai_controls._read_resets(
            context(request=_Request(response(payload))))

        self.assertTrue(any("[brand_new]" in line for line in result.lines))
        self.assertEqual(result.actions, ())


class RedeemTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_reports_windows_reset(self):
        request = _Request(response({"code": "reset", "windows_reset": 2}))
        result = await openai_controls._redeem(context(request=request), "credit-1")

        self.assertEqual(result.lines, ("Reset redeemed: 2 window(s) reset.",))
        method, url, kwargs = request.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, openai_controls.OPENAI_CHATGPT_CONSUME_RESET_URL)
        body = json.loads(kwargs["body"].decode("utf-8"))
        self.assertEqual(body["credit_id"], "credit-1")
        self.assertIsInstance(body["redeem_request_id"], str)
        self.assertEqual(
            kwargs["headers_in"]["Content-Type"], "application/json")
        self.assertEqual(kwargs["retry_max_attempts"], 1)

    async def test_already_redeemed_is_not_an_error(self):
        result = await openai_controls._redeem(
            context(request=_Request(response({"code": "already_redeemed"}))),
            "credit-1")

        self.assertEqual(result.lines, ("Already redeemed (no change).",))

    async def test_ineligible_403_has_distinct_message(self):
        result = await openai_controls._redeem(
            context(request=_Request(response(
                {"detail": {"code": "rate_limit_reset_ineligible"}},
                status=403, reason="Forbidden"))),
            "credit-1")

        self.assertIn("not currently eligible", result.lines[0])

    async def test_ambiguous_failure_reports_request_id_without_retry(self):
        request = _Request(OSError("connection reset"))

        result = await openai_controls._redeem(
            context(request=request), "credit-1")

        self.assertEqual(len(request.calls), 1)
        self.assertIn("Could not confirm the reset", result.lines[0])
        self.assertEqual(result.document, None)


class UsageReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_renders_reported_windows(self):
        payload = {
            "plan_type": "plus",
            "rate_limit": {
                "primary_window": {
                    "used_percent": 56,
                    "limit_window_seconds": 18000,
                    "reset_after_seconds": 3600,
                },
            },
            "rate_limit_reset_credits": {"available_count": 3},
        }

        result = await openai_controls._read_usage(
            context(request=_Request(response(payload))))

        self.assertIn("Plan: plus", result.lines)
        self.assertTrue(any("56% used" in line for line in result.lines))
        self.assertIn("Banked limit resets: 3 available", result.lines)

    async def test_missing_windows_is_graceful(self):
        result = await openai_controls._read_usage(
            context(request=_Request(response({}))))

        self.assertIn("No usage windows were reported.", result.lines)


class AccountTargetTests(unittest.TestCase):
    def test_account_spec_rejects_inference_urls(self):
        spec = openai_controls._auth_spec(
            auth.CredentialRef.openai_subscription())

        with self.assertRaises(auth.CredentialUnavailable):
            auth.validate_authorization_target(
                spec, auth.OPENAI_CHATGPT_RESPONSES_URL)

    def test_account_spec_accepts_account_urls(self):
        spec = openai_controls._auth_spec(
            auth.CredentialRef.openai_subscription())

        for url in openai_controls.OPENAI_CHATGPT_ACCOUNT_URLS:
            with self.subTest(url=url):
                auth.validate_authorization_target(spec, url)


if __name__ == "__main__":
    unittest.main()
