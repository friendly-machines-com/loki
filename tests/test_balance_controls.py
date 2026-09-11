import json
import unittest

from loki_agent import authentications as auth
from loki_agent import deepseek_controls
from loki_agent import http_client
from loki_agent import openrouter_controls
from loki_agent import provider_controls


class _Provider:
    def __init__(self, chat_url):
        self.chat_url = chat_url


class _Config:
    def __init__(self, chat_url, credential=None):
        self.chat_provider = _Provider(chat_url)
        self.auth_spec = auth.AuthSpec(
            credential or auth.CredentialRef.environment("EXAMPLE_API_KEY"),
            "bearer",
            authorized_origins=frozenset({auth.authorization_origin(chat_url)}),
        )


class _Authority:
    async def lease(self, credential, rejected_generation=None):
        return auth.CredentialLease(credential, "secret-token")

    def available(self):
        return frozenset()


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
        url="https://example.test/",
        status=status,
        reason=reason,
        headers={},
        body=json.dumps(payload).encode("utf-8"),
    )


def context(config, request=None):
    return provider_controls.ControlContext(
        config=config,
        credential_authority=_Authority(),
        request=request or _Request(),
    )


OPENROUTER = _Config(
    "https://openrouter.ai/api/v1/chat/completions",
    auth.CredentialRef.environment("OPENROUTER_API_KEY"),
)
DEEPSEEK = _Config(
    "https://api.deepseek.com/chat/completions",
    auth.CredentialRef.environment("DEEPSEEK_API_KEY"),
)


class AvailabilityTests(unittest.TestCase):
    def test_balance_applies_by_endpoint_origin(self):
        self.assertIsNotNone(
            provider_controls.find_control(context(OPENROUTER), "balance"))
        self.assertIsNotNone(
            provider_controls.find_control(context(DEEPSEEK), "balance"))

    def test_unrelated_endpoint_has_no_balance(self):
        other = _Config("https://example.test/v1/chat/completions")
        self.assertIsNone(
            provider_controls.find_control(context(other), "balance"))
        self.assertEqual(
            provider_controls.available_controls(context(other)), [])


class OpenRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_renders_spend_limit_and_usage(self):
        payload = {"data": {
            "label": "key",
            "limit": 10.0,
            "limit_remaining": 4.25,
            "limit_reset": "monthly",
            "usage": 5.75,
            "usage_monthly": 5.75,
            "usage_daily": 1.1,
            "is_free_tier": False,
            "is_management_key": False,
        }}
        request = _Request(response(payload))

        result = await openrouter_controls._read_balance(
            context(OPENROUTER, request))

        self.assertIn("  Spend limit: $10.00", result.lines)
        self.assertIn("  Remaining: $4.25", result.lines)
        self.assertIn("  Usage (today): $1.10", result.lines)
        self.assertIn("  Limit reset: monthly", result.lines)
        self.assertEqual(len(request.calls), 1)
        self.assertEqual(request.calls[0][1], openrouter_controls.OPENROUTER_KEY_URL)

    async def test_no_limit_is_explicit(self):
        request = _Request(response({"data": {"limit": None}}))

        result = await openrouter_controls._read_balance(
            context(OPENROUTER, request))

        self.assertIn(
            "  Spend limit: none set for this key", result.lines)

    async def test_management_key_also_reads_account_credits(self):
        request = _Request(
            response({"data": {"is_management_key": True, "limit": None}}),
            response({"data": {"total_credits": 100.5, "total_usage": 25.75}}),
        )

        result = await openrouter_controls._read_balance(
            context(OPENROUTER, request))

        self.assertEqual(len(request.calls), 2)
        self.assertIn("  Account credits: $100.50", result.lines)
        self.assertIn("  Account remaining: $74.75", result.lines)

    async def test_regular_key_does_not_call_credits_endpoint(self):
        request = _Request(
            response({"data": {"is_management_key": False}}))

        await openrouter_controls._read_balance(context(OPENROUTER, request))

        self.assertEqual(len(request.calls), 1)

    def test_credential_required(self):
        config = _Config("https://openrouter.ai/api/v1/chat/completions", None)
        config.auth_spec = auth.AuthSpec(None)
        with self.assertRaises(auth.CredentialUnavailable):
            openrouter_controls._credential(context(config))


class DeepSeekTests(unittest.IsolatedAsyncioTestCase):
    async def test_renders_balance_infos(self):
        payload = {
            "is_available": True,
            "balance_infos": [{
                "currency": "CNY",
                "total_balance": "110.00",
                "granted_balance": "10.00",
                "topped_up_balance": "100.00",
            }],
        }

        result = await deepseek_controls._read_balance(
            context(DEEPSEEK, _Request(response(payload))))

        self.assertIn("  Balance sufficient for API calls: yes", result.lines)
        self.assertIn(
            "  CNY: 110.00 total (10.00 granted, 100.00 topped up)",
            result.lines)

    async def test_insufficient_balance_is_visible(self):
        result = await deepseek_controls._read_balance(
            context(DEEPSEEK, _Request(response({
                "is_available": False, "balance_infos": [],
            }))))

        self.assertIn("  Balance sufficient for API calls: no", result.lines)
        self.assertIn("  (no balance reported)", result.lines)


if __name__ == "__main__":
    unittest.main()
