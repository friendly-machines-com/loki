import json
import unittest

from loki_agent import authentications as auth
from loki_agent import http_client
from loki_agent import provider_controls
from loki_agent import zai_controls


class _Provider:
    def __init__(self, chat_url):
        self.chat_url = chat_url


class _Config:
    def __init__(self, chat_url, credential=None):
        self.chat_provider = _Provider(chat_url)
        self.auth_spec = auth.AuthSpec(
            credential or auth.CredentialRef.environment("ZAI_API_KEY"),
            "bearer",
            authorized_origins=frozenset({auth.authorization_origin(chat_url)}),
        )


class _Authority:
    async def lease(self, credential, rejected_generation=None):
        return auth.CredentialLease(credential, "api-key")

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
        url="https://api.z.ai/",
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


ZAI = _Config("https://api.z.ai/api/coding/paas/v4/chat/completions")
BIGMODEL = _Config(
    "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions")

QUOTA_PAYLOAD = {
    "success": True,
    "code": 200,
    "data": {
        "level": "Max",
        "limits": [
            {"type": "TOKENS_LIMIT", "unit": 3, "number": 5,
             "percentage": 80, "nextResetTime": 4102444800000},
            {"type": "TOKENS_LIMIT", "unit": 6, "number": 1,
             "percentage": 76, "nextResetTime": 4102444800000},
            {"type": "TIME_LIMIT", "unit": 5, "number": 1,
             "percentage": 2, "currentValue": 2, "usage": 100},
        ],
    },
}

# Mirrors a live response (2026-09-17): the extra data fields the console
# also sends must not disturb parsing.
CARDS_PAYLOAD = {
    "code": 200,
    "msg": "Operation successful",
    "success": True,
    "data": {
        "customerId": 73911762657470674,
        "targetType": "PERSONAL",
        "organizationId": None,
        "projectId": None,
        "lastFiveHourResetTime": None,
        "lastWeekResetTime": "2026-09-10 12:34:56",
        "fiveHourResets": [
            {"recordId": 101, "expireTime": "2100-01-01 00:00:00",
             "available": True},
            {"recordId": 102, "expireTime": "2000-01-01 00:00:00",
             "available": False},
        ],
        "weekResets": [
            {"recordId": 201, "expireTime": "2100-02-01 00:00:00",
             "available": True},
        ],
    },
}

REFUSED = {"code": 1001, "success": False,
           "msg": "Authentication parameter not received in Header"}


class AvailabilityTests(unittest.TestCase):
    def test_zai_connections_expose_usage_and_resets(self):
        for config in (ZAI, BIGMODEL):
            with self.subTest(config=config.chat_provider.chat_url):
                specs = provider_controls.available_controls(context(config))
                self.assertEqual(
                    [spec.id for spec in specs], ["usage", "resets"])

    def test_unrelated_endpoint_has_no_controls(self):
        other = _Config("https://example.test/v1/chat/completions")
        self.assertEqual(
            provider_controls.available_controls(context(other)), [])

    def test_no_credential_has_no_controls(self):
        config = _Config(ZAI.chat_provider.chat_url, None)
        config.auth_spec = auth.AuthSpec(None)
        self.assertEqual(
            provider_controls.available_controls(context(config)), [])


class AuthStyleTests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_key_is_tried_first_and_succeeds(self):
        request = _Request(response(QUOTA_PAYLOAD))

        _payload, style = await zai_controls._authorized_json(
            context(ZAI, request), "GET",
            "https://api.z.ai/api/monitor/usage/quota/limit")

        self.assertEqual(style, "key")
        self.assertEqual(len(request.calls), 1)
        headers = request.calls[0][2]["headers_in"]
        self.assertEqual(headers["Authorization"], "api-key")

    async def test_refusal_under_raw_key_falls_back_to_bearer(self):
        request = _Request(response(REFUSED), response(QUOTA_PAYLOAD))

        payload, style = await zai_controls._authorized_json(
            context(ZAI, request), "GET",
            "https://api.z.ai/api/monitor/usage/quota/limit")

        self.assertEqual(style, "bearer")
        self.assertEqual(len(request.calls), 2)
        self.assertEqual(
            request.calls[1][2]["headers_in"]["Authorization"],
            "Bearer api-key")
        self.assertTrue(payload["success"])

    async def test_refusal_under_every_style_is_reported(self):
        request = _Request(response(REFUSED), response(REFUSED))

        with self.assertRaises(zai_controls._AuthRefused) as raised:
            await zai_controls._authorized_json(
                context(ZAI, request), "GET",
                "https://api.z.ai/api/monitor/usage/quota/limit")

        self.assertIn("session token", str(raised.exception))

    async def test_http_refusal_also_falls_back(self):
        request = _Request(
            response(REFUSED, status=403, reason="Forbidden"),
            response(QUOTA_PAYLOAD))

        _payload, style = await zai_controls._authorized_json(
            context(ZAI, request), "GET",
            "https://api.z.ai/api/monitor/usage/quota/limit")

        self.assertEqual(style, "bearer")

    async def test_provider_error_is_raised_without_fallback(self):
        request = _Request(response({
            "code": 1308, "success": False, "msg": "insufficient quota"}))

        with self.assertRaises(OSError) as raised:
            await zai_controls._authorized_json(
                context(ZAI, request), "GET",
                "https://api.z.ai/api/monitor/usage/quota/limit")

        self.assertIn("1308", str(raised.exception))
        self.assertEqual(len(request.calls), 1)


class UsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_renders_quota_windows(self):
        result = await zai_controls._read_usage(
            context(ZAI, _Request(response(QUOTA_PAYLOAD))))

        self.assertIn("Z.ai Coding Plan - live", result.lines)
        self.assertIn("Plan: Max", result.lines)
        self.assertTrue(any(
            line.startswith(
                "  5-hour prompt pool: 80% used, 20% remaining")
            for line in result.lines))
        self.assertTrue(any(
            line.startswith(
                "  Weekly quota: 76% used, 24% remaining")
            for line in result.lines))
        self.assertIn(
            "  Tool calls: 2% used, 98% remaining, 2 of 100 used",
            result.lines)
        self.assertTrue(any(
            "resets 2100-01-01" in line for line in result.lines))

    async def test_unknown_limit_combination_is_still_shown(self):
        payload = {"success": True, "data": {"limits": [
            {"type": "TOKENS_LIMIT", "unit": 9, "number": 9,
             "percentage": 10}]}}

        result = await zai_controls._read_usage(
            context(ZAI, _Request(response(payload))))

        self.assertIn(
            "  TOKENS_LIMIT (unit=9 number=9): 10% used, 90% remaining",
            result.lines)

    async def test_no_limits_is_explicit(self):
        result = await zai_controls._read_usage(
            context(ZAI, _Request(response({"success": True, "data": {}}))))

        self.assertIn("No usage limits were reported.", result.lines)

    def test_credential_required(self):
        config = _Config(ZAI.chat_provider.chat_url, None)
        config.auth_spec = auth.AuthSpec(None)
        with self.assertRaises(auth.CredentialUnavailable):
            zai_controls._credential(context(config))


class ResetsTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_cards_and_offers_only_available_actions(self):
        result = await zai_controls._read_resets(
            context(ZAI, _Request(response(CARDS_PAYLOAD))))

        self.assertIn("  5-hour reset cards: 1 available", result.lines)
        self.assertIn("  Weekly reset cards: 1 available", result.lines)
        self.assertIn("  Total: 2 available, 1 expired", result.lines)
        self.assertTrue(any(
            line.strip().startswith("1. expires 2100-01-01")
            for line in result.lines))
        self.assertEqual(len(result.actions), 2)
        five_hour = result.actions[0]
        self.assertEqual(five_hour.id, "use:101")
        self.assertIn("5-hour", five_hour.title)
        self.assertIn(
            "Use the 5-hour reset card", five_hour.confirm)
        self.assertIn("expires 2100-01-01", five_hour.confirm)
        self.assertEqual(result.actions[1].id, "use:201")
        self.assertEqual(
            [card["recordId"] for card in result.document["cards"]],
            [101, 102, 201])

    async def test_no_cards_is_explicit(self):
        result = await zai_controls._read_resets(
            context(ZAI, _Request(response({"success": True, "data": {}}))))

        self.assertIn("  (no reset cards were reported)", result.lines)
        self.assertIn("  Total: 0 available, 0 expired", result.lines)
        self.assertEqual(result.actions, ())

    async def test_cards_endpoint_is_under_biz(self):
        request = _Request(response(CARDS_PAYLOAD))

        await zai_controls._read_resets(context(ZAI, request))

        url = request.calls[0][1]
        self.assertEqual(
            url,
            "https://api.z.ai/api/biz/customer-package-reset/list"
            "?targetType=PERSONAL")


class UseTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_card_with_idempotency_request_id(self):
        request = _Request(response({"code": 200, "success": True}))

        result = await zai_controls._use(context(ZAI, request),
                                         "FIVE_HOUR", 101)

        self.assertEqual(len(request.calls), 1)
        method, url, kwargs = request.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(
            url, "https://api.z.ai/api/biz/customer-package-reset/use")
        body = json.loads(kwargs["body"])
        self.assertEqual(body["targetType"], "PERSONAL")
        self.assertEqual(body["resetType"], "FIVE_HOUR")
        self.assertEqual(body["recordId"], 101)
        self.assertIsInstance(body["requestId"], str)
        self.assertIn(
            "Reset card used: 5-hour quota is back to 100%",
            result.lines[0])
        self.assertEqual(
            result.document["request_id"], body["requestId"])

    async def test_action_runs_the_use_request(self):
        request = _Request(response({"code": 200, "success": True}))
        card = {"recordId": 101, "expireTime": "2100-01-01 00:00:00",
                "available": True}

        action = zai_controls._use_action(
            context(ZAI, request), "FIVE_HOUR", "5-hour", card)
        result = await action.run()

        self.assertEqual(action.id, "use:101")
        self.assertTrue(request.calls)
        self.assertIn("back to 100%", result.lines[0])

    async def test_transport_failure_reports_request_id(self):
        request = _Request(OSError("connection reset"))

        result = await zai_controls._use(context(ZAI, request),
                                         "WEEK", 201)

        self.assertIn("Could not confirm the reset", result.lines[0])
        self.assertIn("request ", result.lines[0])
        self.assertIn("connection reset", result.lines[1])

    async def test_refusal_under_every_style_is_not_ambiguous(self):
        request = _Request(response(REFUSED), response(REFUSED))

        result = await zai_controls._use(context(ZAI, request),
                                         "WEEK", 201)

        self.assertIn("Reset card not used", result.lines[0])
        self.assertIn("session token", result.lines[0])


if __name__ == "__main__":
    unittest.main()
