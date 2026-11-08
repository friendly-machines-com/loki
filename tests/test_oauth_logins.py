import asyncio
import base64
import json
import urllib.parse
import unittest

from process_lifecycle_fixtures import observe_fixture
from loki_agent import http_client
from loki_agent import oauth_logins


def jwt(payload):
    def encode(value):
        data = json.dumps(
            value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(
            data).rstrip(b"=").decode("ascii")

    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


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


class OpenAIBrowserLoginTests(unittest.IsolatedAsyncioTestCase):
    async def _callback(self, login, request):
        query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(login.authorization_url).query)
        redirect = urllib.parse.urlsplit(
            query["redirect_uri"][0])
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", redirect.port)
        writer.write(request)
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        return response

    async def test_wrong_state_cannot_finish_login(self):
        login = await oauth_logins.start_openai_browser_login(
            ports=(0,))
        query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(login.authorization_url).query)
        redirect = urllib.parse.urlsplit(
            query["redirect_uri"][0])

        async def callback(state):
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", redirect.port)
            target = (
                "/auth/callback?"
                + urllib.parse.urlencode({
                    "code": "code",
                    "state": state,
                }))
            writer.write(
                f"GET {target} HTTP/1.1\r\n\r\n".encode("ascii"))
            await writer.drain()
            answer = await reader.read()
            writer.close()
            await writer.wait_closed()
            return answer

        answer = await callback("wrong")

        self.assertIn(b"400 Bad Request", answer)
        self.assertFalse(login._result.done())
        login.close()
        await login._server.wait_closed()

    async def test_malformed_callback_cannot_finish_login(self):
        login = await oauth_logins.start_openai_browser_login(
            ports=(0,))

        answer = await self._callback(
            login, b"POST /auth/callback HTTP/1.1\r\n\r\n")

        self.assertIn(b"400 Bad Request", answer)
        self.assertFalse(login._result.done())
        login.close()
        await login._server.wait_closed()

    async def test_oversized_callback_cannot_finish_login(self):
        login = await oauth_logins.start_openai_browser_login(
            ports=(0,))
        request = (
            b"GET /auth/callback HTTP/1.1\r\nX-Padding: "
            + b"x" * oauth_logins.CALLBACK_HEADER_MAX_BYTES
            + b"\r\n\r\n"
        )

        answer = await self._callback(login, request)

        self.assertIn(b"400 Bad Request", answer)
        self.assertFalse(login._result.done())
        login.close()
        await login._server.wait_closed()

    async def test_browser_login_waits_without_a_duration(self):
        # A callback that has not arrived yet is not an error: the login
        # waits for the result without a local duration, and cancellation
        # remains the stop mechanism.
        login = await oauth_logins.start_openai_browser_login(
            ports=(0,))
        task = asyncio.create_task(login.complete())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertFalse(task.done())
        self.assertTrue(login._server.is_serving())

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(login._server.is_serving())
        self.assertTrue(login._result.cancelled())

    async def test_browser_login_cancellation_closes_callback_server(self):
        login = await oauth_logins.start_openai_browser_login(
            ports=(0,))
        task = asyncio.create_task(login.complete())
        await asyncio.sleep(0)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertFalse(login._server.is_serving())
        self.assertTrue(login._result.cancelled())


class OpenAIDeviceLoginTests(unittest.IsolatedAsyncioTestCase):
    async def test_device_request_rejects_invalid_interval(self):
        async def request(_method, url, **_kwargs):
            return response(url, 200, {
                "device_auth_id": "device",
                "user_code": "code",
                "interval": "0",
            })

        with self.assertRaisesRegex(
                oauth_logins.OAuthLoginError, "invalid interval"):
            await oauth_logins.request_openai_device_authorization(
                request=request)

    async def test_device_poll_rejects_unexpected_http_status(self):
        authorization = oauth_logins.DeviceAuthorization(
            verification_url="https://auth.example/device",
            user_code="ABCD-EFGH",
            device_auth_id="device",
            interval=1,
        )

        async def request(_method, url, **_kwargs):
            return response(url, 500, {})

        with self.assertRaisesRegex(
                oauth_logins.OAuthLoginError, "HTTP 500"):
            await oauth_logins.complete_openai_device_login(
                authorization, request=request)

    async def test_device_poll_keeps_polling_without_an_overall_duration(self):
        authorization = oauth_logins.DeviceAuthorization(
            verification_url="https://auth.example/device",
            user_code="ABCD-EFGH",
            device_auth_id="device",
            interval=1,
        )
        polls = []

        async def request(_method, url, **_kwargs):
            polls.append(url)
            return response(url, 403, {})

        async def sleep(_delay):
            # Provider pacing without wall-clock: one scheduling turn each.
            await asyncio.sleep(0)

        task = asyncio.create_task(oauth_logins.complete_openai_device_login(
            authorization, request=request, sleep=sleep))
        try:
            for _ in range(8):
                await asyncio.sleep(0)
            # Pending authorization keeps polling; there is no local overall
            # duration that would abandon it.
            self.assertGreaterEqual(len(polls), 3)
            self.assertFalse(task.done())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_device_poll_cancellation_propagates(self):
        authorization = oauth_logins.DeviceAuthorization(
            verification_url="https://auth.example/device",
            user_code="ABCD-EFGH",
            device_auth_id="device",
            interval=1,
        )
        sleeping = asyncio.Event()

        async def request(_method, url, **_kwargs):
            return response(url, 403, {})

        async def sleep(_delay):
            sleeping.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            oauth_logins.complete_openai_device_login(
                authorization,
                request=request,
                sleep=sleep,
            ))
        try:
            # The poll can fail before reaching the held sleep. Borrow its
            # actual task for readiness; this fixture still owns retirement.
            await observe_fixture(sleeping.wait(), task)
            task.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_token_error_does_not_include_response_body(self):
        secret_body = b"refresh_token=must-not-appear"

        async def request(_method, url, **_kwargs):
            return response(url, 400, secret_body)

        with self.assertRaises(
                oauth_logins.OAuthLoginError) as raised:
            await oauth_logins.exchange_openai_authorization_code(
                "code",
                "http://localhost/callback",
                oauth_logins.PkceCodes("verifier", "challenge"),
                request=request,
            )

        self.assertNotIn("must-not-appear", str(raised.exception))
