"""Asynchronous interactive OAuth protocols.

This module obtains a new token set but never persists it.  The public Loki
authentication command owns presentation and storage, while the ordinary
credential supervisor owns later refreshes.  Keeping those stages separate
ensures a failed or cancelled login cannot damage the previously stored
credential.

OpenAI's browser flow is OAuth authorization-code with PKCE.  Its device-code
flow uses OpenAI-specific endpoints rather than the standard RFC 8628 field
layout.  The constants and request shapes below deliberately mirror the
current open-source Codex client so an upstream protocol change is confined
to this module.  The corresponding upstream implementations are
``codex-rs/login/src/server.rs`` and
``codex-rs/login/src/device_code_auth.rs`` in openai/codex.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass

from . import authentications, http_client


OPENAI_ISSUER = "https://auth.openai.com"
OPENAI_AUTHORIZE_URL = f"{OPENAI_ISSUER}/oauth/authorize"
OPENAI_TOKEN_URL = f"{OPENAI_ISSUER}/oauth/token"
OPENAI_DEVICE_USER_CODE_URL = (
    f"{OPENAI_ISSUER}/api/accounts/deviceauth/usercode")
OPENAI_DEVICE_POLL_URL = (
    f"{OPENAI_ISSUER}/api/accounts/deviceauth/token")
OPENAI_DEVICE_VERIFICATION_URL = f"{OPENAI_ISSUER}/codex/device"
OPENAI_DEVICE_REDIRECT_URL = f"{OPENAI_ISSUER}/deviceauth/callback"
OPENAI_OAUTH_SCOPE = (
    "openid profile email offline_access "
    "api.connectors.read api.connectors.invoke"
)
OPENAI_CALLBACK_PORTS = (1455, 1457)
OPENAI_LOGIN_TIMEOUT_S = 15 * 60
OPENAI_LOGIN_REQUEST_TIMEOUT_S = 30
OPENAI_LOGIN_MAX_BYTES = 256 * 1024
CALLBACK_HEADER_MAX_BYTES = 16 * 1024
CALLBACK_READ_TIMEOUT_S = 5
_DEVICE_USER_CODE_RE = re.compile(r"[A-Za-z0-9-]{1,128}\Z")


class OAuthLoginError(RuntimeError):
    pass


@dataclass(frozen=True)
class PkceCodes:
    verifier: str
    challenge: str


@dataclass(frozen=True)
class DeviceAuthorization:
    verification_url: str
    user_code: str
    device_auth_id: str
    interval: int


def generate_pkce() -> PkceCodes:
    verifier = base64.urlsafe_b64encode(
        secrets.token_bytes(64)).rstrip(b"=").decode("ascii")
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    return PkceCodes(verifier, challenge)


def _pkce_challenge(verifier):
    return base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")


def _decode_json_response(response, operation):
    if response.truncated:
        raise OAuthLoginError(
            f"OpenAI {operation} response exceeded the size limit")
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OAuthLoginError(
            f"OpenAI {operation} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise OAuthLoginError(
            f"OpenAI {operation} returned a non-object response")
    return value


async def _request(
        method, url, *, body=b"", headers=None, request=None):
    request_function = request or http_client.async_http_request
    try:
        return await request_function(
            method,
            url,
            body=body,
            headers_in=headers or {},
            timeout=OPENAI_LOGIN_REQUEST_TIMEOUT_S,
            max_bytes=OPENAI_LOGIN_MAX_BYTES,
            retry_max_attempts=1,
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        # Login requests can contain an authorization code or verifier.
        # Avoid copying an arbitrary transport exception, and potentially its
        # URL, into terminal output or logs.
        raise OAuthLoginError(
            "OpenAI login transport failed") from error


def _login_tokens(data, now=None):
    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    id_token = data.get("id_token")
    if not isinstance(access_token, str) or not access_token:
        raise OAuthLoginError(
            "OpenAI token response omitted access_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise OAuthLoginError(
            "OpenAI token response omitted refresh_token")
    if not isinstance(id_token, str) or not id_token:
        raise OAuthLoginError(
            "OpenAI token response omitted id_token")
    identity = id_token or access_token
    timestamp = time.time() if now is None else now
    try:
        return authentications.OpenAITokenSet(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            account_id=authentications.token_account_id(identity),
            fedramp=authentications.token_fedramp(identity),
            expires_at=authentications.jwt_expiration(access_token),
            last_refresh=timestamp,
        ).normalized(timestamp)
    except ValueError as error:
        raise OAuthLoginError(
            f"OpenAI returned invalid tokens: {error}") from error


async def exchange_openai_authorization_code(
        code: str,
        redirect_uri: str,
        pkce: PkceCodes,
        *,
        request=None,
        now=None):
    if not isinstance(code, str) or not code:
        raise OAuthLoginError("OpenAI callback omitted its authorization code")
    body = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": authentications.OPENAI_OAUTH_CLIENT_ID,
        "code_verifier": pkce.verifier,
    }).encode("ascii")
    response = await _request(
        "POST",
        OPENAI_TOKEN_URL,
        body=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "originator": authentications.OPENAI_ORIGINATOR,
        },
        request=request,
    )
    if not 200 <= response.status < 300:
        raise OAuthLoginError(
            f"OpenAI token exchange returned HTTP {response.status}")
    return _login_tokens(
        _decode_json_response(response, "token exchange"),
        now=now,
    )


def openai_authorization_url(
        redirect_uri: str, pkce: PkceCodes, state: str):
    query = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": authentications.OPENAI_OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OPENAI_OAUTH_SCOPE,
        "code_challenge": pkce.challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": authentications.OPENAI_ORIGINATOR,
    })
    return f"{OPENAI_AUTHORIZE_URL}?{query}"


async def _write_callback_response(writer, status, body):
    encoded = body.encode("utf-8")
    writer.write(
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(encoded)}\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n"
        "\r\n".encode("ascii") + encoded)
    with contextlib.suppress(
            BrokenPipeError, ConnectionError, OSError):
        await writer.drain()


async def _handle_callback(reader, writer, state, result):
    status = "400 Bad Request"
    body = "<p>Invalid Loki authentication callback.</p>"
    complete = None
    try:
        raw = await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"),
            timeout=CALLBACK_READ_TIMEOUT_S,
        )
        if len(raw) > CALLBACK_HEADER_MAX_BYTES:
            raise ValueError("callback headers too large")
        request_line = raw.split(b"\r\n", 1)[0].decode("ascii")
        method, target, version = request_line.split(" ")
        if method != "GET" or version not in {"HTTP/1.0", "HTTP/1.1"}:
            raise ValueError("invalid callback request")
        parsed = urllib.parse.urlsplit(target)
        if parsed.path != "/auth/callback":
            status = "404 Not Found"
            body = "<p>Unknown Loki authentication callback.</p>"
        else:
            query = urllib.parse.parse_qs(
                parsed.query,
                keep_blank_values=True,
                max_num_fields=16,
            )
            states = query.get("state", [])
            if len(states) != 1:
                raise ValueError("invalid OAuth state")
            returned_state = states[0]
            if not secrets.compare_digest(returned_state, state):
                raise ValueError("invalid OAuth state")
            errors = query.get("error", [])
            if len(errors) > 1:
                raise ValueError("invalid OAuth error")
            error = errors[0] if errors else None
            if error is not None:
                complete = OAuthLoginError(
                    "OpenAI authorization was declined")
                body = (
                    "<p>OpenAI authorization failed. Return to the "
                    "terminal.</p>")
            else:
                codes = query.get("code", [])
                if len(codes) != 1 or not codes[0]:
                    raise ValueError("missing authorization code")
                complete = codes[0]
                status = "200 OK"
                body = (
                    "<p>Authorization received. You can return to "
                    "Loki.</p>")
    except (
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            asyncio.TimeoutError,
            TypeError,
            UnicodeError,
            ValueError,
    ):
        pass
    try:
        await _write_callback_response(writer, status, body)
    finally:
        writer.close()
        with contextlib.suppress(
                BrokenPipeError, ConnectionError, OSError):
            await writer.wait_closed()
    if complete is not None and not result.done():
        if isinstance(complete, BaseException):
            result.set_exception(complete)
        else:
            result.set_result(complete)


class OpenAIBrowserLogin:
    def __init__(
            self, server, result, redirect_uri, pkce, state,
            request=None):
        self._server = server
        self._result = result
        self._redirect_uri = redirect_uri
        self._pkce = pkce
        self._request = request
        self.authorization_url = openai_authorization_url(
            redirect_uri, pkce, state)

    async def complete(self):
        try:
            code = await asyncio.wait_for(
                self._result, OPENAI_LOGIN_TIMEOUT_S)
            return await exchange_openai_authorization_code(
                code,
                self._redirect_uri,
                self._pkce,
                request=self._request,
            )
        except asyncio.TimeoutError as error:
            raise OAuthLoginError(
                "OpenAI browser login timed out") from error
        finally:
            self.close()
            await self._server.wait_closed()

    def close(self):
        self._server.close()
        if not self._result.done():
            self._result.cancel()


async def start_openai_browser_login(
        *, ports=OPENAI_CALLBACK_PORTS, request=None):
    loop = asyncio.get_running_loop()
    result = loop.create_future()
    pkce = generate_pkce()
    state = secrets.token_urlsafe(32)
    server = None
    last_error = None
    for port in ports:
        try:
            server = await asyncio.start_server(
                lambda reader, writer: _handle_callback(
                    reader, writer, state, result),
                host="127.0.0.1",
                port=port,
                limit=CALLBACK_HEADER_MAX_BYTES,
            )
            break
        except OSError as error:
            last_error = error
    if server is None:
        raise OAuthLoginError(
            "could not bind the OpenAI login callback ports") from last_error
    actual_port = server.sockets[0].getsockname()[1]
    redirect_uri = f"http://localhost:{actual_port}/auth/callback"
    return OpenAIBrowserLogin(
        server,
        result,
        redirect_uri,
        pkce,
        state,
        request=request,
    )


async def request_openai_device_authorization(*, request=None):
    body = json.dumps({
        "client_id": authentications.OPENAI_OAUTH_CLIENT_ID,
    }, separators=(",", ":")).encode("utf-8")
    response = await _request(
        "POST",
        OPENAI_DEVICE_USER_CODE_URL,
        body=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        request=request,
    )
    if not 200 <= response.status < 300:
        raise OAuthLoginError(
            f"OpenAI device authorization returned HTTP "
            f"{response.status}")
    data = _decode_json_response(response, "device authorization")
    device_auth_id = data.get("device_auth_id")
    user_code = data.get("user_code", data.get("usercode"))
    interval_value = data.get("interval", "5")
    if not isinstance(device_auth_id, str) or not device_auth_id:
        raise OAuthLoginError(
            "OpenAI device authorization omitted device_auth_id")
    if not isinstance(user_code, str) or not user_code:
        raise OAuthLoginError(
            "OpenAI device authorization omitted user_code")
    if _DEVICE_USER_CODE_RE.fullmatch(user_code) is None:
        raise OAuthLoginError(
            "OpenAI device authorization returned invalid user_code")
    try:
        if isinstance(interval_value, bool):
            raise ValueError()
        interval = int(interval_value)
    except (TypeError, ValueError) as error:
        raise OAuthLoginError(
            "OpenAI device authorization returned invalid interval") from error
    if interval < 1 or interval > 60:
        raise OAuthLoginError(
            "OpenAI device authorization returned invalid interval")
    return DeviceAuthorization(
        verification_url=OPENAI_DEVICE_VERIFICATION_URL,
        user_code=user_code,
        device_auth_id=device_auth_id,
        interval=interval,
    )


async def complete_openai_device_login(
        authorization: DeviceAuthorization, *,
        request=None,
        sleep=asyncio.sleep,
        clock=None):
    clock_function = clock or asyncio.get_running_loop().time
    deadline = clock_function() + OPENAI_LOGIN_TIMEOUT_S
    while True:
        body = json.dumps({
            "device_auth_id": authorization.device_auth_id,
            "user_code": authorization.user_code,
        }, separators=(",", ":")).encode("utf-8")
        response = await _request(
            "POST",
            OPENAI_DEVICE_POLL_URL,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            request=request,
        )
        if 200 <= response.status < 300:
            data = _decode_json_response(response, "device-code poll")
            code = data.get("authorization_code")
            verifier = data.get("code_verifier")
            challenge = data.get("code_challenge")
            if not all(
                    isinstance(value, str) and value
                    for value in (code, verifier, challenge)):
                raise OAuthLoginError(
                    "OpenAI device-code poll returned invalid PKCE data")
            try:
                expected_challenge = _pkce_challenge(verifier)
            except UnicodeEncodeError as error:
                raise OAuthLoginError(
                    "OpenAI device-code poll returned invalid PKCE data"
                ) from error
            if not secrets.compare_digest(
                    challenge, expected_challenge):
                raise OAuthLoginError(
                    "OpenAI device-code poll returned inconsistent "
                    "PKCE data")
            return await exchange_openai_authorization_code(
                code,
                OPENAI_DEVICE_REDIRECT_URL,
                PkceCodes(verifier, challenge),
                request=request,
            )
        if response.status not in {403, 404}:
            raise OAuthLoginError(
                f"OpenAI device-code poll returned HTTP "
                f"{response.status}")
        remaining = deadline - clock_function()
        if remaining <= 0:
            raise OAuthLoginError(
                "OpenAI device login timed out")
        await sleep(min(authorization.interval, remaining))
