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

Anthropic OAuth requires a provider-approved native client registration.
The loopback callback and PKCE are shared, but Anthropic's token exchange
uses JSON rather than a form body.
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
OPENAI_CALLBACK_PORTS = [1455, 1457]
LOGIN_REQUEST_TIMEOUT_S = 30
LOGIN_MAX_BYTES = 256 * 1024
CALLBACK_HEADER_MAX_BYTES = 16 * 1024
CALLBACK_READ_TIMEOUT_S = 5
_DEVICE_USER_CODE_RE = re.compile(r"[A-Za-z0-9-]{1,128}\Z")

ANTHROPIC_AUTHORIZE_URL = "https://claude.com/cai/oauth/authorize"
ANTHROPIC_CALLBACK_PATH = "/callback"


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


def _decode_json_response(response, operation, provider="OpenAI"):
    if response.truncated:
        raise OAuthLoginError(
            f"{provider} {operation} response exceeded the size limit")
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (ValueError, RecursionError) as error:
        raise OAuthLoginError(
            f"{provider} {operation} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise OAuthLoginError(
            f"{provider} {operation} returned a non-object response")
    return value


async def _request(method, url, *, body=b"", headers=None, request=None, provider="OpenAI"):
    request_function = request or http_client.async_http_request
    try:
        # Waiting for these responses deliberately has no duration: they
        # mint or consume one-time authorization material and may involve
        # server-side processing, so abandoning one on a guessed budget
        # could report a failure that never happened. Connection setup and
        # delivery stay bounded; cancellation still applies.
        return await request_function(
            method,
            url,
            body=body,
            headers_in=headers or {},
            timeout=LOGIN_REQUEST_TIMEOUT_S,
            response_timeout=None,
            max_bytes=LOGIN_MAX_BYTES,
            retry_max_attempts=1,
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        # Login requests can contain an authorization code or verifier.
        # Avoid copying an arbitrary transport exception, and potentially its
        # URL, into terminal output or logs.
        raise OAuthLoginError(
            f"{provider} login transport failed") from error


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
        # The callback URL can contain a one-time authorization code.
        "Cache-Control: no-store\r\n"
        "Referrer-Policy: no-referrer\r\n"
        "Connection: close\r\n"
        "\r\n".encode("ascii") + encoded)
    with contextlib.suppress(
            BrokenPipeError, ConnectionError, OSError):
        await writer.drain()


async def _read_callback(reader, state, provider, path):
    try:
        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=CALLBACK_READ_TIMEOUT_S)
        if len(raw) > CALLBACK_HEADER_MAX_BYTES:
            raise ValueError("callback headers too large")
        request_line = raw.split(b"\r\n", 1)[0].decode("ascii")
        method, target, version = request_line.split(" ")
        if method != "GET" or version not in {"HTTP/1.0", "HTTP/1.1"}:
            raise ValueError("invalid callback request")
        parsed = urllib.parse.urlsplit(target)
        if parsed.scheme or parsed.netloc or parsed.fragment or parsed.path != path:
            return "404 Not Found", "<p>Unknown Loki authentication callback.</p>", None
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True, max_num_fields=16)
        states = query.get("state", [])
        if len(states) != 1 or not secrets.compare_digest(states[0], state):
            raise ValueError("invalid OAuth state")
        codes = query.get("code", [])
        errors = query.get("error", [])
        if len(errors) == 1 and errors[0] and not codes:
            return (
                "400 Bad Request", "<p>Authorization failed. Return to the terminal.</p>",
                OAuthLoginError(f"{provider} authorization was declined"),
            )
        if errors or len(codes) != 1 or not codes[0]:
            raise ValueError("invalid OAuth response")
        return "200 OK", "<p>Authorization received. You can return to Loki.</p>", codes[0]
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, OSError, TypeError, ValueError):
        return "400 Bad Request", "<p>Invalid Loki authentication callback.</p>", None


async def _handle_callback(reader, writer, state, result, provider, path):
    try:
        status, body, outcome = await _read_callback(reader, state, provider, path)
        if outcome is not None and not result.done():
            # Commit the first valid callback before yielding to the socket;
            # a disconnected browser cannot undo a received authorization.
            if isinstance(outcome, OAuthLoginError):
                result.set_exception(outcome)
            else:
                result.set_result(outcome)
        await _write_callback_response(writer, status, body)
    finally:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


class _BrowserLogin:
    """One CLI login owns its listener and at most one code exchange.

    Both OAuth flows use the same loopback/PKCE lifecycle. Only the callback
    path, authorization query and token request differ, so the provider
    supplies those instead of running a second listener implementation.
    """

    def __init__(self, server, result, authorization_url, exchange):
        self._server = server
        self._result = result
        self.authorization_url = authorization_url
        self._exchange = exchange
        self._completing = False

    async def complete(self):
        """Wait for consent, then exchange the code once."""
        if self._completing:
            raise OAuthLoginError("An authorization code can only be exchanged once")
        self._completing = True
        try:
            # A slow user is not a failed login. Wait until consent or cancellation.
            code = await self._result
            # Stop accepting connections as soon as the callback is no longer needed.
            self._server.close()
            return await self._exchange(code)
        finally:
            await self.aclose()

    async def aclose(self):
        self.close()
        await self._server.wait_closed()

    def close(self):
        self._server.close()
        if not self._result.done():
            self._result.cancel()
        elif not self._result.cancelled():
            # The browser launcher can fail before complete() consumes the future.
            self._result.exception()


async def _start_browser_login(provider, path, ports, redirect_host, authorize, exchange):
    result = asyncio.get_running_loop().create_future()
    pkce = generate_pkce()
    state = secrets.token_urlsafe(32)
    last_error = None
    for port in ports:
        try:
            server = await asyncio.start_server(
                lambda reader, writer: _handle_callback(reader, writer, state, result, provider, path),
                host="127.0.0.1", port=port, limit=CALLBACK_HEADER_MAX_BYTES,
            )
            break
        except OSError as error:
            last_error = error
    else:
        raise OAuthLoginError(f"could not bind the {provider} login callback ports") from last_error
    actual_port = server.sockets[0].getsockname()[1]
    redirect_uri = f"http://{redirect_host}:{actual_port}{path}"
    return _BrowserLogin(
        server, result, authorize(redirect_uri, pkce, state),
        lambda code: exchange(code, redirect_uri, pkce, state),
    )


async def start_openai_browser_login(*, ports=OPENAI_CALLBACK_PORTS, request=None):
    return await _start_browser_login(
        "OpenAI", "/auth/callback", ports, "localhost", openai_authorization_url,
        lambda code, redirect, pkce, _state: exchange_openai_authorization_code(
            code, redirect, pkce, request=request),
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
        sleep=asyncio.sleep):
    """Poll until the authorization completes, is declined, or is cancelled.

    There is no local overall duration: the authorization response carries
    no provider expiry to honor, and a guessed completion bound would
    abandon valid logins. Pacing follows the provider interval; user
    cancellation remains the stop mechanism.
    """
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
                    for value in [code, verifier, challenge]):
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
        await sleep(authorization.interval)


def anthropic_authorization_url(redirect_uri: str, pkce: PkceCodes, state: str, *, client_id: str):
    query = urllib.parse.urlencode({
        "code": "true",
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(sorted(authentications.ANTHROPIC_OAUTH_SCOPES)),
        "code_challenge": pkce.challenge,
        "code_challenge_method": "S256",
        "state": state,
    })
    return f"{ANTHROPIC_AUTHORIZE_URL}?{query}"


async def exchange_anthropic_authorization_code(
        code: str, redirect_uri: str, pkce: PkceCodes, state: str, *,
        client_id: str, request=None, now=None):
    if not isinstance(code, str) or not code:
        raise OAuthLoginError("Anthropic callback omitted its authorization code")
    body = json.dumps({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": pkce.verifier,
        "state": state,
    }, separators=(",", ":")).encode("utf-8")
    response = await _request(
        "POST", authentications.ANTHROPIC_TOKEN_URL, body=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        request=request, provider="Anthropic",
    )
    if not 200 <= response.status < 300:
        raise OAuthLoginError(f"Anthropic token exchange returned HTTP {response.status}")
    data = _decode_json_response(response, "token exchange", "Anthropic")
    timestamp = time.time() if now is None else now
    try:
        authentications.validate_anthropic_token_type(data)
        requested = authentications.ANTHROPIC_OAUTH_SCOPES
        # OAuth permits omission when the granted scopes equal the request.
        scopes = authentications.parse_anthropic_scopes(data["scope"]) if "scope" in data else requested
        if not scopes.issubset(requested):
            raise ValueError("Anthropic granted scopes Loki did not request")
        if authentications.ANTHROPIC_INFERENCE_SCOPE not in scopes:
            raise ValueError("Anthropic did not grant inference access")
        return authentications.AnthropicTokenSet(
            access_token=data.get("access_token"),
            refresh_token=data.get("refresh_token"),
            client_id=client_id,
            scopes=scopes,
            account=authentications.parse_anthropic_account(data),
            expires_at=authentications.anthropic_access_expiration(data.get("expires_in"), timestamp),
            last_refresh=timestamp,
        ).normalized(timestamp)
    except ValueError as error:
        raise OAuthLoginError(str(error)) from error


async def start_anthropic_browser_login(*, client_id: str, request=None):
    try:
        authentications.validate_anthropic_client_id(client_id)
    except ValueError as error:
        raise OAuthLoginError(str(error)) from error
    # An ephemeral port avoids competing with other logins or programs.
    # Use the IP literal in the redirect because the listener is IPv4-only.
    return await _start_browser_login(
        "Anthropic", ANTHROPIC_CALLBACK_PATH, [0], "127.0.0.1",
        lambda redirect, pkce, state: anthropic_authorization_url(redirect, pkce, state, client_id=client_id),
        lambda code, redirect, pkce, state: exchange_anthropic_authorization_code(
            code, redirect, pkce, state, client_id=client_id, request=request),
    )
