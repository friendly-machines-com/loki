"""User-facing persistent authentication commands.

Authentication is deliberately a supervisor command, not a terminal session
operation and not an ACP method.  Both frontends consume the same resulting
credential store on their next startup.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
import webbrowser

from . import credential_storages, oauth_logins


def _parser(program):
    parser = argparse.ArgumentParser(prog=f"{program} auth")
    commands = parser.add_subparsers(dest="command", required=True)

    login = commands.add_parser(
        "login", help="sign in to a provider")
    login.add_argument("provider", choices=["openai", "anthropic"])
    login.add_argument(
        "--client-id",
        help="override the bundled Anthropic OAuth client ID (Anthropic only)",
    )
    method = login.add_mutually_exclusive_group()
    method.add_argument(
        "--device-code",
        action="store_true",
        help="use the headless/device-code flow",
    )
    method.add_argument(
        "--no-browser",
        action="store_true",
        help="print the browser URL without trying to open it",
    )

    status = commands.add_parser(
        "status", help="show persistent login status")
    status.add_argument(
        "provider", nargs="?", choices=["openai", "anthropic"])

    logout = commands.add_parser(
        "logout", help="remove a persistent login")
    logout.add_argument("provider", choices=["openai", "anthropic"])
    return parser


def _time_text(value):
    if value is None:
        return "unknown"
    try:
        return datetime.datetime.fromtimestamp(
            value, tz=datetime.timezone.utc
        ).isoformat()
    except (OverflowError, OSError, ValueError):
        return f"timestamp {value:g}"


async def _complete_browser_login(login, *, no_browser, provider):
    try:
        print("Open this URL in a browser:")
        print(login.authorization_url)
        if not no_browser:
            try:
                # The standard library's system-browser launcher is synchronous.
                # A text-mode browser must not block the loop that serves its callback.
                opened = await asyncio.to_thread(webbrowser.open, login.authorization_url, new=2)
            except (OSError, webbrowser.Error):
                opened = False
            if not opened:
                print("The browser could not be opened automatically; use the URL above.", file=sys.stderr)
        print(f"Waiting for {provider} authorization...")
        return await login.complete()
    finally:
        # Cancellation can arrive during browser launch, before complete().
        await login.aclose()


async def _login_openai(arguments, storage):
    if arguments.client_id is not None:
        print("--client-id is only supported for Anthropic login.", file=sys.stderr)
        return 2
    if arguments.device_code:
        authorization = (
            await oauth_logins.request_openai_device_authorization())
        print("Open this URL in a browser:")
        print(authorization.verification_url)
        print("Enter this one-time code:")
        print(authorization.user_code)
        print("Waiting for OpenAI authorization...")
        tokens = await oauth_logins.complete_openai_device_login(
            authorization)
    else:
        login = await oauth_logins.start_openai_browser_login()
        tokens = await _complete_browser_login(login, no_browser=arguments.no_browser, provider="OpenAI")

    await storage.store_openai_login(tokens)
    print("Logged in: OpenAI ChatGPT subscription")
    if tokens.account_id:
        print(f"Account: {tokens.account_id!r}")
    print(f"Credentials: {storage.file_path!r}")
    return 0


def _show_openai_status(storage):
    stored = storage.load_openai_subscription()
    if stored is None:
        print("OpenAI ChatGPT subscription: not logged in")
        return 1
    if stored.state != "active" or stored.tokens is None:
        print(
            "OpenAI ChatGPT subscription: login required "
            f"({stored.state})")
        return 1
    print("OpenAI ChatGPT subscription: logged in")
    if stored.tokens.account_id:
        print(f"Account: {stored.tokens.account_id!r}")
    print(f"Access token expires: {_time_text(stored.tokens.expires_at)}")
    print(f"Last refresh: {_time_text(stored.tokens.last_refresh)}")
    return 0


async def _login_anthropic(arguments, storage):
    if arguments.device_code:
        print(
            "Anthropic does not offer a device-code flow; use the browser "
            "flow (optionally with --no-browser).",
            file=sys.stderr,
        )
        return 2
    login = await oauth_logins.start_anthropic_browser_login(client_id=arguments.client_id)
    tokens = await _complete_browser_login(login, no_browser=arguments.no_browser, provider="Anthropic")

    await storage.store_anthropic_login(tokens)
    print("Logged in: Anthropic subscription")
    if tokens.account is not None:
        identity = tokens.account.email_address or tokens.account.uuid
        print(f"Account: {identity!r}")
    print(f"Credentials: {storage.file_path!r}")
    return 0


def _show_anthropic_status(storage):
    stored = storage.load_anthropic_subscription()
    if stored is None:
        print("Anthropic subscription: not logged in")
        return 1
    if stored.state != "active" or stored.tokens is None:
        print("Anthropic subscription: no active login")
        return 1
    print("Anthropic subscription: logged in")
    if stored.tokens.account is not None:
        identity = (
            stored.tokens.account.email_address
            or stored.tokens.account.uuid)
        print(f"Account: {identity!r}")
    print(f"Access token expires: {_time_text(stored.tokens.expires_at)}")
    print(f"Last refresh: {_time_text(stored.tokens.last_refresh)}")
    return 0


async def run(arguments, *, program="loki", storage=None):
    parsed = _parser(program).parse_args(arguments)
    credential_storage = (
        storage if storage is not None else credential_storages.JsonCredentialStorage())
    try:
        if parsed.command == "login":
            if parsed.provider == "anthropic":
                return await _login_anthropic(parsed, credential_storage)
            return await _login_openai(parsed, credential_storage)
        if parsed.command == "status":
            if parsed.provider == "anthropic":
                return _show_anthropic_status(credential_storage)
            if parsed.provider == "openai":
                return _show_openai_status(credential_storage)
            openai_code = _show_openai_status(credential_storage)
            anthropic_code = _show_anthropic_status(credential_storage)
            # Providers are alternatives; a missing optional login must not
            # make an existing login look like a failed status check.
            return min(openai_code, anthropic_code)
        if parsed.command == "logout":
            if parsed.provider == "anthropic":
                removed = await credential_storage.remove_anthropic_subscription()
                if removed:
                    print("Logged out: Anthropic subscription")
                else:
                    print("Anthropic subscription: not logged in")
                return 0
            removed = await credential_storage.remove_openai_subscription()
            if removed:
                print("Logged out: OpenAI ChatGPT subscription")
            else:
                print("OpenAI ChatGPT subscription: not logged in")
            return 0
        raise AssertionError("unhandled authentication command")
    finally:
        if storage is None:
            credential_storage.close()


def main(arguments, *, program="loki") -> int:
    try:
        return asyncio.run(run(arguments, program=program))
    except KeyboardInterrupt:
        print("Authentication cancelled.", file=sys.stderr)
        return 130
    except (
            credential_storages.CredentialStorageError,
            oauth_logins.OAuthLoginError,
    ) as error:
        print(f"Authentication error: {error}", file=sys.stderr)
        return 2
