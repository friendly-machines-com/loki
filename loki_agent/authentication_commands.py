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
    login.add_argument("provider", choices=["openai"])
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
        "provider", nargs="?", choices=["openai"])

    logout = commands.add_parser(
        "logout", help="remove a persistent login")
    logout.add_argument("provider", choices=["openai"])
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


async def _login_openai(arguments, storage):
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
        print("Open this URL in a browser:")
        print(login.authorization_url)
        if not arguments.no_browser:
            try:
                opened = webbrowser.open(
                    login.authorization_url, new=2)
            except webbrowser.Error:
                opened = False
            if not opened:
                print(
                    "The browser could not be opened automatically; "
                    "use the URL above.",
                    file=sys.stderr,
                )
        print("Waiting for OpenAI authorization...")
        tokens = await login.complete()

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


async def run(arguments, *, program="loki", storage=None):
    parsed = _parser(program).parse_args(arguments)
    credential_storage = (
        storage or credential_storages.JsonCredentialStorage())
    if parsed.command == "login":
        return await _login_openai(parsed, credential_storage)
    if parsed.command == "status":
        return _show_openai_status(credential_storage)
    if parsed.command == "logout":
        removed = (
            await credential_storage.remove_openai_subscription())
        if removed:
            print("Logged out: OpenAI ChatGPT subscription")
        else:
            print("OpenAI ChatGPT subscription: not logged in")
        return 0
    raise AssertionError("unhandled authentication command")


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
