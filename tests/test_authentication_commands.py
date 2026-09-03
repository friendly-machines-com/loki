import asyncio
import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from loki_agent import authentication_commands
from loki_agent import authentications
from loki_agent import credential_storages


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tokens():
    return authentications.OpenAITokenSet(
        access_token="access-secret",
        refresh_token="refresh-secret",
        id_token="id-secret",
        account_id="account",
        expires_at=10**12,
        last_refresh=100,
    )


class AuthenticationCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.storage = credential_storages.JsonCredentialStorage(
            os.path.join(self.temporary.name, "credentials"))

    async def test_browser_login_persists_only_after_completion(self):
        class Login:
            authorization_url = "https://auth.example/authorize"

            async def complete(inner_self):
                self.assertIsNone(
                    self.storage.load_openai_subscription())
                return tokens()

        output = io.StringIO()
        with mock.patch.object(
                authentication_commands.oauth_logins,
                "start_openai_browser_login",
                new=mock.AsyncMock(return_value=Login())), \
                mock.patch.object(
                    authentication_commands.webbrowser,
                    "open",
                    return_value=True) as browser, \
                contextlib.redirect_stdout(output):
            status = await authentication_commands.run(
                ["login", "openai"], storage=self.storage)

        self.assertEqual(status, 0)
        browser.assert_called_once_with(
            "https://auth.example/authorize", new=2)
        self.assertEqual(
            self.storage.load_openai_subscription().tokens,
            tokens().normalized(),
        )
        rendered = output.getvalue()
        self.assertNotIn("access-secret", rendered)
        self.assertNotIn("refresh-secret", rendered)
        self.assertIn("OpenAI ChatGPT subscription", rendered)

    async def test_device_login_uses_device_protocol(self):
        authorization = mock.Mock(
            verification_url="https://auth.example/device",
            user_code="ABCD-EFGH",
        )
        with mock.patch.object(
                authentication_commands.oauth_logins,
                "request_openai_device_authorization",
                new=mock.AsyncMock(return_value=authorization)), \
                mock.patch.object(
                    authentication_commands.oauth_logins,
                    "complete_openai_device_login",
                    new=mock.AsyncMock(return_value=tokens())) as complete, \
                mock.patch.object(
                    authentication_commands.webbrowser,
                    "open") as browser, \
                contextlib.redirect_stdout(io.StringIO()):
            status = await authentication_commands.run(
                ["login", "openai", "--device-code"],
                storage=self.storage,
            )

        self.assertEqual(status, 0)
        complete.assert_awaited_once_with(authorization)
        browser.assert_not_called()

    async def test_browser_launcher_runs_outside_the_callback_loop(self):
        class Login:
            authorization_url = "https://auth.example/authorize"

            async def complete(inner_self):
                return tokens()

        offload = mock.AsyncMock(return_value=True)
        with mock.patch.object(
                authentication_commands.oauth_logins,
                "start_openai_browser_login",
                new=mock.AsyncMock(return_value=Login())), \
                mock.patch.object(
                    authentication_commands.asyncio,
                    "to_thread",
                    new=offload), \
                contextlib.redirect_stdout(io.StringIO()):
            await authentication_commands.run(
                ["login", "openai"], storage=self.storage)

        offload.assert_awaited_once_with(
            authentication_commands.webbrowser.open,
            "https://auth.example/authorize",
            new=2,
        )

    async def test_failed_login_preserves_previous_credential(self):
        previous = tokens()
        await self.storage.store_openai_login(previous)

        class Login:
            authorization_url = "https://auth.example/authorize"

            async def complete(inner_self):
                raise authentication_commands.oauth_logins.OAuthLoginError(
                    "declined")

        with mock.patch.object(
                authentication_commands.oauth_logins,
                "start_openai_browser_login",
                new=mock.AsyncMock(return_value=Login())), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(
                    authentication_commands.oauth_logins.OAuthLoginError):
                await authentication_commands.run(
                    ["login", "openai", "--no-browser"],
                    storage=self.storage,
                )

        self.assertEqual(
            self.storage.load_openai_subscription().tokens,
            previous.normalized(),
        )

    async def test_cancelled_login_preserves_previous_credential(self):
        previous = tokens()
        await self.storage.store_openai_login(previous)

        class Login:
            authorization_url = "https://auth.example/authorize"

            async def complete(inner_self):
                raise asyncio.CancelledError()

        with mock.patch.object(
                authentication_commands.oauth_logins,
                "start_openai_browser_login",
                new=mock.AsyncMock(return_value=Login())), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(asyncio.CancelledError):
                await authentication_commands.run(
                    ["login", "openai", "--no-browser"],
                    storage=self.storage,
                )

        self.assertEqual(
            self.storage.load_openai_subscription().tokens,
            previous.normalized(),
        )

    async def test_status_and_logout_do_not_render_tokens(self):
        await self.storage.store_openai_login(tokens())
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            status = await authentication_commands.run(
                ["status", "openai"], storage=self.storage)
            logout = await authentication_commands.run(
                ["logout", "openai"], storage=self.storage)

        self.assertEqual(status, 0)
        self.assertEqual(logout, 0)
        self.assertIsNone(
            self.storage.load_openai_subscription())
        rendered = output.getvalue()
        self.assertNotIn("access-secret", rendered)
        self.assertNotIn("refresh-secret", rendered)

    async def test_status_reports_interrupted_refresh(self):
        await self.storage.store_openai_login(tokens())

        async def refresh(_value):
            raise authentications.RefreshTransientError(
                "ambiguous", request_may_have_been_sent=True)

        with self.assertRaises(
                authentications.RefreshTransientError):
            await self.storage.rotate_openai_subscription(
                tokens().normalized(), refresh=refresh)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = await authentication_commands.run(
                ["status"], storage=self.storage)

        self.assertEqual(status, 1)
        self.assertIn("login required", output.getvalue())


class AuthenticationEntrypointTests(unittest.TestCase):
    def test_real_terminal_and_acp_accept_same_persistent_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_home = os.path.join(temporary, "config")
            storage = credential_storages.JsonCredentialStorage(
                os.path.join(
                    config_home, "loki", "credentials"))
            asyncio.run(storage.store_openai_login(tokens()))
            environment = dict(os.environ)
            environment.update({
                "HOME": temporary,
                "XDG_CONFIG_HOME": config_home,
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_PROVIDER": "dummy",
                "LOKI_MODEL": "dummy-model",
            })

            terminal = subprocess.run(
                [
                    os.path.join(ROOT, "loki.py"),
                    "--headless",
                ],
                input="",
                env=environment,
                cwd=temporary,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
            acp = subprocess.run(
                [os.path.join(ROOT, "loki-acp")],
                input="",
                env=environment,
                cwd=temporary,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )

        self.assertEqual(terminal.returncode, 0, terminal.stderr)
        self.assertEqual(acp.returncode, 0, acp.stderr)

    def test_real_loki_status_and_logout_use_xdg_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_home = os.path.join(temporary, "config")
            storage = credential_storages.JsonCredentialStorage(
                os.path.join(config_home, "loki", "credentials"))
            asyncio.run(
                storage.store_openai_login(tokens()))
            environment = dict(os.environ)
            environment.update({
                "HOME": temporary,
                "XDG_CONFIG_HOME": config_home,
            })
            command = [
                os.path.join(ROOT, "loki.py"),
                "auth",
            ]

            status = subprocess.run(
                command + ["status", "openai"],
                env=environment,
                cwd=temporary,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            logout = subprocess.run(
                command + ["logout", "openai"],
                env=environment,
                cwd=temporary,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(logout.returncode, 0, logout.stderr)
        self.assertIn(
            "OpenAI ChatGPT subscription: logged in",
            status.stdout,
        )
        self.assertNotIn("access-secret", status.stdout)
        self.assertNotIn("refresh-secret", status.stdout)
