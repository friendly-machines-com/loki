import asyncio
import base64
import hashlib
import json
import threading
import time
import urllib.parse
import urllib.request
import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from loki_entrypoints import configure_container, entrypoint

from loki_agent import authentication_commands
from loki_agent import authentications
from loki_agent import credential_storages
from loki_agent import credential_supervisors, http_client, oauth_logins
from loki_agent.credentials import CredentialStore
from tests.test_oauth_logins import jwt, response


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

    async def test_browser_command_through_durable_subscription_lifecycle(self):
        await self._subscription_workflow(device=False)

    async def test_device_command_through_durable_subscription_lifecycle(self):
        await self._subscription_workflow(device=True)

    async def _subscription_workflow(self, *, device):
        output, errors = io.StringIO(), io.StringIO()
        access = jwt({"exp": int(time.time()) + 3600, "nonce": "access-secret"})
        identity = jwt({"https://api.openai.com/auth": {
            "chatgpt_account_id": "account", "chatgpt_account_is_fedramp": True}})
        query = {}
        requests = []
        callback_answers = []
        loop_thread = threading.get_ident()
        poll_times = []
        refresh_started, release_refresh = asyncio.Event(), asyncio.Event()

        def challenge(verifier):
            # Independent oracle: never call the production PKCE helper.
            return base64.urlsafe_b64encode(hashlib.sha256(
                verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")

        def browser(url, *, new):
            self.assertNotEqual(threading.get_ident(), loop_thread)
            self.assertEqual(new, 2)
            query.update(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query))
            for key, value in {
                    "response_type": "code", "scope": oauth_logins.OPENAI_OAUTH_SCOPE,
                    "code_challenge_method": "S256",
                    "codex_cli_simplified_flow": "true",
                    "id_token_add_organizations": "true",
                    "originator": authentications.OPENAI_ORIGINATOR}.items():
                self.assertEqual(query[key], [value])
            self.assertIsNone(self.storage.load_openai_subscription())
            target = query["redirect_uri"][0] + "?" + urllib.parse.urlencode({
                "code": "authorization-secret", "state": query["state"][0]})
            with urllib.request.urlopen(target, timeout=5) as answer:
                callback_answers.append((answer.status, answer.read()))
            return True

        async def request(method, url, **kwargs):
            requests.append(url)
            self.assertEqual(method, "POST")
            self.assertEqual(kwargs["retry_max_attempts"], 1)
            content_type = kwargs["headers_in"]["Content-Type"]
            if content_type == "application/json":
                body = json.loads(kwargs["body"])
            else:
                self.assertEqual(content_type, "application/x-www-form-urlencoded")
                body = {k: v[0] for k, v in urllib.parse.parse_qs(
                    kwargs["body"].decode("ascii")).items()}
            self.assertEqual(body.get("client_id", authentications.OPENAI_OAUTH_CLIENT_ID),
                             authentications.OPENAI_OAUTH_CLIENT_ID)
            if body.get("grant_type") == "refresh_token":
                self.assertEqual(body["refresh_token"], "refresh-secret")
                self.assertEqual(self.storage.load_openai_subscription().state,
                                 "refreshing")
                refresh_started.set()
                await release_refresh.wait()
                return response(url, 200, {"access_token": "access-rotated-secret",
                                           "refresh_token": "refresh-rotated-secret",
                                           "id_token": identity})
            self.assertIsNone(self.storage.load_openai_subscription())
            if url == oauth_logins.OPENAI_DEVICE_USER_CODE_URL:
                return response(url, 200, {"device_auth_id": "device-secret",
                                           "user_code": "ABCD-EFGH", "interval": "1"})
            if url == oauth_logins.OPENAI_DEVICE_POLL_URL:
                self.assertEqual(body, {"device_auth_id": "device-secret",
                                        "user_code": "ABCD-EFGH"})
                self.assertIn("ABCD-EFGH", output.getvalue())
                self.assertIn(oauth_logins.OPENAI_DEVICE_VERIFICATION_URL,
                              output.getvalue())
                poll_times.append(asyncio.get_running_loop().time())
                if len(poll_times) == 1:
                    return response(url, 403, {})
                return response(url, 200, {"authorization_code": "authorization-secret",
                                           "code_verifier": "device-verifier-secret",
                                           "code_challenge": challenge("device-verifier-secret")})
            self.assertEqual(url, oauth_logins.OPENAI_TOKEN_URL)
            self.assertEqual(body["grant_type"], "authorization_code")
            self.assertEqual(body["code"], "authorization-secret")
            if device:
                self.assertEqual(body["redirect_uri"], oauth_logins.OPENAI_DEVICE_REDIRECT_URL)
                self.assertEqual(body["code_verifier"], "device-verifier-secret")
            else:
                self.assertEqual(body["redirect_uri"], query["redirect_uri"][0])
                self.assertEqual(challenge(body["code_verifier"]),
                                 query["code_challenge"][0])
                self.assertGreaterEqual(len(body["code_verifier"]), 43)
            return response(url, 200, {"access_token": access,
                                       "refresh_token": "refresh-secret", "id_token": identity})

        with mock.patch.object(http_client, "async_http_request", new=request), \
                mock.patch.object(authentication_commands.webbrowser, "open",
                                  side_effect=browser) as launcher, \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            args = ["login", "openai"] + (["--device-code"] if device else [])
            self.assertEqual(await asyncio.wait_for(authentication_commands.run(
                args, storage=self.storage), 10), 0)
            if device:
                launcher.assert_not_called()
                self.assertEqual(len(poll_times), 2)
                self.assertGreaterEqual(poll_times[1] - poll_times[0], 1)
                self.assertEqual(requests, [
                    oauth_logins.OPENAI_DEVICE_USER_CODE_URL,
                    oauth_logins.OPENAI_DEVICE_POLL_URL,
                    oauth_logins.OPENAI_DEVICE_POLL_URL,
                    oauth_logins.OPENAI_TOKEN_URL])
            else:
                launcher.assert_called_once()
                self.assertEqual(callback_answers[0][0], 200)
                redirect = urllib.parse.urlsplit(query["redirect_uri"][0])
                with self.assertRaises(OSError):
                    await asyncio.open_connection("127.0.0.1", redirect.port)
                self.assertEqual(requests, [oauth_logins.OPENAI_TOKEN_URL])
            stored = self.storage.load_openai_subscription()
            self.assertEqual(stored.state, "active")
            self.assertEqual(stored.tokens.access_token, access)
            self.assertEqual(stored.tokens.refresh_token, "refresh-secret")
            self.assertEqual(stored.tokens.id_token, identity)
            self.assertEqual(stored.tokens.account_id, "account")
            self.assertTrue(stored.tokens.fedramp)
            from loki_agent import private_files
            for path in (self.storage.directory, self.storage.file_path):
                self.assertFalse(private_files.describe_path(path).group_or_other_access)
            self.assertFalse(any(name.startswith(".tokens.json.")
                                 for name in os.listdir(self.storage.directory)))
            self.assertNotIn(access, repr(stored.tokens))
            reopened = credential_storages.JsonCredentialStorage(self.storage.directory)
            supervisor = credential_supervisors.CredentialSupervisor(CredentialStore({}), reopened)
            ref = authentications.CredentialRef.openai_subscription()
            spec = authentications.AuthSpec(
                ref, scheme="openai-subscription",
                authorized_urls=authentications.OPENAI_CHATGPT_CODEX_URLS)
            url = authentications.OPENAI_CHATGPT_RESPONSES_URL
            headers, lease = await authentications.authorized_request_headers(supervisor.broker, spec, url)
            self.assertEqual(headers["Authorization"], "Bearer " + access)
            with self.assertRaises(authentications.CredentialUnavailable):
                await authentications.authorized_request_headers(supervisor.broker, spec,
                                                                 "https://evil.invalid/")
            tasks = [asyncio.create_task(authentications.authorized_request_headers(
                supervisor.broker, spec, url, rejected_generation=lease.generation))
                for _ in range(2)]
            try:
                await asyncio.wait_for(refresh_started.wait(), 5)
            finally:
                release_refresh.set()
            rotated = await asyncio.wait_for(asyncio.gather(*tasks), 5)
            self.assertEqual([h["Authorization"] for h, _ in rotated],
                             ["Bearer access-rotated-secret"] * 2)
            self.assertEqual(requests.count(authentications.OPENAI_REFRESH_URL), 2)
            durable = credential_storages.JsonCredentialStorage(self.storage.directory)
            self.assertEqual(durable.load_openai_subscription().tokens.refresh_token,
                             "refresh-rotated-secret")
            restarted = credential_supervisors.CredentialSupervisor(CredentialStore({}), durable)
            self.assertEqual((await restarted.broker.lease(ref)).value, "access-rotated-secret")
            revision = durable.load_document()["revision"]
            self.assertEqual(await authentication_commands.run(["status", "openai"], storage=durable), 0)
            self.assertEqual(await authentication_commands.run(["logout", "openai"], storage=durable), 0)
            self.assertEqual(await authentication_commands.run(["logout", "openai"], storage=durable), 0)
            self.assertEqual(await authentication_commands.run(["status", "openai"], storage=durable), 1)
            self.assertGreater(durable.load_document()["revision"], revision)
            self.assertEqual(durable.load_document()["credentials"], {})
            logged_out = credential_supervisors.CredentialSupervisor(
                CredentialStore({}),
                credential_storages.JsonCredentialStorage(self.storage.directory))
            self.assertFalse(logged_out.inventory.has_ref(ref))
            with self.assertRaises(authentications.CredentialUnavailable):
                await logged_out.broker.lease(ref)
        self.assertEqual(errors.getvalue(), "")
        self.assertIn("Logged in: OpenAI ChatGPT subscription", output.getvalue())
        for secret in (access, identity, "refresh-secret", "access-rotated-secret",
                       "refresh-rotated-secret", "device-secret", "authorization-secret"):
            self.assertNotIn(secret, output.getvalue())

    async def test_failed_login_preserves_previous_credential(self):
        await self._interrupted_browser(cancel=False)

    async def test_cancelled_login_preserves_previous_credential(self):
        await self._interrupted_browser(cancel=True)

    async def _interrupted_browser(self, *, cancel):
        previous = tokens()
        await self.storage.store_openai_login(previous)
        ready = asyncio.Event()

        class Output(io.StringIO):
            def write(inner_self, text):
                result = super().write(text)
                if text.startswith(oauth_logins.OPENAI_AUTHORIZE_URL):
                    ready.set()
                return result

        output = Output()
        with contextlib.redirect_stdout(output):
            task = asyncio.create_task(authentication_commands.run(
                ["login", "openai", "--no-browser"], storage=self.storage))
            try:
                await asyncio.wait_for(ready.wait(), 5)
                url = next(line for line in output.getvalue().splitlines()
                           if line.startswith(oauth_logins.OPENAI_AUTHORIZE_URL))
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
                redirect = urllib.parse.urlsplit(query["redirect_uri"][0])
                if cancel:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    reader, writer = await asyncio.open_connection("127.0.0.1", redirect.port)
                    target = "/auth/callback?" + urllib.parse.urlencode({
                        "state": query["state"][0], "error": "access_denied"})
                    try:
                        writer.write(f"GET {target} HTTP/1.1\r\n\r\n".encode("ascii"))
                        await writer.drain()
                        self.assertIn(b"400 Bad Request", await reader.read())
                    finally:
                        writer.close()
                        await writer.wait_closed()
                    with self.assertRaisesRegex(oauth_logins.OAuthLoginError, "declined"):
                        await asyncio.wait_for(task, 5)
                with self.assertRaises(OSError):
                    await asyncio.open_connection("127.0.0.1", redirect.port)
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
        self.assertEqual(self.storage.load_openai_subscription().tokens,
                         previous.normalized())

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
    def test_dummy_frontend_startup_compatibility_not_subscription_consumption(self):
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
                "XDG_STATE_HOME": os.path.join(temporary, "state"),
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_PROVIDER": "dummy",
                "LOKI_MODEL": "dummy-model",
            })

            workspace = os.path.join(temporary, "workspace")
            os.makedirs(workspace)
            configure_container(environment, workspace)
            terminal = subprocess.run(
                [
                    entrypoint("loki"),
                    "--headless",
                ],
                input="",
                env=environment,
                cwd=workspace,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
            acp = subprocess.run(
                [entrypoint("loki-acp")],
                input="",
                env=environment,
                cwd=workspace,
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
                entrypoint("loki"),
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
