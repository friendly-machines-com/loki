"""Credential supervisor and dependency-free runtime-isolation tests."""

import asyncio
import errno
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from loki_agent import acp
from loki_agent import authentications
from loki_agent import credential_runtimes
from loki_agent import credential_storages
from loki_agent import credential_supervisors
from loki_agent import paths
from loki_agent import runtime_isolations
from loki_agent.credentials import CredentialStore


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PathTests(unittest.TestCase):
    def test_credential_directory_uses_xdg_config_home(self):
        self.assertEqual(
            paths.credential_directory({
                "HOME": "/ignored",
                "XDG_CONFIG_HOME": "/configuration",
            }),
            "/configuration/loki/credentials",
        )

    def test_credential_directory_falls_back_to_home_config(self):
        with mock.patch.dict(
                os.environ, {"HOME": "/home/tester"}, clear=True):
            self.assertEqual(
                paths.credential_directory(),
                "/home/tester/.config/loki/credentials",
            )


class UnshareSelectionTests(unittest.TestCase):
    def test_python_312_os_unshare_is_preferred(self):
        standard = mock.Mock()
        libc = mock.Mock()

        with mock.patch.object(
                runtime_isolations.os, "unshare", standard,
                create=True):
            runtime_isolations._unshare_user_and_mount_namespaces(
                libc)

        standard.assert_called_once_with(
            runtime_isolations._CLONE_NEWUSER
            | runtime_isolations._CLONE_NEWNS)
        self.assertEqual(libc.mock_calls, [])

    def test_pre_312_libc_unshare_fallback_is_used_only_when_missing(self):
        class Unshare:
            def __init__(self):
                self.calls = []

            def __call__(self, flags):
                self.calls.append(flags)
                return 0

        class Libc:
            unshare = Unshare()

        with mock.patch.object(
                runtime_isolations.os, "unshare", None,
                create=True):
            runtime_isolations._unshare_user_and_mount_namespaces(
                Libc())

        self.assertEqual(
            Libc.unshare.calls,
            [
                runtime_isolations._CLONE_NEWUSER
                | runtime_isolations._CLONE_NEWNS,
            ],
        )


@unittest.skipUnless(
    sys.platform.startswith("linux") and os.path.isdir("/proc/self"),
    "Linux user and mount namespaces",
)
class LinuxIsolationTests(unittest.TestCase):
    def test_runtime_hides_only_its_credential_directory_and_drops_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            credentials = os.path.join(directory, "credentials")
            os.mkdir(credentials)
            marker = os.path.join(credentials, "secret")
            with open(marker, "w", encoding="ascii") as stream:
                stream.write("supervisor-visible")

            code = r"""
import ctypes
import json
import os
import subprocess
import sys

from loki_agent.runtime_isolations import isolate_credential_directory

target = sys.argv[1]
isolated = isolate_credential_directory(target)
try:
    os.listdir(target)
except PermissionError:
    hidden = True
else:
    hidden = False
try:
    with open(os.path.join(target, "secret"), encoding="ascii") as stream:
        stream.read()
except (FileNotFoundError, PermissionError):
    marker_hidden = True
else:
    marker_hidden = False
tool = subprocess.run(
    [
        sys.executable,
        "-c",
        (
            "import sys\n"
            "try:\n"
            "    stream = open(sys.argv[1], encoding='ascii')\n"
            "except (FileNotFoundError, PermissionError):\n"
            "    print('hidden')\n"
            "else:\n"
            "    stream.close()\n"
            "    print('visible')\n"
        ),
        os.path.join(target, "secret"),
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    check=False,
)
status = {}
with open("/proc/self/status", encoding="ascii") as stream:
    for line in stream:
        key, separator, value = line.partition(":")
        if separator and key in {
                "CapEff", "CapPrm", "CapBnd", "NoNewPrivs"}:
            status[key] = value.strip()
libc = ctypes.CDLL(None, use_errno=True)
unmount_result = libc.umount2(os.fsencode(target), 0)
print(json.dumps({
    "isolated": isolated,
    "hidden": hidden,
    "marker_hidden": marker_hidden,
    "tool_hidden": tool.returncode == 0 and tool.stdout.strip() == "hidden",
    "status": status,
    "unmount_result": unmount_result,
    "unmount_errno": ctypes.get_errno(),
}))
"""
            process = subprocess.run(
                [sys.executable, "-c", code, credentials],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertTrue(result["isolated"])
            self.assertTrue(result["hidden"])
            self.assertTrue(result["marker_hidden"])
            self.assertTrue(result["tool_hidden"])
            self.assertEqual(result["status"]["CapEff"], "0" * 16)
            self.assertEqual(result["status"]["CapPrm"], "0" * 16)
            self.assertEqual(result["status"]["CapBnd"], "0" * 16)
            self.assertEqual(result["status"]["NoNewPrivs"], "1")
            self.assertEqual(result["unmount_result"], -1)
            self.assertEqual(result["unmount_errno"], errno.EPERM)

            # The child changed only its private mount namespace.
            with open(marker, encoding="ascii") as stream:
                self.assertEqual(
                    stream.read(), "supervisor-visible")


class CredentialSupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_and_acp_supervisors_load_same_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = credential_storages.JsonCredentialStorage(
                os.path.join(temporary, "credentials"))
            stored_tokens = authentications.OpenAITokenSet(
                access_token="access-secret",
                refresh_token="refresh-secret",
                account_id="account",
                expires_at=10**12,
            )
            await storage.store_openai_login(stored_tokens)
            terminal_supervisor = (
                credential_supervisors.CredentialSupervisor(
                    CredentialStore({}), storage))
            acp_front = acp.Front(
                lambda: None,
                lambda _message: None,
                CredentialStore({}),
                storage,
            )
            credential = (
                authentications.CredentialRef.openai_subscription())

            terminal_lease = await terminal_supervisor.broker.lease(
                credential)
            acp_lease = await acp_front.credential_broker.lease(
                credential)

            self.assertTrue(
                terminal_supervisor.inventory.has_ref(credential))
            self.assertTrue(acp_front.credentials.has_ref(credential))
            self.assertEqual(
                terminal_lease.value, "access-secret")
            self.assertEqual(acp_lease.value, "access-secret")
            self.assertFalse(hasattr(terminal_lease, "refresh_token"))
            self.assertFalse(hasattr(acp_lease, "refresh_token"))

    async def test_persistent_subscription_is_leased_not_inherited(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = credential_storages.JsonCredentialStorage(
                os.path.join(temporary, "credentials"))
            tokens = authentications.OpenAITokenSet(
                access_token="access-secret",
                refresh_token="refresh-secret",
                account_id="account",
                expires_at=10**12,
            )
            await storage.store_openai_login(tokens)
            supervisor = credential_supervisors.CredentialSupervisor(
                CredentialStore({}), storage)
            credential = (
                authentications.CredentialRef.openai_subscription())

            self.assertTrue(supervisor.inventory.has_ref(credential))
            self.assertNotIn(
                "access-secret", repr(supervisor.environment))
            self.assertNotIn(
                "refresh-secret", repr(supervisor.inventory))
            lease = await supervisor.broker.lease(credential)

            self.assertEqual(lease.value, "access-secret")
            self.assertTrue(lease.refreshable)
            self.assertFalse(hasattr(lease, "refresh_token"))

    async def test_subscription_redelegates_to_nested_runtime(self):
        credential = (
            authentications.CredentialRef.openai_subscription())
        broker = authentications.CredentialBroker()
        broker.install_openai_subscription(
            authentications.OpenAITokenSet(
                access_token="access-secret",
                refresh_token="refresh-secret",
                expires_at=10**12,
            ))
        outer = await credential_supervisors.RuntimeDelegation.create(
            broker, {credential})
        outer_owner = os.dup(outer.owner_read_fd)
        outer_capability = os.dup(outer.credential_fd)
        outer.child_spawned()
        outer_runtime = None
        inner = None
        inner_runtime = None
        try:
            outer_runtime = (
                await credential_runtimes.CredentialRuntime.connect(
                    outer_owner, outer_capability))
            outer_session = types.SimpleNamespace(
                credential_authority=None)
            outer_runtime.install(outer_session)

            inner = (
                await credential_supervisors.RuntimeDelegation.create(
                    outer_session.credential_authority,
                    {credential},
                ))
            inner_owner = os.dup(inner.owner_read_fd)
            inner_capability = os.dup(inner.credential_fd)
            inner.child_spawned()
            inner_runtime = (
                await credential_runtimes.CredentialRuntime.connect(
                    inner_owner, inner_capability))
            inner_session = types.SimpleNamespace(
                credential_authority=None)
            inventory = inner_runtime.install(inner_session)

            lease = await inner_session.credential_authority.lease(
                credential)

            self.assertTrue(inventory.has_ref(credential))
            self.assertEqual(lease.value, "access-secret")
            self.assertTrue(lease.refreshable)
            self.assertFalse(hasattr(lease, "refresh_token"))
        finally:
            if inner_runtime is not None:
                await inner_runtime.close()
            if inner is not None:
                await inner.close()
            if outer_runtime is not None:
                await outer_runtime.close()
            await outer.close()

    async def test_incomplete_refresh_is_not_installed(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = credential_storages.JsonCredentialStorage(
                os.path.join(temporary, "credentials"))
            tokens = authentications.OpenAITokenSet(
                access_token="access-secret",
                refresh_token="refresh-secret",
                expires_at=10**12,
            )
            await storage.store_openai_login(tokens)
            current = storage.load_openai_subscription().tokens

            async def refresh(_value):
                raise authentications.RefreshTransientError(
                    "ambiguous", request_may_have_been_sent=True)

            with self.assertRaises(
                    authentications.RefreshTransientError):
                await storage.rotate_openai_subscription(
                    current, refresh=refresh)
            supervisor = credential_supervisors.CredentialSupervisor(
                CredentialStore({}), storage)

            self.assertFalse(supervisor.inventory.has_ref(
                authentications.CredentialRef.openai_subscription()))

    async def test_broker_refresh_updates_persistent_token_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = credential_storages.JsonCredentialStorage(
                os.path.join(temporary, "credentials"))
            tokens = authentications.OpenAITokenSet(
                access_token="access-old",
                refresh_token="refresh-old",
                expires_at=1,
                last_refresh=1,
            )
            await storage.store_openai_login(tokens)
            calls = []

            async def refresh(value):
                calls.append(value)
                return authentications.RefreshResult(
                    access_token="access-new",
                    refresh_token="refresh-new",
                )

            rotate_persisted = storage.rotate_openai_subscription

            async def rotate(current):
                return await rotate_persisted(
                    current,
                    refresh=refresh,
                    clock=lambda: 100,
                )

            storage.rotate_openai_subscription = rotate
            supervisor = credential_supervisors.CredentialSupervisor(
                CredentialStore({}), storage)
            credential = (
                authentications.CredentialRef.openai_subscription())

            lease = await supervisor.broker.lease(credential)

            self.assertEqual(lease.value, "access-new")
            self.assertEqual(calls, ["refresh-old"])
            self.assertEqual(
                storage.load_openai_subscription().tokens.refresh_token,
                "refresh-new",
            )

    async def test_static_environment_credential_is_leased_through_capability(
            self):
        name = "EXAMPLE_API_KEY"
        value = "supervisor-only-secret"
        supervisor = credential_supervisors.CredentialSupervisor(
            CredentialStore({name: value}))
        delegation = await supervisor.delegate()
        owner_fd = os.dup(delegation.owner_read_fd)
        capability_fd = os.dup(delegation.credential_fd)
        delegation.child_spawned()
        runtime = None
        try:
            runtime = await credential_runtimes.CredentialRuntime.connect(
                owner_fd, capability_fd)
            self.assertIsNotNone(runtime)
            session = types.SimpleNamespace(credential_authority=None)
            inventory = runtime.install(session)
            credential = authentications.CredentialRef.environment(name)

            self.assertIs(
                session.credential_authority,
                runtime.credential_client,
            )
            self.assertTrue(inventory.has_ref(credential))
            self.assertEqual(inventory.get(name), "")
            lease = await session.credential_authority.lease(credential)
            self.assertEqual(lease.value, value)
        finally:
            if runtime is not None:
                await runtime.close()
            await delegation.close()

    async def test_terminal_runtime_receives_only_delegated_descriptors(self):
        store = CredentialStore({
            "LOKI_API_KEY": "must-not-enter-runtime-environment",
            "LOKI_MODEL": "model",
        })
        supervisor = credential_supervisors.CredentialSupervisor(
            store)
        spawned = {}

        class Process:
            returncode = 0

            async def wait(self):
                return 7

        async def spawn(*args, **kwargs):
            spawned["args"] = args
            spawned["kwargs"] = kwargs
            return Process()

        with mock.patch.object(
                credential_supervisors.asyncio,
                "create_subprocess_exec",
                new=spawn):
            status = await supervisor.run_terminal_runtime(
                "/installed/bin/loki", ["--headless"])

        self.assertEqual(status, 7)
        self.assertEqual(spawned["args"][0:2], (
            "/installed/bin/loki", "--runtime"))
        self.assertIn("--session-owner-fd", spawned["args"])
        self.assertIn(
            "--credential-capability-fd", spawned["args"])
        self.assertEqual(
            spawned["args"][-2:], ("--", "--headless"))
        self.assertEqual(len(spawned["kwargs"]["pass_fds"]), 2)
        self.assertTrue(spawned["kwargs"]["close_fds"])
        self.assertNotIn(
            "LOKI_API_KEY", spawned["kwargs"]["env"])
        self.assertNotIn(
            "must-not-enter-runtime-environment",
            repr(spawned["kwargs"]["env"]),
        )

    async def test_supervisor_revokes_then_allows_clean_runtime_exit(self):
        store = CredentialStore({})
        supervisor = credential_supervisors.CredentialSupervisor(
            store)
        revoked = asyncio.Event()
        closed = asyncio.Event()

        class Delegation:
            def child_arguments(self):
                return []

            def child_fds(self):
                return ()

            def child_spawned(self):
                pass

            def revoke_now(self):
                revoked.set()

            async def close(self):
                closed.set()

        class Process:
            returncode = None
            terminated = False
            killed = False

            async def wait(self):
                await revoked.wait()
                self.returncode = 23
                return self.returncode

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

        process = Process()

        async def spawn(*args, **kwargs):
            return process

        with mock.patch.object(
                supervisor, "delegate",
                new=mock.AsyncMock(return_value=Delegation())), \
                mock.patch.object(
                    credential_supervisors.asyncio,
                    "create_subprocess_exec",
                    new=spawn):
            task = asyncio.create_task(
                supervisor.run_terminal_runtime("/loki", []))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(revoked.is_set())
        self.assertTrue(closed.is_set())
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)


class CredentialRuntimeCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_closes_even_when_transport_close_fails(self):
        class Owner:
            closed = False

            async def close(self):
                self.closed = True

        class Client:
            async def close(self):
                raise OSError("transport close failed")

        owner = Owner()
        runtime = credential_runtimes.CredentialRuntime(
            owner, Client())

        with self.assertRaisesRegex(OSError, "transport close failed"):
            await runtime.close()
        self.assertTrue(owner.closed)


if __name__ == "__main__":
    unittest.main()
