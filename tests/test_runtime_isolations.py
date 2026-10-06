"""Credential supervisor and dependency-free runtime-isolation tests."""

import asyncio
import contextlib
import errno
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from loki_agent import acp
from loki_agent import authentications
from loki_agent import credential_runtimes
from loki_agent import credential_storages
from loki_agent import credential_supervisors
from loki_agent import paths
from loki_agent import runtime_isolation
from loki_agent import runtime_isolations
from loki_agent import windows_api
from loki_agent.credentials import CredentialStore


def _private_end(endpoint):
    """A copy of an endpoint the delegation still owns.

    POSIX duplicates the descriptor so ``child_spawned`` can close the
    delegation's copy; a Windows endpoint is used directly and the delegation
    is closed last instead.
    """
    return os.dup(endpoint) if os.name == "posix" else endpoint


def _hand_off(delegation):
    """Release the delegation's own copies of the child ends (POSIX only)."""
    if os.name == "posix":
        delegation.child_spawned()


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PathTests(unittest.TestCase):
    def test_credential_directory_uses_xdg_config_home(self):
        # os.path.join, not a literal: the separator is the platform's.
        self.assertEqual(
            paths.credential_directory({
                "HOME": "/ignored",
                "XDG_CONFIG_HOME": "/configuration",
            }),
            os.path.join("/configuration", "loki", "credentials"),
        )

    def test_credential_directory_falls_back_to_home_config_on_posix(self):
        config_home = os.path.join("/home/tester", ".config")
        with mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(
                    os.path, "expanduser", return_value=config_home), \
                mock.patch.dict(
                    os.environ, {"HOME": "/home/tester"}, clear=True):
            self.assertEqual(
                paths.credential_directory(),
                os.path.join(config_home, "loki", "credentials"),
            )

    def test_credential_directory_uses_local_app_data_on_windows(self):
        base = os.path.join(tempfile.gettempdir(), "Local")
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(windows_api, "known_folder",
                                  return_value=base), \
                mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                paths.credential_directory(),
                os.path.join(base, "loki", "credentials"),
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


@unittest.skipUnless(sys.platform.startswith("linux"),
                     "the POSIX enforcement seam (unshare, /proc); the "
                     "Windows side is test_runtime_gate")
class LinuxIsolationTests(unittest.TestCase):
    """The POSIX enforcement seam of the isolation property.

    The property is the same on Windows -- the contained runtime cannot read
    the credential tree, and its ambient cwd is not that tree -- but the seam
    is not.  POSIX hides the tree from a same-UID process by unsharing user and
    mount namespaces inside the runtime itself, which is the production call
    these tests exercise.  Windows enforces the same property at
    ``CreateProcess`` with an AppContainer token, and the runtime only verifies
    its token; that side is
    ``test_windows_appcontainers.AppContainerTests.test_runtime_gate`` and, for
    the cwd half,
    ``test_windows_runtime.IsolationSeamTests.test_runtime_cwd_is_the_supervisor_cwd_not_the_workspace``.
    They are separate because the subject, the fixtures and the observations
    differ, not because the property does.
    """

    def test_storage_free_runtime_sets_no_new_privileges_from_known_state(self):
        # A real child can inherit an already-set flag. This native-call seam
        # starts at zero as well as one, so omitting the setter cannot pass.
        for initial in [0, 1]:
            with self.subTest(initial=initial), \
                    tempfile.TemporaryDirectory() as directory:
                state = initial
                calls = []

                def prctl(*arguments):
                    nonlocal state
                    values = list(argument.value for argument in arguments)
                    calls.append(values)
                    self.assertEqual(values, [38, 1, 0, 0, 0])
                    state = 1
                    return 0

                libc = mock.Mock(prctl=mock.Mock(side_effect=prctl))
                with mock.patch.object(runtime_isolations.ctypes, "CDLL",
                                       return_value=libc), \
                        mock.patch.object(
                            runtime_isolations,
                            "_unshare_user_and_mount_namespaces") as unshare:
                    self.assertFalse(
                        runtime_isolations.isolate_credential_directory(
                            os.path.join(directory, "missing")))
                    self.assertEqual(state, 1)
                    self.assertEqual(calls, [[38, 1, 0, 0, 0]])
                    # Repeated storage-free initialization is still protected.
                    self.assertFalse(
                        runtime_isolations.isolate_credential_directory(
                            os.path.join(directory, "missing")))
                    self.assertEqual(state, 1)
                    self.assertEqual(calls, [[38, 1, 0, 0, 0]] * 2)
                    unshare.assert_not_called()

    def test_missing_credentials_native_privilege_transition_and_exec(self):
        with tempfile.TemporaryDirectory() as directory:
            code = r"""
import ctypes
import json
import subprocess
import sys
from loki_agent.runtime_isolations import isolate_credential_directory

libc = ctypes.CDLL(None)
before = libc.prctl(39, 0, 0, 0, 0)  # PR_GET_NO_NEW_PRIVS
if before == 1:
    sys.exit(77)  # Irreversible inherited state cannot witness a transition.
assert before == 0, before
assert isolate_credential_directory(sys.argv[1]) is False
assert libc.prctl(39, 0, 0, 0, 0) == 1
child = subprocess.run([
    sys.executable, "-c",
    "import ctypes; print(ctypes.CDLL(None).prctl(39, 0, 0, 0, 0))",
], capture_output=True, text=True, check=True, timeout=5)
print(json.dumps({"before": before, "after": 1,
                  "exec_child": int(child.stdout)}))
"""
            process = subprocess.run(
                [sys.executable, "-c", code,
                 os.path.join(directory, "missing")],
                cwd=ROOT, capture_output=True, text=True, timeout=10,
            )
            if process.returncode == 77:
                self.skipTest("native transition unavailable: NO_NEW_PRIVS "
                              "was already inherited; stateful seam still runs")
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout),
                             {"before": 0, "after": 1, "exec_child": 1})

    def test_no_new_privileges_failure_is_fatal_without_credentials(self):
        libc = mock.Mock()
        libc.prctl.return_value = -1
        with mock.patch.object(runtime_isolations.ctypes, "CDLL",
                               return_value=libc), \
                mock.patch.object(runtime_isolations.ctypes, "get_errno",
                                  return_value=errno.EPERM), \
                mock.patch.object(runtime_isolations.os.path, "isdir",
                                  return_value=False):
            with self.assertRaisesRegex(
                    runtime_isolations.RuntimeIsolationError,
                    "PR_SET_NO_NEW_PRIVS"):
                runtime_isolations.isolate_credential_directory("/missing")
        self.assertEqual(
            [argument.value for argument in libc.prctl.call_args.args],
            [runtime_isolations._PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0],
        )

    def test_credential_cover_allows_workspace_but_denies_runtime_and_exec_child(self):
        # Windows side:
        # test_windows_appcontainers.AppContainerTests.test_runtime_gate.
        with tempfile.TemporaryDirectory() as directory:
            credentials = os.path.join(directory, "credentials")
            os.mkdir(credentials)
            marker = os.path.join(credentials, "secret")
            with open(marker, "w", encoding="ascii") as stream:
                stream.write("supervisor-visible")
            workspace = os.path.join(directory, "workspace")
            os.mkdir(workspace)
            control = os.path.join(workspace, "allowed")
            with open(control, "w", encoding="ascii") as stream:
                stream.write("workspace-visible")

            code = r"""
import ctypes
import json
import os
import subprocess
import sys

from loki_agent.runtime_isolations import isolate_credential_directory

target, workspace, starting_cwd = sys.argv[1:]
secret = os.path.join(target, "secret")
control = os.path.join(workspace, "allowed")
with open(secret, encoding="ascii") as stream:
    assert stream.read() == "supervisor-visible"
with open(control, encoding="ascii") as stream:
    assert stream.read() == "workspace-visible"
os.chdir(starting_cwd)
isolated = isolate_credential_directory(target)
assert os.getcwd() == starting_cwd
if starting_cwd == target:
    try:
        open("secret").close()
    except (FileNotFoundError, PermissionError):
        pass
    else:
        raise AssertionError("pre-mount cwd still exposes the credential file")
with open(control, encoding="ascii") as stream:
    assert stream.read() == "workspace-visible"
with open(os.path.join(workspace, "runtime-created"), "w") as stream:
    stream.write("allowed write")
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
            "import ctypes, sys\n"
            "assert ctypes.CDLL(None).prctl(39, 0, 0, 0, 0) == 1\n"
            "assert open(sys.argv[2]).read() == 'workspace-visible'\n"
            "try:\n"
            "    stream = open(sys.argv[1], encoding='ascii')\n"
            "except (FileNotFoundError, PermissionError):\n"
            "    print('hidden')\n"
            "else:\n"
            "    stream.close()\n"
            "    print('visible')\n"
        ),
        secret,
        control,
    ],
    timeout=5,
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
                "CapEff", "CapPrm", "CapInh", "CapBnd", "NoNewPrivs"}:
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
            for starting_cwd in [workspace, credentials]:
                with self.subTest(starting_cwd=starting_cwd):
                    process = subprocess.run(
                        [sys.executable, "-c", code, credentials,
                         workspace, starting_cwd],
                        cwd=ROOT, capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(process.returncode, 0, process.stderr)
                    result = json.loads(process.stdout)
                    self.assertTrue(result["isolated"])
                    self.assertTrue(result["hidden"])
                    self.assertTrue(result["marker_hidden"])
                    self.assertTrue(result["tool_hidden"], process.stderr)
                    for key in ["CapEff", "CapPrm", "CapInh", "CapBnd"]:
                        self.assertEqual(result["status"][key], "0" * 16)
                    self.assertEqual(result["status"]["NoNewPrivs"], "1")
                    self.assertEqual(result["unmount_result"], -1)
                    self.assertEqual(result["unmount_errno"], errno.EPERM)

                    # The runtime can change the workspace, not the parent's
                    # credential view. Observe both before fixture cleanup.
                    with open(marker, encoding="ascii") as stream:
                        self.assertEqual(stream.read(), "supervisor-visible")
                    with open(os.path.join(workspace, "runtime-created"),
                              encoding="ascii") as stream:
                        self.assertEqual(stream.read(), "allowed write")


def launch_patches(spawn):
    """Replace whatever starts the runtime, so the supervisor can be observed.

    On POSIX the seam is the thin ``create_subprocess_exec`` delegation, so
    replacing that one call runs the real seam around the stub.  On Windows the
    runtime is a real AppContainer process -- it needs a configured workspace,
    a profile and ACLs -- and the delegation channel is not ported yet, so the
    real launch cannot run here at all.  The whole platform seam is stubbed
    there instead, which keeps these tests on the supervisor's own behaviour
    (which descriptors and environment the runtime is handed, and the
    revoke-before-wait ordering); the gate and the launch themselves are
    covered by ``test_windows_runtime`` and the AppContainer investigation.
    """
    if os.name != "nt":
        return [mock.patch.object(
            credential_supervisors.asyncio, "create_subprocess_exec",
            new=spawn)]
    return [
        mock.patch.object(runtime_isolation, "configured_workspace",
                          return_value=None),
        mock.patch.object(runtime_isolation, "start_runtime", new=spawn),
        mock.patch.object(runtime_isolation, "close_runtime_process",
                          new=lambda process: None),
    ]


class CredentialSupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_subscription_lease_rotate_reopen_and_logout(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = os.path.join(temporary, "credentials")
            storage = credential_storages.JsonCredentialStorage(directory)
            now = time.time()
            tokens = authentications.OpenAITokenSet(
                access_token="access-secret", refresh_token="refresh-secret",
                id_token="identity-secret", account_id="account",
                fedramp=True, expires_at=10**12, last_refresh=now,
            )
            await storage.store_openai_login(tokens)
            self.assertEqual(
                credential_storages.JsonCredentialStorage(directory)
                .load_openai_subscription().tokens, tokens)
            calls = []

            async def refresh(value):
                calls.append(value)
                # The attempt must be durable and secret-free before sending.
                pending = credential_storages.JsonCredentialStorage(directory)
                record = pending.load_openai_subscription()
                self.assertEqual(record.state, "refreshing")
                self.assertIsNone(record.tokens)
                if value == "refresh-secret":
                    return authentications.RefreshResult(
                        access_token="access-new", refresh_token="refresh-new")
                self.assertEqual(value, "refresh-new")
                return authentications.RefreshResult(
                    access_token="access-next", refresh_token="refresh-next")

            rotate_persisted = storage.rotate_openai_subscription

            async def rotate(current):
                return await rotate_persisted(
                    current, refresh=refresh, clock=lambda: now)

            storage.rotate_openai_subscription = rotate
            terminal = credential_supervisors.CredentialSupervisor(
                CredentialStore({}), storage)
            front = acp.Front(lambda: None, lambda message: None,
                              CredentialStore({}), storage)
            credential = authentications.CredentialRef.openai_subscription()
            leases = []
            for broker, inventory in (
                    (terminal.broker, terminal.inventory),
                    (front.credential_broker, front.credentials)):
                self.assertTrue(inventory.has_ref(credential))
                for secret in [tokens.access_token, tokens.refresh_token,
                               tokens.id_token]:
                    self.assertNotIn(secret, repr(terminal.environment))
                    self.assertNotIn(secret, repr(inventory))
                lease = await broker.lease(credential)
                leases.append(lease)
                self.assertEqual(lease.value, tokens.access_token)
                self.assertTrue(lease.refreshable)
                self.assertFalse(hasattr(lease, "refresh_token"))
            self.assertEqual(calls, [])

            # One supervisor refreshes; the other's stale generation must
            # adopt the durable result, not spend the old refresh token twice.
            for broker, old in zip(
                    (terminal.broker, front.credential_broker), leases):
                lease = await broker.lease(
                    credential, rejected_generation=old.generation)
                self.assertEqual(lease.value, "access-new")
                self.assertGreater(lease.generation, old.generation)
                self.assertFalse(hasattr(lease, "refresh_token"))
            self.assertEqual(calls, ["refresh-secret"])
            expected = authentications.OpenAITokenSet(
                access_token="access-new", refresh_token="refresh-new",
                id_token="identity-secret", account_id="account",
                fedramp=True, last_refresh=now,
            )
            reopened = credential_storages.JsonCredentialStorage(directory)
            self.assertEqual(reopened.load_openai_subscription().tokens,
                             expected)
            fresh_terminal = credential_supervisors.CredentialSupervisor(
                CredentialStore({}), reopened)
            fresh_front = acp.Front(
                lambda: None, lambda message: None,
                CredentialStore({}), reopened)
            for broker in [fresh_terminal.broker,
                           fresh_front.credential_broker]:
                self.assertEqual((await broker.lease(credential)).value,
                                 "access-new")
            self.assertEqual(calls, ["refresh-secret"])

            # Also preserve the automatic expiry trigger from the old durable
            # refresh test, not just the rejected-generation route.
            await storage.store_openai_login(authentications.OpenAITokenSet(
                access_token="access-new", refresh_token="refresh-new",
                id_token="identity-secret", account_id="account",
                fedramp=True, expires_at=1, last_refresh=now))
            expired = credential_supervisors.CredentialSupervisor(
                CredentialStore({}), storage)
            self.assertEqual((await expired.broker.lease(credential)).value,
                             "access-next")
            self.assertEqual(calls, ["refresh-secret", "refresh-new"])
            self.assertEqual(reopened.load_openai_subscription().tokens,
                             authentications.OpenAITokenSet(
                                 access_token="access-next",
                                 refresh_token="refresh-next",
                                 id_token="identity-secret",
                                 account_id="account", fedramp=True,
                                 last_refresh=now))

            await reopened.remove_openai_subscription()
            self.assertIsNone(
                credential_storages.JsonCredentialStorage(directory)
                .load_openai_subscription())
            logged_out = credential_supervisors.CredentialSupervisor(
                CredentialStore({}),
                credential_storages.JsonCredentialStorage(directory))
            self.assertFalse(logged_out.inventory.has_ref(credential))
            with self.assertRaises(authentications.CredentialUnavailable):
                await logged_out.broker.lease(credential)

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
        outer_owner = _private_end(outer.owner_child)
        outer_capability = _private_end(outer.credential_child)
        _hand_off(outer)
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
            inner_owner = _private_end(inner.owner_child)
            inner_capability = _private_end(inner.credential_child)
            _hand_off(inner)
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

    async def test_static_environment_credential_is_leased_through_capability(
            self):
        name = "EXAMPLE_API_KEY"
        value = "supervisor-only-secret"
        supervisor = credential_supervisors.CredentialSupervisor(
            CredentialStore({name: value}))
        delegation = await supervisor.delegate()
        owner_fd = _private_end(delegation.owner_child)
        capability_fd = _private_end(delegation.credential_child)
        _hand_off(delegation)
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

        with contextlib.ExitStack() as stack:
            for patch in launch_patches(spawn):
                stack.enter_context(patch)
            status = await supervisor.run_terminal_runtime(
                "/installed/bin/loki", ["--headless"])

        self.assertEqual(status, 7)
        if os.name == "posix":
            # The POSIX seam builds the command line, so its shape is asserted
            # here.  Windows launch command shape is the launch's business.
            self.assertEqual(spawned["args"][0:2], (
                "/installed/bin/loki", "--runtime"))
            self.assertIn("--session-owner-fd", spawned["args"])
            self.assertIn(
                "--credential-capability-fd", spawned["args"])
            self.assertEqual(
                spawned["args"][-2:], ("--", "--headless"))
            self.assertEqual(len(spawned["kwargs"]["pass_fds"]), 2)
            self.assertTrue(spawned["kwargs"]["close_fds"])
            environment = spawned["kwargs"]["env"]
        else:
            # start_runtime(executable, arguments, workspace, environment,
            # delegation): the environment is the property under test.
            environment = spawned["args"][3]
        self.assertNotIn("LOKI_API_KEY", environment)
        self.assertNotIn(
            "must-not-enter-runtime-environment",
            repr(environment),
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

            def child_spawn_kwargs(self):
                return {}

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

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                supervisor, "delegate",
                new=mock.AsyncMock(return_value=Delegation())))
            for patch in launch_patches(spawn):
                stack.enter_context(patch)
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
    async def test_supervisor_releases_process_after_success_error_and_cancellation(self):
        from process_lifecycle_fixtures import process_lifecycle

        for outcome in ['success', 'error', 'cancel']:
            with self.subTest(outcome=outcome):
                async with process_lifecycle('loki') as fixture:
                    if outcome != 'success':
                        fixture.supervisor.environment.update({
                            'LOKI_STREAM': '1',
                            'LOKI_DUMMY_STREAM_CHUNKS': '["alive", "done"]',
                            'LOKI_DUMMY_STREAM_GATE': os.path.join(fixture.workspace, 'release'),
                        })
                    entered = asyncio.Event()
                    start_runtime = runtime_isolation.start_runtime

                    async def launch(*args, **kwargs):
                        process = fixture.record_process(await start_runtime(*args, **kwargs))
                        wait = process.wait
                        first_wait = True

                        async def observed_wait():
                            nonlocal first_wait
                            if first_wait:
                                first_wait = False
                                if outcome != 'success':
                                    # Witness a real child past its containment
                                    # gate and talking to the credential broker.
                                    await fixture.ready.wait()
                                    self.assertIsNone(process.returncode)
                                entered.set()
                                if outcome == 'error':
                                    raise OSError('injected wait failure')
                                if outcome == 'cancel':
                                    # Hold the supervisor at a cancellable wait;
                                    # the real child and its resources stay live.
                                    await asyncio.Future()
                            else:
                                self.assertIsNone(fixture.delegations[0].owner_parent)
                            return await wait()

                        process.wait = observed_wait
                        return process

                    with mock.patch.object(runtime_isolation, 'start_runtime', new=launch):
                        arguments = ['--headless', '--prompt', 'hello',
                                     '--shell-cwd', fixture.workspace]
                        task = asyncio.create_task(fixture.supervisor.run_terminal_runtime(
                            fixture.launcher, arguments))
                        try:
                            await asyncio.wait_for(entered.wait(), 10)
                            if outcome == 'cancel':
                                task.cancel()
                                with self.assertRaises(asyncio.CancelledError):
                                    await asyncio.wait_for(task, 10)
                            elif outcome == 'error':
                                with self.assertRaisesRegex(OSError, 'injected wait failure'):
                                    await asyncio.wait_for(task, 10)
                            else:
                                self.assertEqual(await asyncio.wait_for(task, 10), 0)
                        finally:
                            if not task.done():
                                task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                    await fixture.assert_released(self)

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


@unittest.skipUnless(os.name == "posix",
                     "the POSIX worker spawn shape; Windows launch is covered "
                     "by test_windows_runtime")
class StartWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_spawn_is_piped_delegated_and_sessioned(self):
        spawned = {}

        class Endpoint:
            def __init__(self, *handles):
                self._handles = handles

            def handles(self):
                return self._handles

        class Delegation:
            owner_child = Endpoint(7)
            credential_child = Endpoint(9)

            def child_arguments(self):
                return ["--session-owner-fd", "7",
                        "--credential-capability-fd", "9"]

            def child_spawn_kwargs(self):
                return {"pass_fds": [7, 9]}

        async def spawn(*args, **kwargs):
            spawned["args"] = args
            spawned["kwargs"] = kwargs
            return object()

        with mock.patch.object(
                runtime_isolation, "worker_command",
                return_value=["/installed/bin/loki", "--worker"]), \
                mock.patch.object(asyncio, "create_subprocess_exec",
                                  new=spawn):
            await runtime_isolation.start_worker(
                "/work", {"SAFE": "value"}, Delegation())

        self.assertEqual(spawned["args"], (
            "/installed/bin/loki", "--worker",
            "--session-owner-fd", "7",
            "--credential-capability-fd", "9",
        ))
        kwargs = spawned["kwargs"]
        self.assertIs(kwargs["stdin"], asyncio.subprocess.PIPE)
        self.assertIs(kwargs["stdout"], asyncio.subprocess.PIPE)
        self.assertIsNone(kwargs["stderr"])
        self.assertTrue(kwargs["close_fds"])
        self.assertEqual(kwargs["env"], {"SAFE": "value"})
        self.assertEqual(kwargs["pass_fds"], [7, 9])
        self.assertTrue(kwargs["start_new_session"])


class WindowsWorkerStdioTests(unittest.IsolatedAsyncioTestCase):
    """Portable native-call simulation, including on native Windows runners."""

    async def test_worker_stdio_ends_are_handed_over_and_released(self):
        # The contained worker's stdio: the child's ends are the only ones
        # marked inheritable and the only ones named as its standard handles,
        # the front keeps the complementary ends, and the front's copies of the
        # child's ends do not outlive the launch -- otherwise the worker's exit
        # could never be seen as end of file.  Windows-only in effect,
        # exercised here with the Win32 and launcher calls mocked.
        from loki_agent import windows_runtime
        from loki_agent import windows_subprocesses

        async def exercise(launch_error=None):
            transport = object.__new__(
                windows_subprocesses.ContainedWorkerTransport)
            transport._loop = asyncio.get_running_loop()
            transport._exit_task = None
            transport._closed = True
            contained = mock.Mock(pid=4242, returncode=None)
            contained.wait = mock.AsyncMock(return_value=0)
            closed = []
            error = None
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    windows_subprocesses, "pipe",
                    side_effect=[(11, 12), (13, 14)]))
                inherit = stack.enter_context(mock.patch.object(
                    windows_subprocesses.api, "set_handle_information"))
                stack.enter_context(mock.patch.object(
                    windows_subprocesses.api, "close_handle",
                    side_effect=closed.append))
                null = stack.enter_context(mock.patch.object(
                    windows_runtime, "worker_stdout_null"))
                if launch_error is None:
                    call = stack.enter_context(mock.patch.object(
                        windows_runtime, "launch", return_value=contained))
                else:
                    call = stack.enter_context(mock.patch.object(
                        windows_runtime, "launch", side_effect=launch_error))
                null.return_value.__enter__.return_value = 101
                try:
                    transport._start(
                        args=None, shell=False,
                        stdin=windows_subprocesses.PIPE,
                        stdout=windows_subprocesses.PIPE, stderr=None,
                        bufsize=0, workspace="/work",
                        environment={"SAFE": "value"},
                        arguments=["--worker", "--session-owner-fd", "r=7"],
                        inherited_handles=[7, 9], current_directory="/cwd")
                except BaseException as raised:  # noqa: BLE001 - asserted below
                    error = raised
                finally:
                    if transport._exit_task is not None:
                        transport._exit_task.cancel()
            return transport, inherit, call, closed, error

        transport, inherit, call, closed, error = await exercise()
        self.assertIsNone(error)

        # stdin is (child_read=11, front_write=12); stdout is
        # (front_read=13, child_write=14).
        self.assertEqual(inherit.call_args_list,
                         [mock.call(11, 1, 1), mock.call(14, 1, 1)])
        self.assertEqual(call.call_args.kwargs["stdio"], (11, 14))
        self.assertEqual(call.call_args.args[1],
                         ["--worker", "--session-owner-fd", "r=7",
                          "--stdout-null-handle", "101"])
        self.assertEqual(call.call_args.args[4], [7, 9, 101])
        self.assertEqual(call.call_args.args[3], "/work")
        self.assertEqual(call.call_args.kwargs["current_directory"], "/cwd")
        self.assertCountEqual(closed, [11, 14])
        self.assertEqual(transport._proc.pid, 4242)
        self.assertEqual(transport._proc.stdin.handle, 12)
        self.assertEqual(transport._proc.stdout.handle, 13)
        with mock.patch.object(windows_subprocesses.api, "close_handle"):
            transport._proc.stdin.close()
            transport._proc.stdout.close()

        # A launch that fails hands nothing over and releases every end.
        _transport, _inherit, _call, closed, error = await exercise(
            OSError("launch failed"))
        self.assertIsInstance(error, OSError)
        self.assertCountEqual(closed, [11, 12, 13, 14])


if __name__ == "__main__":
    unittest.main()
