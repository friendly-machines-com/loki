"""Credential supervisor and dependency-free runtime-isolation tests."""

import asyncio
import contextlib
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


class LinuxIsolationTests(unittest.TestCase):
    def test_runtime_rebinds_cwd_through_credential_cover(self):
        with tempfile.TemporaryDirectory() as directory:
            credentials = os.path.join(directory, "credentials")
            os.mkdir(credentials)
            marker = os.path.join(credentials, "secret")
            with open(marker, "w", encoding="ascii") as stream:
                stream.write("supervisor-visible")

            code = r"""
import json
import os
import sys

from loki_agent.runtime_isolations import isolate_credential_directory

target = sys.argv[1]
os.chdir(target)
isolated = isolate_credential_directory(target)
try:
    with open("secret", encoding="ascii") as stream:
        stream.read()
except (FileNotFoundError, PermissionError):
    relative_hidden = True
else:
    relative_hidden = False
print(json.dumps({
    "isolated": isolated,
    "cwd": os.getcwd(),
    "relative_hidden": relative_hidden,
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
            self.assertEqual(result["cwd"], credentials)
            self.assertTrue(result["relative_hidden"])

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
                return {"pass_fds": (7, 9)}

        if os.name == "nt":
            # The same spawn, contained: the session cwd keys the container,
            # the delegation ends cross as handles, and the ambient cwd stays
            # the front's.
            from loki_agent import windows_subprocesses

            async def spawn(**kwargs):
                spawned["kwargs"] = kwargs
                return object()

            with mock.patch.object(
                    runtime_isolation.windows_runtime, "required_workspace",
                    return_value="/recorded/work") as gate, \
                    mock.patch.object(windows_subprocesses,
                                      "create_worker_process", new=spawn):
                await runtime_isolation.start_worker(
                    "/work", {"SAFE": "value"}, Delegation())

            self.assertEqual(gate.call_args.args, ("/work",))
            kwargs = spawned["kwargs"]
            self.assertEqual(kwargs["workspace"], "/recorded/work")
            self.assertEqual(kwargs["environment"], {"SAFE": "value"})
            self.assertEqual(kwargs["arguments"], [
                "--worker", "--session-owner-fd", "7",
                "--credential-capability-fd", "9"])
            self.assertEqual(kwargs["inherited_handles"], [7, 9])
            self.assertEqual(kwargs["current_directory"], os.getcwd())
            return

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
        self.assertEqual(kwargs["pass_fds"], (7, 9))
        self.assertTrue(kwargs["start_new_session"])

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
