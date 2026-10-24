"""Credential supervisor and dependency-free runtime-isolation tests."""

import asyncio
import contextlib
import errno
import json
import os
import stat
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

    def test_missing_credentials_are_fatal_from_known_privilege_states(self):
        for initial in [0, 1]:
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as directory:
                state = initial

                def prctl(*arguments):
                    nonlocal state
                    self.assertEqual(
                        [argument.value for argument in arguments], [38, 1, 0, 0, 0])
                    state = 1
                    return 0

                libc = mock.Mock(prctl=mock.Mock(side_effect=prctl))
                with mock.patch.object(runtime_isolations.ctypes, 'CDLL', return_value=libc):
                    with self.assertRaisesRegex(
                            runtime_isolations.RuntimeIsolationError, 'credential protection'):
                        runtime_isolations.isolate_credential_directory(None)
                    self.assertEqual(state, 1)
                    not_directory = os.path.join(directory, 'file')
                    with open(not_directory, 'w'):
                        pass
                    fd = os.open(not_directory, os.O_RDONLY)
                    with self.assertRaisesRegex(
                            runtime_isolations.RuntimeIsolationError, 'not a directory'):
                        runtime_isolations.isolate_credential_directory(fd)
                    with self.assertRaises(OSError):
                        os.fstat(fd)

    def test_no_new_privileges_failure_is_fatal_without_credentials(self):
        libc = mock.Mock()
        libc.prctl.return_value = -1
        with mock.patch.object(runtime_isolations.ctypes, "CDLL",
                               return_value=libc), \
                mock.patch.object(runtime_isolations.ctypes, "get_errno",
                                  return_value=errno.EPERM):
            with self.assertRaisesRegex(
                    runtime_isolations.RuntimeIsolationError,
                    "PR_SET_NO_NEW_PRIVS"):
                runtime_isolations.isolate_credential_directory(None)
        self.assertEqual(
            [argument.value for argument in libc.prctl.call_args.args],
            [runtime_isolations._PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0],
        )

    @contextlib.contextmanager
    def directory_startup(self, failure=None):
        original = types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_dev=1, st_ino=2)
        target = types.SimpleNamespace(st_mode=original.st_mode, st_dev=1,
                                       st_ino=3 if failure == 'identity' else 2)
        closed = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(runtime_isolations.ctypes, 'CDLL'))
            stack.enter_context(mock.patch.object(runtime_isolations, '_set_no_new_privileges'))
            unshare = stack.enter_context(mock.patch.object(
                runtime_isolations, '_unshare_user_and_mount_namespaces',
                side_effect=runtime_isolations.RuntimeIsolationError('namespace') if failure == 'namespace' else None))
            stack.enter_context(mock.patch.object(runtime_isolations, '_map_current_identity'))
            stack.enter_context(mock.patch.object(runtime_isolations.os, 'getcwd', return_value='/trusted/source'))

            def kernel_name(path):
                unshare.assert_not_called()
                self.assertEqual(path, '/proc/self/fd/80')
                if failure == 'name':
                    raise FileNotFoundError()
                return '/moved/credentials'

            stack.enter_context(mock.patch.object(runtime_isolations.os, 'readlink', side_effect=kernel_name))
            stack.enter_context(mock.patch.object(runtime_isolations.os, 'fstat', side_effect=[original, target]))

            def reopen(*args):
                unshare.assert_called_once()
                self.assertEqual(args[0], '/moved/credentials')
                if failure == 'reopen':
                    raise FileNotFoundError()
                return 81

            stack.enter_context(mock.patch.object(runtime_isolations.os, 'open', side_effect=reopen))
            mount = stack.enter_context(mock.patch.object(
                runtime_isolations, '_mount_private_credential_view',
                side_effect=runtime_isolations.RuntimeIsolationError('mount') if failure == 'mount' else None))
            stack.enter_context(mock.patch.object(
                runtime_isolations.os, 'chdir',
                side_effect=OSError('cwd') if failure == 'cwd' else None))
            stack.enter_context(mock.patch.object(
                runtime_isolations, '_drop_namespace_capabilities',
                side_effect=runtime_isolations.RuntimeIsolationError('capabilities') if failure == 'capabilities' else None))
            stack.enter_context(mock.patch.object(runtime_isolations.os, 'close', side_effect=closed.append))
            yield [mount, closed]

    def test_mount_target_is_bound_to_the_checked_directory_handle(self):
        with self.directory_startup() as [mount, closed]:
            self.assertTrue(runtime_isolations.isolate_credential_directory(80))
            self.assertEqual(mount.call_args.args[1], '/proc/self/fd/81')
            self.assertCountEqual(closed, [80, 81])

    def test_startup_directory_handles_close_on_every_failure(self):
        for failure in ['name', 'namespace', 'reopen', 'identity', 'mount', 'cwd', 'capabilities']:
            with self.subTest(failure=failure), self.directory_startup(failure) as [mount, closed]:
                with self.assertRaises(runtime_isolations.RuntimeIsolationError):
                    runtime_isolations.isolate_credential_directory(80)
                self.assertCountEqual(closed, [80] if failure in ['name', 'namespace', 'reopen'] else [80, 81])
                if failure in ['name', 'namespace', 'reopen', 'identity']:
                    mount.assert_not_called()

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
directory_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
pinned_directory = os.fstat(directory_fd)
isolated = isolate_credential_directory(directory_fd)
try:
    os.fstat(directory_fd)
except OSError:
    pass
else:
    raise AssertionError("startup directory descriptor remains open")
with os.scandir('/proc/self/fd') as entries:
    for entry in entries:
        try:
            held = os.fstat(int(entry.name))
        except OSError:
            continue
        assert (held.st_dev != pinned_directory.st_dev or
                held.st_ino != pinned_directory.st_ino), 'underlying directory reference remains open'
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

    def test_native_reopen_identity_bind_ancestry_and_cwd_failure(self):
        # Real syscalls select both objects. Only the interval at which this
        # disposable fixture renames a name is controlled; stat/open/mount are
        # not substituted. A bind-path test is not coverage of every alias.
        code = r'''
import ctypes,json,os,sys
from loki_agent import paths
from loki_agent import runtime_isolations as isolation
mode,base,view,moved,workspace,inherited = sys.argv[1:]
fd = int(inherited)
pinned = os.fstat(fd)
libc = ctypes.CDLL(None, use_errno=True)
target = paths.credential_directory({'XDG_CONFIG_HOME': base})
if mode.startswith('bind-'):
    isolation._set_no_new_privileges(libc)
    uid,gid = os.getuid(),os.getgid()
    isolation._unshare_user_and_mount_namespaces(libc)
    isolation._map_current_identity(uid,gid)
    mount = libc.mount
    mount.argtypes = [ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,
                      ctypes.c_ulong,ctypes.c_void_p]
    mount.restype = ctypes.c_int
    for source,destination,flags in [
            [None,b'/',isolation._MS_REC | isolation._MS_PRIVATE],
            [os.fsencode(base),os.fsencode(view),4096]]:
        if mount(source,destination,None,flags,None) != 0:
            raise OSError(ctypes.get_errno(), 'fixture bind mount failed')
    target = paths.credential_directory({'XDG_CONFIG_HOME': view})
    selected = os.open(target,os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    assert os.path.samestat(pinned,os.fstat(selected))
    os.close(fd)
    fd = selected
with os.scandir(target) as entries:
    secret = next(entry.path for entry in entries if entry.name == paths.CREDENTIAL_FILE_NAME)
with open(secret) as stream:
    assert 'native-retained-access' in stream.read()
original_unshare = isolation._unshare_user_and_mount_namespaces
def unshare(native):
    original_unshare(native)
    if mode in ['replacement','renamed-back']:
        os.rename(target,moved)
        if mode == 'replacement':
            os.mkdir(target,0o700)
        else:
            os.rename(moved,target)
isolation._unshare_user_and_mount_namespaces = unshare
original_mount = isolation._mount_private_credential_view
def cover(native,name):
    original_mount(native,name)
    if mode == 'cwd-removed':
        os.rmdir(workspace)
isolation._mount_private_credential_view = cover
os.chdir(target if mode == 'bind-credentials' else workspace)
refused = False
try:
    assert isolation.isolate_credential_directory(fd)
except isolation.RuntimeIsolationError as error:
    if mode == 'replacement':
        assert 'credential location changed' in str(error),repr(error)
    elif mode == 'cwd-removed':
        assert isinstance(error.__cause__,FileNotFoundError),repr(error)
        assert 'working directory' in str(error),repr(error)
    else:
        raise
    refused = True
assert refused == (mode in ['replacement','cwd-removed'])
with os.scandir('/proc/self/fd') as entries:
    for entry in entries:
        try:
            held = os.fstat(int(entry.name))
        except OSError:
            continue
        assert not os.path.samestat(pinned,held),'startup directory reference survived'
if not refused:
    try:
        with open(secret) as stream:
            stream.read()
    except (PermissionError,FileNotFoundError):
        pass
    else:
        raise AssertionError('selected credential view is readable')
    if mode == 'bind-credentials':
        try:
            os.open(paths.CREDENTIAL_FILE_NAME,os.O_RDONLY)
        except (PermissionError,FileNotFoundError):
            pass
        else:
            raise AssertionError('cwd retained the pre-cover directory')
print(json.dumps({'mode':mode,'refused':refused,'startup_references_closed':True}))
'''
        for mode in ['replacement', 'renamed-back', 'bind-workspace',
                     'bind-credentials', 'cwd-removed']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as base, \
                    tempfile.TemporaryDirectory() as view, \
                    tempfile.TemporaryDirectory() as moved, \
                    tempfile.TemporaryDirectory() as workspace:
                storage = credential_storages.JsonCredentialStorage(
                    paths.credential_directory({'XDG_CONFIG_HOME': base}))
                try:
                    tokens = authentications.OpenAITokenSet(
                        access_token='native-retained-access', refresh_token='native-retained-refresh',
                        expires_at=10**12, last_refresh=100)
                    asyncio.run(storage.store_openai_login(tokens))
                    os.rmdir(moved)
                    process = subprocess.run(
                        [sys.executable, '-c', code, mode, base, view, moved, workspace,
                         str(storage.directory_handle)], pass_fds=[storage.directory_handle],
                        cwd=ROOT, capture_output=True, text=True)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    self.assertTrue(json.loads(process.stdout)['startup_references_closed'])
                    self.assertEqual(storage.load_openai_subscription().tokens, tokens.normalized())
                    # The failed or completed child must not consume the
                    # parent's retained store or redirect its next publication.
                    asyncio.run(storage.remove_openai_subscription())
                    self.assertIsNone(storage.load_openai_subscription())
                finally:
                    storage.close()

    def test_isolated_peer_cannot_recover_another_startup_directory(self):
        target_code = r'''
import ctypes,json,os,sys
from loki_agent import runtime_isolations as isolation
fd,stage = int(sys.argv[1]),sys.argv[2]
libc = ctypes.CDLL(None,use_errno=True)
if stage == 'after-unshare':
    isolation._set_no_new_privileges(libc)
    uid,gid = os.getuid(),os.getgid()
    isolation._unshare_user_and_mount_namespaces(libc)
    isolation._map_current_identity(uid,gid)
print(json.dumps({'pid':os.getpid(),'dumpable':libc.prctl(3,0,0,0,0)}),flush=True)
sys.stdin.read(1)
'''
        observer_code = r'''
import json,os,sys
from loki_agent import paths,runtime_isolations as isolation
root,pid,reference = sys.argv[1:]
def read_at(directory):
    fd = os.open(directory,os.O_RDONLY | os.O_DIRECTORY)
    try:
        token = os.open(paths.CREDENTIAL_FILE_NAME,os.O_RDONLY,dir_fd=fd)
        with os.fdopen(token) as stream:
            return stream.read()
    finally:
        os.close(fd)
def attempt(directory):
    try:
        read_at(directory)
    except OSError as error:
        return error.errno
    raise AssertionError('isolated observer recovered synthetic credentials')
isolation.isolate_credential_directory(os.open(root,os.O_RDONLY | os.O_DIRECTORY))
# This is the actual kernel reference route, not a rewritten configured path.
foreign = f'/proc/{pid}/fd/{reference}'
print(json.dumps([attempt(root),attempt(foreign)]),flush=True)
'''
        parent_code = r'''
import json,os,subprocess,sys
from loki_agent import paths,process_protections
fd,root = int(sys.argv[1]),sys.argv[2]
target_code,observer_code = sys.argv[3:]
process_protections.protect_credential_process()
os.set_inheritable(fd,False)
try:
    for stage in ['before-unshare','after-unshare']:
        target = subprocess.Popen(
            [sys.executable,'-c',target_code,str(fd),stage],pass_fds=[fd],close_fds=True,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            line = target.stdout.readline()
            if not line:
                raise AssertionError('startup witness failed: '+target.stderr.read())
            ready = json.loads(line)
            assert ready['dumpable'] == 1
            foreign = f'/proc/{ready["pid"]}/fd/{fd}'
            directory = os.open(foreign,os.O_RDONLY | os.O_DIRECTORY)
            try:
                token = os.open(paths.CREDENTIAL_FILE_NAME,os.O_RDONLY,dir_fd=directory)
                with os.fdopen(token) as stream:
                    assert 'proc-reference-access' in stream.read()
            finally:
                os.close(directory)
            observer = subprocess.run(
                [sys.executable,'-c',observer_code,root,str(ready['pid']),str(fd)],
                capture_output=True,text=True,close_fds=True)
            assert observer.returncode == 0,observer.stderr
            errors = json.loads(observer.stdout)
            assert all(error in [1,13] for error in errors),errors
        finally:
            if target.poll() is None:
                target.stdin.write('x')
                target.stdin.flush()
            output,errors = target.communicate()
            assert target.returncode == 0,errors
finally:
    os.close(fd)
print('matched startup-reference controls passed')
'''
        with tempfile.TemporaryDirectory() as directory:
            storage = credential_storages.JsonCredentialStorage(directory)
            try:
                tokens = authentications.OpenAITokenSet(
                    access_token='proc-reference-access', refresh_token='proc-reference-refresh',
                    expires_at=10**12, last_refresh=100)
                asyncio.run(storage.store_openai_login(tokens))
                process = subprocess.run(
                    [sys.executable, '-c', parent_code, str(storage.directory_handle),
                     directory, target_code, observer_code],
                    pass_fds=[storage.directory_handle], cwd=ROOT,
                    capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn('matched startup-reference controls passed', process.stdout)
                self.assertEqual(storage.load_openai_subscription().tokens, tokens.normalized())
            finally:
                storage.close()


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
                lease = await broker.lease(credential, destination=authentications.OPENAI_CHATGPT_RESPONSES_URL)
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
                    credential, rejected_generation=old.generation, destination=authentications.OPENAI_CHATGPT_RESPONSES_URL)
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
                self.assertEqual((await broker.lease(credential, destination=authentications.OPENAI_CHATGPT_RESPONSES_URL)).value,
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
            self.assertEqual((await expired.broker.lease(credential, destination=authentications.OPENAI_CHATGPT_RESPONSES_URL)).value,
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
                await logged_out.broker.lease(credential, destination=authentications.OPENAI_CHATGPT_RESPONSES_URL)

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
                credential, destination=authentications.OPENAI_CHATGPT_RESPONSES_URL)

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
        destination = "https://example.test/data"
        await supervisor.broker.approve_destinations(
            authentications.CredentialRef.environment(name), [destination])
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
            lease = await session.credential_authority.lease(credential, destination=destination)
            self.assertEqual(lease.value, value)
        finally:
            if runtime is not None:
                await runtime.close()
            await delegation.close()

    async def test_terminal_runtime_receives_channels_and_startup_directory(self):
        store = CredentialStore({
            "LOKI_API_KEY": "must-not-enter-runtime-environment",
            "LOKI_MODEL": "model",
        })
        directory = self.enterContext(tempfile.TemporaryDirectory())
        storage = credential_storages.JsonCredentialStorage(directory)
        self.addCleanup(storage.close)
        supervisor = credential_supervisors.CredentialSupervisor(store, storage)
        spawned = {}

        class Process:
            returncode = 0

            async def wait(self):
                return 7

        async def spawn(*args, **kwargs):
            spawned["args"] = args
            spawned["kwargs"] = kwargs
            if os.name == 'posix':
                directory_fd = int(args[args.index('--credential-directory-fd') + 1])
                spawned['directory_fd'] = directory_fd
                self.assertGreaterEqual(directory_fd, 3)
                self.assertNotEqual(directory_fd, storage.directory_handle)
                self.assertTrue(os.path.samestat(
                    os.fstat(directory_fd), os.fstat(storage.directory_handle)))
                self.assertFalse(os.get_inheritable(directory_fd))
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
            self.assertIn("--credential-directory-fd", spawned["args"])
            self.assertIn(spawned['directory_fd'], spawned["kwargs"]["pass_fds"])
            self.assertEqual(len(spawned["kwargs"]["pass_fds"]), 3)
            with self.assertRaises(OSError):
                os.fstat(spawned['directory_fd'])
            self.assertIsNone(storage.load_openai_subscription())
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
        directory = self.enterContext(tempfile.TemporaryDirectory())
        storage = credential_storages.JsonCredentialStorage(directory)
        self.addCleanup(storage.close)
        supervisor = credential_supervisors.CredentialSupervisor(store, storage)
        revoked = asyncio.Event()
        closed = asyncio.Event()
        waiting = asyncio.Event()

        class Delegation:
            def child_arguments(self):
                return []

            def child_spawn_kwargs(self, credential_directory=None):
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
                waiting.set()
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
            await waiting.wait()
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
        directory = self.enterContext(tempfile.TemporaryDirectory())
        storage = credential_storages.JsonCredentialStorage(directory)
        self.addCleanup(storage.close)
        delegation = await credential_supervisors.RuntimeDelegation.create(
            authentications.CredentialBroker())
        self.addAsyncCleanup(delegation.close)
        spawned = {}

        async def spawn(*args, **kwargs):
            spawned['args'] = args
            spawned['kwargs'] = kwargs
            fd = int(args[args.index('--credential-directory-fd') + 1])
            spawned['directory_fd'] = fd
            self.assertGreaterEqual(fd, 3)
            self.assertNotEqual(fd, storage.directory_handle)
            self.assertTrue(os.path.samestat(
                os.fstat(fd), os.fstat(storage.directory_handle)))
            self.assertFalse(os.get_inheritable(fd))
            return object()

        with mock.patch.object(
                runtime_isolation, 'worker_command',
                return_value=['/installed/bin/loki', '--worker']), \
                mock.patch.object(asyncio, 'create_subprocess_exec', new=spawn):
            await runtime_isolation.start_worker(
                '/work', {'SAFE': 'value'}, delegation,
                credential_directory=storage.directory_handle)

        self.assertEqual(list(spawned['args']), [
            '/installed/bin/loki', '--worker', *delegation.child_arguments(),
            '--credential-directory-fd', str(spawned['directory_fd'])])
        kwargs = spawned['kwargs']
        self.assertIs(kwargs['stdin'], asyncio.subprocess.PIPE)
        self.assertIs(kwargs['stdout'], asyncio.subprocess.PIPE)
        self.assertIsNone(kwargs['stderr'])
        self.assertTrue(kwargs['close_fds'])
        self.assertEqual(kwargs['env'], {'SAFE': 'value'})
        self.assertEqual(kwargs['pass_fds'], [
            delegation.owner_child, delegation.credential_child, spawned['directory_fd']])
        self.assertTrue(kwargs['start_new_session'])
        with self.assertRaises(OSError):
            os.fstat(spawned['directory_fd'])
        reused = os.dup2(storage.directory_handle, spawned['directory_fd'], inheritable=False)
        try:
            delegation.child_spawned()
            await delegation.close()
            self.assertTrue(os.path.samestat(
                os.fstat(reused), os.fstat(storage.directory_handle)))
            self.assertIsNone(storage.load_openai_subscription())
        finally:
            os.close(reused)

    async def test_directory_copy_lives_until_creation_settles_and_closes_on_each_outcome(self):
        for kind in ['terminal', 'worker']:
            for outcome in ['success', 'failure', 'cancel']:
                with self.subTest(kind=kind, outcome=outcome), \
                        tempfile.TemporaryDirectory() as directory:
                    storage = credential_storages.JsonCredentialStorage(directory)
                    delegation = await credential_supervisors.RuntimeDelegation.create(
                        authentications.CredentialBroker())
                    entered, release = asyncio.Event(), asyncio.Event()
                    inherited = []

                    async def spawn(*args, **kwargs):
                        fd = int(args[args.index('--credential-directory-fd') + 1])
                        inherited.append(fd)
                        self.assertIn(fd, kwargs['pass_fds'])
                        entered.set()
                        await release.wait()
                        if outcome == 'failure':
                            raise OSError('creation refused')
                        return object()

                    try:
                        with mock.patch.object(asyncio, 'create_subprocess_exec', new=spawn), \
                                mock.patch.object(runtime_isolation, 'worker_command',
                                                  return_value=['/selected/loki-acp', '--worker']):
                            if kind == 'terminal':
                                operation = runtime_isolation.start_runtime(
                                    '/selected/loki', [], None, {}, delegation,
                                    credential_directory=storage.directory_handle)
                            else:
                                operation = runtime_isolation.start_worker(
                                    directory, {}, delegation,
                                    credential_directory=storage.directory_handle)
                            task = asyncio.create_task(operation)
                            try:
                                await entered.wait()
                                self.assertTrue(os.path.samestat(
                                    os.fstat(inherited[0]), os.fstat(storage.directory_handle)))
                                if outcome == 'cancel':
                                    task.cancel()
                                    with self.assertRaises(asyncio.CancelledError):
                                        await task
                                else:
                                    release.set()
                                    if outcome == 'failure':
                                        with self.assertRaisesRegex(OSError, 'creation refused'):
                                            await task
                                    else:
                                        await task
                            finally:
                                release.set()
                                if not task.done() and not task.cancelling():
                                    task.cancel()
                                await asyncio.gather(task, return_exceptions=True)
                        with self.assertRaises(OSError):
                            os.fstat(inherited[0])
                        self.assertIsNone(storage.load_openai_subscription())
                    finally:
                        await delegation.close()
                        storage.close()

    @unittest.skipUnless(sys.platform.startswith('linux'), 'real Linux runtime creation')
    async def test_cancelled_native_creation_reaps_unpublished_child_before_releasing_copy(self):
        from asyncio.base_subprocess import BaseSubprocessTransport
        from process_lifecycle_fixtures import process_lifecycle
        from loki_agent import host_ipc

        for kind, repeat in [['terminal', False], ['worker', False],
                             ['terminal', True], ['worker', True]]:
            with self.subTest(kind=kind, repeated_cancellation=repeat):
                async with process_lifecycle('loki' if kind == 'terminal' else 'loki-acp') as fixture:
                    entered, release = asyncio.Event(), asyncio.Event()
                    transports, copies = [], []
                    connect = BaseSubprocessTransport._connect_pipes
                    duplicate = host_ipc.duplicate_for_child
                    storage = fixture.supervisor.storage
                    original = os.fstat(storage.directory_handle)

                    def copy(fd):
                        result = duplicate(fd)
                        if os.path.samestat(original, os.fstat(fd)):
                            copies.append(result)
                        return result

                    async def connect_pipes(transport, waiter):
                        # The real OS child already exists, but the public
                        # creation coroutine has not published its Process.
                        transports.append(transport)
                        entered.set()
                        await release.wait()
                        await connect(transport, waiter)

                    front = None
                    with mock.patch.object(BaseSubprocessTransport, '_connect_pipes', new=connect_pipes), \
                            mock.patch.object(host_ipc, 'duplicate_for_child', side_effect=copy):
                        if kind == 'terminal':
                            operation = fixture.supervisor.run_terminal_runtime(
                                fixture.launcher, ['--headless', '--prompt', 'creation check'])
                        else:
                            front = acp.Front(
                                lambda: None, lambda message: None,
                                CredentialStore(fixture.supervisor.environment), storage)
                            operation = front._open_worker(
                                cwd=fixture.workspace, open_method='session/new')
                        task = asyncio.create_task(operation)
                        try:
                            await entered.wait()
                            self.assertEqual(len(copies), 1)
                            self.assertIsNone(transports[0].get_returncode())
                            task.cancel()
                            await asyncio.sleep(0)
                            if repeat:
                                task.cancel()
                                await asyncio.sleep(0)
                            self.assertFalse(task.done())
                            self.assertTrue(os.path.samestat(original, os.fstat(copies[0])))
                            release.set()
                            with self.assertRaises(asyncio.CancelledError):
                                await task
                        finally:
                            release.set()
                            if not task.done() and not task.cancelling():
                                task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    self.assertTrue(transports[0].is_closing())
                    self.assertIsNotNone(transports[0].get_returncode())
                    with self.assertRaises(ChildProcessError):
                        os.waitpid(transports[0].get_pid(), os.WNOHANG)
                    if front is not None:
                        self.assertFalse(front.workers)
                        self.assertFalse(front._sessions)
                    held = []
                    with os.scandir('/proc/self/fd') as entries:
                        for entry in entries:
                            try:
                                candidate = os.fstat(int(entry.name))
                            except OSError:
                                continue
                            if os.path.samestat(original, candidate):
                                held.append(entry.name)
                    # Only storage owns the original object now; a recycled
                    # numeric fd is not mistaken for a surviving launch copy.
                    self.assertEqual(len(held), 1)
                    self.assertIsNone(storage.load_openai_subscription())

    @unittest.skipUnless(sys.platform.startswith('linux'), 'real Linux runtime creation')
    async def test_cancelled_completed_creation_retires_the_unpublished_result(self):
        from process_lifecycle_fixtures import process_lifecycle

        for kind in ['terminal', 'worker']:
            with self.subTest(kind=kind):
                async with process_lifecycle('loki' if kind == 'terminal' else 'loki-acp') as fixture:
                    if kind == 'terminal':
                        # This absent workspace file holds the real dummy turn
                        # live; a leaf alone would refer to the runner's cwd.
                        fixture.supervisor.environment.update({
                            'LOKI_STREAM': '1',
                            'LOKI_DUMMY_STREAM_CHUNKS': '["alive", "done"]',
                            'LOKI_DUMMY_STREAM_GATE': os.path.join(fixture.workspace, 'release'),
                        })
                    create = asyncio.create_subprocess_exec
                    created = []

                    async def spawn(*args, **kwargs):
                        process = await create(*args, **kwargs)
                        created.append(fixture.record_process(process))
                        self.assertIsNone(process.returncode)
                        # Creation returns a real Process just as cancellation
                        # reaches its caller: no result has been published yet.
                        task.cancel()
                        return process

                    front = None
                    with mock.patch.object(asyncio, 'create_subprocess_exec', new=spawn):
                        if kind == 'terminal':
                            operation = fixture.supervisor.run_terminal_runtime(
                                fixture.launcher, ['--headless', '--prompt', 'creation check'])
                        else:
                            front = acp.Front(
                                lambda: None, lambda message: None,
                                CredentialStore(fixture.supervisor.environment), fixture.supervisor.storage)
                            operation = front._open_worker(
                                cwd=fixture.workspace, open_method='session/new')
                        task = asyncio.create_task(operation)
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    self.assertEqual(len(created), 1)
                    await fixture.assert_released(self)
                    if front is not None:
                        self.assertFalse(front.workers)
                        self.assertFalse(front._sessions)
                    self.assertIsNone(fixture.supervisor.storage.load_openai_subscription())

    async def test_copy_acquisition_failure_leaves_storage_and_delegation_usable(self):
        from loki_agent import host_ipc

        directory = self.enterContext(tempfile.TemporaryDirectory())
        storage = credential_storages.JsonCredentialStorage(directory)
        self.addCleanup(storage.close)
        delegation = await credential_supervisors.RuntimeDelegation.create(
            authentications.CredentialBroker())
        self.addAsyncCleanup(delegation.close)
        for kind in ['terminal', 'worker']:
            with self.subTest(kind=kind), mock.patch.object(
                    host_ipc, 'duplicate_for_child', side_effect=OSError('copy refused')), \
                    mock.patch.object(asyncio, 'create_subprocess_exec') as spawn:
                with self.assertRaisesRegex(OSError, 'copy refused'):
                    if kind == 'terminal':
                        await runtime_isolation.start_runtime(
                            '/selected/loki', [], None, {}, delegation,
                            credential_directory=storage.directory_handle)
                    else:
                        await runtime_isolation.start_worker(
                            directory, {}, delegation,
                            credential_directory=storage.directory_handle)
                spawn.assert_not_called()
            self.assertIsNone(storage.load_openai_subscription())
            os.fstat(delegation.owner_child)
            os.fstat(delegation.credential_child)


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
