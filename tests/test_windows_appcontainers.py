"""Bounded AppContainer hypotheses, not a production sandbox or certification.

Documentation contracts used by these probes:
- https://www.microsoft.com/en-us/msrc/windows-security-servicing-criteria
  states AppContainer's capability-scoped read/tamper isolation goal.
- https://learn.microsoft.com/en-us/windows/win32/secauthz/implementing-an-appcontainer
  describes dual-principal checks and SECURITY_CAPABILITIES process creation.
- https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-setnamedsecurityinfow
  allows DACL changes by WRITE_DAC holders OR owners. Its precise interaction
  with the AppContainer check remains a documentation gap: re-grant denial is
  a hypothesis tested here, not inferred from a denied read.
- https://learn.microsoft.com/en-us/windows/win32/secauthz/security-information
  defines protected DACLs as not inheriting ACEs.
- https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-deletefilew
  permits deletion through either file DELETE or parent delete-child access.
- https://learn.microsoft.com/en-us/windows/win32/procthread/inheritance
  says inherited handles retain access. Only a diagnostic sink is inherited.
- https://learn.microsoft.com/en-us/windows/win32/procthread/process-security-and-access-rights
  defines VM_READ/WRITE and DUP_HANDLE. Broker handle-open denial is tested.
- https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
  documents CreateProcess child association and KILL_ON_JOB_CLOSE, but also
  WMI/breakaway exceptions. No all-launch-path containment claim is made.

All files are disposable. No network capabilities, real tokens, runtime Loki
imports, elevation requests, COM/WMI broker tests, or durability claims.
"""

from contextlib import ExitStack
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import uuid

# Explicit sibling loading also works under isolated (-I) Python. This imports
# only the existing standalone probe declarations, never the Loki runtime.
primitives = runpy.run_path(str(Path(__file__).with_name(
    'test_windows_primitives.py')))
NativeCalls = primitives['NativeCalls']
HANDLE = C.c_void_p
ULONG = C.c_uint32


class SecurityCapabilities(C.Structure):
    _fields_ = [('sid', HANDLE), ('capabilities', HANDLE),
                ('count', ULONG), ('reserved', ULONG)]


class StartupInfos(C.Structure):
    _fields_ = [('cb', ULONG), ('reserved', W.LPWSTR),
                ('desktop', W.LPWSTR), ('title', W.LPWSTR),
                ('x', ULONG), ('y', ULONG), ('xsize', ULONG), ('ysize', ULONG),
                ('xchars', ULONG), ('ychars', ULONG), ('fill', ULONG),
                ('flags', ULONG), ('show', C.c_uint16),
                ('reserved_size', C.c_uint16), ('reserved_bytes', HANDLE),
                ('stdin', HANDLE), ('stdout', HANDLE), ('stderr', HANDLE)]


class ExtendedStartups(C.Structure):
    _fields_ = [('startup', StartupInfos), ('attributes', HANDLE)]


class ProcessInfos(C.Structure):
    _fields_ = [('process', HANDLE), ('thread', HANDLE),
                ('pid', ULONG), ('tid', ULONG)]


class BasicLimits(C.Structure):
    _fields_ = [('process_time', C.c_int64), ('job_time', C.c_int64),
                ('flags', ULONG), ('minimum_ws', C.c_size_t),
                ('maximum_ws', C.c_size_t), ('active', ULONG),
                ('affinity', C.c_size_t), ('priority', ULONG), ('scheduling', ULONG)]


class ExtendedLimits(C.Structure):
    _fields_ = [('basic', BasicLimits), ('io', C.c_uint64 * 6),
                ('process_memory', C.c_size_t), ('job_memory', C.c_size_t),
                ('peak_process', C.c_size_t), ('peak_job', C.c_size_t)]


class AppContainers(NativeCalls):
    def __init__(self):
        super().__init__()
        self.userenv = C.WinDLL('userenv')
        self.profile = self.bind(self.userenv, 'CreateAppContainerProfile',
                                 C.c_int32, W.LPCWSTR, W.LPCWSTR, W.LPCWSTR,
                                 HANDLE, ULONG, C.POINTER(HANDLE))
        self.delete_profile = self.bind(self.userenv, 'DeleteAppContainerProfile',
                                        C.c_int32, W.LPCWSTR)
        self.free_sid = self.bind(self.advapi, 'FreeSid', HANDLE, HANDLE)
        self.convert_sd = self.bind(
            self.advapi, 'ConvertStringSecurityDescriptorToSecurityDescriptorW',
            W.BOOL, W.LPCWSTR, ULONG, C.POINTER(HANDLE), HANDLE)
        self.get_dacl = self.bind(self.advapi, 'GetSecurityDescriptorDacl',
                                  W.BOOL, HANDLE, C.POINTER(W.BOOL),
                                  C.POINTER(HANDLE), C.POINTER(W.BOOL))
        self.set_security = self.bind(
            self.advapi, 'SetNamedSecurityInfoW', ULONG, W.LPWSTR, C.c_int,
            ULONG, HANDLE, HANDLE, HANDLE, HANDLE)
        self.initialize = self.bind(
            self.kernel, 'InitializeProcThreadAttributeList', W.BOOL, HANDLE,
            ULONG, ULONG, C.POINTER(C.c_size_t))
        self.update = self.bind(self.kernel, 'UpdateProcThreadAttribute',
                                W.BOOL, HANDLE, ULONG, C.c_size_t, HANDLE,
                                C.c_size_t, HANDLE, HANDLE)
        self.delete_attributes = self.bind(
            self.kernel, 'DeleteProcThreadAttributeList', None, HANDLE)
        self.create_process = self.bind(
            self.kernel, 'CreateProcessW', W.BOOL, W.LPCWSTR, W.LPWSTR,
            HANDLE, HANDLE, W.BOOL, ULONG, HANDLE, W.LPCWSTR,
            C.POINTER(ExtendedStartups), C.POINTER(ProcessInfos))
        self.create_job = self.bind(self.kernel, 'CreateJobObjectW', HANDLE,
                                    HANDLE, W.LPCWSTR)
        self.set_job = self.bind(self.kernel, 'SetInformationJobObject', W.BOOL,
                                 HANDLE, C.c_int, HANDLE, ULONG)
        self.assign_job = self.bind(self.kernel, 'AssignProcessToJobObject',
                                    W.BOOL, HANDLE, HANDLE)
        self.resume = self.bind(self.kernel, 'ResumeThread', ULONG, HANDLE)
        self.wait = self.bind(self.kernel, 'WaitForSingleObject', ULONG,
                              HANDLE, ULONG)
        self.exit_code = self.bind(self.kernel, 'GetExitCodeProcess', W.BOOL,
                                   HANDLE, C.POINTER(ULONG))
        self.terminate = self.bind(self.kernel, 'TerminateProcess', W.BOOL,
                                   HANDLE, ULONG)
        self.open_process = self.bind(self.kernel, 'OpenProcess', HANDLE,
                                      ULONG, W.BOOL, ULONG)

    @staticmethod
    def hresult(result):
        if result < 0:
            raise OSError('HRESULT 0x%08x' % (result & 0xffffffff))

    def acl(self, path, sddl):
        descriptor, acl = HANDLE(), HANDLE()
        self.check(self.convert_sd(sddl, 1, C.byref(descriptor), None))
        try:
            present, defaulted = W.BOOL(), W.BOOL()
            self.check(self.get_dacl(descriptor, C.byref(present),
                                     C.byref(acl), C.byref(defaulted)))
            if not present.value or not acl.value:
                raise ValueError('probe requires an explicit non-null DACL')
            # DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION.
            error = self.set_security(str(path), 1, 4 | 0x80000000,
                                      None, None, acl, None)
            if error:
                raise C.WinError(error)
        finally:
            self.local_free(descriptor)

    def app_identity(self, process=None):
        token = HANDLE()
        if process is None:
            process = self.current_process()
        self.check(self.open_token(process, 8, C.byref(token)))
        try:
            size, is_app = ULONG(), ULONG()
            self.check(self.token_info(token, 29, C.byref(is_app), 4,
                                       C.byref(size)))
            # TOKEN_APPCONTAINER_INFORMATION's SID needs storage in addition
            # to its pointer header; do not pass a pointer-sized output buffer.
            result = self.token_info(token, 31, None, 0, C.byref(size))
            if result or C.get_last_error() != 122 or not size.value:
                raise OSError('unexpected AppContainer SID sizing result')
            buffer = C.create_string_buffer(size.value)
            self.check(self.token_info(token, 31, buffer, len(buffer),
                                       C.byref(size)))
            package = HANDLE.from_buffer(buffer)
            return is_app.value, self.sid(package) if package.value else None
        finally:
            self.check(self.close(token))

    def launch(self, command, sid, workspace, output):
        import msvcrt
        size = C.c_size_t()
        self.initialize(None, 2, 0, C.byref(size))
        if C.get_last_error() != 122 or not size.value:
            raise C.WinError(C.get_last_error())
        attributes = C.create_string_buffer(size.value)
        self.check(self.initialize(attributes, 2, 0, C.byref(size)))
        process = ProcessInfos()
        job = None
        try:
            capabilities = SecurityCapabilities(sid, None, 0, 0)
            self.check(self.update(attributes, 0, 0x20009,
                                   C.byref(capabilities), C.sizeof(capabilities),
                                   None, None))
            job = self.check(self.create_job(None, None))
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            self.check(self.set_job(job, 9, C.byref(limits), C.sizeof(limits)))
            with open(output, 'wb', buffering=0) as log, open(os.devnull, 'rb') as null:
                handles = (HANDLE * 2)(msvcrt.get_osfhandle(log.fileno()),
                                       msvcrt.get_osfhandle(null.fileno()))
                os.set_handle_inheritable(handles[0], True)
                os.set_handle_inheritable(handles[1], True)
                try:
                    self.check(self.update(attributes, 0, 0x20002, handles,
                                           C.sizeof(handles), None, None))
                    startup = ExtendedStartups()
                    startup.startup.cb = C.sizeof(startup)
                    startup.startup.flags = 0x100  # STARTF_USESTDHANDLES
                    startup.startup.stdin = handles[1]
                    startup.startup.stdout = handles[0]
                    startup.startup.stderr = handles[0]
                    startup.attributes = C.cast(attributes, HANDLE)
                    text = C.create_unicode_buffer(subprocess.list2cmdline(command))
                    # Start suspended: no target code executes before assignment
                    # to the kill-on-close job. There is no unsandboxed fallback.
                    self.check(self.create_process(
                        command[0], text, None, None, True, 0x80000 | 4,
                        None, str(workspace), C.byref(startup), C.byref(process)))
                finally:
                    os.set_handle_inheritable(handles[0], False)
                    os.set_handle_inheritable(handles[1], False)
                identity = self.app_identity(process.process)
                if identity != (1, self.sid(sid)):
                    raise RuntimeError('wrong suspended child identity: %r' %
                                       (identity,))
                print(json.dumps({'broker_verified_child_identity': identity}),
                      flush=True)
                self.check(self.assign_job(job, process.process))
                if self.resume(process.thread) == 0xffffffff:
                    raise C.WinError(C.get_last_error())
                wait = self.wait(process.process, 45000)
                if wait != 0:
                    raise TimeoutError('AppContainer wait returned 0x%x' % wait)
                code = ULONG()
                self.check(self.exit_code(process.process, C.byref(code)))
                return code.value
        finally:
            try:
                if process.process:
                    if self.wait(process.process, 0) == 258:
                        self.check(self.terminate(process.process, 1))
                        if self.wait(process.process, 10000) != 0:
                            raise TimeoutError('AppContainer process did not exit')
            finally:
                # Closing the job kills associated descendants even if the root
                # has already exited. This is not proof against broker escapes.
                with ExitStack() as cleanup:
                    cleanup.callback(self.delete_attributes, attributes)
                    for handle in (process.process, process.thread, job):
                        if handle:
                            cleanup.callback(
                                lambda handle=handle: self.check(self.close(handle)))


def private_dacl(owner):
    return 'D:P(A;OICI;FA;;;%s)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)' % owner


def attempts(native, manifest):
    """Each attempt operates on a distinct disposable target."""
    secret = Path(manifest['secret'])
    workspace = Path(manifest['workspace'])
    owner = manifest['owner']
    package = manifest['package']
    outcomes = []

    def denied(name, operation):
        try:
            operation()
        except OSError as error:
            outcomes.append({'operation': name, 'winerror': error.winerror,
                             'denied': error.winerror == 5})
        else:
            outcomes.append({'operation': name, 'denied': False})

    denied('read', lambda: (secret / 'read').read_bytes())
    denied('truncate', lambda: (secret / 'truncate').write_bytes(b'changed'))

    def open_rights(path, rights):
        handle = native.open(path, access=rights)
        native.check(native.close(handle))

    denied('append-right', lambda: open_rights(secret / 'read', 4))
    denied('write-dacl-right', lambda: open_rights(secret / 'regrant', 0x40000))
    denied('write-owner-right', lambda: open_rights(secret / 'regrant', 0x80000))
    denied('trusted-script-write', lambda: open_rights(__file__, 0x40000000))
    denied('delete', lambda: (secret / 'delete').unlink())
    replacement = workspace / 'replacement'
    replacement.write_bytes(b'changed')
    denied('replace', lambda: os.replace(replacement, secret / 'replace'))
    denied('hardlink-read', lambda: (workspace / 'alias').read_bytes())
    denied('junction-read', lambda: (workspace / 'junction' / 'read').read_bytes())
    grant = private_dacl(owner) + '(A;OICI;FA;;;%s)' % package
    denied('regrant-file-dacl', lambda: native.acl(secret / 'regrant', grant))
    denied('regrant-directory-dacl', lambda: native.acl(secret, grant))
    denied('rename-directory', lambda: os.rename(secret, workspace / 'stolen'))

    for name, rights in [('broker-memory-read', 0x10),
                         ('broker-memory-write', 0x20 | 8),
                         ('broker-handle-duplication', 0x40),
                         ('broker-dacl-write', 0x40000)]:
        def open_broker(rights=rights):
            handle = native.check(native.open_process(rights, False,
                                                      manifest['broker_pid']))
            native.check(native.close(handle))
        denied(name, open_broker)
    return outcomes


def contained(manifest_path, descendant=False):
    native = AppContainers()
    manifest = json.loads(Path(manifest_path).read_text())
    details = native.token_details(include_groups=True)
    primitives['require_standard_user'](details, manifest['owner'])
    identity = native.app_identity()
    if identity != (1, manifest['package']):
        raise RuntimeError('unexpected AppContainer identity: %r' % (identity,))
    print(json.dumps({'appcontainer_identity': identity, 'user': details['user'],
                      'descendant': descendant}), flush=True)
    outcomes = attempts(native, manifest)
    for result in outcomes:
        print(json.dumps(result), flush=True)
    failed = any(not result['denied'] for result in outcomes)
    if not descendant:
        result = subprocess.run([sys.executable, '-I', '-u', __file__,
                                 '--descendant', manifest_path],
                                capture_output=True, timeout=20, text=True)
        print(result.stdout, flush=True)
        print(result.stderr, file=sys.stderr, flush=True)
        failed |= result.returncode != 0
    return 1 if failed else 0


class AppContainerMarshallingTests(unittest.TestCase):
    def test_package_sid_uses_sized_buffer_and_closes_token(self):
        native = AppContainers.__new__(AppContainers)
        native.current_process = lambda: 1
        native.close = mock.Mock(return_value=1)
        native.sid = lambda pointer: 'SID-%d' % pointer.value
        queries = []

        def open_token(process, access, pointer):
            C.cast(pointer, C.POINTER(HANDLE)).contents.value = 42
            return 1

        def query(token, kind, buffer, length, returned):
            queries.append((kind, length))
            size = C.cast(returned, C.POINTER(ULONG)).contents
            if kind == 29:
                C.cast(buffer, C.POINTER(ULONG)).contents.value = 1
                size.value = 4
                return 1
            size.value = 80  # deliberately larger than a pointer
            if buffer is None:
                return 0
            self.assertGreaterEqual(length, 80)
            HANDLE.from_buffer(buffer).value = 123
            return 1

        native.open_token = open_token
        native.token_info = query
        with mock.patch.object(C, 'get_last_error', return_value=122,
                               create=True):
            self.assertEqual(native.app_identity(), (1, 'SID-123'))
        self.assertEqual(queries, [(29, 4), (31, 0), (31, 80)])
        native.close.assert_called_once()
        self.assertEqual(native.close.call_args.args[0].value, 42)

    def test_failed_identity_query_closes_token(self):
        native = AppContainers.__new__(AppContainers)
        native.current_process = lambda: 1
        native.open_token = mock.Mock(return_value=1)
        native.close = mock.Mock(return_value=1)
        native.token_info = mock.Mock(side_effect=OSError('query failed'))
        with self.assertRaisesRegex(OSError, 'query failed'):
            native.app_identity()
        native.close.assert_called_once()


@unittest.skip('requires explicit --standard-user invocation in disposable CI stage')
class AppContainerTests(unittest.TestCase):
    def test_same_user_containment(self):
        native = AppContainers()
        details = native.token_details(include_groups=True)
        owner = details['user']
        primitives['require_standard_user'](details, owner)
        name = 'Loki.Probes.' + uuid.uuid4().hex
        sid = HANDLE()
        native.hresult(native.profile(name, name, name, None, 0, C.byref(sid)))
        # LIFO: directories/processes are cleaned before deleting the profile.
        self.addCleanup(lambda: native.hresult(native.delete_profile(name)))
        self.addCleanup(lambda: native.free_sid(sid))
        package = native.sid(sid)
        print(json.dumps({'profile': name, 'package': package}), flush=True)
        # Supervisor explicitly grants control of these disposable CI copies to
        # this standard user. No administrator-owned installation ACL is edited.
        stage = Path(__file__).parent
        if (not stage.name.startswith('LokiStorageProbes-') or
                Path(sys.base_prefix) != stage / 'runtime'):
            raise RuntimeError('refusing ACL changes outside disposable CI copies')
        result = subprocess.run(
            ['icacls.exe', str(stage), '/grant', '*%s:(OI)(CI)RX' % package,
             '/T', '/Q'], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        # Remove inherited package grants before making the private targets.
        native.acl(root, private_dacl(owner) +
                   '(A;;0x20;;;%s)' % package)  # traverse only, not inherited
        secret = root / 'credentials'
        secret.mkdir()
        names = ('read', 'truncate', 'delete', 'replace', 'regrant')
        for leaf in names:
            target = secret / leaf
            target.write_bytes(b'FAKE-CREDENTIAL')
            handle = native.open(target, access=0x20000)  # READ_CONTROL
            try:
                self.assertEqual(native.owner(handle), owner,
                                 'fixture must exercise same-user ownership')
            finally:
                native.check(native.close(handle))
        workspace = root / 'workspace'
        workspace.mkdir()
        native.acl(workspace, private_dacl(owner) +
                   '(A;OICI;FA;;;%s)' % package)
        os.link(secret / 'read', workspace / 'alias')
        junction = workspace / 'junction'
        result = subprocess.run(
            ['cmd.exe', '/d', '/c', 'mklink', '/J',
             str(junction).replace('/', '\\'), str(secret).replace('/', '\\')],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.addCleanup(lambda: os.rmdir(junction))
        manifest = workspace / 'manifest.json'
        manifest.write_text(json.dumps({'secret': str(secret),
                                        'workspace': str(workspace),
                                        'owner': owner, 'package': package,
                                        'broker_pid': os.getpid()}))
        log = root / 'contained.log'
        try:
            code = native.launch([sys.executable, '-I', '-u', __file__,
                                  '--contained', str(manifest)],
                                 sid, workspace, log)
        finally:
            if log.exists():
                print(log.read_text(errors='replace'), flush=True)
        # Check integrity independently of the child-reported operation results.
        for leaf in names:
            self.assertEqual((secret / leaf).read_bytes(), b'FAKE-CREDENTIAL')
        self.assertEqual(code, 0, 'AppContainer probe failed; see native log')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] in ('--contained', '--descendant'):
        sys.exit(contained(sys.argv[2], sys.argv[1] == '--descendant'))
    if len(sys.argv) == 3 and sys.argv[1] == '--standard-user':
        details = AppContainers().token_details(include_groups=True)
        primitives['require_standard_user'](details, sys.argv[2])
        print('Standard-user AppContainer broker verified.', flush=True)
        del sys.argv[1:]
        AppContainerTests.__unittest_skip__ = False
    unittest.main(verbosity=2)
