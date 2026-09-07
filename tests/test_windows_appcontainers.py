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
imports, elevation requests, or durability claims. The companion
windows_escapes.py probes selected token and ShellExecute/WMI launch routes;
its documented limits remain distinct from complete escape resistance.
"""

from contextlib import ExitStack
import ctypes as C
from ctypes import wintypes as W
import errno
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
escape_helpers = runpy.run_path(str(Path(__file__).with_name('windows_escapes.py')))
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
        self.userenv = C.WinDLL('userenv', use_last_error=True)
        self.create_environment = self.bind(
            self.userenv, 'CreateEnvironmentBlock', W.BOOL,
            C.POINTER(HANDLE), HANDLE, W.BOOL)
        self.destroy_environment = self.bind(
            self.userenv, 'DestroyEnvironmentBlock', W.BOOL, HANDLE)
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

    def profile_paths(self):
        """Restore only account/OS paths, never inherited runner configuration.

        https://learn.microsoft.com/en-us/windows/win32/api/userenv/nf-userenv-createenvironmentblock
        specifies TOKEN_QUERY | TOKEN_DUPLICATE for a primary token, FALSE to
        avoid inheriting the caller's environment, and DestroyEnvironmentBlock
        for cleanup. User-profile variables require a loaded profile; our CI
        supervisor already launches this account with LoadUserProfile=True.
        That contract does not identify the cause of AppContainer error 203.
        """
        allowed = {'USERPROFILE', 'LOCALAPPDATA', 'APPDATA', 'HOMEDRIVE',
                   'HOMEPATH', 'SYSTEMDRIVE', 'PROGRAMDATA', 'PROGRAMFILES',
                   'PROGRAMFILES(X86)', 'PROGRAMW6432', 'COMMONPROGRAMFILES',
                   'COMMONPROGRAMFILES(X86)', 'COMMONPROGRAMW6432'}
        token, block = HANDLE(), HANDLE()
        self.check(self.open_token(self.current_process(), 8 | 2,
                                   C.byref(token)))
        with ExitStack() as cleanup:
            cleanup.callback(lambda: self.check(self.close(token)))
            self.check(self.create_environment(C.byref(block), token, False))
            cleanup.callback(lambda: self.check(self.destroy_environment(block)))
            if not block.value:
                raise OSError('CreateEnvironmentBlock returned a null block')
            selected = {}
            address = block.value
            while True:
                entry = C.wstring_at(address)
                if not entry:
                    break
                name, separator, value = entry.partition('=')
                if separator and name.upper() in allowed and value:
                    selected[name.upper()] = value
                # wchar_t is UTF-16 on Windows: non-BMP characters occupy two
                # code units even when Python len() counts a single character.
                if C.sizeof(C.c_wchar) == 2:
                    address += len(entry.encode('utf-16-le', 'surrogatepass')) + 2
                else:
                    address += (len(entry) + 1) * C.sizeof(C.c_wchar)
            missing = {'USERPROFILE', 'LOCALAPPDATA', 'APPDATA'} - selected.keys()
            if missing:
                raise OSError('loaded profile environment missing: %s' %
                              ', '.join(sorted(missing)))
            return selected

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
                wait = self.wait(process.process, 90000)
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


def access_outcome(name, operation):
    try:
        operation()
    except OSError as error:
        # https://docs.python.org/3/library/exceptions.html#OSError documents
        # both C errno and native winerror; unspecified attributes can be None.
        # Accept only explicit ACCESS_DENIED, or errno-based PermissionError
        # with EACCES when no native code exists. A sharing violation, missing
        # path, invalid parameter, or other failure must not certify denial.
        winerror = getattr(error, 'winerror', None)
        denied = (winerror == 5 or
                  (winerror is None and isinstance(error, PermissionError)
                   and error.errno == errno.EACCES))
        return {'operation': name, 'denied': denied,
                'outcome': 'access-denied' if denied else 'unexpected-error',
                'exception': type(error).__name__, 'errno': error.errno,
                'winerror': winerror, 'message': str(error)}
    return {'operation': name, 'denied': False, 'outcome': 'allowed'}


def attempts(native, manifest):
    """Each attempt operates on a distinct disposable target."""
    secret = Path(manifest['secret'])
    workspace = Path(manifest['workspace'])
    owner = manifest['owner']
    package = manifest['package']
    outcomes = []

    def denied(name, operation):
        result = access_outcome(name, operation)
        outcomes.append(result)
        print(json.dumps(result), flush=True)

    def open_rights(path, rights):
        handle = native.open(path, access=rights)
        native.check(native.close(handle))

    # Positive controls: the same APIs and DACL descriptor must actually work
    # on a permitted object. A universally broken reader/ACL setter is not
    # evidence that only credential access was denied.
    grant = private_dacl(owner) + '(A;OICI;FA;;;%s)' % package

    def workspace_control():
        control = workspace / 'allowed-control'
        control.write_bytes(b'allowed')
        native.acl(control, grant)
        open_rights(control, 0x80000000)  # GENERIC_READ
        open_rights(control, 2)  # FILE_WRITE_DATA
        if control.read_bytes() != b'allowed':
            raise AssertionError('workspace read/write control failed')
        control.unlink()

    control_result = access_outcome('workspace-controls', workspace_control)
    control_result['expected'] = 'allowed'
    outcomes.append(control_result)
    print(json.dumps(control_result), flush=True)

    denied('read', lambda: (secret / 'read').read_bytes())
    denied('truncate', lambda: (secret / 'truncate').write_bytes(b'changed'))
    denied('native-read-right', lambda: open_rights(secret / 'read', 0x80000000))
    denied('native-write-right', lambda: open_rights(secret / 'truncate', 2))
    denied('native-hardlink-read', lambda: open_rights(
        workspace / 'alias', 0x80000000))
    denied('native-junction-read', lambda: open_rights(
        workspace / 'junction' / 'read', 0x80000000))

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
    failed = any(result['outcome'] != result.get('expected', 'access-denied')
                 for result in outcomes)
    if not descendant:
        try:
            api = escape_helpers['Escapes'](
                native, primitives, ExtendedStartups, ProcessInfos)
            failed |= not escape_helpers['probe'](
                api, manifest, __file__, manifest_path, access_outcome)
        except Exception as error:
            print(json.dumps({'operation': 'escape-probe-setup',
                              'outcome': 'unexpected-error',
                              'exception': type(error).__name__,
                              'message': str(error)}), flush=True)
            failed = True
        result = subprocess.run([sys.executable, '-I', '-u', __file__,
                                 '--descendant', manifest_path],
                                capture_output=True, timeout=20, text=True)
        print(result.stdout, flush=True)
        print(result.stderr, file=sys.stderr, flush=True)
        failed |= result.returncode != 0
    return 1 if failed else 0


class EscapeResultTests(unittest.TestCase):
    def test_denial_after_creation_is_not_successful_containment(self):
        result = escape_helpers['finalize_result'](
            {'outcome': 'access-denied', 'denied': True}, 1,
            ('access-denied', 'contained'))
        self.assertEqual(result['outcome'], 'uninspectable-launch')
        self.assertFalse(result['passed'])
        self.assertFalse(result['denied'])
        result = escape_helpers['finalize_result'](
            {'outcome': 'access-denied', 'denied': True}, 0, ('access-denied',))
        self.assertTrue(result['passed'])

    def test_privilege_adjustment_requires_not_assigned_and_disabled_state(self):
        check = escape_helpers['privilege_rejected']
        self.assertTrue(check(True, 1300, False))
        for succeeded, error, enabled in [(True, 0, True), (True, 0, False),
                                          (True, 1300, True), (False, 5, False),
                                          (True, 87, False)]:
            with self.subTest(succeeded=succeeded, error=error, enabled=enabled):
                self.assertFalse(check(succeeded, error, enabled))

    def test_witness_identity_requires_user_and_container(self):
        check = escape_helpers['identity_matches']
        self.assertTrue(check({'user': 'user', 'app': 1, 'package': 'pkg'},
                              'user', 'pkg'))
        for user, app, package in [('other', 1, 'pkg'), ('user', 0, None),
                                   ('user', 1, 'other')]:
            self.assertFalse(check({'user': user, 'app': app, 'package': package},
                                   'user', 'pkg'))

    def test_wmi_wrapper_failures_are_not_wmi_denial(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        api.created = 0
        arguments = ('python.exe', 'probe.py', Path('manifest.json'),
                     Path('report.json'), 'user', 'package')
        with mock.patch.dict(os.environ, {'SystemRoot': 'C:\\Windows'}), \
                mock.patch('builtins.print'):
            for code in (2, 3):
                reply = subprocess.CompletedProcess([], 0, json.dumps(
                    {'kind': 'return', 'code': code, 'pid': 0}), '')
                with mock.patch.object(subprocess, 'run', return_value=reply):
                    self.assertEqual(api.wmi_launch(*arguments)['outcome'],
                                     'access-denied')
            reply = subprocess.CompletedProcess([], 0, json.dumps(
                {'kind': 'return', 'code': 8, 'pid': 0}), '')
            with mock.patch.object(subprocess, 'run', return_value=reply):
                with self.assertRaisesRegex(RuntimeError, 'unexpected WMI'):
                    api.wmi_launch(*arguments)
            with mock.patch.object(subprocess, 'run', side_effect=PermissionError(
                    errno.EACCES, 'cannot launch PowerShell')):
                with self.assertRaisesRegex(RuntimeError, 'wrapper did not complete'):
                    api.wmi_launch(*arguments)

    def test_powershell_timeout_preserves_partial_diagnostics(self):
        run = escape_helpers['run_powershell']
        for stdout, stderr in [(b'partial reply', b'wmi: before-class'),
                               ('partial reply', 'wmi: before-class'),
                               (None, None)]:
            with self.subTest(stdout=stdout), \
                    mock.patch.dict(os.environ, {'SystemRoot': 'C:\\Windows'}), \
                    mock.patch('builtins.print') as printed, \
                    mock.patch.object(subprocess, 'run', side_effect=(
                        subprocess.TimeoutExpired(['encoded-command'], 15,
                                                  output=stdout, stderr=stderr))):
                with self.assertRaisesRegex(RuntimeError, 'timed out after 15'):
                    run('script', 'wmi')
                diagnostic = json.loads(printed.call_args.args[0])
                self.assertTrue(diagnostic['timed_out'])
                self.assertEqual(diagnostic['probe'], 'wmi')
                self.assertEqual(diagnostic['stdout'],
                                 None if stdout is None else 'partial reply')
                self.assertEqual(diagnostic['stderr'],
                                 None if stderr is None else 'wmi: before-class')
                self.assertTrue(printed.call_args.kwargs['flush'])

    def test_powershell_startup_requires_successful_reply(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        for code, output, valid in [(0, 'startup-control: ready\n', True),
                                    (1, 'startup-control: ready\n', False),
                                    (0, '', False), (0, 'unexpected', False)]:
            with self.subTest(code=code, output=output), \
                    mock.patch.dict(os.environ, {'SystemRoot': 'C:\\Windows'}), \
                    mock.patch('builtins.print'), \
                    mock.patch.object(subprocess, 'run', return_value=(
                        subprocess.CompletedProcess([], code, output, ''))) as launch:
                if valid:
                    self.assertEqual(api.powershell_startup(), {'outcome': 'ready'})
                else:
                    with self.assertRaises(RuntimeError):
                        api.powershell_startup()
                self.assertEqual(launch.call_args.kwargs['timeout'], 15)
                self.assertNotIn('env', launch.call_args.kwargs)

    def test_wmi_wrapper_protocol_failures_remain_errors(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        api.created = 0
        arguments = ('python.exe', 'probe.py', Path('manifest.json'),
                     Path('report.json'), 'user', 'package')
        for code, output in [(0, ''), (0, 'not json'),
                             (1, '{"kind":"return","code":2,"pid":0}')]:
            with self.subTest(code=code, output=output), \
                    mock.patch.dict(os.environ, {'SystemRoot': 'C:\\Windows'}), \
                    mock.patch('builtins.print'), \
                    mock.patch.object(subprocess, 'run', return_value=(
                        subprocess.CompletedProcess([], code, output, ''))):
                with self.assertRaises((RuntimeError, ValueError)):
                    api.wmi_launch(*arguments)
                self.assertEqual(api.created, 0)


class AccessOutcomeTests(unittest.TestCase):
    def test_errno_permission_denial_keeps_diagnostics(self):
        error = PermissionError(errno.EACCES, 'permission denied')
        result = access_outcome('read', mock.Mock(side_effect=error))
        self.assertEqual(result['outcome'], 'access-denied')
        self.assertTrue(result['denied'])
        self.assertEqual(result['errno'], errno.EACCES)
        self.assertIsNone(result['winerror'])
        self.assertEqual(result['exception'], 'PermissionError')
        self.assertEqual(result['message'], str(error))

    def test_native_error_takes_precedence_over_errno(self):
        for code, expected in [(5, True), (32, False), (87, False), (0, False)]:
            with self.subTest(winerror=code):
                error = PermissionError(errno.EACCES, 'native result')
                error.winerror = code
                result = access_outcome('native', mock.Mock(side_effect=error))
                self.assertEqual(result['denied'], expected)
                self.assertEqual(result['winerror'], code)

    def test_other_errors_do_not_certify_denial(self):
        for error in (FileNotFoundError(errno.ENOENT, 'missing'),
                      IsADirectoryError(errno.EISDIR, 'directory'),
                      PermissionError(errno.EPERM, 'not the expected EACCES'),
                      OSError(errno.EINVAL, 'invalid'), OSError('unknown')):
            with self.subTest(error=error):
                result = access_outcome('read', mock.Mock(side_effect=error))
                self.assertFalse(result['denied'])
                self.assertEqual(result['outcome'], 'unexpected-error')

    def test_success_is_distinct_from_an_error(self):
        self.assertEqual(access_outcome('read', lambda: b''),
                         {'operation': 'read', 'denied': False,
                          'outcome': 'allowed'})

    def test_programming_errors_propagate(self):
        with self.assertRaises(ValueError):
            access_outcome('read', mock.Mock(side_effect=ValueError('bug')))


class AppContainerMarshallingTests(unittest.TestCase):
    def exercise_profile_paths(self, entries, *, create_fails=False):
        native = AppContainers.__new__(AppContainers)
        native.current_process = lambda: 1
        native.close = mock.Mock(return_value=1)
        native.destroy_environment = mock.Mock(return_value=1)
        buffer = C.create_unicode_buffer('\0'.join(entries) + '\0\0')

        def open_token(process, rights, result):
            self.assertEqual(rights, 8 | 2)
            C.cast(result, C.POINTER(HANDLE)).contents.value = 42
            return 1

        def create_environment(result, token, inherit):
            self.assertEqual(token.value, 42)
            self.assertFalse(inherit)
            if create_fails:
                raise OSError('environment creation failed')
            C.cast(result, C.POINTER(HANDLE)).contents.value = C.addressof(buffer)
            return 1

        native.open_token = open_token
        native.create_environment = create_environment
        try:
            return native.profile_paths()
        finally:
            native.close.assert_called_once()
            self.assertEqual(native.close.call_args.args[0].value, 42)
            if create_fails:
                native.destroy_environment.assert_not_called()
            else:
                native.destroy_environment.assert_called_once()
                self.assertEqual(
                    native.destroy_environment.call_args.args[0].value,
                    C.addressof(buffer))

    def test_profile_paths_exclude_secrets_and_controlled_launch_settings(self):
        result = self.exercise_profile_paths([
            'UserProfile=C:\\Users\\probe-\U0001f600',
            'LocalAppData=C:\\Users\\probe\\AppData\\Local',
            'AppData=C:\\Users\\probe\\AppData\\Roaming',
            'SystemDrive=C:', 'PATH=unwanted-path', 'TEMP=unwanted-temp',
            'TMP=unwanted-temp', 'PYTHONPATH=unwanted-modules',
            'GITHUB_TOKEN=sentinel', 'OTHER_SECRET=sentinel', '=C:=C:\\other'])
        self.assertEqual(result, {
            'USERPROFILE': 'C:\\Users\\probe-\U0001f600',
            'LOCALAPPDATA': 'C:\\Users\\probe\\AppData\\Local',
            'APPDATA': 'C:\\Users\\probe\\AppData\\Roaming', 'SYSTEMDRIVE': 'C:'})

    def test_missing_profile_paths_free_block_and_close_token(self):
        with self.assertRaisesRegex(OSError, 'loaded profile environment missing'):
            self.exercise_profile_paths(['USERPROFILE=C:\\Users\\probe'])

    def test_environment_creation_error_still_closes_token(self):
        with self.assertRaisesRegex(OSError, 'environment creation failed'):
            self.exercise_profile_paths([], create_fails=True)

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
    if len(sys.argv) == 4 and sys.argv[1] == '--launch-witness':
        native = AppContainers()
        api = escape_helpers['Escapes'](
            native, primitives, ExtendedStartups, ProcessInfos)
        escape_helpers['witness'](api, json.loads(Path(sys.argv[2]).read_text()),
                                  Path(sys.argv[3]), access_outcome)
        sys.exit(0)
    if len(sys.argv) == 3 and sys.argv[1] in ('--contained', '--descendant'):
        sys.exit(contained(sys.argv[2], sys.argv[1] == '--descendant'))
    if len(sys.argv) == 3 and sys.argv[1] == '--standard-user':
        native = AppContainers()
        details = native.token_details(include_groups=True)
        primitives['require_standard_user'](details, sys.argv[2])
        print('Standard-user AppContainer broker verified.', flush=True)
        paths = native.profile_paths()
        print(json.dumps({'profile_paths_restored': sorted(paths),
                          'previously_present': sorted(
                              name for name in paths if name in os.environ)}),
              flush=True)
        # Only the single-purpose, verified broker changes its environment.
        # CreateProcessW(NULL environment) then inherits these selected paths,
        # plus the CI supervisor's controlled PATH/TEMP/TMP and OS bootstrap.
        os.environ.update(paths)
        del sys.argv[1:]
        AppContainerTests.__unittest_skip__ = False
    unittest.main(verbosity=2)
