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
import time
import unittest
from unittest import mock
import uuid

# Explicit sibling loading also works under isolated (-I) Python. This imports
# only the existing standalone probe declarations, never the Loki runtime.
primitives = runpy.run_path(str(Path(__file__).with_name(
    'test_windows_primitives.py')))
escape_helpers = runpy.run_path(str(Path(__file__).with_name('windows_escapes.py')))
NativeCalls = primitives['NativeCalls']
Guids = escape_helpers['Guids']
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
        self.in_job = self.bind(self.kernel, 'IsProcessInJob', W.BOOL,
                                HANDLE, HANDLE, C.POINTER(W.BOOL))
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

    def launch(self, command, sid, workspace, output, observe=None,
               deadline=None):
        """Run one contained child under one monotonic budget.

        The deadline is fixed before the observer runs--an explicit one, or
        now + 90 s when none is given--and the final wait spends only what
        remains, so observation time can never extend the total budget for
        any caller."""
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
                if observe is not None:
                    # Fix the budget before observation starts in every
                    # caller, explicit or not.
                    deadline = _scenario_deadline(deadline)
                    observe(process.process, job)
                wait = self.wait(process.process,
                                 _remaining_milliseconds(deadline))
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


def cleanup_tree(directory, depth):
    """Disposable tree, with no inherited observer/job handles or service IPC.

    Child stdio uses files under the package-writable workspace: the contained
    witness cannot open the NUL device (native errno 13 at cea20fa), so the
    devnull constant is not an option inside the sandbox.
    """
    if depth < 2:
        child_stdin = open(directory / ('stdio-%d.in' % depth), 'wb')
        child_stdin.close()
        with ExitStack() as files:
            stdin = files.enter_context(
                open(directory / ('stdio-%d.in' % depth), 'rb'))
            stdout = files.enter_context(
                open(directory / ('stdio-%d.out' % depth), 'wb'))
            subprocess.Popen([sys.executable, '-I', '-u', __file__,
                              '--cleanup-tree', str(directory), str(depth + 1)],
                             close_fds=True, stdin=stdin, stdout=stdout,
                             stderr=subprocess.STDOUT)
    report = directory / ('ready-%d' % depth)
    temporary = report.with_suffix('.tmp')
    temporary.write_text(str(os.getpid()))
    os.replace(temporary, report)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if depth == 0 and (directory / 'release').exists():
            return
        time.sleep(0.02)
    raise TimeoutError('cleanup witness exceeded its safety deadline')


def broker_death(directory, package, report):
    """Disposable sub-broker owning a job, killed without running any code.

    The parent observes the tree through handles it opened itself, so the
    dying broker's kill-on-close job is exercised with no finally path, no
    graceful termination, and no administrative sweep involved."""
    native = AppContainers()
    api = escape_helpers['Escapes'](native, primitives, ExtendedStartups, ProcessInfos)
    sid = HANDLE()
    native.check(api.parse_sid(package, C.byref(sid)))
    try:
        def observe(root_process, job):
            deadline = time.monotonic() + 20
            reports = [directory / ('ready-%d' % i) for i in range(3)]
            while not all(path.exists() for path in reports):
                if native.wait(root_process, 0) != 258:
                    raise RuntimeError('broker-death root exited before readiness')
                if time.monotonic() >= deadline:
                    raise TimeoutError('broker-death tree readiness timed out')
                time.sleep(0.02)
            payload = {'broker_pid': os.getpid(),
                       'tree_pids': [int(path.read_text()) for path in reports]}
            temporary = report.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload))
            os.replace(temporary, report)
            # Stay alive holding the job until the parent kills this broker.
            deadline = time.monotonic() + 60
            while not (directory / 'release').exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('broker-death witness exceeded deadline')
                time.sleep(0.05)
        native.launch([sys.executable, '-I', '-u', __file__, '--cleanup-tree',
                       str(directory), '0'], sid, directory,
                      directory / 'broker-death.log', observe=observe)
    finally:
        native.free_sid(sid)


def race_tree(directory, depth):
    """Spawn bounded short-lived children continuously until stopped.

    Every child appends its own pid to the shared track file and then sleeps,
    so the parent can enumerate the exact set of processes that existed while
    shutdown raced creation. Child stdio uses workspace files (the contained
    witness cannot open the NUL device)."""
    report = directory / ('ready-%d' % depth)
    temporary = report.with_suffix('.tmp')
    temporary.write_text(str(os.getpid()))
    os.replace(temporary, report)
    track = directory / 'race-pids'
    marker = ('import os, sys, time;'
              ' stream = open(sys.argv[1], "a");'
              ' stream.write(str(os.getpid()) + chr(10)); stream.close();'
              ' time.sleep(30)')
    for index in range(40):
        with ExitStack() as files:
            stdin = files.enter_context(
                open(directory / ('race-stdio-%d.in' % index), 'wb'))
            stdin.close()
            stdin = files.enter_context(
                open(directory / ('race-stdio-%d.in' % index), 'rb'))
            stdout = files.enter_context(
                open(directory / ('race-stdio-%d.out' % index), 'wb'))
            subprocess.Popen([sys.executable, '-I', '-c', marker, str(track)],
                             close_fds=True, stdin=stdin, stdout=stdout,
                             stderr=subprocess.STDOUT)
        time.sleep(0.05)
    deadline = time.monotonic() + 20
    while not (directory / 'stop').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('race witness exceeded its safety deadline')
        time.sleep(0.02)


def peer_readiness_failure(peer, log):
    wait_error = None
    try:
        code = peer.wait(timeout=5)
        detail = 'exit code %s' % code
    except (OSError, subprocess.SubprocessError) as error:
        wait_error = error
        detail = '%s: %s' % (type(error).__name__, error)
    log.flush()
    log.seek(0)
    output = log.read().decode(errors='replace')
    raise RuntimeError('unrestricted peer did not become ready; wait: %s; log: %s'
                       % (detail, output)) from wait_error


def task_witness(request_path):
    """Read the supervisor-registered task's request: manifest and report.

    The task action is fixed at registration (S4U self-registration by the
    standard-user broker is denied on this Server image; native run
    1670996), so the broker selects the witness target via this file.
    """
    request = json.loads(request_path.read_text())
    return json.loads(Path(request['manifest']).read_text()), Path(request['report'])


def verify_task_route(report_path, owner, native=None, api=None):
    """The control must prove the route live, dangerous, and honestly reported.

    With kernel access provided, the witness process is independently
    inspected: its token must match both the owner and its own report, so a
    forged or mismatched self-report fails the control. The credential read
    must be 'allowed'; any other outcome (denial or error) means the route is
    not proven dangerous and the control fails.
    """
    payload = json.loads(report_path.read_text())
    if payload['token']['user'] != owner or payload['token']['app'] != 0:
        raise RuntimeError('control witness did not run unrestricted: %r'
                           % payload['token'])
    if payload['credential_access'].get('outcome') != 'allowed':
        raise RuntimeError('control witness could not read credentials: %r'
                           % payload['credential_access'])
    if native is not None and api is not None:
        handle = native.check(native.open_process(
            0x1000 | 0x100000 | 1, False, payload['pid']))
        try:
            token = api.token(handle, 8)
            try:
                snapshot = api.snapshot(token)
            finally:
                api.close(token)
        finally:
            api.close(handle)
        if snapshot != payload['token']:
            raise RuntimeError('kernel token disagrees with witness report: '
                               '%r vs %r' % (snapshot, payload['token']))
    return payload


def register_user_com_server(clsid, command):
    """Per-user local COM server registration (arm 1): no admin, no
    capabilities, disposable, broker-owned hive."""
    import winreg
    path = r'Software\Classes\CLSID\%s\LocalServer32' % clsid
    key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, path, 0,
                             winreg.KEY_SET_VALUE)
    try:
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, command)
    finally:
        key.Close()


def unregister_user_com_server(clsid):
    import winreg
    for path in (r'Software\Classes\CLSID\%s\LocalServer32' % clsid,
                 r'Software\Classes\CLSID\%s' % clsid):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)
        except FileNotFoundError:
            # A blocked registration never wrote this key; anything else is a
            # real cleanup failure and must stay visible.
            pass


def witness_verdict(kernel, read_outcome, owner, package):
    """Classify a kernel-inspected worker witness.

    Only a same-user, same-package, AppContainer witness with an explicitly
    denied credential read is 'contained'. An unrestricted or otherwise
    anomalous witness is a boundary violation regardless of the read result;
    a non-definitive read outcome is inconclusive. Callers fail on anything
    but 'contained'."""
    contained = (kernel['user'] == owner and kernel['app'] == 1 and
                 kernel.get('package') == package)
    if not contained:
        if kernel['user'] != owner or kernel['app'] == 0:
            return 'violation-unrestricted-witness'
        return 'violation-anomalous-witness'
    if read_outcome == 'allowed':
        return 'violation-credentials-read'
    if read_outcome != 'access-denied':
        return 'inconclusive-read-outcome'
    return 'contained'


def reap_activation_wrapper(process):
    """Kill and reap a wrapper; its pipes close even when reaping raises.

    Every step is attempted regardless of earlier failures, the first
    failure is preserved and raised, and both pipe streams are closed before
    returning or propagating.
    """
    failure = None
    try:
        if process.poll() is None:
            process.kill()
    except Exception as error:
        failure = error
    try:
        process.communicate(timeout=5)
    except Exception as error:
        if failure is None:
            failure = error
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError as error:
                if failure is None:
                    failure = error
    if failure is not None:
        raise failure


def activate_and_observe(command, report_path, owner, native, api):
    """Run one activation while inspecting its witness live.

    Polling continues for the full bounded deadline even after the wrapper
    exits: the OS launches the witness asynchronously, so a report can still
    appear once the activation call itself has failed or hung. Cleanup is
    never silent: on the normal path a failed reap raises, and while a
    primary error propagates the cleanup failure is printed so the primary
    error still wins.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    payload = None
    output = error = ''
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if report_path.exists():
                payload = verify_task_route(report_path, owner, native, api)
                break
            time.sleep(0.05)
        try:
            output, error = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            output, error = process.communicate(timeout=5)
    except BaseException:
        # The primary error wins; a failing cleanup is reported, not hidden.
        try:
            reap_activation_wrapper(process)
        except Exception as cleanup:
            print(json.dumps({'probe': 'com-control',
                              'cleanup_error': repr(cleanup)}), flush=True)
        raise
    reap_activation_wrapper(process)
    return payload, {'exit': process.returncode, 'stdout': output,
                     'stderr': error}


def _scenario_deadline(deadline, now=None):
    """The single budget start: an explicit deadline, or now + 90 seconds."""
    if deadline is not None:
        return deadline
    if now is None:
        now = time.monotonic()
    return now + 90


def _remaining_milliseconds(deadline, now=None):
    """Wait budget for a monotonic deadline; the classic 90 s when None."""
    if deadline is None:
        return 90000
    if now is None:
        now = time.monotonic()
    return max(0, int((deadline - now) * 1000))


def watch_worker_witnesses(remaining_reports, deadline, root_alive, inspect):
    """Observe armed witness reports until all appear or bounds are hit.

    One active bound governs everything: the scenario deadline while the
    root runs, shortened to a two-second grace once the root exits, with no
    fallback to the longer scenario bound afterwards. Every report read and
    every inspection starts only while that bound is in the future, so an
    earlier inspection that consumes time can never push a later one past
    it. Returns {name: (payload, kernel)} for every report observed;
    unobserved names simply stay out of the result.
    """
    observed = {}

    def sweep(active_deadline):
        # Re-check before EACH report: a slow step earlier in this sweep must
        # not license another past the bound.
        for name, report_path in list(remaining_reports.items()):
            if time.monotonic() >= active_deadline:
                return
            if not report_path.exists():
                continue
            if time.monotonic() >= active_deadline:
                # exists() consumed the budget; never start the read past
                # the bound.
                return
            payload = json.loads(report_path.read_text())
            if time.monotonic() >= active_deadline:
                # The read consumed the budget; never inspect past it.
                return
            observed[name] = (payload, inspect(payload))
            del remaining_reports[name]

    while remaining_reports:
        alive = root_alive()
        active = deadline if alive else min(deadline, time.monotonic() + 2)
        if time.monotonic() >= active:
            break
        sweep(active)
        if not remaining_reports:
            break
        if alive:
            time.sleep(0.05)
            continue
        while remaining_reports and time.monotonic() < active:
            sweep(active)
            time.sleep(0.05)
        break
    return observed


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

    def test_peer_access_denial_stops_before_secondary_operation(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        api.n = mock.Mock()
        api.n.check.side_effect = PermissionError(errno.EACCES, 'open denied')
        api.read_memory = mock.Mock()
        api.duplicate_handle = mock.Mock()
        target = {'pid': 42, 'address': 123, 'file_handle': 456}
        with mock.patch('builtins.print'):
            for operation in (api.peer_memory, api.peer_handle):
                with self.assertRaises(PermissionError):
                    operation(target)
        api.read_memory.assert_not_called()
        api.duplicate_handle.assert_not_called()
        api.n.close.assert_not_called()

    def test_peer_secondary_denial_does_not_hide_process_access(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        api.n = mock.Mock()
        api.n.check.side_effect = lambda value: value
        api.n.open_process.return_value = 77
        api.n.close.return_value = True
        api.read_memory = mock.Mock(return_value=False)
        api.duplicate_handle = mock.Mock(return_value=False)
        target = {'pid': 42, 'address': 123, 'file_handle': 456}
        with mock.patch('builtins.print'), \
                mock.patch.object(C, 'get_last_error', return_value=5, create=True):
            for operation in (api.peer_memory, api.peer_handle):
                result = operation(target)
                self.assertEqual(result['outcome'], 'process-access-granted')
                self.assertEqual(result['winerror'], 5)
                self.assertFalse(escape_helpers['finalize_result'](
                    result, 0, ('access-denied',))['passed'])
        self.assertEqual(api.n.close.call_args_list, [mock.call(77), mock.call(77)])

    def test_duplicate_peer_handle_is_closed_on_success(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        api.n = mock.Mock()
        api.n.check.side_effect = lambda value: value
        api.n.open_process.return_value = 77
        api.n.close.return_value = True

        def duplicate(source, handle, destination, result, rights, inherit, options):
            C.cast(result, C.POINTER(HANDLE)).contents.value = 88
            self.assertFalse(inherit)
            self.assertEqual(options, 2)
            return True
        api.duplicate_handle = duplicate
        with mock.patch('builtins.print'):
            result = api.peer_handle({'pid': 42, 'file_handle': 456})
        self.assertTrue(result['duplicated'])
        closed = [getattr(call.args[0], 'value', call.args[0])
                  for call in api.n.close.call_args_list]
        self.assertEqual(closed, [88, 77])

    def test_escape_close_failure_is_not_access_denial(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        api.n = mock.Mock()
        api.n.close.return_value = False
        with mock.patch.object(C, 'get_last_error', return_value=5, create=True):
            with self.assertRaisesRegex(RuntimeError, 'CloseHandle failed'):
                api.close(77)

    def impersonation_free_api(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        api.created = 0
        api.startup_type = ExtendedStartups
        api.process_type = ProcessInfos
        api.n = mock.Mock()
        api.n.check.side_effect = lambda value: value
        api.n.current_process.return_value = 1
        api.n.close.return_value = True
        api.token = mock.Mock(return_value=7)

        def duplicate(source, rights, attributes, level, kind, out):
            C.cast(out, C.POINTER(HANDLE)).contents.value = 9
            return 1
        api.duplicate = duplicate
        return api

    def test_token_launch_denial_is_reported_at_create_stage(self):
        api = self.impersonation_free_api()
        creator = mock.Mock(return_value=0)
        for error, outcome in [(1314, 'access-denied'), (5, 'access-denied'),
                               (87, 'parameter-rejected-inconclusive'),
                               (6, 'unexpected-launch-result')]:
            with self.subTest(winerror=error), \
                    mock.patch('builtins.print'), \
                    mock.patch.object(C, 'get_last_error', return_value=error,
                                      create=True):
                result = api.token_launch(creator, 0xB, 'python.exe', 'user',
                                          'pkg')
            self.assertEqual(result['phase'], 'create')
            self.assertEqual(result['winerror'], error)
            self.assertEqual(result['outcome'], outcome)
            self.assertEqual(api.created, 0)
            passed = escape_helpers['finalize_result'](
                result, 0, ('access-denied',))['passed']
            self.assertEqual(passed, outcome == 'access-denied')
        self.assertEqual(creator.call_count, 4)

    def test_cleanup_tree_children_use_workspace_files(self):
        # Contained witnesses cannot open 'nul' (native errno 13), so the
        # functional assertions below require real workspace files for child
        # stdio; the devnull constant would fail them (it is an int sentinel).
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            clock = iter([0, 10, 200])  # readiness wait expires immediately
            with mock.patch.object(time, 'monotonic',
                                   side_effect=lambda: next(clock)), \
                    mock.patch.object(time, 'sleep'), \
                    mock.patch.object(subprocess, 'Popen') as popen:
                with self.assertRaises(TimeoutError):
                    cleanup_tree(directory, 0)
            self.assertEqual(popen.call_count, 1)
            kwargs = popen.call_args.kwargs
            self.assertTrue(kwargs['close_fds'])
            self.assertEqual(Path(kwargs['stdin'].name).name, 'stdio-0.in')
            self.assertEqual(Path(kwargs['stdout'].name).name, 'stdio-0.out')
            self.assertIs(kwargs['stderr'], subprocess.STDOUT)

    def test_token_api_binding_signatures(self):
        native = mock.Mock()
        bindings = {}

        def bind(library, name, result, *arguments):
            bindings[name] = (result, arguments)
            return mock.Mock()
        native.bind = bind
        with mock.patch.object(C, 'WinDLL', create=True):
            escape_helpers['Escapes'](native, primitives, ExtendedStartups, ProcessInfos)
        startup, process = C.POINTER(ExtendedStartups), C.POINTER(ProcessInfos)
        self.assertEqual(bindings['CreateProcessWithTokenW'],
                         (W.BOOL, (HANDLE, ULONG, W.LPCWSTR, W.LPWSTR, ULONG,
                                   HANDLE, W.LPCWSTR, startup, process)))
        self.assertEqual(bindings['CreateProcessAsUserW'],
                         (W.BOOL, (HANDLE, W.LPCWSTR, W.LPWSTR, HANDLE, HANDLE,
                                   W.BOOL, ULONG, HANDLE, W.LPCWSTR, startup, process)))
        self.assertEqual(bindings['GetCurrentThread'], (HANDLE, ()))
        self.assertEqual(bindings['OpenThreadToken'],
                         (W.BOOL, (HANDLE, ULONG, W.BOOL, C.POINTER(HANDLE))))

    def test_token_creators_use_documented_call_shapes(self):
        api = self.impersonation_free_api()
        api.create_as_user = mock.Mock(return_value=0)
        api.create_with_token = mock.Mock(return_value=0)
        startup, child = ExtendedStartups(), ProcessInfos()
        with mock.patch('builtins.print'):
            api.launch_as_user(9, 'exe', 'cmd', startup, child)
            api.launch_with_token(9, 'exe', 'cmd', startup, child)
        user_args = api.create_as_user.call_args.args
        self.assertEqual(len(user_args), 11)
        self.assertEqual(user_args[:3], (9, 'exe', 'cmd'))
        self.assertIsNone(user_args[3])
        self.assertIsNone(user_args[4])
        self.assertIs(user_args[5], False)
        self.assertEqual(user_args[6], 0x80000 | 4)
        token_args = api.create_with_token.call_args.args
        self.assertEqual(len(token_args), 9)
        self.assertEqual(token_args[:4], (9, 0, 'exe', 'cmd'))
        self.assertEqual(token_args[4], 0x80000 | 4)

    def test_token_launch_success_requires_inspected_containment(self):
        api = self.impersonation_free_api()
        api.snapshot = mock.Mock(return_value={'user': 'user', 'app': 1,
                                               'package': 'pkg'})
        api.retire = mock.Mock()
        creator = mock.Mock(return_value=1)
        with mock.patch('builtins.print'):
            result = api.token_launch(creator, 0xB, 'python.exe', 'user', 'pkg')
        self.assertEqual(result['outcome'], 'contained')
        self.assertEqual(api.created, 1)
        api.retire.assert_called_once()
        # Source and primary duplicate handles are both released.
        closed = [getattr(call.args[0], 'value', call.args[0])
                  for call in api.n.close.call_args_list]
        self.assertIn(7, closed)
        self.assertIn(9, closed)

    def impersonation_child_api(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        api.n = mock.Mock()
        api.n.check.side_effect = lambda value: value
        api.n.current_process.return_value = 1
        api.n.close.return_value = True
        api.token = mock.Mock(return_value=7)

        def duplicate(source, rights, attributes, level, kind, out):
            C.cast(out, C.POINTER(HANDLE)).contents.value = 9
            return 1
        api.duplicate = duplicate
        api.snapshot = mock.Mock(return_value={'user': 'u', 'app': 1,
                                               'package': 'p'})
        api.current_thread = mock.Mock(return_value=11)
        api.open_thread_token = mock.Mock(return_value=False)
        api.revert = mock.Mock(return_value=True)
        return api

    def test_set_thread_token_failures_are_gated_by_error_code(self):
        classify = mock.Mock(return_value={'outcome': 'access-denied'})
        manifest = {'secret': 'unused'}
        for error, outcome in [(5, 'access-denied'),
                               (87, 'unexpected-set-thread-token-result'),
                               (6, 'unexpected-set-thread-token-result')]:
            api = self.impersonation_child_api()
            api.set_thread_token = mock.Mock(return_value=False)
            with self.subTest(winerror=error), \
                    mock.patch('builtins.print'), \
                    mock.patch.object(C, 'set_last_error', create=True), \
                    mock.patch.object(C, 'get_last_error', return_value=error,
                                      create=True):
                result = escape_helpers['impersonation'](api, manifest,
                                                         classify)
            self.assertEqual(result['outcome'], outcome)
            self.assertEqual(result.get('winerror'), error)
            # A failed assignment must never leave the revert path unchecked.
            api.revert.assert_not_called()
            api.set_thread_token.assert_called_once()

    def test_failed_reversion_exits_without_retry_or_cleanup(self):
        # Exercise real process termination, not a mocked _exit that unwinds
        # Python finally blocks and thereby tests a different failure path.
        script = r'''
import atexit
import ctypes as C
import runpy
import sys
from unittest import mock
module = runpy.run_path(sys.argv[1])
api = module['EscapeResultTests']().impersonation_child_api()
api.set_thread_token = mock.Mock(return_value=True)
api.n.close.side_effect = lambda handle: print('CLEANUP', flush=True) or True
atexit.register(lambda: print('ATEXIT', flush=True))
def open_thread(thread, rights, self_flag, out):
    if thread != 11:
        raise AssertionError('wrong thread handle')
    C.cast(out, C.POINTER(C.c_void_p)).contents.value = 13
    return True
api.open_thread_token = open_thread
calls = []
def revert():
    calls.append(1)
    print('REVERT', flush=True)
    return len(calls) > 1  # A retry would succeed; it must never happen.
api.revert = revert
if sys.argv[2] == 'query-error':
    api.snapshot.side_effect = [api.snapshot.return_value, RuntimeError('query failed')]
with mock.patch.object(C, 'set_last_error', create=True), \
     mock.patch.object(C, 'get_last_error', return_value=0, create=True):
    module['escape_helpers']['impersonation'](
        api, {'secret': 'unused'}, lambda *args: {'outcome': 'access-denied'})
print('RETURNED', flush=True)
'''
        for mode in ('success', 'query-error'):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    [sys.executable, '-I', '-u', '-c', script, __file__, mode],
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 70, result.stderr)
                self.assertEqual(result.stdout, 'REVERT\n')
                self.assertEqual(result.stderr, '')

    def test_successful_reversion_precedes_cleanup_and_residual_query(self):
        api = self.impersonation_child_api()
        api.set_thread_token = mock.Mock(return_value=True)
        events = []

        def open_thread(thread, rights, self_flag, out):
            self.assertEqual(thread, 11)
            if not events:
                events.append('open')
                C.cast(out, C.POINTER(HANDLE)).contents.value = 13
                return True
            self.assertEqual(events, ['open', 'revert', 'close-13'])
            events.append('residual')
            return False
        api.open_thread_token = open_thread
        api.revert.side_effect = lambda: events.append('revert') or True
        api.n.close.side_effect = lambda h: events.append(
            'close-%s' % getattr(h, 'value', h)) or True
        with mock.patch.object(C, 'set_last_error', create=True), \
                mock.patch.object(C, 'get_last_error', return_value=1008, create=True):
            result = escape_helpers['impersonation'](
                api, {'secret': 'unused'}, lambda *args: {'outcome': 'access-denied'})
        self.assertEqual(result['outcome'], 'impersonated-contained')
        api.revert.assert_called_once_with()
        self.assertEqual(events, ['open', 'revert', 'close-13', 'residual',
                                  'close-9', 'close-7'])

    def test_peer_readiness_failure_preserves_log_and_wait_error(self):
        for failure in (None, subprocess.TimeoutExpired(['peer'], 5), OSError('wait failed')):
            with self.subTest(failure=failure), tempfile.TemporaryFile('w+b') as log:
                log.write(b'peer startup diagnostics')
                peer = mock.Mock()
                peer.wait.return_value = 9
                peer.wait.side_effect = failure
                with self.assertRaisesRegex(RuntimeError, 'peer startup diagnostics') as caught:
                    peer_readiness_failure(peer, log)
                self.assertIs(caught.exception.__cause__, failure)
                detail = 'exit code 9' if failure is None else type(failure).__name__
                self.assertIn(detail, str(caught.exception))

    def test_impersonation_launch_relays_child_report(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        report = '{"outcome": "impersonated-contained"}'
        replies = [subprocess.CompletedProcess([], 0, 'noise\n' + report, ''),
                   subprocess.CompletedProcess([], 1, report, 'boom'),
                   subprocess.CompletedProcess([], 0, '', ''),
                   subprocess.TimeoutExpired(['child'], 20)]
        outcomes = [{'outcome': 'impersonated-contained'}, RuntimeError,
                    RuntimeError, RuntimeError]
        for reply, outcome in zip(replies, outcomes):
            expired = isinstance(reply, subprocess.TimeoutExpired)
            with self.subTest(exit=getattr(reply, 'returncode', 'timeout')), \
                    mock.patch('builtins.print'), \
                    mock.patch.object(
                        subprocess, 'run',
                        side_effect=[reply] if expired else None,
                        return_value=None if expired else reply):
                if isinstance(outcome, dict):
                    self.assertEqual(api.impersonation_launch('s', 'm'), outcome)
                else:
                    with self.assertRaises(outcome):
                        api.impersonation_launch('s', 'm')

    def test_scheduled_task_invoke_outcomes(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        for code, outcome in [(1, 'invoke-rejected'), (0, 'invoke-accepted')]:
            reply = subprocess.CompletedProcess([], code, '', 'ERROR: Access is denied.')
            with self.subTest(exit=code), \
                    mock.patch.dict(os.environ, {'SystemRoot': 'C:\\Windows'}), \
                    mock.patch('builtins.print'), \
                    mock.patch.object(subprocess, 'run', return_value=reply):
                result = api.task_invoke('LokiProbe-x')
            self.assertEqual(result['outcome'], outcome)
            self.assertTrue(escape_helpers['finalize_result'](
                result, 0, ('invoke-rejected',))['passed'] ==
                (outcome == 'invoke-rejected'))
        with mock.patch.dict(os.environ, {'SystemRoot': 'C:\\Windows'}), \
                mock.patch('builtins.print'), \
                mock.patch.object(subprocess, 'run', side_effect=(
                    subprocess.TimeoutExpired(['schtasks'], 15))):
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                api.task_invoke('LokiProbe-x')

    def test_task_route_control_requires_live_dangerous_route(self):
        with tempfile.TemporaryDirectory() as name:
            report = Path(name) / 'r.json'
            live = {'pid': 42, 'token': {'user': 'u', 'app': 0},
                    'credential_access': {'outcome': 'allowed'}}
            report.write_text(json.dumps(live))
            self.assertEqual(verify_task_route(report, 'u'), live)
            for token, access in [
                    ({'user': 'u', 'app': 1}, 'allowed'),
                    ({'user': 'other', 'app': 0}, 'allowed'),
                    ({'user': 'u', 'app': 0}, 'access-denied'),
                    ({'user': 'u', 'app': 0}, 'unexpected-error')]:
                report.write_text(json.dumps(
                    {'pid': 42, 'token': token,
                     'credential_access': {'outcome': access}}))
                with self.subTest(token=token, access=access):
                    with self.assertRaisesRegex(RuntimeError, 'control witness'):
                        verify_task_route(report, 'u')

    def test_task_control_kernel_token_must_match_witness(self):
        with tempfile.TemporaryDirectory() as name:
            report = Path(name) / 'r.json'
            report.write_text(json.dumps(
                {'pid': 42, 'token': {'user': 'u', 'app': 0},
                 'credential_access': {'outcome': 'allowed'}}))
            native, api = mock.Mock(), mock.Mock()
            native.check.side_effect = lambda value: value
            native.open_process.return_value = 7
            api.token.return_value = 8
            api.close.return_value = True

            def snapshot(token):
                return {'user': 'u', 'app': 0}
            for kernel, matches in [(snapshot, True),
                                    (lambda token: {'user': 'other', 'app': 0},
                                     False),
                                    (lambda token: {'user': 'u', 'app': 1},
                                     False)]:
                api.snapshot.side_effect = kernel
                with self.subTest(matches=matches):
                    if matches:
                        verify_task_route(report, 'u', native, api)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'kernel'):
                            verify_task_route(report, 'u', native, api)

    def test_witness_verdict_only_accepts_denied_contained_witnesses(self):
        verdict = witness_verdict
        contained = {'user': 'u', 'app': 1, 'package': 'pkg'}
        cases = [
            (contained, 'access-denied', 'contained'),
            (contained, 'allowed', 'violation-credentials-read'),
            (contained, 'unexpected-error', 'inconclusive-read-outcome'),
            ({'user': 'u', 'app': 0, 'package': None}, 'access-denied',
             'violation-unrestricted-witness'),
            ({'user': 'other', 'app': 1, 'package': 'pkg'}, 'access-denied',
             'violation-unrestricted-witness'),
            ({'user': 'u', 'app': 1, 'package': 'other'}, 'access-denied',
             'violation-anomalous-witness'),
        ]
        for kernel, read, expected in cases:
            with self.subTest(kernel=kernel, read=read):
                self.assertEqual(verdict(kernel, read, 'u', 'pkg'), expected)

    def fake_activation_process(self, alive=True):
        process = mock.Mock()
        process.poll.return_value = None if alive else 0
        process.communicate.side_effect = lambda timeout=5: ('out', 'err')
        process.returncode = 0
        return process

    def test_activation_observation_survives_wrapper_exit(self):
        module = sys.modules[__name__]
        report = mock.Mock()
        report.exists.side_effect = [False, False, True]
        payload = {'pid': 1}
        clock = iter(range(0, 100))
        with mock.patch.object(subprocess, 'Popen',
                               return_value=self.fake_activation_process(False)), \
                mock.patch.object(module, 'verify_task_route',
                                  return_value=payload) as verify, \
                mock.patch.object(time, 'monotonic',
                                  side_effect=lambda: next(clock)), \
                mock.patch.object(time, 'sleep'):
            result = activate_and_observe(['x'], report, 'u', None, None)
        # The wrapper exited immediately; polling still caught the report.
        self.assertEqual(result[0], payload)
        verify.assert_called_once_with(report, 'u', None, None)

    def test_activation_observation_kills_wrapper_on_error(self):
        module = sys.modules[__name__]
        report = mock.Mock()
        report.exists.return_value = True
        clock = iter(range(0, 100))
        process = self.fake_activation_process(alive=True)
        with mock.patch.object(subprocess, 'Popen', return_value=process), \
                mock.patch.object(module, 'verify_task_route',
                                  side_effect=RuntimeError('bad report')), \
                mock.patch.object(time, 'monotonic',
                                  side_effect=lambda: next(clock)), \
                mock.patch.object(time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'bad report'):
                activate_and_observe(['x'], report, 'u', None, None)
        # A live wrapper is killed and reaped even when verification raised.
        process.kill.assert_called_once()
        self.assertGreaterEqual(process.communicate.call_count, 1)

    def test_activation_observation_times_out_without_report(self):
        module = sys.modules[__name__]
        report = mock.Mock()
        report.exists.return_value = False
        clock = iter([0, 30] + [31] * 50)
        process = self.fake_activation_process(alive=True)
        with mock.patch.object(subprocess, 'Popen', return_value=process), \
                mock.patch.object(module, 'verify_task_route') as verify, \
                mock.patch.object(time, 'monotonic',
                                  side_effect=lambda: next(clock)), \
                mock.patch.object(time, 'sleep'):
            result = activate_and_observe(['x'], report, 'u', None, None)
        self.assertIsNone(result[0])
        verify.assert_not_called()
        process.kill.assert_called_once()

    def test_activation_cleanup_failure_surfaces_on_normal_path(self):
        module = sys.modules[__name__]
        report = mock.Mock()
        report.exists.return_value = False
        clock = iter([0, 30] + [31] * 50)
        process = self.fake_activation_process(alive=True)
        process.communicate.side_effect = [('out', 'err'),
                                           RuntimeError('reap failed')]
        with mock.patch.object(subprocess, 'Popen', return_value=process), \
                mock.patch.object(module, 'verify_task_route') as verify, \
                mock.patch.object(time, 'monotonic',
                                  side_effect=lambda: next(clock)), \
                mock.patch.object(time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'reap failed'):
                activate_and_observe(['x'], report, 'u', None, None)
        verify.assert_not_called()

    def test_activation_cleanup_failure_is_reported_not_hidden(self):
        module = sys.modules[__name__]
        report = mock.Mock()
        report.exists.return_value = True
        clock = iter(range(0, 100))
        process = self.fake_activation_process(alive=True)
        process.communicate.side_effect = RuntimeError('cleanup broke')
        with mock.patch.object(subprocess, 'Popen', return_value=process), \
                mock.patch.object(module, 'verify_task_route',
                                  side_effect=RuntimeError('bad report')), \
                mock.patch.object(time, 'monotonic',
                                  side_effect=lambda: next(clock)), \
                mock.patch.object(time, 'sleep'), \
                mock.patch('builtins.print') as printed:
            with self.assertRaisesRegex(RuntimeError, 'bad report'):
                activate_and_observe(['x'], report, 'u', None, None)
        diagnostic = json.loads(printed.call_args.args[0])
        self.assertEqual(diagnostic['probe'], 'com-control')
        self.assertIn('cleanup broke', diagnostic['cleanup_error'])

    def test_watcher_observes_report_at_root_exit_boundary(self):
        with tempfile.TemporaryDirectory() as name:
            report = Path(name) / 'r.json'
            report.write_text(json.dumps({'pid': 7}))
            captured = watch_worker_witnesses(
                {'w': report}, time.monotonic() + 1,
                lambda: False, lambda payload: {'kernel': True})
        # Root already exited; the sweep-before-exit-check ordering still
        # observes the report instead of skipping it.
        self.assertEqual(captured['w'][0]['pid'], 7)
        self.assertEqual(captured['w'][1], {'kernel': True})

    def test_watcher_grace_catches_late_report_after_root_exit(self):
        report = mock.Mock()
        report.exists.side_effect = [False, False, True]
        report.read_text.return_value = json.dumps({'pid': 9})
        # Fine-grained clock within the two-second grace; all three report
        # checks are accounted for, and the capture happens inside the
        # grace sweeps themselves.
        clock = iter([0.0, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30,
                      1.35, 1.40])
        with mock.patch.object(time, 'monotonic',
                               side_effect=lambda: next(clock)), \
                mock.patch.object(time, 'sleep'):
            captured = watch_worker_witnesses(
                {'w': report}, 50.0, lambda: False, lambda payload: None)
        self.assertEqual(captured['w'][0]['pid'], 9)
        self.assertEqual(report.exists.call_count, 3)

    def test_watcher_ignores_report_after_grace_expires(self):
        report = mock.Mock()
        # The report would exist on the third check, but only after the
        # two-second post-exit grace has already expired.
        report.exists.side_effect = [False, False, True]
        report.read_text.return_value = json.dumps({'pid': 11})
        clock = iter([0.0, 1.0, 1.1, 1.5, 1.9, 2.1])
        with mock.patch.object(time, 'monotonic',
                               side_effect=lambda: next(clock)), \
                mock.patch.object(time, 'sleep'):
            captured = watch_worker_witnesses(
                {'w': report}, 50.0, lambda: False, lambda payload: None)
        self.assertEqual(captured, {})
        self.assertEqual(report.exists.call_count, 2)

    def test_watcher_slow_inspection_cannot_license_another(self):
        reports = {}
        for name in ('a', 'b'):
            report = mock.Mock()
            report.exists.return_value = True
            report.read_text.return_value = json.dumps({'pid': 1})
            reports[name] = report
        clock = [0.0]

        # The first inspection consumes the entire remaining budget.
        def inspect(payload):
            clock[0] = 20.0
            return 'inspected'
        with mock.patch.object(time, 'monotonic',
                               side_effect=lambda: clock[0]):
            captured = watch_worker_witnesses(
                reports, 10.0, lambda: True, inspect)
        self.assertEqual(captured, {'a': ({'pid': 1}, 'inspected')})
        self.assertIn('b', reports)
        reports['b'].exists.assert_not_called()

    def test_watcher_slow_exists_cannot_license_read(self):
        report = mock.Mock()
        clock = [0.0]
        # exists() consumes the entire remaining budget; the read must not
        # start afterwards.
        report.exists.side_effect = lambda: clock.__setitem__(0, 20.0) or True
        with mock.patch.object(time, 'monotonic',
                               side_effect=lambda: clock[0]):
            captured = watch_worker_witnesses(
                {'w': report}, 10.0, lambda: True, lambda payload: None)
        self.assertEqual(captured, {})
        report.read_text.assert_not_called()

    def test_watcher_never_sweeps_after_deadline(self):
        report = mock.Mock()
        report.exists.return_value = False
        # Deadline reached at the sweep's own check: no report is read, and
        # the loop-bound check after sleep stops the watcher.
        clock = iter([0.0, 5.0, 5.0])
        with mock.patch.object(time, 'monotonic',
                               side_effect=lambda: next(clock)), \
                mock.patch.object(time, 'sleep'):
            captured = watch_worker_witnesses(
                {'w': report}, 5.0, lambda: True, lambda payload: None)
        self.assertEqual(captured, {})
        self.assertEqual(report.exists.call_count, 0)

    def test_watcher_returns_empty_at_deadline_without_reports(self):
        report = mock.Mock()
        report.exists.return_value = False
        clock = iter([0, 5, 6, 7])
        with mock.patch.object(time, 'monotonic',
                               side_effect=lambda: next(clock)), \
                mock.patch.object(time, 'sleep'):
            captured = watch_worker_witnesses(
                {'w': report}, 5.5, lambda: True, lambda payload: None)
        self.assertEqual(captured, {})
        # One report read inside the loop; the deadline expires before the
        # second sweep can start.
        self.assertEqual(report.exists.call_count, 1)

    def test_reap_closes_pipes_even_when_reaping_fails(self):
        process = self.fake_activation_process(alive=True)
        process.communicate.side_effect = RuntimeError('reap failed')
        with self.assertRaisesRegex(RuntimeError, 'reap failed'):
            reap_activation_wrapper(process)
        process.stdout.close.assert_called_once()
        process.stderr.close.assert_called_once()
        process.kill.assert_called_once()

    def test_scenario_deadline_defaults_before_observation(self):
        self.assertEqual(_scenario_deadline(123.0), 123.0)
        self.assertEqual(_scenario_deadline(None, now=100.0), 190.0)

    def test_remaining_milliseconds_computes_shared_budget(self):
        self.assertEqual(_remaining_milliseconds(None), 90000)
        self.assertEqual(_remaining_milliseconds(110.0, now=100.0), 10000)
        self.assertEqual(_remaining_milliseconds(90.0, now=100.0), 0)
        self.assertEqual(_remaining_milliseconds(100.25, now=100.0), 250)

    def test_guid_from_text_matches_memory_layout(self):
        guid = escape_helpers['guid_from_text']
        # IID_IUnknown has a canonical little-endian memory layout.
        raw = C.string_at(C.byref(guid('{00000000-0000-0000-C000-000000000046}')),
                          C.sizeof(Guids))
        self.assertEqual(raw, bytes.fromhex('0000000000000000C000000000000046'))
        # Field values parse big-endian from text; memory is little-endian.
        raw = C.string_at(
            C.byref(guid('{11223344-5566-7788-99AA-BBCCDDEEFF00}')),
            C.sizeof(Guids))
        self.assertEqual(raw, bytes.fromhex('443322116655887799AABBCCDDEEFF00'))

    def test_com_activation_outcomes(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        for hresult, phase, outcome in [
                (-2147024891, 'co-create', 'activation-denied'),
                (-2147221164, 'co-create', 'class-not-registered'),
                (0, 'co-create', 'unexpected-activation-result'),
                (-2147467259, 'co-create', 'unexpected-activation-result'),
                (-2147024891, 'co-initialize', 'unexpected-activation-result')]:
            reply = subprocess.CompletedProcess(
                [], 0, json.dumps({'hresult': hresult, 'phase': phase}), '')
            with self.subTest(hresult=hresult, phase=phase), \
                    mock.patch('builtins.print'), \
                    mock.patch.object(subprocess, 'run', return_value=reply):
                result = api.com_launch('{00000000-0000-0000-0000-000000000000}')
            self.assertEqual(result['outcome'], outcome)
            passed = escape_helpers['finalize_result'](
                result, 0, ('activation-denied', 'class-not-registered'))['passed']
            self.assertEqual(passed, outcome in ('activation-denied',
                                                 'class-not-registered'))
        with mock.patch('builtins.print'), \
                mock.patch.object(subprocess, 'run', side_effect=(
                    subprocess.TimeoutExpired(['com'], 20))):
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                api.com_launch('{00000000-0000-0000-0000-000000000000}')

    def test_task_witness_reads_request_and_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            manifest = {'secret': 's', 'workspace': 'w', 'owner': 'u'}
            (directory / 'm.json').write_text(json.dumps(manifest))
            (directory / 'request.json').write_text(json.dumps(
                {'manifest': str(directory / 'm.json'),
                 'report': str(directory / 'r.json')}))
            loaded, report = task_witness(directory / 'request.json')
            self.assertEqual(loaded, manifest)
            self.assertEqual(report, directory / 'r.json')

    def test_wmi_operands_are_quoted_without_json_cmdlets(self):
        import base64
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        api.created = 0
        report = Path("work's $name `tick __REQUEST__") / 'report.json'
        arguments = ('python.exe', "probe's.py", Path('manifest.json'),
                     report, 'user', 'package')
        reply = subprocess.CompletedProcess(
            [], 0, '{"kind":"return","code":2,"pid":0}', '')
        with mock.patch.dict(os.environ, {'SystemRoot': 'C:\\Windows'}), \
                mock.patch('builtins.print'), \
                mock.patch.object(subprocess, 'run', return_value=reply) as launch:
            api.wmi_launch(*arguments)
        script = base64.b64decode(launch.call_args.args[0][-1]).decode('utf-16-le')
        command = subprocess.list2cmdline(
            ['python.exe', '-I', '-u', "probe's.py", '--launch-witness',
             'manifest.json', str(report)])
        assignments = ("$command = ('" + command.replace("'", "''") + "')\n" +
                       "$directory = ('" + str(report.parent).replace("'", "''") +
                       "')\n")
        self.assertTrue(script.startswith(assignments))
        self.assertNotIn('ConvertFrom-Json', script)
        self.assertNotIn('ConvertTo-Json', script)

    def test_wmi_numeric_exception_reply_preserves_denial_classification(self):
        api = escape_helpers['Escapes'].__new__(escape_helpers['Escapes'])
        api.created = 0
        arguments = ('python.exe', 'probe.py', Path('manifest.json'),
                     Path('report.json'), 'user', 'package')
        for hresult, denied in [(-2147024891, True), (-2147217405, True),
                                (-2147024809, False)]:
            reply = subprocess.CompletedProcess([], 0, json.dumps(
                {'kind': 'exception', 'hresult': hresult}), 'native exception message')
            with self.subTest(hresult=hresult), \
                    mock.patch.dict(os.environ, {'SystemRoot': 'C:\\Windows'}), \
                    mock.patch('builtins.print') as printed, \
                    mock.patch.object(subprocess, 'run', return_value=reply):
                if denied:
                    self.assertEqual(api.wmi_launch(*arguments)['outcome'],
                                     'access-denied')
                else:
                    with self.assertRaisesRegex(RuntimeError, 'unexpected WMI'):
                        api.wmi_launch(*arguments)
                self.assertEqual(json.loads(printed.call_args.args[0])['stderr'],
                                 'native exception message')
                self.assertEqual(api.created, 0)

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
        # Readiness and diagnostics live outside worker-writable storage. Only
        # identifiers are copied into the contained manifest, never live handles.
        peer_report = root / 'peer.json'
        peer_log = open(root / 'peer.log', 'w+b')
        self.addCleanup(peer_log.close)
        peer = subprocess.Popen(
            [sys.executable, '-I', '-u', __file__, '--peer', str(secret / 'read'),
             str(peer_report)], stdin=subprocess.DEVNULL, stdout=peer_log,
            stderr=subprocess.STDOUT, close_fds=True)

        def retire_peer():
            if peer.poll() is None:
                peer.terminate()
            peer.wait(timeout=10)
        self.addCleanup(retire_peer)
        deadline = time.monotonic() + 15
        while not peer_report.exists():
            if peer.poll() is not None or time.monotonic() >= deadline:
                peer_readiness_failure(peer, peer_log)
            time.sleep(0.02)
        target = json.loads(peer_report.read_text())
        self.assertEqual(target['pid'], peer.pid)
        api = escape_helpers['Escapes'](native, primitives, ExtendedStartups, ProcessInfos)
        # Kernel identity plus real memory/handle controls prove the target and
        # helpers work for the unrestricted same-user broker before denial tests.
        process = native.check(native.open_process(0x1000, False, peer.pid))
        try:
            token = api.token(process, 8)
            try:
                snapshot = api.snapshot(token)
                self.assertEqual(snapshot['user'], owner)
                self.assertEqual(snapshot['app'], 0)
            finally:
                api.close(token)
        finally:
            api.close(process)
        memory_control = api.peer_memory(target)
        handle_control = api.peer_handle(target)
        self.assertTrue(memory_control['read_succeeded'])
        self.assertTrue(memory_control['marker_matches'])
        self.assertTrue(handle_control['duplicated'])
        print(json.dumps({'unrestricted_peer_controls': {
            'memory': memory_control, 'handle': handle_control}}), flush=True)
        manifest = workspace / 'manifest.json'
        # Prearranged own-account scheduled task (design B): the supervisor
        # registers a disposable demand-start S4U task whose fixed bootstrap
        # reads its witness target from the request file this broker owns.
        # Registration is denied on this runner image; without the fixture the
        # route stays explicitly untested (setup-blocked), never silently
        # passing, and the containment probes below still run.
        task_name = None
        control_report = root / ('task-control-' + uuid.uuid4().hex + '.json')
        task_report = root / ('task-report-' + uuid.uuid4().hex + '.json')
        fixture_path = Path(__file__).parent / 'task-fixture.json'
        if fixture_path.exists():
            fixture = json.loads(fixture_path.read_text())
            task_name = fixture['task']
        # The witness reads the manifest at run time, so it must exist before
        # any task invocation; a blocked fixture records null and the worker
        # probe skips itself.
        manifest.write_text(json.dumps({'secret': str(secret),
                                        'workspace': str(workspace),
                                        'owner': owner, 'package': package,
                                        'broker_pid': os.getpid(), 'peer': target,
                                        'scheduled_task': task_name}))
        if task_name is None:
            print(json.dumps({'scheduled_task_setup': 'blocked; route untested'}),
                  flush=True)
        else:
            request_path = Path(fixture['request'])
            schtasks = str(Path(os.environ['SystemRoot']) / 'System32/schtasks.exe')

            def run_schtasks(*arguments):
                result = subprocess.run([schtasks, *arguments], capture_output=True,
                                        text=True, timeout=30)
                self.assertEqual(result.returncode, 0, '%s: %s%s' % (
                    ' '.join(arguments), result.stdout, result.stderr))
                return result

            request_path.write_text(json.dumps({'manifest': str(manifest),
                                                'report': str(control_report)}))
            run_schtasks('/Run', '/TN', task_name)
            deadline = time.monotonic() + 20
            while not control_report.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError('scheduled-task control witness did not report')
                time.sleep(0.05)
            control = verify_task_route(control_report, owner, native, api)
            print(json.dumps({'scheduled_task_control': control}), flush=True)
            # The witness self-exits ten seconds after publishing; wait it out
            # so a still-running instance cannot mask the worker attempt.
            time.sleep(12)
            # Retarget the fixed bootstrap at the worker-attempt report.
            request_path.write_text(json.dumps({'manifest': str(manifest),
                                                'report': str(task_report)}))
        # Per-user local COM server (arm 1, no admin): register, prove the
        # route is live and dangerous via an unrestricted control activation,
        # then retarget the registration at the worker-attempt report. If the
        # control cannot launch, the route stays explicitly untested.
        com_control_report = root / ('com-control-' + uuid.uuid4().hex + '.json')
        com_report = root / ('com-report-' + uuid.uuid4().hex + '.json')
        clsid = '{%s}' % uuid.uuid4()

        def com_witness(report):
            return subprocess.list2cmdline(
                [sys.executable, '-I', '-u', __file__, '--launch-witness',
                 str(manifest), str(report)])

        register_user_com_server(clsid, com_witness(com_control_report))
        self.addCleanup(unregister_user_com_server, clsid)
        escapes = Path(__file__).with_name('windows_escapes.py')
        # Each live route arms one independent worker attempt with its own
        # per-arm report; no arm substitutes for another.
        com_targets = []
        payload, detail = activate_and_observe(
            [sys.executable, '-I', '-u', str(escapes), '--com-activate', clsid],
            com_control_report, owner, native, api)
        if payload is not None:
            print(json.dumps({'com_control': payload, 'scope': 'user'}),
                  flush=True)
            # Wait out the control witness so a live instance cannot mask the
            # worker attempt, then retarget the registration.
            time.sleep(12)
            register_user_com_server(clsid, com_witness(com_report))
            com_targets.append({'name': 'user', 'clsid': clsid,
                                'report': str(com_report)})
        else:
            print(json.dumps({'com_setup': 'control-failed',
                              'scope': 'user',
                              'activation': detail}), flush=True)
        # Machine-wide registration (arm 2): the supervisor registered a
        # disposable local server whose fixed bootstrap reads this request
        # file. The arm runs unconditionally, and when its control goes live
        # it receives its own worker attempt regardless of arm 1; a separate
        # control report path is mandatory, since a late arm-1 witness must
        # never count as this arm's evidence. (Arm 1's 9757654 CLASSNOTREG is
        # retracted: the activation used a mis-built GUID, so per-user
        # session-0 visibility is unmeasured, not denied.)
        com_fixture_path = Path(__file__).parent / 'com-fixture.json'
        if com_fixture_path.exists():
            machine = json.loads(com_fixture_path.read_text())
            # Setup and control failures in one arm must not suppress the
            # other. A differing result narrows the investigation, but does
            # not by itself identify the offending registration value.
            for scope, clsid_field, request_field in (
                    ('machine', 'clsid', 'request'),
                    ('machine-noappid', 'clsid_noappid', 'request_noappid')):
                with self.subTest(com_control=scope):
                    if clsid_field not in machine:
                        print(json.dumps({'com_setup': 'registration-unavailable',
                                          'scope': scope}), flush=True)
                        continue
                    machine_cls = machine[clsid_field]
                    com_request = Path(machine[request_field])
                    control_report_path = root / (
                        'com-' + scope + '-control-' + uuid.uuid4().hex + '.json')
                    com_request.write_text(json.dumps(
                        {'manifest': str(manifest),
                         'report': str(control_report_path)}))
                    payload, detail = activate_and_observe(
                        [sys.executable, '-I', '-u', str(escapes), '--com-activate',
                         machine_cls], control_report_path, owner, native, api)
                    if payload is None:
                        print(json.dumps({'com_setup': 'control-failed',
                                          'scope': scope,
                                          'activation': detail}), flush=True)
                        continue
                    print(json.dumps({'com_control': payload, 'scope': scope}),
                          flush=True)
                    time.sleep(12)
                    # The no-AppID comparison is control-only. It never
                    # becomes a worker target or substitutes for the AppID arm.
                    if scope == 'machine':
                        machine_report = root / (
                            'com-machine-report-' + uuid.uuid4().hex + '.json')
                        com_request.write_text(json.dumps(
                            {'manifest': str(manifest),
                             'report': str(machine_report)}))
                        com_targets.append({'name': scope, 'clsid': machine_cls,
                                            'report': str(machine_report)})
        manifest.write_text(json.dumps({'secret': str(secret),
                                        'workspace': str(workspace),
                                        'owner': owner, 'package': package,
                                        'broker_pid': os.getpid(), 'peer': target,
                                        'scheduled_task': task_name,
                                        'com': com_targets}))
        for mode in ('normal-root-exit', 'terminated-root'):
            with self.subTest(cleanup=mode):
                directory = workspace / mode
                directory.mkdir()
                retained = []

                def observe(root_process, job):
                    deadline = time.monotonic() + 20
                    reports = [directory / ('ready-%d' % i) for i in range(3)]
                    while not all(path.exists() for path in reports):
                        if native.wait(root_process, 0) != 258:
                            raise RuntimeError('cleanup root exited before readiness')
                        if time.monotonic() >= deadline:
                            raise TimeoutError('cleanup tree readiness timed out')
                        time.sleep(0.02)
                    for path in reports[1:]:
                        pid = int(path.read_text())
                        handle = native.check(native.open_process(
                            0x1000 | 0x100000 | 1, False, pid))
                        member = W.BOOL()
                        try:
                            native.check(native.in_job(handle, job, C.byref(member)))
                            if not member.value:
                                raise RuntimeError('reported process outside fixture job')
                            self.assertEqual(native.app_identity(handle), (1, package))
                        except BaseException:
                            api.close(handle)
                            raise
                        retained.append(handle)
                        self.assertEqual(native.wait(handle, 0), 258)
                    if mode == 'terminated-root':
                        native.check(native.terminate(root_process, 1))
                    else:
                        (directory / 'release').write_text('release')

                try:
                    code = native.launch(
                        [sys.executable, '-I', '-u', __file__, '--cleanup-tree',
                         str(directory), '0'], sid, directory, root / (mode + '.log'),
                        observe=observe)
                    self.assertEqual(code, 1 if mode == 'terminated-root' else 0)
                    # launch() has closed its job. Observe death before fallback
                    # retirement or the administrative account-wide sweep.
                    states = [native.wait(handle, 5000) for handle in retained]
                    print(json.dumps({'cleanup': mode, 'descendant_waits': states}),
                          flush=True)
                    self.assertEqual(states, [0, 0])
                finally:
                    try:
                        diagnostic = root / (mode + '.log')
                        if diagnostic.exists():
                            print(diagnostic.read_text(errors='replace'), flush=True)
                    finally:
                        with ExitStack() as cleanup:
                            for handle in retained:
                                cleanup.callback(api.retire, handle)
        with self.subTest(cleanup='broker-death'):
            directory = workspace / 'broker-death'
            directory.mkdir()
            report = root / 'broker-death.json'
            broker_log = open(root / 'broker-death-broker.log', 'w+b')
            sub_broker = subprocess.Popen(
                [sys.executable, '-I', '-u', __file__, '--broker-death',
                 str(directory), package, str(report)],
                stdin=subprocess.DEVNULL, stdout=broker_log,
                stderr=subprocess.STDOUT, close_fds=True)
            retained = []
            try:
                deadline = time.monotonic() + 30
                while not report.exists():
                    if sub_broker.poll() is not None or time.monotonic() >= deadline:
                        broker_log.flush()
                        broker_log.seek(0)
                        self.fail('broker-death sub-broker did not report; log: %s'
                                  % broker_log.read().decode(errors='replace'))
                    time.sleep(0.05)
                payload = json.loads(report.read_text())
                self.assertEqual(payload['broker_pid'], sub_broker.pid)
                for pid in payload['tree_pids']:
                    handle = native.check(native.open_process(
                        0x1000 | 0x100000 | 1, False, pid))
                    member = W.BOOL()
                    try:
                        # No job handle crosses the process boundary: membership
                        # is verified against any job plus AppContainer identity.
                        native.check(native.in_job(handle, None, C.byref(member)))
                        self.assertTrue(member.value)
                        self.assertEqual(native.app_identity(handle), (1, package))
                    except BaseException:
                        api.close(handle)
                        raise
                    retained.append(handle)
                    self.assertEqual(native.wait(handle, 0), 258)
                # Kill the disposable broker without executing any of its code.
                broker_handle = native.check(native.open_process(
                    0x1000 | 0x100000 | 1, False, sub_broker.pid))
                try:
                    native.check(native.terminate(broker_handle, 1))
                    sub_broker.wait(timeout=10)
                finally:
                    api.close(broker_handle)
                # Every broker handle, including its job, is gone now: the tree
                # must die with no finally path and before any fallback sweep.
                states = [native.wait(handle, 10000) for handle in retained]
                print(json.dumps({'cleanup': 'broker-death',
                                  'descendant_waits': states}), flush=True)
                self.assertEqual(states, [0, 0, 0])
            finally:
                if sub_broker.poll() is None:
                    sub_broker.terminate()
                sub_broker.wait(timeout=10)
                broker_log.close()
                with ExitStack() as cleanup:
                    for handle in retained:
                        cleanup.callback(api.retire, handle)
        with self.subTest(cleanup='shutdown-race'):
            directory = workspace / 'shutdown-race'
            directory.mkdir()
            retained = []

            def observe(root_process, job):
                # Let creation and shutdown genuinely overlap: only require the
                # root plus a handful of recorded children before terminating.
                deadline = time.monotonic() + 20
                ready = directory / 'ready-0'
                track = directory / 'race-pids'
                while (not ready.exists() or not track.exists() or
                       len(track.read_text().splitlines()) < 5):
                    if native.wait(root_process, 0) != 258:
                        raise RuntimeError('race root exited before children')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('race tree readiness timed out')
                    time.sleep(0.01)
                native.check(native.terminate(root_process, 1))

            try:
                code = native.launch(
                    [sys.executable, '-I', '-u', __file__, '--race-tree',
                     str(directory), '0'], sid, directory,
                    root / 'shutdown-race.log', observe=observe)
                self.assertEqual(code, 1)
                # launch() has returned, so its job is closed. Any recorded
                # child that is still alive survived creation racing shutdown.
                pids = [int(value) for value in
                        (directory / 'race-pids').read_text().split()]
                survivors = []
                for pid in pids:
                    handle = native.open_process(0x1000 | 0x100000 | 1, False, pid)
                    if not handle:
                        continue
                    try:
                        if native.wait(handle, 3000) != 0:
                            survivors.append(pid)
                    finally:
                        api.close(handle)
                print(json.dumps({'cleanup': 'shutdown-race',
                                  'children': len(pids),
                                  'survivors': survivors}), flush=True)
                self.assertEqual(survivors, [])
            finally:
                try:
                    diagnostic = root / 'shutdown-race.log'
                    if diagnostic.exists():
                        print(diagnostic.read_text(errors='replace'), flush=True)
                finally:
                    with ExitStack() as cleanup:
                        for handle in retained:
                            cleanup.callback(api.retire, handle)
        log = root / 'contained.log'
        self.assertIsNone(peer.poll(), 'unrestricted target exited before attacks')

        def witness_kernel_token(payload):
            """Inspect a witness process only while verifiably still running."""
            try:
                handle = native.check(native.open_process(
                    0x1000 | 0x100000 | 1, False, payload['pid']))
            except OSError:
                return None
            try:
                if native.wait(handle, 0) != 258:
                    # Already exited: this would not be a live inspection.
                    return None
                token = api.token(handle, 8)
                try:
                    return api.snapshot(token)
                finally:
                    api.close(token)
            except OSError:
                return None
            finally:
                api.close(handle)

        # Live observation during the scenario: witnesses self-exit ten
        # seconds after publishing, so kernel inspection must happen while
        # the root process still runs. The observer and the final wait share
        # one deadline, so observation never extends the launch budget.
        live_witnesses = {}
        scenario_deadline = time.monotonic() + 90

        def observe_worker_witnesses(root_process, job):
            watched = dict(
                [('scheduled-task', task_report)] +
                [('com-' + entry['name'], Path(entry['report']))
                 for entry in com_targets])
            live_witnesses.update(watch_worker_witnesses(
                watched, scenario_deadline,
                lambda: native.wait(root_process, 0) == 258,
                witness_kernel_token))

        try:
            code = native.launch([sys.executable, '-I', '-u', __file__,
                                  '--contained', str(manifest)],
                                 sid, workspace, log,
                                 observe=observe_worker_witnesses,
                                 deadline=scenario_deadline)
        finally:
            if log.exists():
                print(log.read_text(errors='replace'), flush=True)
        self.assertIsNone(peer.poll(), 'unrestricted target died during attacks')
        # Worker-side evidence: only a witness observed live while the
        # scenario ran is decisive. A report that escaped live observation,
        # a witness that could not be kernel-inspected while alive, or a
        # token disagreement each fail closed rather than fall back to the
        # self-report.
        watched_reports = [('scheduled-task', task_report,
                            'contained worker invoked an unrestricted task')]
        watched_reports += [
            ('com-' + entry['name'], Path(entry['report']),
             'contained worker launched an unrestricted COM server')
            for entry in com_targets]
        for name, report_path, message in watched_reports:
            live = live_witnesses.get(name)
            if live is None:
                if report_path.exists():
                    self.fail('%s: witness report appeared but was never '
                              'observed live' % name)
                print(json.dumps({'witness': name,
                                  'result': 'no report; invocation was '
                                            'rejected or the route was not '
                                            'armed'}), flush=True)
                continue
            payload, kernel = live
            if kernel is None:
                self.fail('%s: witness was not kernel-inspectable while '
                          'live' % name)
            if kernel != payload['token']:
                self.fail('%s: kernel token disagrees with witness report'
                          % name)
            verdict = witness_verdict(
                kernel, payload['credential_access'].get('outcome'),
                owner, package)
            print(json.dumps({'witness': name, 'verdict': verdict,
                              'report': payload, 'kernel_token': kernel}),
                  flush=True)
            self.assertEqual(verdict, 'contained',
                             '%s: %s' % (message, verdict))
        # Check integrity independently of the child-reported operation results.
        for leaf in names:
            self.assertEqual((secret / leaf).read_bytes(), b'FAKE-CREDENTIAL')
        self.assertEqual(code, 0, 'AppContainer probe failed; see native log')


if __name__ == '__main__':
    if '-Embedding' in sys.argv:
        # COM appends this flag to local-server command lines; the witness
        # dispatch must not be confused by it.
        sys.argv.remove('-Embedding')
    if len(sys.argv) == 5 and sys.argv[1] == '--broker-death':
        broker_death(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
        sys.exit(0)
    if len(sys.argv) == 4 and sys.argv[1] == '--race-tree':
        race_tree(Path(sys.argv[2]), int(sys.argv[3]))
        sys.exit(0)
    if len(sys.argv) == 4 and sys.argv[1] == '--cleanup-tree':
        cleanup_tree(Path(sys.argv[2]), int(sys.argv[3]))
        sys.exit(0)
    if len(sys.argv) == 4 and sys.argv[1] == '--peer':
        escape_helpers['peer'](AppContainers(), sys.argv[2], Path(sys.argv[3]))
        sys.exit(0)
    if len(sys.argv) == 4 and sys.argv[1] == '--launch-witness':
        native = AppContainers()
        api = escape_helpers['Escapes'](
            native, primitives, ExtendedStartups, ProcessInfos)
        escape_helpers['witness'](api, json.loads(Path(sys.argv[2]).read_text()),
                                  Path(sys.argv[3]), access_outcome)
        sys.exit(0)
    if len(sys.argv) == 3 and sys.argv[1] == '--token-impersonation':
        native = AppContainers()
        api = escape_helpers['Escapes'](
            native, primitives, ExtendedStartups, ProcessInfos)
        report = escape_helpers['impersonation'](
            api, json.loads(Path(sys.argv[2]).read_text()), access_outcome)
        print(json.dumps(report), flush=True)
        sys.exit(0 if report['outcome'] in ('impersonated-contained',
                                            'access-denied') else 1)
    if len(sys.argv) == 3 and sys.argv[1] in ('--task-witness', '--com-witness'):
        manifest, report = task_witness(Path(sys.argv[2]))
        native = AppContainers()
        api = escape_helpers['Escapes'](
            native, primitives, ExtendedStartups, ProcessInfos)
        escape_helpers['witness'](api, manifest, report, access_outcome)
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
