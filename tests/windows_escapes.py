"""Test-only escape attempts; no production sandbox or elevation requests.

Contracts (not complete escape-resistance guarantees):
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute
  Alternate parent requires PROCESS_CREATE_PROCESS; inherited state includes
  token/job. A failed prerequisite is reported at that stage, not as a launch.
https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-adjusttokenprivileges
  Cannot add absent privileges; nonzero return still requires GetLastError.
https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-duplicatetokenex
  Duplication preserves a security context; handle rights are not identity.
https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntsetinformationtoken
  TokenUser is read-only for this API. This does not specify all lowbox setters.
https://learn.microsoft.com/en-us/windows/win32/api/shellapi/ns-shellapi-shellexecuteinfow
  NOCLOSEPROCESS requests a handle, but success need not return one. NO_UI does
  NOT suppress security prompts. We use open on the staged Python, never runas.
https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/create-method-in-class-win32-process
  WMI 0 means creation, not completion or confinement; 2/3 mean denied/privilege
  failure. Other return values and uninspectable launches fail these probes.
https://learn.microsoft.com/en-us/windows/win32/wmisdk/wmi-error-constants
  WBEM_E_ACCESS_DENIED is 0x80041003, not a generic WMI failure.

Shell/WMI witnesses only query their own token and attempt a fake-file read.
They self-exit after ten seconds even if the launching probe loses its handle.
The administrative CI fixture additionally cleans up this unique user's
processes; this is cleanup, not part of the confinement claim.
"""

import base64
from contextlib import ExitStack
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


HANDLE = C.c_void_p
ULONG = C.c_uint32
LONG = C.c_int32


class Luids(C.Structure):
    _fields_ = [('low', ULONG), ('high', LONG)]


class PrivilegeEntries(C.Structure):
    _fields_ = [('luid', Luids), ('attributes', ULONG)]


class TokenPrivileges(C.Structure):
    _fields_ = [('count', ULONG), ('entries', PrivilegeEntries * 1)]


class ShellInfos(C.Structure):
    _fields_ = [('size', ULONG), ('mask', ULONG), ('window', HANDLE),
                ('verb', W.LPCWSTR), ('file', W.LPCWSTR),
                ('parameters', W.LPCWSTR), ('directory', W.LPCWSTR),
                ('show', C.c_int), ('instance', HANDLE), ('idlist', HANDLE),
                ('class_name', W.LPCWSTR), ('class_key', HANDLE),
                ('hotkey', ULONG), ('icon', HANDLE), ('process', HANDLE)]


def privilege_rejected(succeeded, error, enabled_after):
    # ERROR_NOT_ALL_ASSIGNED is meaningful even with a TRUE function return.
    return bool(succeeded) and error == 1300 and not enabled_after


def finalize_result(result, created, accepted):
    result = dict(result, created_processes=created)
    if result['outcome'] == 'access-denied' and created:
        result.update(outcome='uninspectable-launch', denied=False)
    result['passed'] = result['outcome'] in accepted
    return result


def identity_matches(snapshot, owner, package):
    return (snapshot['user'] == owner and snapshot['app'] == 1 and
            snapshot['package'] == package)


def run_powershell(script, probe):
    shell = Path(os.environ['SystemRoot']) / (
        'System32/WindowsPowerShell/v1.0/powershell.exe')
    command = [str(shell), '-NoLogo', '-NoProfile', '-NonInteractive',
               '-EncodedCommand',
               base64.b64encode(script.encode('utf-16-le')).decode()]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired as error:
        # TimeoutExpired can contain bytes even when text=True was requested.
        def text(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else value
        print(json.dumps({'probe': probe, 'timed_out': True,
                          'stdout': text(error.stdout),
                          'stderr': text(error.stderr)}), flush=True)
        raise RuntimeError('%s wrapper timed out after 15 seconds' % probe) from error
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError('%s wrapper did not complete: %s' % (probe, error)) from error
    print(json.dumps({'probe': probe, 'wrapper_exit': result.returncode,
                      'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
    if result.returncode:
        raise RuntimeError('%s PowerShell wrapper failed' % probe)
    return result.stdout


class Escapes:
    def __init__(self, native, primitives, startup_type, process_type):
        self.n = native
        self.created = 0  # Successful launches in this sequential probe batch.
        self.primitives = primitives
        self.startup_type = startup_type
        self.process_type = process_type
        bind = native.bind
        self.duplicate = bind(native.advapi, 'DuplicateTokenEx', W.BOOL,
                              HANDLE, ULONG, HANDLE, C.c_int, C.c_int,
                              C.POINTER(HANDLE))
        self.adjust = bind(native.advapi, 'AdjustTokenPrivileges', W.BOOL,
                           HANDLE, W.BOOL, HANDLE, ULONG, HANDLE, HANDLE)
        self.lookup_value = bind(native.advapi, 'LookupPrivilegeValueW', W.BOOL,
                                 W.LPCWSTR, W.LPCWSTR, C.POINTER(Luids))
        self.lookup_name = bind(native.advapi, 'LookupPrivilegeNameW', W.BOOL,
                                W.LPCWSTR, C.POINTER(Luids), W.LPWSTR,
                                C.POINTER(ULONG))
        self.parse_sid = bind(native.advapi, 'ConvertStringSidToSidW', W.BOOL,
                              W.LPCWSTR, C.POINTER(HANDLE))
        self.set_token = bind(native.ntdll, 'NtSetInformationToken', LONG,
                              HANDLE, C.c_int, HANDLE, ULONG)
        self.shell = C.WinDLL('shell32', use_last_error=True)
        self.ole = C.WinDLL('ole32')
        self.shell_execute = bind(self.shell, 'ShellExecuteExW', W.BOOL,
                                  C.POINTER(ShellInfos))
        self.co_initialize = bind(self.ole, 'CoInitializeEx', LONG, HANDLE, ULONG)
        self.co_uninitialize = bind(self.ole, 'CoUninitialize', None)

    def token(self, process, rights):
        token = HANDLE()
        self.n.check(self.n.open_token(process, rights, C.byref(token)))
        return token

    def buffer(self, token, kind):
        size = ULONG()
        result = self.n.token_info(token, kind, None, 0, C.byref(size))
        error = 0 if result else C.get_last_error()
        if result or error != 122 or not size.value:
            raise OSError('token class %d sizing: success=%s, error=%s, size=%s' %
                          (kind, bool(result), error, size.value))
        buffer = C.create_string_buffer(size.value)
        self.n.check(self.n.token_info(token, kind, buffer, len(buffer),
                                       C.byref(size)))
        return buffer

    def privileges(self, token):
        buffer = self.buffer(token, 3)
        count = ULONG.from_buffer(buffer).value
        offset = TokenPrivileges.entries.offset
        if offset + count * C.sizeof(PrivilegeEntries) > len(buffer):
            raise OSError('invalid TOKEN_PRIVILEGES size')
        entries = (PrivilegeEntries * count).from_buffer(buffer, offset)
        result = {}
        for entry in entries:
            size = ULONG()
            success = self.lookup_name(None, C.byref(entry.luid), None,
                                       C.byref(size))
            if success or C.get_last_error() != 122:
                raise OSError('unexpected privilege-name sizing result')
            name = C.create_unicode_buffer(size.value + 1)
            size.value = len(name)
            self.n.check(self.lookup_name(None, C.byref(entry.luid), name,
                                          C.byref(size)))
            result[name.value] = entry.attributes
        return result

    def groups(self, token, kind):
        buffer = self.buffer(token, kind)
        count = ULONG.from_buffer(buffer).value
        if not count:
            return []
        entry_type = self.primitives['SidAttributes']
        offset = self.primitives['TokenGroups'].Groups.offset
        if offset + count * C.sizeof(entry_type) > len(buffer):
            raise OSError('invalid token group buffer')
        entries = (entry_type * count).from_buffer(buffer, offset)
        return sorted([self.n.sid(entry.Sid), entry.Attributes] for entry in entries)

    def snapshot(self, token):
        user = self.buffer(token, 1)
        app, returned = ULONG(), ULONG()
        self.n.check(self.n.token_info(token, 29, C.byref(app), 4,
                                       C.byref(returned)))
        package = None
        if app.value:
            buffer = self.buffer(token, 31)
            package = self.n.sid(HANDLE.from_buffer(buffer))
        return {'user': self.n.sid(HANDLE.from_buffer(user)),
                'app': app.value, 'package': package,
                'privileges': self.privileges(token),
                'groups': self.groups(token, 2),
                'restricting_sids': self.groups(token, 11),
                'capabilities': self.groups(token, 30)}

    def clone(self, source, rights):
        token = HANDLE()
        self.n.check(self.duplicate(source, rights, None, 2, 1, C.byref(token)))
        return token

    def close(self, handle):
        self.n.check(self.n.close(handle))

    def retire(self, process):
        # For Shell/WMI, token inspection may itself be denied. Still attempt
        # cleanup; witnesses also have a bounded lifetime independent of us.
        try:
            if self.n.wait(process, 0) == 258:
                self.n.check(self.n.terminate(process, 1))
                if self.n.wait(process, 10000) != 0:
                    raise TimeoutError('escape-probe child did not exit')
        finally:
            self.close(process)

    def alternate_parent(self, parent, executable, owner, package):
        size = C.c_size_t()
        self.n.initialize(None, 1, 0, C.byref(size))
        if C.get_last_error() != 122 or not size.value:
            raise OSError('parent attribute sizing failed')
        buffer = C.create_string_buffer(size.value)
        self.n.check(self.n.initialize(buffer, 1, 0, C.byref(size)))
        with ExitStack() as cleanup:
            cleanup.callback(self.n.delete_attributes, buffer)
            handle = HANDLE(parent)
            self.n.check(self.n.update(buffer, 0, 0x20000, C.byref(handle),
                                       C.sizeof(handle), None, None))
            startup = self.startup_type()
            startup.startup.cb = C.sizeof(startup)
            startup.attributes = C.cast(buffer, HANDLE)
            child = self.process_type()
            command = C.create_unicode_buffer(subprocess.list2cmdline(
                [executable, '-I', '-c', 'pass']))
            self.n.check(self.n.create_process(
                executable, command, None, None, False, 0x80000 | 4,
                None, None, C.byref(startup), C.byref(child)))
            self.created += 1
            cleanup.callback(self.close, child.thread)
            cleanup.callback(self.retire, child.process)
            # Inspect before executing any child code, even if it escaped.
            token = self.token(child.process, 8)
            cleanup.callback(self.close, token)
            snapshot = self.snapshot(token)
            return {'outcome': ('contained' if identity_matches(
                snapshot, owner, package) else 'escaped'), 'token': snapshot,
                'pid': child.pid, 'executed': False}

    def inspect_witness(self, process, report, owner, package):
        with ExitStack() as cleanup:
            cleanup.callback(self.retire, process)
            token = self.token(process, 8)
            cleanup.callback(self.close, token)
            # Kernel-query result, not just a witness assertion of identity.
            snapshot = self.snapshot(token)
            deadline = time.monotonic() + 5
            while not report.exists():
                if self.n.wait(process, 0) != 258:
                    raise RuntimeError('launch witness exited without a report')
                if time.monotonic() >= deadline:
                    raise TimeoutError('launch witness report timed out')
                time.sleep(0.02)
            payload = json.loads(report.read_text())
            safe = (identity_matches(snapshot, owner, package) and
                    payload['token'] == snapshot and
                    payload['credential_access']['outcome'] == 'access-denied')
            return {'outcome': 'contained' if safe else 'escaped',
                    'token': snapshot, 'witness': payload}

    def shell_launch(self, executable, script, manifest, report, owner, package):
        self.n.hresult(self.co_initialize(None, 2 | 4))
        try:
            info = ShellInfos()
            info.size = C.sizeof(info)
            info.mask = 0x40 | 0x100 | 0x400
            info.verb = 'open'
            info.file = executable
            info.parameters = subprocess.list2cmdline(
                ['-I', '-u', script, '--launch-witness', str(manifest), str(report)])
            info.directory = str(report.parent)
            info.show = 0
            self.n.check(self.shell_execute(C.byref(info)))
            self.created += 1  # Conservatively treat Shell success as launch.
            if not info.process:
                raise RuntimeError('ShellExecute succeeded without inspectable handle')
            return self.inspect_witness(info.process, report, owner, package)
        finally:
            self.co_uninitialize()

    def powershell_startup(self):
        # Same direct-descendant launch, environment and deadline as WMI; no
        # broker access or process creation inside this control script.
        output = run_powershell("""
[Console]::Error.WriteLine('startup-control: entered')
[Console]::Error.Flush()
[Console]::Out.WriteLine('startup-control: ready')
[Console]::Out.Flush()
""", 'powershell-startup')
        if output.strip() != 'startup-control: ready':
            raise RuntimeError('PowerShell startup control returned an invalid reply')
        return {'outcome': 'ready'}

    def wmi_launch(self, executable, script, manifest, report, owner, package):
        command = subprocess.list2cmdline(
            [executable, '-I', '-u', script, '--launch-witness', str(manifest),
             str(report)])
        # The PowerShell process is itself a direct sandbox descendant. Failure
        # to start PowerShell or produce this protocol is NOT WMI denial evidence.
        # Single-quoted literals prevent interpolation of paths/arguments. Build
        # assignments separately so operand text is never a template placeholder.

        def literal(value):
            value = value.replace("'", "''")
            # Keep typographic quotes out of PowerShell's string tokenizer.
            for quote in '\u2018\u2019\u201a\u201b':
                value = value.replace(quote, "' + [char]%d + '" % ord(quote))
            return "('" + value + "')"
        ps = ('$command = ' + literal(command) + '\n$directory = ' +
              literal(str(report.parent)) + '\n') + r"""
$ErrorActionPreference = 'Stop'
function Mark-Phase($phase) {
    [Console]::Error.WriteLine($phase)
    [Console]::Error.Flush()
}
try {
    Mark-Phase 'wmi: before-class'
    $class = [wmiclass]'\\.\root\cimv2:Win32_Process'
    Mark-Phase 'wmi: after-class'
    Mark-Phase 'wmi: before-create'
    $result = $class.Create($command, $directory, $null)
    Mark-Phase 'wmi: after-create'
    [Console]::Out.WriteLine(('{{"kind":"return","code":{0},"pid":{1}}}' -f [int]$result.ReturnValue, [int]$result.ProcessId))
    [Console]::Out.Flush()
} catch {
    $errorObject = $_.Exception.GetBaseException()
    [Console]::Error.WriteLine($errorObject.Message)
    [Console]::Error.Flush()
    [Console]::Out.WriteLine(('{{"kind":"exception","hresult":{0}}}' -f [int]$errorObject.HResult))
    [Console]::Out.Flush()
}
"""
        reply = json.loads(run_powershell(ps, 'wmi'))
        if reply['kind'] == 'exception':
            status = reply['hresult'] & 0xffffffff
            if status not in (0x80070005, 0x80041003):
                raise RuntimeError('unexpected WMI exception: %r' % reply)
            return {'outcome': 'access-denied', 'phase': 'wmi-exception',
                    'reply': reply}
        if reply['kind'] != 'return':
            raise RuntimeError('unexpected WMI reply: %r' % reply)
        if reply['code'] in (2, 3):
            return {'outcome': 'access-denied', 'phase': 'wmi-create', 'reply': reply}
        if reply['code'] == 0:
            self.created += 1
        if reply['code'] != 0 or reply['pid'] <= 0:
            raise RuntimeError('unexpected WMI creation result: %r' % reply)
        # QUERY_LIMITED_INFORMATION | SYNCHRONIZE | TERMINATE. Failure here is
        # an uninspectable successful launch, NOT evidence WMI denied creation.
        process = self.n.open_process(0x1000 | 0x100000 | 1, False, reply['pid'])
        if not process:
            error = C.get_last_error()
            raise RuntimeError('WMI created pid %s; inspection failed: %s' %
                               (reply['pid'], error))
        return self.inspect_witness(process, report, owner, package)


def witness(api, manifest, report, classify):
    import os
    token = api.token(api.n.current_process(), 8)
    try:
        snapshot = api.snapshot(token)
    finally:
        api.close(token)
    outcome = classify('witness-credential-read',
                       lambda: (Path(manifest['secret']) / 'read').read_bytes())
    payload = {'pid': os.getpid(), 'token': snapshot, 'credential_access': outcome}
    temporary = report.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload))
    os.replace(temporary, report)
    # Bound even a successfully escaped process; no broker/parent survival or
    # successful TerminateProcess call is required for this payload to exit.
    time.sleep(10)


def probe(api, manifest, script, manifest_path, classify):
    import sys
    results = []
    owner, package = manifest['owner'], manifest['package']

    def run(name, operation, accepted=('access-denied',)):
        # Distinguish API rejection from malformed calls/unsupported wrappers.
        detail = {}

        def execute():
            detail.update(operation() or {})
        before = api.created
        try:
            result = classify(name, execute)
        except Exception as error:
            result = {'operation': name, 'outcome': 'unexpected-error',
                      'exception': type(error).__name__, 'message': str(error)}
        if result['outcome'] == 'allowed':
            result.update(detail)
        # Denial during inspection/cleanup of an already created process is
        # NOT denial of launch. This also handles Shell success with no handle.
        result = finalize_result(result, api.created - before, accepted)
        results.append(result)
        print(json.dumps(result), flush=True)

    def broker_handle(rights):
        return api.n.check(api.n.open_process(rights, False, manifest['broker_pid']))

    def broker_access():
        api.close(broker_handle(0x80))
    run('broker-create-process-access', broker_access)

    def broker_parent():
        print(json.dumps({'probe': 'alternate-broker-parent',
                          'phase': 'open-parent'}), flush=True)
        parent = broker_handle(0x80)
        try:
            print(json.dumps({'probe': 'alternate-broker-parent',
                              'phase': 'create-suspended'}), flush=True)
            return api.alternate_parent(parent, sys.executable, owner, package)
        finally:
            api.close(parent)
    run('alternate-broker-parent', broker_parent)
    run('alternate-self-parent-control', lambda: api.alternate_parent(
        api.n.current_process(), sys.executable, owner, package), ('contained',))

    def broker_token():
        process = broker_handle(0x1000)
        try:
            token = api.token(process, 8 | 2 | 1)
            api.close(token)
        finally:
            api.close(process)
    run('broker-token-acquisition', broker_token)

    # A fresh duplicate for every mutation: failed experiments cannot alter the
    # token used by later filesystem checks or the ordinary descendant control.
    source = api.token(api.n.current_process(), 8 | 2)
    try:
        before = api.snapshot(source)
        print(json.dumps({'worker_token_snapshot': before}), flush=True)

        def duplicate_control():
            clone = api.clone(source, 8)
            try:
                after = api.snapshot(clone)
                return {'outcome': 'unchanged' if after == before else 'changed',
                        'token': after}
            finally:
                api.close(clone)
        run('duplicate-token-control', duplicate_control, ('unchanged',))

        for privilege in ('SeDebugPrivilege', 'SeBackupPrivilege',
                          'SeRestorePrivilege', 'SeTakeOwnershipPrivilege',
                          'SeImpersonatePrivilege', 'SeAssignPrimaryTokenPrivilege',
                          'SeTcbPrivilege', 'SeCreateTokenPrivilege'):
            def enable(privilege=privilege):
                clone = api.clone(source, 8 | 0x20)
                try:
                    request = TokenPrivileges()
                    request.count = 1
                    api.n.check(api.lookup_value(None, privilege,
                                                 C.byref(request.entries[0].luid)))
                    request.entries[0].attributes = 2
                    C.set_last_error(0)
                    success = api.adjust(clone, False, C.byref(request), 0,
                                         None, None)
                    error = C.get_last_error()
                    if not success:
                        raise C.WinError(error)
                    try:
                        after = api.privileges(clone)
                    except OSError as query_error:
                        raise RuntimeError(
                            'privilege adjustment returned %s; cannot inspect state: %s'
                            % (error, query_error)) from query_error
                    enabled = bool(after.get(privilege, 0) & 2)
                    rejected = privilege_rejected(success, error, enabled)
                    return {'outcome': 'not-assigned' if rejected else 'unexpected-adjustment',
                            'winerror': error, 'privilege': privilege,
                            'before': before['privileges'].get(privilege),
                            'after': after.get(privilege)}
                finally:
                    api.close(clone)
            run('enable-' + privilege, enable, ('not-assigned', 'access-denied'))

        def change_user():
            with ExitStack() as cleanup:
                clone = api.clone(source, 8 | 0x80)
                cleanup.callback(api.close, clone)
                sid = HANDLE()
                api.n.check(api.parse_sid('S-1-5-18', C.byref(sid)))
                cleanup.callback(api.n.local_free, sid)
                user = api.primitives['SidAttributes'](sid, 0)
                status = api.set_token(clone, 1, C.byref(user), C.sizeof(user))
                print(json.dumps({'operation': 'set-token-user-status',
                                  'ntstatus': '0x%08x' % (status & 0xffffffff)}),
                      flush=True)
                try:
                    after = api.snapshot(clone)
                except OSError as query_error:
                    raise RuntimeError('cannot inspect token after NtSetInformationToken') from query_error
                # STATUS_INVALID_INFO_CLASS for the documented read-only class.
                rejected = (status & 0xffffffff) == 0xc0000003
                return {'outcome': ('read-only-rejection' if rejected and after == before
                                    else 'unexpected-token-change-result'),
                        'ntstatus': '0x%08x' % (status & 0xffffffff), 'token': after}
        run('set-token-user-to-system', change_user,
            ('read-only-rejection', 'access-denied'))
    finally:
        api.close(source)

    # A failed startup control is diagnostic failure, not denial. run() records
    # it without suppressing the independent WMI attempt below.
    run('powershell-startup', api.powershell_startup, ('ready',))

    for name, launcher in [('shell-execute', api.shell_launch),
                           ('wmi-create', api.wmi_launch)]:
        report = Path(manifest['workspace']) / (name + '-' + uuid.uuid4().hex + '.json')
        # Runtime/protocol errors are caught for diagnostics but never promoted
        # to access denial. Continue to the other broker after an unknown error.
        try:
            run(name, lambda: launcher(sys.executable, script, manifest_path,
                                       report, owner, package),
                ('access-denied', 'contained'))
        except Exception as error:
            result = {'operation': name, 'outcome': 'unexpected-error',
                      'exception': type(error).__name__, 'message': str(error),
                      'passed': False}
            results.append(result)
            print(json.dumps(result), flush=True)
    return all(result['passed'] for result in results)
