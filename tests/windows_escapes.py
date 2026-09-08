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
import sys
import time
import uuid


HANDLE = C.c_void_p
ULONG = C.c_uint32
LONG = C.c_int32


class Luids(C.Structure):
    _fields_ = [('low', ULONG), ('high', LONG)]


class Guids(C.Structure):
    _fields_ = [('data1', ULONG), ('data2', W.WORD),
                ('data3', W.WORD), ('data4', C.c_ubyte * 8)]


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
        self.read_memory = bind(native.kernel, 'ReadProcessMemory', W.BOOL,
                                HANDLE, HANDLE, HANDLE, C.c_size_t,
                                C.POINTER(C.c_size_t))
        self.duplicate_handle = bind(native.kernel, 'DuplicateHandle', W.BOOL,
                                     HANDLE, HANDLE, HANDLE, C.POINTER(HANDLE),
                                     ULONG, W.BOOL, ULONG)
        # SetThreadToken needs TOKEN_IMPERSONATE on the supplied token.
        # CreateProcessAsUserW (advapi32) and CreateProcessWithTokenW have
        # DIFFERENT shapes: 11 args including inheritable-handles, versus 9
        # args with dwLogonFlags. Their absence of worker privileges
        # (SeAssignPrimaryTokenPrivilege / SeImpersonatePrivilege) is the
        # expected denial stage.
        self.set_thread_token = bind(native.advapi, 'SetThreadToken', W.BOOL,
                                     HANDLE, HANDLE)
        self.open_thread_token = bind(native.advapi, 'OpenThreadToken', W.BOOL,
                                      HANDLE, ULONG, W.BOOL, C.POINTER(HANDLE))
        self.revert = bind(native.advapi, 'RevertToSelf', W.BOOL)
        self.current_thread = bind(native.kernel, 'GetCurrentThread', HANDLE)
        self.create_as_user = bind(
            native.advapi, 'CreateProcessAsUserW', W.BOOL, HANDLE, W.LPCWSTR,
            W.LPWSTR, HANDLE, HANDLE, W.BOOL, ULONG, HANDLE, W.LPCWSTR,
            C.POINTER(startup_type), C.POINTER(process_type))
        self.create_with_token = bind(
            native.advapi, 'CreateProcessWithTokenW', W.BOOL, HANDLE, ULONG,
            W.LPCWSTR, W.LPWSTR, ULONG, HANDLE, W.LPCWSTR,
            C.POINTER(startup_type), C.POINTER(process_type))

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
        # Cleanup failure is never evidence that the attempted access was denied.
        if not self.n.close(handle):
            raise RuntimeError('escape probe CloseHandle failed: %s' % C.get_last_error())

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

    def peer_memory(self, target):
        print(json.dumps({'probe': 'peer-memory', 'phase': 'open-process'}), flush=True)
        process = self.n.check(self.n.open_process(0x10, False, target['pid']))
        try:
            # Acquiring this protected right already violates the fixture's
            # expected boundary. Later read errors must not become denial passes.
            buffer = C.create_string_buffer(len(b'FAKE-PEER-MEMORY'))
            size = C.c_size_t()
            success = self.read_memory(process, target['address'], buffer,
                                       len(buffer), C.byref(size))
            return {'outcome': 'process-access-granted', 'read_succeeded': bool(success),
                    'bytes_read': size.value,
                    'marker_matches': buffer.raw == b'FAKE-PEER-MEMORY',
                    'winerror': 0 if success else C.get_last_error()}
        except OSError as error:
            raise RuntimeError('peer process opened but memory probe failed') from error
        finally:
            self.close(process)

    def peer_handle(self, target):
        print(json.dumps({'probe': 'peer-handle', 'phase': 'open-process'}), flush=True)
        process = self.n.check(self.n.open_process(0x40, False, target['pid']))
        handle = HANDLE()
        try:
            success = self.duplicate_handle(process, target['file_handle'],
                                            self.n.current_process(), C.byref(handle),
                                            0, False, 2)  # DUPLICATE_SAME_ACCESS
            return {'outcome': 'process-access-granted',
                    'duplicated': bool(success),
                    'winerror': 0 if success else C.get_last_error()}
        except OSError as error:
            raise RuntimeError('peer process opened but handle probe failed') from error
        finally:
            with ExitStack() as cleanup:
                cleanup.callback(self.close, process)
                if handle.value:
                    cleanup.callback(self.close, handle)

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

    def launch_as_user(self, primary, executable, command, startup, child):
        return self.create_as_user(primary, executable, command, None, None,
                                   False, 0x80000 | 4, None, None,
                                   C.byref(startup), C.byref(child))

    def launch_with_token(self, primary, executable, command, startup, child):
        # Nine-argument contract; dwLogonFlags 0 loads no profile for the
        # bounded suspended witness.
        return self.create_with_token(primary, 0, executable, command,
                                      0x80000 | 4, None, None,
                                      C.byref(startup), C.byref(child))

    def token_launch(self, launcher, rights, executable, owner, package):
        """Launch through a token-based creator using a fresh primary duplicate.

        Duplication and creation denial are reported at their named stage and
        only for documented denial errors; other failures are probe errors.
        An unexpected success stays suspended until its token is inspected, so
        no uninspected code runs under a possibly widened identity.
        """
        print(json.dumps({'probe': 'token-launch', 'phase': 'duplicate'}),
              flush=True)
        source = self.token(self.n.current_process(), 8 | 2)
        try:
            primary = HANDLE()
            self.n.check(self.duplicate(source, rights, None, 2, 1,
                                        C.byref(primary)))
        finally:
            self.close(source)
        try:
            startup = self.startup_type()
            startup.startup.cb = C.sizeof(startup)
            child = self.process_type()
            command = C.create_unicode_buffer(subprocess.list2cmdline(
                [executable, '-I', '-c', 'pass']))
            print(json.dumps({'probe': 'token-launch', 'phase': 'create'}),
                  flush=True)
            success = launcher(primary, executable, command, startup, child)
            if not success:
                error = C.get_last_error()
                if error in (5, 1314):  # ACCESS_DENIED, PRIVILEGE_NOT_HELD
                    return {'outcome': 'access-denied', 'phase': 'create',
                            'winerror': error}
                if error == 87:  # ERROR_INVALID_PARAMETER: native cea20fa
                    # showed the DuplicateTokenEx primary does not meet this
                    # API's undocumented logon-session requirement. Inconclusive
                    # for this route; never denial and never an escape claim.
                    return {'outcome': 'parameter-rejected-inconclusive',
                            'phase': 'create', 'winerror': error}
                # Invalid parameters and unknown failures are probe errors,
                # never containment evidence.
                return {'outcome': 'unexpected-launch-result',
                        'phase': 'create', 'winerror': error}
            self.created += 1  # Conservatively treat success as a launch.
            with ExitStack() as cleanup:
                cleanup.callback(self.close, child.thread)
                cleanup.callback(self.retire, child.process)
                token = self.token(child.process, 8)
                cleanup.callback(self.close, token)
                snapshot = self.snapshot(token)
                return {'outcome': ('contained' if identity_matches(
                    snapshot, owner, package) else 'escaped'), 'token': snapshot}
        finally:
            self.close(primary)

    def task_invoke(self, name):
        """Invoke a prearranged, test-owned scheduled task from the worker.

        The scheduler service, not this process, would launch the witness, so
        a rejected invocation is the denial signal; whether a launch actually
        happened is decided by the broker from the witness report, which this
        worker cannot read or forge (protected root).
        """
        schtasks = Path(os.environ['SystemRoot']) / 'System32/schtasks.exe'
        print(json.dumps({'probe': 'scheduled-task', 'phase': 'invoke'}), flush=True)
        try:
            result = subprocess.run([str(schtasks), '/Run', '/TN', name],
                                    capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('scheduled-task wrapper timed out') from error
        print(json.dumps({'probe': 'scheduled-task',
                          'wrapper_exit': result.returncode,
                          'stdout': result.stdout, 'stderr': result.stderr}),
              flush=True)
        if result.returncode:
            return {'outcome': 'invoke-rejected', 'exit': result.returncode,
                    'output': (result.stderr or result.stdout).strip()}
        # Accepted launch requests still need broker-side witness evidence;
        # never classify acceptance as containment or denial.
        return {'outcome': 'invoke-accepted'}

    def com_launch(self, clsid):
        """Activate a registered local COM server from the worker.

        The wrapper is bounded at 20 seconds because a launched server that
        never registers a class object hangs the activation handshake; a
        timeout fails the scenario, and whether a launch actually happened is
        decided by the broker from the witness report, which this worker
        cannot read or forge (protected root).
        """
        print(json.dumps({'probe': 'com', 'phase': 'activate'}), flush=True)
        command = [sys.executable, '-I', '-u', os.path.abspath(__file__),
                   '--com-activate', clsid]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=20)
        except subprocess.TimeoutExpired as error:
            def text(value):
                if isinstance(value, bytes):
                    return value.decode(errors='replace')
                return value
            print(json.dumps({'probe': 'com', 'timed_out': True,
                              'stdout': text(error.stdout),
                              'stderr': text(error.stderr)}), flush=True)
            raise RuntimeError('com activation wrapper timed out') from error
        print(json.dumps({'probe': 'com', 'wrapper_exit': result.returncode,
                          'stdout': result.stdout, 'stderr': result.stderr}),
              flush=True)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if result.returncode or not lines:
            raise RuntimeError('com activation wrapper failed')
        reply = json.loads(lines[-1])
        status = reply['hresult'] & 0xffffffff
        if status == 0x80070005:  # E_ACCESSDENIED
            return {'outcome': 'activation-denied', 'phase': 'co-create',
                    'hresult': status}
        if status == 0x80040154:  # REGDB_E_CLASSNOTREG
            return {'outcome': 'class-not-registered', 'phase': 'co-create',
                    'hresult': status}
        # Success or unknown failures never count as containment; the broker
        # witness decides violations.
        return {'outcome': 'unexpected-activation-result', 'phase': 'co-create',
                'hresult': status}

    def impersonation_launch(self, script, manifest):
        """Run the thread-token experiment in a disposable contained child.

        Installing a thread token contaminates the calling thread, so this
        never runs in the probe worker itself; the child inherits the same
        AppContainer identity and reports its own classification.
        """
        try:
            result = subprocess.run(
                [sys.executable, '-I', '-u', script, '--token-impersonation',
                 str(manifest)], capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('impersonation child timed out') from error
        print(json.dumps({'impersonation_child_exit': result.returncode,
                          'stdout': result.stdout, 'stderr': result.stderr}),
              flush=True)
        if result.returncode:
            raise RuntimeError('impersonation child failed')
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError('impersonation child produced no report')
        return json.loads(lines[-1])


def guid_from_text(text):
    """GUID structure filled from canonical text.

    Textual groups are big-endian hex of the field VALUES while the struct
    stores those values in little-endian memory, so each group parses
    big-endian and the struct performs the encoding. A little-endian parse
    byte-swaps the first three fields and activates a different CLSID than
    the one registered (defect at 9757654: CLASSNOTREG proved nothing).
    """
    value = Guids()
    raw = bytes.fromhex(text.strip('{}').replace('-', ''))
    value.data1 = int.from_bytes(raw[0:4], 'big')
    value.data2 = int.from_bytes(raw[4:6], 'big')
    value.data3 = int.from_bytes(raw[6:8], 'big')
    C.memmove(value.data4, raw[8:16], 8)
    return value


def com_activation(clsid):
    """Local-server COM activation wrapper; prints one JSON hresult reply.

    Runs in a disposable process so a hung handshake can be killed without
    contaminating the caller. The launched witness never registers a class
    object, so a successful OS launch typically ends in a handshake failure
    or hang rather than S_OK; the witness report is the real evidence.
    """
    from ctypes import wintypes as W

    ole = C.WinDLL('ole32')
    ole.CoInitializeEx.restype = C.c_long  # HRESULT
    ole.CoCreateInstance.restype = C.c_long
    ole.CoCreateInstance.argtypes = [C.POINTER(Guids), C.c_void_p, W.DWORD,
                                     C.POINTER(Guids), C.POINTER(C.c_void_p)]
    hr = ole.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
    if hr != 0:
        print(json.dumps({'hresult': hr, 'phase': 'co-initialize'}), flush=True)
        return
    instance = C.c_void_p()
    hr = ole.CoCreateInstance(
        C.byref(guid_from_text(clsid)), None, 0x4,  # CLSCTX_LOCAL_SERVER
        C.byref(guid_from_text('{00000000-0000-0000-C000-000000000046}')),
        C.byref(instance))
    print(json.dumps({'hresult': hr, 'phase': 'co-create'}), flush=True)


def peer(native, secret, report):
    """Fixed unrestricted target; no worker-controlled commands or IPC service."""
    marker = C.create_string_buffer(b'FAKE-PEER-MEMORY')
    handle = native.open(secret)
    try:
        payload = {'pid': os.getpid(), 'address': C.addressof(marker),
                   'file_handle': handle}
        temporary = report.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload))
        os.replace(temporary, report)
        # The owning fixture also retains the process and terminates it in
        # finally; this lifetime bound is not the observer's cleanup evidence.
        time.sleep(180)
    finally:
        native.check(native.close(handle))


def impersonation(api, manifest, classify):
    """Thread-token experiment for the disposable child; never the worker.

    SetThreadToken/RevertToSelf contracts:
    https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadtoken
    Microsoft directs shutting the process down if RevertToSelf fails; this
    child exits nonzero so the parent reports a protocol error, never denial.
    """
    source = api.token(api.n.current_process(), 8 | 2)
    try:
        duplicate = HANDLE()
        # SecurityImpersonation level, impersonation type, TOKEN_QUERY|IMPERSONATE.
        api.n.check(api.duplicate(source, 8 | 4, None, 2, 2, C.byref(duplicate)))
        try:
            before = api.snapshot(duplicate)
            baseline = classify('impersonation-baseline-read',
                                lambda: (Path(manifest['secret']) /
                                         'read').read_bytes())
            if baseline['outcome'] != 'access-denied':
                raise RuntimeError('pre-impersonation credential read not denied')
            C.set_last_error(0)
            success = api.set_thread_token(None, duplicate)
            error = C.get_last_error()
            if not success:
                if error == 5:  # ERROR_ACCESS_DENIED
                    return {'outcome': 'access-denied',
                            'phase': 'set-thread-token', 'winerror': error}
                # Parameter/handle failures are probe errors, never denial.
                return {'outcome': 'unexpected-set-thread-token-result',
                        'winerror': error}
            thread = HANDLE()
            try:
                # NULL is not a documented current-thread selector here; use
                # the GetCurrentThread pseudo-handle.
                if not api.open_thread_token(api.current_thread(), 8, False,
                                             C.byref(thread)):
                    raise RuntimeError('impersonation installed but thread '
                                       'token cannot be opened: %s' %
                                       C.get_last_error())
                current = api.snapshot(thread)
                preserved = all(current[key] == before[key]
                                for key in ('user', 'app', 'package'))
                under = classify(
                    'impersonated-credential-read',
                    lambda: (Path(manifest['secret']) / 'read').read_bytes())
            finally:
                # This mode runs only in a disposable child. On restore failure
                # do not inspect, retry, print, unwind cleanup, or run atexit
                # callbacks under an identity we could not restore. Exit 70 is
                # reported by the parent as a hard probe failure, never denial.
                if not api.revert():
                    os._exit(70)
                if thread.value:
                    api.close(thread)
            residual = HANDLE()
            leftover = api.open_thread_token(api.current_thread(), 8,
                                             False, C.byref(residual))
            residual_error = 0 if leftover else C.get_last_error()
            if leftover:
                api.close(residual)
            clean = not leftover and residual_error == 1008
            contained = (preserved and
                         under['outcome'] == 'access-denied' and clean)
            return {'outcome': ('impersonated-contained' if contained
                                else 'unexpected-impersonation-result'),
                    'identity_preserved': preserved,
                    'access_under_impersonation': under,
                    'residual_thread_token': not clean}
        finally:
            api.close(duplicate)
    finally:
        api.close(source)


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

    target = manifest['peer']
    run('peer-memory-read', lambda: api.peer_memory(target))
    run('peer-handle-duplication', lambda: api.peer_handle(target))
    for name, rights in [('memory-write', 0x20 | 8), ('dacl-write', 0x40000)]:
        def peer_access(rights=rights):
            process = api.n.check(api.n.open_process(rights, False, target['pid']))
            api.close(process)
        run('peer-' + name, peer_access)

    def peer_parent():
        print(json.dumps({'probe': 'alternate-peer-parent', 'phase': 'open-parent'}),
              flush=True)
        parent = api.n.check(api.n.open_process(0x80, False, target['pid']))
        try:
            return api.alternate_parent(parent, sys.executable, owner, package)
        finally:
            api.close(parent)
    run('alternate-peer-parent', peer_parent, ('access-denied', 'contained'))

    def peer_token():
        print(json.dumps({'probe': 'peer-token', 'phase': 'open-process'}), flush=True)
        process = api.n.check(api.n.open_process(0x1000, False, target['pid']))
        try:
            print(json.dumps({'probe': 'peer-token', 'phase': 'open-token'}), flush=True)
            token = api.token(process, 8 | 2 | 1)
            try:
                snapshot = api.snapshot(token)
                return {'outcome': 'token-acquired', 'token': snapshot}
            except OSError as error:
                raise RuntimeError('peer token acquired but inspection failed') from error
            finally:
                api.close(token)
        finally:
            api.close(process)
    run('peer-token-acquisition', peer_token)

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

        def enable_working_set():
            privilege = 'SeIncreaseWorkingSetPrivilege'
            if before['privileges'].get(privilege) != 0:
                raise RuntimeError('present-disabled privilege control unavailable')
            clone = api.clone(source, 8 | 0x20)
            try:
                request = TokenPrivileges()
                request.count = 1
                api.n.check(api.lookup_value(None, privilege,
                                             C.byref(request.entries[0].luid)))
                request.entries[0].attributes = 2
                C.set_last_error(0)
                success = api.adjust(clone, False, C.byref(request), 0, None, None)
                error = C.get_last_error()
                if not success:
                    raise C.WinError(error)
                try:
                    after = api.snapshot(clone)
                except OSError as query_error:
                    raise RuntimeError('cannot inspect adjusted working-set token') from query_error
                # Enabling this present privilege is not itself an escape. No
                # duplicate is installed; identity, package and groups must stay.
                expected = dict(before, privileges=dict(before['privileges']))
                expected['privileges'][privilege] = after['privileges'].get(privilege)
                attrs = after['privileges'].get(privilege)
                valid = (after == expected and
                         ((error == 0 and attrs in (2, 10)) or
                          (error == 1300 and attrs == 0)))
                return {'outcome': 'contained-adjustment' if valid else 'unexpected-adjustment',
                        'winerror': error, 'before': before, 'after': after}
            finally:
                api.close(clone)
        run('enable-present-working-set-privilege', enable_working_set,
            ('contained-adjustment', 'access-denied'))

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

    # Thread-token installation runs only in a disposable contained child.
    run('thread-token-impersonation',
        lambda: api.impersonation_launch(script, manifest_path),
        ('impersonated-contained', 'access-denied'))
    primary_rights = 8 | 2 | 1  # TOKEN_QUERY|DUPLICATE|ASSIGN_PRIMARY
    for name, launcher in [('create-process-as-user', api.launch_as_user),
                           ('create-process-with-token', api.launch_with_token)]:
        def token_launcher(launcher=launcher):
            return api.token_launch(launcher, primary_rights, sys.executable,
                                    owner, package)
        run(name, token_launcher, ('access-denied', 'contained'))

    # Prearranged own-account task registered by the broker. Only a rejected
    # invocation passes; acceptance is resolved by the broker's witness check.
    if manifest.get('scheduled_task'):
        run('scheduled-task',
            lambda: api.task_invoke(manifest['scheduled_task']),
            ('invoke-rejected',))

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
    com = manifest.get('com') or {}
    if com.get('clsid'):
        run('com-activation', lambda: api.com_launch(com['clsid']),
            ('activation-denied', 'class-not-registered'))

    return all(result['passed'] for result in results)


if __name__ == '__main__' and len(sys.argv) == 3 \
        and sys.argv[1] == '--com-activate':
    com_activation(sys.argv[2])
