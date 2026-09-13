"""Native Windows experiments, independent of Loki's Unix-only imports.

Only disposable data is used. These probe OS operations, not the credential
protocol, ACL security, or power-loss durability. Run directly or via unittest.
"""

import asyncio
import ctypes as C
from ctypes import wintypes as W
import errno
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


# This file is copied and run on its own by the standard-user probe, so it must
# not import ``loki_agent``: the staged copy has no package beside it.  The
# production primitives are exercised in test_windows_credential_primitives.py,
# which runs in the suite where the package is importable.


# Exact Windows ABI types (not ctypes.c_long, which is 64 bits on Unix).
ULONG = C.c_uint32
LONG = C.c_int32
HANDLE = C.c_void_p
DELETE = 0x10000
READ_CONTROL = 0x20000
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
SHARE_ALL = 7
BACKUP = 0x02000000
OPEN_REPARSE = 0x00200000


class UnicodeStrings(C.Structure):
    _fields_ = [('Length', C.c_uint16), ('MaximumLength', C.c_uint16),
                ('Buffer', C.c_void_p)]


class ObjectAttributes(C.Structure):
    _fields_ = [('Length', ULONG), ('RootDirectory', HANDLE),
                ('ObjectName', C.POINTER(UnicodeStrings)),
                ('Attributes', ULONG), ('SecurityDescriptor', HANDLE),
                ('SecurityQualityOfService', HANDLE)]


class IoStatuses(C.Structure):
    # The first member is a union of NTSTATUS and a pointer.
    _fields_ = [('Status', C.c_size_t), ('Information', C.c_size_t)]


class RenameInfos(C.Structure):
    _fields_ = [('ReplaceIfExists', C.c_ubyte), ('RootDirectory', HANDLE),
                ('FileNameLength', ULONG), ('FileName', C.c_uint16 * 1)]


class SidAttributes(C.Structure):
    _fields_ = [('Sid', HANDLE), ('Attributes', ULONG)]


class TokenGroups(C.Structure):
    _fields_ = [('GroupCount', ULONG), ('Groups', SidAttributes * 1)]


class Acls(C.Structure):
    """ACL header that precedes the ACE array in a security descriptor."""

    _fields_ = [('revision', C.c_ubyte), ('sbz1', C.c_ubyte),
                ('size', C.c_uint16), ('ace_count', C.c_uint16),
                ('sbz2', C.c_uint16)]


class AceHeaders(C.Structure):
    _fields_ = [('type', C.c_ubyte), ('flags', C.c_ubyte),
                ('size', C.c_uint16)]


SE_FILE_OBJECT = 1
LABEL_SECURITY_INFORMATION = 0x10
SYSTEM_MANDATORY_LABEL_ACE_TYPE = 0x11

# Probe-only declarations for the transport/scrub/identity questions.  Kept
# local because this file is staged without the package.
GENERIC_WRITE = 0x40000000
LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
FILE_ID_INFO_CLASS = 18
FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400


class Overlapped(C.Structure):
    """OVERLAPPED: the offset a LockFileEx range starts at."""

    _fields_ = [('Internal', C.c_size_t), ('InternalHigh', C.c_size_t),
                ('Offset', ULONG), ('OffsetHigh', ULONG), ('hEvent', HANDLE)]


class FileIdInfo(C.Structure):
    """FILE_ID_INFO: volume serial plus the 128-bit file id."""

    _fields_ = [('VolumeSerialNumber', C.c_ulonglong),
                ('FileId', C.c_ubyte * 16)]


PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x00020016
EXTENDED_STARTUPINFO_PRESENT = 0x00080000


class Coord(C.Structure):
    _fields_ = [('x', C.c_int16), ('y', C.c_int16)]


class ProbeStartupInfo(C.Structure):
    _fields_ = [('cb', ULONG), ('reserved', W.LPWSTR), ('desktop', W.LPWSTR),
                ('title', W.LPWSTR), ('x', ULONG), ('y', ULONG),
                ('xsize', ULONG), ('ysize', ULONG), ('xchars', ULONG),
                ('ychars', ULONG), ('fill', ULONG), ('flags', ULONG),
                ('show', C.c_uint16), ('reserved_size', C.c_uint16),
                ('reserved_bytes', HANDLE), ('stdin', HANDLE),
                ('stdout', HANDLE), ('stderr', HANDLE)]


class SharedStartupInfo(C.Structure):
    _fields_ = [('startup', ProbeStartupInfo), ('attributes', HANDLE)]


class ProbeProcessInfo(C.Structure):
    _fields_ = [('process', HANDLE), ('thread', HANDLE),
                ('pid', ULONG), ('tid', ULONG)]


def conpty_probe(root):
    """Attach a child to a pseudoconsole and record what comes back.

    Decides whether the terminal-attach seam is reachable with the documented
    synchronous ConPTY calls as a standard user: create the session, launch a
    child on it, read the child's output through the pipe, and close.
    """
    kernel = C.WinDLL('kernel32', use_last_error=True)
    create_pipe = kernel.CreatePipe
    create_pipe.argtypes = [C.POINTER(HANDLE), C.POINTER(HANDLE), C.c_void_p,
                            ULONG]
    create_pipe.restype = W.BOOL
    create_pseudo = kernel.CreatePseudoConsole
    create_pseudo.argtypes = [Coord, HANDLE, HANDLE, ULONG, C.POINTER(HANDLE)]
    create_pseudo.restype = C.c_long
    close_pseudo = kernel.ClosePseudoConsole
    close_pseudo.argtypes = [HANDLE]
    close_pseudo.restype = None
    initialize = kernel.InitializeProcThreadAttributeList
    initialize.argtypes = [C.c_void_p, ULONG, ULONG, C.POINTER(C.c_size_t)]
    initialize.restype = W.BOOL
    update = kernel.UpdateProcThreadAttribute
    update.argtypes = [C.c_void_p, ULONG, C.c_size_t, C.c_void_p, C.c_size_t,
                       C.c_void_p, C.c_void_p]
    update.restype = W.BOOL
    delete = kernel.DeleteProcThreadAttributeList
    delete.argtypes = [C.c_void_p]
    delete.restype = None
    create_process = kernel.CreateProcessW
    create_process.argtypes = [W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p,
                               W.BOOL, ULONG, C.c_void_p, W.LPCWSTR,
                               C.POINTER(ProbeStartupInfo),
                               C.POINTER(ProbeProcessInfo)]
    create_process.restype = W.BOOL
    read_file = kernel.ReadFile
    read_file.argtypes = [HANDLE, C.c_void_p, ULONG, C.POINTER(ULONG),
                          C.c_void_p]
    read_file.restype = W.BOOL
    peek = kernel.PeekNamedPipe
    peek.argtypes = [HANDLE, C.c_void_p, ULONG, C.POINTER(ULONG),
                     C.POINTER(ULONG), C.POINTER(ULONG)]
    peek.restype = W.BOOL

    def pair():
        read, write = HANDLE(), HANDLE()
        if not create_pipe(C.byref(read), C.byref(write), None, 0):
            raise C.WinError(C.get_last_error())
        return read, write

    input_read, input_write = pair()
    output_read, output_write = pair()
    hpc = HANDLE()
    record = {'probe': 'conpty'}
    status = create_pseudo(Coord(80, 24), input_read, output_write, 0,
                           C.byref(hpc))
    if status < 0:
        for handle in (input_read, input_write, output_read, output_write):
            kernel.CloseHandle(handle)
        record.update({'created': False, 'hresult': '0x%08x' % (status & 0xffffffff)})
        print(json.dumps(record), flush=True)
        return 0
    record['created'] = True
    attributes = None
    process = ProbeProcessInfo()
    try:
        size = C.c_size_t()
        # The sizing call is documented to fail with ERROR_INSUFFICIENT_BUFFER
        # while filling in the required size; only a wrong error is a defect.
        initialize(None, 1, 0, C.byref(size))
        if C.get_last_error() != 122 or not size.value:
            raise C.WinError(C.get_last_error())
        storage = C.create_string_buffer(size.value)
        attributes = C.cast(storage, C.c_void_p)
        if not initialize(attributes, 1, 0, C.byref(size)):
            raise C.WinError(C.get_last_error())
        if not update(attributes, 0, PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                      C.cast(hpc, C.c_void_p), C.sizeof(hpc), None, None):
            raise C.WinError(C.get_last_error())
        startup = SharedStartupInfo()
        startup.startup.cb = C.sizeof(SharedStartupInfo)
        startup.attributes = attributes
        command = '"%s" /d /c echo conpty-ok' % os.path.join(
            os.environ.get('SystemRoot', 'C:\\Windows'), 'System32',
            'cmd.exe')
        text = C.create_unicode_buffer(command)
        if not create_process(None, text, None, None, False,
                              EXTENDED_STARTUPINFO_PRESENT, None, str(root),
                              C.byref(startup.startup),
                              C.byref(process)):
            raise C.WinError(C.get_last_error())
        # The pseudoconsole keeps its copies; release ours so the channel can
        # detect a broken pipe when the child exits.
        kernel.CloseHandle(input_read)
        kernel.CloseHandle(output_write)
        output = bytearray()
        deadline = time.monotonic() + 8
        available, transferred = ULONG(), ULONG()
        # Keep draining after the child exits: the pseudoconsole flushes its
        # final frame when the session closes, and a poll that stops at process
        # exit loses it.
        while time.monotonic() < deadline and b'conpty-ok' not in output:
            if not peek(output_read, None, 0, None, C.byref(available), None):
                record['peek_failed'] = True
                break
            if available.value:
                buffer = C.create_string_buffer(available.value)
                if not read_file(output_read, buffer, available.value,
                                 C.byref(transferred), None):
                    record['read_failed'] = True
                    break
                output.extend(buffer.raw[:transferred.value])
            else:
                time.sleep(0.005)
        record['saw_marker'] = b'conpty-ok' in output
        record['output'] = output.decode('utf-8', 'replace')
        kernel.WaitForSingleObject(process.process, 5000)
        exit_code = ULONG()
        kernel.GetExitCodeProcess(process.process, C.byref(exit_code))
        record['exit_code'] = exit_code.value
    except OSError as error:
        record['error'] = str(error)
        record['winerror'] = getattr(error, 'winerror', None)
    finally:
        if process.process:
            kernel.CloseHandle(process.process)
        if process.thread:
            kernel.CloseHandle(process.thread)
        close_pseudo(hpc)
        if attributes is not None:
            delete(attributes)
        kernel.CloseHandle(input_write)
        kernel.CloseHandle(output_read)
    print(json.dumps(record), flush=True)
    return 0 if record.get('saw_marker') else 2


def mandatory_label_sid(sacl_address):
    """Return the address of the mandatory-label SID inside a SACL, or None.

    An object's integrity label is not in the DACL: it is a
    SYSTEM_MANDATORY_LABEL_ACE (type 0x11) in the SACL, whose SID follows the
    ACE header and the access mask.  Walking it by hand keeps those offsets
    visible, and the walk is regression-tested with a synthetic buffer -- a
    wrong offset would silently report "no label" on the platform that matters.
    """
    if not sacl_address:
        return None
    acl = Acls.from_address(sacl_address)
    ace_address = sacl_address + C.sizeof(Acls)
    for _ in range(acl.ace_count):
        ace = AceHeaders.from_address(ace_address)
        if ace.type == SYSTEM_MANDATORY_LABEL_ACE_TYPE:
            return ace_address + C.sizeof(AceHeaders) + C.sizeof(ULONG)
        if not ace.size:
            break
        ace_address += ace.size
    return None


def require_standard_user(details, expected_sid):
    """Reject elevated and filtered administrator tokens, not just elevation."""
    if details['user'] != expected_sid:
        raise RuntimeError('probe is not running as the requested account')
    if details['elevation'] != 0:
        raise RuntimeError('probe token is elevated')
    # Check every group, regardless of SE_GROUP_ENABLED or USE_FOR_DENY_ONLY.
    if any(group['sid'] == 'S-1-5-32-544' for group in details['groups']):
        raise RuntimeError('probe token contains the Administrators SID')


class NativeCalls:
    """Test-only declarations; no runtime adapter or compatibility fallback."""

    def __init__(self):
        self.kernel = C.WinDLL('kernel32', use_last_error=True)
        self.advapi = C.WinDLL('advapi32', use_last_error=True)
        self.ntdll = C.WinDLL('ntdll')
        self.create = self.bind(self.kernel, 'CreateFileW', HANDLE,
                                W.LPCWSTR, ULONG, ULONG, HANDLE, ULONG,
                                ULONG, HANDLE)
        self.close = self.bind(self.kernel, 'CloseHandle', W.BOOL, HANDLE)
        self.set_info = self.bind(
            self.kernel, 'SetFileInformationByHandle', W.BOOL,
            HANDLE, C.c_int, HANDLE, ULONG)
        self.ntcreate = self.bind(
            self.ntdll, 'NtCreateFile', LONG, C.POINTER(HANDLE), ULONG,
            C.POINTER(ObjectAttributes), C.POINTER(IoStatuses), HANDLE,
            ULONG, ULONG, ULONG, ULONG, HANDLE, ULONG)
        self.ntset = self.bind(
            self.ntdll, 'NtSetInformationFile', LONG, HANDLE,
            C.POINTER(IoStatuses), HANDLE, ULONG, C.c_int)
        self.get_security = self.bind(
            self.advapi, 'GetSecurityInfo', ULONG, HANDLE, C.c_int, ULONG,
            C.POINTER(HANDLE), HANDLE, HANDLE, HANDLE, C.POINTER(HANDLE))
        self.sid_string = self.bind(self.advapi, 'ConvertSidToStringSidW',
                                    W.BOOL, HANDLE, C.POINTER(HANDLE))
        self.local_free = self.bind(self.kernel, 'LocalFree', HANDLE, HANDLE)
        self.open_token = self.bind(self.advapi, 'OpenProcessToken', W.BOOL,
                                    HANDLE, ULONG, C.POINTER(HANDLE))
        self.current_process = self.bind(self.kernel, 'GetCurrentProcess',
                                         HANDLE)
        self.token_info = self.bind(self.advapi, 'GetTokenInformation',
                                    W.BOOL, HANDLE, C.c_int, HANDLE, ULONG,
                                    C.POINTER(ULONG))
        # Probe additions: environment block, byte-range lock, file identity.
        self.lock_file = self.bind(
            self.kernel, 'LockFileEx', W.BOOL, HANDLE, ULONG, ULONG, ULONG,
            ULONG, C.POINTER(Overlapped))
        self.unlock_file = self.bind(
            self.kernel, 'UnlockFileEx', W.BOOL, HANDLE, ULONG, ULONG, ULONG,
            C.POINTER(Overlapped))
        self.file_information = self.bind(
            self.kernel, 'GetFileInformationByHandleEx', W.BOOL, HANDLE,
            C.c_int, HANDLE, ULONG)
        self.get_environment = self.bind(
            self.kernel, 'GetEnvironmentStringsW', C.c_void_p)
        self.free_environment = self.bind(
            self.kernel, 'FreeEnvironmentStringsW', W.BOOL, C.c_void_p)

    @staticmethod
    def bind(dll, name, result, *args):
        call = getattr(dll, name)
        call.restype = result
        call.argtypes = args
        return call

    @staticmethod
    def check(result):
        if not result:
            raise C.WinError(C.get_last_error())
        return result

    def open(self, path, access=GENERIC_READ, share=SHARE_ALL, flags=0):
        handle = self.create(str(path), access, share, None, 3, flags, None)
        if handle == C.c_void_p(-1).value:
            raise C.WinError(C.get_last_error())
        return handle

    def relative(self, directory, name, options=0x40 | 0x200000):
        # FILE_NON_DIRECTORY_FILE | FILE_OPEN_REPARSE_POINT. Neither this
        # leaf flag nor RootDirectory forbids intermediate reparse traversal.
        encoded = name.encode('utf-16-le')
        text = C.create_string_buffer(encoded + b'\0\0')
        string = UnicodeStrings(len(encoded), len(encoded) + 2,
                                C.cast(text, HANDLE))
        attrs = ObjectAttributes(C.sizeof(ObjectAttributes), directory,
                                 C.pointer(string), 0x40, None, None)
        io = IoStatuses()
        result = HANDLE()
        # Synchronous handle for CRT binary reads; SYNCHRONIZE access is
        # required with FILE_SYNCHRONOUS_IO_NONALERT (0x20).
        status = self.ntcreate(C.byref(result), GENERIC_READ | 0x100000,
                               C.byref(attrs), C.byref(io), None, 0, SHARE_ALL,
                               1, options | 0x20, None, 0)
        if status < 0:
            raise OSError('NtCreateFile NTSTATUS=0x%08x' % (status & 0xffffffff))
        return result.value

    @staticmethod
    def rename_buffer(directory, name):
        encoded = name.encode('utf-16-le')
        # The Win32 documentation describes FileName as NUL-terminated even
        # though FileNameLength excludes the terminator. Provide both, with
        # room for structure padding and the complete UTF-16 string.
        buffer = C.create_string_buffer(C.sizeof(RenameInfos) + len(encoded)
                                        + 2)
        info = RenameInfos.from_buffer(buffer)
        info.ReplaceIfExists = 1
        info.RootDirectory = directory
        info.FileNameLength = len(encoded)
        C.memmove(C.addressof(buffer) + RenameInfos.FileName.offset,
                  encoded, len(encoded))
        return buffer

    def rename(self, source, directory, name, *, native=False):
        buffer = self.rename_buffer(directory, name)
        size = C.sizeof(buffer)
        diagnostic = {'operation': ('NtSetInformationFile' if native else
                                    'SetFileInformationByHandle'),
                      'root_relative': directory is not None,
                      'name': name, 'buffer_size': size,
                      'name_offset': RenameInfos.FileName.offset}
        if native:
            io = IoStatuses()
            status = self.ntset(source, C.byref(io), buffer, size, 10)
            diagnostic['ntstatus'] = '0x%08x' % (status & 0xffffffff)
            print(json.dumps(diagnostic), flush=True)
            if status < 0:
                raise OSError('NtSetInformationFile NTSTATUS=0x%08x' %
                              (status & 0xffffffff))
        else:
            result = self.set_info(source, 3, buffer, size)
            error = 0 if result else C.get_last_error()
            diagnostic['winerror'] = error
            print(json.dumps(diagnostic), flush=True)
            if not result:
                raise C.WinError(error)

    def delete(self, source):
        disposition = C.c_ubyte(1)
        self.check(self.set_info(source, 4, C.byref(disposition), 1))

    def sid(self, pointer):
        text = HANDLE()
        self.check(self.sid_string(pointer, C.byref(text)))
        try:
            return C.wstring_at(text)
        finally:
            self.local_free(text)

    def owner(self, handle):
        owner, descriptor = HANDLE(), HANDLE()
        error = self.get_security(handle, 1, 1, C.byref(owner), None, None,
                                  None, C.byref(descriptor))
        if error:
            raise C.WinError(error)
        try:
            return self.sid(owner)
        finally:
            self.local_free(descriptor)

    def mandatory_label(self, handle):
        """Return the object's mandatory integrity label as a SID string.

        Returns None when the object carries no label.  Distinguishing an
        integrity denial from a DACL denial needs this: both surface as
        ERROR_ACCESS_DENIED, so the levels are what tell them apart.
        """
        sacl, descriptor = HANDLE(), HANDLE()
        error = self.get_security(handle, SE_FILE_OBJECT,
                                  LABEL_SECURITY_INFORMATION, None, None,
                                  None, C.byref(sacl), C.byref(descriptor))
        if error:
            raise C.WinError(error)
        try:
            sid = mandatory_label_sid(sacl.value)
            return None if sid is None else self.sid(HANDLE(sid))
        finally:
            self.local_free(descriptor)

    def token_details(self, *, include_groups=False):
        token = HANDLE()
        self.check(self.open_token(self.current_process(), 8, C.byref(token)))
        try:
            result = {}
            queries = [('user', 1), ('default_owner', 4)]
            if include_groups:
                queries.append(('groups', 2))
            for label, kind in queries:
                size = ULONG()
                succeeded = self.token_info(token, kind, None, 0,
                                            C.byref(size))
                error = 0 if succeeded else C.get_last_error()
                print(json.dumps({'token_query': label, 'size': size.value,
                                  'succeeded': bool(succeeded),
                                  'winerror': error}), flush=True)
                if succeeded or error != 122 or not size.value:
                    raise OSError('unexpected token sizing result: %s, %s, %s'
                                  % (label, error, size.value))
                buffer = C.create_string_buffer(size.value)
                self.check(self.token_info(token, kind, buffer, size.value,
                                           C.byref(size)))
                if kind == 2:
                    count = ULONG.from_buffer(buffer).value
                    offset = TokenGroups.Groups.offset
                    if offset + count * C.sizeof(SidAttributes) > len(buffer):
                        raise OSError('TOKEN_GROUPS exceeds query buffer')
                    groups = (SidAttributes * count).from_buffer(buffer, offset)
                    result[label] = [{'sid': self.sid(group.Sid),
                                      'attributes': group.Attributes}
                                     for group in groups]
                else:
                    result[label] = self.sid(HANDLE.from_buffer(buffer))
            # TOKEN_ELEVATION is a single DWORD. A zero-length sizing query
            # can return ERROR_BAD_LENGTH rather than INSUFFICIENT_BUFFER.
            elevation = ULONG()
            size = ULONG()
            self.check(self.token_info(token, 20, C.byref(elevation),
                                       C.sizeof(elevation), C.byref(size)))
            result['elevation'] = elevation.value
            return result
        finally:
            self.check(self.close(token))


def child(mode, root, stage=None):
    """Subprocess checkpoints acknowledge completion, not durable storage."""
    root = Path(root)

    if mode == 'conpty':
        return conpty_probe(root)

    if mode == 'second-user-create':
        # Create the credential directory exactly as the storage does: a
        # private os.mkdir, whose protected DACL names only the owner, SYSTEM
        # and Administrators.  Another standard user must then be refused it.
        credentials = root / 'credentials'
        credentials.mkdir(mode=0o700)
        (credentials / 'tokens.json').write_text('{"secret": "probe"}\n')
        print(json.dumps({'created': str(credentials)}), flush=True)
        return 0

    if mode == 'second-user-read':
        # Run as a different standard user: every credential object must refuse
        # both listing the directory and reading the JSON.  The parent is listed
        # first as a positive control -- a refuse-everything process would
        # otherwise make the denials below vacuous.
        credentials = root / 'credentials'
        result = {}
        attempts = (
            ('parent', lambda: os.listdir(root)),
            ('directory', lambda: os.listdir(credentials)),
            ('file', lambda: (credentials / 'tokens.json').read_bytes()),
        )
        for label, attempt in attempts:
            try:
                attempt()
            except PermissionError:
                result[label] = 'denied'
            except OSError as error:
                result[label] = 'error'
                result[label + '_winerror'] = getattr(error, 'winerror', None)
            else:
                result[label] = 'granted'
        print(json.dumps(result), flush=True)
        return 0 if (result.get('parent') == 'granted'
                     and result.get('directory') == 'denied'
                     and result.get('file') == 'denied') else 2

    import msvcrt
    if mode == 'lock':
        with open(root / 'lock', 'r+b', buffering=0) as stream:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                print(json.dumps({'errno': error.errno}), flush=True)
                return 0 if stage == 'contended' and error.errno == errno.EACCES else 2
            if stage == 'exit':
                os._exit(0)  # No explicit unlock or Python cleanup.
            if stage == 'hold':
                (root / 'ready').write_text('locked')
                time.sleep(120)
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            return 0 if stage != 'contended' else 3

    def checkpoint(name):
        if name == stage:
            (root / 'ready').write_text(name)
            time.sleep(120)

    with open(root / 'temporary', 'xb') as stream:
        checkpoint('create')
        stream.write(b'{"generation": "new"}\n')
        checkpoint('write')
        stream.flush()
        checkpoint('flush')
        os.fsync(stream.fileno())
        checkpoint('fsync')
    checkpoint('close')
    os.replace(root / 'temporary', root / 'target')
    checkpoint('replace')
    return 0


class WindowsProbeMarshallingTests(unittest.TestCase):
    """Portable boundary regressions; mocks are not native Windows evidence."""

    def test_rename_buffer_has_utf16_terminator_outside_counted_name(self):
        for name in ('published', 'long-name-\U0001f600', 'C:\\dir\\published'):
            with self.subTest(name=name):
                buffer = NativeCalls.rename_buffer(123, name)
                info = RenameInfos.from_buffer(buffer)
                encoded = name.encode('utf-16-le')
                self.assertEqual(info.FileNameLength, len(encoded))
                self.assertEqual(info.RootDirectory, 123)
                self.assertEqual(info.ReplaceIfExists, 1)
                start = RenameInfos.FileName.offset
                self.assertEqual(buffer.raw[start:start + len(encoded) + 2],
                                 encoded + b'\0\0')

    # S-1-16-4096: revision 1, one subauthority, authority 16, level 0x1000.
    MANDATORY_LABEL_SID = bytes([1, 1, 0, 0, 0, 0, 0, 16]) + (
        4096).to_bytes(4, 'little')

    @staticmethod
    def mandatory_label_buffer(sid, *, prefix_aces=()):
        """Build a synthetic SACL whose last ACE carries ``sid``.

        ``prefix_aces`` are (type, payload) pairs placed before it, so the walk
        has to skip ACEs rather than assume the label comes first.
        """
        aces = list(prefix_aces) + [(SYSTEM_MANDATORY_LABEL_ACE_TYPE, sid)]
        size = C.sizeof(Acls) + sum(
            C.sizeof(AceHeaders) + C.sizeof(ULONG) + len(payload)
            for _type, payload in aces)
        buffer = C.create_string_buffer(size)
        acl = Acls.from_buffer(buffer)
        acl.revision = 2
        acl.size = size
        acl.ace_count = len(aces)
        address = C.addressof(buffer) + C.sizeof(Acls)
        for ace_type, payload in aces:
            ace = AceHeaders.from_address(address)
            ace.type = ace_type
            ace.size = C.sizeof(AceHeaders) + C.sizeof(ULONG) + len(payload)
            C.memmove(address + C.sizeof(AceHeaders) + C.sizeof(ULONG),
                      payload, len(payload))
            address += ace.size
        return buffer

    def test_mandatory_label_walk_finds_the_sid_after_header_and_mask(self):
        buffer = self.mandatory_label_buffer(self.MANDATORY_LABEL_SID)

        found = mandatory_label_sid(C.addressof(buffer))

        self.assertEqual(
            C.string_at(found, len(self.MANDATORY_LABEL_SID)),
            self.MANDATORY_LABEL_SID)

    def test_mandatory_label_walk_skips_other_aces(self):
        buffer = self.mandatory_label_buffer(
            self.MANDATORY_LABEL_SID, prefix_aces=[(0x00, b'\x01' * 12)])

        found = mandatory_label_sid(C.addressof(buffer))

        self.assertEqual(
            C.string_at(found, len(self.MANDATORY_LABEL_SID)),
            self.MANDATORY_LABEL_SID)

    def test_mandatory_label_walk_returns_none_when_absent(self):
        self.assertIsNone(mandatory_label_sid(0))
        buffer = self.mandatory_label_buffer(
            self.MANDATORY_LABEL_SID, prefix_aces=[(0x00, b'\x01' * 12)])
        Acls.from_buffer(buffer).ace_count = 1  # count stops before the label

        self.assertIsNone(mandatory_label_sid(C.addressof(buffer)))

    def exercise_token_queries(self, *, fail_elevation=False,
                               include_groups=False):
        native = NativeCalls.__new__(NativeCalls)
        native.current_process = lambda: 1
        native.close = mock.Mock(return_value=1)
        native.sid = lambda pointer: 'SID-%s' % (
            pointer.value if isinstance(pointer, HANDLE) else pointer)
        queries = []

        def open_token(process, access, result):
            C.cast(result, C.POINTER(HANDLE)).contents.value = 42
            return 1

        def token_info(token, kind, buffer, length, returned):
            queries.append((kind, buffer is None))
            size = C.cast(returned, C.POINTER(ULONG)).contents
            if kind == 20:
                self.assertIsNotNone(buffer, 'fixed-size query must not probe')
                self.assertEqual(length, 4)
                if fail_elevation:
                    raise OSError('elevation query failed')
                C.cast(buffer, C.POINTER(ULONG)).contents.value = 1
                size.value = 4
                return 1
            if kind == 2:
                size.value = TokenGroups.Groups.offset + C.sizeof(SidAttributes)
                if buffer is None:
                    return 0
                ULONG.from_buffer(buffer).value = 1
                group = SidAttributes.from_buffer(buffer,
                                                  TokenGroups.Groups.offset)
                group.Sid = 200
                group.Attributes = 16  # deny-only must remain visible
                return 1
            size.value = C.sizeof(HANDLE) * 2
            if buffer is None:
                return 0  # mock get_last_error supplies INSUFFICIENT_BUFFER
            C.cast(buffer, C.POINTER(HANDLE)).contents.value = kind
            return 1

        native.open_token = open_token
        native.token_info = token_info
        with mock.patch.object(C, 'get_last_error', return_value=122,
                               create=True):
            if fail_elevation:
                with self.assertRaisesRegex(OSError, 'elevation query failed'):
                    native.token_details()
            else:
                expected = {'user': 'SID-1', 'default_owner': 'SID-4',
                            'elevation': 1}
                if include_groups:
                    expected['groups'] = [{'sid': 'SID-200', 'attributes': 16}]
                self.assertEqual(native.token_details(
                    include_groups=include_groups), expected)
        native.close.assert_called_once()
        self.assertEqual(native.close.call_args.args[0].value, 42)
        expected_queries = [(1, True), (1, False), (4, True), (4, False)]
        if include_groups:
            expected_queries.extend([(2, True), (2, False)])
        self.assertEqual(queries, expected_queries + [(20, False)])

    def test_token_groups_preserve_deny_only_membership(self):
        self.exercise_token_queries(include_groups=True)

    def test_standard_user_guard_accepts_expected_unelevated_user(self):
        require_standard_user({'user': 'expected', 'elevation': 0,
                               'groups': [{'sid': 'S-1-5-32-545',
                                           'attributes': 7}]}, 'expected')

    def test_standard_user_guard_rejects_wrong_user_and_elevation(self):
        for user, elevation in [('other', 0), ('expected', 1)]:
            with self.subTest(user=user, elevation=elevation):
                with self.assertRaises(RuntimeError):
                    require_standard_user({'user': user, 'elevation': elevation,
                                           'groups': []}, 'expected')

    def test_standard_user_guard_rejects_admin_sid_with_any_attributes(self):
        for attributes in (0, 4, 16):
            with self.subTest(attributes=attributes):
                with self.assertRaisesRegex(RuntimeError, 'Administrators'):
                    require_standard_user(
                        {'user': 'expected', 'elevation': 0,
                         'groups': [{'sid': 'S-1-5-32-544',
                                     'attributes': attributes}]}, 'expected')

    def test_standard_user_guard_requires_group_evidence(self):
        with self.assertRaises(KeyError):
            require_standard_user({'user': 'expected', 'elevation': 0},
                                  'expected')

    def test_token_elevation_uses_fixed_size_buffer(self):
        self.exercise_token_queries()

    def test_failed_token_elevation_still_closes_token(self):
        self.exercise_token_queries(fail_elevation=True)


@unittest.skipUnless(os.name == 'nt', 'requires native Windows Python')
class WindowsPrimitiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.native = NativeCalls()

    def handle(self, *args, **kwargs):
        handle = self.native.open(*args, **kwargs)
        self.addCleanup(lambda: self.native.check(self.native.close(handle)))
        return handle

    def read_handle(self, handle):
        import msvcrt
        # open_osfhandle transfers ownership; only the fd/file closes it.
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            self.native.close(handle)
            raise
        with os.fdopen(fd, 'rb') as stream:
            return stream.read()

    def command(self, mode, stage):
        return [sys.executable, str(Path(__file__).resolve()),
                '--child', mode, str(self.root), stage]

    def run_child(self, mode, stage):
        result = subprocess.run(self.command(mode, stage), timeout=15,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def stop_at(self, mode, stage):
        ready = self.root / 'ready'
        ready.unlink(missing_ok=True)
        process = subprocess.Popen(self.command(mode, stage),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        def stop():
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate(timeout=10)
            print(json.dumps({'stage': stage, 'exit': process.returncode,
                              'stdout': stdout.decode(errors='replace'),
                              'stderr': stderr.decode(errors='replace')}),
                  flush=True)

        self.addCleanup(stop)
        deadline = time.monotonic() + 15
        while not ready.exists():
            if process.poll() is not None:
                self.fail('child exited before checkpoint: %s' % stage)
            if time.monotonic() >= deadline:
                self.fail('child checkpoint timed out: %s' % stage)
            time.sleep(0.02)
        return process

    def test_environment_and_owner_observations(self):
        import socket
        # Recorded, not asserted: CPython gates AF_UNIX on HAVE_SYS_UN_H, which
        # no MSVC pyconfig.h defines, but a mingw/MSYS2 interpreter is built
        # with configure and its toolchain decides.  A transport that works on
        # one Windows python and not the other is worse than none, so the
        # answer has to come from each interpreter actually running.
        print(json.dumps({
            'interpreter': sys.executable,
            'af_unix': hasattr(socket, 'AF_UNIX'),
        }), flush=True)
        directory = self.root / 'private'
        directory.mkdir(mode=0o700)
        handle = self.handle(directory, access=READ_CONTROL, flags=BACKUP)
        owner = self.native.owner(handle)
        print(json.dumps({'python': sys.version, 'executable': sys.executable,
                          'platform': platform.platform(),
                          'owner': owner}), flush=True)
        details = self.native.token_details()
        print(json.dumps({'token': details}), flush=True)
        self.assertTrue(owner.startswith('S-1-'))
        # Observe rather than invent a TokenUser == owner policy. This does
        # not audit the DACL or exercise another/elevation-restricted token.

    def test_empty_file_nonblocking_lock_and_normal_exit(self):
        import msvcrt
        lock = self.root / 'lock'
        lock.touch()
        with open(lock, 'r+b', buffering=0) as stream:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                self.run_child('lock', 'contended')
                self.assertEqual(lock.stat().st_size, 0)
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        self.run_child('lock', 'acquire')
        self.run_child('lock', 'exit')
        self.run_child('lock', 'acquire')

    def test_cancelled_nonblocking_retry_closes_descriptor(self):
        import msvcrt
        (self.root / 'lock').touch()
        process = self.stop_at('lock', 'hold')
        descriptors = []

        async def exercise():
            attempted = asyncio.Event()

            async def waiter():
                with open(self.root / 'lock', 'r+b', buffering=0) as stream:
                    descriptors.append(stream.fileno())
                    while True:
                        stream.seek(0)
                        try:
                            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        except OSError as error:
                            if error.errno != errno.EACCES:
                                raise
                            attempted.set()
                            await asyncio.sleep(0.05)
                        else:
                            stream.seek(0)
                            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                            self.fail('acquired lock while child owns it')

            task = asyncio.create_task(waiter())
            try:
                await asyncio.wait_for(attempted.wait(), timeout=5)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(exercise())
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])
        self.run_child('lock', 'contended')
        process.kill()
        process.wait(timeout=10)
        self.run_child('lock', 'acquire')

    def test_process_termination_releases_lock(self):
        (self.root / 'lock').touch()
        process = self.stop_at('lock', 'hold')
        self.run_child('lock', 'contended')
        process.kill()
        process.wait(timeout=10)
        self.run_child('lock', 'acquire')

    def test_crt_open_allows_read_write_but_blocks_delete(self):
        path = self.root / 'file'
        path.write_bytes(b'old')
        with open(path, 'rb'):
            with open(path, 'r+b') as writer:
                writer.write(b'new')
            with self.assertRaises(OSError) as caught:
                os.replace(path, self.root / 'moved')
            self.assertEqual(caught.exception.winerror, 32)
        os.replace(path, self.root / 'moved')
        self.assertEqual((self.root / 'moved').read_bytes(), b'new')

    def test_sharing_is_checked_in_both_directions(self):
        path = self.root / 'file'
        path.touch()
        # Existing reader permits writes, but a new writer which refuses
        # read sharing still conflicts with the existing reader's access.
        self.handle(path, access=GENERIC_READ, share=SHARE_ALL)
        with self.assertRaises(OSError) as caught:
            self.handle(path, access=GENERIC_WRITE, share=2)
        self.assertEqual(caught.exception.winerror, 32)
        self.handle(path, access=GENERIC_WRITE, share=SHARE_ALL)
        other = self.root / 'other'
        other.touch()
        self.handle(other, access=GENERIC_READ, share=1)
        with self.assertRaises(OSError) as caught:
            self.handle(other, access=GENERIC_WRITE, share=SHARE_ALL)
        self.assertEqual(caught.exception.winerror, 32)

    def test_relative_open_retains_renamed_directory(self):
        directory = self.root / 'directory'
        directory.mkdir()
        (directory / 'child').write_bytes(b'original\r\n\x1a\x00\xff')
        handle = self.handle(directory, flags=BACKUP)
        moved = self.root / 'moved'
        os.rename(directory, moved)
        directory.mkdir()
        (directory / 'child').write_bytes(b'decoy')
        child_handle = self.native.relative(handle, 'child')
        self.assertEqual(self.read_handle(child_handle),
                         b'original\r\n\x1a\x00\xff')
        with self.assertRaises(OSError):
            self.native.relative(handle, 'missing')

    def test_leaf_reparse_open_does_not_block_ancestor_junction(self):
        directory = self.root / 'directory'
        directory.mkdir()
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'child').write_bytes(b'outside')
        junction = directory / 'junction'
        # Native MinGW pathlib emits forward slashes; cmd's mklink parser
        # treats them as switches. Only convert separators for this command,
        # without normalizing components or changing any runtime operands.
        result = subprocess.run(
            ['cmd.exe', '/d', '/c', 'mklink', '/J',
             str(junction).replace('/', '\\'), str(outside).replace('/', '\\')],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.addCleanup(lambda: os.rmdir(junction))
        handle = self.handle(directory, flags=BACKUP)
        self.assertEqual(self.read_handle(
            self.native.relative(handle, 'junction\\child')), b'outside')
        # FILE_DIRECTORY_FILE | FILE_OPEN_REPARSE_POINT opens the junction
        # itself. Use GetFileInformationByHandleEx(FileAttributeTagInfo).
        leaf = self.native.relative(handle, 'junction', 1 | 0x200000)
        try:
            query = self.native.bind(
                self.native.kernel, 'GetFileInformationByHandleEx', W.BOOL,
                HANDLE, C.c_int, HANDLE, ULONG)
            attributes = (ULONG * 2)()
            self.native.check(query(leaf, 9, attributes, C.sizeof(attributes)))
            self.assertTrue(attributes[0] & 0x400)
            self.assertEqual(attributes[1], 0xa0000003)  # mount point tag
        finally:
            self.native.check(self.native.close(leaf))

    def check_source_handle_rename(self, *, native=False, absolute=False):
        source = self.root / 'temporary'
        source.write_bytes(b'selected')
        handle = self.handle(source, access=GENERIC_READ | DELETE)
        moved = self.root / 'moved'
        os.rename(source, moved)
        source.write_bytes(b'decoy')
        destination = self.root / 'destination'
        destination.mkdir()
        directory = self.handle(destination, flags=BACKUP)
        retained = self.root / 'retained'
        os.rename(destination, retained)
        destination.mkdir()
        published = retained / 'published'
        published.write_bytes(b'old')
        # The absolute-path Win32 control only tests source identity; the
        # relative cases additionally test retained destination identity.
        name = str(published).replace('/', '\\') if absolute else 'published'
        self.native.rename(handle, None if absolute else directory, name,
                           native=native)
        # A CRT read would itself refuse delete sharing against our existing
        # DELETE-access source handle; this reader deliberately shares delete.
        self.assertEqual(self.read_handle(self.native.open(published)),
                         b'selected')
        self.assertEqual(source.read_bytes(), b'decoy')
        self.assertFalse(moved.exists())
        self.assertFalse((destination / 'published').exists())

    def test_win32_absolute_rename_retains_source_identity(self):
        self.check_source_handle_rename(absolute=True)

    def test_win32_relative_rename_retains_both_identities(self):
        self.check_source_handle_rename()

    def test_native_relative_rename_retains_both_identities(self):
        # An independent experiment, not a fallback which hides Win32 failure.
        self.check_source_handle_rename(native=True)

    def test_source_handle_delete_ignores_replaced_name(self):
        source = self.root / 'temporary'
        source.write_bytes(b'selected')
        selected = self.root / 'selected'
        # LIFO cleanup: close source, then check deletion, then remove tempdir.
        self.addCleanup(lambda: self.assertFalse(selected.exists()))
        handle = self.handle(source, access=GENERIC_READ | DELETE)
        os.rename(source, selected)
        source.write_bytes(b'decoy')
        self.native.delete(handle)
        self.assertEqual(source.read_bytes(), b'decoy')

    def test_kill_at_write_checkpoints_preserves_complete_target(self):
        for stage in ('create', 'write', 'flush', 'fsync', 'close', 'replace'):
            with self.subTest(stage=stage):
                (self.root / 'temporary').unlink(missing_ok=True)
                target = self.root / 'target'
                target.write_bytes(b'{"generation": "old"}\n')
                process = self.stop_at('write', stage)
                process.kill()
                process.wait(timeout=10)
                expected = 'new' if stage == 'replace' else 'old'
                self.assertEqual(json.loads(target.read_bytes()),
                                 {'generation': expected})
                # Stale temporary files are expected before publication.
                self.assertEqual((self.root / 'temporary').exists(),
                                 stage != 'replace')

    def test_failed_replacement_keeps_both_files(self):
        target = self.root / 'target'
        temporary = self.root / 'temporary'
        target.write_bytes(b'old')
        temporary.write_bytes(b'new')
        with open(target, 'rb'):
            with self.assertRaises(OSError) as caught:
                os.replace(temporary, target)
            print(json.dumps({'operation': 'replace_open_destination',
                              'winerror': caught.exception.winerror}),
                  flush=True)
            # MoveFileExW reported ACCESS_DENIED (5) for an open destination
            # on all three CI runtimes; moving an open source reported 32.
            self.assertIn(caught.exception.winerror, (5, 32))
        self.assertEqual(target.read_bytes(), b'old')
        self.assertEqual(temporary.read_bytes(), b'new')
        # Closing the reader must remove the obstruction, not merely happen
        # to coincide with some unrelated permission failure.
        os.replace(temporary, target)
        self.assertEqual(target.read_bytes(), b'new')
        self.assertFalse(temporary.exists())

    # -- transport / scrub / identity probes -------------------------------
    # Each records what Windows actually does; the answer decides whether the
    # production port needs code or only a test.  Recorded, not asserted,
    # except where the documented contract is the observation.

    def environment_block(self):
        pointer = self.native.get_environment()
        if not pointer:
            raise C.WinError(C.get_last_error())
        try:
            entries, address = [], pointer
            while True:
                text = C.wstring_at(address)
                if not text:
                    break
                entries.append(text)
                address += (len(text) + 1) * C.sizeof(C.c_wchar)
            return entries
        finally:
            self.native.free_environment(pointer)

    def file_identity(self, handle):
        info = FileIdInfo()
        self.native.check(self.native.file_information(
            handle, FILE_ID_INFO_CLASS, C.cast(C.byref(info), HANDLE),
            C.sizeof(info)))
        return info.VolumeSerialNumber, bytes(info.FileId)

    def test_credential_environment_block_after_removal(self):
        # Does removing a variable through os.environ also remove it from the
        # native block a child would receive, or does the original record
        # survive as it does in Linux /proc/<pid>/environ?
        name, value = 'LOKI_SCRUB_PROBE', 'top-secret'
        os.environ[name] = value
        self.addCleanup(os.environ.pop, name, None)
        before = self.environment_block()
        del os.environ[name]
        after = self.environment_block()
        print(json.dumps({
            'probe': 'environment-block-removal',
            'before_has_name': any(e.startswith(name + '=') for e in before),
            'before_has_value': any(e.endswith('=' + value) for e in before),
            'after_has_name': any(e.startswith(name + '=') for e in after),
            'after_has_value': any(e.endswith('=' + value) for e in after),
        }), flush=True)

    def test_lockfileex_contention_reports_the_error(self):
        # LockFileEx documents FAIL_IMMEDIATELY as "returns immediately" but
        # names no error; record the actual code rather than assume 33.
        path = self.root / 'lockfile'
        path.touch()
        first = self.handle(path, access=GENERIC_READ | GENERIC_WRITE)
        second = self.handle(path, access=GENERIC_READ | GENERIC_WRITE)
        overlapped = Overlapped()
        self.native.check(self.native.lock_file(
            first, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0, 1, 0, C.byref(overlapped)))
        try:
            result = self.native.lock_file(
                second, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                0, 1, 0, C.byref(overlapped))
            print(json.dumps({'probe': 'lockfileex-contention',
                              'succeeded': bool(result),
                              'winerror': C.get_last_error()}), flush=True)
        finally:
            self.native.unlock_file(first, 0, 1, 0, C.byref(overlapped))

    def test_file_identity_survives_rename(self):
        # FILE_ID_INFO (Windows 8 / Server 2012+): volume serial + 128-bit id
        # identify an open object across a rename.
        target = self.root / 'identity'
        publish = self.root / 'identity-published'
        target.write_bytes(b'x')
        handle = self.handle(target, access=GENERIC_READ)
        before = self.file_identity(handle)
        os.rename(target, publish)
        moved = self.handle(publish, access=GENERIC_READ)
        after = self.file_identity(moved)
        print(json.dumps({'probe': 'file-id-identity',
                          'same': before == after,
                          'volume_serial': before[0],
                          'file_id': before[1].hex()}), flush=True)

    def test_junction_is_a_reparse_point(self):
        # Junctions are the unprivileged directory alias; record whether
        # creation needs elevation and how the alias is typed.
        real = self.root / 'real'
        real.mkdir()
        alias = self.root / 'alias'
        result = subprocess.run(
            ['cmd.exe', '/d', '/c', 'mklink', '/J', str(alias), str(real)],
            capture_output=True, text=True, timeout=15)
        record = {'probe': 'junction-alias', 'mklink_exit': result.returncode}
        if alias.exists():
            stat = os.lstat(alias)
            record['is_directory'] = os.path.isdir(alias)
            record['reparse_tag'] = hex(getattr(stat, 'st_reparse_tag', 0))
            record['has_reparse_attribute'] = bool(
                stat.st_file_attributes & FILE_ATTRIBUTE_REPARSE_POINT
                if hasattr(stat, 'st_file_attributes') else False)
        print(json.dumps(record), flush=True)
        self.assertEqual(result.returncode, 0,
                         result.stdout + result.stderr)

    def test_pseudoconsole_attaches_a_child(self):
        # ConPTY: the documented synchronous path must launch a child on the
        # pseudoconsole and return its output to a standard user.
        self.run_child('conpty', 'conpty')

    def test_path_resolution_semantics(self):
        # What Windows resolves for a directory junction, `..`, and identity:
        # the file_paths expectations have to be written from this, not from
        # the POSIX kernel behaviour.
        real = self.root / 'real'
        (real / 'child').mkdir(parents=True)
        (real / 'child' / 'file').write_text('x')
        alias = self.root / 'alias'
        subprocess.run(
            ['cmd.exe', '/d', '/c', 'mklink', '/J', str(alias), str(real)],
            capture_output=True, text=True, timeout=15)
        record = {'probe': 'path-semantics', 'alias_exists': alias.exists(),
                  'alias_is_dir': os.path.isdir(alias)}
        record['realpath_alias'] = os.path.realpath(str(alias))
        record['dotdot_plain'] = os.path.realpath(str(real / 'child' / '..'))
        record['dotdot_through_alias'] = os.path.realpath(
            str(alias / 'child' / '..'))
        try:
            record['samefile'] = os.path.samefile(
                str(alias / 'child' / 'file'), str(real / 'child' / 'file'))
        except OSError as error:
            record['samefile_error'] = str(error)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        sys.exit(child(*sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == '--standard-user':
        if len(sys.argv) != 3 or os.name != 'nt':
            raise RuntimeError('--standard-user requires native Windows and a SID')
        details = NativeCalls().token_details(include_groups=True)
        print(json.dumps({'standard_user_token': details}), flush=True)
        require_standard_user(details, sys.argv[2])
        print('Standard-user token verified before running probes.', flush=True)
        del sys.argv[1:]
    unittest.main(verbosity=2)
