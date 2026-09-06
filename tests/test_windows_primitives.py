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

    def token_details(self):
        token = HANDLE()
        self.check(self.open_token(self.current_process(), 8, C.byref(token)))
        try:
            result = {}
            for label, kind in [('user', 1), ('default_owner', 4)]:
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


def child(mode, root, stage):
    """Subprocess checkpoints acknowledge completion, not durable storage."""
    import msvcrt

    root = Path(root)
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

    def exercise_token_queries(self, *, fail_elevation=False):
        native = NativeCalls.__new__(NativeCalls)
        native.current_process = lambda: 1
        native.close = mock.Mock(return_value=1)
        native.sid = lambda pointer: 'SID-%s' % pointer.value
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
                self.assertEqual(native.token_details(),
                                 {'user': 'SID-1', 'default_owner': 'SID-4',
                                  'elevation': 1})
        native.close.assert_called_once()
        self.assertEqual(native.close.call_args.args[0].value, 42)
        self.assertEqual(queries, [(1, True), (1, False), (4, True),
                                   (4, False), (20, False)])

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


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        sys.exit(child(*sys.argv[2:]))
    unittest.main(verbosity=2)
