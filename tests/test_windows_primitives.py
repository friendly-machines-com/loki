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

    def rename(self, source, directory, name):
        encoded = name.encode('utf-16-le')
        size = max(C.sizeof(RenameInfos),
                   RenameInfos.FileName.offset + len(encoded))
        buffer = C.create_string_buffer(size)
        info = RenameInfos.from_buffer(buffer)
        info.ReplaceIfExists = 1
        info.RootDirectory = directory
        info.FileNameLength = len(encoded)
        C.memmove(C.addressof(buffer) + RenameInfos.FileName.offset,
                  encoded, len(encoded))
        self.check(self.set_info(source, 3, buffer, size))

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
            for label, kind in [('user', 1), ('default_owner', 4),
                                ('elevation', 20)]:
                size = ULONG()
                self.token_info(token, kind, None, 0, C.byref(size))
                if C.get_last_error() != 122:  # ERROR_INSUFFICIENT_BUFFER
                    raise C.WinError(C.get_last_error())
                buffer = C.create_string_buffer(size.value)
                self.check(self.token_info(token, kind, buffer, size,
                                           C.byref(size)))
                result[label] = (ULONG.from_buffer(buffer).value
                                 if kind == 20 else
                                 self.sid(HANDLE.from_buffer(buffer)))
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
        details = self.native.token_details()
        print(json.dumps({'python': sys.version, 'executable': sys.executable,
                          'platform': platform.platform(),
                          'owner': owner, 'token': details}), flush=True)
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
        result = subprocess.run(
            ['cmd.exe', '/d', '/c', 'mklink', '/J', str(junction), str(outside)],
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

    def test_source_handle_rename_and_delete_ignore_replaced_name(self):
        source = self.root / 'temporary'
        source.write_bytes(b'selected')
        # Check deletion after source closure but before directory/temp cleanup.
        selected = self.root / 'retained' / 'selected'
        self.addCleanup(lambda: self.assertFalse(selected.exists()))
        handle = self.handle(source, access=GENERIC_READ | DELETE)
        os.rename(source, self.root / 'moved')
        source.write_bytes(b'decoy')
        destination = self.root / 'destination'
        destination.mkdir()
        directory = self.handle(destination, flags=BACKUP)
        retained = self.root / 'retained'
        os.rename(destination, retained)
        destination.mkdir()
        (retained / 'published').write_bytes(b'old')
        self.native.rename(handle, directory, 'published')
        # A CRT read would itself refuse delete sharing against our existing
        # DELETE-access source handle; this reader deliberately shares delete.
        self.assertEqual(self.read_handle(
            self.native.open(retained / 'published')), b'selected')
        self.assertEqual(source.read_bytes(), b'decoy')
        self.assertFalse((destination / 'published').exists())
        # Move again and replace the published name before handle deletion.
        selected = retained / 'selected'
        os.rename(retained / 'published', selected)
        (retained / 'published').write_bytes(b'another decoy')
        self.native.delete(handle)
        # Deletion completes at final handle close (registered cleanup).
        self.assertEqual((retained / 'published').read_bytes(), b'another decoy')
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
            self.assertEqual(caught.exception.winerror, 32)
        self.assertEqual(target.read_bytes(), b'old')
        self.assertEqual(temporary.read_bytes(), b'new')
        temporary.unlink()
        self.assertEqual(target.read_bytes(), b'old')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        sys.exit(child(*sys.argv[2:]))
    unittest.main(verbosity=2)
