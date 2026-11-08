"""Publication ownership and permission-setting checks, not pathname-race certification."""

import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import files as file_operations
from loki_agent import private_files


def _dacl_permissions(sddl):
    if sddl is None:
        return None
    # SetSecurityInfo may record auto-inheritance even on a protected ACL.
    # AI is historical metadata; retain protection and every ACE in the oracle.
    controls, separator, aces = sddl.partition('(')
    return controls.replace('AI', '') + separator + aces


class SyncedPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.directory = private_files.open_directory(self.root)
        self.addCleanup(private_files.close, self.directory)
        self.target = os.path.join(self.root, 'state')

    def test_complete_create_replace_sync_and_borrowed_directory_workflow(self):
        events = []
        write = private_files.write
        sync = private_files.fsync
        close = private_files.close
        replace = private_files.replace_at

        def partial(handle, data):
            return write(handle, data[:max(1, len(data) // 2)])

        def synchronized(handle):
            sync(handle)
            events.append('directory-sync' if handle == self.directory else 'file-sync')

        def closed(handle):
            self.assertNotEqual(handle, self.directory)
            close(handle)
            events.append('file-close')

        def published(directory, temporary, name):
            self.assertEqual(directory, self.directory)
            replace(directory, temporary, name)
            events.append('replace')

        for content in [b'first complete contents', b'replacement complete contents']:
            events.clear()
            with mock.patch.object(private_files, 'write', side_effect=partial) as writing, \
                    mock.patch.object(private_files, 'fsync', side_effect=synchronized), \
                    mock.patch.object(private_files, 'close', side_effect=closed), \
                    mock.patch.object(private_files, 'replace_at', side_effect=published):
                private_files.publish_synced_bytes_at(self.directory, 'state', content)
            self.assertGreater(writing.call_count, 1)
            self.assertEqual(events, ['file-sync', 'file-close', 'replace', 'directory-sync'])
            with open(self.target, 'rb') as stream:
                self.assertEqual(stream.read(), content)
            self.assertEqual(os.listdir(self.root), ['state'])
            self.assertTrue(private_files.information(self.directory).directory)

    def _assert_unpublished_failure(self, operation):
        with open(self.target, 'wb') as stream:
            stream.write(b'original')
        with mock.patch.object(private_files, operation, side_effect=OSError('publication failed')), \
                mock.patch.object(private_files, 'close', wraps=private_files.close) as close:
            with self.assertRaisesRegex(OSError, 'publication failed'):
                private_files.publish_synced_bytes_at(self.directory, 'state', b'replacement')
        self.assertEqual(close.call_count, 1)
        with open(self.target, 'rb') as stream:
            self.assertEqual(stream.read(), b'original')
        self.assertEqual(os.listdir(self.root), ['state'])
        self.assertTrue(private_files.information(self.directory).directory)

    def test_zero_progress_does_not_publish(self):
        with open(self.target, 'wb') as stream:
            stream.write(b'original')
        with mock.patch.object(private_files, 'write', return_value=0), self.assertRaises(OSError):
            private_files.publish_synced_bytes_at(self.directory, 'state', b'replacement')
        with open(self.target, 'rb') as stream:
            self.assertEqual(stream.read(), b'original')
        self.assertEqual(os.listdir(self.root), ['state'])
        self.assertTrue(private_files.information(self.directory).directory)

    def test_file_sync_failure_removes_only_its_unpublished_temporary(self):
        self._assert_unpublished_failure('fsync')

    def test_replace_failure_removes_only_its_unpublished_temporary(self):
        self._assert_unpublished_failure('replace_at')

    def test_close_failure_does_not_retry_a_consumed_handle(self):
        close = private_files.close

        def closed_then_failed(handle):
            close(handle)
            raise OSError('close failed after release')

        with mock.patch.object(private_files, 'close', side_effect=closed_then_failed) as closing:
            with self.assertRaisesRegex(OSError, 'close failed after release'):
                private_files.publish_synced_bytes_at(self.directory, 'state', b'unpublished')
        closing.assert_called_once()
        self.assertEqual(os.listdir(self.root), [])
        self.assertTrue(private_files.information(self.directory).directory)

    def test_directory_sync_failure_does_not_remove_the_published_destination(self):
        sync = private_files.fsync

        def failed_directory_sync(handle):
            if handle == self.directory:
                raise OSError('directory sync failed')
            sync(handle)

        with mock.patch.object(private_files, 'fsync', side_effect=failed_directory_sync):
            with self.assertRaisesRegex(OSError, 'directory sync failed'):
                private_files.publish_synced_bytes_at(self.directory, 'state', b'already published')
        with open(self.target, 'rb') as stream:
            self.assertEqual(stream.read(), b'already published')
        self.assertEqual(os.listdir(self.root), ['state'])
        self.assertTrue(private_files.information(self.directory).directory)


class AtomicWritePermissionTests(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / 'target'
        self.target.write_text('old')
        self.target.chmod(0o640)
        self.mode = stat.S_IMODE(self.target.stat().st_mode)

    def test_descriptor_cleanup_failure_cannot_discard_existing_target_permissions(self):
        from loki_agent import windows_api

        failure = windows_api.WindowsResourceError('LocalFree failed', status=windows_api.ERROR_FILE_NOT_FOUND)
        with mock.patch.object(file_operations, 'sys', mock.Mock(platform='win32')), \
                mock.patch.object(windows_api, 'dacl_sddl', side_effect=failure), \
                mock.patch.object(windows_api, 'set_handle_dacl') as apply:
            with self.assertRaises(windows_api.WindowsResourceError) as caught:
                file_operations._apply_destination_permissions('selected-target', 17, self.mode)
        self.assertIs(caught.exception, failure)
        apply.assert_not_called()

    def test_buffered_save_publishes_complete_utf8_contents(self):
        content = 'complete UTF-8 contents \u2603' * 8192
        file_operations.atomic_write_text(str(self.target), content)
        self.assertEqual(self.target.read_bytes(), content.encode('utf-8'))
        self.assertEqual(os.listdir(self.root), ['target'])

    def test_existing_parent_needs_no_recursive_ancestor_inspection(self):
        # A contained writer can use its workspace without stat access to C:\\
        # or other ancestors. The native owning-runtime workflow exercises that
        # actual boundary; this regression does not depend on host ACLs.
        with mock.patch.object(os, 'makedirs', side_effect=PermissionError(
                'ancestor metadata denied')) as create:
            file_operations.atomic_write_text(str(self.target), 'replacement')
        create.assert_not_called()
        self.assertEqual(self.target.read_text(), 'replacement')
        self.assertEqual(os.listdir(self.root), ['target'])

    def test_missing_parent_is_created_after_native_acquisition_reports_missing(self):
        target = self.root / 'new-parent' / 'nested' / 'file'
        file_operations.atomic_write_text(str(target), 'new contents')
        self.assertEqual(target.read_bytes(), b'new contents')
        self.assertEqual(sorted(os.listdir(target.parent)), ['file'])

    def test_directory_access_denial_does_not_try_recursive_creation(self):
        failure = PermissionError('selected directory denied')
        with mock.patch.object(private_files, 'open_directory', side_effect=failure), \
                mock.patch.object(os, 'makedirs', wraps=os.makedirs) as create:
            with self.assertRaises(PermissionError) as refused:
                file_operations.atomic_write_text(str(self.target), 'replacement')
        self.assertIs(refused.exception, failure)
        create.assert_not_called()
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(os.listdir(self.root), ['target'])

    @unittest.skipIf(sys.platform == 'win32', 'POSIX regular-file size limit')
    def test_native_partial_write_failure_preserves_destination(self):
        # Exercise a real short regular-file write, not a fake buffered writer
        # that violates the standard write/flush contract.
        import resource
        import signal

        limits = resource.getrlimit(resource.RLIMIT_FSIZE)
        previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, [2, limits[1]])
            with self.assertRaises(OSError):
                file_operations.atomic_write_text(str(self.target), 'complete contents')
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, limits)
            signal.signal(signal.SIGXFSZ, previous)
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(os.listdir(self.root), ['target'])

    def test_temporary_name_failure_closes_the_acquired_directory(self):
        opened = []
        open_directory, close = private_files.open_directory, private_files.close

        def opening(*args, **kwargs):
            handle = open_directory(*args, **kwargs)
            opened.append(handle)
            return handle

        with mock.patch.object(private_files, 'open_directory', side_effect=opening), \
                mock.patch.object(private_files, 'close', wraps=close) as closing, \
                mock.patch.object(file_operations.uuid, 'uuid4', side_effect=OSError('name failed')):
            try:
                with self.assertRaisesRegex(OSError, 'name failed'):
                    file_operations.atomic_write_text(str(self.target), 'replacement')
                self.assertEqual(len(opened), 1)
                closing.assert_called_once_with(opened[0])
            finally:
                # Retire a leaked baseline handle when this regression is red;
                # never retry a handle already consumed by the implementation.
                for handle in opened:
                    if mock.call(handle) not in closing.call_args_list:
                        close(handle)
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(os.listdir(self.root), ['target'])

    def test_buffered_flush_failure_does_not_publish(self):
        opening = private_files.binary_file
        failure = OSError('buffer flush failed')

        def failed_flush(*args, **kwargs):
            stream = opening(*args, **kwargs)
            stream.flush = mock.Mock(side_effect=failure)
            return stream

        with mock.patch.object(private_files, 'binary_file', side_effect=failed_flush), \
                self.assertRaisesRegex(OSError, 'buffer flush failed'):
            file_operations.atomic_write_text(str(self.target), 'replacement')
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(os.listdir(self.root), ['target'])

    def test_buffered_wrapper_failure_consumes_the_file_once(self):
        created = []
        create = private_files.create_exclusive_at

        def creating(*args, **kwargs):
            handle = create(*args, **kwargs)
            created.append(handle)
            return handle

        with mock.patch.object(private_files, 'create_exclusive_at', side_effect=creating), \
                mock.patch.object(os, 'fdopen', side_effect=OSError('wrapper failed')), \
                self.assertRaisesRegex(OSError, 'wrapper failed'):
            file_operations.atomic_write_text(str(self.target), 'replacement')
        self.assertEqual(len(created), 1)
        with self.assertRaises(OSError):
            private_files.information(created[0])
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(os.listdir(self.root), ['target'])

    def test_ledger_name_failure_closes_the_acquired_directory(self):
        from loki_agent import windows_state

        opened = []
        open_tree, close = private_files.open_directory_tree, private_files.close

        def opening(*args, **kwargs):
            handle = open_tree(*args, **kwargs)
            opened.append(handle)
            return handle

        with mock.patch.object(windows_state.paths, 'loki_state_dir', return_value=str(self.root)), \
                mock.patch.object(private_files, 'open_directory_tree', side_effect=opening), \
                mock.patch.object(private_files, 'close', wraps=close) as closing, \
                mock.patch.object(windows_state.secrets, 'token_hex', side_effect=OSError('name failed')):
            try:
                with self.assertRaisesRegex(OSError, 'name failed'):
                    windows_state.save_ledger({'version': 1, 'workspaces': {}})
                self.assertEqual(len(opened), 1)
                closing.assert_called_once_with(opened[0])
            finally:
                for handle in opened:
                    if mock.call(handle) not in closing.call_args_list:
                        close(handle)
        self.assertEqual(os.listdir(self.root), ['target'])

    def test_native_fchmod_runs_after_flush_before_close(self):
        if sys.platform == "win32":
            # No descriptor chmod on Windows; the equivalent property is that
            # the destination's DACL survives the write.
            from loki_agent import windows_api

            user = windows_api.current_user_sid()
            for descriptor in [None, f'D:P(A;;FA;;;{user})(A;;FR;;;SY)',
                               f'D:(A;;FA;;;{user})(A;;FR;;;SY)']:
                with self.subTest(descriptor=descriptor):
                    self.target.unlink()
                    if descriptor is not None:
                        handle = windows_api.open_with_access(
                            str(self.target), windows_api.AccessMask.GENERIC_WRITE,
                            creation=windows_api.FileCreateDisposition.CREATE_NEW,
                            flags=windows_api.FileAttribute.FILE_ATTRIBUTE_NORMAL,
                            security_descriptor=descriptor)
                        windows_api.close_handle(handle)
                    self.target.write_text('old')
                    before = windows_api.dacl_sddl(str(self.target))
                    with mock.patch.object(windows_api, 'set_handle_dacl',
                                           wraps=windows_api.set_handle_dacl) as apply:
                        file_operations.atomic_write_text(str(self.target), 'new contents')
                    if descriptor is None:
                        apply.assert_not_called()
                    else:
                        apply.assert_called_once()
                    self.assertEqual(self.target.read_text(), 'new contents')
                    self.assertEqual(_dacl_permissions(windows_api.dacl_sddl(str(self.target))),
                                     _dacl_permissions(before))
                    self.assertEqual(set(self.root.iterdir()), {self.target})
                    if descriptor is not None and descriptor.startswith('D:P'):
                        # The same save must preserve permissions even when the
                        # temporary's readback differs only in protection, has
                        # different ACEs, or reports a NULL DACL. Only the real
                        # inherited exact-match control above may avoid a setter;
                        # the preservation oracle's AI projection is not a waiver.
                        target_acl = windows_api.dacl_sddl(str(self.target))
                        self.assertTrue(target_acl.startswith('D:P'))
                        for current in [target_acl.replace('D:P', 'D:', 1),
                                        f'D:P(A;;FR;;;{user})', None]:
                            with self.subTest(current=current), \
                                    mock.patch.object(windows_api, 'handle_dacl_sddl',
                                                      return_value=current), \
                                    mock.patch.object(windows_api, 'set_handle_dacl',
                                                      wraps=windows_api.set_handle_dacl) as apply:
                                file_operations.atomic_write_text(str(self.target), 'copied permissions')
                            apply.assert_called_once()
                            self.assertEqual(apply.call_args.args[1], target_acl)
                            self.assertEqual(self.target.read_text(), 'copied permissions')
                            self.assertEqual(_dacl_permissions(windows_api.dacl_sddl(str(self.target))),
                                             _dacl_permissions(target_acl))
                            self.assertEqual(set(self.root.iterdir()), {self.target})
            return
        real_fchmod = os.fchmod
        content = 'new contents'

        def set_mode(fd, mode):
            self.assertEqual(os.fstat(fd).st_size, len(content.encode('utf-8')))
            self.assertEqual(mode, self.mode)
            real_fchmod(fd, mode)

        with mock.patch.object(os, 'fchmod', side_effect=set_mode) as fchmod, \
                mock.patch.object(os, 'chmod', side_effect=AssertionError(
                    'pathname chmod must not run')):
            file_operations.atomic_write_text(str(self.target), content)
        fchmod.assert_called_once()
        self.assertEqual(self.target.read_text(), content)
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), self.mode)
        self.assertEqual(set(self.root.iterdir()), {self.target})

    def test_native_fchmod_sets_new_file_mode(self):
        if sys.platform == "win32":
            # A new file keeps the inherited default DACL rather than gaining
            # one that denies the user.
            from loki_agent import windows_api
            new_file = self.root / 'new-file'
            file_operations.atomic_write_text(str(new_file), 'new')
            self.assertEqual(new_file.read_text(), 'new')
            self.assertIsNotNone(windows_api.dacl_sddl(str(new_file)))
            self.assertEqual(set(self.root.iterdir()), {self.target, new_file})
            return
        new_file = self.root / 'new-file'
        with mock.patch.object(os, 'fchmod', wraps=os.fchmod) as fchmod, \
                mock.patch.object(os, 'chmod', side_effect=AssertionError(
                    'pathname chmod must not run')):
            file_operations.atomic_write_text(str(new_file), 'new')
        fchmod.assert_called_once()
        self.assertEqual(fchmod.call_args.args[1], 0o666 & ~file_operations._UMASK)
        self.assertEqual(stat.S_IMODE(new_file.stat().st_mode),
                         0o666 & ~file_operations._UMASK)

    def test_replaced_temporary_name_cannot_redirect_fchmod(self):
        if sys.platform == "win32":
            from loki_agent import private_files, windows_api

            unrelated = self.root / 'unrelated'
            unrelated.write_text('untouched')
            user = windows_api.current_user_sid()
            # This fixture needs a genuinely protected ACL: after unlinking a
            # temporary the setter cannot recompute parent inheritance. Create
            # the target with D:P, as in the permission-preservation cases above;
            # the named setter alone does not apply that control flag.
            self.target.unlink()
            handle = windows_api.open_with_access(
                str(self.target), windows_api.AccessMask.GENERIC_WRITE,
                creation=windows_api.FileCreateDisposition.CREATE_NEW,
                flags=windows_api.FileAttribute.FILE_ATTRIBUTE_NORMAL,
                security_descriptor=f'D:P(A;;FA;;;{user})(A;;FR;;;SY)')
            windows_api.close_handle(handle)
            self.target.write_text('old')
            windows_api.set_named_dacl(str(unrelated), f'D:P(A;;FA;;;{user})')
            target_acl = windows_api.dacl_sddl(str(self.target))
            unrelated_acl = windows_api.dacl_sddl(str(unrelated))
            self.assertNotEqual(target_acl, unrelated_acl)
            apply = windows_api.set_handle_dacl

            def replace_name_then_set_acl(handle, sddl):
                temporary, = self.root.glob('.target.*.tmp')
                # A mismatched ACL must be settled while the created object
                # is still empty, not after exposing bytes under the old grant.
                self.assertEqual(windows_api.by_handle_file_information(handle).nFileSizeLow, 0)
                temporary.unlink()
                os.link(unrelated, temporary)
                apply(handle, sddl)
                self.assertEqual(_dacl_permissions(windows_api.handle_dacl_sddl(handle)),
                                 _dacl_permissions(target_acl))
                self.assertEqual(windows_api.dacl_sddl(str(unrelated)), unrelated_acl)

            # A hard link requires no symlink privilege. Stop before publication:
            # this proves permission assignment, not race-free entry replacement.
            with mock.patch.object(windows_api, 'set_handle_dacl',
                                   side_effect=replace_name_then_set_acl), \
                    mock.patch.object(private_files, 'replace_at',
                                      side_effect=OSError('stop before publication')):
                with self.assertRaisesRegex(OSError, 'stop before publication'):
                    file_operations.atomic_write_text(str(self.target), 'new')
            self.assertEqual(self.target.read_text(), 'old')
            self.assertEqual(unrelated.read_text(), 'untouched')
            self.assertEqual(set(self.root.iterdir()), {self.target, unrelated})
            return

        from loki_agent import private_files

        unrelated = self.root / 'unrelated'
        unrelated.write_text('untouched')
        unrelated.chmod(0o600)
        real_create = private_files.create_exclusive_at
        real_fchmod = os.fchmod
        temporary = {}

        def create_temporary(directory_fd, name, *args, **kwargs):
            fd = real_create(directory_fd, name, *args, **kwargs)
            temporary.update(
                dir_fd=directory_fd,
                name=name,
                identity=os.fstat(fd),
            )
            return fd

        def replace_name_then_set_mode(fd, mode):
            # Unlink the temporary entry and swap in a symlink to unrelated
            # within the open destination directory.
            private_files.unlink_at(temporary['dir_fd'], temporary['name'])
            os.symlink(str(unrelated), temporary['name'],
                       dir_fd=temporary['dir_fd'])
            self.assertTrue(os.path.samestat(
                os.fstat(fd), temporary['identity']))
            real_fchmod(fd, mode)
            self.assertEqual(stat.S_IMODE(os.fstat(fd).st_mode), self.mode)
            self.assertEqual(stat.S_IMODE(unrelated.stat().st_mode), 0o600)

        # Stop before publication: we are testing that mode assignment is bound
        # to the descriptor, not entry publication.
        with mock.patch.object(private_files, 'create_exclusive_at',
                               side_effect=create_temporary), \
                mock.patch.object(os, 'fchmod',
                                  side_effect=replace_name_then_set_mode), \
                mock.patch.object(os, 'chmod', side_effect=AssertionError(
                    'pathname chmod must not run')), \
                mock.patch.object(private_files, 'replace_at',
                                  side_effect=OSError('stop before publication')):
            with self.assertRaisesRegex(OSError, 'stop before publication'):
                file_operations.atomic_write_text(str(self.target), 'new')

        self.assertEqual(unrelated.read_text(), 'untouched')
        self.assertEqual(stat.S_IMODE(unrelated.stat().st_mode), 0o600)
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(set(self.root.iterdir()), {self.target, unrelated})

    def test_fchmod_error_does_not_fall_back_to_pathname(self):
        if sys.platform == "win32":
            # A real mismatch must still update the ACL or fail without writing
            # bytes, publishing, or retrying through the temporary pathname.
            from loki_agent import private_files, windows_api
            user = windows_api.current_user_sid()
            windows_api.set_named_dacl(str(self.target), f'D:P(A;;FA;;;{user})(A;;FR;;;SY)')
            for failure in [PermissionError('mode denied'),
                            windows_api.WindowsApiError('ACL denied', status=5)]:
                with self.subTest(failure=type(failure).__name__), \
                        mock.patch.object(windows_api, 'set_handle_dacl', side_effect=failure), \
                        mock.patch.object(private_files, 'binary_file', side_effect=AssertionError(
                            'stream creation must wait for permissions')), \
                        mock.patch.object(windows_api, 'set_named_dacl', side_effect=AssertionError(
                            'pathname fallback must not run')):
                    with self.assertRaises(type(failure)) as refused:
                        file_operations.atomic_write_text(str(self.target), 'new')
                self.assertIs(refused.exception, failure)
                self.assertEqual(self.target.read_text(), 'old')
                self.assertEqual(set(self.root.iterdir()), {self.target})
            return
        with mock.patch.object(os, 'fchmod', create=True,
                               side_effect=PermissionError('mode denied')), \
                mock.patch.object(os, 'chmod') as chmod:
            with self.assertRaisesRegex(PermissionError, 'mode denied'):
                file_operations.atomic_write_text(str(self.target), 'new')
        chmod.assert_not_called()
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(set(self.root.iterdir()), {self.target})

    def test_creation_permission_control_is_explicit(self):
        """Only an editor temporary requests DACL control, never a capture by default."""
        from loki_agent import _private_files_windows, windows_api

        with mock.patch.object(_private_files_windows, '_open_relative') as opening:
            for control in [False, True]:
                with self.subTest(permission_control=control):
                    _private_files_windows.create_exclusive_at(
                        123, 'temporary', 0o600, permission_control=control)
                    access = opening.call_args.args[2]
                    self.assertEqual(bool(access & windows_api.AccessMask.WRITE_DAC), control)
                    self.assertTrue(access & windows_api.AccessMask.GENERIC_WRITE)

    def test_permission_read_error_does_not_publish(self):
        if sys.platform == "win32":
            # Unknown destination permissions and failed temporary readback
            # both refuse publication. Neither is evidence of an exact match,
            # nor permission to expose content first or try a pathname setter.
            from loki_agent import private_files, windows_api
            for operation in ['dacl_sddl', 'handle_dacl_sddl']:
                failure = windows_api.WindowsApiError('dacl read failed', status=5)
                with self.subTest(operation=operation), \
                        mock.patch.object(windows_api, operation, side_effect=failure), \
                        mock.patch.object(private_files, 'binary_file', side_effect=AssertionError(
                            'stream creation must wait for permissions')), \
                        mock.patch.object(windows_api, 'set_named_dacl', side_effect=AssertionError(
                            'pathname fallback must not run')):
                    with self.assertRaises(windows_api.WindowsApiError) as refused:
                        file_operations.atomic_write_text(str(self.target), 'new')
                self.assertIs(refused.exception, failure)
                self.assertEqual(self.target.read_text(), 'old')
                self.assertEqual(set(self.root.iterdir()), {self.target})
            return
        original_stat = os.stat

        def denied_permissions(path, *args, **kwargs):
            if path == str(self.target):
                raise PermissionError('mode denied')
            return original_stat(path, *args, **kwargs)

        with mock.patch.object(os, 'stat', side_effect=denied_permissions):
            with self.assertRaisesRegex(PermissionError, 'mode denied'):
                file_operations.atomic_write_text(str(self.target), 'new')
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(set(self.root.iterdir()), {self.target})
