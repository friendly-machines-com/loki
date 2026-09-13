"""Permission-setting checks; these do not certify pathname publication."""

import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from loki_agent import loki


class AtomicWritePermissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / 'target'
        self.target.write_text('old')
        self.target.chmod(0o640)
        self.mode = stat.S_IMODE(self.target.stat().st_mode)

    def test_native_fchmod_runs_after_flush_before_close(self):
        if os.name != "posix":
            # No descriptor chmod on Windows; the equivalent property is that
            # the destination's DACL survives the write.
            from loki_agent import windows_api
            before = windows_api.dacl_sddl(str(self.target))
            loki._atomic_write_text(str(self.target), 'new contents')
            self.assertEqual(self.target.read_text(), 'new contents')
            self.assertEqual(windows_api.dacl_sddl(str(self.target)), before)
            self.assertEqual(set(self.root.iterdir()), {self.target})
            return
        real_fchmod = os.fchmod
        descriptors = []
        content = 'new contents'

        def set_mode(fd, mode):
            descriptors.append(fd)
            self.assertEqual(os.fstat(fd).st_size, len(content.encode('utf-8')))
            self.assertEqual(mode, self.mode)
            real_fchmod(fd, mode)

        with mock.patch.object(os, 'fchmod', side_effect=set_mode) as fchmod, \
                mock.patch.object(os, 'chmod', side_effect=AssertionError(
                    'pathname chmod must not run')):
            loki._atomic_write_text(str(self.target), content)
        fchmod.assert_called_once()
        self.assertEqual(self.target.read_text(), content)
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), self.mode)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])
        self.assertEqual(set(self.root.iterdir()), {self.target})

    def test_native_fchmod_sets_new_file_mode(self):
        if os.name != "posix":
            # A new file keeps the inherited default DACL rather than gaining
            # one that denies the user.
            from loki_agent import windows_api
            new_file = self.root / 'new-file'
            loki._atomic_write_text(str(new_file), 'new')
            self.assertEqual(new_file.read_text(), 'new')
            self.assertIsNotNone(windows_api.dacl_sddl(str(new_file)))
            self.assertEqual(set(self.root.iterdir()), {self.target, new_file})
            return
        new_file = self.root / 'new-file'
        with mock.patch.object(os, 'fchmod', wraps=os.fchmod) as fchmod, \
                mock.patch.object(os, 'chmod', side_effect=AssertionError(
                    'pathname chmod must not run')):
            loki._atomic_write_text(str(new_file), 'new')
        fchmod.assert_called_once()
        self.assertEqual(fchmod.call_args.args[1], 0o666 & ~loki._UMASK)
        self.assertEqual(stat.S_IMODE(new_file.stat().st_mode),
                         0o666 & ~loki._UMASK)

    def test_replaced_temporary_name_cannot_redirect_fchmod(self):
        if os.name != "posix":
            # The swap needs an unprivileged symlink, which this platform does
            # not offer; the property that defends against it is that the DACL
            # is applied through the open handle, so assert that call instead.
            from loki_agent import windows_api
            with mock.patch.object(windows_api, 'set_handle_dacl') as apply:
                loki._atomic_write_text(str(self.target), 'new')
            apply.assert_called_once()
            self.assertEqual(self.target.read_text(), 'new')
            self.assertEqual(set(self.root.iterdir()), {self.target})
            return
        unrelated = self.root / 'unrelated'
        unrelated.write_text('untouched')
        unrelated.chmod(0o600)
        real_mkstemp = tempfile.mkstemp
        real_fchmod = os.fchmod
        temporary = {}

        def create_temporary(*args, **kwargs):
            fd, path = real_mkstemp(*args, **kwargs)
            temporary.update(fd=fd, path=path, identity=os.fstat(fd))
            return fd, path

        def replace_name_then_set_mode(fd, mode):
            os.unlink(temporary['path'])
            os.symlink(unrelated, temporary['path'])
            self.assertTrue(os.path.samestat(
                os.fstat(fd), temporary['identity']))
            real_fchmod(fd, mode)
            self.assertEqual(stat.S_IMODE(os.fstat(fd).st_mode), self.mode)
            self.assertEqual(stat.S_IMODE(unrelated.stat().st_mode), 0o600)

        # Stop before rename: selecting the source entry for publication is a
        # separate remaining issue, not something fchmod claims to protect.
        with mock.patch.object(tempfile, 'mkstemp', side_effect=create_temporary), \
                mock.patch.object(os, 'fchmod',
                                  side_effect=replace_name_then_set_mode), \
                mock.patch.object(os, 'chmod', side_effect=AssertionError(
                    'pathname chmod must not run')), \
                mock.patch.object(os, 'replace',
                                  side_effect=OSError('stop before publication')):
            with self.assertRaisesRegex(OSError, 'stop before publication'):
                loki._atomic_write_text(str(self.target), 'new')
        self.assertEqual(unrelated.read_text(), 'untouched')
        self.assertEqual(stat.S_IMODE(unrelated.stat().st_mode), 0o600)
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(set(self.root.iterdir()), {self.target, unrelated})
        with self.assertRaises(OSError):
            os.fstat(temporary['fd'])

    def test_fchmod_error_does_not_fall_back_to_pathname(self):
        if os.name != "posix":
            # A failed DACL application must fail the write, not fall back to
            # anything pathname-based.
            from loki_agent import windows_api
            with mock.patch.object(windows_api, 'set_handle_dacl',
                                   side_effect=PermissionError('mode denied')):
                with self.assertRaisesRegex(PermissionError, 'mode denied'):
                    loki._atomic_write_text(str(self.target), 'new')
            self.assertEqual(self.target.read_text(), 'old')
            self.assertEqual(set(self.root.iterdir()), {self.target})
            return
        with mock.patch.object(os, 'fchmod', create=True,
                               side_effect=PermissionError('mode denied')), \
                mock.patch.object(os, 'chmod') as chmod:
            with self.assertRaisesRegex(PermissionError, 'mode denied'):
                loki._atomic_write_text(str(self.target), 'new')
        chmod.assert_not_called()
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(set(self.root.iterdir()), {self.target})

    def test_missing_fchmod_uses_pathname_fallback(self):
        if os.name != "posix":
            # No destination DACL to preserve: the write succeeds and the new
            # file carries the inherited default DACL.
            from loki_agent import windows_api
            new_file = self.root / 'new-file'
            loki._atomic_write_text(str(new_file), 'new')
            self.assertEqual(new_file.read_text(), 'new')
            self.assertIsNotNone(windows_api.dacl_sddl(str(new_file)))
            self.assertEqual(set(self.root.iterdir()), {self.target, new_file})
            return
        real_chmod = os.chmod
        with mock.patch.object(os, 'fchmod', None, create=True), \
                mock.patch.object(os, 'chmod', wraps=real_chmod) as chmod:
            loki._atomic_write_text(str(self.target), 'new')
        chmod.assert_called_once()
        temporary_path, mode = chmod.call_args.args
        self.assertEqual(Path(temporary_path).parent, self.root)
        self.assertNotEqual(Path(temporary_path), self.target)
        self.assertEqual(mode, self.mode)
        self.assertEqual(self.target.read_text(), 'new')
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), self.mode)
        self.assertEqual(set(self.root.iterdir()), {self.target})

    def test_fallback_error_cleans_up_without_publication(self):
        if os.name != "posix":
            # A DACL read that fails for a reason other than "not found" must
            # abort the write and publish nothing.
            from loki_agent import windows_api
            failure = windows_api.WindowsApiError('dacl read failed', status=5)
            with mock.patch.object(windows_api, 'dacl_sddl',
                                   side_effect=failure):
                with self.assertRaises(windows_api.WindowsApiError):
                    loki._atomic_write_text(str(self.target), 'new')
            self.assertEqual(self.target.read_text(), 'old')
            self.assertEqual(set(self.root.iterdir()), {self.target})
            return
        with mock.patch.object(os, 'fchmod', None, create=True), \
                mock.patch.object(os, 'chmod',
                                  side_effect=PermissionError('mode denied')):
            with self.assertRaisesRegex(PermissionError, 'mode denied'):
                loki._atomic_write_text(str(self.target), 'new')
        self.assertEqual(self.target.read_text(), 'old')
        self.assertEqual(set(self.root.iterdir()), {self.target})
