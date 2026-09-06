"""Path lookup regressions and the existing atomic-write/symlink contract."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from loki_agent import loki, savefiles, sessions, terminal_frontend


@unittest.skipUnless(os.name == 'posix', 'POSIX filesystem traversal tests')
class FilePathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        self.elsewhere = self.root / 'elsewhere'
        (self.elsewhere / 'child').mkdir(parents=True)
        (self.project / 'link').symlink_to(self.elsewhere / 'child')
        self.session = sessions.Session(shell_cwd=str(self.project))
        session_patch = mock.patch.object(loki, '_DEFAULT_SESSION', self.session)
        session_patch.start()
        self.addCleanup(session_patch.stop)
        state_patch = mock.patch.object(loki, 'file_state', {})
        state_patch.start()
        self.addCleanup(state_patch.stop)

    def test_read_and_edit_follow_kernel_dotdot_lookup(self):
        wrong = self.project / 'text.txt'
        right = self.elsewhere / 'text.txt'
        wrong.write_text('wrong target')
        right.write_text('correct target')
        path = 'link/../text.txt'
        self.assertIn('correct target', loki.run_read(path))
        self.assertNotIn('wrong target', loki.run_read(path))
        self.assertIn('Successfully', loki.run_edit(path, 'correct', 'edited'))
        self.assertEqual(right.read_text(), 'edited target')
        self.assertEqual(wrong.read_text(), 'wrong target')

    def test_write_uses_the_kernel_selected_parent_for_temporary_file(self):
        right = self.elsewhere / 'text.txt'
        wrong = self.project / 'text.txt'
        right.write_text('right')
        wrong.write_text('wrong')
        path = str(self.project) + '/link/../text.txt'
        loki.run_read(path)
        real_open = os.open
        created_inodes = []

        def observe_open(name, flags, *args, **kwargs):
            fd = real_open(name, flags, *args, **kwargs)
            if flags & os.O_EXCL:
                created_inodes.append(os.stat(os.path.dirname(name)).st_ino)
            return fd

        with mock.patch.object(loki.os, 'open', side_effect=observe_open):
            self.assertIn('Successfully', loki.run_write(path, 'updated'))
        self.assertEqual(created_inodes, [self.elsewhere.stat().st_ino])
        self.assertEqual(right.read_text(), 'updated')
        self.assertEqual(wrong.read_text(), 'wrong')
        self.assertFalse(list(self.elsewhere.glob('.*.tmp')))

    def test_final_symlink_survives_atomic_target_replacement(self):
        target = self.project / 'target'
        target.write_text('old')
        target.chmod(0o640)
        alias = self.project / 'alias'
        alias.symlink_to('target')
        hard_link = self.project / 'hard-link'
        os.link(target, hard_link)
        original_inode = target.stat().st_ino
        # Read through one spelling, edit through another: existing alias
        # sharing in the read-before-write cache must remain available.
        loki.run_read(str(target))
        self.assertIn('Successfully', loki.run_edit(str(alias), 'old', 'edited'))
        self.assertTrue(alias.is_symlink())
        self.assertEqual(os.readlink(alias), 'target')
        self.assertEqual(target.read_text(), 'edited')
        self.assertEqual(target.stat().st_mode & 0o777, 0o640)
        self.assertNotEqual(target.stat().st_ino, original_inode)
        self.assertEqual(hard_link.read_text(), 'old')
        self.assertIn('Successfully', loki.run_write(str(alias), 'written'))
        self.assertTrue(alias.is_symlink())
        self.assertEqual(target.read_text(), 'written')

    def test_dangling_final_symlink_creates_target_and_preserves_link(self):
        alias = self.project / 'alias'
        alias.symlink_to('../elsewhere/new/subdir/target')
        self.assertIn('Successfully', loki.run_write(str(alias), 'created'))
        self.assertTrue(alias.is_symlink())
        self.assertEqual(
            (self.elsewhere / 'new/subdir/target').read_text(), 'created')

    def test_final_symlink_target_dotdot_uses_kernel_traversal(self):
        right = self.elsewhere / 'target'
        wrong = self.project / 'target'
        right.write_text('right')
        wrong.write_text('wrong')
        alias = self.project / 'alias'
        alias.symlink_to('link/../target')
        loki.run_read(str(alias))
        self.assertIn('Successfully', loki.run_write(str(alias), 'updated'))
        self.assertEqual(right.read_text(), 'updated')
        self.assertEqual(wrong.read_text(), 'wrong')
        self.assertEqual(os.readlink(alias), 'link/../target')

    def test_invalid_traversal_does_not_read_a_simplified_path(self):
        target = self.project / 'target'
        target.write_text('must not read')
        (self.project / 'plain').write_text('not a directory')
        for relative in ('missing/../target', 'plain/../target', 'target/'):
            with self.subTest(path=relative):
                with self.assertRaises(OSError):
                    with open(str(self.project) + '/' + relative):
                        pass
                result = loki.run_read(relative)
                self.assertTrue(result.startswith('Error:'), result)
                self.assertNotIn('must not read', result)

    def test_parent_creation_cannot_bypass_read_before_overwrite(self):
        target = self.project / 'target'
        target.write_text('must preserve')
        result = loki.run_write('missing/../target', 'overwrite')
        self.assertTrue(result.startswith('Error:'), result)
        self.assertEqual(target.read_text(), 'must preserve')

    def test_stale_write_does_not_recreate_a_deleted_parent(self):
        parent = self.project / 'deleted-parent'
        parent.mkdir()
        target = parent / 'target'
        target.write_text('original')
        loki.run_read(str(target))
        target.unlink()
        parent.rmdir()
        self.assertIn('Error checking current file contents',
                      loki.run_write(str(target), 'wrong'))
        self.assertFalse(parent.exists())

    def test_non_directory_and_trailing_slash_writes_fail(self):
        target = self.project / 'target'
        target.write_text('original')
        (self.project / 'plain').write_text('not a directory')
        loki.run_read(str(target))
        for relative in ('plain/../target', 'target/'):
            with self.subTest(path=relative):
                self.assertTrue(loki.run_write(relative, 'wrong').startswith('Error:'))
        self.assertEqual(target.read_text(), 'original')

    def test_symlink_loops_fail_without_replacing_links(self):
        a = self.project / 'a'
        b = self.project / 'b'
        a.symlink_to('b')
        b.symlink_to('a')
        self.assertTrue(loki.run_write(str(a), 'wrong').startswith('Error:'))
        self.assertTrue(a.is_symlink())
        self.assertTrue(b.is_symlink())

    def test_file_operands_do_not_expand_tilde(self):
        (self.project / '~').mkdir()
        (self.project / '~' / 'target').write_text('literal')
        home = self.root / 'home'
        home.mkdir()
        (home / 'target').write_text('home target')
        with mock.patch.dict(os.environ, {'HOME': str(home)}):
            self.assertIn('literal', loki.run_read('~/target'))
            self.assertIn('Successfully', loki.run_write('~/target', 'updated'))
        self.assertEqual((home / 'target').read_text(), 'home target')

    @unittest.skipUnless(os.path.isdir('/proc/self/fd'), 'needs proc fd paths')
    def test_unlinked_fd_read_does_not_authorize_a_different_named_file(self):
        original = self.project / 'unlinked'
        original.write_text('same bytes')
        with open(original) as held:
            original.unlink()
            fd_path = f'/proc/self/fd/{held.fileno()}'
            self.assertIn('same bytes', loki.run_read(fd_path))
            other = self.project / 'unlinked (deleted)'
            other.write_text('same bytes')
            result = loki.run_write(str(other), 'unreviewed replacement')
            self.assertTrue(result.startswith('Error:'), result)
            self.assertEqual(other.read_text(), 'same bytes')

    def test_cd_validates_path_without_changing_process_cwd(self):
        original_cwd = os.getcwd()
        path = str(self.project) + '/link/..'
        result = loki.change_shell_cwd(path)
        self.assertTrue(os.path.samefile(result, self.elsewhere))
        self.assertEqual(os.getcwd(), original_cwd)
        (self.elsewhere / 'target').write_text('correct cwd')
        self.assertIn('correct cwd', loki.run_read('target'))
        with self.assertRaises(FileNotFoundError):
            loki.change_shell_cwd(str(self.project) + '/missing/..')
        self.assertEqual(loki.current_cwd(), result)

    def test_image_lookup_and_invalid_traversals(self):
        correct = b'\x89PNG\r\n\x1a\ncorrect'
        (self.elsewhere / 'image.png').write_bytes(correct)
        (self.project / 'image.png').write_bytes(b'not an image')
        image = terminal_frontend.load_image_attachment(
            'link/../image.png', base_dir=str(self.project))
        self.assertEqual(image.byte_size, len(correct))
        for path in ('missing/../image.png', 'image.png/'):
            with self.subTest(path=path):
                with self.assertRaises(terminal_frontend.ImageAttachmentError):
                    terminal_frontend.load_image_attachment(
                        path, base_dir=str(self.project))

    def test_resume_paths_preserve_traversal_and_save_through_symlink(self):
        literal = str(self.project) + '/link/../session.json'
        resolved = savefiles.resolve_chat_log_path(
            literal, str(self.project), str(self.root / 'logs'), loki._resolve_path)
        self.assertEqual(resolved, literal)
        alias = self.elsewhere / 'session.json'
        alias.symlink_to('actual.json')
        (self.elsewhere / 'actual.json').write_text('old')
        self.session.replace_transcript([], [], {}, {}, resolved)
        loki._atomic_write_text(self.session.chat_log_path, 'new')
        self.assertTrue(alias.is_symlink())
        self.assertEqual((self.elsewhere / 'actual.json').read_text(), 'new')
        # Resumed sessions already retain the selected save target. Do not
        # change that policy if the original alias is later redirected.
        (self.elsewhere / 'other.json').write_text('unrelated')
        alias.unlink()
        alias.symlink_to('other.json')
        loki._atomic_write_text(self.session.chat_log_path, 'saved again')
        self.assertEqual((self.elsewhere / 'actual.json').read_text(), 'saved again')
        self.assertEqual((self.elsewhere / 'other.json').read_text(), 'unrelated')
