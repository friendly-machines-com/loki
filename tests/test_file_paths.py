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

    def test_write_and_edit_keep_target_selected_before_validation(self):
        original = self.project / 'original'
        other = self.project / 'other'
        alias = self.project / 'alias'
        atomic_write = loki._atomic_write_text

        def redirect_before_publication(path, content):
            alias.unlink()
            alias.symlink_to('other')
            return atomic_write(path, content)

        for operation in ('Write', 'Edit'):
            with self.subTest(operation=operation):
                original.write_text('reviewed')
                other.write_text('unrelated')
                if alias.is_symlink():
                    alias.unlink()
                alias.symlink_to('original')
                self.assertIn('reviewed', loki.run_read(str(alias)))
                with mock.patch.object(loki, '_atomic_write_text',
                                       side_effect=redirect_before_publication):
                    if operation == 'Write':
                        result = loki.run_write(str(alias), 'updated')
                    else:
                        result = loki.run_edit(str(alias), 'reviewed', 'updated')
                self.assertIn('Successfully', result)
                self.assertEqual(original.read_text(), 'updated')
                self.assertEqual(other.read_text(), 'unrelated')

    def test_snapshot_key_is_not_recomputed_after_observation(self):
        original = self.project / 'original'
        other = self.project / 'other'
        original.write_text('same contents')
        other.write_text('same contents')
        alias = self.project / 'alias'
        alias.symlink_to('original')
        observe = loki._file_observation

        def redirect_after_observation(path, expected_stat=None):
            result = observe(path, expected_stat=expected_stat)
            alias.unlink()
            alias.symlink_to('other')
            return result

        with mock.patch.object(loki, '_file_observation',
                               side_effect=redirect_after_observation):
            self.assertIn('same contents', loki.run_read(str(alias)))
        self.assertIn(str(original), loki.file_state)
        self.assertNotIn(str(other), loki.file_state)
        self.assertIn('You must Read', loki.run_write(str(other), 'wrong'))
        self.assertEqual(other.read_text(), 'same contents')

    def test_unverified_read_key_does_not_authorize_alias_replacement(self):
        original = self.project / 'original'
        other = self.project / 'other'
        alias = self.project / 'alias'
        original.write_text('same contents')
        other.write_text('same contents')
        alias.symlink_to('original')
        state_key = loki._file_state_key

        def redirect_during_key_selection(path, expected_stat=None):
            alias.unlink()
            alias.symlink_to('other')
            key = state_key(path, expected_stat=expected_stat)
            alias.unlink()
            alias.symlink_to('original')
            return key

        with mock.patch.object(loki, '_file_state_key',
                               side_effect=redirect_during_key_selection):
            self.assertIn('same contents', loki.run_read(str(alias)))
        alias.unlink()
        alias.write_text('same contents')
        self.assertIn('You must Read', loki.run_write(str(alias), 'wrong'))
        self.assertEqual(alias.read_text(), 'same contents')

    def test_cd_keeps_selected_directory_when_alias_changes(self):
        first = self.project / 'first'
        second = self.project / 'second'
        first.mkdir()
        second.mkdir()
        alias = self.project / 'alias'
        alias.symlink_to('first')
        loki.change_shell_cwd(str(alias))
        alias.unlink()
        alias.symlink_to('second')
        self.assertIn('Successfully', loki.run_write('new-file', 'contents'))
        self.assertEqual((first / 'new-file').read_text(), 'contents')
        self.assertFalse((second / 'new-file').exists())

    def test_dangling_parent_link_retains_directory_creation_policy(self):
        alias = self.project / 'alias'
        alias.symlink_to('../elsewhere/new-directory')
        result = loki.run_write(str(alias / 'new-file'), 'contents')
        self.assertIn('Successfully', result)
        self.assertTrue(alias.is_symlink())
        self.assertEqual(
            (self.elsewhere / 'new-directory' / 'new-file').read_text(),
            'contents')

    def test_failed_dotdot_lookup_in_link_target_creates_nothing(self):
        alias = self.project / 'alias'
        alias.symlink_to('missing/child/../../missing/new-file')
        result = loki.run_write(str(alias), 'contents')
        self.assertTrue(result.startswith('Error:'), result)
        self.assertFalse((self.project / 'missing').exists())
        self.assertTrue(alias.is_symlink())

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
        self.assertFalse((self.project / 'missing').exists())

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

    def test_new_chat_selects_save_target_before_later_alias_changes(self):
        literal = str(self.project) + '/link/../new-session.json'
        loki.new_chat_log(literal)
        self.assertEqual(self.session.chat_log_path,
                         str(self.elsewhere / 'new-session.json'))
        (self.project / 'link').unlink()
        (self.project / 'link').symlink_to(self.project)
        loki._atomic_write_text(self.session.chat_log_path, 'snapshot')
        self.assertEqual((self.elsewhere / 'new-session.json').read_text(),
                         'snapshot')
        self.assertFalse((self.root / 'new-session.json').exists())

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
