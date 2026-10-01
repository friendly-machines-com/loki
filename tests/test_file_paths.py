"""Path lookup regressions and the existing atomic-write/symlink contract."""

import base64
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from loki_agent import loki, savefiles, sessions, terminal_frontend


class StatIdentityTests(unittest.TestCase):

    def test_observation_handles_ctime_changes_by_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'observed'
            payload = b'known independently hashed content'
            path.write_bytes(payload)
            original = path.stat()
            for platform in ('posix', 'nt'):
                for expected in (None, original):
                    with self.subTest(platform=platform,
                                      expected_stat=expected is not None):
                        calls = 0

                        def changing_metadata(fd):
                            nonlocal calls
                            calls += 1
                            return SimpleNamespace(
                                st_dev=original.st_dev,
                                st_ino=original.st_ino,
                                st_size=original.st_size,
                                st_mtime_ns=original.st_mtime_ns,
                                st_ctime_ns=original.st_ctime_ns + calls)

                        # Real bytes/open, substituted metadata: test the
                        # observation consumer, not a private tuple layout.
                        with mock.patch.object(loki.os, 'name', platform), \
                                mock.patch.object(loki.os, 'fstat',
                                                  side_effect=changing_metadata):
                            if platform == 'posix':
                                with self.assertRaisesRegex(
                                        OSError, 'changed while it was being read'):
                                    loki._file_observation(str(path), expected)
                            else:
                                observed = loki._file_observation(str(path), expected)
                                self.assertEqual(observed.digest,
                                                 hashlib.sha256(payload).hexdigest())
                                self.assertEqual(observed.size, len(payload))
                                self.assertEqual(calls, 2)


class FilePathTests(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        self.elsewhere = self.root / 'elsewhere'
        (self.elsewhere / 'child').mkdir(parents=True)
        (self.project / 'link').symlink_to(
            self.elsewhere / 'child', target_is_directory=True)
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
        # Authorization is per spelling: the record names the alias the caller
        # read, and nothing recorded a read of the other name.
        self.assertIn(loki._file_key(str(alias)), loki.file_state)
        self.assertNotIn(loki._file_key(str(other)), loki.file_state)
        self.assertIn('You must Read', loki.run_write(str(other), 'wrong'))
        self.assertEqual(other.read_text(), 'same contents')

    def test_a_replaced_alias_is_caught_by_what_was_read(self):
        # Authorization is per spelling, and what it authorises is decided by
        # the observation recorded at the read: a replacement whose contents
        # are not the contents that were read is refused, while a byte-identical
        # replacement clobbers nothing and is allowed.
        alias = self.project / 'alias'
        original = self.project / 'original'
        original.write_text('same contents')
        alias.symlink_to('original')
        self.assertIn('same contents', loki.run_read(str(alias)))
        alias.unlink()
        alias.write_text('different contents')
        self.assertIn('changed on disk', loki.run_write(str(alias), 'wrong'))
        self.assertEqual(alias.read_text(), 'different contents')
        alias.write_text('same contents')
        self.assertIn('Successfully', loki.run_write(str(alias), 'updated'))
        self.assertEqual(alias.read_text(), 'updated')

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
        if os.name != 'posix':
            # Windows refuses to open a dangling symlink for writing; the link
            # and the absent target are left untouched.
            self.assertTrue(result.startswith('Error:'), result)
            self.assertTrue(alias.is_symlink())
            self.assertFalse((self.elsewhere / 'new-directory').exists())
            return
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

    def test_cwd_read_edit_write_publish_atomically_through_aliases(self):
        process_cwd = os.getcwd()
        for chained in (False, True):
            with self.subTest(chained=chained):
                case = f'case-{int(chained)}'
                for base in (self.project, self.elsewhere):
                    (base / case).mkdir()
                # POSIX follows link/.. into elsewhere; Windows cancels it
                # lexically into project. The other copy must never change.
                selected_base = (self.elsewhere if os.name == 'posix'
                                 else self.project)
                other_base = (self.project if os.name == 'posix'
                              else self.elsewhere)
                directory = selected_base / case
                target = directory / 'text.txt'
                other = other_base / case / 'text.txt'
                target.write_text('old')
                target.chmod(0o640)
                other.write_text('unrelated target')
                alias = directory / 'alias'
                links = [alias]
                if chained:
                    intermediate = directory / 'intermediate'
                    intermediate.symlink_to('text.txt')
                    alias.symlink_to('intermediate')
                    links.append(intermediate)
                else:
                    alias.symlink_to('text.txt')
                link_operands = {link: os.readlink(link) for link in links}
                old_link = directory / 'hard-link'
                os.link(target, old_link)
                original_inode = target.stat().st_ino
                literal_directory = str(self.project) + f'/link/../{case}'
                selected = loki.change_shell_cwd(literal_directory)
                self.assertTrue(os.path.samefile(selected, directory))
                self.assertEqual(os.getcwd(), process_cwd)
                operand = literal_directory + '/alias'

                # Reading the terminus does not authorize another spelling.
                loki.run_read(str(target))
                self.assertIn('You must Read',
                              loki.run_edit(operand, 'old', 'edited'))
                self.assertEqual(target.read_text(), 'old')
                self.assertIn('old', loki.run_read(operand))
                self.assertIn('old', loki.run_read('alias'))
                real_open = loki.private_files.open_directory
                real_create = loki.private_files.create_exclusive_at
                opened, created = [], []

                def observe_open(name):
                    handle = real_open(name)
                    opened.append(handle)
                    return handle

                def observe_create(handle, name, mode):
                    created.append(handle)
                    return real_create(handle, name, mode)

                with mock.patch.object(loki.private_files, 'open_directory',
                                       side_effect=observe_open), \
                        mock.patch.object(loki.private_files,
                                          'create_exclusive_at',
                                          side_effect=observe_create):
                    self.assertIn('Successfully',
                                  loki.run_edit(operand, 'old', 'edited'))
                    self.assertEqual(target.read_text(), 'edited')
                    edited_inode = target.stat().st_ino
                    self.assertNotEqual(edited_inode, original_inode)
                    self.assertEqual(old_link.read_text(), 'old')
                    self.assertIn('Successfully',
                                  loki.run_write(operand, 'written'))
                self.assertEqual(len(opened), 2)
                self.assertEqual(created, opened)
                self.assertNotEqual(target.stat().st_ino, edited_inode)
                self.assertEqual(target.read_text(), 'written')
                self.assertEqual(old_link.read_text(), 'old')
                self.assertEqual(other.read_text(), 'unrelated target')
                for link, original in link_operands.items():
                    self.assertTrue(link.is_symlink())
                    self.assertEqual(os.readlink(link), original)
                    self.assertTrue(os.path.samefile(link, target))
                if os.name == 'posix':
                    self.assertEqual(target.stat().st_mode & 0o777, 0o640)
                self.assertFalse(list(directory.glob('.*.tmp')))
                self.assertEqual(os.getcwd(), process_cwd)
                invalid = literal_directory + '/missing/..'
                if os.name == 'posix':
                    with self.assertRaises(FileNotFoundError):
                        loki.change_shell_cwd(invalid)
                else:
                    self.assertTrue(os.path.samefile(
                        loki.change_shell_cwd(invalid), directory))
                self.assertTrue(os.path.samefile(loki.current_cwd(), directory))
                self.assertFalse((directory / 'missing').exists())

    def test_dangling_final_symlink_creates_target_and_preserves_link(self):
        alias = self.project / 'alias'
        alias.symlink_to('../elsewhere/new/subdir/target')
        result = loki.run_write(str(alias), 'created')
        if os.name != 'posix':
            # As above: a dangling symlink cannot be opened for writing here.
            self.assertTrue(result.startswith('Error:'), result)
            self.assertTrue(alias.is_symlink())
            self.assertFalse((self.elsewhere / 'new').exists())
            return
        self.assertIn('Successfully', result)
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
        original_link = os.readlink(alias)
        if os.name != 'posix':
            # Windows refuses a symlink whose target contains `..` (WinError
            # 123), so the write is refused and both files stay untouched.
            self.assertTrue(
                loki.run_write(str(alias), 'updated').startswith('Error:'))
            self.assertEqual(right.read_text(), 'right')
        else:
            # POSIX applies `..` in the link target through the kernel.
            loki.run_read(str(alias))
            self.assertIn('Successfully', loki.run_write(str(alias), 'updated'))
            self.assertEqual(right.read_text(), 'updated')
        self.assertEqual(wrong.read_text(), 'wrong')
        self.assertTrue(alias.is_symlink())
        self.assertEqual(os.readlink(alias), original_link)

    def test_invalid_traversal_does_not_read_a_simplified_path(self):
        target = self.project / 'target'
        target.write_text('must not read')
        (self.project / 'plain').write_text('not a directory')
        if os.name != 'posix':
            # Windows cancels `..` lexically before the kernel sees the path, so
            # the missing/non-directory component is dropped and the path
            # simplifies to the existing target, which is then read normally;
            # the run confirms that outcome.  A trailing slash on a file is
            # invalid outright.
            for relative in ('missing/../target', 'plain/../target'):
                self.assertIn('must not read', loki.run_read(relative))
            self.assertTrue(loki.run_read('target/').startswith('Error:'))
            return
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
        if os.name != 'posix':
            # `plain/..` cancels lexically here, so the target is written; only
            # the trailing slash on a file is invalid.  Read-before-write keys
            # on the caller's spelling, so the spelling being written must be
            # the one that was read.
            loki.run_read('plain/../target')
            self.assertIn('Successfully',
                          loki.run_write('plain/../target', 'wrong'))
            self.assertEqual(target.read_text(), 'wrong')
            self.assertTrue(
                loki.run_write('target/', 'wrong').startswith('Error:'))
            return
        for relative in ('plain/../target', 'target/'):
            with self.subTest(path=relative):
                self.assertTrue(loki.run_write(relative, 'wrong').startswith('Error:'))
        self.assertEqual(target.read_text(), 'original')

    def test_symlink_loops_fail_without_replacing_links(self):
        a = self.project / 'a'
        b = self.project / 'b'
        a.symlink_to('b')
        b.symlink_to('a')
        original_links = {a: os.readlink(a), b: os.readlink(b)}
        original_names = set(self.project.iterdir())
        self.assertTrue(loki.run_write(str(a), 'wrong').startswith('Error:'))
        for link, operand in original_links.items():
            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), operand)
        self.assertEqual(set(self.project.iterdir()), original_names)

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
        self.assertEqual((self.project / '~' / 'target').read_text(), 'updated')

    def test_unlinked_fd_read_does_not_authorize_a_different_named_file(self):
        if os.name != 'posix':
            # No /proc/self/fd here; the property this keeps -- a name that was
            # never read cannot authorize a write -- is asserted directly.
            other = self.project / 'unlinked (deleted)'
            other.write_text('same bytes')
            result = loki.run_write(str(other), 'unreviewed replacement')
            self.assertTrue(result.startswith('Error:'), result)
            self.assertEqual(other.read_text(), 'same bytes')
            return
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

    def test_image_lookup_and_invalid_traversals(self):
        correct = b'\x89PNG\r\n\x1a\ncorrect'
        (self.elsewhere / 'image.png').write_bytes(correct)
        (self.project / 'image.png').write_bytes(b'not an image')
        if os.name != 'posix':
            # Lexical `..` selects the project's non-image copy here.
            with self.assertRaises(terminal_frontend.ImageAttachmentError):
                terminal_frontend.load_image_attachment(
                    'link/../image.png', base_dir=str(self.project))
        else:
            image = terminal_frontend.load_image_attachment(
                'link/../image.png', base_dir=str(self.project))
            self.assertEqual(image.byte_size, len(correct))
            self.assertEqual(image.media_type, 'image/png')
            self.assertEqual(base64.b64decode(image.encoded_data), correct)
            self.assertTrue(os.path.samefile(
                image.path, self.elsewhere / 'image.png'))
            (self.elsewhere / 'image.png').write_bytes(b'changed after staging')
            self.assertEqual(base64.b64decode(image.encoded_data), correct)
        for path in ('missing/../image.png', 'image.png/'):
            with self.subTest(path=path):
                with self.assertRaises(terminal_frontend.ImageAttachmentError):
                    terminal_frontend.load_image_attachment(
                        path, base_dir=str(self.project))

    def test_new_chat_uses_the_path_it_was_given(self):
        # The path handed to a new chat is composed by the runtime, not taken
        # from a model operand, so it is used as given.  A name that goes
        # through a link is therefore followed again when the log is written --
        # the write is not pinned to the object that name denoted here.
        literal = str(self.project) + '/link/../new-session.json'
        loki.new_chat_log(literal)
        self.assertEqual(self.session.chat_log_path, literal)
        base = self.project if os.name != 'posix' else self.elsewhere
        loki._atomic_write_text(self.session.chat_log_path, 'snapshot')
        self.assertEqual((base / 'new-session.json').read_text(), 'snapshot')

    def test_resume_writes_through_the_path_it_was_given(self):
        # A resumed session stores the path it was handed and writes through
        # that name.  Whatever the name denotes at write time is what is
        # written: a link redirected after load is followed, and the write is
        # not pinned to the object the name denoted when the log was loaded.
        # Containment is unaffected -- the container's grants still bound where
        # a write can land.
        literal = str(self.project) + '/link/../session.json'
        resolved = savefiles.resolve_chat_log_path(
            literal, str(self.project), str(self.root / 'logs'), loki._resolve_path)
        self.assertEqual(resolved, literal)
        # POSIX applies `..` in the name through the kernel and lands in
        # elsewhere; Windows cancels it lexically and lands in the project.
        base = self.project if os.name != 'posix' else self.elsewhere
        alias = base / 'session.json'
        alias.symlink_to('actual.json')
        (base / 'actual.json').write_text('old')
        self.session.replace_transcript([], [], {}, {}, resolved)
        self.assertEqual(self.session.chat_log_path, literal)
        loki._atomic_write_text(self.session.chat_log_path, 'new')
        self.assertTrue(alias.is_symlink())
        self.assertEqual((base / 'actual.json').read_text(), 'new')
        # Redirecting the link afterwards redirects later writes.
        (base / 'other.json').write_text('unrelated')
        alias.unlink()
        alias.symlink_to('other.json')
        loki._atomic_write_text(self.session.chat_log_path, 'saved again')
        self.assertEqual((base / 'other.json').read_text(), 'saved again')
        self.assertEqual((base / 'actual.json').read_text(), 'new')
