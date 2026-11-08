"""Path lookup regressions and the existing atomic-write/symlink contract."""

import asyncio
import base64
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
from config_fixtures import setUpModule as set_up_config
from response_header_fixtures import setUpModule as set_up_response_headers

from loki_agent import loki, private_files, savefiles, sessions, terminal_frontend
from loki_agent import files as file_operations


def setUpModule():
    set_up_config()
    set_up_response_headers()


class FileObservationTests(unittest.TestCase):

    def test_read_preserves_bounded_utf8_newlines_and_full_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, payload, expected in [
                    ['text', b'known\r\ncontent', '1\tknown\n2\tcontent'],
                    ['binary', b'\xff\x00binary', 'binary (8 bytes)'],
                    ['incomplete', b'\xc3', 'binary (1 bytes)'],
                    ['empty', b'', 'is empty'],
                    ['image.png', b'\xffimage', 'is an image (6 bytes)'],
                    ['capped', b'visible\r\n' + b'x' * 65526 + b'\xff', '1\tvisible']]:
                path = os.path.join(directory, name)
                with open(path, 'wb') as stream:
                    stream.write(payload)
                observations = {}
                with self.subTest(name=name), \
                        mock.patch.object(file_operations, 'READ_CHAR_CAP',
                                          8 if name == 'capped' else file_operations.READ_CHAR_CAP):
                    result = file_operations.run_read(path, cwd=directory, observations=observations)
                    self.assertIn(expected, result)
                    observed = observations[path]
                    self.assertEqual(observed.digest, hashlib.sha256(payload).hexdigest())
                    self.assertEqual(observed.size, len(payload))

    def test_regular_reader_keeps_one_descriptor_owner_on_every_exit(self):
        """The wrapper borrows the descriptor; this scope closes it even if wrapping fails."""
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, 'owned-reader')
            with open(target, 'wb') as stream:
                stream.write(b'known\r\ncontent')
            native_open, native_close = os.open, os.close
            for phase in ['success', 'blocking', 'wrapper', 'body']:
                if phase == 'blocking' and os.name != 'posix':
                    continue  # Windows uses binary descriptors, not this mode switch.
                opened = []

                def opening(*args, **kwargs):
                    descriptor = native_open(*args, **kwargs)
                    opened.append(descriptor)
                    return descriptor

                def wrapping(*args, **kwargs):
                    if os.name == 'posix':
                        self.assertTrue(os.get_blocking(args[0]))
                    if phase == 'wrapper':
                        raise OSError('wrapper failed')
                    return open(*args, **kwargs)

                with self.subTest(phase=phase), \
                        mock.patch.object(file_operations.os, 'open', side_effect=opening), \
                        mock.patch.object(file_operations.os, 'close', wraps=native_close) as closing, \
                        mock.patch.object(file_operations.os, 'set_blocking', wraps=os.set_blocking,
                                          side_effect=OSError('blocking failed') if phase == 'blocking' else None), \
                        mock.patch.object(file_operations, 'open', side_effect=wrapping, create=True):
                    if phase == 'success':
                        with file_operations._open_regular_file(target) as source:
                            self.assertEqual(source.read(), 'known\ncontent')
                    else:
                        with self.assertRaisesRegex(OSError, phase):
                            with file_operations._open_regular_file(target):
                                raise OSError('body failed')
                    self.assertEqual(len(opened), 1)
                    closing.assert_called_once_with(opened[0])


class FilePathTests(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        self.elsewhere = self.root / 'elsewhere'
        (self.elsewhere / 'child').mkdir(parents=True)
        self.session = sessions.Session(shell_cwd=str(self.project))
        self.addCleanup(self.session.response_headers.close)
        session_patch = mock.patch.object(loki, '_DEFAULT_SESSION', self.session)
        session_patch.start()
        self.addCleanup(session_patch.stop)

    def directory_link(self):
        # Only link-specific invariants need this fixture. Requiring symlink
        # creation in setUp prevented unrelated file tests from running on a
        # standard Windows account without that privilege.
        (self.project / 'link').symlink_to(self.elsewhere / 'child', target_is_directory=True)

    def assert_refused(self, tool, arguments, diagnostic=None):
        # Exercise the real result boundary rather than treating error-looking
        # prose from a text helper as evidence that the tool failed.
        result = asyncio.run(loki.dispatch_tool_async(tool, arguments))
        self.assertFalse(result['ok'], result)
        if diagnostic is not None:
            self.assertIn(diagnostic, result['content'])
        return result

    def test_read_history_belongs_to_each_conversation(self):
        """One conversation's Read must not satisfy another conversation's guard."""
        target = os.path.join(self.project, 'conversation-file')
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write('reviewed')
        self.assertIn('reviewed', loki.run_read(target))
        other = sessions.Session(shell_cwd=str(self.project))
        self.addCleanup(other.response_headers.close)
        with mock.patch.object(loki, '_DEFAULT_SESSION', other):
            for tool, arguments in [
                    ['Write', {'file_path': target, 'content': 'wrong'}],
                    ['Edit', {'file_path': target, 'old_string': 'reviewed', 'new_string': 'wrong'}]]:
                with self.subTest(tool=tool):
                    result = asyncio.run(loki.dispatch_tool_async(tool, arguments))
                    self.assertFalse(result['ok'], result)
                    self.assertIn('You must Read', result['content'])
        with open(target, encoding='utf-8') as stream:
            self.assertEqual(stream.read(), 'reviewed')
        self.assertIn('Successfully', loki.run_edit(target, 'reviewed', 'updated'))

    def test_new_and_loaded_conversations_discard_live_read_history(self):
        """Replacing a transcript discards live records; saved JSON cannot restore them."""
        target = os.path.join(self.project, 'conversation-file')
        saved = os.path.join(self.project, 'saved-conversation.json')
        forged = {'shell_cwd': str(self.project), 'file_observations': {
            target: {'digest': hashlib.sha256(b'reviewed').hexdigest(), 'size': 8}}}
        with open(saved, 'w', encoding='utf-8') as stream:
            stream.write(savefiles.serialize_chat_log([], [], forged))
        for transition in ['new', 'load']:
            with self.subTest(transition=transition):
                with open(target, 'w', encoding='utf-8') as stream:
                    stream.write('reviewed')
                self.assertIn('reviewed', loki.run_read(target))
                if transition == 'new':
                    loki.new_chat_log(os.path.join(self.project, 'new-conversation.json'))
                else:
                    loki.load_chat_log(saved)
                result = asyncio.run(loki.dispatch_tool_async(
                    'Write', {'file_path': target, 'content': 'wrong'}))
                self.assertFalse(result['ok'], result)
                self.assertIn('You must Read', result['content'])
                self.assertIn('reviewed', loki.run_read(target))
                self.assertIn('Successfully', loki.run_edit(target, 'reviewed', 'updated'))

    def test_read_history_remains_bounded_and_eviction_requires_read_again(self):
        """A conversation's bookkeeping must not grow with every filename ever touched."""
        targets = [os.path.join(self.project, f'bounded-{number}') for number in range(3)]
        with mock.patch.object(file_operations, 'READ_PATHS_LIMIT', 2):
            for target in targets:
                self.assertIn('Successfully', loki.run_write(target, 'contents'))
            self.assertEqual(len(self.session.file_observations), 2)
            self.assert_refused('Write', {'file_path': targets[0], 'content': 'wrong'}, 'You must Read')
            self.assertIn('contents', loki.run_read(targets[0]))
            self.assertIn('Successfully', loki.run_edit(targets[0], 'contents', 'updated'))
            self.assertEqual(len(self.session.file_observations), 2)

    def test_failed_reads_do_not_record_observations_but_empty_regular_files_do(self):
        """Only a successful opened-file read can satisfy the later overwrite guard."""
        missing = os.path.join(self.project, 'absent-read')
        for target in [missing, str(self.project)]:
            with self.subTest(target=target):
                result = asyncio.run(loki.dispatch_tool_async('Read', {'file_path': target}))
                self.assertFalse(result['ok'], result)
                self.assertNotIn(target, self.session.file_observations)
        empty = os.path.join(self.project, 'empty-read')
        with open(empty, 'wb'):
            pass
        result = asyncio.run(loki.dispatch_tool_async('Read', {'file_path': empty}))
        self.assertTrue(result['ok'], result)
        self.assertIn('empty', result['content'])
        self.assertIn('Successfully', loki.run_write(empty, 'now populated'))

    def test_write_and_edit_keep_target_selected_before_validation(self):
        original = self.project / 'original'
        other = self.project / 'other'
        alias = self.project / 'alias'
        atomic_write = file_operations.atomic_write_text

        def redirect_before_publication(path, content):
            alias.unlink()
            alias.symlink_to('other')
            return atomic_write(path, content)

        for operation in ['Write', 'Edit']:
            with self.subTest(operation=operation):
                original.write_text('reviewed')
                other.write_text('unrelated')
                if alias.is_symlink():
                    alias.unlink()
                alias.symlink_to('original')
                self.assertIn('reviewed', loki.run_read(str(alias)))
                with mock.patch.object(file_operations, 'atomic_write_text',
                                       side_effect=redirect_before_publication):
                    if operation == 'Write':
                        result = loki.run_write(str(alias), 'updated')
                    else:
                        result = loki.run_edit(str(alias), 'reviewed', 'updated')
                self.assertIn('Successfully', result)
                self.assertEqual(original.read_text(), 'updated')
                self.assertEqual(other.read_text(), 'unrelated')

    def test_change_between_steps_is_detected_beyond_the_display_cap(self):
        """Capped Unicode display must not discard the later full-file change warning."""
        target = os.path.join(self.project, 'read-selection')
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write('\u00e9\u732b value-a')
        original = os.stat(target)
        with mock.patch.object(file_operations, 'READ_CHAR_CAP', 2):
            result = loki.run_read(target)
        self.assertIn('\u00e9\u732b', result)
        self.assertIn('truncated at 2 characters', result)
        self.assertNotIn('value-a', result)
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write('\u00e9\u732b value-b')
        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assert_refused('Write', {'file_path': target, 'content': 'wrong'}, 'changed on disk')
        self.assert_refused('Edit', {'file_path': target, 'old_string': 'value-b',
                                     'new_string': 'updated'}, 'changed on disk')
        with open(target, encoding='utf-8') as stream:
            self.assertEqual(stream.read(), '\u00e9\u732b value-b')
        self.assertIn('value-b', loki.run_read(target))
        self.assertIn('Successfully', loki.run_edit(target, 'value-b', 'updated'))

    def test_read_rejects_a_file_changed_during_its_content_check(self):
        target = self.project / 'changing-read'
        target.write_text('reviewed')
        real_open = open
        changed = False

        def opening(*args, **kwargs):
            stream = real_open(*args, **kwargs)
            if len(args) > 1 and args[1] == 'rb':
                read = stream.read

                def changing_read(*arguments):
                    nonlocal changed
                    content = read(*arguments)
                    if content and not changed:
                        changed = True
                        with real_open(target, 'ab') as writer:
                            writer.write(b' appended')
                    return content

                stream.read = changing_read
            return stream

        with mock.patch.object(file_operations, 'open', side_effect=opening, create=True):
            result = asyncio.run(loki.dispatch_tool_async('Read', {'file_path': str(target)}))
        self.assertTrue(changed)
        self.assertFalse(result['ok'], result)
        self.assertIn('changed while it was being read', result['content'])
        self.assertNotIn(str(target), self.session.file_observations)

    def test_write_rejects_replacement_during_target_selection(self):
        target = self.project / 'selected-write'
        replacement = self.project / 'replacement'
        target.write_text('reviewed')
        replacement.write_text('reviewed')
        loki.run_read(str(target))
        follow = file_operations._follow_final_link
        changed = False

        def replacing_selection(path):
            nonlocal changed
            selected = follow(path)
            if not changed:
                changed = True
                os.replace(replacement, target)
            return selected

        with mock.patch.object(file_operations, '_follow_final_link', side_effect=replacing_selection):
            result = asyncio.run(loki.dispatch_tool_async(
                'Write', {'file_path': str(target), 'content': 'must not publish'}))
        self.assertTrue(changed)
        self.assertFalse(result['ok'], result)
        self.assertIn('path changed during target selection', result['content'])
        self.assertEqual(target.read_text(), 'reviewed')

    def test_edit_validates_the_buffer_it_will_change(self):
        target = os.path.join(self.project, 'edit-selection')
        reviewed = 'old reviewed contents'
        unread = 'old unread contents'
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write(reviewed)
        self.assertIn(reviewed, loki.run_read(target))
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write(unread)
        real_open = open
        interleaved = False

        def opening(path, mode='r', **kwargs):
            stream = real_open(path, mode, **kwargs)
            if mode == 'r' and not interleaved:
                read = stream.read

                def replacing_read(*arguments):
                    nonlocal interleaved
                    data = read(*arguments)
                    interleaved = True
                    with real_open(target, 'w', encoding='utf-8') as replacement:
                        replacement.write(reviewed)
                    return data

                stream.read = replacing_read
            return stream

        with mock.patch.object(file_operations, 'open', side_effect=opening, create=True):
            self.assert_refused('Edit', {'file_path': target, 'old_string': 'old',
                                         'new_string': 'new'}, 'changed on disk')
        self.assertTrue(interleaved)
        with open(target, encoding='utf-8') as stream:
            self.assertEqual(stream.read(), reviewed)

    @unittest.skipUnless(os.name == 'posix', 'native FIFO replacement')
    def test_special_file_replacement_is_not_read_or_overwritten(self):
        target = os.path.join(self.project, 'special-operation')
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write('reviewed')
        self.assertIn('reviewed', loki.run_read(target))
        os.unlink(target)
        os.mkfifo(target)
        for failure, operation in [
                [OSError, lambda: loki.run_read(target)],
                [ValueError, lambda: loki.run_write(target, 'wrong')],
                [OSError, lambda: loki.run_edit(target, 'reviewed', 'wrong')]]:
            with self.subTest(direct_operation=operation), \
                    self.assertRaisesRegex(failure, 'file is not a regular file'):
                operation()
            self.assertTrue(stat.S_ISFIFO(os.stat(target).st_mode))
        calls = [
            ['Read', {'file_path': target}],
            ['Write', {'file_path': target, 'content': 'wrong'}],
            ['Edit', {'file_path': target, 'old_string': 'reviewed', 'new_string': 'wrong'}],
        ]
        for name, arguments in calls:
            with self.subTest(tool=name):
                result = asyncio.run(loki.dispatch_tool_async(name, arguments))
                self.assertFalse(result['ok'], result)
                self.assertIn('file is not a regular file', result['content'])
                self.assertTrue(stat.S_ISFIFO(os.stat(target).st_mode))

    @unittest.skipUnless(os.path.isdir('/proc/self/fd'), 'native open-resource accounting')
    def test_observation_scope_closes_on_binary_success_and_metadata_failure(self):
        target = os.path.join(self.project, 'observation-resources')
        with open(target, 'wb') as stream:
            stream.write(b'\xffbinary')
        before = len(os.listdir('/proc/self/fd'))
        self.assertIn('binary', loki.run_read(target))
        self.assertEqual(len(os.listdir('/proc/self/fd')), before)
        with mock.patch.object(file_operations.os, 'fstat', side_effect=OSError('metadata unavailable')):
            self.assert_refused('Read', {'file_path': target}, 'metadata unavailable')
        self.assertEqual(len(os.listdir('/proc/self/fd')), before)

    def test_edit_keeps_existing_universal_newline_behavior(self):
        target = os.path.join(self.project, 'edit-newlines')
        for newline in ['\r\n', '\r', '\n']:
            with self.subTest(newline=newline):
                with open(target, 'wb') as stream:
                    stream.write(f'old{newline}reviewed{newline}'.encode('utf-8'))
                self.assertIn('old', loki.run_read(target))
                self.assertIn('Successfully', loki.run_edit(target, 'old', 'new'))
                with open(target, 'rb') as stream:
                    self.assertEqual(stream.read(), b'new\nreviewed\n')

    def test_alias_retarget_after_read_does_not_record_other_name(self):
        original = self.project / 'original'
        other = self.project / 'other'
        original.write_text('same contents')
        other.write_text('same contents')
        alias = self.project / 'alias'
        alias.symlink_to('original')
        self.assertIn('same contents', loki.run_read(str(alias)))
        alias.unlink()
        alias.symlink_to('other')
        # The record names the alias the caller read, not the other name.
        self.assertIn(str(alias), self.session.file_observations)
        self.assertNotIn(str(other), self.session.file_observations)
        self.assert_refused('Write', {'file_path': str(other), 'content': 'wrong'}, 'You must Read')
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
        self.assert_refused('Write', {'file_path': str(alias), 'content': 'wrong'}, 'changed on disk')
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

    def test_editor_publication_follows_a_native_parent_directory_link(self):
        """An ordinary filename's parent follows native lookup, unlike a protected state root."""
        self.directory_link()
        target = os.path.join(self.elsewhere, 'child', 'parent-link-file')
        operand = os.path.join(self.project, 'link', 'parent-link-file')
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write('reviewed')
        self.assertIn('reviewed', loki.run_read(operand))
        self.assertIn('Successfully', loki.run_edit(operand, 'reviewed', 'updated'))
        with open(target, encoding='utf-8') as stream:
            self.assertEqual(stream.read(), 'updated')
        self.assertTrue(os.path.islink(os.path.join(self.project, 'link')))
        with self.assertRaises(OSError):
            private_files.open_directory_tree(os.path.join(self.project, 'link'))

    def test_dangling_parent_link_retains_directory_creation_policy(self):
        alias = self.project / 'alias'
        alias.symlink_to('../elsewhere/new-directory')
        if os.name != 'posix':
            # Windows refuses to open a dangling symlink for writing; the link
            # and the absent target are left untouched.
            self.assert_refused('Write', {'file_path': str(alias / 'new-file'), 'content': 'contents'})
            self.assertTrue(alias.is_symlink())
            self.assertFalse((self.elsewhere / 'new-directory').exists())
            return
        self.assertIn('Successfully', loki.run_write(str(alias / 'new-file'), 'contents'))
        self.assertTrue(alias.is_symlink())
        self.assertEqual(
            (self.elsewhere / 'new-directory' / 'new-file').read_text(),
            'contents')

    def test_failed_dotdot_lookup_in_link_target_creates_nothing(self):
        alias = self.project / 'alias'
        alias.symlink_to('missing/child/../../missing/new-file')
        self.assert_refused('Write', {'file_path': str(alias), 'content': 'contents'})
        self.assertFalse((self.project / 'missing').exists())
        self.assertTrue(alias.is_symlink())

    def test_cwd_read_edit_write_publish_atomically_through_aliases(self):
        self.directory_link()
        process_cwd = os.getcwd()
        for chained in [False, True]:
            with self.subTest(chained=chained):
                case = f'case-{int(chained)}'
                for base in [self.project, self.elsewhere]:
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
                self.assert_refused('Edit', {'file_path': operand, 'old_string': 'old',
                                             'new_string': 'edited'}, 'You must Read')
                self.assertEqual(target.read_text(), 'old')
                self.assertIn('old', loki.run_read(operand))
                self.assertIn('old', loki.run_read('alias'))
                real_open = private_files.open_directory
                real_create = private_files.create_exclusive_at
                opened, created = [], []

                def observe_open(name, **options):
                    handle = real_open(name, **options)
                    opened.append(handle)
                    return handle

                def observe_create(handle, name, mode, **options):
                    created.append(handle)
                    return real_create(handle, name, mode, **options)

                with mock.patch.object(private_files, 'open_directory',
                                       side_effect=observe_open), \
                        mock.patch.object(private_files,
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
                self.assertTrue(os.path.samefile(loki.current_shell_cwd(), directory))
                self.assertFalse((directory / 'missing').exists())

    def test_dangling_final_symlink_creates_target_and_preserves_link(self):
        alias = self.project / 'alias'
        alias.symlink_to('../elsewhere/new/subdir/target')
        if os.name != 'posix':
            # As above: a dangling symlink cannot be opened for writing here.
            self.assert_refused('Write', {'file_path': str(alias), 'content': 'created'})
            self.assertTrue(alias.is_symlink())
            self.assertFalse((self.elsewhere / 'new').exists())
            return
        self.assertIn('Successfully', loki.run_write(str(alias), 'created'))
        self.assertTrue(alias.is_symlink())
        self.assertEqual(
            (self.elsewhere / 'new/subdir/target').read_text(), 'created')

    def test_final_symlink_target_dotdot_uses_kernel_traversal(self):
        self.directory_link()
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
            self.assert_refused('Write', {'file_path': str(alias), 'content': 'updated'})
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
            for relative in ['missing/../target', 'plain/../target']:
                self.assertIn('must not read', loki.run_read(relative))
            self.assert_refused('Read', {'file_path': 'target/'})
            return
        for relative in ['missing/../target', 'plain/../target', 'target/']:
            with self.subTest(path=relative):
                with self.assertRaises(OSError):
                    with open(str(self.project) + '/' + relative):
                        pass
                result = self.assert_refused('Read', {'file_path': relative})
                self.assertNotIn('must not read', result['content'])

    def test_parent_creation_cannot_bypass_read_before_overwrite(self):
        target = self.project / 'target'
        target.write_text('must preserve')
        self.assert_refused('Write', {'file_path': 'missing/../target', 'content': 'overwrite'})
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
        result = asyncio.run(loki.dispatch_tool_async(
            'Write', {'file_path': str(target), 'content': 'wrong'}))
        self.assertFalse(result['ok'], result)
        self.assertIn('checking current file contents', result['content'])
        self.assertFalse(parent.exists())

    def test_non_directory_and_trailing_slash_writes_fail(self):
        target = self.project / 'target'
        target.write_text('original')
        (self.project / 'plain').write_text('not a directory')
        loki.run_read(str(target))
        previous_cwd = self.session.shell_cwd
        for number in [errno.EACCES, errno.EIO, errno.ELOOP]:
            failure = OSError(number, 'link lookup refused')
            with self.subTest(errno=number), mock.patch.object(
                    file_operations.os, 'readlink', side_effect=failure):
                self.assert_refused('Write', {'file_path': str(target), 'content': 'wrong'}, 'link lookup refused')
                self.assert_refused('Edit', {'file_path': str(target), 'old_string': 'original',
                                             'new_string': 'wrong'}, 'link lookup refused')
                with self.assertRaises(OSError) as refused:
                    file_operations.atomic_write_text(str(target), 'wrong')
                self.assertIs(refused.exception, failure)
                with self.assertRaises(OSError) as refused:
                    loki.change_shell_cwd(str(self.project))
                self.assertIs(refused.exception, failure)
            self.assertEqual(target.read_text(), 'original')
            self.assertEqual(self.session.shell_cwd, previous_cwd)
            self.assertFalse(list(self.project.glob('.*.tmp')))
        if os.name != 'posix':
            # `plain/..` cancels lexically here, so the target is written; only
            # the trailing slash on a file is invalid.  Read-before-write keys
            # on the caller's spelling, so the spelling being written must be
            # the one that was read.
            loki.run_read('plain/../target')
            self.assertIn('Successfully',
                          loki.run_write('plain/../target', 'wrong'))
            self.assertEqual(target.read_text(), 'wrong')
            self.assert_refused('Write', {'file_path': 'target/', 'content': 'wrong'})
            return
        for relative in ['plain/../target', 'target/']:
            with self.subTest(path=relative):
                self.assert_refused('Write', {'file_path': relative, 'content': 'wrong'})
        self.assertEqual(target.read_text(), 'original')

    def test_symlink_loops_fail_without_replacing_links(self):
        a = self.project / 'a'
        b = self.project / 'b'
        a.symlink_to('b')
        b.symlink_to('a')
        original_links = {a: os.readlink(a), b: os.readlink(b)}
        original_names = set(self.project.iterdir())
        self.assert_refused('Write', {'file_path': str(a), 'content': 'wrong'})
        for link, operand in original_links.items():
            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), operand)
        self.assertEqual(set(self.project.iterdir()), original_names)

    def test_file_operands_do_not_expand_tilde(self):
        literal = os.path.join(self.project, '~')
        literal_target = os.path.join(literal, 'target')
        home = os.path.join(self.root, 'home')
        home_target = os.path.join(home, 'target')
        os.mkdir(literal)
        os.mkdir(home)
        with open(literal_target, 'w', encoding='utf-8') as stream:
            stream.write('literal')
        with open(home_target, 'w', encoding='utf-8') as stream:
            stream.write('home target')
        process_cwd = os.getcwd()
        with mock.patch.dict(os.environ, {'HOME': home}):
            self.assertIn('literal', loki.run_read('~/target'))
            self.assertIn('Successfully', loki.run_write('~/target', 'updated'))
            self.assertTrue(os.path.samefile(loki.change_shell_cwd('~'), literal))
        with open(home_target, encoding='utf-8') as stream:
            self.assertEqual(stream.read(), 'home target')
        with open(literal_target, encoding='utf-8') as stream:
            self.assertEqual(stream.read(), 'updated')
        self.assertEqual(os.getcwd(), process_cwd)

    def test_cd_preserves_quoted_edge_spaces_in_the_operand(self):
        target = os.path.join(self.project, ' edge spaces ')
        os.mkdir(target)
        operand = f'"{target}"'
        self.assertTrue(loki.change_shell_cwd_from_text(operand))
        self.assertTrue(os.path.samefile(loki.current_shell_cwd(), target))

    def test_unlinked_fd_read_does_not_authorize_a_different_named_file(self):
        if os.name != 'posix':
            # No /proc/self/fd here; the property this keeps -- a name that was
            # never read cannot authorize a write -- is asserted directly.
            other = self.project / 'unlinked (deleted)'
            other.write_text('same bytes')
            self.assert_refused('Write', {'file_path': str(other), 'content': 'unreviewed replacement'})
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
            self.assert_refused('Write', {'file_path': str(other), 'content': 'unreviewed replacement'})
            self.assertEqual(other.read_text(), 'same bytes')

    def test_image_lookup_and_invalid_traversals(self):
        self.directory_link()
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
        for path in ['missing/../image.png', 'image.png/']:
            with self.subTest(path=path):
                with self.assertRaises(terminal_frontend.ImageAttachmentError):
                    terminal_frontend.load_image_attachment(
                        path, base_dir=str(self.project))

    @unittest.skipUnless(os.name == 'posix', 'native FIFO and regular-file symlink lookup')
    def test_saved_log_readers_validate_the_opened_file_and_keep_alias_names(self):
        target = os.path.join(self.project, 'chat-target.json')
        alias = os.path.join(self.project, 'chat-alias.json')
        document = savefiles.serialize_chat_log([], [], {'shell_cwd': str(self.project)})
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write(document)
        os.symlink('chat-target.json', alias)
        loki.load_chat_log(alias)
        self.assertEqual(self.session.chat_log_path, alias)
        self.assertTrue(os.path.islink(alias))
        entries = {entry.name: entry.path for entry in savefiles.chat_log_entries(str(self.project))}
        self.assertEqual(entries, {'chat-target.json': target, 'chat-alias.json': alias})
        self.assertEqual(savefiles.text_for(alias), document)
        os.unlink(target)
        os.mkfifo(target)
        for path in [target, alias]:
            with self.subTest(path=path):
                with self.assertRaisesRegex(OSError, 'not a regular file'):
                    loki.load_chat_log(path)
                self.assertEqual(savefiles.text_for(path), '')
                self.assertEqual(self.session.chat_log_path, alias)
        self.assertEqual(savefiles.filtered_chat_log_entries('not present', str(self.project)), [])
        os.unlink(target)
        os.mkdir(target)
        with self.assertRaisesRegex(OSError, 'not a regular file'):
            loki.load_chat_log(alias)
        self.assertEqual(savefiles.text_for(alias), '')

    def test_preparsed_saved_log_keeps_the_admitted_snapshot(self):
        filename = os.path.join(str(self.project), 'prepared.json')
        admitted = savefiles.serialize_chat_log([], [], {
            'shell_cwd': str(self.project), 'reasoning_effort': 'high'})
        with open(filename, 'w', encoding='utf-8') as stream:
            stream.write(admitted)
        with savefiles.open_chat_log(filename) as (stream, _facts):
            loaded = savefiles.read_chat_log(stream)
        with open(filename, 'w', encoding='utf-8') as stream:
            stream.write(savefiles.serialize_chat_log([], [], {
                'shell_cwd': str(self.project), 'reasoning_effort': 'low'}))

        loki.load_chat_log(filename, loaded)

        self.assertEqual(self.session.reasoning_effort_preference, 'high')
        self.assertEqual(self.session.session_state['reasoning_effort'], 'high')
        self.assertEqual(self.session.chat_log_path, filename)

    def test_malformed_saved_log_closes_reader_and_preserves_session(self):
        filename = os.path.join(str(self.project), 'malformed.json')
        with open(filename, 'w', encoding='utf-8') as stream:
            stream.write('{invalid')
        previous = os.path.join(str(self.project), 'previous.json')
        self.session.chat_log_path = previous
        self.session.reasoning_effort_preference = 'high'
        self.session.session_state = {'reasoning_effort': 'high'}
        with mock.patch.object(savefiles, 'read_chat_log', wraps=savefiles.read_chat_log) as parse, \
                self.assertRaises(json.JSONDecodeError):
            loki.load_chat_log(filename)

        parse.assert_called_once()
        self.assertTrue(parse.call_args.args[0].closed)
        self.assertEqual(self.session.chat_log_path, previous)
        self.assertEqual(self.session.reasoning_effort_preference, 'high')
        self.assertEqual(self.session.session_state, {'reasoning_effort': 'high'})
        self.assertEqual(self.session.transcript_items, [])
        self.assertEqual(self.session.shell_cwd, str(self.project))

    def test_new_chat_name_selection_does_not_create_directories_before_save(self):
        parent = os.path.join(self.project, 'new-parent')
        filename = os.path.join(parent, 'session.json')
        loki.new_chat_log(filename)
        self.assertEqual(self.session.chat_log_path, filename)
        self.assertFalse(os.path.exists(parent))
        loki.save_chat_log()
        self.assertTrue(os.path.isdir(parent))
        self.assertTrue(os.path.isfile(filename))

    def test_new_chat_uses_the_path_it_was_given(self):
        self.directory_link()
        # The path handed to a new chat is composed by the runtime, not taken
        # from a model operand, so it is used as given.  A name that goes
        # through a link is therefore followed again when the log is written --
        # the write is not pinned to the object that name denoted here.
        literal = str(self.project) + '/link/../new-session.json'
        loki.new_chat_log(literal)
        self.assertEqual(self.session.chat_log_path, literal)
        base = self.project if os.name != 'posix' else self.elsewhere
        file_operations.atomic_write_text(self.session.chat_log_path, 'snapshot')
        self.assertEqual((base / 'new-session.json').read_text(), 'snapshot')

    def test_resume_writes_through_the_path_it_was_given(self):
        self.directory_link()
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
        file_operations.atomic_write_text(self.session.chat_log_path, 'new')
        self.assertTrue(alias.is_symlink())
        self.assertEqual((base / 'actual.json').read_text(), 'new')
        # Redirecting the link afterwards redirects later writes.
        (base / 'other.json').write_text('unrelated')
        alias.unlink()
        alias.symlink_to('other.json')
        file_operations.atomic_write_text(self.session.chat_log_path, 'saved again')
        self.assertEqual((base / 'other.json').read_text(), 'saved again')
        self.assertEqual((base / 'actual.json').read_text(), 'new')
