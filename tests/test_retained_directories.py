"""Trusted state owners follow retained objects, not stolen ancestor names."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from job_fixtures import job_manager
from loki_agent import authentications, credential_storages, files, jobs
from loki_agent import private_files, response_headers, settings, windows_api


def token(label):
    return authentications.OpenAITokenSet(
        access_token=label, refresh_token=label + '-refresh', id_token='id',
        account_id='account', expires_at=10**12, last_refresh=100)


class RetainedDirectoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parent = self.root / 'state'
        self.parent.mkdir()
        self.moved = self.root / 'moved-state'

    def replace_parent(self):
        self.decoy = self.parent
        try:
            self.parent.rename(self.moved)
        except PermissionError as error:
            if os.name != 'nt' or error.winerror not in [
                    windows_api.ERROR_ACCESS_DENIED, windows_api.ERROR_SHARING_VIOLATION]:
                raise
            # A retained child-directory handle can prevent an ancestor rename
            # on Windows. Do not close it to manufacture a successful attack.
            # Prove ordinary rename works, report the blocked setup, and still
            # exercise all reads/writes against the unchanged original tree.
            control = self.root / 'unheld-control'
            control.mkdir()
            control.rename(self.root / 'renamed-control')
            print(json.dumps({'ancestor_rename': 'blocked', 'winerror': error.winerror,
                              'decoy_replacement_exercised': False}), flush=True)
            self.moved = self.parent
            self.decoy = self.root / 'separate-decoy'
        self.decoy.mkdir()

    async def test_job_reads_writes_and_new_jobs_ignore_recreated_parent(self):
        manager = job_manager(str(self.parent / 'jobs'), cleanup=self.addCleanup,
                              async_cleanup=self.addAsyncCleanup)
        # The manager must already be anchored before the first child launch.
        self.replace_parent()
        decoy_session = self.decoy / 'jobs' / manager.session_id
        decoy_job = decoy_session / '1'
        decoy_job.mkdir(parents=True)
        (decoy_job / 'stdout.log').write_text('FORGED STDOUT')
        (decoy_job / 'stderr.log').write_text('FORGED STDERR')
        (decoy_job / 'job.json').write_text('FORGED METADATA')
        first, status, stdout, stderr = await manager.run_exec([
            sys.executable, '-c', "import sys; print('real output'); print('real error', file=sys.stderr)"])
        self.assertEqual(status, 'completed')
        self.assertEqual(stdout, 'real output' + os.linesep)
        self.assertEqual(stderr, 'real error' + os.linesep)
        self.assertIn('real output', manager.job_status(first.id))
        self.assertNotIn('FORGED', manager.job_status(first.id))
        manager._append_spool(first, 'stderr.log', b'owner diagnostic\n')
        first.description = 'updated metadata'
        manager._write_metadata(first)
        original = self.moved / 'jobs' / manager.session_id / first.id
        self.assertIn('owner diagnostic', (original / 'stderr.log').read_text())
        self.assertEqual(json.loads((original / 'job.json').read_text())['description'], 'updated metadata')
        self.assertEqual((decoy_job / 'job.json').read_text(), 'FORGED METADATA')
        second, _, second_out, _ = await manager.run_exec([sys.executable, '-c', "print('second')"])
        self.assertEqual(second_out, 'second' + os.linesep)
        self.assertTrue((original.parent / second.id / 'stdout.log').exists())
        self.assertFalse((decoy_session / second.id).exists())
        with self.assertRaises(OSError):
            await manager.run_exec([str(self.root / 'does-not-exist')])
        self.assertIn('launch failed', (original.parent / '3' / 'stderr.log').read_text())
        self.assertEqual((decoy_job / 'stdout.log').read_text(), 'FORGED STDOUT')
        await manager.aclose()
        self.assertIsNone(manager._session_fd)
        with self.assertRaisesRegex(ValueError, 'closed'):
            manager.job_status(first.id)

    async def test_job_output_handles_allow_overwrite_and_truncate_but_refuse_replacement(self):
        """Jobs own their capture bytes, but creation must not overwrite an existing entry."""
        directory = private_files.open_directory(str(self.parent))
        self.addCleanup(private_files.close, directory)
        for name in ['stdout.log', 'stderr.log']:
            with self.subTest(name=name):
                with jobs.JobManager._open_spool_stream(directory, name) as stream:
                    stream.write(b'original longer output')
                    stream.flush()
                    stream.seek(0)
                    stream.write(b'new')
                    stream.flush()
                    stream.truncate(3)
                self.assertEqual(jobs._read_spool_tail(directory, name, 100), 'new')
                with self.assertRaises(FileExistsError):
                    jobs.JobManager._open_spool_stream(directory, name)
                self.assertEqual(jobs._read_spool_tail(directory, name, 100), 'new')

    async def test_launched_jobs_rewrite_captures_without_redirecting_job_state(self):
        """Inherited output is mutable data; real process exit, not saved text, owns status."""
        manager = job_manager(str(self.parent), cleanup=self.addCleanup,
                              async_cleanup=self.addAsyncCleanup)
        script = (
            "import os,sys\n"
            "for fd, text in [[1, sys.argv[1]], [2, sys.argv[2]]]:\n"
            "    data = text.encode('ascii')\n"
            "    os.write(fd, b'original longer output' * 32)\n"
            "    os.lseek(fd, 0, os.SEEK_SET)\n"
            "    os.write(fd, data)\n"
            "    os.ftruncate(fd, len(data))\n"
            "sys.exit(7)\n")
        completed = []
        for background in [False, True]:
            with self.subTest(background=background):
                stdout = json.dumps({'status': 'running', 'pid': 0, 'background': background})
                stderr = json.dumps({'status': 'failed', 'exit_code': 0, 'background': background})
                command = [sys.executable, '-c', script, stdout, stderr]
                if background:
                    job = await manager.run_background_exec(command)
                    await job.process.wait()
                else:
                    job, status, output, errors = await manager.run_exec(command)
                    self.assertEqual(status, 'completed')
                    self.assertEqual(output, stdout)
                    self.assertEqual(errors, stderr)
                report = manager.job_status(job.id)
                self.assertEqual(job.status, 'exited')
                self.assertEqual(job.exit_code, 7)
                self.assertEqual(job.pid, job.process.pid)
                self.assertTrue(report.startswith(
                    f'job_id: {job.id}\nstatus: exited\npid: {job.process.pid}\n'), report)
                self.assertIn('\nexit_code: 7\n', report)
                with open(job.metadata_path, 'w', encoding='utf-8') as stream:
                    json.dump({'id': 'forged', 'status': 'running', 'pid': 0, 'exit_code': 0}, stream)
                self.assertIn(f'{job.id}. status=exited pid={job.pid} exit=7 ', manager.list_jobs())
                self.assertIn('\nexit_code: 7\n', manager.job_status(job.id))
                completed.append([job, stdout, stderr])
                # A later launch gets its own endpoints. Rewriting those must
                # not redirect the earlier job's captures.
                for other, expected_output, expected_errors in completed:
                    with open(other.stdout_path, 'rb') as stream:
                        self.assertEqual(stream.read(), expected_output.encode('ascii'))
                    with open(other.stderr_path, 'rb') as stream:
                        self.assertEqual(stream.read(), expected_errors.encode('ascii'))
                    self.assertEqual(other.exit_code, 7)
                    self.assertEqual(other.pid, other.process.pid)
                    self.assertEqual(manager.jobs[other.id].pid, other.process.pid)

    async def test_credential_reads_refresh_approvals_and_logout_ignore_decoy(self):
        storage = credential_storages.JsonCredentialStorage(str(self.parent / 'credentials'))
        self.addCleanup(storage.close)
        original_tokens = token('original')
        await storage.store_openai_login(original_tokens)
        ref = authentications.CredentialRef.environment('API_KEY')
        await storage.approve_destinations(ref, ['https://original.invalid/v1'])
        self.replace_parent()
        decoy = credential_storages.JsonCredentialStorage(str(self.decoy / 'credentials'))
        self.addCleanup(decoy.close)
        await decoy.store_openai_login(token('FORGED'))
        await decoy.approve_destinations(ref, ['https://evil.invalid/v1'])
        self.assertEqual(storage.load_openai_subscription().tokens, original_tokens.normalized())
        self.assertEqual(storage.approved_destinations(ref), frozenset(['https://original.invalid/v1']))

        async def refresh(value):
            self.assertEqual(value, original_tokens.refresh_token)
            return authentications.RefreshResult(access_token='rotated', refresh_token='rotated-refresh')

        rotated = await storage.rotate_openai_subscription(original_tokens, refresh=refresh)
        self.assertEqual(rotated.access_token, 'rotated')
        reopened = credential_storages.JsonCredentialStorage(str(self.moved / 'credentials'))
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.load_openai_subscription().tokens, rotated)
        self.assertEqual(decoy.load_openai_subscription().tokens.access_token, 'FORGED')
        self.assertTrue(await storage.remove_openai_subscription())
        self.assertIsNone(reopened.load_openai_subscription())
        self.assertEqual(decoy.load_openai_subscription().tokens.access_token, 'FORGED')
        storage.close()
        with self.assertRaisesRegex(credential_storages.CredentialStorageError, 'closed'):
            storage.load_document()
        with self.assertRaisesRegex(credential_storages.CredentialStorageError, 'closed'):
            storage.directory_handle

    async def test_response_reads_merges_and_publication_ignore_decoy(self):
        store = response_headers.Store(str(self.parent / 'headers.json'))
        self.addCleanup(store.close)
        observe = store.observer('https://provider.invalid/v1', None, 'model')
        observe(200, {'x-original': 'original'})
        await store.save()
        self.replace_parent()
        decoy = response_headers.Store(str(self.decoy / 'headers.json'))
        self.addCleanup(decoy.close)
        decoy.observer('https://forged.invalid/v1', None, 'forged')(200, {'x-forged': 'FORGED'})
        await decoy.save()
        self.assertEqual(store.snapshot()['endpoints'][0]['endpoint'], 'https://provider.invalid/v1')
        observe(200, {'x-updated': 'updated'})
        await store.save()
        reopened = response_headers.Store(str(self.moved / 'headers.json'))
        self.addCleanup(reopened.close)
        headers = reopened.snapshot()['endpoints'][0]['headers']
        self.assertEqual(headers['x-original']['value'], 'original')
        self.assertEqual(headers['x-updated']['value'], 'updated')
        self.assertEqual(decoy.snapshot()['endpoints'][0]['endpoint'], 'https://forged.invalid/v1')
        store.close()
        with self.assertRaisesRegex(ValueError, 'closed'):
            store.snapshot()

    async def test_background_monitor_is_joined_before_handle_close_without_killing_job(self):
        manager = job_manager(str(self.parent / 'jobs'), cleanup=self.addCleanup,
                              async_cleanup=self.addAsyncCleanup)
        release = self.root / 'release'
        job = await manager.run_background_exec([
            sys.executable, '-c',
            "import pathlib,sys,time\nwhile not pathlib.Path(sys.argv[1]).exists(): time.sleep(.01)",
            str(release)])
        try:
            await manager.aclose()
            self.assertIsNone(job.process.returncode)
            self.assertIsNone(manager._session_fd)
            self.assertFalse(manager._monitors)
        finally:
            release.touch()
            await job.process.wait()
            await manager.aclose()

    async def _assert_publication_collision(self, file, publish, error, retained):
        """Exclusive-create failure grants no ownership of the entry that blocked it."""
        with open(file, 'rb') as stream:
            original = stream.read()
        create = private_files.create_exclusive_at
        foreign = b'belongs to another publication'
        collisions = []

        def collide(root, name, mode, **options):
            fd = create(root, name, mode, **options)
            try:
                private_files.write(fd, foreign)
            finally:
                private_files.close(fd)
            collisions.append(name)
            return create(root, name, mode, **options)  # The real exclusive create must fail.

        with mock.patch.object(private_files, 'create_exclusive_at', side_effect=collide):
            with self.assertRaises(error):
                result = publish()
                if asyncio.iscoroutine(result):
                    await result
        self.assertEqual(len(collisions), 1)
        with open(file, 'rb') as stream:
            self.assertEqual(stream.read(), original)
        fd = private_files.open_read_at(retained, collisions[0])
        try:
            self.assertEqual(private_files.read(fd, len(foreign) + 1), foreign)
        finally:
            private_files.close(fd)

    async def test_editor_collision_never_deletes_an_unacquired_temporary(self):
        file = os.path.join(self.parent, 'editor')
        with open(file, 'wb') as stream:
            stream.write(b'original editor contents')
        directory = private_files.open_directory(str(self.parent))
        self.addCleanup(private_files.close, directory)
        await self._assert_publication_collision(
            file, lambda: files.atomic_write_text(file, 'replacement'), FileExistsError, directory)

    async def test_metadata_collision_never_deletes_an_unacquired_temporary(self):
        file = os.path.join(self.parent, 'metadata')
        with open(file, 'wb') as stream:
            stream.write(b'original metadata')
        directory = private_files.open_directory(str(self.parent))
        self.addCleanup(private_files.close, directory)
        await self._assert_publication_collision(
            file, lambda: files.atomic_write_text_at(directory, 'metadata', 'replacement'), FileExistsError, directory)

    async def test_settings_collision_never_deletes_an_unacquired_temporary(self):
        file = os.path.join(self.parent, 'settings')
        with open(file, 'wb') as stream:
            stream.write(b'[terminal]\nshow_bash_stdout = false\n')
        directory = private_files.open_directory(str(self.parent))
        self.addCleanup(private_files.close, directory)
        await self._assert_publication_collision(file, lambda: settings.update_user_settings(
            {'terminal.show_bash_stdout': True}, file=file), settings.SettingsError, directory)

    async def test_headers_collision_never_deletes_an_unacquired_temporary(self):
        file = os.path.join(self.parent, 'headers')
        with open(file, 'wb') as stream:
            stream.write(b'{"version": 1, "endpoints": []}')
        store = response_headers.Store(file)
        self.addCleanup(store.close)
        store.observer('https://example.invalid/v1', None, 'model')(200, {'x-test': 'value'})
        await self._assert_publication_collision(file, store.save, FileExistsError, store.directory_handle)

    async def test_credentials_collision_never_deletes_an_unacquired_temporary(self):
        storage = credential_storages.JsonCredentialStorage(os.path.join(self.parent, 'credentials'))
        self.addCleanup(storage.close)
        await storage.store_openai_login(token('original'))
        await self._assert_publication_collision(
            storage.file_path, lambda: storage.store_openai_login(token('replacement')),
            credential_storages.CredentialStorageError, storage.directory_handle)

    async def test_relative_atomic_publication_cleans_failed_temporary(self):
        directory = private_files.open_directory(str(self.parent))
        self.addCleanup(private_files.close, directory)
        files.atomic_write_text_at(directory, 'metadata', 'original')
        self.replace_parent()
        with mock.patch.object(private_files, 'replace_at', side_effect=OSError('publish failed')):
            with self.assertRaisesRegex(OSError, 'publish failed'):
                files.atomic_write_text_at(directory, 'metadata', 'replacement')
        self.assertEqual((self.moved / 'metadata').read_text(), 'original')
        self.assertEqual([entry.name for entry in self.moved.iterdir()], ['metadata'])
        self.assertEqual(list(self.decoy.iterdir()), [])

    async def test_existing_and_missing_state_directories_need_no_ancestor_stat(self):
        with mock.patch.object(private_files.os, 'makedirs', side_effect=AssertionError('ancestor mkdir')):
            existing = private_files.open_directory_tree(str(self.parent))
        private_files.close(existing)
        target = self.parent / 'new' / 'nested'
        with mock.patch.object(private_files.os, 'stat', side_effect=AssertionError('ancestor stat')):
            directory = private_files.open_directory_tree(str(target))
        try:
            self.assertTrue(private_files.information(directory).directory)
            self.assertTrue(target.is_dir())
        finally:
            private_files.close(directory)

    async def test_atomic_temporary_collision_does_not_delete_another_file(self):
        directory = private_files.open_directory(str(self.parent))
        self.addCleanup(private_files.close, directory)
        temporary = self.parent / '.metadata.fixed.tmp'
        temporary.write_text('another owner')
        with mock.patch.object(files.uuid, 'uuid4', return_value=type('Id', (), {'hex': 'fixed'})()):
            with self.assertRaises(FileExistsError):
                files.atomic_write_text_at(directory, 'metadata', 'replacement')
        self.assertEqual(temporary.read_text(), 'another owner')
        self.assertFalse((self.parent / 'metadata').exists())

    @unittest.skipUnless(os.name == 'posix', 'POSIX symlink and FIFO primitives')
    async def test_job_symlink_directory_and_nonregular_spools_are_refused(self):
        manager = job_manager(str(self.parent / 'jobs'), cleanup=self.addCleanup,
                              async_cleanup=self.addAsyncCleanup)
        target = self.root / 'target'
        target.mkdir()
        (Path(manager.session_dir) / '1').symlink_to(target, target_is_directory=True)
        with self.assertRaises(OSError):
            await manager.run_exec([sys.executable, '-c', "print('must not run')"])
        self.assertEqual(list(target.iterdir()), [])
        job, _, _, _ = await manager.run_exec([sys.executable, '-c', "print('ok')"])
        spool = Path(job.stdout_path)
        spool.unlink()
        os.mkfifo(spool)
        with self.assertRaisesRegex(OSError, 'regular file'):
            manager.job_status(job.id)


class WindowsRelativeDirectoryTests(unittest.TestCase):
    def test_generic_file_information_does_not_interpret_a_credential_dacl(self):
        from loki_agent import _private_files_windows as backend

        result = windows_api.ByHandleFileInformation()
        result.dwFileAttributes = windows_api.FileAttribute.FILE_ATTRIBUTE_DIRECTORY
        result.nFileSizeLow = 7
        with mock.patch.object(windows_api, 'by_handle_file_information', return_value=result), \
                mock.patch.object(windows_api, 'handle_dacl_sddl',
                                  side_effect=AssertionError('credential privacy parser')):
            info = backend.information(77)
        self.assertTrue(info.directory)
        self.assertFalse(info.reparse_point)
        self.assertEqual(info.size, 7)
        self.assertIsInstance(info, private_files.FileInfo)
        self.assertNotIsInstance(info, private_files.FileFacts)

    def test_relative_directory_handle_refuses_reparse_points_and_closes_on_error(self):
        from loki_agent import _private_files_windows as backend

        for attributes, succeeds in [[0x10, True], [0x410, False]]:
            with self.subTest(attributes=attributes), mock.patch.object(
                    windows_api, 'nt_create_file', return_value=77) as create, mock.patch.object(
                    windows_api, 'by_handle_file_information',
                    return_value=type('Information', (), {'dwFileAttributes': attributes})()), mock.patch.object(
                    windows_api, 'close_handle') as close:
                if succeeds:
                    self.assertEqual(backend.open_directory_at(12, 'child'), 77)
                    close.assert_not_called()
                else:
                    with self.assertRaisesRegex(OSError, 'reparse'):
                        backend.open_directory_at(12, 'child')
                    close.assert_called_once_with(77)
                self.assertEqual(list(create.call_args.args[:2]), [12, 'child'])
                self.assertTrue(create.call_args.args[4] & windows_api.NtCreateOptions.FILE_DIRECTORY_FILE)
