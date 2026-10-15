"""Trusted state owners follow retained objects, not stolen ancestor names."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import authentications, credential_storages, files, jobs
from loki_agent import private_files, response_headers, windows_api


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
        self.parent.rename(self.moved)
        self.parent.mkdir()

    async def test_job_reads_writes_and_new_jobs_ignore_recreated_parent(self):
        manager = jobs.JobManager(str(self.parent / 'jobs'))
        self.addAsyncCleanup(manager.aclose)
        # The manager must already be anchored before the first child launch.
        self.replace_parent()
        decoy_session = self.parent / 'jobs' / manager.session_id
        decoy_job = decoy_session / '1'
        decoy_job.mkdir(parents=True)
        (decoy_job / 'stdout.log').write_text('FORGED STDOUT')
        (decoy_job / 'stderr.log').write_text('FORGED STDERR')
        (decoy_job / 'job.json').write_text('FORGED METADATA')
        first, status, stdout, stderr = await manager.run_exec([
            sys.executable, '-c', "import sys; print('real output'); print('real error', file=sys.stderr)"])
        self.assertEqual(status, 'completed')
        self.assertEqual(stdout, 'real output\n')
        self.assertEqual(stderr, 'real error\n')
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
        self.assertEqual(second_out, 'second\n')
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

    async def test_credential_reads_refresh_approvals_and_logout_ignore_decoy(self):
        storage = credential_storages.JsonCredentialStorage(str(self.parent / 'credentials'))
        self.addCleanup(storage.close)
        original_tokens = token('original')
        await storage.store_openai_login(original_tokens)
        ref = authentications.CredentialRef.environment('API_KEY')
        await storage.approve_destinations(ref, ['https://original.invalid/v1'])
        self.replace_parent()
        decoy = credential_storages.JsonCredentialStorage(str(self.parent / 'credentials'))
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

    async def test_response_reads_merges_and_publication_ignore_decoy(self):
        store = response_headers.Store(str(self.parent / 'headers.json'))
        self.addCleanup(store.close)
        observe = store.observer('https://provider.invalid/v1', None, 'model')
        observe(200, {'x-original': 'original'})
        await store.save()
        self.replace_parent()
        decoy = response_headers.Store(str(self.parent / 'headers.json'))
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
        manager = jobs.JobManager(str(self.parent / 'jobs'))
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
        self.assertEqual(list(self.parent.iterdir()), [])

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
        manager = jobs.JobManager(str(self.parent / 'jobs'))
        self.addAsyncCleanup(manager.aclose)
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
