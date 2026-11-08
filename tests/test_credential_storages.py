import asyncio
from contextlib import closing
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from loki_agent import authentications
from loki_agent import private_files
from loki_agent import credential_storages
from loki_agent import file_locks
from loki_agent import paths
from loki_agent import windows_api


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tokens(access="access-a", refresh="refresh-a"):
    return authentications.OpenAITokenSet(
        access_token=access,
        refresh_token=refresh,
        id_token="id-a",
        account_id="account-a",
        expires_at=10**12,
        last_refresh=100,
    )


def _marker_lines(path):
    """How many process markers a file has, or zero before it exists."""
    try:
        with open(path, encoding="ascii") as stream:
            return len(stream.readlines())
    except FileNotFoundError:
        return 0


class JsonCredentialStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = os.path.join(
            self.temporary.name, "loki", "credentials")
        self.storage = credential_storages.JsonCredentialStorage(
            self.directory)
        self.addCleanup(self.storage.close)

    def _symlink(self, target, link, *, target_is_directory=False):
        try:
            os.symlink(target, link, target_is_directory=target_is_directory)
        except OSError as error:
            if sys.platform != 'win32' or getattr(error, 'winerror', None) != 1314:
                raise
            self.skipTest('this Windows account cannot create the symbolic-link fixture')

    @unittest.skipUnless(sys.platform == 'win32', 'native Windows owner-mismatch control')
    async def test_native_existing_different_owner_is_refused_without_repair(self):
        selected = self.temporary.name + os.sep + 'existing-owner-control'
        os.mkdir(selected, 0o700)
        handle = private_files.open_directory(selected)
        try:
            if private_files.describe(handle).owned_by_current_user:
                self.skipTest('this token already creates user-owned objects by default')
            owner = windows_api.handle_owner_sid(handle)
            dacl = windows_api.handle_dacl_sddl(handle)
        finally:
            private_files.close(handle)
        with self.assertRaises(credential_storages.CredentialStorageError):
            credential_storages.JsonCredentialStorage(selected)
        handle = private_files.open_directory(selected)
        try:
            self.assertEqual(windows_api.handle_owner_sid(handle), owner)
            self.assertEqual(windows_api.handle_dacl_sddl(handle), dacl)
        finally:
            private_files.close(handle)

    async def test_single_component_relative_directory_is_created_and_validated(self):
        """A valid bare directory must not fail because parent extraction produced ''.

        The helper process isolates its cwd change; the real constructor must
        request private creation and validate the selected directory object.
        """
        code = r'''
import os,stat,sys
from loki_agent import credential_storages
os.chdir(sys.argv[1])
store = credential_storages.JsonCredentialStorage('credentials')
try:
    store.ensure_directory()
    assert store.load_openai_subscription() is None
    if os.name == 'posix':
        assert stat.S_IMODE(os.stat('credentials').st_mode) == 0o700
finally:
    store.close()
print('relative private store validated')
'''
        process = subprocess.run(
            [sys.executable, '-c', code, self.temporary.name], cwd=ROOT,
            stdin=subprocess.DEVNULL, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('relative private store validated', process.stdout)

    async def test_existing_private_directory_is_not_changed(self):
        self.storage.ensure_directory()
        with mock.patch.object(os, 'chmod', side_effect=AssertionError(
                'existing permissions must not be changed')):
            if os.name == 'posix':
                before = os.stat(self.directory)
                self.storage.ensure_directory()
                after = os.stat(self.directory)
                self.assertTrue(os.path.samestat(before, after))
                self.assertEqual(
                    stat.S_IMODE(after.st_mode), stat.S_IMODE(before.st_mode))
            else:
                # Windows has no mode bits; "not changed" is the DACL string.
                before = windows_api.dacl_sddl(self.directory)
                self.storage.ensure_directory()
                self.assertEqual(windows_api.dacl_sddl(self.directory), before)

    async def test_existing_shared_directory_is_rejected_without_repair(self):
        os.makedirs(self.directory, exist_ok=True)
        if os.name == 'posix':
            modes = [0o755, 0o750, 0o770]
            for mode in modes:
                with self.subTest(mode=oct(mode)):
                    os.chmod(self.directory, mode)
                    with mock.patch.object(
                            os, 'chmod', side_effect=AssertionError(
                                'existing permissions must not be changed')):
                        with self.assertRaisesRegex(
                                credential_storages.CredentialStorageError,
                                'adjust its permissions.*Loki will not change'):
                            await self.storage.store_openai_login(tokens())
                    self.assertEqual(
                        stat.S_IMODE(os.stat(self.directory).st_mode), mode)
                    self.assertFalse(os.path.exists(self.storage.file_path))
                    self.assertFalse(os.path.exists(self.storage.lock_path))
        else:
            # POSIX widens the mode; Windows widens the DACL.  The DACL is set
            # through the editor's mutation, the only caller that rewrites one.
            from loki_agent import windows_containers, windows_state

            widened = windows_state.add_package_ace(
                windows_api.dacl_sddl(self.directory),
                'S-1-5-21-0-0-0-1004', windows_state.Access.READ)
            windows_containers.set_dacl_sddl(self.directory, widened)
            with self.assertRaisesRegex(
                    credential_storages.CredentialStorageError,
                    'adjust its permissions.*Loki will not change'):
                await self.storage.store_openai_login(tokens())
            self.assertFalse(os.path.exists(self.storage.file_path))
            self.assertFalse(os.path.exists(self.storage.lock_path))

    async def test_login_failure_preserves_previous_tokens(self):
        await self.storage.store_openai_login(tokens())
        with mock.patch.object(
                self.storage, "_write_document_at",
                side_effect=credential_storages.CredentialStorageError(
                    "disk full")):
            with self.assertRaises(
                    credential_storages.CredentialStorageError):
                await self.storage.store_openai_login(
                    tokens("access-b", "refresh-b"))

        self.assertEqual(
            self.storage.load_openai_subscription().tokens,
            tokens().normalized(),
        )

    async def test_rejects_group_readable_json(self):
        await self.storage.store_openai_login(tokens())
        if os.name == 'posix':
            os.chmod(self.storage.file_path, 0o640)
        else:
            # The Windows equivalent of "group readable" is a DACL that grants
            # another trustee.
            from loki_agent import windows_containers, windows_state

            widened = windows_state.add_package_ace(
                windows_api.dacl_sddl(self.storage.file_path),
                'S-1-5-21-0-0-0-1004', windows_state.Access.READ)
            windows_containers.set_dacl_sddl(self.storage.file_path, widened)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "group or other"):
            self.storage.load_document()

    async def test_rejects_symlinked_credential_directory(self):
        real = os.path.join(self.temporary.name, "real")
        os.mkdir(real)
        alias = os.path.join(self.temporary.name, 'linked-credentials')
        self._symlink(real, alias, target_is_directory=True)

        # POSIX lstat sees a link as neither file nor directory; a Windows
        # directory link is a directory that is also a reparse point.  Both
        # refuse, with the message each mechanism can state.
        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "not a directory|reparse point"):
            credential_storages.JsonCredentialStorage(alias)

    async def test_rejects_symlinked_json_file(self):
        self.storage.ensure_directory()
        target = os.path.join(self.temporary.name, "target")
        with open(target, "w", encoding="ascii") as stream:
            stream.write("{}")
        self._symlink(target, self.storage.file_path)

        # POSIX refuses the open with O_NOFOLLOW; Windows opens the link itself
        # and refuses it as a reparse point.  Both refuse, with a message each
        # mechanism can state.
        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "could not open credential JSON|reparse point"):
            self.storage.load_document()

    async def test_rejects_symlinked_lock_file(self):
        self.storage.ensure_directory()
        target = os.path.join(self.temporary.name, "target")
        with open(target, "w", encoding="ascii") as stream:
            stream.write("")
        self._symlink(target, self.storage.lock_path)

        # Same divergence as the JSON file: O_NOFOLLOW refuses on POSIX, the
        # reparse-point check refuses on Windows.
        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "could not open credential lock|reparse point"):
            await self.storage.store_openai_login(tokens())

    async def test_failed_lock_open_is_refused(self):
        self.storage.ensure_directory()
        self._symlink('unused-target', self.storage.lock_path)
        for _ in range(3):
            # O_NOFOLLOW refuses the lock open on POSIX; on Windows the
            # link opens and is refused as a reparse point.
            with self.assertRaisesRegex(
                    credential_storages.CredentialStorageError,
                    'could not open credential lock|reparse point'):
                await self.storage.store_openai_login(tokens())

    async def test_setup_failure_after_lock_open_propagates(self):
        directory_fd = self.storage._directory_fd
        lock_fd = self.storage._open_lock_at(directory_fd)
        with mock.patch.object(self.storage, '_open_lock_at',
                               return_value=lock_fd), \
                mock.patch.object(file_locks, 'try_lock_exclusive',
                                  side_effect=RuntimeError('setup failed')):
            with self.assertRaisesRegex(RuntimeError, 'setup failed'):
                await self.storage.store_openai_login(tokens())

    async def test_lock_close_error_propagates(self):
        directory_fd = self.storage._directory_fd
        lock_fd = self.storage._open_lock_at(directory_fd)
        # Patch the platform close the storage actually calls: os.close on
        # POSIX, CloseHandle on Windows.  Patching os.close would inject
        # nothing on Windows, where the token is not a descriptor.
        close = private_files.close

        def close_then_report_error(token):
            close(token)
            if token == lock_fd:
                raise OSError('lock close failed')

        with mock.patch.object(self.storage, '_open_lock_at',
                               return_value=lock_fd), \
                mock.patch.object(private_files, 'close',
                                  side_effect=close_then_report_error):
            with self.assertRaisesRegex(OSError, 'lock close failed'):
                async with self.storage._locked_document():
                    pass

    async def test_cancellation_while_waiting_is_propagated(self):
        directory_fd = self.storage._directory_fd
        lock_fd = self.storage._open_lock_at(directory_fd)
        with mock.patch.object(self.storage, '_open_lock_at',
                               return_value=lock_fd), \
                mock.patch.object(file_locks, 'try_lock_exclusive',
                                  side_effect=BlockingIOError):
            task = asyncio.create_task(self.storage.store_openai_login(tokens()))
            try:
                await asyncio.sleep(0)
                self.assertFalse(task.done())
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

    async def test_rejects_wrong_file_owner(self):
        facts = private_files.FileFacts(
            regular=True, directory=False, reparse_point=False, size=0,
            owned_by_current_user=False, group_or_other_access=False)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "not owned by this user"):
            self.storage._validate_secret_file(facts, "JSON file")

    async def test_rejects_wrong_directory_owner(self):
        facts = private_files.FileFacts(
            regular=False, directory=True, reparse_point=False, size=0,
            owned_by_current_user=False, group_or_other_access=False)

        with mock.patch.object(
                private_files, "describe", return_value=facts):
            with self.assertRaisesRegex(
                    credential_storages.CredentialStorageError,
                    "directory is not owned"):
                self.storage.ensure_directory()

    async def test_rejects_duplicate_json_keys(self):
        await self.storage.store_openai_login(tokens())
        with open(
                self.storage.file_path, "w", encoding="utf-8") as stream:
            stream.write(
                f'{{"version":{credential_storages.FORMAT_VERSION},'
                f'"version":{credential_storages.FORMAT_VERSION},'
                '"revision":0,"credentials":{}}')

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "duplicate key"):
            self.storage.load_document()

    async def test_rejects_excessively_long_json_integer(self):
        await self.storage.store_openai_login(tokens())
        with open(
                self.storage.file_path, "w", encoding="ascii") as stream:
            stream.write(
                f'{{"version":{credential_storages.FORMAT_VERSION},'
                '"revision":'
                + "9" * 5000
                + ',"credentials":{}}')

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "credential JSON is invalid"):
            self.storage.load_document()

    async def test_rejects_unsupported_schema_version(self):
        await self.storage.store_openai_login(tokens())
        with open(
                self.storage.file_path, "w",
                encoding="ascii") as stream:
            json.dump({
                "version": credential_storages.FORMAT_VERSION + 1,
                "revision": 0,
                "credentials": {},
            }, stream)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "unsupported.*version"):
            self.storage.load_document()

    async def test_rejects_invalid_openai_record_field(self):
        await self.storage.store_openai_login(tokens())
        with open(
                self.storage.file_path,
                encoding="utf-8") as stream:
            document = json.load(stream)
        record = document["credentials"][
            credential_storages.OPENAI_CREDENTIAL_KEY]
        record["fedramp"] = "false"
        with open(
                self.storage.file_path, "w",
                encoding="utf-8") as stream:
            json.dump(document, stream)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "FedRAMP flag"):
            self.storage.load_openai_subscription()

    async def test_rejects_oversized_json(self):
        await self.storage.store_openai_login(tokens())
        with open(
                self.storage.file_path, "wb") as stream:
            stream.write(
                b"x" * (
                    credential_storages.MAX_CREDENTIAL_FILE_BYTES + 1))

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "size limit"):
            self.storage.load_document()

    async def test_ambiguous_rotation_is_durably_fail_closed(self):
        current = tokens()
        await self.storage.store_openai_login(current)

        async def refresh(_value):
            raise authentications.RefreshTransientError(
                "lost response",
                request_may_have_been_sent=True,
            )

        with self.assertRaises(
                authentications.RefreshTransientError):
            await self.storage.rotate_openai_subscription(
                current.normalized(), refresh=refresh)

        stored = self.storage.load_openai_subscription()
        self.assertEqual(stored.state, "reauth-required")
        with open(
                self.storage.file_path, encoding="utf-8") as stream:
            persisted = stream.read()
        self.assertNotIn("refresh-a", persisted)

        restarted = credential_storages.JsonCredentialStorage(
            self.directory)
        self.addCleanup(restarted.close)
        replayed = False

        async def must_not_refresh(_value):
            nonlocal replayed
            replayed = True

        with self.assertRaises(
                authentications.RefreshPermanentError):
            await restarted.rotate_openai_subscription(
                current.normalized(), refresh=must_not_refresh)
        self.assertFalse(replayed)

    async def test_pre_send_failure_restores_active_record(self):
        current = tokens()
        await self.storage.store_openai_login(current)

        async def refresh(_value):
            raise authentications.RefreshTransientError(
                "offline",
                request_may_have_been_sent=False,
            )

        with self.assertRaises(
                authentications.RefreshTransientError):
            await self.storage.rotate_openai_subscription(
                current.normalized(), refresh=refresh)

        stored = self.storage.load_openai_subscription()
        self.assertEqual(stored.state, "active")
        self.assertEqual(stored.tokens, current.normalized())

    async def test_cancelled_rotation_is_durably_fail_closed(self):
        current = tokens()
        await self.storage.store_openai_login(current)
        consumer = asyncio.current_task()
        cancelling = consumer.cancelling()
        requested = []
        refreshing = []
        cancel_handle = None

        async def refresh(_value):
            nonlocal cancel_handle
            # Observe the real disk publication while refresh is in progress.
            # Assert outside production catches, which can wrap a read failure.
            refreshing.append(self.storage.load_openai_subscription().state)
            cancel_handle = asyncio.get_running_loop().call_soon(
                lambda: requested.append(consumer.cancel()))
            await asyncio.Future()

        try:
            with self.assertRaises(asyncio.CancelledError):
                await self.storage.rotate_openai_subscription(
                    current.normalized(), refresh=refresh)
            self.assertEqual(requested, [True])
            self.assertEqual(consumer.cancelling(), cancelling + 1)
        finally:
            if cancel_handle is not None:
                cancel_handle.cancel()
            for accepted in requested:
                if accepted and consumer.cancelling() > cancelling:
                    consumer.uncancel()
        self.assertEqual(consumer.cancelling(), cancelling)
        self.assertEqual(refreshing, ["refreshing"])
        self.assertEqual(
            self.storage.load_openai_subscription().state,
            "reauth-required",
        )

    async def test_invalid_grant_is_durably_fail_closed(self):
        current = tokens()
        await self.storage.store_openai_login(current)

        async def refresh(_value):
            raise authentications.RefreshPermanentError(
                "invalid grant")

        with self.assertRaises(
                authentications.RefreshPermanentError):
            await self.storage.rotate_openai_subscription(
                current.normalized(), refresh=refresh)

        self.assertEqual(
            self.storage.load_openai_subscription().state,
            "reauth-required",
        )

    async def test_failed_final_write_leaves_refreshing_tombstone(self):
        current = tokens()
        await self.storage.store_openai_login(current)
        real_write = self.storage._write_document_at
        writes = 0

        def fail_final_write(directory_fd, document):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise credential_storages.CredentialStorageError(
                    "disk full")
            return real_write(directory_fd, document)

        async def refresh(_value):
            return authentications.RefreshResult(
                access_token="access-b",
                refresh_token="refresh-b",
            )

        with mock.patch.object(
                self.storage, "_write_document_at",
                side_effect=fail_final_write):
            with self.assertRaises(
                    authentications.RefreshIndeterminateError):
                await self.storage.rotate_openai_subscription(
                    current.normalized(), refresh=refresh)

        stored = self.storage.load_openai_subscription()
        self.assertEqual(stored.state, "refreshing")
        replayed = False

        async def must_not_refresh(_value):
            nonlocal replayed
            replayed = True

        with self.assertRaises(
                authentications.RefreshPermanentError):
            await self.storage.rotate_openai_subscription(
                current,
                refresh=must_not_refresh,
            )
        self.assertFalse(replayed)

    async def test_two_instances_send_only_one_rotation(self):
        first = credential_storages.JsonCredentialStorage(
            self.directory)
        second = credential_storages.JsonCredentialStorage(
            self.directory)
        current = tokens().normalized()
        await first.store_openai_login(current)
        calls = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def refresh(value):
            calls.append(value)
            started.set()
            await release.wait()
            return authentications.RefreshResult(
                access_token="access-b",
                refresh_token="refresh-b",
            )

        first_task = asyncio.create_task(
            first.rotate_openai_subscription(
                current, refresh=refresh))
        await started.wait()
        second_task = asyncio.create_task(
            second.rotate_openai_subscription(
                current, refresh=refresh))
        await asyncio.sleep(0.1)
        release.set()
        results = await asyncio.gather(first_task, second_task)

        self.assertEqual(calls, ["refresh-a"])
        self.assertEqual(
            {result.refresh_token for result in results},
            {"refresh-b"},
        )


class CredentialDocumentTests(unittest.TestCase):
    def test_two_processes_send_only_one_rotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = os.path.join(
                temporary, "loki", "credentials")
            storage = credential_storages.JsonCredentialStorage(
                directory)
            asyncio.run(storage.store_openai_login(tokens()))
            calls_path = os.path.join(temporary, "calls")
            release_path = os.path.join(temporary, "release")
            loaded_path = os.path.join(temporary, "loaded")
            command = [
                sys.executable,
                "-m",
                "tests.credential_rotation_processes",
                directory,
                calls_path,
                release_path,
                loaded_path,
            ]
            processes = []
            try:
                first = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                processes.append(first)
                deadline = time.monotonic() + 5
                while not os.path.exists(calls_path):
                    if first.poll() is not None:
                        break
                    if time.monotonic() >= deadline:
                        self.fail("first refresh process did not start")
                    time.sleep(0.01)
                if first.poll() is not None:
                    stdout, stderr = first.communicate()
                    self.fail(
                        "first refresh process failed: "
                        f"{stdout!r} {stderr!r}")

                second = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                processes.append(second)
                # Wait until the second process has read the credential
                # document, rather than guessing with a sleep.  It reads the
                # document outside the lock, so it must read while the first
                # still holds it; if it reads after publication it sees the
                # refreshed tokens and rotates a second time.
                deadline = time.monotonic() + 5
                while _marker_lines(loaded_path) < 2:
                    if second.poll() is not None:
                        stdout, stderr = second.communicate()
                        self.fail(
                            "second refresh process failed: "
                            f"{stdout!r} {stderr!r}")
                    if time.monotonic() >= deadline:
                        self.fail(
                            "second refresh process did not read the document")
                    time.sleep(0.01)
                with open(
                        calls_path, encoding="ascii") as stream:
                    self.assertEqual(len(stream.readlines()), 1)
                with open(
                        release_path, "w",
                        encoding="ascii") as stream:
                    stream.write("release\n")

                results = [
                    process.communicate(timeout=5)
                    for process in processes
                ]
                for process, (stdout, stderr) in zip(
                        processes, results):
                    self.assertEqual(
                        process.returncode, 0, stderr)
                    self.assertEqual(
                        stdout.strip(), "refresh-b")
                with open(
                        calls_path, encoding="ascii") as stream:
                    self.assertEqual(len(stream.readlines()), 1)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
                    else:
                        if process.stdout is not None:
                            process.stdout.close()
                        if process.stderr is not None:
                            process.stderr.close()

    def test_unknown_records_are_preserved_by_openai_login(self):
        with tempfile.TemporaryDirectory() as temporary:
            with closing(credential_storages.JsonCredentialStorage(
                    os.path.join(temporary, "credentials"))) as storage:
                storage.ensure_directory()
                document = {
                    "version": credential_storages.FORMAT_VERSION,
                    "revision": 3,
                    "credentials": {
                        "future:value": {"type": "future"},
                    },
                }
                storage._write_document_at(storage.directory_handle, document)

                async def save():
                    await storage.store_openai_login(tokens())

                asyncio.run(save())
                self.assertEqual(
                    storage.load_document()["credentials"]["future:value"],
                    {"type": "future"},
                )


class PrivateDirectorySupportTests(unittest.TestCase):
    """The Windows mkdir(mode) floor is enforced at creation, not by metadata.

    ``requires-python`` states the same floor but only installers enforce it,
    so a checkout run is covered only by the runtime check.
    """

    def test_posix_is_unrestricted_regardless_of_version(self):
        for version in ((3, 9, 0), (3, 11, 9), (3, 12, 3)):
            with self.subTest(version=version), \
                    mock.patch.object(sys, "platform", "linux"), \
                    mock.patch.object(sys, "version_info", version):
                self.assertIsNone(paths.private_directory_support_error())

    def test_windows_versions_without_the_mkdir_fix_are_refused(self):
        for version in ((3, 10, 11), (3, 11, 0), (3, 11, 9),
                        (3, 12, 0), (3, 12, 3)):
            with self.subTest(version=version), \
                    mock.patch.object(sys, "platform", "win32"), \
                    mock.patch.object(sys, "version_info", version):
                message = paths.private_directory_support_error()
                self.assertIn("Windows", message)
                self.assertIn("3.11.10 or later", message)

    def test_windows_versions_with_the_mkdir_fix_are_allowed(self):
        for version in ((3, 11, 10), (3, 11, 16), (3, 12, 4),
                        (3, 12, 10), (3, 13, 0), (3, 14, 7)):
            with self.subTest(version=version), \
                    mock.patch.object(sys, "platform", "win32"), \
                    mock.patch.object(sys, "version_info", version):
                self.assertIsNone(paths.private_directory_support_error())

    def test_ensure_directory_refuses_before_creating_anything(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = os.path.join(temporary, "loki")
            with mock.patch.object(sys, "platform", "win32"), \
                    mock.patch.object(sys, "version_info", (3, 11, 9)):
                with self.assertRaises(
                        credential_storages.CredentialStorageError):
                    credential_storages.JsonCredentialStorage(os.path.join(parent, "credentials"))
            # Refusal must precede makedirs/mkdir: nothing may be created.
            self.assertFalse(os.path.exists(parent))


class WindowsCredentialFilePrimitiveTests(unittest.TestCase):
    """Portable checks for the Windows primitive layer's own logic.

    The native calls cannot run here, so these drive the layer around them with
    ``windows_api`` mocked: the single-component name guard, the NTSTATUS /
    Win32 to OSError translation the storage protocol branches on, and the
    FileFacts a handle produces.  The calls themselves are exercised by the
    Windows investigation experiments.
    """

    def test_native_close_failure_translates_once_to_storage_oserror(self):
        from loki_agent import _private_files_windows

        failure = windows_api.WindowsResourceError('CloseHandle failed', status=5)
        with mock.patch.object(windows_api, 'close_handle', side_effect=failure) as close:
            with self.assertRaises(OSError) as caught:
                _private_files_windows.close(17)
        self.assertIs(type(caught.exception), OSError)
        self.assertIs(caught.exception.__cause__, failure)
        close.assert_called_once_with(17)

    def test_cleanup_failure_never_retries_private_directory_creation(self):
        from loki_agent import _private_files_windows

        failure = windows_api.WindowsResourceError('LocalFree failed', status=windows_api.ERROR_PATH_NOT_FOUND)
        with mock.patch.object(windows_api, 'current_user_sid', return_value='S-1-5-21-9'), \
                mock.patch.object(windows_api, 'create_directory', side_effect=failure) as create:
            with self.assertRaises(OSError) as caught:
                _private_files_windows.make_private_directory(r'parent\selected')
        self.assertIs(type(caught.exception), OSError)
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(create.call_args.args[0], r'parent\selected')

    def test_private_directory_creation_tries_the_raw_native_operand_before_parent_navigation(self):
        from loki_agent import _private_files_windows
        user = 'S-1-5-21-1-2-3-1001'
        raw = r'base\link\..\new\credentials'
        parent = r'base\link\..\new'
        calls = []

        def create(path, *, security_descriptor):
            calls.append(path)
            self.assertTrue(security_descriptor.startswith('O:' + user + 'D:P'))
            if len(calls) == 1:
                raise windows_api.WindowsApiError('parent missing', status=windows_api.ERROR_PATH_NOT_FOUND)
            return True

        with mock.patch.object(windows_api, 'current_user_sid', return_value=user), \
                mock.patch.object(windows_api, 'create_directory', side_effect=create), \
                mock.patch.object(windows_api, 'set_handle_dacl') as repair:
            _private_files_windows.make_private_directory(raw)
        self.assertEqual(calls, [raw, parent, raw])
        repair.assert_not_called()

    def test_existing_private_directory_is_not_repaired_and_other_errors_are_not_retried(self):
        from loki_agent import _private_files_windows
        with mock.patch.object(windows_api, 'current_user_sid', return_value='S-1-5-21-1-2-3-1001'), \
                mock.patch.object(windows_api, 'create_directory', return_value=False) as create, \
                mock.patch.object(windows_api, 'set_handle_dacl') as repair:
            _private_files_windows.make_private_directory('selected')
            create.assert_called_once()
            repair.assert_not_called()
            create.reset_mock()
            create.side_effect = windows_api.WindowsApiError('denied', status=windows_api.ERROR_ACCESS_DENIED)
            with self.assertRaises(PermissionError):
                _private_files_windows.make_private_directory('selected')
            create.assert_called_once()

    def test_user_owned_file_creation_is_explicit_and_never_requests_write_dac(self):
        from loki_agent import _private_files_windows
        user = 'S-1-5-21-1-2-3-1001'
        for operation in [_private_files_windows.create_exclusive_at, _private_files_windows.open_lock_file_at]:
            for owned in [False, True]:
                with self.subTest(operation=operation.__name__, user_owned=owned), \
                        mock.patch.object(windows_api, 'current_user_sid', return_value=user) as identity, \
                        mock.patch.object(windows_api, 'nt_create_file', return_value=77) as create:
                    self.assertEqual(operation(17, 'secret', 0o600, user_owned=owned), 77)
                    self.assertFalse(create.call_args.args[2] & windows_api.AccessMask.WRITE_DAC)
                    if owned:
                        identity.assert_called_once()
                        self.assertEqual(create.call_args.kwargs['security_descriptor'], 'O:' + user)
                    else:
                        identity.assert_not_called()
                        self.assertNotIn('security_descriptor', create.call_args.kwargs)

    def test_a_name_that_is_not_one_component_is_refused(self):
        from loki_agent import _private_files_windows

        for name in ["", ".", "..", "a/b", "a\\b", "C:name"]:
            with self.subTest(name=name), self.assertRaises(
                    credential_storages.CredentialStorageError):
                _private_files_windows.open_read_at(object(), name)

    @staticmethod
    def _failing_call(status):
        def call(*args, **kwargs):
            raise windows_api.WindowsApiError("call failed", status=status)
        return call

    def test_missing_names_map_to_filenotfound(self):
        from loki_agent import _private_files_windows

        statuses = (windows_api.STATUS_OBJECT_NAME_NOT_FOUND,
                    windows_api.STATUS_OBJECT_PATH_NOT_FOUND,
                    windows_api.ERROR_FILE_NOT_FOUND,
                    windows_api.ERROR_PATH_NOT_FOUND)
        for status in statuses:
            with self.subTest(status=status), mock.patch.object(
                    windows_api, "nt_create_file",
                    side_effect=self._failing_call(status)):
                with self.assertRaises(FileNotFoundError):
                    _private_files_windows.open_read_at(
                        object(), "tokens.json")

    def test_a_name_collision_maps_to_fileexists(self):
        from loki_agent import _private_files_windows

        with mock.patch.object(
                windows_api, "nt_create_file",
                side_effect=self._failing_call(
                    windows_api.STATUS_OBJECT_NAME_COLLISION)):
            with self.assertRaises(FileExistsError):
                _private_files_windows.create_exclusive_at(
                    object(), "tokens.json", 0o600)

    def test_access_denied_maps_to_permissionerror(self):
        from loki_agent import _private_files_windows

        with mock.patch.object(
                windows_api, "nt_create_file",
                side_effect=self._failing_call(
                    windows_api.STATUS_ACCESS_DENIED)):
            with self.assertRaises(PermissionError):
                _private_files_windows.open_lock_file_at(
                    object(), "tokens.lock", 0o600)

    def test_an_unnamed_status_is_a_plain_oserror(self):
        from loki_agent import _private_files_windows

        with mock.patch.object(
                windows_api, "nt_create_file",
                side_effect=self._failing_call(0xDEADBEEF)):
            with self.assertRaises(OSError) as caught:
                _private_files_windows.open_read_at(
                    object(), "tokens.json")
        self.assertNotIsInstance(caught.exception, FileNotFoundError)

    @staticmethod
    def _private_sddl(owner):
        return (f"D:P(A;OICI;FA;;;{owner})"
                "(A;OICI;FA;;;S-1-5-18)(A;OICI;FA;;;S-1-5-32-544)")

    def _created_file_mocks(self, information, owner, sddl):
        return (
            mock.patch.object(windows_api, "nt_create_file",
                              return_value=0x77),
            mock.patch.object(windows_api, "by_handle_file_information",
                              return_value=information),
            mock.patch.object(windows_api, "handle_owner_sid",
                              return_value=owner),
            mock.patch.object(windows_api, "handle_dacl_sddl",
                              return_value=sddl),
            mock.patch.object(windows_api, "current_user_sid",
                              return_value=owner),
        )

    def test_a_new_file_that_is_not_private_is_refused_before_writing(self):
        # The privacy check is the store's: it describes the file it just
        # created, before writing, and refuses anything not private.  The
        # primitives only open and describe.
        shared = private_files.FileFacts(
            regular=True, directory=False, reparse_point=False, size=0,
            owned_by_current_user=True, group_or_other_access=True)
        with tempfile.TemporaryDirectory() as root:
            directory = root + os.sep + 'credentials'
            storage = credential_storages.JsonCredentialStorage(directory)
            directory_fd = storage._directory_fd
            try:
                with mock.patch.object(private_files, "describe",
                                       return_value=shared):
                    with self.assertRaises(
                            credential_storages.CredentialStorageError):
                        storage._write_document_at(
                            directory_fd,
                            credential_storages._empty_document())
            finally:
                storage.close()
            # The refused temporary file is cleaned up, not left behind.
            self.assertEqual(
                [entry for entry in os.listdir(directory)
                 if entry.startswith(".")], [])

    def test_a_new_private_file_is_returned(self):
        from loki_agent import _private_files_windows

        information = windows_api.ByHandleFileInformation()
        information.dwFileAttributes = (
            windows_api.FileAttribute.FILE_ATTRIBUTE_NORMAL)
        owner = "S-1-5-21-1-2-3-1001"
        patches = self._created_file_mocks(
            information, owner, self._private_sddl(owner))
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            handle = _private_files_windows.create_exclusive_at(
                object(), "tokens.json", 0o600)
        self.assertEqual(handle, 0x77)

    # The DACL the Windows runners report for os.mkdir(mode=0o700).  Its
    # trustees are SDDL aliases, which is what the first implementation
    # compared against full SIDs and so judged every private directory shared.
    _MEASURED_PRIVATE_SDDL = (
        "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;OW)")

    def test_the_kernel_alias_dacl_is_judged_private(self):
        from loki_agent import _private_files_windows

        self.assertFalse(_private_files_windows._is_shared(
            self._MEASURED_PRIVATE_SDDL, "S-1-5-21-1-2-3-1001"))

    def test_an_alias_for_another_trustee_is_refused(self):
        from loki_agent import _private_files_windows

        sddl = self._MEASURED_PRIVATE_SDDL + "(A;;FR;;;WD)"
        self.assertTrue(_private_files_windows._is_shared(
            sddl, "S-1-5-21-1-2-3-1001"))

    def test_an_inheritable_grant_to_another_trustee_is_refused(self):
        from loki_agent import _private_files_windows

        # Inherit-only, so the directory's own access is clean, but a file
        # created in it inherits the grant: for a directory that is not private.
        sddl = self._MEASURED_PRIVATE_SDDL + "(A;IO;FR;;;S-1-5-21-1-2-3-1004)"
        self.assertTrue(_private_files_windows._is_shared(
            sddl, "S-1-5-21-1-2-3-1001", inheritable=True))
        self.assertFalse(_private_files_windows._is_shared(
            sddl, "S-1-5-21-1-2-3-1001", inheritable=False))

    def test_facts_for_a_private_file_owned_by_the_caller(self):
        from loki_agent import _private_files_windows

        information = windows_api.ByHandleFileInformation()
        information.dwFileAttributes = (
            windows_api.FileAttribute.FILE_ATTRIBUTE_NORMAL)
        information.nFileSizeLow = 1234
        owner = "S-1-5-21-1-2-3-1001"
        with mock.patch.object(windows_api, "by_handle_file_information",
                               return_value=information), \
                mock.patch.object(windows_api, "handle_owner_sid",
                                  return_value=owner), \
                mock.patch.object(windows_api, "handle_dacl_sddl",
                                  return_value=self._private_sddl(owner)), \
                mock.patch.object(windows_api, "current_user_sid",
                                  return_value=owner):
            facts = _private_files_windows.describe(object())
        self.assertTrue(facts.regular)
        self.assertFalse(facts.directory)
        self.assertFalse(facts.reparse_point)
        self.assertEqual(facts.size, 1234)
        self.assertTrue(facts.owned_by_current_user)
        self.assertFalse(facts.group_or_other_access)

    def test_a_dacl_that_grants_another_trustee_is_not_private(self):
        from loki_agent import _private_files_windows

        owner = "S-1-5-21-1-2-3-1001"
        sddl = (f"D:P(A;OICI;FA;;;{owner})"
                "(A;OICI;FA;;;S-1-5-18)(A;OICI;FA;;;S-1-5-32-544)"
                "(A;;FR;;;S-1-5-21-1-2-3-1004)")
        self.assertTrue(
            _private_files_windows._is_shared(sddl, owner))

    def test_an_absent_dacl_is_not_private(self):
        from loki_agent import _private_files_windows

        # No DACL means everyone has full access; it must read as shared, never
        # as "no grant" because there was nothing to parse.
        self.assertTrue(_private_files_windows._is_shared(
            None, "S-1-5-21-1-2-3-1001"))

    def test_a_reparse_point_and_a_foreign_owner_are_reported(self):
        from loki_agent import _private_files_windows

        information = windows_api.ByHandleFileInformation()
        information.dwFileAttributes = (
            windows_api.FileAttribute.FILE_ATTRIBUTE_REPARSE_POINT)
        with mock.patch.object(windows_api, "by_handle_file_information",
                               return_value=information), \
                mock.patch.object(windows_api, "handle_owner_sid",
                                  return_value="S-1-5-21-1-2-3-1004"), \
                mock.patch.object(windows_api, "handle_dacl_sddl",
                                  return_value=None), \
                mock.patch.object(windows_api, "current_user_sid",
                                  return_value="S-1-5-21-1-2-3-1001"):
            facts = _private_files_windows.describe(object())
        self.assertTrue(facts.reparse_point)
        self.assertFalse(facts.owned_by_current_user)
        self.assertTrue(facts.group_or_other_access)


class CredentialStorageLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """Creation, publication and login/logout share one storage on either OS.

    FileFacts supplies the common contract; actual mode/DACL observations remain
    independent of publication. Refusals keep their own fresh fixtures above.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = os.path.join(
            self.temporary.name, "loki", "credentials")
        with mock.patch.object(os, 'chmod', side_effect=AssertionError(
                'directory permissions must not be repaired')), \
                mock.patch.object(windows_api, 'set_handle_dacl', side_effect=AssertionError(
                    'existing directory permissions must not be repaired')):
            self.storage = credential_storages.JsonCredentialStorage(
                self.directory)
            self.addCleanup(self.storage.close)

    async def test_private_creation_login_and_logout_share_one_lifecycle(self):
        facts = private_files.describe(self.storage.directory_handle)
        self.assertTrue(facts.directory)
        self.assertTrue(facts.owned_by_current_user)
        self.assertFalse(facts.group_or_other_access)
        if os.name == 'posix':
            self.assertEqual(stat.S_IMODE(os.stat(self.directory).st_mode), 0o700)
        stored = await self.storage.store_openai_login(tokens())
        self.assertEqual(stored.state, "active")
        for name in [paths.CREDENTIAL_LOCK_FILE_NAME, paths.CREDENTIAL_FILE_NAME]:
            with self.subTest(name=name):
                handle = private_files.open_read_at(self.storage.directory_handle, name)
                try:
                    facts = private_files.describe(handle)
                    self.assertTrue(facts.owned_by_current_user)
                    self.assertFalse(facts.group_or_other_access)
                finally:
                    private_files.close(handle)
        self.storage.ensure_directory()
        self.assertFalse(private_files.describe_path(
            self.directory).group_or_other_access)
        self.assertFalse(private_files.describe_path(
            self.storage.file_path).group_or_other_access)
        self.assertEqual(
            self.storage.load_openai_subscription().tokens,
            tokens().normalized())
        self.assertTrue(await self.storage.remove_openai_subscription())
        self.assertFalse(await self.storage.remove_openai_subscription())


if __name__ == "__main__":
    unittest.main()
