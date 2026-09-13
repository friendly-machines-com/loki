import asyncio
import errno
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
from loki_agent import credential_files
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


class JsonCredentialStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = os.path.join(
            self.temporary.name, "loki", "credentials")
        self.storage = credential_storages.JsonCredentialStorage(
            self.directory)

    async def test_directory_creation_requests_private_mode_without_chmod(self):
        mkdir = os.mkdir
        with mock.patch.object(os, 'mkdir', wraps=mkdir) as create, \
                mock.patch.object(os, 'chmod', side_effect=AssertionError(
                    'directory permissions must not be repaired')):
            self.storage.ensure_directory()
        # The mode is requested on every platform; on Windows it becomes a
        # private DACL, which os.stat cannot show.  POSIX asserts the resulting
        # bits here, Windows asserts the DACL in WindowsCredentialStorageTests.
        create.assert_any_call(self.directory, 0o700)
        if os.name == 'posix':
            self.assertEqual(
                stat.S_IMODE(os.stat(self.directory).st_mode), 0o700)

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
            modes = (0o755, 0o750, 0o770)
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

    async def test_login_is_atomic_private_and_loadable(self):
        stored = await self.storage.store_openai_login(tokens())

        self.assertEqual(stored.state, "active")
        if os.name == 'posix':
            self.assertEqual(
                stat.S_IMODE(os.stat(self.directory).st_mode), 0o700)
            self.assertEqual(
                stat.S_IMODE(os.stat(self.storage.file_path).st_mode),
                0o600,
            )
        else:
            # No mode bits on Windows; the primitives read the DACL.
            self.assertFalse(
                credential_files.describe_path(
                    self.directory).group_or_other_access)
            self.assertFalse(
                credential_files.describe_path(
                    self.storage.file_path).group_or_other_access)
        loaded = self.storage.load_openai_subscription()
        self.assertEqual(loaded.tokens, tokens().normalized())
        self.assertNotIn(
            "access-a",
            repr(loaded.tokens),
        )
        leftovers = [
            name for name in os.listdir(self.directory)
            if name.startswith(".tokens.json.")
        ]
        self.assertEqual(leftovers, [])

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

    async def test_logout_preserves_document_tombstone(self):
        await self.storage.store_openai_login(tokens())

        self.assertTrue(
            await self.storage.remove_openai_subscription())
        self.assertFalse(
            await self.storage.remove_openai_subscription())

        document = self.storage.load_document()
        self.assertGreater(document["revision"], 0)
        self.assertEqual(document["credentials"], {})

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
        os.makedirs(os.path.dirname(self.directory))
        os.symlink(real, self.directory)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "not a directory"):
            self.storage.load_document()

    async def test_rejects_symlinked_json_file(self):
        self.storage.ensure_directory()
        target = os.path.join(self.temporary.name, "target")
        with open(target, "w", encoding="ascii") as stream:
            stream.write("{}")
        os.symlink(target, self.storage.file_path)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "could not open credential JSON"):
            self.storage.load_document()

    async def test_rejects_symlinked_lock_file(self):
        self.storage.ensure_directory()
        target = os.path.join(self.temporary.name, "target")
        with open(target, "w", encoding="ascii") as stream:
            stream.write("")
        os.symlink(target, self.storage.lock_path)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "could not open credential lock"):
            await self.storage.store_openai_login(tokens())

    def assert_descriptor_closed(self, fd):
        try:
            os.fstat(fd)
        except OSError as error:
            self.assertEqual(error.errno, errno.EBADF)
        else:
            # Keep a failing regression test from leaking its real descriptor.
            os.close(fd)
            self.fail(f"descriptor {fd} was left open")

    async def test_failed_lock_open_closes_directory_descriptor(self):
        self.storage.ensure_directory()
        os.symlink('unused-target', self.storage.lock_path)
        open_directory = self.storage._open_directory
        opened = []

        def record_directory():
            fd = open_directory()
            opened.append(fd)
            return fd

        with mock.patch.object(self.storage, '_open_directory',
                               side_effect=record_directory):
            for _ in range(3):
                with self.assertRaisesRegex(
                        credential_storages.CredentialStorageError,
                        'could not open credential lock'):
                    await self.storage.store_openai_login(tokens())
                self.assert_descriptor_closed(opened[-1])

    async def test_setup_failure_after_lock_open_closes_both_descriptors(self):
        directory_fd = self.storage._open_directory()
        lock_fd = self.storage._open_lock_at(directory_fd)
        try:
            with mock.patch.object(self.storage, '_open_directory',
                                   return_value=directory_fd), \
                    mock.patch.object(self.storage, '_open_lock_at',
                                      return_value=lock_fd), \
                    mock.patch.object(asyncio, 'get_running_loop',
                                      side_effect=RuntimeError('setup failed')):
                with self.assertRaisesRegex(RuntimeError, 'setup failed'):
                    await self.storage.store_openai_login(tokens())
        finally:
            # Check both even if one assertion fails.
            try:
                self.assert_descriptor_closed(lock_fd)
            finally:
                self.assert_descriptor_closed(directory_fd)

    async def test_lock_close_error_still_closes_directory_descriptor(self):
        directory_fd = self.storage._open_directory()
        lock_fd = self.storage._open_lock_at(directory_fd)
        close = os.close

        def close_then_report_error(fd):
            close(fd)
            if fd == lock_fd:
                raise OSError('lock close failed')

        try:
            with mock.patch.object(self.storage, '_open_directory',
                                   return_value=directory_fd), \
                    mock.patch.object(self.storage, '_open_lock_at',
                                      return_value=lock_fd), \
                    mock.patch.object(os, 'close', side_effect=close_then_report_error):
                with self.assertRaisesRegex(OSError, 'lock close failed'):
                    async with self.storage._locked_document():
                        pass
        finally:
            try:
                self.assert_descriptor_closed(lock_fd)
            finally:
                self.assert_descriptor_closed(directory_fd)

    async def test_cancellation_while_waiting_closes_both_descriptors(self):
        directory_fd = self.storage._open_directory()
        lock_fd = self.storage._open_lock_at(directory_fd)
        with mock.patch.object(self.storage, '_open_directory',
                               return_value=directory_fd), \
                mock.patch.object(self.storage, '_open_lock_at',
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
                try:
                    self.assert_descriptor_closed(lock_fd)
                finally:
                    self.assert_descriptor_closed(directory_fd)

    async def test_rejects_wrong_file_owner(self):
        facts = credential_files.FileFacts(
            regular=True, directory=False, reparse_point=False, size=0,
            owned_by_current_user=False, group_or_other_access=False)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "not owned by this user"):
            self.storage._validate_secret_file(facts, "JSON file")

    async def test_rejects_wrong_directory_owner(self):
        facts = credential_files.FileFacts(
            regular=False, directory=True, reparse_point=False, size=0,
            owned_by_current_user=False, group_or_other_access=False)

        with mock.patch.object(
                credential_files, "describe_path", return_value=facts):
            with self.assertRaisesRegex(
                    credential_storages.CredentialStorageError,
                    "directory is not owned"):
                self.storage.ensure_directory()

    async def test_rejects_duplicate_json_keys(self):
        self.storage.ensure_directory()
        with open(
                self.storage.file_path, "w", encoding="utf-8") as stream:
            stream.write(
                f'{{"version":{credential_storages.FORMAT_VERSION},'
                f'"version":{credential_storages.FORMAT_VERSION},'
                '"revision":0,"credentials":{}}')
        os.chmod(self.storage.file_path, 0o600)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "duplicate key"):
            self.storage.load_document()

    async def test_rejects_excessively_long_json_integer(self):
        self.storage.ensure_directory()
        with open(
                self.storage.file_path, "w", encoding="ascii") as stream:
            stream.write(
                f'{{"version":{credential_storages.FORMAT_VERSION},'
                '"revision":'
                + "9" * 5000
                + ',"credentials":{}}')
        os.chmod(self.storage.file_path, 0o600)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "credential JSON is invalid"):
            self.storage.load_document()

    async def test_rejects_unsupported_schema_version(self):
        self.storage.ensure_directory()
        with open(
                self.storage.file_path, "w",
                encoding="ascii") as stream:
            json.dump({
                "version": credential_storages.FORMAT_VERSION + 1,
                "revision": 0,
                "credentials": {},
            }, stream)
        os.chmod(self.storage.file_path, 0o600)

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
        os.chmod(self.storage.file_path, 0o600)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "FedRAMP flag"):
            self.storage.load_openai_subscription()

    async def test_rejects_oversized_json(self):
        self.storage.ensure_directory()
        with open(
                self.storage.file_path, "wb") as stream:
            stream.write(
                b"x" * (
                    credential_storages.MAX_CREDENTIAL_FILE_BYTES + 1))
        os.chmod(self.storage.file_path, 0o600)

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "size limit"):
            self.storage.load_document()

    async def test_successful_rotation_is_durable(self):
        current = tokens()
        await self.storage.store_openai_login(current)
        calls = []

        async def refresh(value):
            calls.append(value)
            return authentications.RefreshResult(
                access_token="access-b",
                refresh_token="refresh-b",
                id_token="id-b",
            )

        rotated = await self.storage.rotate_openai_subscription(
            current.normalized(), refresh=refresh, clock=lambda: 200)

        self.assertEqual(calls, ["refresh-a"])
        self.assertEqual(rotated.refresh_token, "refresh-b")
        self.assertEqual(
            self.storage.load_openai_subscription().tokens,
            rotated,
        )

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
        started = asyncio.Event()

        async def refresh(_value):
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            self.storage.rotate_openai_subscription(
                current.normalized(), refresh=refresh))
        await started.wait()
        self.assertEqual(
            self.storage.load_openai_subscription().state,
            "refreshing",
        )
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

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
            command = [
                sys.executable,
                "-m",
                "tests.credential_rotation_processes",
                directory,
                calls_path,
                release_path,
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
                time.sleep(0.2)
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
            storage = credential_storages.JsonCredentialStorage(
                os.path.join(temporary, "credentials"))
            storage.ensure_directory()
            document = {
                "version": credential_storages.FORMAT_VERSION,
                "revision": 3,
                "credentials": {
                    "future:value": {"type": "future"},
                },
            }
            with open(
                    storage.file_path, "w", encoding="utf-8") as stream:
                json.dump(document, stream)
            os.chmod(storage.file_path, 0o600)

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
            storage = credential_storages.JsonCredentialStorage(
                os.path.join(parent, "credentials"))
            with mock.patch.object(sys, "platform", "win32"), \
                    mock.patch.object(sys, "version_info", (3, 11, 9)):
                with self.assertRaises(
                        credential_storages.CredentialStorageError):
                    storage.ensure_directory()
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

    def test_a_name_that_is_not_one_component_is_refused(self):
        from loki_agent import _credential_files_windows

        for name in ("", ".", "..", "a/b", "a\\b", "C:name"):
            with self.subTest(name=name), self.assertRaises(
                    credential_storages.CredentialStorageError):
                _credential_files_windows.open_read_at(object(), name)

    @staticmethod
    def _failing_call(status):
        def call(*args, **kwargs):
            raise windows_api.WindowsApiError("call failed", status=status)
        return call

    def test_missing_names_map_to_filenotfound(self):
        from loki_agent import _credential_files_windows

        statuses = (windows_api.STATUS_OBJECT_NAME_NOT_FOUND,
                    windows_api.STATUS_OBJECT_PATH_NOT_FOUND,
                    windows_api.ERROR_FILE_NOT_FOUND,
                    windows_api.ERROR_PATH_NOT_FOUND)
        for status in statuses:
            with self.subTest(status=status), mock.patch.object(
                    windows_api, "nt_create_file",
                    side_effect=self._failing_call(status)):
                with self.assertRaises(FileNotFoundError):
                    _credential_files_windows.open_read_at(
                        object(), "tokens.json")

    def test_a_name_collision_maps_to_fileexists(self):
        from loki_agent import _credential_files_windows

        with mock.patch.object(
                windows_api, "nt_create_file",
                side_effect=self._failing_call(
                    windows_api.STATUS_OBJECT_NAME_COLLISION)):
            with self.assertRaises(FileExistsError):
                _credential_files_windows.create_exclusive_at(
                    object(), "tokens.json", 0o600)

    def test_access_denied_maps_to_permissionerror(self):
        from loki_agent import _credential_files_windows

        with mock.patch.object(
                windows_api, "nt_create_file",
                side_effect=self._failing_call(
                    windows_api.STATUS_ACCESS_DENIED)):
            with self.assertRaises(PermissionError):
                _credential_files_windows.open_lock_file_at(
                    object(), "tokens.lock", 0o600)

    def test_an_unnamed_status_is_a_plain_oserror(self):
        from loki_agent import _credential_files_windows

        with mock.patch.object(
                windows_api, "nt_create_file",
                side_effect=self._failing_call(0xDEADBEEF)):
            with self.assertRaises(OSError) as caught:
                _credential_files_windows.open_read_at(
                    object(), "tokens.json")
        self.assertNotIsInstance(caught.exception, FileNotFoundError)

    @staticmethod
    def _private_sddl(owner):
        return (f"D:P(A;OICI;FA;;;{owner})"
                "(A;OICI;FA;;;S-1-5-18)(A;OICI;FA;;;S-1-5-32-544)")

    def test_facts_for_a_private_file_owned_by_the_caller(self):
        from loki_agent import _credential_files_windows

        information = windows_api.ByHandleFileInformation()
        information.dwFileAttributes = windows_api.FILE_ATTRIBUTE_NORMAL
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
            facts = _credential_files_windows.describe(object())
        self.assertTrue(facts.regular)
        self.assertFalse(facts.directory)
        self.assertFalse(facts.reparse_point)
        self.assertEqual(facts.size, 1234)
        self.assertTrue(facts.owned_by_current_user)
        self.assertFalse(facts.group_or_other_access)

    def test_a_dacl_that_grants_another_trustee_is_not_private(self):
        from loki_agent import _credential_files_windows

        owner = "S-1-5-21-1-2-3-1001"
        sddl = (f"D:P(A;OICI;FA;;;{owner})"
                "(A;OICI;FA;;;S-1-5-18)(A;OICI;FA;;;S-1-5-32-544)"
                "(A;;FR;;;S-1-5-21-1-2-3-1004)")
        self.assertTrue(
            _credential_files_windows._is_shared(sddl, owner))

    def test_an_absent_dacl_is_not_private(self):
        from loki_agent import _credential_files_windows

        # No DACL means everyone has full access; it must read as shared, never
        # as "no grant" because there was nothing to parse.
        self.assertTrue(_credential_files_windows._is_shared(
            None, "S-1-5-21-1-2-3-1001"))

    def test_a_reparse_point_and_a_foreign_owner_are_reported(self):
        from loki_agent import _credential_files_windows

        information = windows_api.ByHandleFileInformation()
        information.dwFileAttributes = (
            windows_api.FILE_ATTRIBUTE_REPARSE_POINT)
        with mock.patch.object(windows_api, "by_handle_file_information",
                               return_value=information), \
                mock.patch.object(windows_api, "handle_owner_sid",
                                  return_value="S-1-5-21-1-2-3-1004"), \
                mock.patch.object(windows_api, "handle_dacl_sddl",
                                  return_value=None), \
                mock.patch.object(windows_api, "current_user_sid",
                                  return_value="S-1-5-21-1-2-3-1001"):
            facts = _credential_files_windows.describe(object())
        self.assertTrue(facts.reparse_point)
        self.assertFalse(facts.owned_by_current_user)
        self.assertTrue(facts.group_or_other_access)


@unittest.skipUnless(
    os.name == "nt", "the Windows DACL is the private mechanism")
class WindowsCredentialStorageTests(unittest.IsolatedAsyncioTestCase):
    """The storage protocol end to end over the Windows primitives.

    ``JsonCredentialStorageTests`` asserts POSIX mode bits, which do not exist
    on Windows; this runs the same protocol and checks the same property
    through the DACL the primitives read.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = os.path.join(
            self.temporary.name, "loki", "credentials")
        self.storage = credential_storages.JsonCredentialStorage(
            self.directory)

    async def test_login_load_and_logout_round_trip(self):
        stored = await self.storage.store_openai_login(tokens())
        self.assertEqual(stored.state, "active")
        self.assertFalse(credential_files.describe_path(
            self.directory).group_or_other_access)
        self.assertFalse(credential_files.describe_path(
            self.storage.file_path).group_or_other_access)
        self.assertEqual(
            self.storage.load_openai_subscription().tokens,
            tokens().normalized())
        self.assertTrue(await self.storage.remove_openai_subscription())
        self.assertFalse(await self.storage.remove_openai_subscription())


if __name__ == "__main__":
    unittest.main()
