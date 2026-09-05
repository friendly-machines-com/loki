import asyncio
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
from loki_agent import credential_storages


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

    async def test_login_is_atomic_private_and_loadable(self):
        stored = await self.storage.store_openai_login(tokens())

        self.assertEqual(stored.state, "active")
        self.assertEqual(
            stat.S_IMODE(os.stat(self.directory).st_mode), 0o700)
        self.assertEqual(
            stat.S_IMODE(os.stat(self.storage.file_path).st_mode),
            0o600,
        )
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
        os.chmod(self.storage.file_path, 0o640)

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

    async def test_rejects_wrong_file_owner(self):
        file_stat = mock.Mock(
            st_mode=stat.S_IFREG | 0o600,
            st_uid=os.geteuid() + 1,
        )

        with self.assertRaisesRegex(
                credential_storages.CredentialStorageError,
                "not owned by this user"):
            self.storage._validate_secret_file(
                file_stat, "JSON file")

    async def test_rejects_wrong_directory_owner(self):
        directory_stat = mock.Mock(
            st_mode=stat.S_IFDIR | 0o700,
            st_uid=os.geteuid() + 1,
        )

        with mock.patch.object(
                credential_storages.os,
                "lstat",
                return_value=directory_stat):
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
