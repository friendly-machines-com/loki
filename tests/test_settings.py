import asyncio
import configparser
import dataclasses
import pathlib
import tempfile
import unittest
from unittest import mock

from loki_agent import file_locks, paths, private_files, settings


class SettingsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = pathlib.Path(directory.name)
        self.file = self.directory / "settings.ini"

    async def load(self, **kwargs):
        return await settings.load_settings(files=[self.file], **kwargs)

    async def update(self, changes):
        return await settings.update_user_settings(changes, file=self.file)

    async def test_missing_file_uses_immutable_defaults_without_creating_files(self):
        value = await self.load()
        self.assertFalse(value.terminal.show_bash_stdout)
        self.assertEqual(list(self.directory.iterdir()), [])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            value.terminal.show_bash_stdout = True

    async def test_default_location_uses_shared_config_directory_resolution(self):
        config_home = self.directory / "config"
        file = config_home / "loki" / "settings.ini"
        file.parent.mkdir(parents=True)
        file.write_text("[terminal]\nshow_bash_stdout = true\n", encoding="utf-8")
        with mock.patch.dict("os.environ", {"XDG_CONFIG_HOME": str(config_home)}):
            self.assertEqual(paths.loki_config_dir(), str(file.parent))
            value = await settings.load_settings()
        self.assertTrue(value.terminal.show_bash_stdout)

    async def test_missing_section_or_option_uses_default(self):
        for text in ["", "[terminal]\n", "[other]\nvalue = something\n"]:
            with self.subTest(text=text):
                self.file.write_text(text, encoding="utf-8")
                self.assertFalse((await self.load()).terminal.show_bash_stdout)

    async def test_boolean_ini_spellings(self):
        for text, expected in [
                ["true", True], ["YES", True], ["on", True], ["1", True],
                ["false", False], ["no", False], ["off", False], ["0", False]]:
            with self.subTest(text=text):
                self.file.write_text(f"[terminal]\nshow_bash_stdout = {text}\n", encoding="utf-8")
                self.assertIs((await self.load()).terminal.show_bash_stdout, expected)

    async def test_invalid_layers_report_and_preserve_file(self):
        for text in [
                "not ini", "[terminal]\nshow_bash_stdout = maybe\n",
                "[terminal]\nshow_bash_stdout = true\nshow_bash_stdout = false\n",
                "[DEFAULT]\nshow_bash_stdout = true\n[terminal]\n"]:
            with self.subTest(text=text):
                self.file.write_text(text, encoding="utf-8")
                errors = []
                value = await self.load(on_error=errors.append)
                self.assertFalse(value.terminal.show_bash_stdout)
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], settings.SettingsError)
                self.assertEqual(self.file.read_text(encoding="utf-8"), text)

    async def test_unreadable_file_reports_and_defaults(self):
        errors = []
        with mock.patch("builtins.open", side_effect=PermissionError("denied")):
            value = await self.load(on_error=errors.append)
        self.assertFalse(value.terminal.show_bash_stdout)
        self.assertEqual(len(errors), 1)

    async def test_layers_merge_only_explicit_overrides(self):
        upper = self.directory / "local.ini"
        self.file.write_text("[terminal]\nshow_bash_stdout = true\n", encoding="utf-8")
        upper.write_text("[terminal]\n", encoding="utf-8")
        value = await settings.load_settings(files=[self.file, upper])
        self.assertTrue(value.terminal.show_bash_stdout)
        upper.write_text("[terminal]\nshow_bash_stdout = false\n", encoding="utf-8")
        value = await settings.load_settings(files=[self.file, upper])
        self.assertFalse(value.terminal.show_bash_stdout)

    async def test_invalid_upper_layer_does_not_discard_valid_lower_layer(self):
        upper = self.directory / "local.ini"
        self.file.write_text("[terminal]\nshow_bash_stdout = true\n", encoding="utf-8")
        upper.write_text("[terminal]\nshow_bash_stdout = invalid\n", encoding="utf-8")
        errors = []
        value = await settings.load_settings(files=[self.file, upper], on_error=errors.append)
        self.assertTrue(value.terminal.show_bash_stdout)
        self.assertEqual(len(errors), 1)

    async def test_updates_round_trip_true_and_explicit_false(self):
        for value in [True, False]:
            with self.subTest(value=value):
                updated = await self.update({"terminal.show_bash_stdout": value})
                self.assertIs(updated.terminal.show_bash_stdout, value)
                self.assertIs((await self.load()).terminal.show_bash_stdout, value)
                self.assertIn(f"show_bash_stdout = {str(value).lower()}", self.file.read_text())

    async def test_update_rereads_and_preserves_unrelated_values_without_interpolation(self):
        self.file.write_text(
            "[terminal]\nFutureOption = %value%\n[other]\npath = %(literal)s\n",
            encoding="utf-8")
        await self.update({"terminal.show_bash_stdout": True})
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str
        parser.read(self.file, encoding="utf-8")
        self.assertEqual(parser["terminal"]["FutureOption"], "%value%")
        self.assertEqual(parser["other"]["path"], "%(literal)s")
        parser.set("other", "new", "later edit")
        with self.file.open("w", encoding="utf-8") as target:
            parser.write(target)
        await self.update({"terminal.show_bash_stdout": False})
        parser.read(self.file, encoding="utf-8")
        self.assertEqual(parser["other"]["new"], "later edit")

    async def test_removing_override_restores_default(self):
        await self.update({"terminal.show_bash_stdout": True})
        updated = await self.update({"terminal.show_bash_stdout": None})
        self.assertFalse(updated.terminal.show_bash_stdout)
        self.assertNotIn("show_bash_stdout", self.file.read_text())

    async def test_removing_missing_override_is_harmless(self):
        updated = await self.update({"terminal.show_bash_stdout": None})
        self.assertFalse(updated.terminal.show_bash_stdout)

    async def test_invalid_changes_do_not_create_files(self):
        for changes in [
                {"terminal.unknown": True}, {"terminal.show_bash_stdout": "true"},
                {"terminal.show_bash_stdout": 1}, []]:
            with self.subTest(changes=changes):
                with self.assertRaises(settings.SettingsError):
                    await self.update(changes)
                self.assertEqual(list(self.directory.iterdir()), [])

    async def test_invalid_existing_file_is_not_overwritten(self):
        text = "[terminal]\nshow_bash_stdout = invalid\n"
        self.file.write_text(text, encoding="utf-8")
        with self.assertRaises(settings.SettingsError):
            await self.update({"terminal.show_bash_stdout": True})
        self.assertEqual(self.file.read_text(), text)

    async def test_write_failure_keeps_original_and_removes_temporary_file(self):
        await self.update({"terminal.show_bash_stdout": False})
        original = self.file.read_bytes()
        with mock.patch.object(private_files, "write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(settings.SettingsError, "disk full"):
                await self.update({"terminal.show_bash_stdout": True})
        self.assertEqual(self.file.read_bytes(), original)
        self.assertFalse(any(path.name.startswith(".settings.ini.") for path in self.directory.iterdir()))
        await self.update({"terminal.show_bash_stdout": True})

    async def test_replace_failure_keeps_original_and_removes_temporary_file(self):
        await self.update({"terminal.show_bash_stdout": False})
        original = self.file.read_bytes()
        with mock.patch.object(private_files, "replace_at", side_effect=OSError("denied")):
            with self.assertRaisesRegex(settings.SettingsError, "denied"):
                await self.update({"terminal.show_bash_stdout": True})
        self.assertEqual(self.file.read_bytes(), original)
        self.assertFalse(any(path.name.startswith(".settings.ini.") for path in self.directory.iterdir()))

    async def test_lock_contention_times_out_without_changing_file(self):
        await self.update({"terminal.show_bash_stdout": False})
        original = self.file.read_bytes()
        with mock.patch.object(file_locks, "try_lock_exclusive", side_effect=BlockingIOError):
            with self.assertRaisesRegex(settings.SettingsError, "busy"):
                await self.update({"terminal.show_bash_stdout": True})
        self.assertEqual(self.file.read_bytes(), original)
        await self.update({"terminal.show_bash_stdout": True})

    async def test_cancelled_lock_wait_releases_resources(self):
        await self.update({"terminal.show_bash_stdout": False})
        tried = asyncio.Event()

        def contend(_fd):
            tried.set()
            raise BlockingIOError

        with mock.patch.object(file_locks, "try_lock_exclusive", side_effect=contend):
            task = asyncio.create_task(self.update({"terminal.show_bash_stdout": True}))
            await tried.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        await self.update({"terminal.show_bash_stdout": True})
        self.assertTrue((await self.load()).terminal.show_bash_stdout)

    @unittest.skipUnless(hasattr(pathlib.Path, "symlink_to"), "requires symlink support")
    async def test_update_keeps_dotfile_symlink(self):
        target = self.directory / "dotfiles" / "loki.ini"
        target.parent.mkdir()
        target.write_text("[terminal]\nshow_bash_stdout = false\n", encoding="utf-8")
        try:
            self.file.symlink_to(target)
        except OSError:
            self.skipTest("symlinks are unavailable")
        await self.update({"terminal.show_bash_stdout": True})
        self.assertTrue(self.file.is_symlink())
        self.assertIn("show_bash_stdout = true", target.read_text())


if __name__ == "__main__":
    unittest.main()
