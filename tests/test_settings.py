import asyncio
import configparser
import contextlib
import dataclasses
import errno
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from process_lifecycle_fixtures import observe_fixture
from loki_agent import file_locks, files, paths, private_files, settings


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

    async def test_settings_validation_does_not_require_credential_privacy_queries(self):
        """A readable settings file needs type checks, not credential ownership/ACL parsing."""
        with mock.patch.object(private_files, 'describe',
                               side_effect=AssertionError('credential privacy query')):
            updated = await self.update({'terminal.show_bash_stdout': True})
        self.assertTrue(updated.terminal.show_bash_stdout)
        self.assertTrue((await self.load()).terminal.show_bash_stdout)

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

    async def test_directory_sync_failure_reports_after_publication_and_releases_the_lock(self):
        await self.update({'terminal.show_bash_stdout': False})
        sync = private_files.fsync

        def failed_directory_sync(handle):
            if private_files.information(handle).directory:
                raise OSError('directory sync failed')
            sync(handle)

        with mock.patch.object(private_files, 'fsync', side_effect=failed_directory_sync):
            with self.assertRaisesRegex(settings.SettingsError, 'directory sync failed'):
                await self.update({'terminal.show_bash_stdout': True})
        self.assertTrue((await self.load()).terminal.show_bash_stdout)
        self.assertFalse(any(name.startswith('.settings.ini.') for name in os.listdir(self.directory)))
        await self.update({'terminal.show_bash_stdout': False})
        self.assertFalse((await self.load()).terminal.show_bash_stdout)

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
            try:
                # Opening/parsing can fail before the lock attempt; the held
                # update, not an elapsed cutoff, owns this observation.
                await observe_fixture(tried.wait(), task)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await self.update({"terminal.show_bash_stdout": True})
        self.assertTrue((await self.load()).terminal.show_bash_stdout)

    async def test_preferences_and_hooks_share_one_read_and_definition(self):
        from loki_agent import tool_runtime

        self.file.write_text(
            "[terminal]\nshow_read_stdout = yes\n"
            "[hook.turn_error.retry]\ncommand.0 = selected interpreter\n"
            "command.1 = script name;#%literal\ncwd = selected cwd\n",
            encoding="utf-8")
        with mock.patch.object(settings, "read_ini", wraps=settings.read_ini) as read:
            value = await self.load(strict=True)
        read.assert_called_once_with(self.file)
        self.assertTrue(value.terminal.show_read_stdout)
        self.assertEqual(len(value.hooks), 1)
        hook = value.hooks[0]
        self.assertEqual(hook.command, ["selected interpreter", "script name;#%literal"])
        self.assertEqual(hook.cwd, "selected cwd")
        self.assertIsNone(hook.timeout_ms)
        self.assertEqual(hook.tools, ["*"])
        pipeline = tool_runtime.build_hook_pipeline(value.hooks, create_exec=mock.AsyncMock())
        self.assertIs(pipeline.turn_error_hooks[0].config, hook)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            hook.cwd = "changed"

    async def test_all_phase_defaults_are_supplied_by_configuration(self):
        rows = [["pre_tool_call", 2000, "deny"], ["pre_tool_gate", 2000, "deny"],
                ["post_tool_call", 10000, "continue"], ["turn_end", None, None],
                ["turn_error", None, None]]
        self.file.write_text("".join(
            f"[hook.{phase}.{phase}]\ncommand.0 = selected-command\ncwd = chosen\n"
            for phase, _, _ in rows), encoding="utf-8")
        value = await self.load(strict=True)
        self.assertEqual([[hook.phase, hook.timeout_ms, hook.on_error]
                          for hook in value.hooks], rows)

    async def test_malformed_known_hook_options_are_not_ignored(self):
        for options in ["command.2 = lost", "command.01 = ambiguous", "command.x = wrong",
                        "command = bare", "command.1 =", "tools.1 = Bash", "cwd.extra = wrong",
                        "timeout_ms = 0", "timeout_ms = invalid", "on_error = unknown",
                        "workspace_side_effects = maybe"]:
            with self.subTest(options=options):
                self.file.write_text(
                    "[hook.pre_tool_call.test]\ncommand.0 = selected-command\n"
                    f"cwd = chosen\n{options}\n", encoding="utf-8")
                with self.assertRaises(settings.HookConfigurationError):
                    await self.load(strict=True)

    async def test_turn_hook_indexed_tool_options_are_rejected(self):
        for phase in ["turn_end", "turn_error"]:
            for option in ["tools.0 = Bash", "on_error = continue", "workspace_side_effects = yes"]:
                with self.subTest(phase=phase, option=option):
                    self.file.write_text(
                        f"[hook.{phase}.test]\ncommand.0 = selected-command\n"
                        f"cwd = chosen\n{option}\n", encoding="utf-8")
                    with self.assertRaisesRegex(settings.HookConfigurationError, "does not support"):
                        await self.load(strict=True)

    async def test_unknown_values_are_reported_and_preserved_by_preference_updates(self):
        self.file.write_text(
            "[terminal]\nFutureOption = %unchanged%\n"
            "[hook.turn_error.first]\ncommand.0 = chosen\ncwd = chosen-cwd\nFutureOption = value\n"
            "[hook.future_phase.later]\nFutureRequirement = value\n"
            "[hook.turn_end.last]\ncommand.0 = notify\ncwd = elsewhere\n"
            "[future_category]\nOption = value\n", encoding="utf-8")
        before = settings.read_ini(self.file)
        warnings = []
        value = await self.load(strict=True, on_error=warnings.append)
        self.assertEqual([hook.hook_id for hook in value.hooks], ["first", "last"])
        self.assertTrue(any("unsupported hook phase" in str(warning) for warning in warnings))
        self.assertTrue(any("FutureOption" in str(warning) for warning in warnings))
        self.assertTrue(any("future_category" in str(warning) for warning in warnings))
        updated = await self.update({"terminal.show_bash_stdout": True})
        self.assertEqual(updated.hooks, value.hooks)
        after = settings.read_ini(self.file)
        self.assertEqual(after.sections(), before.sections())
        for section in before.sections():
            for key, text in before.items(section):
                self.assertEqual(after.get(section, key), text)

    async def test_layers_override_hook_fields_without_reordering_definitions(self):
        self.file.write_text(
            "[hook.turn_error.first]\ncommand.0 = chosen\ncommand.1 = original\ncwd = lower\n"
            "[hook.turn_end.last]\ncommand.0 = notify\ncwd = lower\n", encoding="utf-8")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as resource:
            resource.write("[hook.turn_error.first]\ncommand.1 = upper\ncwd = upper\n")
            upper = resource.name
        self.addCleanup(os.unlink, upper)
        value = await settings.load_settings(files=[self.file, upper], strict=True)
        self.assertEqual([hook.hook_id for hook in value.hooks], ["first", "last"])
        self.assertEqual(value.hooks[0].command, ["chosen", "upper"])
        self.assertEqual(value.hooks[0].cwd, "upper")
        self.assertEqual(value.hooks[1].cwd, "lower")

    async def test_startup_strict_read_does_not_silently_drop_hooks(self):
        self.file.write_text("[broken section\n", encoding="utf-8")
        with self.assertRaises(settings.SettingsError):
            await self.load(strict=True)

    async def test_terminal_startup_passes_the_single_configuration_snapshot(self):
        from loki_agent import loki, terminal_frontend as frontend

        self.file.write_text(
            "[terminal]\nshow_read_stdout = yes\n"
            "[hook.turn_end.notify]\ncommand.0 = notify\ncwd = configured\n",
            encoding="utf-8")
        read_ini = settings.read_ini

        def read(file):
            parser = read_ini(file)
            # Any second read would observe different preferences and no hooks.
            self.file.write_text("[terminal]\nshow_read_stdout = no\n", encoding="utf-8")
            return parser

        async def run(args, configuration):
            self.assertTrue(configuration.terminal.show_read_stdout)
            self.assertEqual(len(configuration.hooks), 1)
            self.assertIs(loki.TOOL_HOOK_PIPELINE.turn_end_hooks[0].config, configuration.hooks[0])
            return 0

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(settings, "user_path", return_value=self.file))
            read_call = stack.enter_context(mock.patch.object(settings, "read_ini", side_effect=read))
            stack.enter_context(mock.patch.object(loki, "TOOL_HOOK_PIPELINE"))
            stack.enter_context(mock.patch.object(
                loki, "_tool_process_creator", return_value=mock.AsyncMock()))
            stack.enter_context(mock.patch.object(frontend, "async_main", new=run))
            stack.enter_context(mock.patch.object(frontend, "current_session", return_value=mock.Mock(job_manager=None)))
            stack.enter_context(mock.patch.object(frontend, "current_chat_log_path", return_value=None))
            stack.enter_context(mock.patch.object(frontend, "terminal_output_mode", return_value=contextlib.nullcontext()))
            stack.enter_context(mock.patch.object(frontend, "initialize_terminal_overlay"))
            stack.enter_context(mock.patch.object(frontend, "restore_terminal_overlay"))
            stack.enter_context(mock.patch.object(frontend.signal, "signal"))
            if hasattr(frontend.signal, "pthread_sigmask"):
                stack.enter_context(mock.patch.object(frontend.signal, "pthread_sigmask"))
            self.assertEqual(await frontend._run_frontend([]), 0)
            read_call.assert_called_once_with(self.file)

    @unittest.skipUnless(os.name == 'posix', 'native POSIX dotdot traversal')
    async def test_invalid_traversal_does_not_update_a_simplified_filename(self):
        original = '[terminal]\nshow_bash_stdout = false\n'
        with open(self.file, 'w', encoding='utf-8') as stream:
            stream.write(original)
        operand = os.fspath(self.directory) + '/missing/../settings.ini'
        with self.assertRaises(settings.SettingsError):
            await settings.update_user_settings({'terminal.show_bash_stdout': True}, file=operand)
        with open(self.file, encoding='utf-8') as stream:
            self.assertEqual(stream.read(), original)
        self.assertEqual(os.listdir(self.directory), ['settings.ini'])

    @unittest.skipUnless(os.name == 'posix', 'native retained directory rename')
    async def test_update_reads_locks_and_publishes_in_the_same_opened_directory(self):
        # These four names address the selected tree, its moved name, and the
        # settings entry in each; neither input nor oracle is canonicalized.
        selected = os.path.join(self.directory, 'selected')
        moved = os.path.join(self.directory, 'moved')
        file = os.path.join(selected, 'settings.ini')
        retained = os.path.join(moved, 'settings.ini')
        os.mkdir(selected)
        original = ('[other]\nmarker = original\n'
                    '[hook.turn_end.notify]\ncommand.0 = original-command\ncwd = hook-cwd\n')
        replacement = ('[other]\nmarker = replacement\n'
                       '[hook.turn_end.notify]\ncommand.0 = replacement-command\ncwd = other-cwd\n')
        with open(file, 'w', encoding='utf-8') as stream:
            stream.write(original)
        open_directory = private_files.open_directory
        changed = False

        def replace_directory(path, **options):
            nonlocal changed
            handle = open_directory(path, **options)
            if path == selected and not changed:
                changed = True
                os.rename(selected, moved)
                os.mkdir(selected)
                with open(file, 'w', encoding='utf-8') as stream:
                    stream.write(replacement)
            return handle

        with mock.patch.object(private_files, 'open_directory', side_effect=replace_directory):
            updated = await settings.update_user_settings({'terminal.show_bash_stdout': True}, file=file)
        self.assertEqual(updated.hooks[0].command, ['original-command'])
        self.assertEqual(updated.hooks[0].cwd, 'hook-cwd')
        self.assertTrue(changed)
        with open(file, encoding='utf-8') as stream:
            self.assertEqual(stream.read(), replacement)
        with open(retained, encoding='utf-8') as stream:
            value = stream.read()
        self.assertIn('marker = original', value)
        self.assertNotIn('replacement', value)
        self.assertIn('show_bash_stdout = true', value)
        self.assertEqual(os.listdir(selected), ['settings.ini'])

    async def test_update_keeps_native_parent_directory_link_lookup(self):
        """Removing realpath must not turn a valid settings parent link into a forbidden root."""
        actual = os.path.join(self.directory, 'actual-parent')
        alias = os.path.join(self.directory, 'parent-link')
        os.mkdir(actual)
        try:
            os.symlink(actual, alias, target_is_directory=True)
        except OSError as error:
            if os.name == 'nt' and getattr(error, 'winerror', None) == 1314:
                self.skipTest('this Windows account cannot create symbolic links')
            raise
        target = os.path.join(actual, 'settings.ini')
        operand = os.path.join(alias, 'settings.ini')
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write('[terminal]\nshow_bash_stdout = false\n')
        await settings.update_user_settings({'terminal.show_bash_stdout': True}, file=operand)
        with open(target, encoding='utf-8') as stream:
            self.assertIn('show_bash_stdout = true', stream.read())
        self.assertTrue(os.path.islink(alias))

    @unittest.skipUnless(hasattr(pathlib.Path, "symlink_to"), "requires symlink support")
    async def test_update_keeps_dotfile_symlink(self):
        target = self.directory / "dotfiles" / "loki.ini"
        target.parent.mkdir()
        target.write_text("[terminal]\nshow_bash_stdout = false\n", encoding="utf-8")
        try:
            self.file.symlink_to(target)
        except OSError:
            self.skipTest("symlinks are unavailable")
        original = target.read_bytes()
        for number in [errno.EACCES, errno.EIO, errno.ELOOP]:
            with self.subTest(errno=number), mock.patch.object(
                    files.os, 'readlink', side_effect=OSError(number, 'link lookup refused')):
                with self.assertRaisesRegex(settings.SettingsError, 'link lookup refused'):
                    await self.update({"terminal.show_bash_stdout": True})
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(self.file.is_symlink())
            self.assertEqual(list(target.parent.iterdir()), [target])
        await self.update({"terminal.show_bash_stdout": True})
        self.assertTrue(self.file.is_symlink())
        self.assertIn("show_bash_stdout = true", target.read_text())


if __name__ == "__main__":
    unittest.main()
