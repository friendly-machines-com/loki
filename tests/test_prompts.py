"""Optional read-only instructions, default export, and conversation snapshots."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from loki_entrypoints import child_environment, entrypoint
from loki_agent import __main__ as launcher
from loki_agent import formats, loki, paths, prompts
from loki_agent.sessions import Session


class SystemPromptTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        patch = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": directory.name})
        patch.start()
        self.addCleanup(patch.stop)
        self.file = Path(paths.loki_config_dir()) / "SYSTEM.md"

    def write_prompt(self, text):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(text, encoding="utf-8")

    def test_missing_config_uses_default_silently_without_creating_anything(self):
        with self.assertNoLogs("loki_agent.prompts", level="WARNING"):
            self.assertEqual(prompts.load_system_prompt(), prompts.DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_literal_user_instructions_are_preserved(self):
        self.write_prompt("User instructions: caf\u00e9.\n{cwd} is literal text.\n")
        expected = self.file.read_bytes()
        self.assertEqual(prompts.load_system_prompt(), expected.decode("utf-8"))
        self.assertEqual(self.file.read_bytes(), expected)

    def test_literal_crlf_bytes_are_preserved_on_every_platform(self):
        self.file.parent.mkdir(parents=True)
        data = 'Literal caf\u00e9\r\n{cwd}\r\n'.encode('utf-8')
        self.file.write_bytes(data)
        self.assertEqual(prompts.load_system_prompt(), data.decode('utf-8'))
        self.assertEqual(self.file.read_bytes(), data)

    @unittest.skipIf(os.name == "nt", "POSIX read-only file and directory modes")
    def test_read_only_config_supports_both_existing_and_missing_prompt(self):
        self.write_prompt("Read-only instructions")
        self.file.chmod(0o444)
        self.file.parent.chmod(0o555)
        try:
            with self.assertNoLogs("loki_agent.prompts", level="WARNING"):
                self.assertEqual(prompts.load_system_prompt(), "Read-only instructions")
            self.file.parent.chmod(0o755)
            self.file.unlink()
            self.file.parent.chmod(0o555)
            with self.assertNoLogs("loki_agent.prompts", level="WARNING"):
                self.assertEqual(prompts.load_system_prompt(), prompts.DEFAULT_SYSTEM_PROMPT)
            self.assertEqual(list(self.file.parent.iterdir()), [])
        finally:
            self.file.parent.chmod(0o755)

    def test_existing_empty_file_is_intentional(self):
        self.write_prompt("")
        self.assertEqual(prompts.load_system_prompt(), "")
        self.assertEqual(self.file.read_bytes(), b"")

    def test_deletion_restores_current_default_without_recreating_file(self):
        self.write_prompt("User instructions")
        with mock.patch.object(prompts, "DEFAULT_SYSTEM_PROMPT", "New default\n"):
            self.assertEqual(prompts.load_system_prompt(), "User instructions")
            self.file.unlink()
            self.assertEqual(prompts.load_system_prompt(), "New default\n")
        self.assertFalse(self.file.exists())

    def test_invalid_utf8_is_reported_and_never_replaced(self):
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b"\xffuser content")
        with self.assertLogs("loki_agent.prompts", level="WARNING"):
            self.assertEqual(prompts.load_system_prompt(), prompts.DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(self.file.read_bytes(), b"\xffuser content")

    def test_unreadable_existing_file_is_reported_and_preserved(self):
        self.write_prompt("Private instructions")
        with mock.patch("builtins.open", side_effect=PermissionError("denied")), \
                self.assertLogs("loki_agent.prompts", level="WARNING"):
            self.assertEqual(prompts.load_system_prompt(), prompts.DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(self.file.read_text(encoding="utf-8"), "Private instructions")

    def test_directory_is_reported_and_never_replaced(self):
        self.file.mkdir(parents=True)
        with self.assertLogs("loki_agent.prompts", level="WARNING"):
            self.assertEqual(prompts.load_system_prompt(), prompts.DEFAULT_SYSTEM_PROMPT)
        self.assertTrue(self.file.is_dir())
        self.assertEqual(list(self.file.iterdir()), [])

    def test_symlinked_instructions_are_read_without_replacing_link(self):
        target = self.root / "instructions.md"
        target.write_text("Linked instructions", encoding="utf-8")
        self.file.parent.mkdir(parents=True)
        try:
            self.file.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlinks unavailable: {error}")
        self.assertEqual(prompts.load_system_prompt(), "Linked instructions")
        self.assertTrue(self.file.is_symlink())
        target.unlink()
        self.assertEqual(prompts.load_system_prompt(), prompts.DEFAULT_SYSTEM_PROMPT)
        self.assertTrue(self.file.is_symlink())
        self.assertFalse(target.exists())

    def install_session(self):
        session = Session(shell_cwd=str(self.root))
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", session)
        patch.start()
        self.addCleanup(patch.stop)
        return session

    def test_custom_base_is_followed_by_fresh_runtime_context_without_tool_list(self):
        self.write_prompt("My instructions with {date} and {cwd} left literal.")
        self.install_session()
        with mock.patch.object(loki.time, "strftime", return_value="2030-01-02"):
            items = loki.initial_transcript_items()
        self.assertEqual(formats.item_text(items[0]),
                         "My instructions with {date} and {cwd} left literal.")
        runtime = "\n".join(formats.item_text(item) for item in items[1:])
        self.assertIn("Current date: 2030-01-02", runtime)
        self.assertIn(f"Current Loki cwd: {self.root}", runtime)
        self.assertNotIn("You have these tools", runtime)
        self.assertNotIn("Read, Write, Edit", runtime)
        self.assertNotIn("Current date:", self.file.read_text(encoding="utf-8"))

    def test_empty_base_still_has_runtime_context(self):
        self.write_prompt("")
        self.install_session()
        items = loki.initial_transcript_items()
        self.assertIn("Current Loki cwd:", "\n".join(formats.item_text(item) for item in items))
        self.assertNotIn(prompts.DEFAULT_SYSTEM_PROMPT, str(items))

    def test_new_chats_reread_but_current_and_resumed_chats_retain_snapshot(self):
        self.write_prompt("Original instructions")
        session = self.install_session()
        original_path = self.root / "original.json"
        loki.new_chat_log(str(original_path))
        original = list(session.transcript_items)
        with contextlib.redirect_stdout(io.StringIO()):
            loki.save_chat_log()
        self.write_prompt("Changed instructions")
        self.assertEqual(session.transcript_items, original)
        loki.new_chat_log(str(self.root / "new.json"))
        self.assertEqual(formats.item_text(session.transcript_items[0]), "Changed instructions")
        with mock.patch.object(prompts, "load_system_prompt", side_effect=AssertionError(
                "resuming must not read today's instructions")):
            loki.load_chat_log(str(original_path), apply_shell_cwd=False)
        self.assertEqual(session.transcript_items, original)

    def test_export_through_real_entrypoint_uses_builtin_and_creates_no_config(self):
        # A file at the config base prevents even accidental directory creation.
        blocked = self.root / "not-a-directory"
        blocked.write_text("untouched", encoding="utf-8")
        result = subprocess.run(
            [entrypoint("loki"), "--print-default-system-prompt"],
            env=child_environment(
                PATH=os.environ.get("PATH", os.defpath),
                HOME=str(self.root), XDG_CONFIG_HOME=str(blocked),
                XDG_STATE_HOME=str(blocked), LOKI_PROVIDER="not-a-provider"),
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, prompts.DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(result.stderr, "")
        self.assertEqual(list(self.root.iterdir()), [blocked])
        self.assertEqual(blocked.read_text(encoding="utf-8"), "untouched")

    def test_export_bypasses_user_config_and_runtime_initialization(self):
        self.write_prompt("User override must not be exported")
        output = io.StringIO()
        with mock.patch.object(launcher.sys, "argv", ["loki", "--print-default-system-prompt"]), \
                mock.patch.object(launcher, "capture_process_credentials", side_effect=AssertionError), \
                mock.patch.object(launcher, "configure_logging", side_effect=AssertionError), \
                mock.patch.object(launcher.runtime_isolation, "preflight", side_effect=AssertionError), \
                contextlib.redirect_stdout(output):
            self.assertEqual(launcher.main(), 0)
        self.assertEqual(output.getvalue(), prompts.DEFAULT_SYSTEM_PROMPT)

    def test_export_rejects_combining_with_runtime_options(self):
        result = subprocess.run(
            [entrypoint("loki"), "--print-default-system-prompt", "--headless"],
            env=child_environment(PATH=os.environ.get("PATH", os.defpath),
                                  XDG_CONFIG_HOME=str(self.root / "missing")),
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("must be used alone", result.stderr)
        self.assertFalse((self.root / "missing").exists())
