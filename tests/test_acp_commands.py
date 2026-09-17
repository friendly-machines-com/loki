import os
import tempfile
import unittest
from unittest import mock

from loki_agent import acp_commands
from loki_agent import loki
from loki_agent.sessions import Session


PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)


def session(shell_cwd):
    return Session(shell_cwd=shell_cwd)


class ParseTests(unittest.TestCase):
    def test_plain_text_is_not_a_command(self):
        self.assertIsNone(acp_commands.parse("hello"))
        self.assertIsNone(acp_commands.parse("/not-a-loki-command"))

    def test_local_commands_parse_with_argument(self):
        self.assertEqual(acp_commands.parse("/pwd"), ("pwd", ""))
        self.assertEqual(acp_commands.parse("/cd /tmp"), ("cd", "/tmp"))
        self.assertEqual(
            acp_commands.parse("/status all --json"),
            ("status", "all --json"))

    def test_skill_names_are_not_local_commands(self):
        # Skills are advertised, then handled by the model's Skill tool.
        self.assertIsNone(acp_commands.parse("/pua do the thing"))

    def test_bang_without_a_command_is_not_a_command(self):
        self.assertIsNone(acp_commands.parse("!"))
        self.assertEqual(acp_commands.parse("!ls"), ("!", "ls"))


class AdvertisementTests(unittest.TestCase):
    def test_local_commands_are_advertised(self):
        with mock.patch.object(loki, "LOKI_CONFIG_DIR", "/nonexistent"):
            names = {
                command["name"]
                for command in acp_commands.advertised_commands()}
        self.assertIn("pwd", names)
        self.assertIn("account", names)
        # /model and /effort are native ACP config options, not commands.
        self.assertNotIn("model", names)
        self.assertNotIn("effort", names)

    def test_skills_are_advertised_with_descriptions(self):
        with tempfile.TemporaryDirectory() as directory:
            skill = os.path.join(directory, "skills", "demo")
            os.makedirs(skill)
            with open(os.path.join(skill, "SKILL.md"), "w") as stream:
                stream.write(
                    "---\nname: demo\ndescription: Does demo things\n"
                    "---\n# Demo\n")
            plugin = os.path.join(directory, "skills", "pack", "inner")
            os.makedirs(plugin)
            with open(os.path.join(plugin, "SKILL.md"), "w") as stream:
                stream.write("# Inner\nFirst line fallback\n")
            with mock.patch.object(loki, "LOKI_CONFIG_DIR", directory):
                commands = {
                    command["name"]: command["description"]
                    for command in acp_commands.advertised_commands()}
        self.assertEqual(commands["demo"], "Does demo things")
        self.assertEqual(commands["pack:inner"], "First line fallback")


class CommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = loki._DEFAULT_SESSION
        self.addCleanup(self._restore)

    def _restore(self):
        loki._DEFAULT_SESSION = self._saved

    def _install(self, shell_cwd):
        loki._DEFAULT_SESSION = session(shell_cwd)
        return loki._DEFAULT_SESSION

    async def test_pwd_reports_the_shell_cwd(self):
        _session = self._install("/tmp")

        outcome = await acp_commands.run("/pwd", _session)

        self.assertEqual(outcome.text, "cwd: /tmp")
        self.assertIsNone(outcome.model_text)

    async def test_cd_changes_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            _session = self._install("/tmp")

            outcome = await acp_commands.run(f"/cd {directory}", _session)

            self.assertEqual(
                outcome.text, f"cwd: {os.path.realpath(directory)}")
            self.assertEqual(_session.shell_cwd, os.path.realpath(directory))

    async def test_cd_failure_is_reported_not_raised(self):
        _session = self._install("/tmp")

        outcome = await acp_commands.run("/cd /no/such/place", _session)

        self.assertTrue(outcome.text.startswith("cd: "))
        self.assertIn("/no/such/place", outcome.text)

    async def test_ps_lists_jobs(self):
        _session = self._install("/tmp")
        with mock.patch.object(loki, "run_jobs", return_value="no jobs"):
            outcome = await acp_commands.run("/ps", _session)

        self.assertEqual(outcome.text, "no jobs")

    async def test_status_without_a_connection_says_so(self):
        _session = self._install("/tmp")

        outcome = await acp_commands.run("/status", _session)

        self.assertIn("No active HTTP chat connection", outcome.text)

    async def test_account_without_controls_is_explicit(self):
        _session = self._install("/tmp")

        outcome = await acp_commands.run("/account", _session)

        self.assertIn("No live account controls", outcome.text)

    async def test_unknown_account_control_is_reported(self):
        _session = self._install("/tmp")

        outcome = await acp_commands.run("/account nope", _session)

        self.assertEqual(outcome.text, "No such account control: nope")

    async def test_image_stages_a_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            _session = self._install(directory)
            path = os.path.join(directory, "shot.png")
            with open(path, "wb") as stream:
                stream.write(PNG)

            outcome = await acp_commands.run("/image shot.png", _session)

            self.assertIsNotNone(outcome.image)
            self.assertEqual(outcome.image.media_type, "image/png")
            block = outcome.image.content_block()
            self.assertEqual(block["type"], "image")
            self.assertIn("Attached image", outcome.text)

    async def test_image_failure_is_reported_not_raised(self):
        _session = self._install("/tmp")

        outcome = await acp_commands.run("/image nope.png", _session)

        self.assertTrue(outcome.text.startswith("image: "))

    async def test_bang_runs_and_prepares_a_model_turn(self):
        _session = self._install("/tmp")

        async def fake_run(command, **kwargs):
            self.assertEqual(command, "echo hi")
            return "hi"

        with mock.patch.object(loki, "run_bash_async", side_effect=fake_run):
            outcome = await acp_commands.run("!echo hi", _session)

        self.assertIn("Running local command: echo hi", outcome.text)
        self.assertIn("I ran the local command `echo hi`", outcome.model_text)
        self.assertIn("hi", outcome.model_text)

    async def test_non_command_returns_none(self):
        _session = self._install("/tmp")

        self.assertIsNone(await acp_commands.run("hello", _session))


if __name__ == "__main__":
    unittest.main()
