import os
import tempfile
import unittest
from unittest import mock

from loki_agent import acp_commands
from loki_agent import loki
from loki_agent import provider_controls
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

    async def test_cd_operand_is_a_literal_path(self):
        # The operand is the rest of the line, a literal path -- not a shell
        # word list.  POSIX tokenising used to eat the separators, so a
        # Windows path reached change_shell_cwd as C:worksub.
        self.assertEqual(
            loki._parse_cd_arg_text(r"C:\work\sub"), r"C:\work\sub")
        # One layer of matching quotes is still removed, and nothing else.
        self.assertEqual(loki._parse_cd_arg_text('"a b"'), "a b")

        _session = self._install("/tmp")

        outcome = await acp_commands.run(r"/cd C:\work\sub", _session)

        self.assertNotIn("C:worksub", outcome.text)

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


class FakeControlTests(unittest.IsolatedAsyncioTestCase):
    """The ACP /account path against a stub control, provider-independent."""

    def setUp(self):
        self._saved = loki._DEFAULT_SESSION
        self.addCleanup(self._restore)
        loki._DEFAULT_SESSION = session("/tmp")
        self._actions = ()

    def _restore(self):
        loki._DEFAULT_SESSION = self._saved

    async def _read(self, context):
        return provider_controls.ControlResult(
            lines=("Reset cards - live", "  5-hour reset cards: 1 available"),
            actions=tuple(self._actions),
        )

    def _patch(self, actions=()):
        self._actions = actions
        spec = provider_controls.ControlSpec(
            id="resets",
            title="Reset cards",
            description="quota reset cards",
            applies=lambda context: True,
            read=self._read,
        )
        for name in ("available_controls", "find_control"):
            patch = mock.patch.object(
                provider_controls, name, return_value=[spec]
                if name == "available_controls" else spec)
            patch.start()
            self.addCleanup(patch.stop)

    async def test_lists_controls_when_called_without_argument(self):
        self._patch()

        outcome = await acp_commands.run(
            "/account", loki.current_session())

        self.assertIn("resets - Reset cards", outcome.text)

    async def test_reads_control_and_offers_action_ids(self):
        ran = []
        action = provider_controls.ControlAction(
            id="use:7", title="Use 5-hour reset card",
            confirm="Use the card?",
            run=lambda: _ran(ran))
        self._patch(actions=(action,))

        outcome = await acp_commands.run(
            "/account resets", loki.current_session())

        self.assertIn("5-hour reset cards: 1 available", outcome.text)
        self.assertIn("use:7 - Use 5-hour reset card", outcome.text)
        self.assertEqual(ran, [])

    async def test_named_action_is_the_confirmation_and_runs(self):
        ran = []

        async def run():
            ran.append(True)
            return provider_controls.ControlResult(
                lines=("Reset card used: Weekly quota is back to 100%.",))

        action = provider_controls.ControlAction(
            id="use:7", title="Use reset card",
            confirm="Use the card? This cannot be undone.",
            run=run)
        self._patch(actions=(action,))

        outcome = await acp_commands.run(
            "/account resets use:7", loki.current_session())

        self.assertEqual(ran, [True])
        self.assertIn("Use the card? This cannot be undone.", outcome.text)
        self.assertIn("back to 100%", outcome.text)

    async def test_unknown_action_is_reported_without_running(self):
        ran = []
        action = provider_controls.ControlAction(
            id="use:7", title="Use reset card", confirm="Use?",
            run=lambda: _ran(ran))
        self._patch(actions=(action,))

        outcome = await acp_commands.run(
            "/account resets use:999", loki.current_session())

        self.assertEqual(outcome.text, "No such action: use:999")
        self.assertEqual(ran, [])


async def _ran(bucket):
    bucket.append(True)
    return provider_controls.ControlResult(lines=("done",))


if __name__ == "__main__":
    unittest.main()
