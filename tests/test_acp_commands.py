import os
import tempfile
import unittest
from unittest import mock

from loki_agent import acp_commands
from loki_agent import loki
from loki_agent import provider_controls, skills
from loki_agent.sessions import Session


def session(shell_cwd):
    return Session(shell_cwd=shell_cwd)


class ParseTests(unittest.TestCase):
    def test_plain_text_is_not_a_command(self):
        self.assertIsNone(acp_commands.parse("hello"))
        self.assertIsNone(acp_commands.parse("/not-a-loki-command"))

    def test_skill_names_are_not_local_commands(self):
        # Skills are advertised, then handled by the model's Skill tool.
        self.assertIsNone(acp_commands.parse("/pua do the thing"))

    def test_bang_without_a_command_is_not_a_command(self):
        self.assertIsNone(acp_commands.parse("!"))
        self.assertEqual(acp_commands.parse("!ls"), ("!", "ls"))


class AdvertisementTests(unittest.TestCase):
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
                demo = loki.run_skill('demo')
                inner = loki.run_skill('pack:inner')
        self.assertEqual(commands["demo"], "Does demo things")
        self.assertEqual(commands["pack:inner"], "First line fallback")
        self.assertIn('# Demo', demo)
        self.assertIn(f'Base directory for this skill: {skill}', demo)
        self.assertIn('# Inner', inner)
        self.assertIn(f'Base directory for this skill: {plugin}', inner)

    def test_skill_name_does_not_become_an_arbitrary_filename(self):
        with tempfile.TemporaryDirectory() as config, tempfile.TemporaryDirectory() as outside:
            path = os.path.join(outside, 'SKILL.md')
            with open(path, 'w', encoding='utf-8') as stream:
                stream.write('not a registered skill')
            with mock.patch.object(loki, 'LOKI_CONFIG_DIR', config), \
                    mock.patch.object(skills, 'read_text') as read:
                for name in [outside, 'missing', '../outside']:
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'unavailable'):
                        loki.run_skill(name)
                read.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'literal colon directory names')
    def test_ambiguous_catalog_identifier_is_neither_advertised_nor_loaded(self):
        with tempfile.TemporaryDirectory() as config:
            direct = os.path.join(config, 'skills', 'pack:inner')
            nested = os.path.join(config, 'skills', 'pack', 'inner')
            kept = os.path.join(config, 'skills', 'kept')
            for directory, body in [[direct, 'direct'], [nested, 'nested'], [kept, 'kept body']]:
                os.makedirs(directory)
                with open(os.path.join(directory, 'SKILL.md'), 'w', encoding='utf-8') as stream:
                    stream.write(body)
            with mock.patch.object(loki, 'LOKI_CONFIG_DIR', config):
                names = [command['name'] for command in acp_commands.advertised_commands()]
                self.assertNotIn('pack:inner', names)
                self.assertIn('kept', names)
                with self.assertRaisesRegex(ValueError, 'unavailable'):
                    loki.run_skill('pack:inner')
                self.assertIn('kept body', loki.run_skill('kept'))

    @unittest.skipUnless(os.name == 'posix', 'native POSIX FIFO replacement')
    def test_skill_reader_checks_the_opened_object_after_catalog_lookup(self):
        with tempfile.TemporaryDirectory() as config:
            directory = os.path.join(config, 'skills', 'demo')
            os.makedirs(directory)
            path = os.path.join(directory, 'SKILL.md')
            with open(path, 'w', encoding='utf-8') as stream:
                stream.write('original regular file')
            read = skills.read_text

            def replace(selected, limit, **kwargs):
                os.unlink(selected)
                os.mkfifo(selected)
                return read(selected, limit, **kwargs)

            with mock.patch.object(loki, 'LOKI_CONFIG_DIR', config), \
                    mock.patch.object(skills, 'read_text', side_effect=replace):
                with self.assertRaisesRegex(OSError, 'not a regular file'):
                    loki.run_skill('demo')


class CommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = loki._DEFAULT_SESSION
        self.addCleanup(self._restore)

    def _restore(self):
        loki._DEFAULT_SESSION = self._saved

    def _install(self, shell_cwd):
        loki._DEFAULT_SESSION = session(shell_cwd)
        return loki._DEFAULT_SESSION

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

    async def test_image_failure_is_reported_not_raised(self):
        _session = self._install("/tmp")

        outcome = await acp_commands.run("/image nope.png", _session)

        self.assertTrue(outcome.text.startswith("image: "))

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
            lines=["Reset cards - live", "  5-hour reset cards: 1 available"],
            actions=list(self._actions),
        )

    def _patch(self, actions=[]):
        self._actions = actions
        spec = provider_controls.ControlSpec(
            id="resets",
            title="Reset cards",
            description="quota reset cards",
            applies=lambda context: True,
            read=self._read,
        )
        for name in ["available_controls", "find_control"]:
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
        self._patch(actions=[action])

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
                lines=["Reset card used: Weekly quota is back to 100%."])

        action = provider_controls.ControlAction(
            id="use:7", title="Use reset card",
            confirm="Use the card? This cannot be undone.",
            run=run)
        self._patch(actions=[action])

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
        self._patch(actions=[action])

        outcome = await acp_commands.run(
            "/account resets use:999", loki.current_session())

        self.assertEqual(outcome.text, "No such action: use:999")
        self.assertEqual(ran, [])


async def _ran(bucket):
    bucket.append(True)
    return provider_controls.ControlResult(lines=["done"])


if __name__ == "__main__":
    unittest.main()
