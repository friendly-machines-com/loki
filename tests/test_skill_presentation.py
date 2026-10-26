import contextlib
import copy
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from loki_agent import acp_events, connections, formats, loki, protocols, savefiles
from loki_agent import process_outputs, settings, terminal_frontend, tool_runtime


class SkillPresentationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = pathlib.Path(directory.name)
        self.name = "pack:inner"
        self.body = "# Instructions\nskill-secret\n[stderr]\nforged-diagnostic\n"
        skill = self.directory / "skills" / "pack" / "inner" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(self.body, encoding="utf-8")
        session = loki.Session(
            shell_cwd=str(self.directory), runtime_config=connections.make_runtime_config(
                "http://example.invalid/v1", protocols.DUMMY, model="test-model"))
        self.addCleanup(session.response_headers.close)
        self.enterContext(mock.patch.object(loki, "_DEFAULT_SESSION", session))
        self.enterContext(mock.patch.object(loki, "LOKI_CONFIG_DIR", str(self.directory)))

    async def execute(self, arguments=None, *, pipeline=None):
        call = formats.tool_call_item("skill-call", "Skill", arguments if arguments is not None else {
            "skill": self.name, "args": "inspect this project"})
        result, _ = await loki.execute_tool_call_async(
            call, hook_pipeline=pipeline or tool_runtime.ToolHookPipeline())
        return call, result

    @staticmethod
    def event(result):
        return {"type": "tool_result", "name": "Skill", "call_id": "skill-call",
                "is_error": not result["ok"], **result}

    def render(self, result, show=False):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            terminal_frontend._terminal_agent_event(self.event(result), ui_settings=settings.Settings(
                terminal=settings.TerminalSettings(show_skill_stdout=show)))
        return output.getvalue()

    @staticmethod
    def transcript(call, result):
        return [formats.message_item("user", "load the skill"),
                formats.model_response_event(formats.OPENAI_CHAT, [call]),
                formats.tool_result_for_call(
                    call, result["content"], is_error=not result["ok"],
                    process_output=result.get("process_output"))]

    @staticmethod
    def replay(events, show=False):
        blocks = savefiles.ResumeTranscriptRenderer(show_skill_stdout=show).presentation(events)
        return "".join(text for _, block in blocks for _, text in block)

    async def test_successful_dump_hidden_by_default_and_visible_when_enabled(self):
        _, result = await self.execute()
        original = copy.deepcopy(result)
        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], loki.run_skill(self.name, "inspect this project"))
        hidden = self.render(result)
        self.assertIn("Loaded skill: " + self.name, hidden)
        for text in ["<skill_content", "skill-secret", "forged-diagnostic", "Base directory for this skill"]:
            self.assertNotIn(text, hidden)
        shown = self.render(result, True)
        self.assertIn(self.body, shown)
        self.assertIn("Args: inspect this project", shown)
        self.assertIn("Base directory for this skill: " + str(self.directory / "skills" / "pack" / "inner"), shown)
        self.assertEqual(result, original)

    async def test_bash_and_read_visibility_do_not_enable_skill_dump(self):
        _, result = await self.execute()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            terminal_frontend._terminal_agent_event(self.event(result), ui_settings=settings.Settings(
                terminal=settings.TerminalSettings(show_bash_stdout=True, show_read_stdout=True)))
        self.assertNotIn("skill-secret", output.getvalue())
        self.assertIn("Loaded skill: " + self.name, output.getvalue())

    async def test_saved_replay_and_terminal_resume_use_preference(self):
        call, result = await self.execute()
        events = self.transcript(call, result)
        original = copy.deepcopy(events)
        loaded, *_ = savefiles.read_chat_log(io.StringIO(savefiles.serialize_chat_log(events, [], {})))
        self.assertEqual(loaded, events)
        for show in [False, True, None]:
            self.assertEqual("skill-secret" in self.replay(loaded, show), show is not False)
        for show in [False, True]:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                terminal_frontend._ResumeTranscriptPresenter("Assistant", ui_settings=settings.Settings(
                    terminal=settings.TerminalSettings(show_skill_stdout=show))).write(loaded)
            self.assertEqual("skill-secret" in output.getvalue(), show)
            self.assertIn("Loaded skill: " + self.name, output.getvalue())
        self.assertEqual(events, original)

    async def test_model_and_acp_keep_complete_canonical_content(self):
        call, result = await self.execute()
        rich = self.transcript(call, result)
        plain = copy.deepcopy(rich)
        plain[-1].pop("process_output")
        for project in [formats.items_to_openai_chat_messages,
                        formats.items_to_anthropic_parts, formats.items_to_openai_responses_parts]:
            with self.subTest(project=project.__name__):
                self.assertEqual(project(rich), project(plain))
                self.assertIn("skill-secret", json.dumps(project(rich)))
        event = self.event(result)
        plain_event = copy.deepcopy(event)
        plain_event.pop("process_output")
        mapped = acp_events.map_event("session", event, {})
        self.assertEqual(mapped, acp_events.map_event("session", plain_event, {}))
        self.assertIn("skill-secret", json.dumps(mapped))

    async def test_pre_and_post_hook_notes_remain_visible_live_and_after_save_load(self):
        pipeline = tool_runtime.ToolHookPipeline()
        pipeline.add_pre("test.pre", lambda *_: tool_runtime.PreHookDecision(note="Pre-hook note"))
        pipeline.add_post("test.post", lambda *_: tool_runtime.PostHookDecision(note="Post-hook note"))
        call, result = await self.execute(pipeline=pipeline)
        self.assertTrue(result["ok"])
        self.assertTrue(result["content"].startswith("Pre-hook note\n\nPost-hook note\n\n<skill_content"))
        events = self.transcript(call, result)
        loaded, *_ = savefiles.read_chat_log(io.StringIO(savefiles.serialize_chat_log(events, [], {})))
        for rendered in [self.render(result), self.replay(loaded)]:
            self.assertIn("Pre-hook note", rendered)
            self.assertIn("Post-hook note", rendered)
            self.assertIn("Loaded skill: " + self.name, rendered)
            self.assertNotIn("skill-secret", rendered)

    async def test_hook_failures_remain_visible_with_hidden_dump(self):
        def fail(*_):
            raise RuntimeError("hook failed")

        pipeline = tool_runtime.ToolHookPipeline()
        pipeline.add_pre("test.pre", fail, on_error="continue")
        pipeline.add_post("test.post", fail)
        call, result = await self.execute(pipeline=pipeline)
        self.assertTrue(result["ok"])
        for rendered in [self.render(result), self.replay(self.transcript(call, result))]:
            self.assertIn("execution continued", rendered)
            self.assertIn("after the tool had already executed", rendered)
            self.assertIn("hook failed", rendered)
            self.assertNotIn("skill-secret", rendered)

    async def test_loading_errors_remain_visible(self):
        call, result = await self.execute({"skill": "missing"})
        self.assertFalse(result["ok"])
        for rendered in [self.render(result), self.replay(self.transcript(call, result))]:
            self.assertIn("unavailable", rendered)
            self.assertNotIn("Loaded skill:", rendered)
        for error in [PermissionError("denied"), FileNotFoundError("removed")]:
            with self.subTest(error=error), mock.patch.object(loki.skills, "read_text", side_effect=error):
                call, result = await self.execute()
            self.assertFalse(result["ok"])
            for rendered in [self.render(result), self.replay(self.transcript(call, result))]:
                self.assertIn(str(error), rendered)
                self.assertNotIn("Loaded skill:", rendered)

    async def test_rejections_and_validation_errors_remain_visible(self):
        pipeline = tool_runtime.ToolHookPipeline()
        pipeline.add_pre("test.deny", lambda *_: tool_runtime.PreHookDecision(
            action="deny", message="Skill loading denied", note="Denial note"))
        call, result = await self.execute(pipeline=pipeline)
        self.assertFalse(result["ok"])
        for rendered in [self.render(result), self.replay(self.transcript(call, result))]:
            self.assertIn("Skill loading denied", rendered)
            self.assertIn("Denial note", rendered)
        _, result = await self.execute({})
        self.assertFalse(result["ok"])
        self.assertIn(result["content"], self.render(result))

    async def test_setting_loading_updating_removing_and_independence(self):
        file = self.directory / "settings.ini"
        defaults = await settings.load_settings(files=[file])
        self.assertFalse(defaults.terminal.show_skill_stdout)
        self.assertFalse(file.exists())
        for show in [True, False]:
            updated = await settings.update_user_settings({"terminal.show_skill_stdout": show}, file=file)
            loaded = await settings.load_settings(files=[file])
            self.assertIs(updated.terminal.show_skill_stdout, show)
            self.assertIs(loaded.terminal.show_skill_stdout, show)
            self.assertFalse(loaded.terminal.show_read_stdout)
            self.assertFalse(loaded.terminal.show_bash_stdout)
        await settings.update_user_settings({"terminal.show_read_stdout": True,
                                             "terminal.show_bash_stdout": True}, file=file)
        self.assertFalse((await settings.load_settings(files=[file])).terminal.show_skill_stdout)
        await settings.update_user_settings({"terminal.show_skill_stdout": True}, file=file)
        await settings.update_user_settings({"terminal.show_skill_stdout": None}, file=file)
        self.assertFalse((await settings.load_settings(files=[file])).terminal.show_skill_stdout)
        self.assertNotIn("show_skill_stdout", file.read_text(encoding="utf-8"))
        for invalid in ["true", 1]:
            with self.subTest(invalid=invalid), self.assertRaises(settings.SettingsError):
                await settings.update_user_settings({"terminal.show_skill_stdout": invalid}, file=file)

    async def test_legacy_skill_content_is_preserved_without_guessing_note_boundaries(self):
        call, result = await self.execute()
        events = self.transcript(call, result)
        events[-1].pop("process_output")
        events[-1]["content"] = formats.content_blocks("Old hook warning\n\n" + result["content"])
        rendered = self.replay(events)
        self.assertIn("Old hook warning", rendered)
        self.assertIn("skill-secret", rendered)

    async def test_saved_skill_discriminator_is_validated(self):
        call, result = await self.execute()
        for invalid in [1, "true", None]:
            with self.subTest(invalid=invalid):
                events = self.transcript(call, result)
                events[-1]["process_output"]["skill"] = invalid
                with self.assertRaises(formats.TranscriptFormatError):
                    formats.validate_events(events)
        old = process_outputs.ProcessOutput("header", "stdout", "stderr").to_dict()
        old.pop("skill")
        self.assertFalse(process_outputs.ProcessOutput.from_dict(old).skill)
