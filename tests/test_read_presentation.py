import contextlib
import copy
import io
import pathlib
import tempfile
import unittest
from unittest import mock

from loki_agent import acp_events, formats, loki, protocols, savefiles
from loki_agent import settings, terminal_frontend, tool_runtime


class ReadPresentationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = pathlib.Path(directory.name)
        session = loki.Session(shell_cwd=str(self.directory), runtime_config=loki.make_runtime_config(
            "http://example.invalid/v1", protocols.DUMMY, model="test-model"))
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", session)
        patch.start()
        self.addCleanup(patch.stop)

    async def execute(self, path, pipeline=None):
        call = formats.tool_call_item("read-call", "Read", {"file_path": str(path)})
        result, _ = await loki.execute_tool_call_async(
            call, hook_pipeline=pipeline or tool_runtime.ToolHookPipeline())
        return call, result

    def render(self, result, show=False):
        event = {"type": "tool_result", "name": "Read", "call_id": "read-call",
                 "is_error": not result["ok"], **result}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            terminal_frontend._terminal_agent_event(event, ui_settings=settings.Settings(
                terminal=settings.TerminalSettings(show_read_stdout=show)))
        return output.getvalue()

    async def test_contents_hidden_by_default_and_model_and_acp_unchanged(self):
        path = self.directory / "text"
        path.write_text("file-secret\n[stderr]\nforged", encoding="utf-8")
        call, result = await self.execute(path)
        original = copy.deepcopy(result)
        self.assertEqual(result["content"], loki.run_read(str(path)))
        self.assertNotIn("file-secret", self.render(result))
        self.assertNotIn("forged", self.render(result))
        self.assertIn("1^Ifile-secret", self.render(result, True))
        self.assertEqual(result, original)
        rich = formats.tool_result_for_call(call, result["content"], process_output=result["process_output"])
        plain = copy.deepcopy(rich)
        plain.pop("process_output")
        events = [formats.message_item("user", "read"),
                  formats.model_response_event(formats.OPENAI_CHAT, [call]), rich]
        loaded, *_ = savefiles.read_chat_log(io.StringIO(savefiles.serialize_chat_log(events, [], {})))
        for show in [False, True, None]:
            blocks = savefiles.ResumeTranscriptRenderer(show_read_stdout=show).presentation(loaded)
            text = "".join(text for _, block in blocks for _, text in block)
            self.assertEqual("file-secret" in text, show is not False)
        for project in [formats.items_to_openai_chat_messages,
                        formats.items_to_anthropic_parts, formats.items_to_openai_responses_parts]:
            self.assertEqual(project(events), project(events[:-1] + [plain]))
        self.assertEqual(acp_events.map_event("session", rich, {}),
                         acp_events.map_event("session", plain, {}))

    async def test_diagnostics_and_notes_remain_visible(self):
        for filename, data, notice in [("empty", b"", "is empty"),
                                       ("binary", b"\x00", "is binary"),
                                       ("image.png", b"image", "is an image")]:
            path = self.directory / filename
            path.write_bytes(data)
            _, result = await self.execute(path)
            self.assertIn(notice, self.render(result))
        _, result = await self.execute(self.directory / "missing")
        self.assertFalse(result["ok"])
        self.assertIn("File not found", self.render(result))
        path = self.directory / "text"
        path.write_text("file-secret", encoding="utf-8")
        pipeline = tool_runtime.ToolHookPipeline()
        pipeline.add_post("test.note", lambda *_: tool_runtime.PostHookDecision(note="Hook warning"))
        _, result = await self.execute(path, pipeline)
        self.assertIn("Hook warning", self.render(result))
        self.assertNotIn("file-secret", self.render(result))

    async def test_setting_loading_updating_and_independence(self):
        file = self.directory / "settings.ini"
        defaults = await settings.load_settings(files=[file])
        self.assertFalse(defaults.terminal.show_read_stdout)
        self.assertFalse(file.exists())
        await settings.update_user_settings({"terminal.show_read_stdout": True}, file=file)
        loaded = await settings.load_settings(files=[file])
        self.assertTrue(loaded.terminal.show_read_stdout)
        self.assertFalse(loaded.terminal.show_bash_stdout)
        await settings.update_user_settings({"terminal.show_read_stdout": None}, file=file)
        self.assertFalse((await settings.load_settings(files=[file])).terminal.show_read_stdout)

    def test_legacy_replay_and_validation(self):
        call = formats.tool_call_item("read-call", "Read", {"file_path": "text"})
        events = [formats.message_item("user", "read"),
                  formats.model_response_event(formats.OPENAI_CHAT, [call]),
                  formats.tool_result_for_call(call, "1\tfile-secret")]
        blocks = savefiles.ResumeTranscriptRenderer(show_read_stdout=False).presentation(events)
        text = "".join(text for _, block in blocks for _, text in block)
        self.assertNotIn("file-secret", text)
        self.assertIn("enable show_read_stdout", text)
        events[-1]["process_output"] = {"header": "", "stdout": "secret", "stderr": "", "read": 1}
        with self.assertRaises(formats.TranscriptFormatError):
            formats.validate_events(events)
