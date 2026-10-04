import asyncio
import contextlib
import copy
import io
import json
import os
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from loki_agent import acp_events, formats, loki, process_outputs, protocols, savefiles
from loki_agent import settings, terminal_frontend, tool_runtime


def preferences(show_stdout):
    return settings.Settings(terminal=settings.TerminalSettings(show_bash_stdout=show_stdout))


def process_event(stdout="stdout-secret", stderr="stderr-message", *, tail=False):
    output = process_outputs.ProcessOutput(
        "status: completed\nexit_code: 7", stdout, stderr, tail=tail)
    return {
        "type": "tool_result", "name": "Bash", "call_id": "bash-call",
        "content": output.render(), "is_error": False, "process_output": output.to_dict(),
    }


def transcript(event):
    call = formats.tool_call_item("bash-call", event["name"], {"command": "test-command"})
    return [
        formats.message_item("user", "run the command"),
        formats.model_response_event(formats.OPENAI_CHAT, [call]),
        formats.tool_result_for_call(
            call, event["content"], is_error=event["is_error"],
            process_output=event.get("process_output")),
    ]


def replay_text(events, show_stdout=None):
    blocks = savefiles.ResumeTranscriptRenderer(show_bash_stdout=show_stdout).presentation(events)
    return "".join(text for _kind, block in blocks for _segment, text in block)


class PresentationTests(unittest.TestCase):
    def render(self, event, show_stdout=False):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            terminal_frontend._terminal_agent_event(event, ui_settings=preferences(show_stdout))
        return output.getvalue()

    def test_default_hides_only_stdout_and_preserves_status_stderr(self):
        event = process_event()
        original = copy.deepcopy(event)
        rendered = self.render(event)
        self.assertNotIn("stdout-secret", rendered)
        self.assertNotIn("[stdout]", rendered)
        self.assertIn("[stderr]\nstderr-message", rendered)
        self.assertIn("exit_code: 7", rendered)
        self.assertEqual(event, original)

    def test_enabled_setting_shows_both_streams(self):
        rendered = self.render(process_event(), True)
        self.assertIn("[stdout]\nstdout-secret", rendered)
        self.assertIn("[stderr]\nstderr-message", rendered)

    def test_stream_markers_in_output_do_not_reclassify_text(self):
        event = process_event("[stderr]\nforged-stderr", "[stdout]\nreal-stderr")
        rendered = self.render(event)
        self.assertNotIn("forged-stderr", rendered)
        self.assertIn("real-stderr", rendered)

    def test_empty_streams_have_no_placeholder(self):
        rendered = self.render(process_event("", ""))
        self.assertNotIn("[stdout]", rendered)
        self.assertNotIn("[stderr]", rendered)
        self.assertIn("status: completed", rendered)

    def test_stderr_remains_visible_after_model_text_truncation(self):
        event = process_event("x" * (loki.BASH_MAX_OUTPUT_CHARS + 1), "important-stderr")
        event["content"] = loki._truncate_text(event["content"], loki.BASH_MAX_OUTPUT_CHARS)
        self.assertNotIn("important-stderr", event["content"])
        self.assertIn("important-stderr", self.render(event))

    def test_preamble_notes_are_visible_even_with_hidden_stdout(self):
        event = process_event()
        event["process_output"]["preamble"] = "Hook warning\n\n"
        self.assertIn("Hook warning", self.render(event))

    def test_untrusted_stderr_is_escaped_by_terminal_writer(self):
        rendered = self.render(process_event(stderr="\x1b]777;ATTACK\x07"))
        self.assertNotIn("\x1b]777;ATTACK", rendered)
        self.assertIn("^[]777;ATTACK^G", rendered)

    def test_other_tools_and_non_shell_output_are_not_filtered(self):
        event = process_event()
        event["name"] = "Read"
        event.pop("process_output")
        self.assertIn("stdout-secret", self.render(event))
        event["name"] = "JobStatus"
        event["process_output"] = process_event()["process_output"]
        event["process_output"]["shell"] = False
        self.assertIn("stdout-secret", self.render(event))

    def test_generic_tool_errors_without_streams_remain_visible(self):
        event = {
            "type": "tool_result", "name": "Bash", "content": "Error: launch denied", "is_error": True,
        }
        self.assertIn("Error: launch denied", self.render(event))

    def test_new_transcript_round_trip_preserves_channels_and_replay_policy(self):
        events = transcript(process_event())
        loaded, _todos, _state, _tools = savefiles.read_chat_log(io.StringIO(
            savefiles.serialize_chat_log(events, [], {})))
        self.assertEqual(loaded, events)
        hidden = replay_text(loaded, False)
        self.assertNotIn("stdout-secret", hidden)
        self.assertIn("stderr-message", hidden)
        self.assertIn("stdout-secret", replay_text(loaded, True))
        self.assertIn("stdout-secret", replay_text(loaded))

    def test_terminal_resume_uses_settings_but_does_not_change_saved_content(self):
        events = transcript(process_event())
        original = copy.deepcopy(events)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            terminal_frontend._ResumeTranscriptPresenter("Assistant").write(events)
        self.assertNotIn("stdout-secret", output.getvalue())
        self.assertIn("stderr-message", output.getvalue())
        self.assertEqual(events, original)

    def test_legacy_bash_replay_is_explicit_not_guessed(self):
        event = process_event()
        event.pop("process_output")
        events = transcript(event)
        hidden = replay_text(events, False)
        self.assertIn("Older combined Bash output is hidden", hidden)
        self.assertNotIn("stdout-secret", hidden)
        self.assertIn("stdout-secret", replay_text(events, True))
        self.assertIn("stdout-secret", replay_text(events))

    def test_model_projections_do_not_include_or_filter_presentation_metadata(self):
        event = process_event()
        rich = transcript(event)
        plain = copy.deepcopy(rich)
        plain[-1].pop("process_output")
        for project in [
                formats.items_to_openai_chat_messages,
                formats.items_to_anthropic_parts,
                formats.items_to_openai_responses_parts]:
            with self.subTest(project=project.__name__):
                self.assertEqual(project(rich), project(plain))
        self.assertIn("stdout-secret", json.dumps(formats.items_to_openai_chat_messages(rich)))

    def test_acp_tool_payload_is_unchanged_by_presentation_metadata(self):
        event = process_event()
        plain = copy.deepcopy(event)
        plain.pop("process_output")
        self.assertEqual(acp_events.map_event("session", event, {}), acp_events.map_event("session", plain, {}))
        self.assertIn("stdout-secret", json.dumps(acp_events.map_event("session", event, {})))

    def test_saved_presentation_data_is_validated(self):
        for field, value in [["stdout", []], ["shell", 1], ["tail", "true"], ["header", None]]:
            with self.subTest(field=field):
                events = transcript(process_event())
                events[-1]["process_output"][field] = value
                with self.assertRaises(formats.TranscriptFormatError):
                    formats.validate_events(events)


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = pathlib.Path(directory.name)
        self.manager = loki.JobManager(str(self.directory))
        self.session = loki.Session(
            shell_cwd=str(self.directory), job_manager=self.manager,
            runtime_config=loki.make_runtime_config(
                "http://example.invalid/v1", protocols.DUMMY, model="test-model"))
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)
        self.addAsyncCleanup(self.manager.close_session_owned)

    async def execute(self, command="test-command", *, pipeline=None, **arguments):
        call = formats.tool_call_item("bash-call", "Bash", {"command": command, **arguments})
        result, _execution = await loki.execute_tool_call_async(
            call, hook_pipeline=pipeline or tool_runtime.ToolHookPipeline())
        return result

    async def test_completed_capture_preserves_original_model_text(self):
        job = types.SimpleNamespace(exit_code=7)
        with mock.patch.object(self.manager, "run_foreground", new=mock.AsyncMock(
                return_value=[job, "completed", "stdout-secret", "stderr-message"])):
            result = await self.execute()
        self.assertEqual(result["content"],
                         "status: completed\nexit_code: 7\n[stdout]\nstdout-secret\n[stderr]\nstderr-message")
        self.assertEqual(result["process_output"]["stdout"], "stdout-secret")
        self.assertEqual(result["process_output"]["stderr"], "stderr-message")

    async def test_timeout_and_cancellation_keep_both_captured_channels(self):
        job = types.SimpleNamespace(exit_code=-15)
        for status in ["timed_out", "cancelled"]:
            with self.subTest(status=status), mock.patch.object(
                    self.manager, "run_foreground", new=mock.AsyncMock(
                        return_value=[job, status, "partial-stdout", "partial-stderr"])):
                result = await self.execute(timeout=100)
                self.assertFalse(result["ok"])
                output = process_outputs.ProcessOutput.from_dict(result["process_output"])
                self.assertIn("partial-stdout", output.stdout)
                self.assertIn("partial-stderr", output.stderr)
                hidden = output.render(show_stdout=False)
                self.assertNotIn("partial-stdout", hidden)
                self.assertIn("partial-stderr", hidden)
                if status == "timed_out":
                    self.assertIn("command timed out", hidden)
                else:
                    self.assertIn("interrupted by user", hidden)
                    self.assertEqual(result["content"],
                                     "Tool call interrupted by user (SIGINT to the job's process group)"
                                     "; partial output:\npartial-stdout")

    async def test_launch_error_remains_visible_in_replay_without_stdout(self):
        with mock.patch.object(self.manager, "run_foreground", new=mock.AsyncMock(side_effect=OSError("launch denied"))):
            result = await self.execute()
        self.assertFalse(result["ok"])
        self.assertEqual(result["process_output"]["stdout"], "")
        event = {"name": "Bash", "content": result["content"], "is_error": True,
                 "process_output": result["process_output"]}
        self.assertIn("launch denied", replay_text(transcript(event), False))

    async def test_rejected_bash_has_no_streams_but_preserves_diagnostic(self):
        call = formats.tool_call_item("bash-call", "Bash", {"command": "test-command"})
        result, _execution = await loki.execute_tool_call_async(
            call, allowed=set(), hook_pipeline=tool_runtime.ToolHookPipeline())
        self.assertFalse(result["ok"])
        self.assertEqual(result["process_output"]["stdout"], "")
        self.assertIn("not available", process_outputs.presentation_text(result["content"], result["process_output"]))

    async def test_empty_bash_command_has_structured_metadata_without_streams(self):
        result = await self.execute("")
        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "status: completed\nexit_code: 0\nno_output_expected: true")
        self.assertEqual(result["process_output"]["stdout"], "")

    async def test_post_hook_notes_are_retained_in_both_views(self):
        pipeline = tool_runtime.ToolHookPipeline()
        pipeline.add_post("test.note", lambda _invocation, _outcome: tool_runtime.PostHookDecision(note="Important hook note"))
        job = types.SimpleNamespace(exit_code=0)
        with mock.patch.object(self.manager, "run_foreground", new=mock.AsyncMock(
                return_value=[job, "completed", "stdout-secret", "stderr-message"])):
            result = await self.execute(pipeline=pipeline)
        self.assertTrue(result["content"].startswith("Important hook note\n\n"))
        hidden = process_outputs.presentation_text(result["content"], result["process_output"])
        self.assertTrue(hidden.startswith("Important hook note\n\n"))
        self.assertNotIn("stdout-secret", hidden)
        self.assertIn("stderr-message", hidden)

    def completed_job(self, shell):
        stdout = self.directory / "stdout.log"
        stderr = self.directory / "stderr.log"
        stdout.write_text("job-stdout", encoding="utf-8")
        stderr.write_text("job-stderr", encoding="utf-8")
        job = loki.Job(
            id="1", command="test", argv=None, shell=shell, description="", background=True,
            spool_dir=str(self.directory), stdout_path=str(stdout), stderr_path=str(stderr),
            metadata_path=str(self.directory / "job.json"), started_at_iso="start",
            status="exited", exit_code=0)
        self.manager.jobs[job.id] = job
        return job

    async def test_shell_job_status_is_filtered_but_explicit_ps_is_complete(self):
        self.completed_job(True)
        result = await loki.dispatch_tool_async("JobStatus", {"job_id": "1"})
        self.assertIn("job-stdout", result["content"])
        hidden = process_outputs.presentation_text(result["content"], result["process_output"])
        self.assertNotIn("job-stdout", hidden)
        self.assertIn("[stderr_tail]\njob-stderr", hidden)
        self.assertIn("job-stdout", loki.run_ps("1"))

    async def test_subagent_job_status_stdout_is_not_filtered(self):
        self.completed_job(False)
        result = await loki.dispatch_tool_async("JobStatus", {"job_id": "1"})
        self.assertNotIn("process_output", result)
        self.assertIn("job-stdout", process_outputs.presentation_text(result["content"]))

    async def test_manual_shell_calls_keep_the_complete_text_projection(self):
        job = types.SimpleNamespace(exit_code=0)
        with mock.patch.object(self.manager, "run_foreground", new=mock.AsyncMock(
                return_value=[job, "completed", "stdout-secret", "stderr-message"])):
            output = await loki.run_bash_async("manual-command")
        self.assertIsInstance(output, str)
        self.assertIn("stdout-secret", output)
        self.assertIn("stderr-message", output)

    async def test_model_tool_loop_preserves_metadata_in_event_and_saved_result(self):
        snapshots = []
        events = []
        call = formats.tool_call_item("bash-call", "Bash", {"command": "test-command"})
        responses = iter([
            formats.DecodedTurn([call], metadata={"protocol": formats.OPENAI_CHAT}),
            formats.DecodedTurn([formats.message_item("assistant", "done")], metadata={"protocol": formats.OPENAI_CHAT}),
        ])

        async def chat(items, **_kwargs):
            snapshots.append(copy.deepcopy(items))
            return next(responses)

        items = [formats.message_item("user", "run it")]
        job = types.SimpleNamespace(exit_code=0)
        with mock.patch.object(self.manager, "run_foreground", new=mock.AsyncMock(
                return_value=[job, "completed", "stdout-secret", "stderr-message"])):
            self.assertEqual(await loki.run_tool_loop_async(
                items, chat_fn=chat, on_event=events.append, hook_pipeline=tool_runtime.ToolHookPipeline()), "done")
        result = next(item for item in items if item["type"] == "tool_result")
        event = next(event for event in events if event["type"] == "tool_result")
        self.assertEqual(result["process_output"], event["process_output"])
        self.assertIn("stdout-secret", formats.item_text(result))
        self.assertIn("stdout-secret", json.dumps(formats.items_to_openai_chat_messages(snapshots[-1])))
        event["process_output"]["stdout"] = "changed event"
        self.assertEqual(result["process_output"]["stdout"], "stdout-secret")
        loaded, _todos, _state, _tools = savefiles.read_chat_log(io.StringIO(savefiles.serialize_chat_log(items, [], {})))
        self.assertNotIn("stdout-secret", replay_text(loaded, False))
        self.assertIn("stderr-message", replay_text(loaded, False))

    async def test_terminal_turn_callback_uses_supplied_preferences(self):
        event = process_event()

        async def tool_loop(_items, **kwargs):
            kwargs["on_event"](event)
            return "done"

        for show in [False, True]:
            with self.subTest(show=show), mock.patch.object(
                    terminal_frontend, "run_tool_loop_async", side_effect=tool_loop), mock.patch.object(
                    terminal_frontend, "_redraw_status"), contextlib.redirect_stdout(io.StringIO()) as output:
                await terminal_frontend.run_terminal_turn_async([], ui_settings=preferences(show))
                self.assertEqual("stdout-secret" in output.getvalue(), show)
                self.assertIn("stderr-message", output.getvalue())

    async def test_terminal_startup_loads_optional_ini_through_settings_facade(self):
        config_dir = self.directory / "config"
        seen = []

        class Input:
            def __init__(inner_self):
                inner_self.user_messages = terminal_frontend.terminals.UserMessageQueue()
                inner_self.user_messages.put_nowait("request")
                inner_self.user_messages.put_nowait(None)
                inner_self.reader = types.SimpleNamespace(
                    cancel_requested=False, cancel_event=asyncio.Event())

            async def __aenter__(inner_self):
                return inner_self

            async def __aexit__(inner_self, *_args):
                pass

        async def turn(_items, **kwargs):
            value = kwargs["ui_settings"]
            seen.append(value.terminal.show_bash_stdout)
            terminal_frontend._terminal_agent_event(process_event(), ui_settings=value)
            return "done"

        config = self.session.runtime_config
        for setting in [None, False, True]:
            with (
                self.subTest(setting=setting),
                mock.patch.object(settings.paths, "loki_config_dir", return_value=str(config_dir)),
                mock.patch.object(terminal_frontend, "input_session", return_value=Input()),
                mock.patch.object(terminal_frontend.terminals, "open_terminal_stdin"),
                mock.patch.object(terminal_frontend, "restore_output_area_after_input"),
                mock.patch.object(terminal_frontend, "_redraw_status"),
                mock.patch.object(terminal_frontend, "explicit_api_base_configured", return_value=True),
                mock.patch.object(terminal_frontend, "build_config_from_env", return_value=config),
                mock.patch.object(terminal_frontend, "new_chat_log_path", return_value=str(self.directory / "chat.json")),
                mock.patch.object(terminal_frontend, "run_terminal_turn_async", side_effect=turn),
                mock.patch.object(loki, "TOOL_HOOK_PIPELINE", tool_runtime.ToolHookPipeline()),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                if setting is not None:
                    await settings.update_user_settings({"terminal.show_bash_stdout": setting})
                self.assertEqual(await terminal_frontend.async_main([]), 0)
                self.assertEqual("stdout-secret" in output.getvalue(), bool(setting))
                self.assertIn("stderr-message", output.getvalue())
        self.assertEqual(seen, [False, False, True])

    async def test_headless_does_not_load_terminal_preferences(self):
        with (
            mock.patch.object(settings, "load_settings", new=mock.AsyncMock()) as load,
            mock.patch.object(terminal_frontend, "build_config_from_env", return_value=self.session.runtime_config),
            mock.patch.object(terminal_frontend.subagents, "run_cli_async", new=mock.AsyncMock()),
        ):
            self.assertEqual(await terminal_frontend.async_main(["--headless", "--prompt=test"]), 0)
            load.assert_not_awaited()

    @unittest.skipUnless(os.name == "posix", "requires a POSIX shell")
    async def test_real_background_bash_status_keeps_channel_identity(self):
        started = await self.execute(
            "printf 'bg-stdout'; printf 'bg-stderr' >&2", run_in_background=True)
        self.assertIn("Started background job", started["content"])
        self.assertEqual(started["process_output"]["stdout"], "")
        job = next(iter(self.manager.jobs.values()))
        try:
            await asyncio.wait_for(job.process.wait(), 3)
            result = await loki.dispatch_tool_async("JobStatus", {"job_id": job.id})
            self.assertEqual(result["process_output"]["stdout"], "bg-stdout")
            self.assertEqual(result["process_output"]["stderr"], "bg-stderr")
            hidden = process_outputs.presentation_text(result["content"], result["process_output"])
            self.assertNotIn("bg-stdout", hidden)
            self.assertIn("bg-stderr", hidden)
        finally:
            if job.process.returncode is None:
                self.manager.stop_job(job.id, force=True)
                await job.process.wait()

    @unittest.skipUnless(os.name == "posix", "requires a POSIX shell")
    async def test_real_foreground_bash_keeps_channels_separate(self):
        result = await self.execute("printf 'real-stdout'; printf 'real-stderr' >&2; exit 9")
        self.assertEqual(result["process_output"]["stdout"], "real-stdout")
        self.assertEqual(result["process_output"]["stderr"], "real-stderr")
        hidden = process_outputs.presentation_text(result["content"], result["process_output"])
        self.assertNotIn("real-stdout", hidden)
        self.assertIn("real-stderr", hidden)
        self.assertIn("exit_code: 9", hidden)


if __name__ == "__main__":
    unittest.main()
