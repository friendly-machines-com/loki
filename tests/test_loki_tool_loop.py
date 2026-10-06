import asyncio
import base64
import copy
import contextlib
import io
import json
import os
import pathlib
import shlex
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from loki_entrypoints import child_environment, configure_container, entrypoint
from response_header_fixtures import setUpModule as set_up_response_headers
from settings_fixtures import setUpModule as set_up_default_settings


from loki_agent import formats
from loki_agent import authentications
from loki_agent import attachments
from loki_agent import credential_runtimes
from loki_agent import credential_supervisors
from loki_agent import http_client
from loki_agent import loki
from loki_agent import subagents
from loki_agent import terminal_frontend
from loki_agent import models as modelsdev
from loki_agent import openai_models
from loki_agent import protocols
from loki_agent.connections import ConnectionDescriptor
from loki_agent.credentials import CredentialInventory, CredentialStore
from loki_agent import savefiles
from loki_agent import terminals
from loki_endpoints import assume_endpoints_approved


_MISSING = object()


def setUpModule():
    set_up_response_headers()
    set_up_default_settings()


def _codex_model(slug="gpt-5-codex", **overrides):
    value = {
        "slug": slug,
        "display_name": slug,
        "visibility": "list",
        "input_modalities": ["text", "image"],
        "supported_reasoning_levels": [],
        "supports_reasoning_summaries": False,
        "default_reasoning_summary": "auto",
        "support_verbosity": False,
        "supports_parallel_tool_calls": True,
        "shell_type": "shell_command",
    }
    value.update(overrides)
    return openai_models.CodexModelRequestProfile.from_catalog_model(value)


def _effort_profile(*values):
    return modelsdev.ReasoningEffortProfile(list(values))


def save_loki_state(names):
    """Snapshot session fields (plus CREDENTIALS) by name."""
    session = loki.current_session()
    out = {}
    for name in names:
        if name == "CREDENTIALS":
            out[name] = getattr(loki, name, _MISSING)
        else:
            out[name] = getattr(session, name, _MISSING)
    return out


def restore_loki_state(saved):
    session = loki.current_session()
    for name, value in saved.items():
        target = loki if name == "CREDENTIALS" else session
        if value is _MISSING:
            try:
                delattr(target, name)
            except AttributeError:
                pass
        else:
            setattr(target, name, value)


class ScriptedInputSession:
    def __init__(self, messages):
        self.messages = list(messages)
        self.user_messages = self
        self.on_submit = lambda text: False
        # Scripted input is never interactive, so terminal_ask_user yields
        # no seam and the Ask tool stays unadvertised.
        self.interactive = False
        self.reader = types.SimpleNamespace(
            cancel_requested=False,
            cancel_event=mock.Mock(),
        )

    async def get(self):
        while True:
            text = self.messages.pop(0)
            # Model the input owner's immediate-command interception, not a
            # consumer-side slash-command handler. Native PTY tests cover
            # actual queue timing.
            if text is None or not self.on_submit(text):
                return text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def modal(self):
        return self

    async def prompt(self, prompt=None, history=None):
        raise AssertionError(f"unexpected real modal prompt: {prompt!r}")


class TerminalImageCommandTests(unittest.TestCase):
    _state_names = [
        "CREDENTIALS",
        "runtime_config",
        "transcript_items",
        "session_todos",
        "session_toolsets",
        "session_state",
        "chat_log_path",
        "chat_log_dirty",
        "job_manager",
        "shell_cwd",
        "previous_shell_cwd",
        "agent_mode",
        "last_instructed_agent_mode",
    ]

    def setUp(self):
        self.saved_state = save_loki_state(self._state_names)

    def tearDown(self):
        restore_loki_state(self.saved_state)

    def test_loader_snapshots_relative_png_and_detects_real_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = b"\x89PNG\r\n\x1a\npayload"
            path = pathlib.Path(tmpdir, "picture.dat")
            path.write_bytes(data)

            image = terminal_frontend.load_image_attachment(
                "picture.dat", base_dir=tmpdir)
            path.write_bytes(b"changed later")

        self.assertEqual(
            os.path.realpath(image.path), os.path.realpath(path))
        self.assertEqual(image.media_type, "image/png")
        self.assertEqual(image.byte_size, len(data))
        self.assertEqual(
            base64.b64decode(image.encoded_data, validate=True), data)
        self.assertEqual(image.content_block(), {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image.encoded_data,
            },
        })

    def test_media_type_detection_covers_supported_formats(self):
        samples = {
            b"\x89PNG\r\n\x1a\n": "image/png",
            b"\xff\xd8\xff\xe0": "image/jpeg",
            b"GIF87a": "image/gif",
            b"GIF89a": "image/gif",
            b"RIFF\x04\x00\x00\x00WEBP": "image/webp",
        }
        for data, expected in samples.items():
            with self.subTest(expected=expected, data=data):
                self.assertEqual(
                    attachments.image_media_type(data), expected)

    def test_loader_rejects_missing_non_image_non_file_and_oversize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = pathlib.Path(tmpdir, "not-image.png")
            text_path.write_text("not an image", encoding="utf-8")
            large_path = pathlib.Path(tmpdir, "large.png")
            large_path.write_bytes(b"\x89PNG\r\n\x1a\npayload")

            with self.assertRaisesRegex(
                    terminal_frontend.ImageAttachmentError,
                    "cannot read"):
                terminal_frontend.load_image_attachment(
                    "missing.png", base_dir=tmpdir)
            with self.assertRaisesRegex(
                    terminal_frontend.ImageAttachmentError,
                    "unsupported image data"):
                terminal_frontend.load_image_attachment(
                    str(text_path), base_dir=tmpdir)
            # A directory is refused on both platforms; Windows reports it
            # through the denied open, so only the contract is asserted here.
            with self.assertRaises(terminal_frontend.ImageAttachmentError):
                terminal_frontend.load_image_attachment(
                    tmpdir, base_dir=tmpdir)
            with self.assertRaisesRegex(
                    terminal_frontend.ImageAttachmentError,
                    "maximum"):
                terminal_frontend.load_image_attachment(
                    str(large_path), base_dir=tmpdir, max_bytes=8)

    def test_command_path_supports_shell_quoting_and_requires_one_path(self):
        self.assertEqual(
            terminal_frontend._image_command_path(
                r'/image "screen shot.png"'),
            "screen shot.png",
        )
        self.assertEqual(
            terminal_frontend._image_command_path(
                r"/image screen\ shot.png"),
            "screen shot.png",
        )
        with self.assertRaisesRegex(
                terminal_frontend.ImageAttachmentError, "usage"):
            terminal_frontend._image_command_path("/image")
        with self.assertRaisesRegex(
                terminal_frontend.ImageAttachmentError, "quot"):
            terminal_frontend._image_command_path('/image "unterminated')

    def _run_terminal(self, messages, tmpdir, turn_runner=None):
        loki.CREDENTIALS = CredentialStore({
            "LOKI_API_BASE":
                "https://provider.example.test/v1/chat/completions",
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_API_KEY": "test-key",
            "LOKI_MODEL": "vision-model",
        })
        loki.current_session().shell_cwd = tmpdir
        session = ScriptedInputSession(messages)
        captured = []

        async def capture_turn(items, **kwargs):
            self.assertTrue(
                terminal_frontend._terminal_activity.turn_running)
            captured.append(copy.deepcopy(items))
            return ""

        stdout = io.StringIO()
        stderr = io.StringIO()
        path = os.path.join(tmpdir, "chat-test.json")

        def input_session(**kwargs):
            session.on_submit = kwargs["on_submit"]
            return session

        with mock.patch(
                "loki_agent.terminal_frontend.input_session",
                side_effect=input_session), mock.patch(
                    "loki_agent.terminal_frontend.new_chat_log_path",
                    return_value=path), mock.patch(
                        "loki_agent.terminal_frontend."
                        "restore_output_area_after_input"), mock.patch(
                            "loki_agent.terminal_frontend."
                            "run_terminal_turn_async",
                            new=turn_runner or capture_turn
                        ), contextlib.redirect_stdout(
                                stdout), contextlib.redirect_stderr(stderr):
            status = asyncio.run(terminal_frontend.async_main([]))
        return status, captured, stdout.getvalue(), stderr.getvalue()

    def test_model_receives_original_user_text(self):
        logical = (
            "first\x1b]0;owned\x07\t\n"
            "second\r\x00\u009b")

        with tempfile.TemporaryDirectory() as tmpdir:
            status, captured, _stdout, _stderr = self._run_terminal(
                [logical, "/quit"], tmpdir)

        self.assertEqual(status, 0)
        self.assertEqual(
            formats.item_text(captured[0][-1]), logical)

    def test_completed_turn_is_saved_before_terminal_cleanup(self):
        async def complete_turn(items, **_kwargs):
            items.append(formats.message_item(
                "assistant", "durable terminal answer"))
            return "durable terminal answer"

        with tempfile.TemporaryDirectory() as tmpdir:
            status, _captured, _stdout, _stderr = self._run_terminal(
                ["durable terminal question", "/quit"],
                tmpdir,
                turn_runner=complete_turn,
            )
            with open(
                    os.path.join(tmpdir, "chat-test.json"),
                    "r", encoding="utf-8") as file_obj:
                events, _todos, _state, _toolsets = (
                    savefiles.read_chat_log(file_obj))

        self.assertEqual(status, 0)
        self.assertIn(
            "durable terminal question",
            [formats.item_text(item) for item in events],
        )
        self.assertIn(
            "durable terminal answer",
            [formats.item_text(item) for item in events],
        )
        self.assertFalse(loki.current_session().chat_log_dirty)

    def test_image_command_attaches_snapshot_to_next_text_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = b"\x89PNG\r\n\x1a\npicture"
            pathlib.Path(tmpdir, "screen shot.png").write_bytes(data)

            status_updates = []
            with mock.patch(
                    "loki_agent.terminal_frontend.terminals."
                    "redraw_status_bar",
                    side_effect=lambda: status_updates.append(
                        terminal_frontend.status_text())):
                status, captured, _stdout, stderr = self._run_terminal(
                    [
                        '/image "screen shot.png"',
                        "What is wrong here?",
                        "Continue without the image.",
                        "/quit",
                    ],
                    tmpdir,
                )

        self.assertEqual(status, 0)
        self.assertEqual(len(captured), 2)
        user = captured[0][-1]
        self.assertEqual(user["type"], "message")
        self.assertEqual(user["role"], "user")
        self.assertEqual(user["content"][0], {
            "type": "text",
            "text": "What is wrong here?",
        })
        self.assertEqual(
            user["content"][1]["source"]["media_type"], "image/png")
        self.assertEqual(
            base64.b64decode(
                user["content"][1]["source"]["data"], validate=True),
            data,
        )
        self.assertNotIn(
            '/image "screen shot.png"',
            [formats.item_text(item) for item in captured[0]],
        )
        self.assertEqual(captured[1][-1], {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Continue without the image.",
            }],
        })
        self.assertIn("Attached image for next prompt:", stderr)
        self.assertIn("images: 1)", status_updates[0])
        self.assertIn("images: 0)", status_updates[1])

    def test_stale_image_delete_after_real_consumption_preserves_new_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pathlib.Path(tmpdir, "same.png").write_bytes(b"\x89PNG\r\n\x1a\npayload")
            with mock.patch.object(
                    terminal_frontend, "_queued_inputs",
                    terminal_frontend._QueuedInputs()):
                status, captured, stdout, _stderr = self._run_terminal(
                    ["/image same.png", "/queue images", "first prompt",
                     "/image same.png", "/queue images delete 1",
                     "second prompt", "/quit"], tmpdir)
        self.assertEqual(status, 0)
        self.assertEqual(len(captured), 2)
        for items in captured:
            self.assertEqual([block["type"] for block in items[-1]["content"]],
                             ["text", "image"])
        self.assertIn("Staged image ID 1 is no longer pending.", stdout)

    def test_empty_prompt_submits_all_staged_images_without_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pathlib.Path(tmpdir, "one.gif").write_bytes(
                b"GIF89aone")
            pathlib.Path(tmpdir, "two.webp").write_bytes(
                b"RIFF\x04\x00\x00\x00WEBPtwo")

            status, captured, _stdout, _stderr = self._run_terminal(
                ["/image one.gif", "/image two.webp", "", "/quit"],
                tmpdir,
            )

        self.assertEqual(status, 0)
        self.assertEqual(len(captured), 1)
        user = captured[0][-1]
        self.assertEqual(
            [block["type"] for block in user["content"]],
            ["image", "image"],
        )
        self.assertEqual(
            [block["source"]["media_type"] for block in user["content"]],
            ["image/gif", "image/webp"],
        )

    def test_turn_status_resets_after_cancellation(self):
        observed = []

        async def cancel_turn(_items, **_kwargs):
            observed.append(
                terminal_frontend._terminal_activity.turn_running)
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as tmpdir:
            status, _captured, _stdout, _stderr = self._run_terminal(
                ["cancel this turn", "/quit"],
                tmpdir,
                turn_runner=cancel_turn,
            )
            with open(
                    os.path.join(tmpdir, "chat-test.json"),
                    "r", encoding="utf-8") as file_obj:
                events, _todos, _state, _toolsets = (
                    savefiles.read_chat_log(file_obj))

        self.assertEqual(status, 0)
        self.assertEqual(observed, [True])
        self.assertIn(
            "cancel this turn",
            [formats.item_text(item) for item in events],
        )
        self.assertFalse(
            terminal_frontend._terminal_activity.turn_running)

    def test_slash_commands_do_not_start_a_turn(self):
        arguments = ["", "all", "99", "stop 99", "kill 99", "stop"]
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
                loki, "run_ps", wraps=loki.run_ps) as ps:
            status, captured, stdout, _stderr = self._run_terminal(
                ["/pwd"] + ["/ps " + argument for argument in arguments]
                + ["/quit"], tmpdir)
        self.assertEqual(
            [call.args[0].strip() for call in ps.call_args_list], arguments)
        self.assertIn("unknown job id", stdout)
        self.assertIn("usage: /ps", stdout)
        self.assertIn("No running, starting, or failed jobs.", stdout)
        self.assertIn("/ps all - history; /ps ID - tail", stdout)
        self.assertIn("/ps stop ID - stop; /ps kill ID - force termination", stdout)
        self.assertEqual(status, 0)
        self.assertEqual(captured, [])
        self.assertFalse(
            terminal_frontend._terminal_activity.turn_running)

    def test_job_redraw_observer_is_scoped_to_terminal_input(self):
        for fail in [False, True]:
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                manager = loki.JobManager(directory)
                previous = mock.Mock()
                manager.on_change = previous
                loki.current_session().job_manager = manager
                with mock.patch.object(terminals, "redraw_status_bar") as redraw:
                    async def turn(_items, **_kwargs):
                        before = redraw.call_count
                        manager.on_change()
                        self.assertEqual(redraw.call_count, before + 1)
                        if fail:
                            raise RuntimeError("turn failed")
                        return ""

                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "turn failed"):
                            self._run_terminal(["test"], directory, turn_runner=turn)
                    else:
                        self.assertEqual(self._run_terminal(
                            ["test", "/quit"], directory, turn_runner=turn)[0], 0)
                    previous.assert_not_called()
                    self.assertIs(manager.on_change, previous)
                    redraw.reset_mock()
                    manager.on_change()
                    previous.assert_called_once_with()
                    redraw.assert_not_called()

    def test_turn_status_resets_after_unexpected_failure(self):
        observed = []

        async def fail_turn(_items, **_kwargs):
            observed.append(
                terminal_frontend._terminal_activity.turn_running)
            raise RuntimeError("turn failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "turn failed"):
                self._run_terminal(
                    ["fail this turn"],
                    tmpdir,
                    turn_runner=fail_turn,
                )
            with open(
                    os.path.join(tmpdir, "chat-test.json"),
                    "r", encoding="utf-8") as file_obj:
                events, _todos, _state, _toolsets = (
                    savefiles.read_chat_log(file_obj))

        self.assertEqual(observed, [True])
        self.assertIn(
            "fail this turn",
            [formats.item_text(item) for item in events],
        )
        self.assertFalse(
            terminal_frontend._terminal_activity.turn_running)


class ProviderReinstallTests(unittest.TestCase):
    def test_reinstall_does_not_carry_custom_auth_to_new_provider(self):
        saved = save_loki_state(["runtime_config"])
        try:
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://custom.example.test/v1/chat/completions",
                protocols.OPENAI_CHAT,
                model="old-model",
                credential_ref=(
                    authentications.CredentialRef.environment(
                        "CUSTOM_API_KEY")),
                auth_header="X-Custom-Key",
            ))

            loki.reinstall_provider(
                model="claude-model",
                url="https://api.anthropic.com",
                provider_kind=protocols.ANTHROPIC_MESSAGES,
                provider_id="anthropic",
                credential_ref=(
                    authentications.CredentialRef.environment(
                        "ANTHROPIC_API_KEY")),
            )

            self.assertIsNone(
                loki.current_config().auth_spec.header_name)
            self.assertEqual(
                loki.current_config().auth_spec.scheme, "anthropic")
        finally:
            restore_loki_state(saved)

    def test_reinstall_does_not_invent_credential_for_new_provider(self):
        saved = save_loki_state(["runtime_config"])
        try:
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://old.example.test/v1/responses",
                protocols.OPENAI_RESPONSES,
                model="old-model",
                credential_ref=(
                    authentications.CredentialRef.environment(
                        "OLD_API_KEY")),
            ))

            loki.reinstall_provider(
                model="new-model",
                url="https://new.example.test/v1/responses",
                provider_id="new",
            )

            self.assertIsNone(loki.current_config().auth_spec)
        finally:
            restore_loki_state(saved)

    def test_reinstall_provider_requires_startup_config(self):
        names = ["runtime_config"]
        old_values = save_loki_state(names)

        try:
            loki.current_session().runtime_config = None
            with self.assertRaises(RuntimeError):
                loki.reinstall_provider(model="model-a")
        finally:
            restore_loki_state(old_values)

    def test_reinstall_preserves_status_only_for_the_same_catalog_entry(self):
        names = ["runtime_config"]
        old_values = save_loki_state(names)

        try:
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://example.test/v1",
                protocols.OPENAI_CHAT,
                model="old-model",
                provider_id="provider",
                credential_ref=authentications.CredentialRef.environment(
                    "PROVIDER_API_KEY"),
                model_status="deprecated",
            ))

            loki.reinstall_provider(model="old-model")
            self.assertEqual(
                loki.current_config().model_status, "deprecated")

            loki.reinstall_provider(model="new-model")
            self.assertIsNone(loki.current_config().model_status)
        finally:
            restore_loki_state(old_values)

    def test_reinstall_preserves_request_profile_only_for_same_model(self):
        saved = save_loki_state(["runtime_config"])
        try:
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://chatgpt.com/backend-api/codex/responses",
                protocols.OPENAI_RESPONSES,
                model="gpt-5.6-sol",
                provider_id="openai-subscription",
                credential_ref=(
                    authentications.CredentialRef.openai_subscription()),
                openai_request_profile=_codex_model(
                    "gpt-5.6-sol", use_responses_lite=True),
            ))

            loki.reinstall_provider(model="gpt-5.6-sol")
            self.assertTrue(
                loki.current_config().chat_provider.responses_lite)

            with self.assertRaisesRegex(
                    protocols.ProtocolError,
                    "require authenticated request profile"):
                loki.reinstall_provider(model="gpt-5.5")
        finally:
            restore_loki_state(saved)


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        assume_endpoints_approved(self)

    def test_reasoning_preference_is_sticky_across_model_capabilities(self):
        saved = save_loki_state([
            "runtime_config",
            "reasoning_effort_preference",
            "session_state",
            "chat_log_path",
            "chat_log_dirty",
        ])
        first = loki.make_runtime_config(
            "https://api.openai.com/v1/responses",
            protocols.OPENAI_RESPONSES,
            model="first",
            provider_id="openai",
            reasoning_effort_profile=_effort_profile("low", "high"),
        )
        narrower = loki.make_runtime_config(
            "https://api.openai.com/v1/responses",
            protocols.OPENAI_RESPONSES,
            model="narrower",
            provider_id="openai",
            reasoning_effort_profile=_effort_profile("low", "medium"),
        )
        unsupported = loki.make_runtime_config(
            "https://api.openai.com/v1/responses",
            protocols.OPENAI_RESPONSES,
            model="unsupported",
        )
        try:
            loki.current_session().session_state = {}
            loki.current_session().chat_log_path = None
            loki.apply_runtime_config(first)
            loki.current_session().reasoning_effort_preference = "high"
            self.assertEqual(loki.effective_reasoning_effort(), "high")

            loki.apply_runtime_config(narrower)
            self.assertIsNone(loki.effective_reasoning_effort())
            self.assertEqual(
                loki.current_reasoning_effort_preference(), "high")
            self.assertIn(
                "preferred high is unavailable",
                loki.reasoning_effort_default_text(),
            )

            loki.apply_runtime_config(unsupported)
            self.assertIsNone(loki.effective_reasoning_effort())
            self.assertIsNone(loki.reasoning_effort_status_text())

            loki.apply_runtime_config(first)
            self.assertEqual(loki.effective_reasoning_effort(), "high")

            loki.set_reasoning_effort(None)
            self.assertIsNone(
                loki.current_reasoning_effort_preference())
            self.assertIsNone(loki.effective_reasoning_effort())
            self.assertEqual(
                loki.reasoning_effort_status_text(),
                "unknown",
            )
        finally:
            restore_loki_state(saved)

    def test_effort_status_shows_only_the_active_value(self):
        config = loki.make_runtime_config(
            "https://api.z.ai/api/paas/v4/chat/completions",
            protocols.OPENAI_CHAT, model="model", provider_id="zai",
            reasoning_effort_profile=modelsdev.ReasoningEffortProfile(
                ["low", "medium", "high"]),
        )
        session = loki.Session(runtime_config=config)
        with mock.patch.object(loki, "_DEFAULT_SESSION", session):
            # No fabricated provider default: with no selection and no
            # advertised default, the level is the model's own, unknown
            # to the catalog.
            for preference, expected in (
                    (None, "unknown"), ("high", "high"),
                    ("unavailable", "unknown")):
                with self.subTest(preference=preference):
                    session.reasoning_effort_preference = preference
                    self.assertEqual(loki.reasoning_effort_status_text(), expected)
                    text = terminal_frontend.status_text()
                    self.assertIn(f"Effort: {expected}, Context:", text)
                    self.assertNotIn("Model default", text)
                    self.assertNotIn("preferred", text)
            session.reasoning_effort_preference = None
            self.assertEqual(
                loki.reasoning_effort_default_text(), "Model default")
            config.reasoning_effort_profile = modelsdev.ReasoningEffortProfile(
                ["low", "medium", "high"], default="low")
            self.assertEqual(loki.reasoning_effort_status_text(), "low")

    def test_delegated_config_reconstructs_reasoning_profile(self):
        profile = _effort_profile("low", "high")
        inventory = CredentialInventory({
            "LOKI_API_BASE": "https://api.openai.com/v1/responses",
            "LOKI_PROVIDER": "openai_responses",
            "LOKI_MODEL": "gpt-test",
            "LOKI_PROVIDER_ID": "openai",
            "LOKI_REASONING_EFFORT_PROFILE": json.dumps(
                profile.to_dict()),
        })

        config = loki.build_config_from_env(credentials=inventory)

        self.assertEqual(config.reasoning_effort_profile, profile)

    def test_reasoning_preference_round_trips_in_session_state(self):
        saved = save_loki_state([
            "runtime_config",
            "reasoning_effort_preference",
            "transcript_items",
            "session_todos",
            "session_toolsets",
            "session_state",
            "chat_log_path",
            "chat_log_dirty",
            "conversation_id",
        ])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-reasoning.json")
                loki.apply_runtime_config(loki.make_runtime_config(
                    "https://api.openai.com/v1/responses",
                    protocols.OPENAI_RESPONSES,
                    model="gpt-test",
                    provider_id="openai",
                    reasoning_effort_profile=_effort_profile(
                        "low", "high"),
                ))
                loki.new_chat_log(path)
                loki.set_thinking_controls({"effort": "high"})

                with open(path, "r", encoding="utf-8") as stream:
                    blob = json.load(stream)
                self.assertEqual(
                    blob["session_state"]["reasoning_effort"], "high")

                loki.current_session().reasoning_effort_preference = None
                loki.load_chat_log(path)
                self.assertEqual(
                    loki.current_reasoning_effort_preference(), "high")
                self.assertEqual(
                    loki.effective_reasoning_effort(), "high")
        finally:
            restore_loki_state(saved)

    def test_tool_loop_snapshots_reasoning_effort_for_all_requests(self):
        saved = save_loki_state([
            "runtime_config",
            "reasoning_effort_preference",
            "session_state",
            "chat_log_path",
            "chat_log_dirty",
        ])
        seen = []

        async def completion(
                items, tools=None, *, codex_turn_state,
                reasoning_effort=None, **kwargs):
            seen.append(reasoning_effort)
            if len(seen) == 1:
                loki.current_session().reasoning_effort_preference = "low"
                return formats.DecodedTurn([
                    formats.tool_call_item(
                        "call_1", "Read", {"file_path": "README.md"}),
                ])
            return formats.DecodedTurn([
                formats.message_item("assistant", "done"),
            ])

        async def dispatch(
                fn_name, args, allowed=None, extra_context=None):
            return {"ok": True, "content": "contents"}

        try:
            loki.current_session().session_state = {}
            loki.current_session().chat_log_path = None
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://api.openai.com/v1/responses",
                protocols.OPENAI_RESPONSES,
                model="gpt-test",
                provider_id="openai",
                reasoning_effort_profile=_effort_profile("low", "high"),
            ))
            loki.current_session().reasoning_effort_preference = "high"
            with (
                    mock.patch.object(
                        loki, "async_chat_completion", new=completion),
                    mock.patch.object(
                        loki, "dispatch_tool_async", new=dispatch)):
                result = asyncio.run(loki.run_tool_loop_async(
                    [formats.message_item("user", "read")],
                    max_loops=3,
                ))
        finally:
            restore_loki_state(saved)

        self.assertEqual(result, "done")
        self.assertEqual(seen, ["high", "high"])

    def test_tool_loop_snapshots_model_default_explicitly(self):
        saved = save_loki_state([
            "runtime_config",
            "reasoning_effort_preference",
            "session_state",
            "chat_log_path",
            "chat_log_dirty",
        ])
        seen = []

        async def completion(
                items, tools=None, *, codex_turn_state,
                reasoning_effort="not-passed", **kwargs):
            seen.append(reasoning_effort)
            loki.current_session().reasoning_effort_preference = "high"
            return formats.DecodedTurn([
                formats.message_item("assistant", "done"),
            ])

        try:
            loki.current_session().session_state = {}
            loki.current_session().chat_log_path = None
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://api.openai.com/v1/responses",
                protocols.OPENAI_RESPONSES,
                model="gpt-test",
                provider_id="openai",
                reasoning_effort_profile=_effort_profile("low", "high"),
            ))
            loki.current_session().reasoning_effort_preference = None
            with mock.patch.object(
                    loki, "async_chat_completion", new=completion):
                result = asyncio.run(loki.run_tool_loop_async(
                    [formats.message_item("user", "hello")],
                ))
        finally:
            restore_loki_state(saved)

        self.assertEqual(result, "done")
        self.assertEqual(seen, [None])

    def test_delegated_config_reconstructs_subscription_authentication(self):
        credential = authentications.CredentialRef.openai_subscription()
        profile = _codex_model(
            "gpt-5-codex", use_responses_lite=True)
        inventory = CredentialInventory({
            "LOKI_API_BASE":
                "https://chatgpt.com/backend-api/codex/responses",
            "LOKI_PROVIDER": "openai_responses",
            "LOKI_MODEL": "gpt-5-codex",
            "LOKI_CREDENTIAL_REF": credential.encode(),
            "LOKI_AUTH_SCHEME": "openai-subscription",
            "LOKI_OPENAI_REQUEST_PROFILE": json.dumps(
                profile.to_dict()),
        }, {credential})

        config = loki.build_config_from_env(credentials=inventory)

        self.assertEqual(config.auth_spec.credential, credential)
        self.assertEqual(config.auth_spec.scheme, "openai-subscription")
        self.assertEqual(
            config.chat_provider.provider_id, "openai-subscription")
        self.assertTrue(config.chat_provider.responses_lite)
        self.assertEqual(
            config.chat_provider.headers[
                protocols.RESPONSES_LITE_HEADER], "true")

    def test_delegated_config_rejects_undelegated_credential(self):
        inventory = CredentialInventory({
            "LOKI_API_BASE": "https://example.test/v1/responses",
            "LOKI_PROVIDER": "openai_responses",
            "LOKI_MODEL": "model",
            "LOKI_CREDENTIAL_REF": "env:NOT_DELEGATED_TOKEN",
        })

        with self.assertRaisesRegex(
                ValueError, "unavailable credential"):
            loki.build_config_from_env(credentials=inventory)

    def test_explicit_streaming_is_opt_in_and_validated(self):
        base = {
            "LOKI_API_BASE": "http://localhost:8000/v1/chat/completions",
            "LOKI_PROVIDER": "openai_chat",
            "LOKI_MODEL": "local-model",
        }

        enabled = loki.build_config_from_env({
            **base, "LOKI_STREAM": "1",
        })
        disabled = loki.build_config_from_env(base)

        self.assertTrue(enabled.stream)
        self.assertFalse(disabled.stream)
        with self.assertRaisesRegex(ValueError, "LOKI_STREAM must be"):
            loki.build_config_from_env({
                **base, "LOKI_STREAM": "sometimes",
            })

    def test_dummy_provider_honors_stream_setting(self):
        config = loki.build_config_from_env({
            "LOKI_API_BASE": "http://dummy.invalid/v1",
            "LOKI_PROVIDER": "dummy",
            "LOKI_STREAM": "1",
        })

        self.assertTrue(config.stream)

    def test_anthropic_prompt_cache_defaults_only_for_anthropic_api(self):
        direct = loki.build_config_from_env({
            "LOKI_API_BASE": "https://api.anthropic.com/v1/messages",
            "LOKI_PROVIDER": "anthropic_messages",
            "LOKI_MODEL": "claude-test",
        })
        compatible = loki.build_config_from_env({
            "LOKI_API_BASE": "https://compatible.example/v1/messages",
            "LOKI_PROVIDER": "anthropic_messages",
            "LOKI_MODEL": "compatible-test",
        })
        opted_in = loki.build_config_from_env({
            "LOKI_API_BASE": "https://compatible.example/v1/messages",
            "LOKI_PROVIDER": "anthropic_messages",
            "LOKI_MODEL": "compatible-test",
            "LOKI_PROMPT_CACHE": "1",
        })

        self.assertTrue(direct.chat_provider.prompt_cache)
        self.assertFalse(compatible.chat_provider.prompt_cache)
        self.assertTrue(opted_in.chat_provider.prompt_cache)
        with self.assertRaisesRegex(ValueError, "LOKI_PROMPT_CACHE must be"):
            loki.build_config_from_env({
                "LOKI_API_BASE":
                    "https://compatible.example/v1/messages",
                "LOKI_PROVIDER": "anthropic_messages",
                "LOKI_MODEL": "compatible-test",
                "LOKI_PROMPT_CACHE": "sometimes",
            })

    def test_no_builtin_connection_exists(self):
        for credential_name in [
                "OPENCODE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            with self.subTest(credential_name=credential_name):
                env = {credential_name: "provider-key"}
                with self.assertRaisesRegex(
                        ValueError,
                        "API endpoint missing"):
                    loki.build_config_from_env(env)

    def test_unrelated_sdk_base_variables_do_not_configure_loki(self):
        for base_name in ["OPENAI_API_BASE", "ANTHROPIC_BASE_URL"]:
            with self.subTest(base_name=base_name):
                env = {
                    base_name: "https://unrelated.example.test/v1",
                    "ANTHROPIC_API_KEY": "unrelated-key",
                }
                credentials = CredentialStore.capture(env)
                self.assertFalse(
                    loki.explicit_api_base_configured(credentials))
                with self.assertRaisesRegex(ValueError, "API endpoint missing"):
                    loki.build_config_from_env(
                        env, credentials=credentials)

    def test_custom_connection_does_not_use_generic_credentials(self):
        for credential_name in [
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENCODE_API_KEY"]:
            with self.subTest(credential_name=credential_name):
                env = {
                    "LOKI_API_BASE":
                        "https://custom.example.test/v1/chat/completions",
                    credential_name: "must-not-be-sent",
                }
                config = loki.build_config_from_env(env)

                self.assertIsNone(config.auth_spec)
                self.assertNotIn(
                    "Authorization", config.chat_provider.headers)
                self.assertNotIn(
                    "x-api-key", config.chat_provider.headers)

    def test_saved_connection_requires_its_exact_credential(self):
        effort_profile = _effort_profile("low", "high")
        descriptor = ConnectionDescriptor(
            provider_id="openrouter",
            provider_name="OpenRouter",
            model="z-ai/glm",
            chat_url="https://openrouter.ai/api/v1/chat/completions",
            models_url="https://openrouter.ai/api/v1/models",
            protocol=protocols.OPENAI_CHAT,
            credential_ref=(
                authentications.CredentialRef.environment(
                    "OPENROUTER_API_KEY")),
            model_status="deprecated",
            reasoning_effort_profile=effort_profile,
        )
        with self.assertRaisesRegex(
                ValueError, "missing env:OPENROUTER_API_KEY"):
            loki.config_from_connection_descriptor(
                descriptor,
                CredentialStore({"LOKI_API_KEY": "wrong-provider-key"}),
            )

        config = loki.config_from_connection_descriptor(
            descriptor,
            CredentialStore({
                "OPENROUTER_API_KEY": "right-key",
                "LOKI_MODEL": "override-model",
            }),
        )
        self.assertEqual(
            config.chat_provider.input_url, descriptor.chat_url)
        self.assertEqual(
            config.auth_spec.credential,
            authentications.CredentialRef.environment(
                "OPENROUTER_API_KEY"))
        self.assertEqual(config.model, "override-model")
        self.assertIsNone(config.model_status)
        self.assertIsNone(config.reasoning_effort_profile)
        self.assertEqual(
            config.chat_provider.provider_id, "openrouter")

        restored = loki.config_from_connection_descriptor(
            descriptor,
            CredentialStore({"OPENROUTER_API_KEY": "right-key"}),
        )
        self.assertEqual(restored.model, "z-ai/glm")
        self.assertEqual(restored.model_status, "deprecated")
        self.assertEqual(
            restored.reasoning_effort_profile, effort_profile)

        protocol_override = loki.config_from_connection_descriptor(
            descriptor,
            CredentialStore({
                "OPENROUTER_API_KEY": "right-key",
                "LOKI_PROVIDER": protocols.OPENAI_RESPONSES,
            }),
        )
        self.assertIsNone(
            protocol_override.reasoning_effort_profile)

    def test_subscription_refresh_keeps_connection_without_valid_selector(
            self):
        credential = (
            authentications.CredentialRef.openai_subscription())
        descriptor = ConnectionDescriptor(
            provider_id="openai-subscription",
            provider_name="OpenAI ChatGPT subscription",
            model="gpt-test",
            chat_url=authentications.OPENAI_CHATGPT_RESPONSES_URL,
            models_url=authentications.OPENAI_CHATGPT_MODELS_REQUEST_URL,
            protocol=protocols.OPENAI_RESPONSES,
            credential_ref=credential,
            auth_scheme="openai-subscription",
            openai_request_profile=_codex_model(
                supports_reasoning_summaries=True,
                default_reasoning_level="max"),
            reasoning_effort_profile=_effort_profile("max"),
        )
        response = {
            "models": [{
                "slug": "gpt-test",
                "display_name": "GPT Test",
                "visibility": "list",
                "input_modalities": ["text"],
                "supported_reasoning_levels": [
                    {"effort": "max"},
                    {"effort": "max"},
                ],
                "default_reasoning_level": "max",
                "supports_reasoning_summaries": True,
                "default_reasoning_summary": "auto",
                "supports_parallel_tool_calls": True,
            }],
        }
        diagnostics = []

        with mock.patch.object(
                modelsdev,
                "fetch_openai_subscription_models",
                new=mock.AsyncMock(return_value=response)):
            refreshed = asyncio.run(
                loki.refresh_connection_descriptor_async(
                    descriptor,
                    object(),
                    diagnostic_writer=diagnostics.append,
                ))

        self.assertEqual(refreshed.model, "gpt-test")
        self.assertIsNotNone(refreshed.openai_request_profile)
        self.assertIsNone(refreshed.reasoning_effort_profile)
        self.assertTrue(any(
            "Ignoring reasoning effort choices" in item
            for item in diagnostics
        ))

    def test_subscription_descriptor_uses_saved_profile_only_when_offline(
            self):
        credential = (
            authentications.CredentialRef.openai_subscription())
        descriptor = ConnectionDescriptor(
            provider_id="openai-subscription",
            provider_name="OpenAI ChatGPT subscription",
            model="gpt-test",
            chat_url=authentications.OPENAI_CHATGPT_RESPONSES_URL,
            models_url=authentications.OPENAI_CHATGPT_MODELS_REQUEST_URL,
            protocol=protocols.OPENAI_RESPONSES,
            credential_ref=credential,
            auth_scheme="openai-subscription",
            openai_request_profile=_codex_model(),
        )
        diagnostics = []

        with mock.patch.object(
                modelsdev,
                "fetch_openai_subscription_models",
                new=mock.AsyncMock(side_effect=OSError("offline"))):
            refreshed = asyncio.run(
                loki.refresh_connection_descriptor_async(
                    descriptor,
                    object(),
                    diagnostic_writer=diagnostics.append,
                ))

        self.assertEqual(refreshed, descriptor)
        self.assertTrue(any("using saved" in item for item in diagnostics))

    def test_saved_subscription_cannot_redirect_access_token(self):
        credential = (
            authentications.CredentialRef.openai_subscription())
        descriptor = ConnectionDescriptor(
            provider_id="openai-subscription",
            provider_name="OpenAI ChatGPT subscription",
            model="gpt-test",
            chat_url="https://attacker.example/v1/responses",
            models_url=None,
            protocol=protocols.OPENAI_RESPONSES,
            credential_ref=credential,
            auth_scheme="openai-subscription",
            openai_request_profile=_codex_model("gpt-test"),
        )

        with self.assertRaises(
                authentications.CredentialUnavailable):
            loki.config_from_connection_descriptor(
                descriptor,
                CredentialInventory({}, {credential}),
            )

    def test_saved_prompt_cache_setting_restores_without_reinference(self):
        descriptor = ConnectionDescriptor(
            provider_id="compatible",
            provider_name="Compatible",
            model="compatible-test",
            chat_url="https://compatible.example/v1/messages",
            models_url="https://compatible.example/v1/models",
            protocol=protocols.ANTHROPIC_MESSAGES,
            prompt_cache=True,
        )

        restored = loki.config_from_connection_descriptor(
            descriptor, CredentialStore({}))
        overridden = loki.config_from_connection_descriptor(
            descriptor, CredentialStore({"LOKI_PROMPT_CACHE": "0"}))

        self.assertTrue(restored.chat_provider.prompt_cache)
        self.assertFalse(overridden.chat_provider.prompt_cache)


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        assume_endpoints_approved(self)
        names = [
            "runtime_config", "CREDENTIALS", "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "job_manager",
            "shell_cwd", "previous_shell_cwd",
        ]
        self.old_values = save_loki_state(names)

    def tearDown(self):
        restore_loki_state(self.old_values)

    def test_provider_model_discovery_does_not_select_a_model(self):
        loki.apply_runtime_config(loki.make_runtime_config(
            "https://provider.example.test/v1/chat/completions",
            protocols.OPENAI_CHAT,
            model="",
            models_url="https://provider.example.test/v1/models",
            credential_ref=authentications.CredentialRef.environment(
                "LOKI_API_KEY"),
        ))
        response = {
            "data": [
                {"id": "first-model"},
                {"id": "second-model"},
            ],
        }

        with mock.patch(
                "loki_agent.loki.async_provider_request",
                new=mock.AsyncMock(
                    return_value=protocols.ProviderResponse(response))):
            loaded_models = asyncio.run(loki.load_models_async())

        self.assertEqual(loaded_models, ["first-model", "second-model"])
        self.assertEqual(loki.current_model(), "")
        self.assertEqual(loki.current_config().model, "")

    def test_explicit_connection_option_requires_complete_loki_config(self):
        self.assertIsNone(loki.explicit_connection_option(
            CredentialStore({
                "LOKI_API_BASE": "http://localhost:8000/v1",
                "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            })))

        option = loki.explicit_connection_option(CredentialStore({
            "LOKI_API_BASE": "http://localhost:8000/v1",
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_MODEL": "private-model",
        }))

        self.assertEqual(option, modelsdev.ExplicitConnectionOption(
            model="private-model",
            api_url="http://localhost:8000/v1",
            protocol=protocols.OPENAI_CHAT,
        ))

    def test_interactive_startup_does_not_fetch_provider_models(self):
        loki.CREDENTIALS = CredentialStore({
            "LOKI_API_BASE":
                "http://localhost:8000/v1/chat/completions",
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_MODEL": "chosen-model",
        })
        session = ScriptedInputSession([None])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-test.json")
            loader = mock.AsyncMock()
            with mock.patch(
                    "loki_agent.terminal_frontend.input_session",
                    return_value=session), mock.patch(
                        "loki_agent.terminal_frontend.new_chat_log_path",
                        return_value=path), mock.patch(
                            "loki_agent.terminal_frontend.restore_output_area_after_input"
                        ), mock.patch(
                            "loki_agent.terminal_frontend.load_models_async",
                            new=loader):
                status = asyncio.run(terminal_frontend.async_main([]))

        loader.assert_not_awaited()
        self.assertEqual(status, 0)
        self.assertEqual(loki.current_model(), "chosen-model")
        self.assertIsNone(loki.current_config().auth_spec)
        self.assertNotIn(
            "Authorization",
            loki.current_config().chat_provider.headers)

    def test_headless_startup_requires_an_explicit_model(self):
        loki.CREDENTIALS = CredentialStore({
            "LOKI_API_BASE":
                "https://provider.example.test/v1/chat/completions",
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_API_KEY": "test-key",
        })
        loader = mock.AsyncMock()
        runner = mock.AsyncMock()
        stderr = io.StringIO()

        with mock.patch(
                "loki_agent.terminal_frontend.load_models_async",
                new=loader), mock.patch(
                    "loki_agent.terminal_frontend.subagents.run_cli_async",
                    new=runner), contextlib.redirect_stderr(stderr):
            status = asyncio.run(terminal_frontend.async_main(["--headless"]))

        loader.assert_not_awaited()
        runner.assert_not_awaited()
        self.assertEqual(status, 2)
        self.assertIn(
            "Configuration error: model missing; set LOKI_MODEL.",
            stderr.getvalue(),
        )

    def test_headless_configuration_failure_returns_usage_error(self):
        loki.CREDENTIALS = CredentialStore({})
        runner = mock.AsyncMock()
        stderr = io.StringIO()

        with mock.patch(
                "loki_agent.terminal_frontend.subagents.run_cli_async",
                new=runner), contextlib.redirect_stderr(stderr):
            status = asyncio.run(terminal_frontend.async_main(["--headless"]))

        runner.assert_not_awaited()
        self.assertEqual(status, 2)
        self.assertIn(
            "Configuration error: API endpoint missing",
            stderr.getvalue(),
        )

    def test_requested_resume_read_failure_returns_error(self):
        loki.CREDENTIALS = CredentialStore({})
        session = ScriptedInputSession([])
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "missing-chat.json")
            with mock.patch(
                    "loki_agent.terminal_frontend.input_session",
                    return_value=session), contextlib.redirect_stderr(stderr):
                status = asyncio.run(
                    terminal_frontend.async_main([f"--resume={missing}"]))

        self.assertEqual(status, 1)
        self.assertIn("Could not resume chat:", stderr.getvalue())

    def test_requested_resume_rejects_invalid_saved_reasoning_effort(self):
        loki.CREDENTIALS = CredentialStore({})
        session = ScriptedInputSession([])
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad-reasoning-chat.json")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(savefiles.serialize_chat_log(
                    [],
                    [],
                    {"reasoning_effort": 42},
                ))
            with mock.patch(
                    "loki_agent.terminal_frontend.input_session",
                    return_value=session), contextlib.redirect_stderr(stderr):
                status = asyncio.run(
                    terminal_frontend.async_main([f"--resume={path}"]))

        self.assertEqual(status, 1)
        self.assertIn(
            "Could not resume chat: invalid saved reasoning effort",
            stderr.getvalue(),
        )

    def test_interactive_resume_accepts_saved_credentialless_connection(self):
        loki.CREDENTIALS = CredentialStore({})
        session = ScriptedInputSession([None])
        resumed_question = (
            "visible\x1b]0;owned\x07 resumed question\nnext\tline")
        assistant_markdown = (
            "## Resume heading\n\n"
            "**visible resumed answer** and `code` "
            "\x1b]0;assistant-owned\x07\u009b\n\n"
            "```python\n"
            "print('raw **inside fence**')\n"
            "```"
        )
        descriptor = ConnectionDescriptor(
            provider_id=None,
            provider_name="Explicit LOKI_* connection",
            model="local-model",
            chat_url="http://localhost:8000/v1/chat/completions",
            models_url="http://localhost:8000/v1/models",
            protocol=protocols.OPENAI_CHAT,
            stream=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-test.json")
            events = loki.initial_transcript_items() + [
                formats.message_item("user", resumed_question),
                formats.model_response_event(
                    protocols.OPENAI_CHAT,
                    [formats.message_item(
                        "assistant", assistant_markdown)],
                    model="local-model",
                ),
            ]
            blob = formats.new_log_blob(
                events, [])
            blob["session_state"] = {
                "shell_cwd": loki.current_cwd(),
                "connection": descriptor.to_dict(),
            }
            pathlib.Path(path).write_text(
                json.dumps(blob), encoding="utf-8")
            confirm = mock.AsyncMock(return_value=True)
            stdout = io.StringIO()
            with mock.patch(
                    "loki_agent.terminal_frontend.input_session",
                    return_value=session), mock.patch(
                        "loki_agent.terminal_frontend.confirm_saved_connection_async",
                        new=confirm), mock.patch(
                            "loki_agent.terminal_frontend."
                            "terminal.markdown_style",
                            True), mock.patch(
                            "loki_agent.terminal_frontend.restore_output_area_after_input"), \
                    contextlib.redirect_stdout(stdout):
                status = asyncio.run(
                    terminal_frontend.async_main([f"--resume={path}"]))

        confirm.assert_awaited_once()
        self.assertEqual(status, 0)
        rendered = stdout.getvalue()
        self.assertIn(
            "User: visible^[]0;owned^G resumed question\n"
            "next^Iline",
            rendered,
        )
        self.assertNotIn("\x1b]0;owned\x07", rendered)
        self.assertNotIn(
            "\x1b]0;assistant-owned\x07", rendered)
        self.assertIn(
            "^[]0;assistant-owned^G\\x9b", rendered)
        self.assertIn(
            "local-model: "
            + terminals.render_markdown(assistant_markdown, style=True),
            rendered,
        )
        self.assertIn(
            "\033[7m## Resume heading\033[27m", rendered)
        self.assertIn(
            "\033[1mvisible resumed answer\033[0m", rendered)
        self.assertIn("\033[36mcode\033[0m", rendered)
        self.assertNotIn("**visible resumed answer**", rendered)
        self.assertIn("raw **inside fence**", rendered)
        self.assertTrue(rendered.endswith("----\n"))
        self.assertEqual(loki.current_model(), "local-model")
        self.assertIsNone(loki.current_config().auth_spec)
        self.assertTrue(loki.current_config().stream)
        self.assertNotIn(
            "Authorization",
            loki.current_config().chat_provider.headers)

    def test_status_commands_inspect_and_save_without_sending_chat(self):
        from loki_agent.response_headers import Store
        loki.CREDENTIALS = CredentialStore({})
        session = ScriptedInputSession([
            "/status", "/status --json", "/status all", "/status all --json",
            "/status save", "/quit"])
        runner = mock.AsyncMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Store(os.path.join(tmpdir, "status.json"))
            store.observer("https://example.test/chat", None, "test")(
                200, {"x-remaining": "8"})
            with mock.patch.object(loki.current_session(), "response_headers", store), \
                    mock.patch.object(terminal_frontend, "input_session",
                                      return_value=session), \
                    mock.patch.object(terminal_frontend, "new_chat_log_path",
                                      return_value=os.path.join(tmpdir, "chat.json")), \
                    mock.patch.object(terminal_frontend, "restore_output_area_after_input"), \
                    mock.patch.object(terminal_frontend, "run_terminal_turn_async", runner), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                status = asyncio.run(terminal_frontend.async_main([]))
            self.assertEqual(status, 0)
            saved = Store(store.path).snapshot()["endpoints"]
            self.assertEqual(saved[0]["headers"]["x-remaining"]["value"], "8")
            current_output, all_output = output.getvalue().split("User: /status all", 1)
            self.assertIn("No active HTTP chat connection", current_output)
            self.assertNotIn("x-remaining", current_output)
            self.assertIn('"endpoints": []', current_output)
            self.assertIn("x-remaining", all_output)
            self.assertIn("All known connections", all_output)
            self.assertIn("unsaved observations are not visible", all_output)
            self.assertIn("Response status saved", output.getvalue())
        runner.assert_not_awaited()
        self.assertFalse(any(formats.item_text(item).startswith("/status")
                             for item in loki.current_transcript()))

    def test_status_defaults_to_current_endpoint_and_credential(self):
        from loki_agent.response_headers import Store
        endpoint = "https://example.test/v1/chat/completions"
        for authenticated in [False, True]:
            with self.subTest(authenticated=authenticated), \
                    tempfile.TemporaryDirectory() as tmpdir:
                environment = {"LOKI_API_BASE": endpoint, "LOKI_MODEL": "model"}
                credential = None
                if authenticated:
                    environment["LOKI_API_KEY"] = "test-key"
                    credential = "env:LOKI_API_KEY"
                loki.CREDENTIALS = CredentialStore(environment)
                store = Store(os.path.join(tmpdir, "status.json"))
                store.observer(endpoint, credential, "model")(
                    200, {"selected-only": "1"})
                store.observer(endpoint, "env:OTHER", "model")(
                    200, {"other-credential-only": "2"})
                store.observer("https://other.example/chat", credential, "model")(
                    200, {"other-endpoint-only": "3"})
                session = ScriptedInputSession([
                    "/status", "/status --json", "/status all", "/quit"])
                runner = mock.AsyncMock()
                with mock.patch.object(loki.current_session(), "response_headers", store), \
                        mock.patch.object(terminal_frontend, "input_session",
                                          return_value=session), \
                        mock.patch.object(terminal_frontend, "new_chat_log_path",
                                          return_value=os.path.join(tmpdir, "chat.json")), \
                        mock.patch.object(terminal_frontend, "restore_output_area_after_input"), \
                        mock.patch.object(terminal_frontend, "run_terminal_turn_async", runner), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(asyncio.run(terminal_frontend.async_main([])), 0)
                current_output, all_output = output.getvalue().split("User: /status all", 1)
                self.assertIn("selected-only", current_output)
                self.assertNotIn("other-credential-only", current_output)
                self.assertNotIn("other-endpoint-only", current_output)
                self.assertIn("Current endpoint and credential", current_output)
                self.assertIn("last observed, not live balances", current_output)
                self.assertIn("other-credential-only", all_output)
                self.assertIn("other-endpoint-only", all_output)
                runner.assert_not_awaited()

    def test_chat_request_without_a_model_is_not_sent(self):
        loki.CREDENTIALS = CredentialStore({
            "LOKI_API_BASE":
                "https://provider.example.test/v1/chat/completions",
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_API_KEY": "test-key",
        })
        session = ScriptedInputSession(["do not send this", "/quit"])
        runner = mock.AsyncMock()
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-test.json")
            with mock.patch(
                    "loki_agent.terminal_frontend.input_session",
                    return_value=session), mock.patch(
                        "loki_agent.terminal_frontend.new_chat_log_path",
                        return_value=path), mock.patch(
                            "loki_agent.terminal_frontend.restore_output_area_after_input"
                        ), mock.patch(
                            "loki_agent.terminal_frontend.run_terminal_turn_async",
                            new=runner), contextlib.redirect_stderr(stderr):
                status = asyncio.run(terminal_frontend.async_main([]))

        runner.assert_not_awaited()
        self.assertEqual(status, 0)
        self.assertNotIn(
            "do not send this",
            [formats.item_text(item) for item in loki.current_transcript()],
        )
        self.assertIn(
            "No model selected; use /model or set LOKI_MODEL.",
            stderr.getvalue(),
        )

    def test_provider_fallback_cancel_preserves_selected_model(self):
        loki.CREDENTIALS = CredentialStore({
            "LOKI_API_BASE":
                "https://provider.example.test/v1/chat/completions",
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_API_KEY": "test-key",
            "LOKI_MODEL": "current-model",
        })
        session = ScriptedInputSession(["/model", "/quit"])

        async def load_provider_models(diagnostic_writer=None):
            loki.current_session().models = ["current-model", "other-model"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-test.json")
            loader = mock.AsyncMock(side_effect=load_provider_models)
            with mock.patch(
                    "loki_agent.terminal_frontend.input_session",
                    return_value=session), mock.patch(
                        "loki_agent.terminal_frontend.new_chat_log_path",
                        return_value=path), mock.patch(
                            "loki_agent.terminal_frontend.restore_output_area_after_input"
                        ), mock.patch(
                            "loki_agent.terminal_frontend.load_models_async",
                            new=loader), mock.patch(
                                "loki_agent.terminal_frontend.modelsdev."
                                "run_model_picker_async",
                                new=mock.AsyncMock(
                                    side_effect=OSError("offline"))), \
                    mock.patch(
                        "loki_agent.terminal_frontend.modelsdev."
                        "run_flat_model_picker_async",
                        new=mock.AsyncMock(return_value=None)):
                status = asyncio.run(terminal_frontend.async_main([]))

        loader.assert_awaited_once()
        self.assertEqual(status, 0)
        self.assertEqual(loki.current_model(), "current-model")
        self.assertEqual(loki.current_config().model, "current-model")

    def test_provider_fallback_selection_preserves_connection(self):
        async def workflow():
            models_url = "https://catalog.example.test/custom/models"
            owner = credential_supervisors.CredentialSupervisor(CredentialStore({
                "LOKI_API_BASE": "https://provider.example.test/v1/chat/completions",
                "LOKI_PROVIDER": protocols.OPENAI_CHAT, "LOKI_API_KEY": "fallback-secret",
                "LOKI_MODELS_URL": models_url,
            }))
            from test_http_client import FakeConnector

            def response(data):
                body = json.dumps(data).encode()
                return b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" + f"Content-Length: {len(body)}\r\n\r\n".encode() + body

            def answer(text):
                return {"object": "chat.completion", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}

            connector = FakeConnector([
                response({"data": [{"id": "first-model"}, {"id": "selected-model"}]}),
                response(answer("fallback durable answer")), response(answer("fallback resumed answer"))])
            catalog_attempts = []
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "chat-fallback.json")
                session = loki.Session(shell_cwd=directory, job_manager=loki.JobManager(directory))
                session.credential_authority = owner.broker
                self.addCleanup(lambda: asyncio.run(session.job_manager.close_session_owned()))

                class Input(ScriptedInputSession):
                    async def prompt(inner_self, prompt=None, history=None, *, initial_text=""):
                        self.assertIn("filter WORDS", prompt)
                        self.assertEqual(initial_text, "filter ")
                        return "2"

                inputs = Input(["/model", "answer with fallback model", None])

                async def connect(host, port, **kwargs):
                    self.assertEqual(port, 443)
                    self.assertIsNotNone(kwargs.get("ssl"))
                    if host == "models.dev":
                        catalog_attempts.append(host)
                        raise OSError("catalog offline")
                    self.assertIn(host, ("catalog.example.test", "provider.example.test"))
                    return await connector.open_connection(host, port, **kwargs)

                saved = modelsdev._index_cache
                modelsdev._index_cache = None
                try:
                    with mock.patch.object(loki, "_DEFAULT_SESSION", session), \
                            mock.patch.object(loki, "CREDENTIALS", owner.inventory), \
                            mock.patch.object(asyncio, "open_connection", side_effect=connect), \
                            mock.patch.object(terminal_frontend, "input_session", return_value=inputs), \
                            mock.patch.object(terminal_frontend, "new_chat_log_path", return_value=path), \
                            mock.patch.object(terminal_frontend, "restore_output_area_after_input"), \
                            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(await asyncio.wait_for(terminal_frontend.async_main([]), 10), 0)
                        loki.save_chat_log()
                        self.assertEqual(loki.current_config().chat_provider.models_url, models_url)
                        await session.job_manager.close_session_owned()
                finally:
                    modelsdev._index_cache = saved
                saved = json.loads(pathlib.Path(path).read_text())
                self.assertEqual(saved["session_state"]["connection"]["models_url"], models_url)
                self.assertEqual(saved["session_state"]["connection"]["model"], "selected-model")
                self.assertEqual(formats.item_text(saved["events"][-1]), "fallback durable answer")
                self.assertNotIn("fallback-secret", pathlib.Path(path).read_text())
                self.assertEqual(catalog_attempts, ["models.dev"])
                fresh = credential_supervisors.CredentialSupervisor(CredentialStore({"LOKI_API_KEY": "fallback-secret"}))
                resumed = loki.Session(shell_cwd=directory, job_manager=loki.JobManager(os.path.join(directory, "resumed-jobs")))
                resumed.credential_authority = fresh.broker
                self.addCleanup(lambda: asyncio.run(resumed.job_manager.close_session_owned()))

                class ResumeInput(ScriptedInputSession):
                    async def prompt(inner_self, prompt=None, history=None):
                        self.assertEqual(prompt, "Use this saved connection? [y/N]: ")
                        return "yes"

                with mock.patch.object(loki, "_DEFAULT_SESSION", resumed), \
                        mock.patch.object(loki, "CREDENTIALS", fresh.inventory), \
                        mock.patch.object(asyncio, "open_connection", side_effect=connect), \
                        mock.patch.object(terminal_frontend, "input_session", return_value=ResumeInput(["continue fallback", None])), \
                        mock.patch.object(terminals, "open_terminal_stdin"), \
                        mock.patch.object(terminal_frontend, "restore_output_area_after_input"), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(await asyncio.wait_for(terminal_frontend.async_main([f"--resume={path}"]), 10), 0)
                    self.assertEqual(loki.current_config().chat_provider.models_url, models_url)
                    loki.save_chat_log()
                    await resumed.job_manager.close_session_owned()
                final = json.loads(pathlib.Path(path).read_text())
                self.assertEqual(final["events"][:len(saved["events"])], saved["events"])
                self.assertEqual(formats.item_text(final["events"][-1]), "fallback resumed answer")
                self.assertEqual(final["session_state"]["connection"], saved["session_state"]["connection"])
                self.assertNotIn("fallback-secret", pathlib.Path(path).read_text())
                self.assertEqual(connector.responses, [])
                requests, payloads = [], []
                for connection, writer in zip(connector.calls, connector.writers):
                    self.assertTrue(writer.closed)
                    self.assertTrue(writer.wait_closed_called)
                    head, body = bytes(writer.data).split(b"\r\n\r\n", 1)
                    lines = head.decode().split("\r\n")
                    method, target, version = lines[0].split(" ")
                    self.assertEqual(version, "HTTP/1.1")
                    headers = dict(line.split(": ", 1) for line in lines[1:])
                    self.assertEqual(headers["Authorization"], "Bearer fallback-secret")
                    requests.append((method, "https://" + connection["host"] + target))
                    if body:
                        payload = json.loads(body)
                        self.assertEqual(payload["model"], "selected-model")
                        payloads.append(payload)
                self.assertEqual(requests, [("GET", models_url)] + [
                    ("POST", "https://provider.example.test/v1/chat/completions")] * 2)
                self.assertIn("fallback durable answer", json.dumps(payloads[1]))

        asyncio.run(workflow())

    def test_explicit_connection_is_selectable_when_modelsdev_is_offline(self):
        explicit_url = "http://localhost:8000/v1"
        loki.CREDENTIALS = CredentialStore({
            "LOKI_API_BASE": explicit_url,
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_API_KEY": "local-key",
            "LOKI_MODEL": "private-model",
        })
        session = ScriptedInputSession(["/model", "/quit"])
        seen_explicit = []

        async def pick_flat(input_fn, model_ids,
                            explicit_connection=None, *, text_writer):
            seen_explicit.append(explicit_connection)
            return explicit_connection

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-test.json")
            with mock.patch(
                    "loki_agent.terminal_frontend.input_session",
                    return_value=session), mock.patch(
                        "loki_agent.terminal_frontend.new_chat_log_path",
                        return_value=path), mock.patch(
                            "loki_agent.terminal_frontend.restore_output_area_after_input"
                        ), mock.patch(
                            "loki_agent.terminal_frontend.modelsdev."
                            "run_model_picker_async",
                            new=mock.AsyncMock(
                                side_effect=OSError("offline"))), mock.patch(
                                    "loki_agent.terminal_frontend.load_models_async",
                                    new=mock.AsyncMock()), mock.patch(
                                        "loki_agent.terminal_frontend.modelsdev."
                                        "run_flat_model_picker_async",
                                        new=mock.AsyncMock(
                                            side_effect=pick_flat)):
                status = asyncio.run(terminal_frontend.async_main([]))

        self.assertEqual(status, 0)
        self.assertEqual(len(seen_explicit), 1)
        self.assertIsInstance(
            seen_explicit[0], modelsdev.ExplicitConnectionOption)
        self.assertEqual(
            loki.current_config().chat_provider.input_url, explicit_url)
        self.assertEqual(loki.current_config().model, "private-model")


class TerminalReasoningEffortTests(unittest.TestCase):
    _state_names = [
        "CREDENTIALS",
        "runtime_config",
        "reasoning_effort_preference",
        "transcript_items",
        "session_todos",
        "session_toolsets",
        "session_state",
        "chat_log_path",
        "chat_log_dirty",
    ]

    def setUp(self):
        assume_endpoints_approved(self)
        self.saved = save_loki_state(self._state_names)

    def tearDown(self):
        restore_loki_state(self.saved)

    def test_effort_command_selects_and_persists_value(self):
        loki.CREDENTIALS = CredentialStore({
            "OPENROUTER_API_KEY": "secret",
        })
        provider = {
            "id": "openrouter",
            "name": "OpenRouter",
            "npm": "@openrouter/ai-sdk-provider",
            "env": ["OPENROUTER_API_KEY"],
            "api": "https://openrouter.ai/api/v1",
        }
        model = {
            "id": "gpt-test",
            "name": "GPT Test",
            "reasoning_options": [{
                "type": "effort",
                "values": ["low", "high"],
            }],
        }
        session = ScriptedInputSession(
            ["/model", "/thinking effort high", "/quit"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-test.json")
            with (
                    mock.patch(
                        "loki_agent.terminal_frontend.input_session",
                        return_value=session),
                    mock.patch(
                        "loki_agent.terminal_frontend.new_chat_log_path",
                        return_value=path),
                    mock.patch(
                        "loki_agent.terminal_frontend."
                        "restore_output_area_after_input"),
                    mock.patch(
                        "loki_agent.terminal_frontend.modelsdev."
                        "run_model_picker_async",
                        new=mock.AsyncMock(return_value=(
                            "openrouter", provider, model))),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO())):
                status = asyncio.run(terminal_frontend.async_main([]))
            with open(path, "r", encoding="utf-8") as stream:
                saved = json.load(stream)

        self.assertEqual(status, 0)
        self.assertEqual(
            loki.current_reasoning_effort_preference(), "high")
        self.assertEqual(
            saved["session_state"]["reasoning_effort"], "high")
        self.assertIn(
            "Effort: high, Context: unknown; /model, /thinking",
            terminal_frontend.status_text(),
        )

    def test_picker_offers_default_then_exact_model_values(self):
        loki.apply_runtime_config(loki.make_runtime_config(
            "https://api.openai.com/v1/responses",
            protocols.OPENAI_RESPONSES,
            model="gpt-test",
            provider_id="openai",
            reasoning_effort_profile=_effort_profile("minimal", "high"),
        ))
        loki.current_session().session_state = {}
        loki.current_session().chat_log_path = None
        loki.current_session().reasoning_effort_preference = "high"

        self.assertEqual(
            terminal_frontend._reasoning_effort_rows(),
            [
                (None, "Model default"),
                ("minimal", "minimal"),
                ("high", "high"),
            ],
        )


class SelectionConversationWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_numeric_selection_approval_and_resumed_conversation(self):
        await self._workflow("numeric")

    async def test_filtered_selection_and_custom_header_resume(self):
        await self._workflow("custom")

    async def test_display_name_filter_and_credentialless_switch_resume(self):
        await self._workflow("credentialless")

    async def _workflow(self, variant):
        from loki_agent import acp_worker, endpoint_pins
        from test_endpoint_pins import _StateDir
        from test_models_dev import DATA

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name)
        source = root / "selection.txt"
        source.write_text("selection witness\n", encoding="utf-8")
        path = str(root / "chat-selection.json")
        raw = copy.deepcopy(DATA)
        raw["openrouter"]["models"]["z-ai/glm-5.2"].update({
            "status": "deprecated", "provider": {
                "api": "https://effective.example/v1", "npm": "@ai-sdk/openai"}})
        raw["openai"] = {
            "id": "openai", "name": "OpenAI", "npm": "@ai-sdk/openai",
            "env": ["OPENAI_API_KEY"],
            "models": {"gpt-test": {"id": "gpt-test", "name": "GPT Test", "status": "deprecated"}}}
        raw["openai"]["models"]["gpt-override"] = {
            "id": "gpt-override", "name": "GPT Override",
            "provider": {"api": "https://normalized-effective.example/v1"}}
        values = {
            "LOKI_API_KEY": "stale-explicit-secret", "OPENROUTER_API_KEY": "selected-router-secret",
            "OPENAI_API_KEY": "selected-openai-secret", "ANTHROPIC_API_KEY": "selected-anthropic-secret",
            "ZHIPU_API_KEY": "unused-zhipu-secret", "LOKI_MAX_TOKENS": "1234",
            "LOKI_ANTHROPIC_VERSION": "2024-01-01", "LOKI_STREAM": "0"}
        owner = credential_supervisors.CredentialSupervisor(CredentialStore(values))
        session = loki.Session(shell_cwd=str(root), job_manager=loki.JobManager(str(root / "jobs")))
        session.credential_authority = owner.broker
        self.addAsyncCleanup(session.job_manager.close_session_owned)
        requests, prompts, approvals = [], [], []
        output = io.StringIO()
        errors = io.StringIO()
        phase = {}
        saved_index = modelsdev._index_cache
        self.addCleanup(setattr, modelsdev, "_index_cache", saved_index)
        modelsdev._index_cache = None

        def reply(text=None, tool=False):
            if phase["protocol"] == protocols.ANTHROPIC_MESSAGES:
                return {"type": "message", "id": "anthropic-selection", "role": "assistant",
                        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1}}
            if phase["protocol"] == protocols.OPENAI_CHAT:
                return {"object": "chat.completion", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
            items = ([{"type": "function_call", "call_id": "selection-read", "name": "Read",
                      "arguments": json.dumps({"file_path": str(source)})}] if tool else
                     [{"type": "message", "role": "assistant", "content": [
                         {"type": "output_text", "text": text}]}])
            return {"object": "response", "status": "completed", "output": items}

        async def request(method, url, **kwargs):
            if method == "GET":
                self.assertEqual(url, modelsdev.MODELS_DEV_URL)
                self.assertEqual(kwargs["headers_in"]["Accept"], "application/json")
                self.assertNotIn("Authorization", kwargs["headers_in"])
                return http_client.HttpResponse(url, 200, "OK", {}, json.dumps(raw).encode())
            self.assertEqual(method, "POST")
            self.assertEqual(url, phase["url"])
            headers = kwargs["headers_in"]
            credential = phase.get("credential")
            if credential:
                self.assertEqual(headers[phase["header"]], phase.get("prefix", "") + credential)
            for header in ["Authorization", "x-api-key", "X-Custom-Key"]:
                if not credential or header != phase["header"]:
                    self.assertNotIn(header, headers)
            payload = json.loads(kwargs["body"])
            self.assertEqual(payload["model"], phase["model"])
            if phase["protocol"] == protocols.ANTHROPIC_MESSAGES:
                self.assertEqual(payload["max_tokens"], 1234)
                self.assertEqual(headers["anthropic-version"], "2024-01-01")
            if phase.get("history"):
                serialized = json.dumps(payload)
                self.assertIn("router durable answer", serialized)
                self.assertIn("selection-read", serialized)
                self.assertIn("1\\tselection witness", serialized)
            for prior_answer in phase.get("resume_answers", []):
                self.assertIn(prior_answer, json.dumps(payload))
            if len(phase["queue"]) == 1 and phase.get("tool"):
                self.assertEqual([(item["call_id"], item["output"]) for item in payload["input"]
                                  if item.get("type") == "function_call_output"],
                                 [("selection-read", "1\tselection witness")])
            requests.append((url, copy.deepcopy(headers), payload))
            data = phase["queue"].pop(0)
            return http_client.HttpResponse(url, 200, "OK", {}, json.dumps(data).encode())

        async def response_chunks(data):
            if phase["protocol"] == protocols.OPENAI_RESPONSES:
                for item in data["output"]:
                    yield ("data: " + json.dumps({"type": "response.output_item.done", "item": item}) + "\n\n").encode()
                yield ("data: " + json.dumps({"type": "response.completed", "response": {**data, "output": []}}) + "\n\n").encode()
            else:
                self.assertEqual(phase["protocol"], protocols.OPENAI_CHAT)
                text = data["choices"][0]["message"]["content"]
                for delta, finish in (({"role": "assistant", "content": text}, None), ({}, "stop")):
                    yield ("data: " + json.dumps({"object": "chat.completion.chunk", "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n").encode()
                yield b"data: [DONE]\n\n"

        # Route only the external connection to a loopback service. The production
        # HTTP serializer, parser, streaming transport and provider decoders run.
        service_tasks, service_errors, wire_requests = [], [], []

        async def serve(reader, writer):
            service_tasks.append(asyncio.current_task())
            try:
                head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
                lines = head.decode("latin-1").split("\r\n")
                method, target, version = lines[0].split(" ")
                self.assertEqual(version, "HTTP/1.1")
                headers = dict(line.split(": ", 1) for line in lines[1:] if line)
                body = await asyncio.wait_for(reader.readexactly(int(headers.get("Content-Length", "0"))), 5)
                url = "https://" + headers["Host"] + target
                wire_requests.append((method, url))
                response = await request(method, url, headers_in=headers, body=body)
                response_body = response.body
                content_type = "application/json"
                if body and json.loads(body).get("stream"):
                    content_type = "text/event-stream"
                    response_body = b"".join([chunk async for chunk in response_chunks(json.loads(response_body))])
                writer.write(b"HTTP/1.1 200 OK\r\n" + f"Content-Type: {content_type}\r\n".encode()
                             + f"Content-Length: {len(response_body)}\r\n\r\n".encode() + response_body)
                await asyncio.wait_for(writer.drain(), 5)
            except Exception as error:
                service_errors.append(error)
            finally:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), 5)

        server = await asyncio.start_server(serve, "127.0.0.1", 0)

        async def close_service():
            server.close()
            await asyncio.wait_for(server.wait_closed(), 5)
            if service_tasks:
                await asyncio.wait_for(asyncio.gather(*service_tasks), 10)

        self.addAsyncCleanup(close_service)
        port = server.sockets[0].getsockname()[1]
        real_connect = asyncio.open_connection

        async def connect(host, port_in, **kwargs):
            from urllib.parse import urlsplit
            self.assertEqual(port_in, 443)
            self.assertIsNotNone(kwargs.get("ssl"))
            self.assertIn(host, ("models.dev", urlsplit(phase["url"]).hostname))
            return await real_connect("127.0.0.1", port)

        def prepare(url, model, protocol, secret=None, *, header="Authorization", history=False, tool=False):
            phase.clear()
            phase.update(url=url, model=model, protocol=protocol, credential=secret, header=header,
                         prefix="Bearer " if header == "Authorization" else "", history=history, tool=tool)
            phase["answer"] = {
                "z-ai/glm-5.2": "router durable answer", "router-replacement": "router replacement durable answer",
                "gpt-test": "openai durable answer", "gpt-override": "override durable answer",
                "claude-sonnet-4-6": "anthropic durable answer", "beta": "credentialless durable answer",
                "explicit-model": "explicit durable answer"}[model]
            phase["queue"] = ([reply(tool=True)] if tool else []) + [reply(text=phase["answer"])]

        async def turn(prompt):
            loki.set_session_connection(loki.active_connection_descriptor())
            loki.current_transcript().append(formats.message_item("user", prompt))
            answer = await asyncio.wait_for(loki.run_tool_loop_async(
                loki.current_transcript(), allowed={"Read"}, max_loops=3), 10)
            self.assertEqual(phase["queue"], [])
            self.assertEqual(answer, phase["answer"])
            loki.mark_chat_log_dirty()
            loki.save_chat_log()

        async def pick(answers, expected_pid, expected_api, expected_ref):
            script = iter(answers)
            previous = endpoint_pins.load()
            provider_prompt_seen = False
            last_input_render_end = len(output.getvalue())

            async def input_fn(prompt=None, history=None, *, initial_text=""):
                nonlocal provider_prompt_seen, last_input_render_end
                if expected_pid == "openrouter" and prompt.startswith("Provider choice") and not provider_prompt_seen:
                    shown = output.getvalue()[render_start:]
                    self.assertIn("1. OpenRouter id=z-ai/glm-5.2", shown)
                    self.assertIn("2. Zhipu AI id=glm-5.2", shown)
                    provider_prompt_seen = True
                prompts.append(prompt)
                answer = next(script)
                if prompt == "Send this credential to this endpoint? [y/N] ":
                    self.assertEqual(endpoint_pins.load(), previous)
                    self.assertEqual(endpoint_pins.status(expected_pid, expected_api, expected_ref)[0],
                                     endpoint_pins.CHANGED if expected_pid in previous else endpoint_pins.NEW)
                    shown = output.getvalue()[last_input_render_end:]
                    self.assertIn(expected_api, shown)
                    self.assertIn(expected_ref, shown)
                    worker = acp_worker.Worker(session, lambda message: None, "approval")
                    worker._option_leaves = {"chosen": next(
                        leaf for members in modelsdev._index_cache[1].values() for leaf in members
                        if leaf[0] == expected_pid
                        and modelsdev.provider_access(leaf[1], owner.inventory).api_url == expected_api)}
                    active = loki.current_config()
                    requests_before = len(requests)
                    with self.assertRaisesRegex(ValueError, "approve it once"):
                        loki.config_from_modelsdev_selection(*worker._option_leaves["chosen"], owner.inventory)
                    self.assertIs(loki.current_config(), active)
                    self.assertEqual(len(requests), requests_before)
                    description = worker.describe_config_selection({"value": "chosen"})
                    self.assertEqual(description["providerId"], expected_pid)
                    self.assertEqual(description["endpoint"], expected_api)
                    self.assertEqual(description["credential"], expected_ref)
                    self.assertEqual(description["changed"], expected_pid in previous)
                    approvals.append((expected_pid, expected_api, expected_ref))
                last_input_render_end = len(output.getvalue())
                return answer

            render_start = len(output.getvalue())
            leaf = await modelsdev.run_model_picker_async(
                input_fn, owner.inventory, text_writer=output.write)
            rendered = output.getvalue()[render_start:]
            self.assertTrue(rendered.startswith("\nUsable models:\n"),
                            {'stdout': repr(rendered), 'stderr': repr(errors.getvalue())})
            self.assertEqual(rendered.count("\nUsable models:\n"), 2 if answers[0].startswith("filter") else 1)
            self.assertEqual(rendered.count("\nUsable providers:\n"), 2 if len(answers) == 5 else 1)
            self.assertEqual(leaf[0], expected_pid)
            approved_pair = {"api": expected_api, "credential": expected_ref}
            self.assertEqual(endpoint_pins.status(expected_pid, expected_api, expected_ref),
                             (endpoint_pins.PINNED, approved_pair))
            self.assertEqual(endpoint_pins.load(), {**previous, expected_pid: approved_pair})
            self.assertEqual(json.loads(pathlib.Path(state_directory, "loki", "provider-endpoints.json").read_text()),
                             {**previous, expected_pid: approved_pair})
            with self.assertRaises(StopIteration):
                next(script)
            return leaf

        with _StateDir() as state_directory, mock.patch.object(loki, "_DEFAULT_SESSION", session), \
                mock.patch.object(loki, "CREDENTIALS", owner.inventory), \
                mock.patch.object(asyncio, "open_connection", side_effect=connect), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            # An explicit endpoint leases only LOKI_API_KEY, never a generic SDK key.
            env = {**values, "LOKI_API_BASE": "https://api.deepseek.com/anthropic",
                   "LOKI_PROVIDER": protocols.ANTHROPIC_MESSAGES, "LOKI_MODEL": "explicit-model"}
            if variant == "custom":
                env["LOKI_AUTH_HEADER"] = "X-Custom-Key"
            config = loki.build_config_from_env(env)
            self.assertEqual(config.auth_spec.scheme, "custom" if variant == "custom" else "anthropic")
            self.assertEqual(config.auth_spec.header_name, "X-Custom-Key" if variant == "custom" else None)
            self.assertNotIn("X-Custom-Key", config.chat_provider.headers)
            self.assertFalse(any(name.endswith("API_KEY") for name in env))
            self.assertEqual(config.auth_spec.credential, authentications.CredentialRef.environment("LOKI_API_KEY"))
            self.assertNotIn("x-api-key", config.chat_provider.headers)
            self.assertEqual(config.chat_provider.input_url, "https://api.deepseek.com/anthropic")
            loki.apply_runtime_config(config)
            loki.new_chat_log(path)
            if variant == "credentialless":
                startup_values = {name: value for name, value in values.items() if name != "LOKI_API_KEY"}
                startup_values.pop("LOKI_STREAM")
                startup_values.update(LOKI_API_BASE="https://local.example/v1",
                                      LOKI_PROVIDER=protocols.OPENAI_CHAT, LOKI_MODEL="beta")
                prepare("https://local.example/v1/chat/completions", "beta", protocols.OPENAI_CHAT)
                with mock.patch.object(loki, "CREDENTIALS", CredentialStore(startup_values)):
                    self.assertEqual(await asyncio.wait_for(terminal_frontend.async_main(
                        ["--headless", "--prompt=credentialless startup"]), 10), 0)
                self.assertIsNone(loki.current_config().auth_spec)
                self.assertFalse(loki.current_config().stream)
                self.assertEqual(phase["queue"], [])
            else:
                prepare("https://api.deepseek.com/anthropic/v1/messages", "explicit-model", protocols.ANTHROPIC_MESSAGES,
                        "stale-explicit-secret", header="X-Custom-Key" if variant == "custom" else "x-api-key")
                await turn("explicit request")

            choices = {"numeric": ["2", "1", "y"],
                       "custom": ["filter openrouter", "1", "filter openrouter", "1", "y"],
                       "credentialless": ["filter zhipu ai", "1", "1", "y"]}[variant]
            leaf = await pick(choices, "openrouter", "https://effective.example/v1", "env:OPENROUTER_API_KEY")
            self.assertEqual(leaf[2]["id"], "z-ai/glm-5.2")
            worker = acp_worker.Worker(session, lambda message: None, "selection")
            worker._option_leaves = {"chosen": leaf}
            self.assertEqual(worker.describe_config_selection({"value": "chosen"}), {})
            selection_credentials = CredentialStore({**values, "LOKI_STREAM": "1"}) if variant == "custom" else owner.inventory
            config = loki.config_from_modelsdev_selection(*leaf, selection_credentials)
            self.assertEqual(config.model_status, "deprecated")
            self.assertEqual(config.auth_spec.credential, authentications.CredentialRef.environment("OPENROUTER_API_KEY"))
            self.assertIsNone(config.auth_spec.header_name)
            loki.apply_runtime_config(config)
            self.assertIn("Model: z-ai/glm-5.2 (deprecated), Context: unknown; /model", terminal_frontend.status_text())
            self.assertEqual(loki.current_config().stream, variant == "custom")
            self.assertNotIn("Authorization", loki.current_config().chat_provider.headers)
            prepare("https://effective.example/v1/responses", "z-ai/glm-5.2", protocols.OPENAI_RESPONSES,
                    "selected-router-secret", tool=True)
            await turn("read the selection witness")
            self.assertEqual(endpoint_pins.status("openrouter", raw["openrouter"]["api"], "env:OPENROUTER_API_KEY")[0],
                             endpoint_pins.CHANGED)
            before_retry = len(prompts)
            await pick(["filter openrouter", "1", "1"], "openrouter", "https://effective.example/v1", "env:OPENROUTER_API_KEY")
            self.assertEqual(len(prompts) - before_retry, 3)
            self.assertEqual(len(approvals), 1)
            persisted_router = json.loads(pathlib.Path(path).read_text())
            self.assertEqual(persisted_router["session_state"]["connection"]["model_status"], "deprecated")

            original_provider = loki.current_config().chat_provider
            loki.reinstall_provider(model="router-replacement")
            self.assertIsNot(loki.current_config().chat_provider, original_provider)
            self.assertEqual(loki.current_model(), "router-replacement")
            self.assertEqual(loki.current_config().model, "router-replacement")
            self.assertEqual(loki.current_config().chat_provider.kind, protocols.OPENAI_RESPONSES)
            self.assertEqual(loki.current_config().chat_provider.max_tokens, 1234)
            self.assertEqual(loki.current_config().stream, variant == "custom")
            self.assertEqual(loki.current_config().auth_spec.credential,
                             authentications.CredentialRef.environment("OPENROUTER_API_KEY"))
            prepare("https://effective.example/v1/responses", "router-replacement", protocols.OPENAI_RESPONSES,
                    "selected-router-secret", history=True)
            await turn("replace model on the same provider")

            # Reapproval replaces, rather than accumulates, a provider's durable pair.
            raw["openrouter"]["models"]["z-ai/glm-5.2"]["provider"]["api"] = "https://replacement.example/v1"
            modelsdev._index_cache = None
            leaf = await pick(["filter openrouter", "1", "1", "yes"], "openrouter",
                              "https://replacement.example/v1", "env:OPENROUTER_API_KEY")
            loki.apply_runtime_config(loki.config_from_modelsdev_selection(*leaf, owner.inventory))
            prepare("https://replacement.example/v1/responses", "z-ai/glm-5.2", protocols.OPENAI_RESPONSES,
                    "selected-router-secret", history=True)
            await turn("replace approved endpoint")
            pin_path = pathlib.Path(state_directory, "loki", "provider-endpoints.json")
            self.assertEqual(json.loads(pin_path.read_text())["openrouter"], {
                "api": "https://replacement.example/v1", "credential": "env:OPENROUTER_API_KEY"})
            self.assertEqual(endpoint_pins.load()["openrouter"], {
                "api": "https://replacement.example/v1", "credential": "env:OPENROUTER_API_KEY"})
            self.assertEqual(
                endpoint_pins.status("openrouter", "https://effective.example/v1", "env:OPENROUTER_API_KEY")[0],
                endpoint_pins.CHANGED)

            # Replace the provider/protocol through another real approved catalog selection.
            leaf = await pick(["filter anthropic", "1", "1", "y"], "anthropic",
                              "https://api.anthropic.com/v1/messages", "env:ANTHROPIC_API_KEY")
            selected = loki.config_from_modelsdev_selection(*leaf, owner.inventory)
            loki.reinstall_provider(model=selected.model, url="https://api.anthropic.com",
                                    provider_kind=selected.chat_provider.kind, provider_id="anthropic",
                                    anthropic_version=selected.chat_provider.headers["anthropic-version"],
                                    credential_ref=selected.auth_spec.credential)
            self.assertEqual(loki.current_config().auth_spec.scheme, "anthropic")
            self.assertEqual(loki.current_config().chat_provider.max_tokens, 1234)
            prepare("https://api.anthropic.com/v1/messages", "claude-sonnet-4-6", protocols.ANTHROPIC_MESSAGES,
                    "selected-anthropic-secret", header="x-api-key", history=True)
            await turn("switch protocol")

            if variant == "credentialless":
                # The actual terminal /model handler replaces a loaded connection
                # and then selects the explicit credentialless option again.
                answers = iter(["GPT Test", "1", "1", "yes", "beta", "1", "1"])

                class SwitchInput(ScriptedInputSession):
                    async def get(inner_self):
                        message = await super().get()
                        if message == "/model" and loki.current_model() == "gpt-test":
                            self.assertIn("Model: gpt-test (deprecated), Context: unknown; /model", terminal_frontend.status_text())
                            self.assertEqual(json.loads(pathlib.Path(path).read_text())["session_state"]["connection"]["model_status"],
                                             "deprecated")
                        if message == "normalized selection":
                            self.assertEqual(loki.current_config().chat_provider.provider_name,
                                             "OpenAI Platform API [endpoint supplied by Loki]")
                            prepare("https://api.openai.com/v1/responses", "gpt-test", protocols.OPENAI_RESPONSES,
                                    "selected-openai-secret", history=True)
                        elif message == "switch to credentialless":
                            self.assertIsNone(loki.current_config().auth_spec)
                            self.assertEqual(loki.current_config().chat_provider.input_url, "https://local.example/v1")
                            self.assertEqual(loki.current_config().chat_provider.provider_name, "Explicit LOKI_* connection")
                            prepare("https://local.example/v1/chat/completions", "beta", protocols.OPENAI_CHAT, history=True)
                        return message

                    async def prompt(inner_self, prompt=None, history=None, *, initial_text=""):
                        answer = next(answers)
                        self.assertEqual(initial_text, "filter " if answer in ["GPT Test", "beta"] else "")
                        if prompt == "Send this credential to this endpoint? [y/N] ":
                            self.assertEqual(
                                endpoint_pins.status("openai", "https://api.openai.com/v1", "env:OPENAI_API_KEY"),
                                (endpoint_pins.NEW, None))
                            self.assertIn("env:OPENAI_API_KEY", output.getvalue())
                        else:
                            self.assertIn("filter WORDS", prompt)
                        return initial_text + answer

                startup_values["LOKI_STREAM"] = "1"
                inputs = SwitchInput(["/model", "normalized selection", "/model", "switch to credentialless", None])
                with mock.patch.object(loki, "CREDENTIALS", CredentialStore(startup_values)), \
                        mock.patch.object(terminal_frontend, "input_session", return_value=inputs), \
                        mock.patch.object(terminals, "open_terminal_stdin"), \
                        mock.patch.object(terminal_frontend, "restore_output_area_after_input"):
                    self.assertEqual(await asyncio.wait_for(terminal_frontend.async_main([f"--resume={path}"]), 10), 0)
                    loki.save_chat_log()
                self.assertEqual(phase["queue"], [])
                with self.assertRaises(StopIteration):
                    next(answers)
            else:
                leaf = await pick(["filter GPT Test", "1", "1", "y"], "openai",
                                  "https://api.openai.com/v1", "env:OPENAI_API_KEY")
                loki.apply_runtime_config(loki.config_from_modelsdev_selection(*leaf, owner.inventory))
                self.assertEqual(loki.current_config().chat_provider.provider_name,
                                 "OpenAI Platform API [endpoint supplied by Loki]")
                self.assertEqual(loki.current_config().chat_provider.models_url, "https://api.openai.com/v1/models")
                prepare("https://api.openai.com/v1/responses", "gpt-test", protocols.OPENAI_RESPONSES,
                        "selected-openai-secret", history=True)
                await turn("normalized selection")
                if variant == "numeric":
                    leaf = await pick(["filter GPT Override", "1", "1", "yes"], "openai",
                                      "https://normalized-effective.example/v1", "env:OPENAI_API_KEY")
                    self.assertEqual(leaf[2]["id"], "gpt-override")
                    loki.apply_runtime_config(loki.config_from_modelsdev_selection(*leaf, owner.inventory))
                    prepare("https://normalized-effective.example/v1/responses", "gpt-override", protocols.OPENAI_RESPONSES,
                            "selected-openai-secret", history=True)
                    await turn("override a normalized provider endpoint")

            expected_prompts = ["read the selection witness", "replace model on the same provider",
                                "replace approved endpoint", "switch protocol", "normalized selection"]
            expected_answers = ["router durable answer", "router replacement durable answer",
                                "router durable answer", "anthropic durable answer", "openai durable answer"]
            if variant != "credentialless":
                expected_prompts.insert(0, "explicit request")
                expected_answers.insert(0, "explicit durable answer")
            if variant == "credentialless":
                expected_prompts.append("switch to credentialless")
                expected_answers.append("credentialless durable answer")
            elif variant == "numeric":
                expected_prompts.append("override a normalized provider endpoint")
                expected_answers.append("override durable answer")
            before = json.loads(pathlib.Path(path).read_text())
            self.assertEqual(before["session_state"]["connection"], loki.active_connection_descriptor().to_dict())
            self.assertEqual([(item["call_id"], formats.item_text(item)) for item in before["events"]
                              if item.get("type") == "tool_result"], [("selection-read", "1\tselection witness")])
            formats.validate_events(before["events"])
            await session.job_manager.close_session_owned()
            fresh_values = {name: value for name, value in values.items() if name.endswith("API_KEY")}
            if variant == "custom":
                fresh_values["LOKI_AUTH_HEADER"] = "X-Custom-Key"
            fresh_owner = credential_supervisors.CredentialSupervisor(CredentialStore(fresh_values))
            self.assertIsNot(fresh_owner.broker, owner.broker)
            resumed = loki.Session(shell_cwd=str(root), job_manager=loki.JobManager(str(root / "resumed-jobs")))
            resumed.credential_authority = fresh_owner.broker
            self.addAsyncCleanup(resumed.job_manager.close_session_owned)
            descriptor = before["session_state"]["connection"]
            if variant == "credentialless":
                self.assertIsNone(descriptor["credential"])
                self.assertEqual(descriptor["model"], "beta")
                self.assertTrue(descriptor["stream"])
                self.assertEqual(descriptor["chat_url"], "https://local.example/v1/chat/completions")
                prepare("https://local.example/v1/chat/completions", "beta", protocols.OPENAI_CHAT, history=True)
            elif variant == "numeric":
                prepare("https://normalized-effective.example/v1/responses", "gpt-override", protocols.OPENAI_RESPONSES,
                        "selected-openai-secret", history=True)
            else:
                prepare("https://api.openai.com/v1/responses", "gpt-test", protocols.OPENAI_RESPONSES,
                        "selected-openai-secret", header="X-Custom-Key", history=True)

            phase["resume_answers"] = list(expected_answers)

            class Input(ScriptedInputSession):
                async def prompt(inner_self, prompt=None, history=None):
                    self.assertEqual(prompt, "Use this saved connection? [y/N]: ")
                    self.assertEqual(resumed.transcript_items, [])
                    self.assertIn(descriptor["chat_url"], output.getvalue())
                    if variant == "credentialless":
                        self.assertIn("Authentication: none\n", output.getvalue())
                        self.assertIn("Streaming: yes\n", output.getvalue())
                        self.assertNotIn("Credential: None", output.getvalue())
                    return "yes"

            inputs = Input(["continue selection", None])
            count = len(requests)
            with mock.patch.object(loki, "_DEFAULT_SESSION", resumed), \
                    mock.patch.object(loki, "CREDENTIALS", fresh_owner.inventory), \
                    mock.patch.object(terminal_frontend, "input_session", return_value=inputs), \
                    mock.patch.object(terminals, "open_terminal_stdin"), \
                    mock.patch.object(terminal_frontend, "restore_output_area_after_input"):
                self.assertEqual(await asyncio.wait_for(terminal_frontend.async_main([f"--resume={path}"]), 10), 0)
                self.assertEqual(loki.current_config().chat_provider.max_tokens, 1234)
                self.assertEqual(loki.current_config().stream, variant == "credentialless")
                self.assertEqual(loki.current_transcript()[:len(before["events"])], before["events"])
                loki.save_chat_log()
                await resumed.job_manager.close_session_owned()
            self.assertEqual(len(requests), count + 1)
            self.assertEqual(phase["queue"], [])
            final = json.loads(pathlib.Path(path).read_text())
            self.assertEqual(final["events"][:len(before["events"])], before["events"])
            self.assertEqual(formats.item_text(final["events"][-1]),
                             {"credentialless": "credentialless durable answer", "numeric": "override durable answer",
                              "custom": "openai durable answer"}[variant])
            self.assertEqual([formats.item_text(item) for item in final["events"]
                              if item.get("type") == "message" and item.get("role") == "user"],
                             expected_prompts + ["continue selection"])
            responses = [item for item in final["events"] if item.get("type") == "model_response"]
            self.assertEqual([formats.item_text(item) for item in responses if not formats.response_tool_calls(item)],
                             expected_answers + [phase["answer"]])
            self.assertEqual([(call["call_id"], call["name"], formats.tool_call_input(call))
                              for item in responses for call in formats.response_tool_calls(item)],
                             [("selection-read", "Read", {"file_path": str(source)})])
            expected_types = []
            for prompt in expected_prompts + ["continue selection"]:
                expected_types.extend(["message", "model_response"])
                if prompt == "read the selection witness":
                    expected_types.extend(["tool_result", "model_response"])
            self.assertEqual([item["type"] for item in final["events"]
                              if item["type"] != "instruction"
                              and (item["type"] != "message" or item.get("role") == "user")], expected_types)
            self.assertNotIn("calls", final)
            self.assertEqual(final["session_state"]["connection"]["chat_url"], descriptor["chat_url"])
            for secret in (value for name, value in values.items() if name.endswith("API_KEY")):
                self.assertNotIn(secret, pathlib.Path(path).read_text() + pin_path.read_text()
                                 + output.getvalue() + errors.getvalue())
            self.assertTrue(any(prompt.startswith("Model choice") for prompt in prompts))
            self.assertTrue(any(prompt.startswith("Provider choice") for prompt in prompts))
            self.assertTrue(all("filter WORDS" in prompt and "empty cancels" in prompt
                                for prompt in prompts if "choice" in prompt.lower()))
            rendered = output.getvalue()
            self.assertIn("\nUsable models:\n", rendered)
            self.assertIn("\nUsable providers:\n", rendered)
            self.assertLess(rendered.index("\nUsable models:\n"), rendered.index("\nUsable providers:\n"))
            if variant == "custom":
                self.assertGreaterEqual(rendered.count("\nUsable providers:\n"), 5)
            await asyncio.wait_for(asyncio.gather(*service_tasks), 10)
            self.assertEqual(service_errors, [])
            self.assertTrue(all(task.done() for task in service_tasks))
            self.assertEqual(len(wire_requests), len(requests) + 2)
            server.close()
            await asyncio.wait_for(server.wait_closed(), 5)
            self.assertFalse(server.is_serving())


class SubscriptionInferenceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_subscription_inference_save_and_resume(self):
        await self._workflow(stream=True)

    async def test_buffered_subscription_inference_save_and_resume(self):
        await self._workflow(stream=False)

    async def _workflow(self, *, stream, approve=True):
        from loki_agent import credential_storages
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            storage = credential_storages.JsonCredentialStorage(str(root / "credentials"))
            await storage.store_openai_login(authentications.OpenAITokenSet(
                access_token="leased-access-sentinel", refresh_token="durable-refresh-sentinel",
                id_token="identity-sentinel", account_id="account-sentinel",
                expires_at=10**12, last_refresh=10**12))
            supervisor = credential_supervisors.CredentialSupervisor(CredentialStore({}), storage)
            chat_id = "5a72cf91-7370-409b-8b39-a68cc21b649e"
            path = str(root / f"chat-{chat_id}.json")
            requests, catalog_requests = [], []
            output, errors = io.StringIO(), io.StringIO()
            initial = loki.Session(shell_cwd=directory, job_manager=loki.JobManager(str(root / "jobs")))
            initial.credential_authority = supervisor.broker
            initial.session_todos = [{"content": "inspect durable workflow", "status": "pending", "priority": "high"}]

            def catalog(updated):
                return {"models": [{
                    "slug": slug, "display_name": slug, "visibility": "list",
                    "input_modalities": ["text"], "use_responses_lite": True,
                    "supports_parallel_tool_calls": updated,
                    "supports_reasoning_summaries": True,
                    "default_reasoning_summary": "none",
                    "supported_reasoning_levels": [{"effort": "high" if updated else "low"}],
                    "default_reasoning_level": "high" if updated else "low",
                } for slug in ["old-model", "gpt-test"]]}

            def message(text):
                return {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": text}]}

            responses = [
                ("first-state-sentinel", [message("continuing")], False),
                ("ignored-state-sentinel", [{"type": "function_call", "call_id": "todo-call",
                                             "name": "TodoRead", "arguments": "{}"}], None),
                (None, [message("first durable answer")], None),
                ("resumed-state-sentinel", [message("resumed durable answer")], None),
                ("next-state-sentinel", [message("next durable answer")], None),
            ]

            def check_headers(headers):
                self.assertEqual(headers["Authorization"], "Bearer leased-access-sentinel")
                self.assertEqual({k.lower(): v for k, v in headers.items()}[
                    "chatgpt-account-id"], "account-sentinel")
                self.assertNotIn("durable-refresh-sentinel", json.dumps(headers))

            def inference(method, url, kwargs):
                self.assertEqual(method, "POST")
                self.assertEqual(url, authentications.OPENAI_CHATGPT_RESPONSES_URL)
                headers = dict(kwargs["headers_in"])
                if "prepare_attempt_headers" in kwargs:
                    kwargs["prepare_attempt_headers"](headers)
                check_headers(headers)
                self.assertEqual(headers[protocols.RESPONSES_LITE_HEADER], "true")
                for name in ["session-id", "thread-id", "x-client-request-id"]:
                    self.assertEqual(headers[name], chat_id)
                payload = json.loads(kwargs["body"])
                self.assertEqual(payload["model"], "gpt-test")
                self.assertEqual(payload["prompt_cache_key"], chat_id)
                requests.append((headers, payload))
                state, items, end_turn = responses.pop(0)
                response = {"id": f"response-{len(requests)}", "object": "response",
                            "status": "completed", "output": items}
                if end_turn is not None:
                    response["end_turn"] = end_turn
                response_headers = {loki.CODEX_TURN_STATE_HEADER: state} if state else {}
                return response_headers, response

            async def request(method, url, **kwargs):
                if method == "GET":
                    self.assertEqual(url, authentications.OPENAI_CHATGPT_MODELS_REQUEST_URL)
                    check_headers(kwargs["headers_in"])
                    catalog_requests.append(dict(kwargs["headers_in"]))
                    data = catalog(len(catalog_requests) > 1)
                    return http_client.HttpResponse(url, 200, "OK", {}, json.dumps(data).encode())
                headers, data = inference(method, url, kwargs)
                kwargs["on_response_headers"](200, headers)
                return http_client.HttpResponse(url, 200, "OK", headers, json.dumps(data).encode())

            @contextlib.asynccontextmanager
            async def streaming(method, url, **kwargs):
                headers, data = inference(method, url, kwargs)
                headers["content-type"] = "text/event-stream"

                async def chunks():
                    for item in data["output"]:
                        yield ("data: " + json.dumps({"type": "response.output_item.done", "item": item}) + "\n\n").encode()
                    yield ("data: " + json.dumps({
                        "type": "response.completed",
                        "response": {**data, "output": []}}) + "\n\n").encode()
                yield http_client.HttpStreamResponse(url, 200, "OK", headers, chunks())

            with mock.patch.object(loki, "_DEFAULT_SESSION", initial), \
                    mock.patch.object(loki, "CREDENTIALS", supervisor.inventory), \
                    mock.patch.object(http_client, "async_http_request", new=request), \
                    mock.patch.object(http_client, "async_http_stream", new=streaming), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                data = await modelsdev.fetch_openai_subscription_models(supervisor.broker)
                entries = modelsdev.add_openai_subscription_catalog(modelsdev.normalize_catalog({}), data)
                provider = entries["openai-subscription"]
                config = loki.config_from_modelsdev_selection(
                    "openai-subscription", provider,
                    provider["models"]["old-model"], supervisor.inventory)
                self.assertTrue(config.stream)
                loki.apply_runtime_config(config)
                loki.reinstall_provider(
                    model="gpt-test",
                    models_url=authentications.OPENAI_CHATGPT_MODELS_REQUEST_URL,
                    openai_request_profile=loki.config_from_modelsdev_selection(
                        "openai-subscription", provider,
                        provider["models"]["gpt-test"], supervisor.inventory).chat_provider.openai_request_profile)
                # Buffered is an explicit supported runtime variant; catalog defaults to SSE.
                loki.reinstall_provider(
                    stream=stream,
                    models_url=authentications.OPENAI_CHATGPT_MODELS_REQUEST_URL)
                ref = authentications.CredentialRef.openai_subscription()
                self.assertEqual(loki.current_config().auth_spec.credential, ref)
                self.assertEqual(loki.current_config().auth_spec.scheme, "openai-subscription")
                self.assertEqual(loki.current_config().chat_provider.models_url,
                                 authentications.OPENAI_CHATGPT_MODELS_REQUEST_URL)
                loki.new_chat_log(path)
                loki.current_session().session_todos = initial.session_todos = [
                    {"content": "inspect durable workflow", "status": "pending", "priority": "high"}]
                loki.current_transcript().append(formats.message_item("user", "run tools"))
                first = await asyncio.wait_for(loki.run_tool_loop_async(
                    loki.current_transcript(), max_loops=4, allowed={"TodoRead"}), 10)
                self.assertEqual(first, "first durable answer")
                results = [item for item in loki.current_transcript() if item.get("type") == "tool_result"]
                self.assertEqual([item["call_id"] for item in results], ["todo-call"])
                self.assertIn("inspect durable workflow", json.dumps(results))
                self.assertIn("todo-call", json.dumps(requests[2][1]))
                self.assertIn("inspect durable workflow", json.dumps(requests[2][1]))
                self.assertEqual([item.get("end_turn") for item in
                                  loki.current_transcript() if item.get("type") == "model_response"][0], False)
                self.assertIn("Todos:\n  1. [pending] (high) inspect durable workflow",
                              formats.item_text(results[0]))
                loki.mark_chat_log_dirty()
                loki.save_chat_log()
                before = pathlib.Path(path).read_bytes()
                first_blob = json.loads(before)
                self.assertNotIn("conversation_id", first_blob["session_state"])
                saved_results = [item for item in first_blob["events"]
                                 if item.get("type") == "tool_result"]
                self.assertEqual(
                    [(item["call_id"], formats.item_text(item)) for item in saved_results],
                    [("todo-call", "Todos:\n  1. [pending] (high) inspect durable workflow")])
                await initial.job_manager.close_session_owned()
                self.assertFalse(any(job.process and job.process.returncode is None
                                     for job in initial.job_manager.jobs.values()))

                reopened_storage = credential_storages.JsonCredentialStorage(storage.directory)
                reopened_owner = credential_supervisors.CredentialSupervisor(CredentialStore({}), reopened_storage)
                resumed = loki.Session(shell_cwd=directory, job_manager=loki.JobManager(str(root / "resumed-jobs")))
                resumed.credential_authority = reopened_owner.broker

                class Input(ScriptedInputSession):
                    async def prompt(inner_self, prompt=None, history=None):
                        self.assertEqual(prompt, "Use this saved connection? [y/N]: ")
                        self.assertEqual(len(catalog_requests), 2)
                        self.assertEqual(len(requests), 3)
                        return "yes" if approve else "no"

                inputs = Input(["continue saved conversation", "start next turn", None])
                with mock.patch.object(loki, "_DEFAULT_SESSION", resumed), \
                        mock.patch.object(terminal_frontend, "input_session", return_value=inputs), \
                        mock.patch.object(terminals, "open_terminal_stdin"), \
                        mock.patch.object(terminal_frontend, "restore_output_area_after_input"):
                    status = await asyncio.wait_for(terminal_frontend.async_main([f"--resume={path}"]), 10)
                    self.assertEqual(status, 0)
                    if not approve:
                        self.assertEqual(pathlib.Path(path).read_bytes(), before)
                        self.assertEqual(len(requests), 3)
                        return
                    self.assertEqual(resumed.conversation_id, chat_id)
                    self.assertTrue(loki.current_config().chat_provider.responses_lite)
                    self.assertTrue(loki.current_config().chat_provider.openai_request_profile.supports_parallel_tool_calls)
                    self.assertEqual(loki.current_config().reasoning_effort_profile.values, ["high"])
                    loki.save_chat_log()
                    await resumed.job_manager.close_session_owned()
                saved = json.loads(pathlib.Path(path).read_bytes())
                connection = saved["session_state"]["connection"]
                self.assertTrue(connection["openai_request_profile"]["supports_parallel_tool_calls"])
                self.assertEqual(connection["reasoning_effort_profile"]["options"][0]["value"], "high")
                self.assertEqual(len(catalog_requests), 2)
                self.assertEqual(responses, [])
                self.assertEqual([h.get(loki.CODEX_TURN_STATE_HEADER) for h, _ in requests],
                                 [None, "first-state-sentinel", "first-state-sentinel", None, None])
                for _, payload in requests[3:]:
                    self.assertIn("first durable answer", json.dumps(payload))
                    self.assertIn("todo-call", json.dumps(payload))
                    self.assertEqual(payload["reasoning"]["effort"], "high")
                durable = pathlib.Path(path).read_text()
                for text in ["first durable answer", "resumed durable answer", "next durable answer",
                             "todo-call", "inspect durable workflow"]:
                    self.assertIn(text, durable)
                for secret in ["leased-access-sentinel", "durable-refresh-sentinel", "identity-sentinel",
                               "first-state-sentinel", "ignored-state-sentinel", "resumed-state-sentinel"]:
                    self.assertNotIn(secret, durable + output.getvalue() + errors.getvalue())
                formats.validate_events(resumed.transcript_items)

    async def test_refused_refreshed_subscription_preserves_saved_bytes(self):
        await self._workflow(stream=True, approve=False)


class ExitStatusTests(unittest.TestCase):
    def test_executable_entry_point_propagates_headless_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            env = child_environment(
                HOME=directory,
                PATH=os.environ.get("PATH", ""),
                TERM="dumb",
                XDG_CONFIG_HOME=os.path.join(directory, "config"),
                XDG_STATE_HOME=os.path.join(directory, "state"),
            )
            workspace = os.path.join(directory, "workspace")
            os.makedirs(workspace)
            configure_container(env, workspace)
            result = subprocess.run(
                [entrypoint("loki"), "--headless"],
                cwd=workspace,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "Configuration error: API endpoint missing",
            result.stderr,
        )

    def test_cleanup_failure_changes_only_successful_status(self):
        old_credentials = loki.CREDENTIALS
        stderr = io.StringIO()
        try:
            for async_status, expected_status in ((0, 1), (2, 2)):
                with self.subTest(async_status=async_status), mock.patch(
                            "loki_agent.terminal_frontend.signal.signal"), mock.patch(
                                "loki_agent.terminal_frontend.signal.pthread_sigmask",
                                create=True,
                            ), mock.patch(
                                "loki_agent.terminal_frontend.initialize_terminal_overlay"
                            ), mock.patch(
                                "loki_agent.terminal_frontend.async_main",
                                new=mock.AsyncMock(
                                    return_value=async_status)), mock.patch(
                                    "loki_agent.terminal_frontend."
                                    "restore_terminal_overlay",
                                    side_effect=OSError("restore failed")
                                ), mock.patch.object(
                                    loki.current_session(),
                                    "chat_log_path",
                                    None,
                                ), contextlib.redirect_stderr(stderr):
                    status = asyncio.run(
                        terminal_frontend._run_frontend([]))

                self.assertEqual(status, expected_status)
        finally:
            loki.CREDENTIALS = old_credentials

        self.assertIn("Cleanup error: OSError: restore failed",
                      stderr.getvalue())


class StatusTextTests(unittest.TestCase):
    def test_local_status_groups_queue_counts_with_the_queue_command(self):
        activity = terminal_frontend.TerminalActivityStatus()
        with mock.patch.object(terminal_frontend, "current_cwd",
                               return_value="/home/dannym/src/loki"), \
                mock.patch.object(terminal_frontend, "current_agent_mode",
                                  return_value="normal"):
            local = terminal_frontend.status_text(activity).split("\n", 1)[1]
        self.assertEqual(
            local,
            "Local: CWD: /home/dannym/src/loki, turn: idle, mode: normal; "
            "/queue(texts: 0, images: 0), /pwd, /cd DIR, /ps, /image PATH, !foo, /quit")
        self.assertNotIn("queued prompts:", local)
        self.assertNotIn("queued images:", local)

    def test_status_bar_bolds_only_nonidle_and_nonzero_activity_values(self):
        for running, messages, images in (
                (True, 0, 0), (False, 2, 0), (False, 0, 3),
                (True, 2, 3), (False, 0, 0)):
            with self.subTest(running=running, messages=messages, images=images):
                activity = terminal_frontend.TerminalActivityStatus(
                    turn_running=running, queued_prompts=messages, queued_images=images)
                with mock.patch.object(terminal_frontend, "_terminal_activity", activity), \
                        mock.patch.object(terminal_frontend, "current_cwd",
                                          return_value="/status/cwd"), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    terminal_frontend._write_status_text()
                    plain = terminal_frontend.status_text(activity)
                rendered = output.getvalue()
                self.assertTrue(rendered.split("\n", 1)[1].startswith(
                    "Local: CWD: /status/cwd, turn: "))
                self.assertEqual(
                    rendered.replace("\033[1m", "").replace("\033[22m", ""), plain)
                for label, value, bold, suffix in [
                        ["turn", "running" if running else "idle", running, ","],
                        ["texts", str(messages), messages != 0, ","],
                        ["images", str(images), images != 0, ")"]]:
                    expected = f"\033[1m{value}\033[22m" if bold else value
                    self.assertIn(f"{label}: {expected}{suffix}", rendered)
                self.assertEqual(rendered.count("\033[1m"),
                                 sum((running, messages != 0, images != 0)))
                self.assertNotIn("\033[0m", rendered)
                self.assertNotIn("\033[", terminal_frontend.status_text(activity))

    def test_ps_bold_tracks_running_jobs_and_stops_at_the_command(self):
        finished = types.SimpleNamespace(
            status="exited", process=types.SimpleNamespace(returncode=0))
        for state, returncode, bold in [
                ["starting", None, False], ["running", None, True],
                ["running", 0, False], ["running", 42, False],
                ["stopping", None, False], ["failed", None, False],
                ["exited", 0, False], ["signaled", -15, False],
                ["timed_out", -15, False]]:
            with self.subTest(state=state, returncode=returncode):
                job = types.SimpleNamespace(
                    status=state, process=types.SimpleNamespace(returncode=returncode))
                session = loki.Session(job_manager=types.SimpleNamespace(
                    jobs={"1": finished, "2": job}))
                activity = terminal_frontend.TerminalActivityStatus()
                with mock.patch.object(loki, "_DEFAULT_SESSION", session), \
                        mock.patch.object(terminal_frontend, "_terminal_activity", activity), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    terminal_frontend._write_status_text()
                    plain = terminal_frontend.status_text()
                rendered = output.getvalue()
                expected = "\033[1m/ps\033[22m" if bold else "/ps"
                self.assertIn(f", {expected}, /image PATH", rendered)
                self.assertEqual(rendered.count("\033[1m"), int(bold))
                self.assertEqual(
                    rendered.replace("\033[1m", "").replace("\033[22m", ""), plain)
                self.assertNotIn("\033[0m", rendered)

    def test_ps_is_plain_without_a_job_manager_or_with_no_jobs(self):
        for manager in (None, types.SimpleNamespace(jobs={})):
            with self.subTest(manager=manager):
                session = loki.Session(job_manager=manager)
                with mock.patch.object(loki, "_DEFAULT_SESSION", session), \
                        mock.patch.object(terminal_frontend, "_terminal_activity",
                                          terminal_frontend.TerminalActivityStatus()), \
                        mock.patch.object(loki, "current_job_manager",
                                          side_effect=AssertionError("status must not create jobs")), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    terminal_frontend._write_status_text()
                self.assertIn(", /ps, /image PATH", output.getvalue())
                self.assertNotIn("\033[1m/ps", output.getvalue())

    def test_remote_side_advertises_status_with_and_without_effort(self):
        for effort in [None, "high"]:
            with self.subTest(effort=effort), mock.patch.object(
                    loki, "reasoning_effort_status_text", return_value=effort):
                text = terminal_frontend.status_text()
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    terminal_frontend._write_status_text()
            for rendered in (text, output.getvalue()):
                remote, local = rendered.split("\n", 1)
                self.assertIn("/status", remote)
                self.assertNotIn("/status", local)
                self.assertIn("/account", remote)
                self.assertNotIn("/account", local)
                self.assertNotIn("/effort", remote)
                self.assertIn("/thinking", remote)
                self.assertIn("/trace thinking", remote)

    def test_status_text_shows_none_when_no_model_is_selected(self):
        names = ["runtime_config"]
        old_values = save_loki_state(names)

        try:
            loki.current_session().runtime_config = None
            with mock.patch.object(loki, "reasoning_effort_status_text",
                                   return_value=None):
                text = terminal_frontend.status_text()
        finally:
            restore_loki_state(old_values)

        self.assertIn("Model: none, Context: unknown; /model", text)
        self.assertNotIn("Model: ;", text)

    def test_activity_status_redraws_only_for_changed_counts(self):
        activity = terminal_frontend.TerminalActivityStatus()

        with mock.patch(
                "loki_agent.terminal_frontend.terminals.redraw_status_bar"
        ) as redraw:
            activity.set_queued_prompts(2)
            activity.set_queued_prompts(2)
            activity.set_queued_images(1)
            activity.set_turn_running(True)
            activity.set_turn_running(True)

        self.assertTrue(activity.turn_running)
        self.assertEqual(activity.queued_prompts, 2)
        self.assertEqual(activity.queued_images, 1)
        self.assertEqual(redraw.call_count, 3)

    def test_status_text_includes_short_api_base_before_model_without_url_secrets(self):
        names = ["runtime_config", "shell_cwd"]
        old_values = save_loki_state(names)

        try:
            loki.current_session().shell_cwd = loki.STARTUP_CWD
            chat_provider = protocols.Provider(
                kind=protocols.OPENAI_CHAT,
                input_url="https://user:pass@example.test:8443/base/path/v1/chat/completions?token=secret#fragment",
                chat_url="https://example.test:8443/base/path/chat/completions",
                models_url=None,
                model_urls=[],
                headers={},
                max_tokens=4096,
            )
            loki.current_session().runtime_config = loki.RuntimeConfig(
                chat_provider=chat_provider,
                model="model-x"
            )
            # model is derived from runtime_config set above

            text = terminal_frontend.status_text(
                terminal_frontend.TerminalActivityStatus(
                    turn_running=True,
                    queued_prompts=2,
                    queued_images=1,
                ))
        finally:
            restore_loki_state(old_values)

        self.assertEqual(
            text,
            "Remote: API: example.test:8443/base/path, Model: model-x, "
            "Context: unknown; "
            "/model, /thinking, /trace thinking, /status, /account\n"
            f"Local: CWD: {loki.STARTUP_CWD}, turn: running, "
            f"mode: {loki.current_agent_mode()}; /queue(texts: 2, images: 1), "
            "/pwd, /cd DIR, /ps, /image PATH, !foo, /quit",
        )
        self.assertNotIn("user", text)
        self.assertNotIn("pass", text)
        self.assertNotIn("token", text)
        self.assertNotIn("secret", text)

    def test_status_text_escapes_controls_in_dynamic_fields(self):
        names = ["runtime_config", "shell_cwd"]
        old_values = save_loki_state(names)

        try:
            loki.current_session().runtime_config = None
            loki.current_session().shell_cwd = (
                "/tmp/unsafe\x1b[2J\nnext")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                terminal_frontend._write_status_text()
            displayed = output.getvalue()
        finally:
            restore_loki_state(old_values)

        self.assertIn(
            "Local: CWD: /tmp/unsafe^[[2J^Jnext, turn: ", displayed)
        self.assertNotIn("\x1b", displayed)
        self.assertEqual(displayed.count("\n"), 1)


@unittest.skipUnless(os.name == "posix", "requires a POSIX pseudo-terminal")
class TerminalJobStatusTtyTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_job_unbolds_ps_without_further_input(self):
        await self._exercise_job(background=True)

    async def test_foreground_job_repaints_ps_while_the_turn_is_running(self):
        await self._exercise_job(background=False)

    async def _exercise_job(self, background):
        import fcntl
        import pty
        import struct
        import termios
        from process_lifecycle_fixtures import ProcessResources

        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 240, 0, 0))

        def child_setup():
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        transport, _protocol = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader),
            os.fdopen(master, "rb", buffering=0))
        process = None
        resources = None
        with tempfile.TemporaryDirectory() as directory:
            environment = child_environment(
                PATH=os.environ.get("PATH", ""), HOME=directory,
                XDG_CONFIG_HOME=os.path.join(directory, "config"),
                XDG_STATE_HOME=os.path.join(directory, "state"),
                TERM="xterm-256color", LOKI_API_BASE="http://dummy.invalid/v1",
                LOKI_PROVIDER="dummy", LOKI_MODEL="dummy",
                LOKI_DUMMY_TOOL_CALL=json.dumps({
                    "name": "Bash", "arguments": {
                        "command": "sleep 2", "run_in_background": background}}))
            try:
                process = await asyncio.create_subprocess_exec(
                    entrypoint("loki"), "--dangerously-skip-permissions",
                    stdin=slave, stdout=slave, stderr=slave,
                    cwd=directory, env=environment, preexec_fn=child_setup)
                resources = ProcessResources(process)
                os.close(slave)
                slave = None

                async def read_until(marker):
                    captured = bytearray()
                    try:
                        async with asyncio.timeout(10):
                            while marker not in captured:
                                chunk = await reader.read(65536)
                                self.assertTrue(chunk, captured.decode("utf-8", errors="replace"))
                                captured.extend(chunk)
                    except TimeoutError:
                        self.fail("No terminal update for " + repr(marker)
                                  + "\n" + captured.decode("utf-8", errors="replace"))
                    return captured

                await read_until(b"User: ")
                os.write(master, b"run a job\n")
                started = await read_until(b"\033[1m/ps\033[22m")
                self.assertIn(b"turn: \033[1mrunning\033[22m", started)
                # No keystrokes or tool calls between these two observations.
                await read_until(b"/cd DIR, /ps, /image PATH")
                os.write(master, b"/quit\n")
                self.assertEqual(await asyncio.wait_for(process.wait(), 5), 0)
                resources.assert_released(self)
            finally:
                if resources is not None:
                    await resources.cleanup()
                if slave is not None:
                    os.close(slave)
                transport.close()


class TerminalOverlayLifecycleTests(unittest.TestCase):
    class RecordingTerminal:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return lambda *args: self.calls.append((name, *args))

    def test_initialize_clears_only_from_cursor_to_end(self):
        terminal = self.RecordingTerminal()

        terminal_frontend.initialize_terminal_overlay(terminal)

        self.assertEqual(terminal.calls, [
            ("hide_cursor",),
            ("enable_bracketed_paste_mode",),
            ("enable_origin_mode",),
            ("clear_to_end_of_screen",),
            ("reset_colors_and_flags",),
            ("set_clipping_region", *terminal_frontend.terminals.output_area),
            ("goto_position", 1, 1),
            ("flush",),
        ])
        self.assertNotIn(("clear_screen",), terminal.calls)

    def test_restore_resets_scroll_region_then_clears_to_end(self):
        terminal = self.RecordingTerminal()

        terminal_frontend.restore_terminal_overlay(terminal)

        self.assertEqual(terminal.calls, [
            ("disable_bracketed_paste_mode",),
            ("disable_clipping_regions",),
            ("disable_origin_mode",),
            ("reset_colors_and_flags",),
            ("goto_position", terminal_frontend.terminals.input_area[0], 1),
            ("clear_to_end_of_screen",),
            ("show_cursor",),
            ("force_end_synchronized_update",),
            ("flush",),
        ])
        self.assertNotIn(("clear_screen",), terminal.calls)


class ApiErrorFormattingTests(unittest.TestCase):
    def test_formatted_error_preserves_full_json_body(self):
        message = "x" * 5000
        error = loki.ApiError(
            "https://example.test/v1/chat/completions",
            429,
            "Too Many Requests",
            json.dumps({"error": {"message": message}}),
        )

        text = error.formatted()

        self.assertIn(message, text)
        self.assertNotIn("body truncated", text)

    def test_formatted_error_preserves_full_raw_body(self):
        body = "not-json:" + ("y" * 5000)
        error = loki.ApiError(
            "https://example.test/v1/chat/completions",
            500,
            "Internal Server Error",
            body,
        )

        text = error.formatted()

        self.assertIn(body, text)
        self.assertNotIn("body truncated", text)


class ResumeTranscriptRendererTests(unittest.TestCase):
    @staticmethod
    def _render(renderer, events):
        # Join a resume presentation the way a text front-end would; the
        # terminal front-end walks the same segments itself.
        return "\n\n".join(
            "".join(text for _kind, text in segments)
            for _block_kind, segments in renderer.presentation(events))

    def test_resume_renderer_shows_provider_notice_without_assistant_text(
            self):
        event = formats.model_response_event(
            formats.OPENAI_RESPONSES,
            [],
            protocol_data={
                "loki": {
                    "provider_notices": [
                        formats.TRUSTED_ACCESS_FOR_CYBER,
                    ],
                },
            },
        )

        text = self._render(
            savefiles.ResumeTranscriptRenderer(assistant_label="Assistant"),
            [event])

        self.assertIn("Trusted Access", text)
        self.assertNotIn("Assistant:", text)

    def test_resume_renderer_shows_provider_result_content_and_failures(self):
        items = [
            formats.model_response_event(
                formats.ANTHROPIC_MESSAGES,
                [{
                    "type": "provider_tool_result",
                    "call_id": "srvtoolu_1",
                    "content": [{
                        "type": "web_search_result",
                        "title": "Visible result",
                    }],
                }],
                status="completed",
            ),
            formats.model_response_event(
                formats.OPENAI_RESPONSES,
                [],
                status="failed",
                protocol_data={
                    formats.OPENAI_RESPONSES: {
                        "error": {"message": "visible failure"},
                    },
                },
            ),
        ]

        text = self._render(
            savefiles.ResumeTranscriptRenderer(assistant_label="Assistant"),
            items)

        self.assertIn("Provider tool result", text)
        self.assertIn("Visible result", text)
        self.assertNotIn("\nNone", text)
        self.assertIn("[Model response failed]", text)
        self.assertIn("visible failure", text)

    def test_terminal_resume_presentation_neutralizes_every_segment_kind(self):
        tool_name = "Read\x1b]0;name\x07\nnext"
        items = [
            formats.model_response_event(
                formats.OPENAI_CHAT,
                [formats.tool_call_item(
                    "call_1", tool_name,
                    {"path": "a\x1b]0;path\x07\nb"})],
                model="model\x1b]0;model\x07\nnext",
            ),
            formats.tool_result_item(
                "call_1",
                "first\x1b]0;result\x07\n"
                "second\u009b \u6a21\u578b",
                name=tool_name,
            ),
        ]
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            terminal_frontend._ResumeTranscriptPresenter(
                "Assistant").write(items)

        rendered = output.getvalue()
        self.assertNotIn("\x1b]0;", rendered)
        self.assertIn(
            "'Read\\x1b]0;name\\x07\\nnext'", rendered)
        self.assertIn("path: 'a\\x1b]0;path\\x07\\n'", rendered)
        self.assertIn("'b'", rendered)
        self.assertIn(
            "first^[]0;result^G\nsecond\\x9b \u6a21\u578b",
            rendered,
        )


class SessionResponsePersistenceTests(unittest.TestCase):
    def test_acp_chat_identity_uses_the_embedded_uuid(self):
        chat_id = "5a72cf91-7370-409b-8b39-a68cc21b649e"
        path = os.path.join(
            "/tmp", f"chat-loki-{chat_id}.json")

        self.assertEqual(
            loki.conversation_id_for_path(path),
            chat_id,
        )


class ProviderToolReplayWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_switch_repair_hooks_real_tools_save_resume_and_replay(self):
        from loki_agent import replays, tool_runtime
        import urllib.parse

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        with contextlib.nullcontext(temporary.name) as directory:
            root = pathlib.Path(directory)
            source = root / 'source.txt'
            source.write_text('marker original\n', encoding='utf-8')
            notes = root / 'notes.md'
            path = str(root / 'chat-switch.json')
            hook_log = root / 'hook-events.jsonl'
            hook_config = root / 'hooks.json'
            script = (
                'import json,pathlib,sys\n'
                'p=json.load(sys.stdin); i=p["invocation"]; a=i["effective_arguments"]\n'
                'if i["call_id"] in ["chat-search", "c-search"]:\n'
                ' assert i["adjustments"][1] == {"hook":"loki.input-repair",\n'
                '  "rule":"optional_null_omission", "path":["blocked_domains"],\n'
                '  "display_path":"$.blocked_domains", "operation":"remove"}\n'
                'with open(sys.argv[1], "a") as f: f.write(json.dumps([p["event"],i["call_id"],a])+"\\n")\n'
                'if p["event"]=="pre_tool_call":\n'
                ' assert isinstance(a["allowed_domains"],list)\n'
                ' assert "blocked_domains" not in a\n'
                ' if i["call_id"] in ["chat-search", "c-search"]:\n'
                '  assert i["original_arguments"]["blocked_domains"] is None\n'
                '  assert i["adjustments"][:2] == [\n'
                '   {"hook":"loki.input-repair", "rule":"json_encoded_array",\n'
                '    "path":["allowed_domains"], "display_path":"$.allowed_domains",\n'
                '    "operation":"replace", "value":["example.com"]},\n'
                '   {"hook":"loki.input-repair", "rule":"optional_null_omission",\n'
                '    "path":["blocked_domains"], "display_path":"$.blocked_domains",\n'
                '    "operation":"remove"}]\n'
                ' a["query"] += " external"\n'
                ' json.dump({"arguments":a},sys.stdout)\n'
                'else:\n'
                ' assert p["outcome"]["executed"] and p["outcome"]["ok"]\n'
                ' pathlib.Path(sys.argv[2]).write_text("marker post\\n")\n'
                ' json.dump({"changed_paths":[sys.argv[2]],"note":"post observed"},sys.stdout)\n'
            )
            hook_config.write_text(json.dumps({
                'pre_tool_call': [{'id': 'external.transform', 'tools': ['WebSearch'],
                                   'command': [sys.executable, '-c', script, str(hook_log), str(source)]}],
                'post_tool_call': [{'id': 'external.post', 'tools': ['WebSearch'],
                                    'command': [sys.executable, '-c', script, str(hook_log), str(source)]}],
            }), encoding='utf-8')
            store = CredentialStore({f'{name}_API_KEY': f'leased-{name}-secret'
                                     for name in ['A', 'B', 'CHAT', 'ANTHROPIC', 'C']})
            owner = credential_supervisors.CredentialSupervisor(store)
            session = loki.Session(shell_cwd=directory, job_manager=loki.JobManager(str(root / 'jobs')))
            session.credential_authority = owner.broker
            self.addAsyncCleanup(session.job_manager.close_session_owned)
            children, dispatched, events, search_queries, captured = [], [], [], [], []
            outcomes, originals, expected_replay, expected_terminal = {}, {}, [], []
            diagnostics = io.StringIO()
            real_dispatch = loki.dispatch_tool_async
            real_spawn = asyncio.create_subprocess_exec
            phase = {}
            source_key = loki._file_key(str(source))

            async def spawn(*args, **kwargs):
                if args[0] == 'test-search-program':
                    # Supply only this workflow's external search response;
                    # tool dispatch, result formatting and child cleanup stay real.
                    self.assertEqual(list(args[1:4]), [
                        '--color=never', '--with-filename', '--line-number'])
                    self.assertEqual(args[4], '--')
                    self.assertIn(args[5], ['marker', 'notes'])
                    args = [sys.executable, '-c',
                            'import pathlib,sys\n'
                            'path=sys.argv[2]\n'
                            'for number,line in enumerate(pathlib.Path(path).read_text().splitlines(),1):\n'
                            ' if sys.argv[1] in line: print(f"{path}:{number}:{line}")\n',
                            *args[5:]]
                child = await real_spawn(*args, **kwargs)
                children.append(child)

                async def reap():
                    if child.returncode is None:
                        child.kill()
                        await asyncio.wait_for(child.wait(), 5)
                self.addAsyncCleanup(reap)
                return child

            async def dispatch(name, args, allowed=None, extra_context=None):
                dispatched.append((name, copy.deepcopy(args)))
                return await real_dispatch(name, args, allowed=allowed, extra_context=extra_context)

            def hook_records():
                return [json.loads(line) for line in hook_log.read_text().splitlines()] if hook_log.exists() else []

            def transform(invocation):
                self.assertIsInstance(invocation.effective_arguments['allowed_domains'], list)
                self.assertEqual(invocation.effective_arguments['query'], 'loki external')
                self.assertEqual(hook_records()[-1][:2], ['pre_tool_call', invocation.call_id])
                arguments = copy.deepcopy(invocation.effective_arguments)
                arguments['query'] += ' custom'
                return tool_runtime.PreHookDecision(arguments=arguments)

            def gate(invocation):
                self.assertEqual(invocation.effective_arguments, {
                    'query': 'loki external custom', 'allowed_domains': ['example.com']})
                if invocation.call_id == 'chat-search':
                    self.assertIn(source_key, loki.file_state)
                return tool_runtime.PreHookDecision()

            def post(invocation, outcome):
                self.assertTrue(outcome.executed)
                self.assertTrue(outcome.ok)
                self.assertEqual(hook_records()[-1][:2], ['post_tool_call', invocation.call_id])
                self.assertEqual(source.read_text(), 'marker post\n')
                return tool_runtime.PostHookDecision()

            def provider_config(name, protocol):
                suffix = {protocols.OPENAI_RESPONSES: 'responses', protocols.OPENAI_CHAT: 'chat/completions',
                          protocols.ANTHROPIC_MESSAGES: 'messages'}[protocol]
                return loki.make_runtime_config(
                    f'https://{name.lower()}.example/v1/{suffix}', protocol,
                    model=f'model-{name}', provider_id=name, provider_name=name,
                    credential_ref=authentications.CredentialRef.environment(f'{name}_API_KEY'))

            def reply(call=None, *, private=False, text=None):
                name, protocol = phase['name'], phase['protocol']
                usage = {'input_tokens': 2, 'output_tokens': 1} if protocol != protocols.OPENAI_CHAT else {'total_tokens': 3}
                if protocol == protocols.OPENAI_RESPONSES:
                    items = []
                    if private:
                        items.append({'type': 'reasoning', 'id': f'private-{name}', 'summary': [],
                                      'encrypted_content': f'opaque-{name}'})
                    if text:
                        items.append({'type': 'message', 'role': 'assistant', 'content': [
                            {'type': 'output_text', 'text': text}]})
                    if call:
                        cid, tool, args = call
                        items.append({'type': 'function_call', 'call_id': cid, 'name': tool,
                                      'arguments': json.dumps(args)})
                    return {'object': 'response', 'status': 'completed', 'model': f'model-{name}',
                            'output': items, 'usage': usage}
                if protocol == protocols.OPENAI_CHAT:
                    message = {'role': 'assistant', 'content': phase['exact_content'] if call else text}
                    if call:
                        cid, tool, args = call
                        message['tool_calls'] = [{'id': cid, 'type': 'function', 'function': {
                            'name': tool, 'arguments': json.dumps(args)}}]
                    return {'object': 'chat.completion', 'id': 'chat-exact', 'model': f'model-{name}',
                            'choices': [{'index': 0, 'message': message,
                                         'finish_reason': 'tool_calls' if call else 'stop'}], 'usage': usage}
                content = []
                if private:
                    content.append({'type': 'thinking', 'thinking': 'private anthropic thought',
                                    'signature': 'anthropic-signature'})
                if text:
                    content.append({'type': 'text', 'text': text})
                if call:
                    cid, tool, args = call
                    content.append({'type': 'tool_use', 'id': cid, 'name': tool, 'input': args})
                return {'type': 'message', 'id': 'anthropic-origin', 'role': 'assistant',
                        'model': f'model-{name}', 'content': content,
                        'stop_reason': 'tool_use' if call else 'end_turn', 'usage': usage}

            async def request(method, url, **kwargs):
                self.assertEqual(method, 'POST')
                if url == loki.DUCKDUCKGO_HTML_SEARCH_URL:
                    query = urllib.parse.parse_qs(kwargs['body'].decode())['q'][0]
                    search_queries.append(query)
                    self.assertNotIn('Authorization', kwargs['headers_in'])
                    html = b'<a class="result__a" href="https://example.com/result">Sentinel result</a>'
                    return http_client.HttpResponse(url, 200, 'OK', {'content-type': 'text/html'}, html)
                config = loki.current_config()
                self.assertEqual(url, config.chat_provider.input_url)
                self.assertEqual(url, phase['url'])
                suffix = {protocols.OPENAI_RESPONSES: 'responses',
                          protocols.OPENAI_CHAT: 'chat/completions',
                          protocols.ANTHROPIC_MESSAGES: 'messages'}[phase['protocol']]
                self.assertEqual(url, f"https://{phase['name'].lower()}.example/v1/{suffix}")
                headers = kwargs['headers_in']
                if phase['protocol'] == protocols.ANTHROPIC_MESSAGES:
                    self.assertEqual(headers['x-api-key'], 'leased-ANTHROPIC-secret')
                    self.assertNotIn('Authorization', headers)
                else:
                    self.assertEqual(headers['Authorization'], f"Bearer leased-{phase['name']}-secret")
                    self.assertNotIn('x-api-key', headers)
                payload = json.loads(kwargs['body'])
                self.assertEqual(payload['model'], f"model-{phase['name']}")
                self.assertNotIn('"execution":', json.dumps(payload))
                if phase['protocol'] == protocols.OPENAI_RESPONSES:
                    wire_calls = [(i['call_id'], json.loads(i['arguments']))
                                  for i in payload['input'] if i.get('type') == 'function_call']
                elif phase['protocol'] == protocols.OPENAI_CHAT:
                    wire_calls = [(i['id'], json.loads(i['function']['arguments']))
                                  for m in payload['messages'] for i in m.get('tool_calls', [])]
                else:
                    wire_calls = [(i['id'], i['input']) for m in payload['messages']
                                  for i in m['content'] if i.get('type') == 'tool_use']
                for cid, arguments in wire_calls:
                    self.assertEqual(arguments, originals[cid])
                validators = {protocols.OPENAI_RESPONSES: '_assert_responses_payload_valid',
                              protocols.OPENAI_CHAT: '_assert_chat_payload_valid',
                              protocols.ANTHROPIC_MESSAGES: '_assert_anthropic_payload_valid'}
                getattr(PrimaryModelSwitchResumeTests, validators[phase['protocol']])(self, payload)
                captured.append((phase['name'], copy.deepcopy(payload)))
                # Opaque origin-only blocks must never reach a foreign provider.
                serialized = json.dumps(payload)
                for origin in ['A', 'B']:
                    if phase['name'] != origin:
                        self.assertNotIn(f'opaque-{origin}', serialized)
                        self.assertNotIn(f'private-{origin}', serialized)
                if phase['name'] != 'ANTHROPIC':
                    self.assertNotIn('anthropic-signature', serialized)
                    self.assertNotIn('private anthropic thought', serialized)
                data = phase['queue'].pop(0)
                return http_client.HttpResponse(url, 200, 'OK', {'content-type': 'application/json'}, json.dumps(data).encode())

            async def turn(name, protocol, calls, prompt, *, private=False, restored=False):
                phase.clear()
                phase.update(name=name, protocol=protocol, exact_content=[
                    {'type': 'text', 'text': 'first block'}, {'type': 'text', 'text': 'second block'}])
                config = loki.current_config() if restored else provider_config(name, protocol)
                if not restored:
                    loki.apply_runtime_config(config)
                loki.set_session_connection(loki.active_connection_descriptor())
                phase['url'] = config.chat_provider.input_url
                phase['queue'] = [reply(call, private=private and i == len(calls) - 1) for i, call in enumerate(calls)]
                answer = f'{prompt} done'
                phase['queue'].append(reply(text=answer))
                transcript = loki.current_transcript()
                transcript.append(formats.message_item('user', prompt))
                start = len(transcript)
                for cid, tool, args in calls:
                    originals[cid] = copy.deepcopy(args)
                allowed = {'Read', 'Write', 'Grep', 'WebSearch'}
                advertised = [spec['definition'] for name, spec in loki.TOOL_REGISTRY.items()
                              if name in allowed]
                loki._remember_session_toolset(advertised)
                result = await asyncio.wait_for(loki.run_tool_loop_async(
                    transcript, allowed=allowed,
                    max_loops=len(calls) + 2, on_event=events.append), 20)
                self.assertEqual(result, answer)
                self.assertEqual(phase['queue'], [])
                self.assertEqual([item['type'] for item in transcript[start:]],
                                 ['model_response', 'tool_result'] * len(calls) + ['model_response'])
                self.assertEqual(loki.current_toolsets(), [advertised])
                expected_replay.append(('user', prompt, ('message', 'user')))
                expected_terminal.append(('message', f'User: {prompt}'))
                for cid, tool, args in calls:
                    if protocol == protocols.OPENAI_CHAT:
                        expected_replay.append(('agent', 'first block\nsecond block', ('message', 'assistant')))
                        expected_terminal.append(('message', 'model-CHAT: first block\nsecond block'))
                    expected_replay.append(('tool', tool, cid))
                    rendered_args = '\n'.join(f'    {key}: {value!r}' for key, value in args.items())
                    expected_terminal.append(('tool_call', f'Tool call: {tool}\n{rendered_args}'))
                    event = next(item for item in transcript[start:] if item.get('type') == 'tool_result' and item['call_id'] == cid)
                    self.assertFalse(event.get('is_error', False), event)
                    outcomes[cid] = formats.item_text(event)
                    expected_replay.append(('tool', f'Tool result: {tool}\n{outcomes[cid]}', cid))
                    expected_terminal.append(('tool_result', f'Tool result: {tool}\n{outcomes[cid]}'))
                    call = next(call for item in transcript[start:] if item.get('type') == 'model_response'
                                for call in formats.response_tool_calls(item) if call['call_id'] == cid)
                    self.assertEqual(formats.tool_call_input(call), args)
                expected_replay.append(('agent', answer, ('message', 'assistant')))
                expected_terminal.append(('message', f'model-{name}: {answer}'))
                loki.mark_chat_log_dirty()
                loki.save_chat_log()
                formats.validate_events(transcript)
                return transcript[start:]

            def reopen():
                new_owner = credential_supervisors.CredentialSupervisor(store)
                new = loki.Session(shell_cwd=directory, job_manager=loki.JobManager(str(root / 'reopened-jobs')))
                new.credential_authority = new_owner.broker
                self.addAsyncCleanup(new.job_manager.close_session_owned)
                return new

            with mock.patch.object(loki, '_DEFAULT_SESSION', session), \
                    mock.patch.object(loki, 'CREDENTIALS', owner.inventory), \
                    mock.patch.object(loki, 'TOOL_HOOK_PIPELINE', tool_runtime.ToolHookPipeline()), \
                    mock.patch.object(loki, 'file_state', {}), \
                    mock.patch.object(loki, 'dispatch_tool_async', new=dispatch), \
                    mock.patch.object(asyncio, 'create_subprocess_exec', new=spawn), \
                    mock.patch.object(loki.executables, 'RIPGREP', 'test-search-program'), \
                    mock.patch.object(http_client, 'async_http_request', new=request), \
                    contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
                loki.apply_runtime_config(provider_config('A', protocols.OPENAI_RESPONSES))
                loki.new_chat_log(path)
                malformed_path = os.path.join(directory, '[source.txt](http://source.txt)')
                await turn('A', protocols.OPENAI_RESPONSES, [
                    ('a-read', 'Read', {'file_path': malformed_path}),
                    ('a-search', 'WebSearch', {'query': 'loki', 'allowed_domains': '["example.com"]', 'blocked_domains': None}),
                ], 'read and search', private=True)
                self.assertEqual(dispatched[0], ('Read', {'file_path': str(source)}))
                self.assertIn(source_key, loki.file_state)
                self.assertEqual(loki.configure_tool_hook_pipeline({'LOKI_HOOKS': str(hook_config)}), str(hook_config))
                pipeline = loki.TOOL_HOOK_PIPELINE
                pipeline.add_pre('custom.transform', transform, matcher=lambda name: name == 'WebSearch')
                pipeline.add_gate('observing.gate', gate, matcher=lambda name: name == 'WebSearch')
                pipeline.add_post('observing.post', post, matcher=lambda name: name == 'WebSearch')
                self.assertEqual(source.read_text(), 'marker original\n')
                await turn('B', protocols.OPENAI_RESPONSES, [
                    ('b-grep', 'Grep', {'pattern': 'marker', 'path': str(source), 'output_mode': 'content'}),
                    ('b-write', 'Write', {'file_path': os.path.join(directory, '[notes.md](http://notes.md)'), 'content': '[notes.md](http://notes.md)'}),
                ], 'switch and write', private=True)
                self.assertEqual(notes.read_text(), '[notes.md](http://notes.md)')
                self.assertIn('marker original', outcomes['b-grep'])
                self.assertIn('opaque-A', json.dumps(captured[2][1]))
                self.assertIn('opaque-B', json.dumps(captured[-1][1]))
                saved = json.loads(pathlib.Path(path).read_text())
                self.assertNotIn('calls', saved)
                self.assertNotIn('"start":', json.dumps(saved))
                self.assertEqual(saved['toolsets'], [[
                    spec['definition'] for name, spec in loki.TOOL_REGISTRY.items()
                    if name in {'Read', 'Write', 'Grep', 'WebSearch'}]])
                first_toolsets = copy.deepcopy(saved['toolsets'])
                await session.job_manager.close_session_owned()
                resumed = reopen()
                with mock.patch.object(loki, '_DEFAULT_SESSION', resumed):
                    loki.file_state.clear()
                    loki.load_chat_log(path)
                    self.assertEqual(loki.current_toolsets(), first_toolsets)
                    descriptor = loki.connection_from_session_state(loki.current_state())
                    self.assertEqual((descriptor.provider_id, descriptor.model), ('B', 'model-B'))
                    loki.apply_runtime_config(loki.config_from_connection_descriptor(descriptor, owner.inventory))
                    await turn('B', protocols.OPENAI_RESPONSES, [
                        ('b-resumed-read', 'Read', {'file_path': str(source)}),
                        ('b-resumed-grep', 'Grep', {'pattern': 'notes', 'path': str(notes), 'output_mode': 'content'})], 'resume B', restored=True)
                    await turn('CHAT', protocols.OPENAI_CHAT, [
                        ('chat-search', 'WebSearch', {'query': 'loki', 'allowed_domains': '["example.com"]', 'blocked_domains': None})], 'chat exact')
                    self.assertNotIn(source_key, loki.file_state)
                    self.assertEqual(source.read_text(), 'marker post\n')
                    chat_followup = captured[-1][1]
                    native = next(m for m in chat_followup['messages'] if m.get('tool_calls', [{}])[0].get('id') == 'chat-search')
                    self.assertEqual(native['content'], phase['exact_content'])
                    self.assertEqual(native['tool_calls'], [{'id': 'chat-search', 'type': 'function', 'function': {
                        'name': 'WebSearch', 'arguments': json.dumps(originals['chat-search'])}}])
                    await turn('ANTHROPIC', protocols.ANTHROPIC_MESSAGES, [
                        ('anthropic-search', 'WebSearch', {'query': 'loki', 'allowed_domains': 'example.com'})], 'anthropic turn', private=True)
                    self.assertIn('anthropic-signature', json.dumps(captured[-1][1]))
                    await turn('C', protocols.OPENAI_RESPONSES, [
                        ('c-search', 'WebSearch', {'query': 'loki', 'allowed_domains': '["example.com"]', 'blocked_domains': None})], 'foreign replay')
                    loki.apply_runtime_config(provider_config('ANTHROPIC', protocols.ANTHROPIC_MESSAGES))
                    loki.set_session_connection(loki.active_connection_descriptor())
                    loki.save_chat_log()
                    await resumed.job_manager.close_session_owned()
                final_session = reopen()
                with mock.patch.object(loki, '_DEFAULT_SESSION', final_session):
                    loki.file_state.clear()
                    loki.load_chat_log(path)
                    descriptor = loki.connection_from_session_state(loki.current_state())
                    self.assertEqual(descriptor.provider_id, 'ANTHROPIC')
                    loki.apply_runtime_config(loki.config_from_connection_descriptor(descriptor, owner.inventory))
                    hooks_before = hook_records()
                    child_count = len(children)
                    self.assertIsNone(loki.configure_tool_hook_pipeline({'LOKI_HOOKS': 'off'}))
                    self.assertFalse(loki.TOOL_HOOK_PIPELINE.has_custom_hooks)
                    await turn('ANTHROPIC', protocols.ANTHROPIC_MESSAGES, [
                        ('off-search', 'WebSearch', {'query': 'loki', 'allowed_domains': 'example.com'})], 'final resume', restored=True)
                    self.assertEqual(hook_records(), hooks_before)
                    self.assertEqual(len(children), child_count)
                    self.assertEqual(search_queries, ['loki'] + ['loki external custom'] * 3 + ['loki'])
                    final_blob = json.loads(pathlib.Path(path).read_text())
                    persisted = final_blob['events']
                    for cid, original in originals.items():
                        call = next(call for event in persisted if event.get('type') == 'model_response'
                                    for call in formats.response_tool_calls(event) if call['call_id'] == cid)
                        self.assertEqual(formats.tool_call_input(call), original)
                        result = next(item for item in persisted if item.get('type') == 'tool_result' and item['call_id'] == cid)
                        self.assertEqual(formats.item_text(result), outcomes[cid])
                    # Assert execution effects independently before using their text as replay goldens.
                    self.assertEqual(outcomes['a-read'].split('\n\n')[-1], '1\tmarker original')
                    self.assertEqual(outcomes['b-resumed-read'], '1\tmarker original')
                    for cid, file, line in [('b-grep', source, 'marker original'),
                                            ('b-resumed-grep', notes, '[notes.md](http://notes.md)')]:
                        self.assertEqual(outcomes[cid].split('[results]\n')[1], f'{file}:1:{line}')
                    for cid in ['a-search', 'chat-search', 'anthropic-search', 'c-search', 'off-search']:
                        query = 'loki external custom' if cid in ['chat-search', 'anthropic-search', 'c-search'] else 'loki'
                        self.assertEqual(
                            outcomes[cid].split('\n\n')[-1],
                            f"WebSearch results for {query!r} (1 results):\n1. Sentinel result\n   https://example.com/result")
                    self.assertEqual(outcomes['b-write'].split('\n\n')[-1], f'Successfully wrote to {notes}')
                    self.assertEqual(dispatched, [
                        ('Read', {'file_path': str(source)}),
                        ('WebSearch', {'query': 'loki', 'allowed_domains': ['example.com']}),
                        ('Grep', {'pattern': 'marker', 'path': str(source), 'output_mode': 'content'}),
                        ('Write', {'file_path': str(notes), 'content': '[notes.md](http://notes.md)'}),
                        ('Read', {'file_path': str(source)}),
                        ('Grep', {'pattern': 'notes', 'path': str(notes), 'output_mode': 'content'}),
                        ('WebSearch', {'query': 'loki external custom', 'allowed_domains': ['example.com']}),
                        ('WebSearch', {'query': 'loki external custom', 'allowed_domains': ['example.com']}),
                        ('WebSearch', {'query': 'loki external custom', 'allowed_domains': ['example.com']}),
                        ('WebSearch', {'query': 'loki', 'allowed_domains': ['example.com']}),
                    ])
                    self.assertEqual(final_blob['toolsets'], first_toolsets)
                    response_records = [i for i in persisted if i.get('type') == 'model_response']
                    self.assertEqual([i['provider'] for i in response_records],
                                     ['A'] * 3 + ['B'] * 6 + ['CHAT'] * 2 + ['ANTHROPIC'] * 2 + ['C'] * 2 + ['ANTHROPIC'] * 2)
                    for record in response_records:
                        name = record['provider']
                        protocol = {'CHAT': protocols.OPENAI_CHAT,
                                    'ANTHROPIC': protocols.ANTHROPIC_MESSAGES}.get(name, protocols.OPENAI_RESPONSES)
                        suffix = {protocols.OPENAI_RESPONSES: 'responses',
                                  protocols.OPENAI_CHAT: 'chat/completions',
                                  protocols.ANTHROPIC_MESSAGES: 'messages'}[protocol]
                        self.assertEqual(record['protocol'], protocol)
                        self.assertEqual(record['endpoint'], f'https://{name.lower()}.example/v1/{suffix}')
                        self.assertEqual(record['requested_model'], f'model-{name}')
                        self.assertEqual(record['status'], 'completed')
                        self.assertEqual(record['model'], f'model-{name}')
                        self.assertEqual(record['usage'], {'total_tokens': 3} if name == 'CHAT' else
                                         {'input_tokens': 2, 'output_tokens': 1})
                    for cid in ['chat-search', 'anthropic-search', 'c-search']:
                        metadata = next(i['execution'] for i in persisted if i.get('call_id') == cid and i.get('type') == 'tool_result')
                        self.assertEqual([(r['hook'], r['phase'], r['status']) for r in metadata['hooks']], [
                            ('external.transform', 'pre_tool_call', 'ok'),
                            ('custom.transform', 'pre_tool_call', 'ok'),
                            ('observing.gate', 'pre_tool_gate', 'ok'),
                            ('external.post', 'post_tool_call', 'ok'),
                            ('observing.post', 'post_tool_call', 'ok'),
                        ])
                        self.assertEqual(metadata['changed_paths'], [str(source)])
                    event_pairs = [(event['type'], event['call_id']) for event in events
                                   if event['type'] in ['tool_input_repaired', 'tool_call']]
                    repaired_ids = {'a-read', 'a-search', 'b-write', 'chat-search', 'anthropic-search', 'c-search', 'off-search'}
                    self.assertEqual(event_pairs, [
                        (event_type, cid) for cid in originals
                        for event_type in (['tool_input_repaired', 'tool_call'] if cid in repaired_ids else ['tool_call'])])
                    self.assertTrue(all(event['cwd'] == directory for event in events if event['type'] == 'tool_call'))
                    repairs = {item['call_id']: item.get('execution', {}).get('adjustments', [])
                               for item in persisted if item.get('type') == 'tool_result'}
                    self.assertEqual([r['rule'] for r in repairs['a-read']], ['path_markdown_autolink'])
                    self.assertEqual([r['rule'] for r in repairs['a-search'][:2]], ['json_encoded_array', 'optional_null_omission'])
                    for cid in ['chat-search', 'anthropic-search', 'c-search', 'off-search']:
                        self.assertEqual(repairs[cid][0]['rule'],
                                         'json_encoded_array' if cid in ['chat-search', 'c-search']
                                         else 'bare_string_array')
                        self.assertEqual(repairs[cid][0]['value'], ['example.com'])
                    for cid in ['a-search', 'chat-search', 'c-search']:
                        self.assertEqual(repairs[cid][1], {
                            'hook': 'loki.input-repair', 'rule': 'optional_null_omission',
                            'path': ['blocked_domains'], 'display_path': '$.blocked_domains',
                            'operation': 'remove'})
                    for cid in ['chat-search', 'anthropic-search', 'c-search']:
                        self.assertEqual([r['hook'] for r in repairs[cid]][-2:], ['external.transform', 'custom.transform'])
                    self.assertEqual([row[:2] for row in hook_records()], [
                        [event, cid] for cid in ['chat-search', 'anthropic-search', 'c-search']
                        for event in ['pre_tool_call', 'post_tool_call']])
                    self.assertEqual(replays.classify_transcript(persisted), expected_replay)
                    presentation = savefiles.ResumeTranscriptRenderer('current').presentation(persisted)
                    self.assertEqual([(kind, ''.join(text for _, text in segments))
                                      for kind, segments in presentation], expected_terminal)
                    terminal_text = '\n\n'.join(''.join(text for _, text in segments) for _, segments in presentation)
                    for name, prompt in [('A', 'read and search'), ('B', 'switch and write'), ('B', 'resume B'),
                                         ('CHAT', 'chat exact'), ('ANTHROPIC', 'anthropic turn'), ('C', 'foreign replay'),
                                         ('ANTHROPIC', 'final resume')]:
                        self.assertIn(f'model-{name}: {prompt} done', terminal_text)
                    self.assertNotIn('private anthropic thought', terminal_text)
                    self.assertNotIn('opaque-A', terminal_text)
                    durable = pathlib.Path(path).read_text()
                    for name in ['A', 'B', 'CHAT', 'ANTHROPIC', 'C']:
                        self.assertNotIn(f'leased-{name}-secret', durable + diagnostics.getvalue())
                    self.assertEqual(len(children), 8)
                    self.assertTrue(all(child.returncode == 0 for child in children))
                    await final_session.job_manager.close_session_owned()


class PrimaryModelSwitchResumeTests(unittest.TestCase):
    _GLOBAL_NAMES = [
        "runtime_config", "chat_log_path", "session_state",
        "chat_log_dirty", "transcript_items", "session_todos",
        "session_toolsets", "shell_cwd", "previous_shell_cwd",
    ]

    @contextlib.contextmanager
    def _isolated_runtime(self):
        session = loki.current_session()
        old_values = {}
        for name in self._GLOBAL_NAMES:
            value = getattr(session, name, _MISSING)
            old_values[name] = _MISSING if value is _MISSING \
                else copy.deepcopy(value)
        try:
            yield
        finally:
            restore_loki_state(old_values)

    @staticmethod
    def _request_sequence(responses, captured):
        queued = list(responses)

        async def request(
                method, request_url, payload=None, request_headers=None,
                report_errors=False, show_timing=False,
                codex_turn_state=None):
            if not queued:
                raise AssertionError("unexpected provider request")
            if method != "POST":
                raise AssertionError(
                    f"unexpected provider method {method!r}")
            captured.append({
                "url": request_url,
                "payload": copy.deepcopy(payload),
                "headers": copy.deepcopy(request_headers),
            })
            return protocols.ProviderResponse(
                copy.deepcopy(queued.pop(0)))

        def assert_exhausted():
            if queued:
                raise AssertionError(
                    f"{len(queued)} provider responses were not consumed")

        request.assert_exhausted = assert_exhausted
        return request

    def _assert_responses_payload_valid(self, payload):
        self.assertIsInstance(payload.get("model"), str)
        self.assertIsInstance(payload.get("input"), list)
        pending = set()
        allowed_types = {
            "message",
            "function_call", "custom_tool_call",
            "function_call_output", "custom_tool_call_output",
            "reasoning",
        } | set(formats._RESPONSES_PROVIDER_TYPES)
        for item in payload["input"]:
            self.assertIsInstance(item, dict)
            item_type = item.get("type")
            self.assertIn(
                item_type, allowed_types,
                f"invalid Responses input item type: {item_type!r}",
            )
            if item_type == "message":
                self.assertIn(
                    item.get("role"),
                    ["system", "developer", "user", "assistant"],
                )
                self.assertIsInstance(item.get("content"), list)
                for block in item["content"]:
                    self.assertIsInstance(block, dict)
                    self.assertIsInstance(block.get("type"), str)
            elif item_type in [
                    "function_call", "custom_tool_call"]:
                call_id = item.get("call_id")
                self.assertIsInstance(call_id, str)
                self.assertIsInstance(item.get("name"), str)
                argument_field = (
                    "input" if item_type == "custom_tool_call"
                    else "arguments")
                self.assertIsInstance(
                    item.get(argument_field), str)
                self.assertNotIn(call_id, pending)
                pending.add(call_id)
            elif item_type in [
                    "function_call_output", "custom_tool_call_output"]:
                call_id = item.get("call_id")
                self.assertIn(
                    call_id, pending,
                    f"Responses output has no preceding call: {call_id!r}",
                )
                pending.remove(call_id)
        self.assertEqual(
            pending, set(),
            f"Responses payload contains dangling calls: {pending!r}",
        )

    def _assert_chat_payload_valid(self, payload):
        self.assertIsInstance(payload.get("model"), str)
        self.assertIsInstance(payload.get("messages"), list)
        pending = set()
        for message in payload["messages"]:
            self.assertIn(
                message.get("role"),
                ["system", "developer", "user", "assistant", "tool",
                 "function"],
            )
            content = message.get("content")
            self.assertTrue(
                content is None
                or isinstance(content, (str, list)),
                f"invalid Chat message content: {content!r}",
            )
            if isinstance(content, list):
                for block in content:
                    self.assertIsInstance(block, dict)
                    self.assertIsInstance(block.get("type"), str)
            for call in message.get("tool_calls", []):
                call_id = call.get("id")
                self.assertIsInstance(call_id, str)
                self.assertEqual(call.get("type"), "function")
                function = call.get("function")
                self.assertIsInstance(function, dict)
                self.assertIsInstance(function.get("name"), str)
                self.assertIsInstance(
                    function.get("arguments"), str)
                self.assertNotIn(call_id, pending)
                pending.add(call_id)
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                self.assertIn(
                    call_id, pending,
                    f"Chat tool result has no preceding call: {call_id!r}",
                )
                pending.remove(call_id)
        self.assertEqual(
            pending, set(),
            f"Chat payload contains dangling calls: {pending!r}",
        )

    def _assert_anthropic_payload_valid(self, payload):
        self.assertIsInstance(payload.get("model"), str)
        self.assertIsInstance(payload.get("messages"), list)
        if "system" in payload:
            self.assertIsInstance(payload["system"], list)
            for block in payload["system"]:
                self.assertIsInstance(block, dict)
                self.assertIsInstance(block.get("type"), str)
        pending = set()
        provider_pending = set()
        provider_result_types = {
            "web_search_tool_result",
            "web_fetch_tool_result",
            "code_execution_tool_result",
            "bash_code_execution_tool_result",
            "text_editor_code_execution_tool_result",
            "tool_search_tool_result",
            "mcp_tool_result",
        }
        for message in payload["messages"]:
            role = message.get("role")
            self.assertIn(role, ["user", "assistant"])
            content = message.get("content")
            self.assertIsInstance(content, list)
            for block in content:
                self.assertIsInstance(block, dict)
                block_type = block.get("type")
                self.assertIsInstance(block_type, str)
                if block_type == "tool_use":
                    call_id = block.get("id")
                    self.assertIsInstance(call_id, str)
                    self.assertIsInstance(block.get("name"), str)
                    self.assertIsInstance(block.get("input"), dict)
                    self.assertNotIn(call_id, pending)
                    pending.add(call_id)
                elif block_type == "tool_result":
                    call_id = block.get("tool_use_id")
                    self.assertIn(
                        call_id, pending,
                        "Anthropic tool result has no preceding tool use",
                    )
                    pending.remove(call_id)
                elif block_type in [
                        "server_tool_use", "mcp_tool_use"]:
                    call_id = block.get("id")
                    self.assertIsInstance(call_id, str)
                    self.assertIsInstance(block.get("name"), str)
                    self.assertIsInstance(block.get("input"), dict)
                    self.assertNotIn(call_id, provider_pending)
                    provider_pending.add(call_id)
                elif block_type in provider_result_types:
                    call_id = block.get("tool_use_id")
                    self.assertIn(
                        call_id, provider_pending,
                        "Anthropic provider result has no preceding "
                        "server tool use",
                    )
                    provider_pending.remove(call_id)
        self.assertEqual(
            pending, set(),
            f"Anthropic payload contains dangling calls: {pending!r}",
        )
        self.assertEqual(
            provider_pending, set(),
            "Anthropic payload contains dangling server calls: "
            f"{provider_pending!r}",
        )

    def test_incomplete_tool_call_and_media_survive_resume_and_switch(self):
        with self._isolated_runtime(), tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-incomplete.json")
            responses_config = loki.make_runtime_config(
                "https://responses.example/v1/responses",
                protocols.OPENAI_RESPONSES,
                model="responses-model",
                provider_id="responses-provider",
                provider_name="Responses Provider",
                credential_ref=authentications.CredentialRef.environment(
                    "RESPONSES_API_KEY"),
            )
            loki.apply_runtime_config(responses_config)
            loki.new_chat_log(path)
            loki.current_transcript().append(
                formats.message_item("user", [
                    formats.text_block("inspect this image"),
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AAAA",
                        },
                    },
                ]))

            captured_incomplete = []
            incomplete_request = self._request_sequence([{
                "object": "response",
                "status": "incomplete",
                "incomplete_details": {
                    "reason": "max_output_tokens",
                },
                "output": [{
                    "type": "function_call",
                    "id": "function_incomplete",
                    "status": "incomplete",
                    "call_id": "call_incomplete",
                    "name": "Read",
                    "arguments": '{"file_path":',
                }],
            }], captured_incomplete)
            dispatches = []

            async def forbidden_dispatch(
                    name, args, allowed=None, extra_context=None):
                dispatches.append((name, copy.deepcopy(args)))
                raise AssertionError(
                    "incomplete tool call must not be dispatched")

            with mock.patch(
                    "loki_agent.loki.async_provider_request",
                    new=incomplete_request), mock.patch(
                        "loki_agent.loki.dispatch_tool_async",
                        new=forbidden_dispatch):
                result = asyncio.run(loki.run_tool_loop_async(
                    loki.current_transcript(),
                    allowed={"Read"},
                ))

            incomplete_request.assert_exhausted()
            self.assertEqual(result, "")
            self.assertEqual(dispatches, [])
            self.assertEqual(
                formats.pending_tool_calls(loki.current_transcript()), [])
            self.assertEqual(
                [event["type"] for event in loki.current_transcript()[-2:]],
                ["model_response", "tool_result"],
            )
            self.assertEqual(
                loki.current_transcript()[-2]["status"], "incomplete")
            self.assertTrue(loki.current_transcript()[-1]["is_error"])
            self.assertIn(
                "provider response was incomplete",
                formats.item_text(loki.current_transcript()[-1]),
            )
            self.assertEqual(len(captured_incomplete), 1)
            self.assertEqual(
                captured_incomplete[0]["url"],
                responses_config.chat_provider.chat_url,
            )
            self._assert_responses_payload_valid(
                captured_incomplete[0]["payload"])
            initial_payload = json.dumps(
                captured_incomplete[0]["payload"])
            self.assertIn(
                "data:image/png;base64,AAAA", initial_payload)
            loki.mark_chat_log_dirty()
            loki.save_chat_log()

            loki.current_session().transcript_items = []
            with contextlib.redirect_stdout(io.StringIO()):
                loki.load_chat_log(path)
            self.assertEqual(
                loki.current_transcript()[-2]["status"], "incomplete")
            self.assertEqual(
                formats.pending_tool_calls(loki.current_transcript()), [])

            chat_config = loki.make_runtime_config(
                "https://chat.example/v1/chat/completions",
                protocols.OPENAI_CHAT,
                model="chat-model",
                provider_id="chat-provider",
                provider_name="Chat Provider",
                credential_ref=authentications.CredentialRef.environment(
                    "CHAT_API_KEY"),
            )
            loki.apply_runtime_config(chat_config)
            loki.set_session_connection(
                loki.active_connection_descriptor())
            loki.current_transcript().append(
                formats.message_item(
                    "user", "recover after the incomplete call"))

            captured_chat = []
            chat_request = self._request_sequence([{
                "id": "chat_response",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Recovered on Chat.",
                    },
                    "finish_reason": "stop",
                }],
            }], captured_chat)
            with mock.patch(
                    "loki_agent.loki.async_provider_request",
                    new=chat_request):
                recovered = asyncio.run(loki.run_tool_loop_async(
                    loki.current_transcript(),
                    allowed={"Read"},
                ))

            chat_request.assert_exhausted()
            self.assertEqual(recovered, "Recovered on Chat.")
            self.assertEqual(len(captured_chat), 1)
            self.assertEqual(
                captured_chat[0]["url"],
                chat_config.chat_provider.chat_url,
            )
            chat_payload = captured_chat[0]["payload"]
            self._assert_chat_payload_valid(chat_payload)
            serialized_chat = json.dumps(chat_payload)
            self.assertIn(
                "data:image/png;base64,AAAA", serialized_chat)
            self.assertIn("call_incomplete", serialized_chat)
            self.assertIn(
                "provider response was incomplete", serialized_chat)
            self.assertIn(
                "recover after the incomplete call", serialized_chat)

            loki.mark_chat_log_dirty()
            loki.save_chat_log()
            loki.current_session().transcript_items = []
            with contextlib.redirect_stdout(io.StringIO()):
                loki.load_chat_log(path)
            descriptor = loki.connection_from_session_state(
                loki.current_state())
            self.assertEqual(descriptor.provider_id, "chat-provider")
            self.assertEqual(
                formats.pending_tool_calls(loki.current_transcript()), [])
            formats.validate_events(loki.current_transcript())

    def test_anthropic_server_tool_replays_at_origin_and_sanitizes_on_switch(
            self):
        with self._isolated_runtime(), tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-server-tool.json")
            provider_a = loki.make_runtime_config(
                "https://anthropic-a.example/v1/messages",
                protocols.ANTHROPIC_MESSAGES,
                model="claude-a",
                provider_id="anthropic-a",
                provider_name="Anthropic A",
                credential_ref=authentications.CredentialRef.environment(
                    "ANTHROPIC_A_API_KEY"),
            )
            loki.apply_runtime_config(provider_a)
            loki.new_chat_log(path)
            loki.current_transcript().append(
                formats.message_item("user", "search for the result"))

            server_content = [
                {
                    "type": "thinking",
                    "thinking": "provider A private thought",
                    "signature": "provider-a-signature",
                },
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_a",
                    "name": "web_search",
                    "input": {"query": "portable result"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_a",
                    "content": [{
                        "type": "web_search_result",
                        "title": "Portable result",
                        "url": "https://example.test/result",
                        "encrypted_content": "provider-a-encrypted",
                    }],
                },
                {
                    "type": "text",
                    "text": "Search completed.",
                    "citations": [{
                        "type": "web_search_result_location",
                        "url": "https://example.test/result",
                        "title": "Portable result",
                        "encrypted_index": "provider-a-index",
                    }],
                },
            ]
            captured_a = []
            requests_a = self._request_sequence([
                {
                    "id": "message_pause",
                    "type": "message",
                    "role": "assistant",
                    "content": server_content,
                    "stop_reason": "pause_turn",
                },
                {
                    "id": "message_final",
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "Provider A final answer.",
                    }],
                    "stop_reason": "end_turn",
                },
            ], captured_a)
            with mock.patch(
                    "loki_agent.loki.async_provider_request",
                    new=requests_a):
                answer_a = asyncio.run(loki.run_tool_loop_async(
                    loki.current_transcript(),
                    allowed={"Read"},
                    max_loops=4,
                ))

            requests_a.assert_exhausted()
            self.assertEqual(answer_a, "Provider A final answer.")
            self.assertEqual(len(captured_a), 2)
            self.assertTrue(all(
                request["url"] == provider_a.chat_provider.chat_url
                for request in captured_a))
            self.assertTrue(all(
                "x-api-key" not in request["headers"]
                for request in captured_a))
            exact_payload = captured_a[1]["payload"]
            self._assert_anthropic_payload_valid(exact_payload)
            self.assertEqual(
                exact_payload["messages"][1]["content"],
                server_content,
            )
            exact_serialized = json.dumps(exact_payload)
            self.assertIn("provider-a-signature", exact_serialized)
            self.assertIn("provider-a-encrypted", exact_serialized)
            self.assertIn("provider-a-index", exact_serialized)
            loki.mark_chat_log_dirty()
            loki.save_chat_log()

            loki.current_session().transcript_items = []
            with contextlib.redirect_stdout(io.StringIO()):
                loki.load_chat_log(path)
            provider_b = loki.make_runtime_config(
                "https://anthropic-b.example/v1/messages",
                protocols.ANTHROPIC_MESSAGES,
                model="claude-b",
                provider_id="anthropic-b",
                provider_name="Anthropic B",
                credential_ref=authentications.CredentialRef.environment(
                    "ANTHROPIC_B_API_KEY"),
            )
            loki.apply_runtime_config(provider_b)
            loki.set_session_connection(
                loki.active_connection_descriptor())
            loki.current_transcript().append(
                formats.message_item(
                    "user", "continue on provider B"))

            captured_b = []
            requests_b = self._request_sequence([{
                "id": "message_b",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "Provider B answer.",
                }],
                "stop_reason": "end_turn",
            }], captured_b)
            with mock.patch(
                    "loki_agent.loki.async_provider_request",
                    new=requests_b):
                answer_b = asyncio.run(loki.run_tool_loop_async(
                    loki.current_transcript(),
                    allowed={"Read"},
                ))

            requests_b.assert_exhausted()
            self.assertEqual(answer_b, "Provider B answer.")
            self.assertEqual(len(captured_b), 1)
            self.assertEqual(
                captured_b[0]["url"],
                provider_b.chat_provider.chat_url,
            )
            self.assertNotIn(
                "x-api-key", captured_b[0]["headers"])
            foreign_payload = captured_b[0]["payload"]
            self._assert_anthropic_payload_valid(foreign_payload)
            foreign_serialized = json.dumps(foreign_payload)
            self.assertIn("Portable result", foreign_serialized)
            self.assertIn("Search completed.", foreign_serialized)
            self.assertIn("Provider A final answer.",
                          foreign_serialized)
            self.assertNotIn("provider A private thought",
                             foreign_serialized)
            self.assertNotIn("provider-a-signature",
                             foreign_serialized)
            self.assertNotIn("provider-a-encrypted",
                             foreign_serialized)
            self.assertNotIn("provider-a-index",
                             foreign_serialized)
            projected_types = [
                block.get("type")
                for message in foreign_payload["messages"]
                for block in message.get("content", [])
            ]
            self.assertIn("tool_use", projected_types)
            self.assertIn("tool_result", projected_types)
            formats.validate_events(loki.current_transcript())


class SavedChatPickerJourneyTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_chats_pick_render_resume_and_continue(self):
        from datetime import datetime
        import uuid

        for selected_index in [2, 3]:
            with self.subTest(selected_index=selected_index), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                chat_dir = root / 'missing' / '.loki' / 'chats'
                self.assertFalse(chat_dir.exists())
                output, errors, terminal_calls, requests = io.StringIO(), io.StringIO(), [], []
                tasks_before = set(asyncio.all_tasks())
                process_cwd = os.getcwd()
                phase = {}

                class RecordingTerminal(terminals._TerminalTextOutput):
                    def goto_position(self, *args):
                        terminal_calls.append(('goto_position', *args))

                    def clear_to_end_of_screen(self):
                        terminal_calls.append(('clear_to_end_of_screen',))

                    def flush(self):
                        terminal_calls.append(('flush',))

                    def set_foreground_color(self, color):
                        pass

                    def set_background_color(self, color):
                        pass

                    def reset_colors_and_flags(self):
                        pass

                terminal = RecordingTerminal()
                terminal.assistant_markdown = terminals.AssistantMarkdownPresentation(terminal)

                class PickerInput(ScriptedInputSession):
                    active = False
                    modal_active = False

                    async def __aenter__(self):
                        self.active = True
                        return self

                    async def __aexit__(self, *args):
                        self.active = False

                    @contextlib.asynccontextmanager
                    async def modal(self):
                        if not self.active or self.modal_active:
                            raise AssertionError('invalid modal ownership')
                        self.modal_active = True
                        try:
                            yield self
                        finally:
                            self.modal_active = False

                    async def prompt(self, prompt=None, history=None):
                        if not self.modal_active:
                            raise AssertionError('picker prompt outside modal')
                        self.menu = output.getvalue()
                        return str(selected_index)

                input_session = PickerInput(['continued request', '/quit'])
                input_session.reader.cancel_event = asyncio.Event()
                input_session.menu = ''
                store = CredentialStore({
                    'LOKI_API_BASE': 'https://picker.example/v1/chat/completions',
                    'LOKI_PROVIDER': protocols.OPENAI_CHAT, 'LOKI_MODEL': 'picker-model',
                })
                config = loki.build_config_from_env(credentials=store)
                paths, saved, mtimes = {}, {}, {'oldest': 1000, 'middle': 2000, 'newest': 3000}
                selected_label = {2: 'middle', 3: 'newest'}[selected_index]

                async def completion(items, tools=None, *args, **kwargs):
                    requests.append(copy.deepcopy(items))
                    label, count = phase['label'], phase['count']
                    phase['count'] += 1
                    if count == 0:
                        if label == 'continued':
                            self.assertEqual(items[:len(saved[selected_label]['events'])],
                                             saved[selected_label]['events'])
                            self.assertEqual(formats.item_text(items[-1]), 'continued request')
                        return formats.DecodedTurn([
                            formats.message_item('assistant', f'prelude {label}'),
                            formats.tool_call_item(f'read-{label}', 'Read',
                                                   {'file_path': f'{label}.txt'}),
                        ])
                    self.assertEqual(count, 1)
                    result = items[-1]
                    self.assertEqual(result['type'], 'tool_result')
                    self.assertEqual(result['call_id'], f'read-{label}')
                    self.assertEqual(formats.item_text(result), f'1\tevidence {label}')
                    self.assertIn(loki._file_key(str(root / f'{label}.txt')), loki.file_state)
                    return formats.DecodedTurn([
                        formats.message_item('assistant', f'answer {label}')])

                async def fresh_session():
                    session = loki.Session(shell_cwd=directory,
                                           job_manager=loki.JobManager(str(root / 'jobs')))
                    self.addAsyncCleanup(session.job_manager.close_session_owned)
                    return session

                def read_blob(path):
                    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

                rendered = []
                real_present = terminal_frontend._ResumeTranscriptPresenter.write

                def observe_present(presenter, events):
                    self.assertEqual(terminal_calls[-3:], [
                        ('goto_position', 1, 1), ('clear_to_end_of_screen',), ('flush',)])
                    self.assertTrue(input_session.active)
                    self.assertFalse(input_session.modal_active)
                    self.assertEqual(events, saved[selected_label]['events'])
                    self.assertEqual(pathlib.Path(paths[selected_label]).read_bytes(), before[selected_label])
                    start = output.tell()
                    real_present(presenter, events)
                    rendered.append(output.getvalue()[start:])

                with mock.patch.object(loki, 'CHAT_LOG_DIR', str(chat_dir)), \
                        mock.patch.object(loki, 'CREDENTIALS', store), \
                        mock.patch.object(loki, 'file_state', {}), \
                        mock.patch.object(loki, 'TOOL_HOOK_PIPELINE', loki.tool_runtime.ToolHookPipeline()), \
                        mock.patch.object(terminal_frontend, '_terminal_activity', terminal_frontend.TerminalActivityStatus()), \
                        mock.patch.object(terminal_frontend, 'terminal', terminal), \
                        mock.patch.object(terminal_frontend, 'async_chat_completion', new=completion), \
                        mock.patch.object(terminals, 'redraw_status_bar'), \
                        mock.patch.object(terminals, 'open_terminal_stdin'), \
                        mock.patch.object(terminal_frontend, 'restore_output_area_after_input'), \
                        contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    # Creation order differs from mtime order, so creation-order picking cannot pass.
                    for label in ['newest', 'oldest', 'middle']:
                        session = await fresh_session()
                        with mock.patch.object(loki, '_DEFAULT_SESSION', session):
                            loki.apply_runtime_config(config)
                            path = loki.new_chat_log_path()
                            self.assertEqual(pathlib.Path(path).parent, chat_dir)
                            uuid.UUID(pathlib.Path(path).stem.removeprefix('chat-'))
                            self.assertTrue(chat_dir.is_dir())
                            if not paths:
                                chat_dir.rmdir()
                                self.assertFalse(chat_dir.exists())
                            loki.new_chat_log(path)
                            self.assertTrue(chat_dir.is_dir())
                            self.assertEqual(loki.current_chat_log_path(), os.path.realpath(path))
                            self.assertTrue(session.chat_log_dirty)
                            self.assertFalse(pathlib.Path(path).exists())
                            (root / f'{label}.txt').write_text(f'evidence {label}\n', encoding='utf-8')
                            session.transcript_items.append(formats.message_item('user', f'{label} request'))
                            phase.update(label=label, count=0)
                            self.assertEqual(await asyncio.wait_for(
                                terminal_frontend.run_terminal_turn_async(session.transcript_items), 5),
                                f'answer {label}')
                            loki.mark_chat_log_dirty()
                            self.assertTrue(loki.save_chat_log())
                            self.assertFalse(session.chat_log_dirty)
                            paths[label], saved[label] = path, read_blob(path)
                            formats.validate_events(saved[label]['events'])
                            self.assertEqual(saved[label]['events'], session.transcript_items)
                            os.utime(path, (mtimes[label], mtimes[label]))
                            await session.job_manager.close_session_owned()
                    self.assertEqual(len(set(paths.values())), 3)
                    before = {label: pathlib.Path(path).read_bytes() for label, path in paths.items()}
                    (root / 'continued.txt').write_text('evidence continued\n', encoding='utf-8')
                    resumed = await fresh_session()
                    self.assertEqual(resumed.transcript_items, [])
                    output.seek(0)
                    output.truncate()
                    phase.update(label='continued', count=0)
                    with mock.patch.object(loki, '_DEFAULT_SESSION', resumed), \
                            mock.patch.object(terminal_frontend, 'input_session', return_value=input_session), \
                            mock.patch.object(terminal_frontend._ResumeTranscriptPresenter, 'write', new=observe_present):
                        self.assertEqual(await asyncio.wait_for(terminal_frontend.async_main(['resume']), 5), 0)
                        self.assertEqual(resumed.chat_log_path, paths[selected_label])
                        self.assertFalse(resumed.chat_log_dirty)
                        self.assertFalse(input_session.active)
                        self.assertFalse(input_session.modal_active)
                        self.assertEqual(input_session.messages, [])
                        final = read_blob(paths[selected_label])
                        self.assertEqual(final['events'][:len(saved[selected_label]['events'])],
                                         saved[selected_label]['events'])
                        self.assertEqual(final['events'], resumed.transcript_items)
                        self.assertEqual(formats.item_text(final['events'][-1]['items'][-1]), 'answer continued')
                        self.assertEqual(final['session_state']['connection'], saved[selected_label]['session_state']['connection'])
                        await resumed.job_manager.close_session_owned()
                        self.assertFalse(resumed.job_manager.jobs)
                    self.assertEqual(len(requests), 8)
                    for label in mtimes:
                        if label != selected_label:
                            self.assertEqual(pathlib.Path(paths[label]).read_bytes(), before[label])
                    self.assertTrue(input_session.menu.startswith('\nSaved sessions:\n'))
                    rows = [line for line in input_session.menu.splitlines()
                            if line.startswith(('  1.', '  2.', '  3.'))]
                    self.assertEqual(rows, [
                        f'  {index}. {datetime.fromtimestamp(mtimes[label]):%Y-%m-%d %H:%M}  '
                        f'{pathlib.Path(paths[label]).stem[5:13]}  {label} request'
                        for index, label in enumerate(('oldest', 'middle', 'newest'), 1)])
                    self.assertEqual(rendered, [
                        f'User: {selected_label} request\n\n'
                        f'picker-model: prelude {selected_label}\n\n'
                        f"Tool call: 'Read'\n    file_path: '{selected_label}.txt'\n\n"
                        "Tool result: 'Read'\n\n"
                        f'picker-model: answer {selected_label}\n----\n'])
                    logical = '\n\n'.join(
                        ''.join(text for _, text in segments)
                        for _, segments in savefiles.ResumeTranscriptRenderer(
                            'picker-model').presentation(saved[selected_label]['events']))
                    self.assertEqual(logical,
                                     f'User: {selected_label} request\n\n'
                                     f'picker-model: prelude {selected_label}\n\n'
                                     f"Tool call: Read\n    file_path: '{selected_label}.txt'\n\n"
                                     f'Tool result: Read\n1\tevidence {selected_label}\n\n'
                                     f'picker-model: answer {selected_label}')
                    instructions = [event for event in saved[selected_label]['events']
                                    if event.get('role') in ['system', 'developer']]
                    self.assertTrue(instructions)
                    for event in instructions:
                        self.assertNotIn(formats.item_text(event), rendered[0])
                    self.assertNotIn('response_metadata', rendered[0])
                    self.assertNotIn('protocol_data', rendered[0])
                    self.assertEqual(errors.getvalue(), '')
                    self.assertEqual(os.getcwd(), process_cwd)
                    self.assertFalse(set(asyncio.all_tasks()) - tasks_before)


class ChatLogPathTests(unittest.TestCase):
    def test_bare_resume_names_resolve_to_local_loki_chat_directory(self):
        self.assertEqual(
            loki.resolve_chat_log_path("abc"),
            os.path.join(loki.CHAT_LOG_DIR, "chat-abc.json"),
        )
        self.assertEqual(
            loki.resolve_chat_log_path("chat-abc.json"),
            os.path.join(loki.CHAT_LOG_DIR, "chat-abc.json"),
        )

    def test_path_like_resume_arguments_stay_explicit(self):
        # normpath on both sides: the argument keeps its own separators.
        self.assertEqual(
            os.path.normpath(loki.resolve_chat_log_path("./chat-abc.json")),
            os.path.normpath(
                os.path.join(loki.STARTUP_CWD, "./chat-abc.json")),
        )
        self.assertEqual(
            os.path.normpath(loki.resolve_chat_log_path("logs/chat-abc.json")),
            os.path.normpath(
                os.path.join(loki.STARTUP_CWD, "logs", "chat-abc.json")),
        )


class SessionPickerTests(unittest.TestCase):
    """Tests for the `--resume` (no arg) session picker.

    Drives run_session_picker_async by monkeypatching get_input_async to feed a
    scripted sequence of inputs, with CHAT_LOG_DIR pointed at a tempdir of
    synthetic chat logs.
    """

    def _write_chat(self, dirpath, chat_id, text, mtime=None):
        path = os.path.join(dirpath, f"chat-{chat_id}.json")
        with open(path, "w") as f:
            f.write(text)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def _make_picker(self, dirpath, inputs):
        """Build a modal session that reads from `inputs` (a list of strings).

        Each call to modal.prompt returns the next input; EOFError when
        the list is exhausted (so infinite loops fail the test rather than
        hang it).
        """
        # Point CHAT_LOG_DIR at the tempdir for _chat_log_paths().
        saved_log_dir = loki.CHAT_LOG_DIR
        loki.CHAT_LOG_DIR = dirpath
        iterator = iter(inputs)

        class _FakeModal:
            def __init__(self):
                self.active = False

            async def __aenter__(self):
                self.active = True
                return self

            async def __aexit__(self, exc_type, exc, tb):
                self.active = False

            async def prompt(self, prompt=None, history=None):
                if not self.active:
                    raise AssertionError("prompt outside modal")
                try:
                    return next(iterator)
                except StopIteration:
                    raise EOFError

        class _FakeSession:
            def modal(self):
                return _FakeModal()

        session = _FakeSession()
        saved_terminal = terminal_frontend.terminal

        class _FakeTerminal:
            def save_cursor_position(self, *a, **k):
                pass

            def restore_cursor_position(self, *a, **k):
                pass

            def clear_to_end_of_screen(self, *a, **k):
                pass

            def goto_position(self, *a, **k):
                pass

            def flush(self, *a, **k):
                pass

            def write_text(
                    self, text, *, multiline=False, file=None):
                terminals._TerminalTextOutput.write_text(
                    self, text, multiline=multiline, file=file)

        terminal_frontend.terminal = _FakeTerminal()

        def restore():
            loki.CHAT_LOG_DIR = saved_log_dir
            terminal_frontend.terminal = saved_terminal

        return restore, session

    def test_picker_filter_matches_all_words_in_any_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "aaa", '{"text":"alpha beta"}', mtime=1000)
            self._write_chat(tmpdir, "bbb", '{"text":"beta gamma"}', mtime=2000)
            restore, session = self._make_picker(
                tmpdir, ["filter beta alpha", "1"])
            try:
                result = asyncio.run(terminal_frontend.run_session_picker_async(session))
            finally:
                restore()
            # Only the aaa log contains both "beta" and "alpha".
            self.assertTrue(result.endswith("chat-aaa.json"))

    def test_picker_filter_404_matches_literal_digits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "aaa", '{"text":"error 404 in nginx"}', mtime=1000)
            self._write_chat(tmpdir, "bbb", '{"text":"something else"}', mtime=2000)
            # Bare "404" should NOT match (parsed as int, out of range, ignored).
            restore, session = self._make_picker(
                tmpdir, ["404", "filter 404", "1"])
            try:
                result = asyncio.run(terminal_frontend.run_session_picker_async(session))
            finally:
                restore()
            self.assertTrue(result.endswith("chat-aaa.json"))

    def test_picker_bare_filter_clears(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "aaa", '{"text":"alpha beta"}', mtime=1000)
            self._write_chat(tmpdir, "bbb", '{"text":"gamma delta"}', mtime=2000)
            # Narrow to one match, then clear with bare "filter", then pick 2.
            (restore, session) = self._make_picker(
                tmpdir, ["filter alpha", "filter", "2"])
            try:
                result = asyncio.run(terminal_frontend.run_session_picker_async(session))
            finally:
                restore()
            # After clearing, both visible; "2" = bbb (newest last).
            self.assertTrue(result.endswith("chat-bbb.json"))

    def test_picker_empty_input_cancels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "aaa", '{"text":"alpha"}', mtime=1000)
            restore, session = self._make_picker(tmpdir, [""])
            try:
                result = asyncio.run(terminal_frontend.run_session_picker_async(session))
            finally:
                restore()
            self.assertIsNone(result)

    def test_picker_preview_handles_partial_json(self):
        # A truncated/garbled log file must not crash preview extraction.
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(
                tmpdir, "broken", '{"text":"hi there this is truncated', mtime=1000)
            # No closing quote, no closing brace -- regex should still grab "hi there...".
            restore, session = self._make_picker(tmpdir, ["1"])
            try:
                result = asyncio.run(terminal_frontend.run_session_picker_async(session))
            finally:
                restore()
            self.assertTrue(result.endswith("chat-broken.json"))

    def test_picker_unrecognized_input_keeps_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "aaa", '{"text":"alpha"}', mtime=1000)
            self._write_chat(tmpdir, "bbb", '{"text":"beta"}', mtime=2000)
            # "alpha" (no prefix) is unrecognized: not "filter ...", not an int,
            # not empty. Should re-render with the current (empty) filter, then
            # "1" selects the first row.
            restore, session = self._make_picker(tmpdir, ["alpha", "1"])
            try:
                result = asyncio.run(terminal_frontend.run_session_picker_async(session))
            finally:
                restore()
            self.assertTrue(result.endswith("chat-aaa.json"))


class ShellCwdTests(unittest.TestCase):
    def test_change_shell_cwd_does_not_change_process_cwd(self):
        names = ["shell_cwd", "previous_shell_cwd"]
        old_values = save_loki_state(names)
        process_cwd = os.getcwd()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loki.change_shell_cwd(tmpdir)
                resolved = os.path.realpath(tmpdir)

                self.assertEqual(loki.current_cwd(), resolved)
                self.assertEqual(os.getcwd(), process_cwd)
                self.assertEqual(
                    loki._resolve_path("file.txt"),
                    os.path.join(resolved, "file.txt"))
        finally:
            restore_loki_state(old_values)

    def test_bash_runs_in_shell_cwd(self):
        names = ["shell_cwd", "previous_shell_cwd", "job_manager"]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                workdir = os.path.join(tmpdir, "work")
                os.mkdir(workdir)
                loki.current_session().job_manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
                loki.change_shell_cwd(workdir)

                # Report a native path rather than the shell's rendering: the
                # Git Bash pwd prints /c/... which no Windows path comparison
                # can use.
                executable = sys.executable.replace(os.sep, "/")
                result = asyncio.run(loki.run_bash_async(
                    f"{shlex.quote(executable)} -c "
                    "\"import os; print(os.getcwd())\""))
                jobs = list(loki.current_job_manager().jobs.values())
        finally:
            restore_loki_state(old_values)

        self.assertIn("[stdout]\n" + os.path.realpath(workdir), result)
        self.assertEqual(os.path.basename(jobs[0].stdout_path), "stdout.log")
        self.assertEqual(os.path.basename(jobs[0].stderr_path), "stderr.log")

    def test_save_chat_log_persists_shell_cwd(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "shell_cwd",
            "previous_shell_cwd",
        ]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cwd = os.path.join(tmpdir, "work")
                os.mkdir(cwd)
                path = os.path.join(tmpdir, "chat-test.json")
                loki.new_chat_log(path)
                loki.change_shell_cwd(cwd)

                loki.save_chat_log()

                with open(path, "r", encoding="utf-8") as f:
                    blob = json.load(f)
        finally:
            restore_loki_state(old_values)

        self.assertEqual(
            blob["session_state"]["shell_cwd"], os.path.realpath(cwd))

    def test_save_chat_log_persists_connection_without_credential_value(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
            "runtime_config", ]
        sentinel = object()
        old_values = {
            name: loki.current_session().__dict__.get(name, sentinel) for name in names}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                config = loki.make_runtime_config(
                    "https://openrouter.ai/api/v1",
                    protocols.OPENAI_CHAT,
                    model="z-ai/glm",
                    provider_id="openrouter",
                    provider_name="OpenRouter",
                    credential_ref=(
                        authentications.CredentialRef.environment(
                            "OPENROUTER_API_KEY")),
                    model_status="deprecated",
                )
                loki.apply_runtime_config(config)
                loki.new_chat_log(path)
                loki.save_chat_log()
                text = pathlib.Path(path).read_text(encoding="utf-8")
                blob = json.loads(text)
        finally:
            restore_loki_state(old_values)

        connection = blob["session_state"]["connection"]
        self.assertEqual(connection["provider_id"], "openrouter")
        self.assertEqual(
            connection["credential"],
            {"kind": "env", "name": "OPENROUTER_API_KEY"},
        )
        self.assertEqual(connection["model_status"], "deprecated")
        self.assertNotIn("api_url", connection)
        self.assertEqual(
            connection["chat_url"],
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertNotIn("do-not-persist-this", text)

    def test_loading_and_clean_cleanup_leave_chat_bytes_unchanged(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "runtime_config",
            "shell_cwd", "previous_shell_cwd",
        ]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                descriptor = ConnectionDescriptor(
                    provider_id="openrouter",
                    provider_name="OpenRouter",
                    model="z-ai/glm",
                    chat_url="https://openrouter.ai/api/v1/chat/completions",
                    models_url="https://openrouter.ai/api/v1/models",
                    protocol=protocols.OPENAI_CHAT,
                    credential_ref=(
                        authentications.CredentialRef.environment(
                            "OPENROUTER_API_KEY")),
                )
                blob = formats.new_log_blob(
                    loki.initial_transcript_items(), [])
                blob["session_state"] = {
                    "shell_cwd": tmpdir,
                    "connection": descriptor.to_dict(),
                    "future_field": {"keep": True},
                }
                original = json.dumps(
                    blob, separators=(",", ":"), sort_keys=True).encode()
                pathlib.Path(path).write_bytes(original)
                loki.current_session().runtime_config = None

                loki.load_chat_log(path)
                saved = loki.save_chat_log()

                self.assertFalse(saved)
                self.assertFalse(loki.current_session().chat_log_dirty)
                self.assertEqual(pathlib.Path(path).read_bytes(), original)
                self.assertEqual(
                    loki.current_state()["connection"], descriptor.to_dict())
                self.assertEqual(
                    loki.current_state()["future_field"], {"keep": True})
        finally:
            restore_loki_state(old_values)

    def test_later_save_preserves_unavailable_loaded_connection(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "runtime_config",
            "shell_cwd", "previous_shell_cwd",
        ]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                descriptor = ConnectionDescriptor(
                    provider_id="openrouter",
                    provider_name="OpenRouter",
                    model="z-ai/glm",
                    chat_url="https://openrouter.ai/api/v1/chat/completions",
                    models_url="https://openrouter.ai/api/v1/models",
                    protocol=protocols.OPENAI_CHAT,
                    credential_ref=(
                        authentications.CredentialRef.environment(
                            "OPENROUTER_API_KEY")),
                )
                blob = formats.new_log_blob(
                    loki.initial_transcript_items(), [])
                legacy_connection = descriptor.to_dict()
                legacy_connection["api_url"] = "https://openrouter.ai/api/v1"
                blob["session_state"] = {
                    "shell_cwd": tmpdir,
                    "connection": legacy_connection,
                    "future_field": "retained",
                }
                pathlib.Path(path).write_text(
                    json.dumps(blob), encoding="utf-8")
                loki.current_session().runtime_config = None

                loki.load_chat_log(path)
                loki.mark_chat_log_dirty()
                self.assertTrue(loki.save_chat_log())

                after = json.loads(
                    pathlib.Path(path).read_text(encoding="utf-8"))
                self.assertEqual(
                    after["session_state"]["connection"],
                    descriptor.to_dict(),
                )
                self.assertEqual(
                    after["session_state"]["future_field"], "retained")
        finally:
            restore_loki_state(old_values)

    def test_resumed_chat_does_not_adopt_explicit_runtime_connection(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "runtime_config", "shell_cwd", "previous_shell_cwd",
        ]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                saved_descriptor = ConnectionDescriptor(
                    provider_id="saved",
                    provider_name="Saved",
                    model="saved-model",
                    chat_url="https://saved.example/v1/chat/completions",
                    models_url="https://saved.example/v1/models",
                    protocol=protocols.OPENAI_CHAT,
                    credential_ref=(
                        authentications.CredentialRef.environment(
                            "SAVED_API_KEY")),
                )
                blob = formats.new_log_blob(
                    loki.initial_transcript_items(), [])
                blob["session_state"] = {
                    "shell_cwd": tmpdir,
                    "connection": saved_descriptor.to_dict(),
                }
                pathlib.Path(path).write_text(
                    json.dumps(blob), encoding="utf-8")
                loki.apply_runtime_config(loki.make_runtime_config(
                    "https://override.example/v1",
                    protocols.OPENAI_CHAT,
                    model="override-model",
                    credential_ref=(
                        authentications.CredentialRef.environment(
                            "LOKI_API_KEY")),
                ))

                loki.load_chat_log(path)
                loki.mark_chat_log_dirty()
                loki.save_chat_log()

                after = json.loads(
                    pathlib.Path(path).read_text(encoding="utf-8"))
                self.assertEqual(
                    after["session_state"]["connection"],
                    saved_descriptor.to_dict(),
                )
        finally:
            restore_loki_state(old_values)

    def test_chat_save_atomically_replaces_existing_snapshot(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
        ]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                loki.new_chat_log(path)
                loki.save_chat_log()
                first_inode = os.stat(path).st_ino

                loki.current_transcript().append(
                    formats.message_item("user", "changed"))
                loki.mark_chat_log_dirty()
                loki.save_chat_log()

                self.assertNotEqual(os.stat(path).st_ino, first_inode)
                self.assertEqual(
                    [name for name in os.listdir(tmpdir)
                     if name.endswith(".tmp")],
                    [],
                )
        finally:
            restore_loki_state(old_values)

    def test_failed_atomic_publish_preserves_previous_snapshot(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
        ]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                loki.new_chat_log(path)
                loki.save_chat_log()
                original = pathlib.Path(path).read_bytes()

                loki.current_transcript().append(
                    formats.message_item("user", "must not publish"))
                loki.mark_chat_log_dirty()
                # Patch the seam the publisher uses, not os.replace: on Windows
                # the publish is private_files.replace_at (a handle-relative
                # rename), so an os.replace patch would never fire there.
                with mock.patch(
                        "loki_agent.loki.private_files.replace_at",
                        side_effect=OSError("publish failed")):
                    with self.assertRaisesRegex(OSError, "publish failed"):
                        loki.save_chat_log()

                self.assertEqual(pathlib.Path(path).read_bytes(), original)
                self.assertTrue(loki.current_session().chat_log_dirty)
                self.assertEqual(
                    [name for name in os.listdir(tmpdir)
                     if name.endswith(".tmp")],
                    [],
                )
        finally:
            restore_loki_state(old_values)

    def test_load_session_state_restores_shell_cwd(self):
        names = ["shell_cwd", "previous_shell_cwd"]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loki.load_session_state({"shell_cwd": tmpdir})

                self.assertEqual(loki.current_cwd(), os.path.realpath(tmpdir))
        finally:
            restore_loki_state(old_values)

    def test_explicit_apply_shell_cwd_false_keeps_the_launch_directory(self):
        names = ["shell_cwd", "previous_shell_cwd"]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as saved_dir:
                with tempfile.TemporaryDirectory() as launch_dir:
                    loki.change_shell_cwd(launch_dir)
                    loki.load_session_state(
                        {"shell_cwd": saved_dir}, apply_shell_cwd=False)

                    self.assertEqual(
                        loki.current_cwd(), os.path.realpath(launch_dir))
        finally:
            restore_loki_state(old_values)

    def test_saved_connection_confirmation_is_explicit(self):
        descriptor = ConnectionDescriptor(
            provider_id="openrouter",
            provider_name="OpenRouter",
            model="z-ai/glm",
            chat_url="https://openrouter.ai/api/v1/chat/completions",
            models_url="https://openrouter.ai/api/v1/models",
            protocol=protocols.OPENAI_CHAT,
            credential_ref=(
                authentications.CredentialRef.environment(
                    "OPENROUTER_API_KEY")),
        )

        class FakeSession:
            def __init__(self, answer):
                self.answer = answer
                self.calls = []

            def modal(self):
                self.calls.append("modal")
                return self

            async def __aenter__(self):
                self.calls.append("enter")
                return self

            async def __aexit__(self, exc_type, exc, tb):
                self.calls.append("exit")

            async def prompt(self, prompt):
                if "[y/N]" not in prompt:
                    raise AssertionError(prompt)
                self.calls.append("prompt")
                return self.answer

        no_session = FakeSession("")
        yes_session = FakeSession("yes")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            declined = asyncio.run(
                terminal_frontend.confirm_saved_connection_async(descriptor, no_session))
            accepted = asyncio.run(
                terminal_frontend.confirm_saved_connection_async(descriptor, yes_session))

        self.assertFalse(declined)
        self.assertTrue(accepted)
        self.assertEqual(
            no_session.calls, ["modal", "enter", "prompt", "exit"])
        self.assertEqual(
            yes_session.calls, ["modal", "enter", "prompt", "exit"])
        rendered = output.getvalue()
        self.assertTrue(rendered.startswith("\nSaved connection:\n"))
        self.assertIn("Saved connection:", rendered)
        self.assertIn("Provider: 'OpenRouter'", rendered)
        self.assertIn("Model: 'z-ai/glm'", rendered)
        self.assertIn(
            "Chat endpoint: "
            "'https://openrouter.ai/api/v1/chat/completions'",
            rendered,
        )
        self.assertIn("Credential: 'OPENROUTER_API_KEY'", rendered)


class SubagentLaunchTests(unittest.TestCase):
    async def _run_delegated_subagent(self, args, credentials=None):
        supervisor = credential_supervisors.CredentialSupervisor(
            credentials or CredentialStore({}))
        delegation = await supervisor.delegate()
        if os.name == "posix":
            # Duplicate the descriptors so the delegation can close its own
            # copies; a Windows endpoint is used directly and the delegation is
            # closed after the subagent finishes instead.
            owner_end = os.dup(delegation.owner_child)
            capability_end = os.dup(delegation.credential_child)
            delegation.child_spawned()
        else:
            owner_end = delegation.owner_child
            capability_end = delegation.credential_child
        try:
            if "--subagent-depth" not in args:
                args = [*args, "--subagent-depth", "1"]
            if "--root-conversation-id" not in args:
                args = [*args, "--root-conversation-id",
                        loki.current_session().root_conversation_id]
            return await subagents.async_main([
                *args,
                "--session-owner-fd", str(loki.host_ipc.reference(owner_end)),
                "--credential-capability-fd",
                str(loki.host_ipc.reference(capability_end)),
            ])
        finally:
            await delegation.close()

    def test_subagent_launch_reuses_its_own_dispatcher(self):
        saved = save_loki_state(["subagent_depth"])
        old_argv = sys.argv[:]
        try:
            results = []
            for depth in [0, 2]:
                loki.current_session().subagent_depth = depth
                for parent_entrypoint in ["./loki.py", "./loki-acp"]:
                    sys.argv = [parent_entrypoint]
                    results.append((
                        depth,
                        parent_entrypoint,
                        loki._subagent_argv("Explore", "inspect this"),
                    ))
        finally:
            sys.argv = old_argv
            restore_loki_state(saved)

        for depth, parent_entrypoint, result in results:
            self.assertEqual(result, [
                parent_entrypoint,
                "--subagent",
                "Explore",
                "--subagent-depth",
                str(depth + 1),
                "--root-conversation-id",
                loki.current_session().root_conversation_id,
                "--prompt",
                "inspect this",
                "--shell-cwd",
                loki.current_cwd(),
            ])

    def test_both_entrypoints_can_run_real_subagent(self):
        saved = save_loki_state([
            "runtime_config", "credential_authority", "job_manager"])
        old_argv = sys.argv[:]

        async def in_process(tmpdir, parent_entrypoint):
            session = loki.current_session()
            session.credential_authority = (
                authentications.CredentialBroker())
            session.job_manager = loki.JobManager(
                os.path.join(tmpdir, "jobs"))
            loki.apply_runtime_config(loki.make_runtime_config(
                "http://dummy.invalid/v1",
                protocols.DUMMY,
                model="dummy-model",
            ))
            # The packaged entrypoint on Windows; the checkout script on POSIX.
            sys.argv = [entrypoint(parent_entrypoint)]
            return await loki.run_agent_async(
                "recursive launch", "inspect this")

        def headless(tmpdir):
            # The terminal subagent re-proves containment, so on Windows it
            # must be spawned from a real contained runtime, which only the
            # headless entrypoint establishes.  The DUMMY provider emits one
            # Agent tool call, and the resulting subagent's "ok" comes back in
            # the final answer.
            workspace = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace, exist_ok=True)
            env = child_environment(
                HOME=tmpdir,
                XDG_CONFIG_HOME=os.path.join(tmpdir, "config"),
                XDG_STATE_HOME=os.path.join(tmpdir, "state"),
                TERM="dumb",
                LOKI_PROVIDER="dummy",
                LOKI_API_BASE="http://dummy.invalid/v1",
                LOKI_MODEL="dummy-model",
                LOKI_DUMMY_REPLY="ok",
                LOKI_DUMMY_TOOL_CALL=json.dumps({
                    "name": "Agent",
                    "arguments": {
                        "description": "recursive launch",
                        "prompt": "inspect this",
                        "subagent_type": "Explore",
                    },
                }),
            )
            configure_container(env, workspace)
            result = subprocess.run(
                [entrypoint("loki"), "--headless", "--prompt",
                 "inspect this"],
                cwd=workspace, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ok", result.stdout)
            self.assertNotIn("Error:", result.stdout)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                for parent_entrypoint in ["loki", "loki-acp"]:
                    with self.subTest(entrypoint=parent_entrypoint):
                        if os.name == "nt" and parent_entrypoint == "loki":
                            headless(tmpdir)
                        else:
                            result = asyncio.run(
                                in_process(tmpdir, parent_entrypoint))
                            self.assertNotIn("Error:", result)
                            self.assertIn("ok", result)
        finally:
            sys.argv = old_argv
            restore_loki_state(saved)

    def test_subagent_inherits_process_cwd_and_receives_shell_cwd(self):
        saved = save_loki_state(["shell_cwd"])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loki.current_session().shell_cwd = tmpdir
                manager = mock.Mock()
                manager.run_exec = mock.AsyncMock(return_value=(
                    types.SimpleNamespace(exit_code=0),
                    "exited",
                    "",
                    "",
                ))
                with mock.patch.object(
                        loki, "current_job_manager",
                        return_value=manager):
                    asyncio.run(loki.run_agent_async(
                        "inspect", "inspect this"))

                args, kwargs = manager.run_exec.await_args
                self.assertEqual(kwargs["cwd"], os.getcwd())
                self.assertTrue(kwargs["session_owned"])
                self.assertTrue(kwargs["subagent"])
                self.assertEqual(
                    args[0][-2:],
                    ["--shell-cwd", tmpdir],
                )
        finally:
            restore_loki_state(saved)

    def test_subagent_receives_only_current_credential_capability(self):
        saved = save_loki_state([
            "runtime_config",
            "reasoning_effort_preference",
            "session_state",
            "chat_log_path",
            "chat_log_dirty",
        ])
        credential = (
            authentications.CredentialRef.openai_subscription())
        profile = _codex_model(
            "gpt-5-codex",
            use_responses_lite=True,
            tool_mode="code_mode_only",
            default_reasoning_level="high",
            supported_reasoning_levels=[{"effort": "high"}],
            supports_reasoning_summaries=True,
            context_window=200000,
            base_instructions="must not enter the child environment",
        )
        effort_profile = _effort_profile("low", "high")
        manager = mock.Mock()
        manager.run_exec = mock.AsyncMock(return_value=(
            types.SimpleNamespace(exit_code=0),
            "completed",
            "",
            "",
        ))
        try:
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://chatgpt.com/backend-api/codex/responses",
                protocols.OPENAI_RESPONSES,
                model="gpt-5-codex",
                provider_id="openai-subscription",
                credential_ref=credential,
                auth_scheme="openai-subscription",
                openai_request_profile=profile,
                reasoning_effort_profile=effort_profile,
            ))
            loki.current_session().session_state = {}
            loki.current_session().chat_log_path = None
            loki.current_session().reasoning_effort_preference = "high"
            with mock.patch.object(
                    loki, "current_job_manager",
                    return_value=manager):
                asyncio.run(loki.run_agent_async(
                    "inspect", "inspect this"))
        finally:
            restore_loki_state(saved)

        _args, kwargs = manager.run_exec.await_args
        self.assertEqual(kwargs["credential_refs"], {credential})
        self.assertEqual(
            kwargs["env"]["LOKI_CREDENTIAL_REF"],
            credential.encode(),
        )
        self.assertEqual(
            kwargs["env"]["LOKI_AUTH_SCHEME"],
            "openai-subscription",
        )
        encoded_profile = json.loads(
            kwargs["env"]["LOKI_OPENAI_REQUEST_PROFILE"])
        self.assertEqual(
            openai_models.CodexModelRequestProfile.from_dict(
                encoded_profile),
            profile,
        )
        self.assertNotIn("base_instructions", encoded_profile)
        self.assertEqual(
            json.loads(
                kwargs["env"]["LOKI_REASONING_EFFORT_PROFILE"]),
            effort_profile.to_dict(),
        )
        self.assertNotIn("LOKI_REASONING_EFFORT", kwargs["env"])
        self.assertEqual(json.loads(kwargs["env"]["LOKI_TURN_THINKING"])["settings"]["effort"], "high")

    def test_subagent_depth_is_bounded_at_the_entrypoint(self):
        self.assertEqual(
            subagents.parse_args([
                "Explore",
                "--subagent-depth",
                str(loki.MAX_SUBAGENT_DEPTH),
            ]).subagent_depth,
            loki.MAX_SUBAGENT_DEPTH,
        )
        for value in ("0", str(loki.MAX_SUBAGENT_DEPTH + 1), "one"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                subagents.parse_args([
                    "Explore", "--subagent-depth", value])

    def test_maximum_depth_neither_advertises_nor_runs_agent(self):
        saved = save_loki_state(["subagent_depth", "agent_mode"])
        session = loki.current_session()
        session.subagent_depth = loki.MAX_SUBAGENT_DEPTH
        captured = {}

        async def fake_loop(messages, allowed, *, thinking):
            self.assertEqual(thinking.traces, "off")
            captured["messages"] = messages
            captured["allowed"] = allowed
            return "done"

        try:
            with mock.patch.object(
                    loki, "run_tool_loop_async", side_effect=fake_loop):
                result = asyncio.run(
                    subagents.run_prompt_async("Explore", "inspect"))
            manager = mock.Mock()
            with mock.patch.object(
                    loki, "current_job_manager",
                    return_value=manager):
                launch_result = asyncio.run(loki.run_agent_async(
                    "nested", "inspect"))
        finally:
            restore_loki_state(saved)

        self.assertEqual(result, "done")
        self.assertNotIn("Agent", captured["allowed"])
        self.assertIn(
            "maximum delegation depth",
            formats.item_text(captured["messages"][0]),
        )
        self.assertIn("maximum subagent depth", launch_result)
        manager.run_exec.assert_not_called()
        manager.run_background_exec.assert_not_called()

    def test_lower_depth_advertises_recursive_agent(self):
        saved = save_loki_state(["subagent_depth", "agent_mode"])
        session = loki.current_session()
        session.subagent_depth = loki.MAX_SUBAGENT_DEPTH - 1
        captured = {}

        async def fake_loop(messages, allowed, *, thinking):
            self.assertEqual(thinking.traces, "off")
            captured["messages"] = messages
            captured["allowed"] = allowed
            return "done"

        try:
            with mock.patch.object(
                    loki, "run_tool_loop_async", side_effect=fake_loop):
                asyncio.run(
                    subagents.run_prompt_async("Explore", "inspect"))
        finally:
            restore_loki_state(saved)

        self.assertIn("Agent", captured["allowed"])
        self.assertIn(
            "delegate another read-only Explore search",
            formats.item_text(captured["messages"][0]),
        )

    def test_subagent_operation_is_cancelled_when_owner_fd_closes(self):
        async def scenario():
            read_fd, write_fd = os.pipe()
            owner = credential_runtimes.SessionOwner(read_fd)
            cancelled = asyncio.Event()

            class CredentialClient:
                async def wait_closed(self):
                    await asyncio.Event().wait()

            async def operation():
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            try:
                runtime = credential_runtimes.CredentialRuntime(
                    owner, CredentialClient())
                task = asyncio.create_task(
                    runtime.run(operation()))
                await asyncio.sleep(0)
                os.close(write_fd)
                write_fd = None
                completed, _result = await asyncio.wait_for(
                    task, timeout=1)
                return completed, cancelled.is_set()
            finally:
                if write_fd is not None:
                    os.close(write_fd)
                await owner.close()

        completed, cancelled = asyncio.run(scenario())
        self.assertFalse(completed)
        self.assertTrue(cancelled)

    def test_owner_revocation_wins_simultaneous_operation_completion(self):
        async def scenario():
            owner_closed = asyncio.get_running_loop().create_future()
            owner_closed.set_result(b"")

            class Owner:
                closed_task = owner_closed

            class CredentialClient:
                async def wait_closed(self):
                    await asyncio.Event().wait()

            ran = False

            async def operation():
                nonlocal ran
                ran = True
                return "too late"

            runtime = credential_runtimes.CredentialRuntime(
                Owner(), CredentialClient())
            completed, result = await runtime.run(operation())
            return completed, result, ran

        completed, result, ran = asyncio.run(scenario())
        self.assertFalse(completed)
        self.assertIsNone(result)
        self.assertFalse(ran)

    def test_runtime_task_cancellation_cancels_its_operation(self):
        async def scenario():
            owner_closed = asyncio.get_running_loop().create_future()
            operation_cancelled = asyncio.Event()

            class Owner:
                closed_task = owner_closed

            class CredentialClient:
                async def wait_closed(self):
                    await asyncio.Event().wait()

            async def operation():
                try:
                    await asyncio.Event().wait()
                finally:
                    operation_cancelled.set()

            runtime = credential_runtimes.CredentialRuntime(
                Owner(), CredentialClient())
            task = asyncio.create_task(runtime.run(operation()))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return operation_cancelled.is_set()

        self.assertTrue(asyncio.run(scenario()))

    def test_owner_close_cancels_pending_credential_handshake(self):
        async def scenario():
            if os.name != "posix":
                owner_parent, owner_child = loki.host_ipc.owner_channel()
                owner = credential_runtimes.SessionOwner(owner_child)
                server_end, client_end = loki.host_ipc.socket_pair()
                capability_fd = loki.host_ipc.prepare_child_socket(client_end)
                try:
                    task = asyncio.create_task(
                        credential_runtimes._connect_while_owned(
                            capability_fd, owner))
                    await asyncio.sleep(0)
                    loki.host_ipc.close_end(owner_parent)
                    result = await asyncio.wait_for(task, timeout=1)
                    self.assertEqual(capability_fd.handles(), [])
                    return result
                finally:
                    await owner.close()
                    loki.host_ipc.close_end(server_end)
            read_fd, write_fd = os.pipe()
            owner = credential_runtimes.SessionOwner(read_fd)
            broker_socket, child_socket = socket.socketpair()
            capability_fd = child_socket.detach()
            try:
                task = asyncio.create_task(
                    credential_runtimes._connect_while_owned(
                        capability_fd, owner))
                await asyncio.sleep(0)
                os.close(write_fd)
                write_fd = None
                result = await asyncio.wait_for(task, timeout=1)
                return result
            finally:
                if write_fd is not None:
                    os.close(write_fd)
                broker_socket.close()
                await owner.close()

        self.assertIsNone(asyncio.run(scenario()))

    def test_subagent_operation_is_cancelled_when_broker_closes(self):
        async def scenario():
            if os.name != "posix":
                owner_parent, owner_child = loki.host_ipc.owner_channel()
                owner = credential_runtimes.SessionOwner(owner_child)

                def release_owner():
                    loki.host_ipc.close_end(owner_parent)
            else:
                read_fd, write_fd = os.pipe()
                owner = credential_runtimes.SessionOwner(read_fd)

                def release_owner():
                    os.close(write_fd)
            broker_closed = asyncio.Event()
            cancelled = asyncio.Event()

            class CredentialClient:
                async def wait_closed(self):
                    await broker_closed.wait()

            async def operation():
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            try:
                runtime = credential_runtimes.CredentialRuntime(
                    owner, CredentialClient())
                task = asyncio.create_task(
                    runtime.run(operation()))
                await asyncio.sleep(0)
                broker_closed.set()
                completed, _result = await asyncio.wait_for(
                    task, timeout=1)
                return completed, cancelled.is_set()
            finally:
                release_owner()
                await owner.close()

        completed, cancelled = asyncio.run(scenario())
        self.assertFalse(completed)
        self.assertTrue(cancelled)

    def test_subagent_refuses_owner_fd_without_credential_capability(self):
        if os.name != "posix":
            owner_parent, owner_child = loki.host_ipc.owner_channel()
            with contextlib.redirect_stderr(io.StringIO()):
                status = asyncio.run(subagents.async_main([
                    "Explore",
                    "--session-owner-fd",
                    str(loki.host_ipc.reference(owner_child)),
                ]))
            self.assertEqual(status, 2)
            # Closure is not observable here: ``_close_descriptors`` closes the
            # endpoint it parsed out of the argv reference -- a second
            # PipeEndpoint over the same handle numbers -- while the caller's
            # ``owner_child`` keeps recording those numbers.
            loki.host_ipc.close_end(owner_parent)
            return
        read_fd, write_fd = os.pipe()
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                status = asyncio.run(subagents.async_main([
                    "Explore",
                    "--session-owner-fd", str(read_fd),
                ]))
        finally:
            os.close(write_fd)

        self.assertEqual(status, 2)

    def test_subagent_cwd_failure_returns_usage_error(self):
        saved = save_loki_state(
            ["credential_authority", "CREDENTIALS", "shell_cwd"])
        try:
            with tempfile.TemporaryDirectory() as directory:
                missing = os.path.join(directory, "missing")
                with contextlib.redirect_stderr(io.StringIO()):
                    status = asyncio.run(
                        self._run_delegated_subagent([
                            "Explore",
                            "--shell-cwd", missing,
                        ]))
        finally:
            restore_loki_state(saved)

        self.assertEqual(status, 2)

    def test_subagent_configuration_failure_returns_usage_error(self):
        saved = save_loki_state(
            ["credential_authority", "CREDENTIALS"])
        try:
            with mock.patch.object(
                    subagents._core,
                    "build_config_from_env",
                    side_effect=ValueError("invalid config")), \
                    contextlib.redirect_stderr(io.StringIO()):
                status = asyncio.run(
                    self._run_delegated_subagent(["Explore"]))
        finally:
            restore_loki_state(saved)

        self.assertEqual(status, 2)

    def test_subagent_model_is_required(self):
        saved = save_loki_state(
            ["credential_authority", "CREDENTIALS", "runtime_config"])
        try:
            with mock.patch.object(
                    subagents._core, "build_config_from_env"), \
                    mock.patch.object(
                        subagents._core, "apply_runtime_config"), \
                    mock.patch.object(
                        subagents._core, "current_model",
                        return_value=""), \
                    contextlib.redirect_stderr(io.StringIO()):
                status = asyncio.run(
                    self._run_delegated_subagent(["Explore"]))
        finally:
            restore_loki_state(saved)

        self.assertEqual(status, 2)

    def test_subagent_tree_identity_is_installed_forwarded_and_restored(self):
        saved = save_loki_state([
            "credential_authority", "CREDENTIALS", "runtime_config",
            "reasoning_effort_preference",
        ])
        root = "35b2d314-fdfb-466f-bbd6-4f479fc82eb4"
        session = loki.current_session()
        previous_root = session.delegated_root_conversation_id
        thread = session.conversation_id

        async def run(*args, thinking):
            self.assertEqual(thinking.traces, "off")
            self.assertEqual(session.root_conversation_id, root)
            self.assertEqual(session.conversation_id, thread)
            argv = loki._subagent_argv("Explore", "nested search")
            nested = subagents.parse_args(argv[2:])
            self.assertEqual(nested.root_conversation_id, root)
            raise RuntimeError("failed child")

        try:
            with mock.patch.object(
                    subagents._core, "build_config_from_env",
                    return_value=loki.make_runtime_config(
                        "dummy://local", protocols.DUMMY, model="dummy")), \
                    mock.patch.object(subagents, "run_cli_async", new=run):
                with self.assertRaisesRegex(RuntimeError, "failed child"):
                    asyncio.run(self._run_delegated_subagent([
                        "Explore", "--root-conversation-id", root,
                    ]))
            self.assertEqual(session.delegated_root_conversation_id, previous_root)
        finally:
            restore_loki_state(saved)

    def test_subagent_cli_applies_explicit_shell_cwd(self):
        saved = save_loki_state([
            "credential_authority", "CREDENTIALS", "runtime_config",
            "shell_cwd",
        ])
        runner = mock.AsyncMock()
        try:
            with tempfile.TemporaryDirectory() as tmpdir, \
                    mock.patch.object(
                        subagents._core, "build_config_from_env"), \
                    mock.patch.object(
                        subagents._core, "apply_runtime_config"), \
                    mock.patch.object(
                        subagents._core, "current_model",
                        return_value="model"), \
                    mock.patch.object(
                        subagents, "run_cli_async",
                        new=runner):
                status = asyncio.run(
                    self._run_delegated_subagent([
                        "Explore",
                        "--prompt", "inspect this",
                        "--shell-cwd", tmpdir,
                    ]))

                self.assertEqual(status, 0)
                self.assertEqual(loki.current_cwd(), tmpdir)
                self.assertEqual(os.getcwd(), loki.STARTUP_CWD)
        finally:
            restore_loki_state(saved)

    def test_subagent_environment_contains_no_request_secret(self):
        names = ["runtime_config"]
        old_values = save_loki_state(names)
        source_env = {
            "PATH": os.environ.get("PATH", ""),
            "OPENROUTER_API_KEY": "unrelated-key",
        }
        CredentialStore.capture(dict(source_env))
        try:
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://example.test/v1",
                protocols.OPENAI_CHAT,
                model="active-model",
                credential_ref=(
                    authentications.CredentialRef.environment(
                        "EXAMPLE_API_KEY")),
                models_url="https://example.test/custom-models",
                max_tokens=12345,
                anthropic_version="2026-01-02",
                auth_header="X-Custom-Key",
            ))
            child_env = loki._subagent_env(environ=source_env)
        finally:
            restore_loki_state(old_values)

        self.assertNotIn("LOKI_API_KEY", child_env)
        self.assertEqual(child_env["LOKI_STREAM"], "0")
        self.assertNotIn("LOKI_OPENAI_REQUEST_PROFILE", child_env)
        self.assertNotIn("LOKI_OPENAI_MODEL_METADATA", child_env)
        self.assertNotIn("LOKI_RESPONSES_LITE", child_env)
        self.assertEqual(child_env["LOKI_MAX_TOKENS"], "12345")
        self.assertNotIn("LOKI_ANTHROPIC_VERSION", child_env)
        self.assertEqual(child_env["LOKI_AUTH_HEADER"], "X-Custom-Key")
        self.assertEqual(
            child_env["LOKI_CREDENTIAL_REF"], "env:EXAMPLE_API_KEY")
        self.assertEqual(child_env["LOKI_AUTH_SCHEME"], "custom")
        self.assertEqual(
            child_env["LOKI_MODELS_URL"],
            "https://example.test/custom-models",
        )
        self.assertNotIn("OPENROUTER_API_KEY", child_env)

    def test_subagent_environment_preserves_credentialless_connection(self):
        names = ["runtime_config"]
        old_values = save_loki_state(names)
        try:
            loki.apply_runtime_config(loki.make_runtime_config(
                "http://localhost:8000/v1",
                protocols.OPENAI_CHAT,
                model="local-model",
                stream=True,
            ))
            child_env = loki._subagent_env(
                environ={"LOKI_API_KEY": "must-not-leak"})
        finally:
            restore_loki_state(old_values)

        self.assertEqual(child_env["LOKI_API_BASE"],
                         "http://localhost:8000/v1")
        self.assertEqual(child_env["LOKI_MODEL"], "local-model")
        self.assertEqual(child_env["LOKI_STREAM"], "1")
        self.assertNotIn("LOKI_API_KEY", child_env)
        self.assertNotIn("LOKI_CREDENTIAL_REF", child_env)
        self.assertNotIn("LOKI_AUTH_SCHEME", child_env)


class RequestTimeCredentialTests(unittest.TestCase):
    def setUp(self):
        self.saved = save_loki_state(
            ["runtime_config", "credential_authority"])

    def tearDown(self):
        restore_loki_state(self.saved)

    def _install_subscription(self, *, stream=False):
        async def refresh(refresh_token):
            self.assertEqual(refresh_token, "refresh-old")
            return authentications.RefreshResult(
                access_token="access-new",
                refresh_token="refresh-new",
            )

        async def rotate(tokens):
            return authentications.refreshed_openai_tokens(
                tokens, await refresh(tokens.refresh_token))

        broker = authentications.CredentialBroker()
        broker.install_openai_subscription(
            authentications.OpenAITokenSet(
                access_token="access-old",
                refresh_token="refresh-old",
                account_id="account",
                expires_at=10**12,
            ),
            rotate=rotate,
        )
        loki.current_session().credential_authority = broker
        loki.apply_runtime_config(loki.make_runtime_config(
            "https://chatgpt.com/backend-api/codex/responses",
            protocols.OPENAI_RESPONSES,
            model="gpt-5-codex",
            provider_id="openai-subscription",
            credential_ref=(
                authentications.CredentialRef.openai_subscription()),
            auth_scheme="openai-subscription",
            stream=stream,
            openai_request_profile=_codex_model("gpt-5-codex"),
        ))

    def _assert_codex_identity(self, headers):
        session = loki.current_session()
        self.assertEqual(headers["session-id"], session.root_conversation_id)
        self.assertEqual(headers["thread-id"], session.conversation_id)
        self.assertEqual(headers["x-client-request-id"], session.conversation_id)

    def test_codex_identity_headers_match_projected_requests(self):
        requests = []
        response = {"object": "response", "status": "completed", "output": []}

        async def buffered(method, url, **kwargs):
            requests.append(kwargs)
            return http_client.HttpResponse(
                url, 200, "OK", {}, json.dumps(response).encode())

        @contextlib.asynccontextmanager
        async def streaming(method, url, **kwargs):
            requests.append(kwargs)

            async def body():
                yield ("data: " + json.dumps({
                    "type": "response.completed", "response": response,
                }) + "\n\n").encode()

            yield http_client.HttpStreamResponse(
                url, 200, "OK", {"content-type": "text/event-stream"}, body())

        session = loki.current_session()
        for stream in [False, True]:
            for root in [None, "35b2d314-fdfb-466f-bbd6-4f479fc82eb4"]:
                with self.subTest(stream=stream, delegated_root=root):
                    self._install_subscription(stream=stream)
                    provider = loki.current_config().chat_provider
                    provider.headers.update({
                        "SESSION-ID": "untrusted",
                        "Thread-Id": "untrusted",
                        "X-Client-Request-ID": "untrusted",
                    })
                    original_headers = dict(provider.headers)
                    with mock.patch.object(
                            session, "delegated_root_conversation_id", root), \
                            mock.patch.object(
                                http_client, "async_http_request", new=buffered), \
                            mock.patch.object(
                                http_client, "async_http_stream", new=streaming):
                        for _ in range(2):
                            asyncio.run(loki.async_chat_completion(
                                [formats.message_item("user", "hello")], tools=[]))
                            request = requests[-1]
                            headers = request["headers_in"]
                            self._assert_codex_identity(headers)
                            payload = json.loads(request["body"])
                            self.assertEqual(
                                headers["thread-id"], payload["prompt_cache_key"])
                            for name in [
                                    "SESSION-ID", "Thread-Id", "X-Client-Request-ID"]:
                                self.assertNotIn(name, headers)
                    self.assertEqual(provider.headers, original_headers)

    def test_codex_identity_headers_are_not_added_to_other_requests(self):
        requests = []

        async def request(method, url, **kwargs):
            requests.append(kwargs["headers_in"])
            return http_client.HttpResponse(url, 200, "OK", {}, b"{}")

        async def streamed(url, payload, headers, *args, **kwargs):
            requests.append(headers)
            return protocols.ProviderResponse({})

        with mock.patch.object(http_client, "async_http_request", new=request), \
                mock.patch.object(
                    loki, "_async_chat_stream_request_once", new=streamed):
            self._install_subscription()
            asyncio.run(loki.async_provider_request(
                "GET", authentications.OPENAI_CHATGPT_MODELS_REQUEST_URL))
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://api.example.test/v1/responses",
                protocols.OPENAI_RESPONSES, model="public-model"))
            asyncio.run(loki.async_provider_request(
                "POST", loki.current_config().chat_provider.chat_url, {}))
            asyncio.run(loki.async_chat_stream_request(
                loki.current_config().chat_provider.chat_url, {}))

        for headers in requests:
            for name in ["session-id", "thread-id", "x-client-request-id"]:
                self.assertNotIn(name, headers)

    def test_buffered_401_refreshes_once_with_same_idempotency_key(self):
        self._install_subscription()
        calls = []

        async def request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            if len(calls) == 1:
                return loki.http_client.HttpResponse(
                    url, 401, "Unauthorized", {}, b"{}")
            return loki.http_client.HttpResponse(
                url, 200, "OK", {}, b"{}")

        with mock.patch.object(
                loki.http_client, "async_http_request", new=request):
            result = asyncio.run(loki.async_provider_request(
                "POST",
                loki.current_config().chat_provider.input_url,
                {"input": []}))

        self.assertEqual(result.payload, {})
        self.assertEqual(len(calls), 2)
        first_headers = calls[0][2]["headers_in"]
        second_headers = calls[1][2]["headers_in"]
        self._assert_codex_identity(first_headers)
        self._assert_codex_identity(second_headers)
        self.assertEqual(
            first_headers["Authorization"], "Bearer access-old")
        self.assertEqual(
            second_headers["Authorization"], "Bearer access-new")
        self.assertEqual(
            first_headers["ChatGPT-Account-ID"], "account")
        self.assertEqual(
            first_headers[loki.LLM_IDEMPOTENCY_HEADER_OPENAI],
            second_headers[loki.LLM_IDEMPOTENCY_HEADER_OPENAI],
        )

    def test_buffered_completion_records_only_header_selected_model(self):
        response_body = json.dumps({
            "object": "response",
            "status": "completed",
            "model": "untrusted-envelope-model",
            "output": [],
        }).encode("utf-8")
        for headers, expected in [
                ({"oPeNaI-MoDeL": "server-selected-model"},
                 "server-selected-model"),
                ({}, "gpt-5-codex"),
        ]:
            with self.subTest(headers=headers):
                self._install_subscription()
                response = loki.http_client.HttpResponse(
                    authentications.OPENAI_CHATGPT_RESPONSES_URL,
                    200,
                    "OK",
                    headers,
                    response_body,
                )
                with mock.patch.object(
                        loki.http_client,
                        "async_http_request",
                        new=mock.AsyncMock(return_value=response)):
                    turn = asyncio.run(loki.async_chat_completion(
                        [formats.message_item("user", "hello")],
                        tools=[],
                    ))

                self.assertEqual(turn.metadata["model"], expected)

    def test_streaming_model_header_precedence(self):
        for http_model, expected in [
                ("http-model", "http-model"),
                (None, "nested-model"),
        ]:
            with self.subTest(http_model=http_model):
                self._install_subscription(stream=True)

                async def response_body():
                    yield (
                        b'data: {"type":"response.metadata","metadata":{'
                        b'"openai_verification_recommendation":['
                        b'"trusted_access_for_cyber",'
                        b'"trusted_access_for_cyber"]}}\n\n'
                        b'data: {"type":"response.completed",'
                        b'"headers":{"OpenAI-Model":"top-level-model"},'
                        b'"response":{"id":"response_1",'
                        b'"headers":{"openai-model":"nested-model"},'
                        b'"status":"completed","output":[]}}\n\n'
                    )

                @contextlib.asynccontextmanager
                async def stream(method, request_url, **kwargs):
                    headers = {
                        "content-type": "text/event-stream",
                    }
                    if http_model is not None:
                        headers["OPENAI-MODEL"] = http_model
                    yield loki.http_client.HttpStreamResponse(
                        request_url,
                        200,
                        "OK",
                        headers,
                        response_body(),
                    )

                with mock.patch.object(
                        loki.http_client,
                        "async_http_stream",
                        new=stream):
                    turn = asyncio.run(loki.async_chat_completion(
                        [formats.message_item("user", "hello")],
                        tools=[],
                    ))

                self.assertEqual(turn.metadata["model"], expected)
                self.assertEqual(
                    formats.provider_notice_codes(turn),
                    [formats.TRUSTED_ACCESS_FOR_CYBER],
                )

    def test_public_responses_ignore_subscription_model_header(self):
        loki.apply_runtime_config(loki.make_runtime_config(
            "https://api.example.test/v1/responses",
            protocols.OPENAI_RESPONSES,
            model="public-model",
        ))
        response = loki.http_client.HttpResponse(
            "https://api.example.test/v1/responses",
            200,
            "OK",
            {"OpenAI-Model": "must-not-be-trusted"},
            b"{}",
        )

        with mock.patch.object(
                loki.http_client,
                "async_http_request",
                new=mock.AsyncMock(return_value=response)):
            result = asyncio.run(loki.async_provider_request(
                "POST", response.url, {}))

        self.assertIsNone(result.effective_model)

    def test_streaming_401_refreshes_once_with_same_idempotency_key(self):
        self._install_subscription(stream=True)
        calls = []

        async def request_once(
                url, payload, headers, on_text_delta, cancel_check,
                codex_turn_state=None, observe=None):
            calls.append(dict(headers))
            if len(calls) == 1:
                raise loki.StreamingApiError(
                    url, 401, "Unauthorized", "{}")
            return protocols.ProviderResponse({})

        with mock.patch.object(
                loki, "_async_chat_stream_request_once",
                new=request_once):
            result = asyncio.run(loki.async_chat_stream_request(
                loki.current_config().chat_provider.input_url,
                {"input": []}))

        self.assertEqual(result.payload, {})
        self._assert_codex_identity(calls[0])
        self._assert_codex_identity(calls[1])
        self.assertEqual(
            calls[0]["Authorization"], "Bearer access-old")
        self.assertEqual(
            calls[1]["Authorization"], "Bearer access-new")
        self.assertEqual(
            calls[0][loki.LLM_IDEMPOTENCY_HEADER_OPENAI],
            calls[1][loki.LLM_IDEMPOTENCY_HEADER_OPENAI],
        )

    def test_streaming_response_failure_retries_when_classified_retryable(self):
        self._install_subscription(stream=True)
        calls = []

        async def request_once(
                url, payload, headers, on_text_delta, cancel_check,
                codex_turn_state=None, observe=None):
            calls.append(dict(headers))
            if len(calls) < 3:
                raise protocols.ResponseApiError(
                    "temporary server failure",
                    code="server_error",
                    retryable=True,
                )
            return protocols.ProviderResponse({"id": "response"})

        with mock.patch.object(
                loki, "_async_chat_stream_request_once",
                new=request_once), \
                mock.patch.object(
                    loki, "HTTP_RETRY_BASE_DELAY_S", 0), \
                mock.patch.object(
                    loki, "HTTP_RETRY_MAX_JITTER_S", 0):
            result = asyncio.run(loki.async_chat_stream_request(
                loki.current_config().chat_provider.input_url,
                {"input": []}))

        self.assertEqual(result.payload, {"id": "response"})
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            len({
                headers[loki.LLM_IDEMPOTENCY_HEADER_OPENAI]
                for headers in calls
            }),
            1,
        )

    def test_terminal_response_failure_is_not_saved_as_a_turn(self):
        transcript = [formats.message_item("user", "hello")]
        events = []

        async def chat_fn(
                items, on_text_delta, *, codex_turn_state):
            raise protocols.ResponseApiError(
                "quota exhausted",
                code="insufficient_quota",
                category="quota",
                retryable=False,
            )

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            on_event=events.append,
            stream_chat=True,
        ))

        self.assertEqual(result, "")
        self.assertEqual(
            [item["type"] for item in transcript],
            ["message"],
        )
        self.assertEqual(
            [event["type"] for event in events],
            ["api_error"],
        )

    def test_streamed_response_failure_reaches_retry_classifier(self):
        self._install_subscription(stream=True)
        calls = []

        async def response_body():
            yield (
                b'data: {"type":"response.failed","response":{'
                b'"status":"failed","error":{"code":"server_error",'
                b'"message":"temporary failure"}}}\n\n'
            )

        @contextlib.asynccontextmanager
        async def fake_http_stream(method, request_url, **kwargs):
            calls.append(kwargs)
            yield loki.http_client.HttpStreamResponse(
                request_url,
                200,
                "OK",
                {
                    "content-type": "text/event-stream",
                    loki.CODEX_TURN_STATE_HEADER: (
                        "first-state"
                        if len(calls) == 1 else "ignored-later-state"),
                },
                response_body(),
            )

        with mock.patch.object(
                loki.http_client, "async_http_stream",
                side_effect=fake_http_stream), \
                mock.patch.object(
                    loki, "HTTP_RETRY_BASE_DELAY_S", 0), \
                mock.patch.object(
                    loki, "HTTP_RETRY_MAX_JITTER_S", 0):
            with self.assertRaises(
                    protocols.ResponseApiError) as raised:
                asyncio.run(loki.async_chat_stream_request(
                    loki.current_config().chat_provider.input_url,
                    {"input": []}))

        self.assertEqual(raised.exception.code, "server_error")
        self.assertEqual(len(calls), 3)
        self.assertNotIn(
            loki.CODEX_TURN_STATE_HEADER,
            calls[0]["headers_in"],
        )
        self.assertEqual(
            calls[1]["headers_in"][loki.CODEX_TURN_STATE_HEADER],
            "first-state",
        )
        self.assertEqual(
            calls[2]["headers_in"][loki.CODEX_TURN_STATE_HEADER],
            "first-state",
        )

    def test_caller_cannot_supply_codex_turn_state(self):
        self._install_subscription(stream=True)
        observed = []

        async def request_once(
                url, payload, headers, on_text_delta, cancel_check,
                codex_turn_state=None, observe=None):
            observed.append(dict(headers))
            return protocols.ProviderResponse({
                "object": "response",
                "status": "completed",
                "output": [],
            })

        with mock.patch.object(
                loki, "_async_chat_stream_request_once",
                new=request_once):
            asyncio.run(loki.async_chat_stream_request(
                loki.current_config().chat_provider.input_url,
                {"input": []},
                request_headers={
                    "X-Codex-Turn-State": "caller-forged",
                },
            ))

        self.assertNotIn("X-Codex-Turn-State", observed[0])
        self.assertNotIn(
            loki.CODEX_TURN_STATE_HEADER, observed[0])

    def test_static_credential_does_not_retry_a_401(self):
        credential = authentications.CredentialRef.environment(
            "EXAMPLE_API_KEY")
        broker = authentications.CredentialBroker()
        broker.install_static(credential, "static-secret")
        loki.current_session().credential_authority = broker
        loki.apply_runtime_config(loki.make_runtime_config(
            "https://example.test/v1/responses",
            protocols.OPENAI_RESPONSES,
            model="model",
            credential_ref=credential,
        ))
        calls = []

        async def request(method, url, **kwargs):
            calls.append(kwargs)
            return loki.http_client.HttpResponse(
                url, 401, "Unauthorized", {}, b"{}")

        with mock.patch.object(
                loki.http_client, "async_http_request", new=request):
            with self.assertRaises(loki.ApiError):
                asyncio.run(loki.async_provider_request(
                    "POST",
                    loki.current_config().chat_provider.input_url,
                    {"input": []}))

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["headers_in"]["Authorization"],
            "Bearer static-secret",
        )


class StreamingCompletionTests(unittest.TestCase):
    def setUp(self):
        self.old_runtime_config = loki.current_config()
        self.old_model = loki.current_model()
        loki.apply_runtime_config(loki.make_runtime_config(
            "http://localhost:8000/v1",
            protocols.OPENAI_CHAT,
            model="local-model",
            stream=True,
        ))

    def tearDown(self):
        loki.current_session().runtime_config = self.old_runtime_config
        # model restored with runtime_config above

    def test_text_delta_arrives_before_stream_completion(self):
        async def scenario():
            first_seen = asyncio.Event()
            release = asyncio.Event()
            deltas = []
            payloads = []

            async def response_body():
                yield (
                    b'data: {"id":"chat_1","object":'
                    b'"chat.completion.chunk","choices":[{"index":0,'
                    b'"delta":{"role":"assistant","content":"hel"},'
                    b'"finish_reason":null}]}\n\n')
                await release.wait()
                yield (
                    b'data: {"id":"chat_1","choices":[{"index":0,'
                    b'"delta":{"content":"lo"},"finish_reason":"stop"}]}'
                    b'\n\ndata: [DONE]\n\n')

            @contextlib.asynccontextmanager
            async def fake_http_stream(method, request_url, **kwargs):
                payloads.append(json.loads(kwargs["body"]))
                yield loki.http_client.HttpStreamResponse(
                    request_url,
                    200,
                    "OK",
                    {"content-type": "text/event-stream"},
                    response_body(),
                )

            def on_delta(text):
                deltas.append(text)
                first_seen.set()

            with mock.patch(
                    "loki_agent.loki.http_client.async_http_stream",
                    side_effect=fake_http_stream):
                task = asyncio.create_task(loki.async_chat_completion(
                    [formats.message_item("user", "hello")],
                    tools=[],
                    on_text_delta=on_delta,
                ))
                await asyncio.wait_for(first_seen.wait(), timeout=1)
                self.assertFalse(task.done())
                release.set()
                items = await task
            return payloads, deltas, items

        payloads, deltas, items = asyncio.run(scenario())

        self.assertEqual(len(payloads), 1)
        self.assertIs(payloads[0]["stream"], True)
        self.assertEqual(deltas, ["hel", "lo"])
        self.assertEqual(
            [item["type"] for item in items],
            ["message"],
        )
        self.assertEqual(formats.item_text(items[0]), "hello")

    def test_responses_stream_returns_at_completed_without_waiting_for_eof(self):
        loki.apply_runtime_config(loki.make_runtime_config(
            "http://localhost:8000/v1/responses",
            protocols.OPENAI_RESPONSES,
            model="local-model",
            stream=True,
        ))

        async def scenario():
            keep_open = asyncio.Event()

            async def response_body():
                yield (
                    b'data: {"type":"response.completed","response":{'
                    b'"id":"response_1","status":"completed","output":[]}}'
                    b'\n\n'
                )
                await keep_open.wait()

            @contextlib.asynccontextmanager
            async def fake_http_stream(method, request_url, **kwargs):
                yield loki.http_client.HttpStreamResponse(
                    request_url,
                    200,
                    "OK",
                    {"content-type": "text/event-stream"},
                    response_body(),
                )

            with mock.patch(
                    "loki_agent.loki.http_client.async_http_stream",
                    side_effect=fake_http_stream):
                return await asyncio.wait_for(
                    loki.async_chat_completion(
                        [formats.message_item("user", "hello")],
                        tools=[],
                    ),
                    timeout=0.2,
                )

        turn = asyncio.run(scenario())

        self.assertTrue(turn.complete)
        self.assertEqual(turn.items, [])

    def test_public_responses_never_receive_codex_turn_state(self):
        loki.apply_runtime_config(loki.make_runtime_config(
            "https://api.openai.com/v1/responses",
            protocols.OPENAI_RESPONSES,
            model="local-model",
            stream=True,
        ))
        observed = []

        async def request_once(
                url, payload, headers, on_text_delta, cancel_check,
                codex_turn_state=None, observe=None):
            observed.append((dict(headers), codex_turn_state))
            return protocols.ProviderResponse({
                "object": "response",
                "status": "completed",
                "output": [],
            })

        with mock.patch.object(
                loki, "_async_chat_stream_request_once",
                new=request_once):
            asyncio.run(loki.async_chat_stream_request(
                loki.current_config().chat_provider.input_url,
                {"input": []},
                codex_turn_state=loki.CodexTurnState(
                    "must-not-leave-subscription"),
            ))

        self.assertIsNone(observed[0][1])
        self.assertNotIn(
            loki.CODEX_TURN_STATE_HEADER, observed[0][0])

    def test_reasoning_deltas_are_silent_and_replay_only_at_origin(self):
        deltas = []
        diagnostics = io.StringIO()

        async def response_body():
            for reasoning in ["The", " user", " said", " Test"]:
                yield (
                    "data: "
                    + json.dumps({
                        "id": "chat_reasoning",
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "reasoning_content": reasoning,
                            },
                            "finish_reason": None,
                        }],
                    })
                    + "\n\n"
                ).encode("utf-8")
            yield (
                b'data: {"id":"chat_reasoning","choices":[{"index":0,'
                b'"delta":{"content":"Working."},'
                b'"finish_reason":"stop"}]}\n\n'
                b'data: [DONE]\n\n'
            )

        @contextlib.asynccontextmanager
        async def fake_http_stream(method, request_url, **kwargs):
            yield loki.http_client.HttpStreamResponse(
                request_url,
                200,
                "OK",
                {"content-type": "text/event-stream"},
                response_body(),
            )

        user = formats.message_item("user", "Test")
        with mock.patch(
                "loki_agent.loki.http_client.async_http_stream",
                side_effect=fake_http_stream), \
                contextlib.redirect_stderr(diagnostics):
            turn = asyncio.run(loki.async_chat_completion(
                [user],
                tools=[],
                on_text_delta=deltas.append,
            ))

        event = turn.to_event()
        origin_payload = (
            loki.current_config().chat_provider.chat_payload(
                [user, event], [], loki.current_model()))
        foreign = loki.make_runtime_config(
            "http://other-localhost:8000/v1",
            protocols.OPENAI_CHAT,
            model="other-model",
            stream=True,
        )
        foreign_payload = foreign.chat_provider.chat_payload(
            [user, event], [], "other-model")

        self.assertEqual(diagnostics.getvalue(), "")
        self.assertEqual(deltas, ["Working."])
        self.assertEqual(formats.item_text(turn.items[0]), "Working.")
        self.assertEqual(
            turn.items[0]["protocol_data"][protocols.OPENAI_CHAT]
            ["fields"]["reasoning_content"],
            "The user said Test",
        )
        self.assertIn(
            "The user said Test", json.dumps(origin_payload))
        self.assertNotIn(
            "The user said Test", json.dumps(foreign_payload))

    def test_normal_json_response_to_stream_request_is_not_resent(self):
        calls = []
        deltas = []
        response = {
            "id": "chat_1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "buffered"},
                "finish_reason": "stop",
            }],
        }

        async def response_body():
            yield json.dumps(response).encode("utf-8")

        @contextlib.asynccontextmanager
        async def fake_http_stream(method, request_url, **kwargs):
            calls.append(json.loads(kwargs["body"]))
            yield loki.http_client.HttpStreamResponse(
                request_url,
                200,
                "OK",
                {"content-type": "application/json"},
                response_body(),
            )

        with mock.patch(
                "loki_agent.loki.http_client.async_http_stream",
                side_effect=fake_http_stream):
            items = asyncio.run(loki.async_chat_completion(
                [formats.message_item("user", "hello")],
                tools=[],
                on_text_delta=deltas.append,
            ))

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["stream"], True)
        self.assertEqual(deltas, [])
        self.assertEqual(formats.item_text(items[0]), "buffered")

    def test_http_rejection_suggests_disabling_streaming(self):
        async def response_body():
            yield b'{"error":{"message":"stream unsupported"}}'

        @contextlib.asynccontextmanager
        async def fake_http_stream(method, request_url, **kwargs):
            yield loki.http_client.HttpStreamResponse(
                request_url,
                400,
                "Bad Request",
                {"content-type": "application/json"},
                response_body(),
            )

        with mock.patch(
                "loki_agent.loki.http_client.async_http_stream",
                side_effect=fake_http_stream):
            with self.assertRaises(loki.StreamingApiError) as raised:
                asyncio.run(loki.async_chat_completion(
                    [formats.message_item("user", "hello")],
                    tools=[],
                ))

        self.assertIn("set LOKI_STREAM=0", raised.exception.formatted())

    def _run_stream_turn(self, body_chunks, content_type):
        deltas = []

        async def response_body():
            for chunk in body_chunks:
                yield chunk

        @contextlib.asynccontextmanager
        async def fake_http_stream(method, request_url, **kwargs):
            yield loki.http_client.HttpStreamResponse(
                request_url,
                200,
                "OK",
                {"content-type": content_type},
                response_body(),
            )

        with mock.patch(
                "loki_agent.loki.http_client.async_http_stream",
                side_effect=fake_http_stream):
            items = asyncio.run(loki.async_chat_completion(
                [formats.message_item("user", "hello")],
                tools=[],
                on_text_delta=deltas.append,
            ))
        return deltas, items

    def test_marker_split_across_chunks_with_lying_content_type(self):
        # The first TCP read ends mid-"data:" while the content-type claims
        # JSON; the sniff must wait for more bytes instead of misparsing the
        # SSE stream as one JSON document.
        deltas, items = self._run_stream_turn([
            b'da',
            b'ta: {"choices":[{"index":0,"delta":'
            b'{"role":"assistant","content":"hi"},'
            b'"finish_reason":null}]}\n\n',
            b'data: [DONE]\n\n',
        ], "application/json")
        self.assertEqual(deltas, ["hi"])
        self.assertEqual(formats.item_text(items[0]), "hi")

    def test_bom_prefixed_stream_is_sniffed_as_sse(self):
        deltas, items = self._run_stream_turn([
            b'\xef\xbb\xbfdata: {"choices":[{"index":0,"delta":'
            b'{"content":"hi"},"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ], "application/json")
        self.assertEqual(deltas, ["hi"])
        self.assertEqual(formats.item_text(items[0]), "hi")

    def test_whitespace_only_prefix_waits_for_deciding_bytes(self):
        # Whitespace before "{" is legal JSON padding split across reads;
        # the sniff must not misroute it while ambiguous.
        response = {"id": "chat_1", "object": "chat.completion",
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": "buffered"},
                        "finish_reason": "stop"}]}
        _, items = self._run_stream_turn([
            b' ', b'  ', json.dumps(response).encode("utf-8")],
            "application/json")
        self.assertEqual(formats.item_text(items[0]), "buffered")

    def test_leading_whitespace_before_data_line_is_not_sse(self):
        # SSE field names may not be preceded by whitespace; the decoder
        # would drop that line silently. The sniff must classify as JSON so
        # the turn fails loudly instead.
        with self.assertRaises(protocols.StreamProtocolError):
            self._run_stream_turn([
                b'   data: {"choices":[{"index":0,"delta":'
                b'{"content":"hi"},"finish_reason":"stop"}]}\n\n',
            ], "application/json")

    def test_ambiguous_then_json_still_parses_as_json(self):
        # The lookahead stops as soon as bytes stop matching an SSE marker
        # prefix ("eve" stops matching at "e{"...), and a JSON document
        # split from its opening brace still parses as JSON.
        response = {"id": "chat_1", "object": "chat.completion",
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": "buffered"},
                        "finish_reason": "stop"}]}
        raw = json.dumps(response).encode("utf-8")
        _, items = self._run_stream_turn(
            [b' ', raw[:1], raw[1:]], "application/json")
        self.assertEqual(formats.item_text(items[0]), "buffered")

    def test_eof_while_ambiguous_yields_short_prefix(self):
        # Stream ends mid-marker: no hang, no crash; the sniffer falls back
        # to the content-type and the JSON path reports the parse error.
        with self.assertRaises(protocols.StreamProtocolError):
            self._run_stream_turn([b'da'], "application/json")

    def test_stream_body_kind_table(self):
        cases = [
            ("text/event-stream", b'data: {}\n\n', "sse"),
            ("text/event-stream", b'\n\ndata: {}\n\n', "sse"),
            ("text/event-stream", b': ping\n\ndata: {}', "sse"),
            ("application/json", b'{"error":"x"}', "json"),
            ("application/json", b'data: {}', "sse"),
            ("", b'\xef\xbb\xbfdata: {}', "sse"),
            ("application/json", b'\xef\xbb\xbfdata: {}', "sse"),
            # Whitespace before a field name is invalid SSE (the decoder
            # drops the line); JSON is the loud failure, not silent loss.
            ("application/json", b'   data: {}\n\n', "json"),
            ("text/event-stream", b'   data: {}\n\n', "sse"),
            ("", b'', "json"),
            ("", b'\n', "json"),
        ]
        for content_type, chunk, expected in cases:
            with self.subTest(chunk=chunk, content_type=content_type):
                self.assertEqual(
                    loki._stream_body_kind(content_type, chunk), expected)

    def test_transport_failure_after_first_event_is_not_retried(self):
        calls = []
        deltas = []

        async def response_body():
            yield (
                b'data: {"choices":[{"index":0,'
                b'"delta":{"content":"partial"}}]}\n\n')
            raise ConnectionResetError("reset")

        @contextlib.asynccontextmanager
        async def fake_http_stream(method, request_url, **kwargs):
            calls.append(request_url)
            yield loki.http_client.HttpStreamResponse(
                request_url,
                200,
                "OK",
                {"content-type": "text/event-stream"},
                response_body(),
            )

        with mock.patch(
                "loki_agent.loki.http_client.async_http_stream",
                side_effect=fake_http_stream):
            with self.assertRaisesRegex(
                    protocols.StreamProtocolError,
                    "after output began"):
                asyncio.run(loki.async_chat_completion(
                    [formats.message_item("user", "hello")],
                    tools=[],
                    on_text_delta=deltas.append,
                ))

        self.assertEqual(calls, [
            "http://localhost:8000/v1/chat/completions"])
        self.assertEqual(deltas, ["partial"])

    def test_cancel_closes_stream_without_final_response(self):
        cancelled = {"value": False}

        async def response_body():
            yield (
                b'data: {"choices":[{"index":0,'
                b'"delta":{"content":"partial"}}]}\n\n')
            await asyncio.sleep(60)

        @contextlib.asynccontextmanager
        async def fake_http_stream(method, request_url, **kwargs):
            yield loki.http_client.HttpStreamResponse(
                request_url,
                200,
                "OK",
                {"content-type": "text/event-stream"},
                response_body(),
            )

        def on_delta(text):
            cancelled["value"] = True

        with mock.patch(
                "loki_agent.loki.http_client.async_http_stream",
                side_effect=fake_http_stream):
            with self.assertRaises(loki.StreamCancelled):
                asyncio.run(loki.async_chat_completion(
                    [formats.message_item("user", "hello")],
                    tools=[],
                    on_text_delta=on_delta,
                    cancel_check=lambda: cancelled["value"],
                ))


class StreamingToolLoopTests(unittest.TestCase):
    def test_stderr_diagnostic_flushes_stdout_first(self):
        class TrackingStdout(io.StringIO):
            def __init__(self):
                super().__init__()
                self.was_flushed = False

            def flush(self):
                self.was_flushed = True
                super().flush()

        class OrderedStderr(io.StringIO):
            def __init__(self, stdout):
                super().__init__()
                self.stdout = stdout

            def write(self, value):
                if value and not self.stdout.was_flushed:
                    raise AssertionError(
                        "stderr was written before stdout was flushed")
                return super().write(value)

        stdout = TrackingStdout()
        stderr = OrderedStderr(stdout)

        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            terminal_frontend._terminal_agent_event({
                "type": "response_incomplete",
                "protocol_data": {"reason": "max_output_tokens"},
            })

        self.assertIn("model response incomplete", stderr.getvalue())

    def test_streamed_text_is_not_printed_again_or_duplicated_in_transcript(
            self):
        transcript = [formats.message_item("user", "hello")]
        events = []

        async def chat_fn(
                items, on_text_delta, *, codex_turn_state):
            on_text_delta("hel")
            on_text_delta("lo")
            return [formats.message_item("assistant", "hello")]

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            on_event=events.append,
            stream_chat=True,
        ))

        self.assertEqual(result, "hello")
        self.assertEqual(len(transcript), 2)
        self.assertEqual(formats.item_text(transcript[1]), "hello")
        self.assertEqual(
            [event["type"] for event in events],
            [
                "assistant_start",
                "assistant_delta",
                "assistant_delta",
                "assistant_end",
            ],
        )
        self.assertFalse(any(
            event["type"] == "assistant_message" for event in events))

    def test_partial_stream_error_is_not_invented_as_response(self):
        transcript = [formats.message_item("user", "hello")]
        events = []

        async def chat_fn(
                items, on_text_delta, *, codex_turn_state):
            on_text_delta("partial")
            raise protocols.StreamProtocolError("broken stream")

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            on_event=events.append,
            stream_chat=True,
        ))

        self.assertEqual(result, "")
        self.assertEqual(
            [event["type"] for event in transcript], ["message"])
        self.assertEqual(
            [event["type"] for event in events],
            [
                "assistant_start",
                "assistant_delta",
                "assistant_end",
                "stream_error",
            ],
        )
        self.assertFalse(events[2]["complete"])

    def test_partial_cancel_is_not_invented_as_response(self):
        transcript = [formats.message_item("user", "hello")]
        events = []

        async def chat_fn(
                items, on_text_delta, *, codex_turn_state):
            on_text_delta("partial")
            raise loki.StreamCancelled()

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            on_event=events.append,
            stream_chat=True,
        ))

        self.assertEqual(result, "")
        self.assertEqual(
            [event["type"] for event in transcript], ["message"])
        self.assertEqual(
            [event["type"] for event in events],
            [
                "assistant_start",
                "assistant_delta",
                "assistant_end",
                "response_cancelled",
            ],
        )


class ResponsesToolLoopTests(unittest.TestCase):
    def test_provider_notice_is_saved_and_emitted_but_not_model_input(self):
        transcript = [formats.message_item("user", "hello")]
        events = []
        turn = formats.DecodedTurn(
            [formats.message_item("assistant", "answer")],
            {
                "protocol": formats.OPENAI_RESPONSES,
                "protocol_data": {
                    "loki": {
                        "provider_notices": [
                            formats.TRUSTED_ACCESS_FOR_CYBER,
                        ],
                    },
                },
            },
        )

        async def chat_fn(items, *, codex_turn_state):
            return turn

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            on_event=events.append,
        ))

        self.assertEqual(result, "answer")
        self.assertEqual(
            [event["type"] for event in events],
            ["provider_notice", "assistant_message"],
        )
        self.assertEqual(
            formats.provider_notice_codes(transcript[1]),
            [formats.TRUSTED_ACCESS_FOR_CYBER],
        )
        _instructions, projected = (
            formats.items_to_openai_responses_parts(transcript))
        self.assertNotIn(
            formats.TRUSTED_ACCESS_FOR_CYBER,
            json.dumps(projected),
        )

    def test_terminal_provider_notice_is_not_labeled_as_assistant(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            terminal_frontend._terminal_agent_event({
                "type": "provider_notice",
                "code": formats.TRUSTED_ACCESS_FOR_CYBER,
            })

        rendered = output.getvalue()
        self.assertIn("Trusted Access", rendered)
        self.assertNotIn("Assistant:", rendered)

    def test_toolless_helper_receives_distinct_explicit_turn_state(self):
        outer_states = []
        inner_states = []

        async def completion(
                items, tools=None, *, codex_turn_state, **kwargs):
            inner_states.append(codex_turn_state)
            codex_turn_state.capture("inner-state")
            return formats.DecodedTurn([
                formats.message_item("assistant", "helper result"),
            ])

        async def outer_chat(items, *, codex_turn_state):
            outer_states.append(codex_turn_state)
            codex_turn_state.capture("outer-state")
            helper = await loki.run_toolless_completion_async([
                formats.message_item("user", "helper prompt"),
            ])
            return formats.DecodedTurn([
                formats.message_item("assistant", helper),
            ])

        with mock.patch.object(
                loki, "async_chat_completion", new=completion):
            result = asyncio.run(loki.run_tool_loop_async(
                [formats.message_item("user", "outer prompt")],
                chat_fn=outer_chat,
            ))

        self.assertEqual(result, "helper result")
        self.assertEqual(len(outer_states), 1)
        self.assertEqual(len(inner_states), 1)
        self.assertIsNot(inner_states[0], outer_states[0])
        self.assertEqual(inner_states[0].value, "inner-state")
        self.assertEqual(outer_states[0].value, "outer-state")

    def test_autonomous_loop_limit_is_hard_and_closes_pending_call(self):
        transcript = [formats.message_item("user", "keep calling")]
        events = []
        dispatched = []
        response_number = 0

        async def chat_fn(items, *, codex_turn_state):
            nonlocal response_number
            response_number += 1
            return formats.DecodedTurn([
                formats.tool_call_item(
                    f"call_{response_number}", "Read",
                    {"file_path": "README.md"}),
            ])

        async def fake_dispatch(fn_name, args, allowed=None,
                                extra_context=None):
            dispatched.append((fn_name, args))
            return {"ok": True, "content": "contents"}

        old_dispatch = loki.dispatch_tool_async
        try:
            loki.dispatch_tool_async = fake_dispatch
            result = asyncio.run(loki.run_tool_loop_async(
                transcript,
                chat_fn=chat_fn,
                on_event=events.append,
                max_loops=2,
            ))
        finally:
            loki.dispatch_tool_async = old_dispatch

        self.assertEqual(result, "")
        self.assertEqual(response_number, 2)
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(
            [item["type"] for item in transcript],
            [
                "message",
                "model_response", "tool_result",
                "model_response", "tool_result",
            ],
        )
        self.assertTrue(transcript[-1]["is_error"])
        self.assertIn("2-response autonomous loop limit",
                      formats.item_text(transcript[-1]))
        self.assertEqual(
            [event["type"] for event in events],
            ["tool_call", "tool_result", "max_loops"],
        )

    def test_empty_incomplete_response_is_a_real_response_event(self):
        transcript = [formats.message_item("user", "hello")]
        records = []
        events = []

        async def chat_fn(items, *, codex_turn_state):
            return formats.DecodedTurn(
                [],
                {
                    "protocol": "openai_responses",
                    "status": "incomplete",
                },
                complete=False,
            )

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            on_event=events.append,
            on_response=lambda turn, event: records.append(
                (turn, event)),
        ))

        self.assertEqual(result, "")
        self.assertEqual(len(transcript), 2)
        self.assertEqual(transcript[1]["type"], "model_response")
        self.assertEqual(transcript[1]["status"], "incomplete")
        self.assertEqual(transcript[1]["items"], [])
        self.assertEqual(len(records), 1)
        self.assertIs(records[0][1], transcript[1])
        self.assertEqual(
            [event["type"] for event in events],
            ["response_incomplete"],
        )

    def test_failed_response_is_saved_and_reported_as_failed(self):
        transcript = [formats.message_item("user", "hello")]
        events = []

        async def chat_fn(items, *, codex_turn_state):
            return formats.DecodedTurn(
                [],
                {
                    "protocol": formats.OPENAI_RESPONSES,
                    "status": "failed",
                    "protocol_data": {
                        formats.OPENAI_RESPONSES: {
                            "error": {
                                "code": "server_error",
                                "message": "failed",
                            },
                        },
                    },
                },
                complete=False,
            )

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            on_event=events.append,
        ))

        self.assertEqual(result, "")
        self.assertEqual(transcript[1]["status"], "failed")
        self.assertEqual(
            [event["type"] for event in events],
            ["response_failed"],
        )
        self.assertEqual(
            events[0]["protocol_data"][formats.OPENAI_RESPONSES]
            ["error"]["code"],
            "server_error",
        )

    def test_incomplete_function_call_is_closed_before_next_user_turn(self):
        transcript = [formats.message_item("user", "read it")]
        events = []

        async def chat_fn(items, *, codex_turn_state):
            return formats.DecodedTurn(
                [formats.tool_call_item(
                    "call_incomplete",
                    "Read",
                    raw_arguments='{"file_path":',
                    parse_error="incomplete JSON",
                    status="incomplete",
                )],
                {
                    "protocol": formats.OPENAI_RESPONSES,
                    "status": "incomplete",
                    "protocol_data": {
                        formats.OPENAI_RESPONSES: {
                            "incomplete_details": {
                                "reason": "max_output_tokens",
                            },
                        },
                    },
                },
                complete=False,
            )

        asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            on_event=events.append,
        ))

        self.assertEqual(
            [item["type"] for item in transcript],
            ["message", "model_response", "tool_result"],
        )
        self.assertTrue(transcript[-1]["is_error"])
        self.assertEqual(formats.pending_tool_calls(transcript), [])
        transcript.append(formats.message_item("user", "continue"))
        chat = formats.items_to_openai_chat_messages(transcript)
        self.assertEqual(
            [message["role"] for message in chat],
            ["user", "assistant", "tool", "user"],
        )
        self.assertEqual(
            [event["type"] for event in events],
            ["response_incomplete"],
        )

    def test_anthropic_pause_turn_continues_without_synthetic_event(self):
        transcript = [formats.message_item("user", "search")]
        requests = []

        async def chat_fn(items, *, codex_turn_state):
            requests.append(copy.deepcopy(items))
            if len(requests) == 1:
                return formats.DecodedTurn(
                    [formats.tool_call_item(
                        "srvtoolu_1",
                        "web_search",
                        {"query": "current news"},
                        execution="provider",
                        protocol_data={
                            formats.ANTHROPIC_MESSAGES: {
                                "native_type": "server_tool_use",
                                "id": "srvtoolu_1",
                            },
                        },
                    )],
                    {
                        "protocol": formats.ANTHROPIC_MESSAGES,
                        "stop_reason": "pause_turn",
                    },
                )
            return formats.DecodedTurn(
                [formats.message_item("assistant", "finished")],
                {
                    "protocol": formats.ANTHROPIC_MESSAGES,
                    "stop_reason": "end_turn",
                },
            )

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            max_loops=3,
        ))

        self.assertEqual(result, "finished")
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            [item["type"] for item in requests[1]],
            ["message", "model_response"],
        )
        self.assertEqual(
            [item["type"] for item in transcript],
            ["message", "model_response", "model_response"],
        )


class HarnessProjectionTests(unittest.TestCase):
    def test_allowed_subset_advertisement_and_real_execution_enforcement(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / 'reviewed.txt'
            target.write_text('read-only sentinel', encoding='utf-8')
            session = loki.Session(shell_cwd=directory)
            requests = []

            async def completion(items, tools, model=None, *, codex_turn_state,
                                 reasoning_effort=None, thinking=None):
                self.assertEqual({tool['function']['name'] for tool in tools},
                                 {'Read', 'Grep'})
                requests.append(copy.deepcopy(items))
                if len(requests) == 1:
                    # Read first: absent the allowed-set gate, the subsequent
                    # Write would be authorized and would really clobber it.
                    return formats.DecodedTurn([
                        formats.tool_call_item('read', 'Read',
                                               {'file_path': str(target)}),
                        formats.tool_call_item('forbidden', 'Write', {
                            'file_path': str(target), 'content': 'clobbered'}),
                    ])
                self.assertEqual(len(requests), 2)
                results = [item for item in items
                           if item.get('type') == 'tool_result']
                self.assertEqual([item['call_id'] for item in results],
                                 ['read', 'forbidden'])
                self.assertFalse(results[0]['is_error'])
                self.assertIn('read-only sentinel',
                              results[0]['content'][0]['text'])
                self.assertTrue(results[1]['is_error'])
                self.assertIn('Tool Write not available in this subagent',
                              results[1]['content'][0]['text'])
                return formats.DecodedTurn([
                    formats.message_item('assistant', 'inspected without changes')])

            transcript = [formats.message_item('user', 'inspect')]
            with mock.patch.object(loki, '_DEFAULT_SESSION', session), \
                    mock.patch.object(loki, 'file_state', {}), \
                    mock.patch.object(loki, 'async_chat_completion', completion):
                result = asyncio.run(loki.run_tool_loop_async(
                    transcript, allowed={'Read', 'Grep'}))
            self.assertEqual(result, 'inspected without changes')
            self.assertEqual(len(requests), 2)
            self.assertEqual(target.read_text(encoding='utf-8'),
                             'read-only sentinel')
            self.assertEqual([item['call_id'] for item in transcript
                              if item.get('type') == 'tool_result'],
                             ['read', 'forbidden'])

    def test_toolless_completion_returns_all_assistant_phases(self):
        async def fake_completion(
                items, tools, *, codex_turn_state,
                reasoning_effort=None, thinking=None):
            self.assertEqual(tools, [])
            return formats.DecodedTurn([
                formats.message_item("assistant", "commentary"),
                formats.message_item("assistant", "final"),
            ])

        old_completion = loki.async_chat_completion
        try:
            loki.async_chat_completion = fake_completion
            result = asyncio.run(loki.run_toolless_completion_async(
                [formats.message_item("user", "hello")]))
        finally:
            loki.async_chat_completion = old_completion

        self.assertEqual(result, "commentary\nfinal")

    def test_failed_provider_call_creates_no_transcript_ghost(self):
        transcript = [formats.message_item("user", "hello")]
        records = []
        body = {"error": {"message": "full marker"}}

        async def chat_fn(items, *, codex_turn_state):
            raise loki.ApiError(
                "https://provider.test/v1/responses",
                429,
                "Too Many Requests",
                json.dumps(body),
            )

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            on_response=lambda turn, event: records.append(
                (turn, event)),
        ))

        self.assertEqual(result, "")
        self.assertEqual(
            [item["type"] for item in transcript], ["message"])
        self.assertEqual(records, [])


class QuestionGuardTests(unittest.TestCase):
    """The question guard: when the user's turn asks a question, the
    agent answers -- state-changing tools are refused for that whole
    turn.

    The guard was added in ed8c342 and its trigger silently dropped in
    1a0c19a. These tests pin the trigger, the gate, and the explore
    tool set so none of them can change silently again.
    """

    def setUp(self):
        self._state = save_loki_state(
            ["agent_mode", "last_instructed_agent_mode", "job_manager"])

    def tearDown(self):
        restore_loki_state(self._state)

    def _context_for(self, text):
        items = [formats.message_item("user", text)]
        return loki.get_tool_loop_extra_context(items)

    def test_question_trigger_boundaries(self):
        # The current trigger is a trailing question mark or literal 'what?'
        # anywhere (case-insensitive), not every occurrence of 'what' or '?'.
        for text, inhibited in (
                ('why does the build fail?', True),
                ('that is odd, WHAT? exactly fails here', True),
                ('what is broken', False),
                ('does it fail? fix it now', False),
                ('please fix the failing tests', False)):
            with self.subTest(text=text):
                self.assertEqual(self._context_for(text)['inhibit_edits'],
                                 "answering the user's question"
                                 if inhibited else False)

    def test_only_a_trailing_user_message_counts(self):
        items = [
            formats.message_item("user", "what is this?"),
            formats.message_item("assistant", "an answer"),
        ]
        self.assertFalse(
            loki.get_tool_loop_extra_context(items)["inhibit_edits"])

    def test_guard_spans_whole_turn_and_resets_next_turn(self):
        # The guard is decided once at turn start and binds the whole
        # turn: a tool call issued after the model already produced
        # text is still refused. The next user turn decides anew.
        transcript = [formats.message_item(
            "user", "why does the build fail?")]
        first_calls = [
            [formats.message_item("assistant", "let me check"),
             formats.tool_call_item("t1", "Bash", {"command": "echo ok"})],
            [formats.message_item("assistant", "because dependencies")],
        ]

        async def scripted_chat(items, *, codex_turn_state):
            return first_calls.pop(0)

        asyncio.run(loki.run_tool_loop_async(
            transcript, chat_fn=scripted_chat))
        # The Bash call in this question turn must have been refused --
        # the refusal reason can only enter the transcript that way.
        self.assertIn("answering the user's question", str(transcript))

        transcript.append(formats.message_item("user", "fix it now"))
        second_calls = [
            [formats.tool_call_item("t2", "Bash", {"command": "echo ok"})],
            [formats.message_item("assistant", "done")],
        ]

        async def scripted_chat2(items, *, codex_turn_state):
            return second_calls.pop(0)

        before = len(transcript)
        asyncio.run(loki.run_tool_loop_async(
            transcript, chat_fn=scripted_chat2))
        self.assertNotIn(
            "answering the user's question", str(transcript[before:]))
        self.assertIn("ok", str(transcript[before:]))

    def test_explore_and_plan_modes_inhibit(self):
        for mode in ["explore", "plan"]:
            with self.subTest(mode=mode):
                loki.current_session().agent_mode = mode
                self.assertEqual(
                    loki.get_tool_loop_extra_context(
                        [formats.message_item("user", "hi")])
                    ["inhibit_edits"],
                    f"{mode} mode")

    def test_question_refuses_changes_to_disposable_reviewed_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = root / 'reviewed.txt'
            target.write_text('preserve these bytes', encoding='utf-8')
            manager = loki.JobManager(str(root / 'jobs'))
            session = loki.Session(shell_cwd=directory, job_manager=manager)
            quote = (subprocess.list2cmdline([str(target)]) if os.name == 'nt'
                     else shlex.quote(str(target)))
            with mock.patch.object(loki, '_DEFAULT_SESSION', session), \
                    mock.patch.object(loki, 'file_state', {}):
                extra = self._context_for('what does run_edit do?')
                read = asyncio.run(loki.dispatch_tool_async(
                    'Read', {'file_path': str(target)}, extra_context=extra))
                self.assertTrue(read['ok'])
                self.assertIn('preserve these bytes', read['content'])
                # Read-before-write is satisfied, so it cannot accidentally
                # supply the refusal if the question guard regresses.
                for name, args in (
                        ('Edit', {'file_path': str(target),
                                  'old_string': 'preserve',
                                  'new_string': 'clobber'}),
                        ('Write', {'file_path': str(target),
                                   'content': 'clobbered'}),
                        ('Bash', {'command': f'echo clobbered > {quote}'})):
                    with self.subTest(tool=name):
                        result = asyncio.run(loki.dispatch_tool_async(
                            name, args, extra_context=extra))
                        self.assertFalse(result['ok'])
                        self.assertIn("answering the user's question",
                                      result['content'])
                        self.assertEqual(target.read_text(encoding='utf-8'),
                                         'preserve these bytes')
                        self.assertEqual(set(root.iterdir()), {target})
                        self.assertEqual(manager.jobs, {})

    def test_refused_bash_grep_recommends_grep_tool(self):
        command = {"command": "grep -rn foo ."}
        message = loki._tool_access_error(
            "Bash",
            args=command,
            extra_context=self._context_for("what does run_edit do?"))
        self.assertIn("answering the user's question", message)
        hint = loki._refused_tool_hint("Bash", command, None)
        self.assertIsNotNone(hint)
        self.assertIn(hint, message)

    def test_refused_bash_without_grep_gets_no_hint(self):
        command = {"command": "echo hi"}
        self.assertIsNone(loki._refused_tool_hint("Bash", command, None))
        message = loki._tool_access_error(
            "Bash",
            args=command,
            extra_context=self._context_for("what does run_edit do?"))
        self.assertIn("answering the user's question", message)
        self.assertNotIn("grep", message.lower())

    def test_grep_hint_respects_subagent_allowed_set(self):
        command = {"command": "grep -rn foo ."}
        allowed = {"Bash", "Read"}
        self.assertIsNone(
            loki._refused_tool_hint("Bash", command, allowed))
        message = loki._tool_access_error(
            "Bash",
            args=command,
            allowed=allowed,
            extra_context=self._context_for("what does run_edit do?"))
        self.assertIn("answering the user's question", message)
        self.assertNotIn("grep", message.lower())

    def test_question_refusal_surfaces_grep_hint_in_transcript(self):
        # Goes through execute_tool_call_async, not dispatch_tool_async:
        # this is the path the real loop takes, and where the hint has to
        # appear for the model (and the terminal) to see it.
        transcript = [formats.message_item(
            "user", "what does this grep do?")]
        replies = [
            [formats.message_item("assistant", "let me look"),
             formats.tool_call_item(
                 "t1", "Bash", {"command": "grep -rn foo ."})],
            [formats.message_item("assistant", "done")],
        ]

        async def scripted_chat(items, *, codex_turn_state):
            return replies.pop(0)

        asyncio.run(loki.run_tool_loop_async(
            transcript, chat_fn=scripted_chat))
        self.assertIn("answering the user's question", str(transcript))
        hint = loki._refused_tool_hint(
            "Bash", {"command": "grep -rn foo ."}, None)
        self.assertIsNotNone(hint)
        self.assertIn(hint, str(transcript))

    def test_explore_tools_registry_shape_is_pinned(self):
        # Any change to the explore-allowed set must show up here,
        # deliberately, as a diff to this expected set.
        self.assertEqual(
            loki.EXPLORE_TOOLS,
            {"Agent", "Read", "Glob", "Grep", "Jobs", "JobStatus", "TodoRead",
             "WebFetch", "WebSearch"})

    def test_plan_tools_registry_shape_is_pinned(self):
        # Plan mode additionally allows TodoWrite (session-scoped plan
        # state) and Ask (clarifying questions asked of the user before
        # the plan is final); anything else landing here must be a
        # deliberate diff.
        self.assertEqual(
            loki.PLAN_TOOLS,
            loki.EXPLORE_TOOLS | {"TodoWrite", "Ask"})

    def test_plan_mode_allows_todowrite_but_blocks_workspace_and_system(self):
        loki.current_session().agent_mode = "plan"
        context = loki.get_tool_loop_extra_context(
            [formats.message_item("user", "plan the refactor")])
        for name in ["TodoWrite", "Read", "Grep"]:
            with self.subTest(tool=name):
                self.assertIsNone(
                    loki._tool_access_error(name, extra_context=context))
        for name in ["Edit", "Write", "Bash", "Skill", "JobStop"]:
            with self.subTest(tool=name):
                self.assertIsNotNone(
                    loki._tool_access_error(name, extra_context=context))

    def test_explore_mode_and_questions_still_block_todowrite(self):
        loki.current_session().agent_mode = "explore"
        explore_context = loki.get_tool_loop_extra_context(
            [formats.message_item("user", "explore this")])
        self.assertIsNone(
            loki._tool_access_error(
                "Agent",
                allowed=loki.EXPLORE_TOOLS,
                extra_context=explore_context,
            ))
        self.assertIsNotNone(
            loki._tool_access_error(
                "TodoWrite", extra_context=explore_context))
        loki.current_session().agent_mode = "normal"
        question_context = loki.get_tool_loop_extra_context(
            [formats.message_item("user", "what about todos?")])
        self.assertIsNotNone(
            loki._tool_access_error(
                "TodoWrite", extra_context=question_context))

    def test_agent_description_matches_explore_tools(self):
        description = next(
            spec["function"]["description"] for spec in loki.TOOLS
            if spec["function"]["name"] == "Agent")
        marker = "(Tools: "
        start = description.find(marker)
        self.assertGreaterEqual(start, 0, "no tool list in description")
        end = description.find(")", start)
        advertised = {
            part.strip()
            for part in description[start + len(marker):end].split(",")}
        self.assertEqual(advertised, loki.EXPLORE_TOOLS)


class ThinkingControlsTests(unittest.TestCase):
    def setUp(self):
        self.session = loki.Session()
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)
        self.session.runtime_config = loki.make_runtime_config(
            "https://api.anthropic.com/v1", protocols.ANTHROPIC_MESSAGES,
            model="claude-opus-4-5", provider_id="anthropic",
            reasoning_effort_profile=modelsdev.ReasoningEffortProfile(
                ["low", "high"]))

    def test_command_parser_is_atomic_and_fields_are_order_free(self):
        for argument in [
                "effort nope",
                "budget zero", "mode manual mode off", "effort high mode",
                "effort high mode invented"]:
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                loki.thinking_command(argument)
            self.assertIsNone(self.session.thinking_mode)
            self.assertIsNone(self.session.thinking_budget)
            self.assertIsNone(self.session.reasoning_effort_preference)
        # A manual budget below the documented floor is a joint-state
        # error: assignment accepts it, projection raises once.
        loki.thinking_command("mode manual budget 1023")
        self.assertEqual(
            (self.session.thinking_mode, self.session.thinking_budget),
            ("manual", 1023))
        with self.assertRaises(ValueError):
            loki.capture_turn_settings()
        loki.thinking_command("budget default mode default")
        # Manual without a budget is a legal, order-free assignment; the
        # allowance pairs with it from any order and projection carries it.
        loki.thinking_command("mode manual")
        self.assertEqual(self.session.thinking_mode, "manual")
        self.assertIsNone(self.session.thinking_budget)
        loki.thinking_command("budget 2048 effort high")
        self.assertEqual(
            (self.session.thinking_mode, self.session.thinking_budget),
            ("manual", 2048))
        self.assertEqual(loki.effective_reasoning_effort(), "high")

    def test_unlisted_model_ride_the_normal_path(self):
        # An unlisted model has a verified spelling; acceptance is the
        # endpoint's to answer, in the open. No trial state exists.
        loki.reinstall_provider(model="claude-future")
        text = loki.thinking_command("mode adaptive")
        self.assertNotIn("Unverified model acceptance", text)
        self.assertEqual(self.session.thinking_mode, "adaptive")
        captured = loki.capture_turn_settings()
        self.assertEqual(captured.mode, "adaptive")
        self.assertFalse(hasattr(captured, "trial"))

    def test_dormant_preferences_survive_and_restore(self):
        loki.thinking_command("mode manual budget 2048")
        loki.thinking_command("mode default")
        self.assertEqual(self.session.thinking_budget, 2048)
        self.assertEqual(
            loki.effective_thinking_settings()[:2], (None, None))
        self.assertIn(
            "Inactive budget preference: 2048", loki.thinking_status_text())
        loki.thinking_command("mode manual")
        self.assertEqual(
            loki.effective_thinking_settings()[:2], ("manual", 2048))

    def test_controls_round_trip_and_saved_state_is_validated(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "chat.json")
            loki.new_chat_log(path)
            loki.thinking_command("mode manual budget 2048 effort high")
            loki.trace_command("thinking on")
            state = dict(loki.current_session().session_state)
            loki.new_chat_log(path)
            loki.load_session_state(state)
            restored = loki.current_session()
            self.assertEqual(
                (restored.thinking_mode, restored.thinking_budget),
                ("manual", 2048))
            self.assertEqual(restored.reasoning_traces, "on")
            self.assertEqual(restored.reasoning_effort_preference, "high")
        for bad in [{"thinking_mode": "bogus"}, {"thinking_budget": True},
                    {"thinking_traces": "loud"},
                    {"reasoning_retention": "keep"}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                loki.load_session_state(bad)

    def test_snapshot_governs_the_whole_turn(self):
        loki.thinking_command("mode manual budget 2048")
        seen = []

        async def completion(items, tools, *, codex_turn_state,
                             reasoning_effort=None, thinking=None):
            seen.append(thinking)
            if len(seen) == 1:
                # A mid-turn preference change must not affect this turn.
                self.session.thinking_mode = None
                return formats.DecodedTurn([formats.tool_call_item(
                    "call", "Read", {"file_path": "x"})])
            return formats.DecodedTurn(
                [formats.message_item("assistant", "done")])

        execute = mock.AsyncMock(return_value=({"ok": True, "content": "ok"}, {}))
        with mock.patch.object(loki, "async_chat_completion", completion), \
                mock.patch.object(loki, "execute_tool_call_async", execute):
            asyncio.run(loki.run_tool_loop_async(
                [formats.message_item("user", "go")]))
        self.assertEqual([settings.mode for settings in seen],
                         ["manual", "manual"])


if __name__ == "__main__":
    unittest.main()
