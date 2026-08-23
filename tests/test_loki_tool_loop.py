import asyncio
import base64
import copy
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("LOKI_API_KEY", "test-key")
os.environ.setdefault("LOKI_API_BASE", "https://api.openai.com/v1/responses")
os.environ.setdefault("LOKI_PROVIDER", "openai_responses")

from loki_agent import formats
from loki_agent import loki
from loki_agent import terminal_frontend
from loki_agent import models as modelsdev
from loki_agent import protocols
from loki_agent.connections import ConnectionDescriptor
from loki_agent.credentials import CredentialStore
from loki_agent import savefiles
from loki_agent import terminals



_MISSING = object()


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
        self.reader = types.SimpleNamespace(
            cancel_requested=False,
            cancel_event=mock.Mock(),
        )

    async def get(self):
        return self.messages.pop(0)

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

        self.assertEqual(image.path, os.path.realpath(path))
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
                    terminal_frontend._image_media_type(data), expected)

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
            with self.assertRaisesRegex(
                    terminal_frontend.ImageAttachmentError,
                    "not a regular file"):
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
        with mock.patch(
                "loki_agent.terminal_frontend.input_session",
                return_value=session), mock.patch(
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
        self.assertIn("queued_images=1", status_updates[0])
        self.assertIn("queued_images=0", status_updates[1])

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

        self.assertEqual(status, 0)
        self.assertEqual(observed, [True])
        self.assertFalse(
            terminal_frontend._terminal_activity.turn_running)

    def test_slash_commands_do_not_start_a_turn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, captured, _stdout, _stderr = self._run_terminal(
                ["/pwd", "/ps", "/quit"],
                tmpdir,
            )

        self.assertEqual(status, 0)
        self.assertEqual(captured, [])
        self.assertFalse(
            terminal_frontend._terminal_activity.turn_running)

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

        self.assertEqual(observed, [True])
        self.assertFalse(
            terminal_frontend._terminal_activity.turn_running)


class ProviderReinstallTests(unittest.TestCase):
    def test_make_runtime_config_builds_provider_and_headers(self):
        config = loki.make_runtime_config(
            "https://api.example.test/v1/messages",
            protocols.ANTHROPIC_MESSAGES,
            "secret-key",
            model="model-a",
            max_tokens=1234,
            anthropic_version="2024-01-01",
            auth_header="X-Custom-Auth",
        )

        self.assertEqual(config.url, "https://api.example.test/v1/messages")
        self.assertEqual(config.provider_kind, protocols.ANTHROPIC_MESSAGES)
        self.assertEqual(config.netloc, "api.example.test")
        self.assertEqual(config.api_key, "secret-key")
        self.assertEqual(config.model, "model-a")
        self.assertEqual(config.chat_provider.max_tokens, 1234)
        self.assertEqual(config.chat_provider.kind, protocols.ANTHROPIC_MESSAGES)
        self.assertEqual(config.headers["X-Custom-Auth"], "secret-key")
        self.assertEqual(config.anthropic_version, "2024-01-01")
        self.assertEqual(config.auth_header, "X-Custom-Auth")

    def test_reinstall_provider_swaps_provider_preserving_settings(self):
        env = {
            "LOKI_API_BASE": "https://example.test/v1/responses",
            "LOKI_PROVIDER": "openai_responses",
            "LOKI_API_KEY": "test-key",
            "LOKI_MODEL": "model-a",
            "LOKI_MAX_TOKENS": "512",
        }
        names = ["runtime_config"]
        old_values = save_loki_state(names)

        try:
            loki.apply_runtime_config(loki.build_config_from_env(env))
            old_provider = loki.current_config().chat_provider

            loki.reinstall_provider(model="model-b")

            self.assertEqual(loki.current_model(), "model-b")
            self.assertEqual(loki.current_config().model, "model-b")
            # A fresh Provider object was built and swapped in.
            self.assertIsNot(loki.current_config().chat_provider, old_provider)
            # Everything else carries over from the previous config.
            self.assertEqual(loki.current_config().chat_provider.kind, protocols.OPENAI_RESPONSES)
            self.assertEqual(loki.current_config().chat_provider.chat_url, "https://example.test/v1/responses")
            self.assertEqual(loki.current_config().chat_provider.max_tokens, 512)
            self.assertEqual(loki.current_config().api_key, "test-key")
            self.assertEqual(loki.current_config().headers["Authorization"], "Bearer test-key")
            # The copied headers field was rebuilt too, not left stale.
            self.assertEqual(loki.current_config().headers,
                             loki.current_config().chat_provider.headers)
        finally:
            restore_loki_state(old_values)

    def test_reinstall_provider_switches_protocol_per_model(self):
        env = {
            "LOKI_API_BASE": "https://example.test/v1/responses",
            "LOKI_PROVIDER": "openai_responses",
            "LOKI_API_KEY": "test-key",
            "LOKI_MODEL": "model-a",
        }
        names = ["runtime_config"]
        old_values = save_loki_state(names)

        try:
            loki.apply_runtime_config(loki.build_config_from_env(env))

            # A future models.dev record maps this model to a different
            # provider + protocol; reinstall must rebuild Provider/headers.
            loki.reinstall_provider(
                model="claude-model",
                url="https://anthropic.example.test",
                provider_kind=protocols.ANTHROPIC_MESSAGES,
                api_key="anthropic-key",
            )

            self.assertEqual(loki.current_model(), "claude-model")
            provider = loki.current_config().chat_provider
            self.assertEqual(provider.kind, protocols.ANTHROPIC_MESSAGES)
            self.assertEqual(provider.chat_url, "https://anthropic.example.test/v1/messages")
            self.assertEqual(provider.headers["x-api-key"], "anthropic-key")
            self.assertEqual(loki.current_config().headers["x-api-key"], "anthropic-key")
            self.assertEqual(loki.current_config().provider_kind, protocols.ANTHROPIC_MESSAGES)
            self.assertEqual(loki.current_config().api_key, "anthropic-key")
        finally:
            restore_loki_state(old_values)

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
                "test-key",
                model="old-model",
                provider_id="provider",
                credential_env="PROVIDER_API_KEY",
                model_status="deprecated",
            ))

            loki.reinstall_provider(model="old-model")
            self.assertEqual(
                loki.current_config().model_status, "deprecated")

            loki.reinstall_provider(model="new-model")
            self.assertIsNone(loki.current_config().model_status)
        finally:
            restore_loki_state(old_values)


class RuntimeConfigTests(unittest.TestCase):
    def test_build_config_uses_explicit_env_key(self):
        env = {
            "LOKI_API_BASE": "https://api.deepseek.com/anthropic",
            "LOKI_PROVIDER": "anthropic_messages",
            "LOKI_API_KEY": "loki-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "OPENAI_API_KEY": "openai-key",
            "LOKI_MODEL": "deepseek-test",
            "LOKI_MAX_TOKENS": "123",
            "LOKI_ANTHROPIC_VERSION": "2024-01-01",
        }

        config = loki.build_config_from_env(env)

        self.assertEqual(config.url, "https://api.deepseek.com/anthropic")
        self.assertEqual(config.provider_kind, protocols.ANTHROPIC_MESSAGES)
        self.assertEqual(config.netloc, "api.deepseek.com")
        self.assertEqual(config.api_key, "loki-key")
        self.assertEqual(config.model, "deepseek-test")
        self.assertEqual(config.chat_provider.max_tokens, 123)
        self.assertEqual(config.headers["x-api-key"], "loki-key")
        self.assertEqual(config.headers["anthropic-version"], "2024-01-01")
        self.assertEqual(config.anthropic_version, "2024-01-01")
        self.assertEqual(config.credential_env, "LOKI_API_KEY")
        self.assertNotIn("LOKI_API_KEY", env)
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("OPENAI_API_KEY", env)

    def test_build_config_accepts_explicit_connection_without_authentication(
            self):
        env = {
            "LOKI_API_BASE": "https://example.test/v1/chat/completions",
            "LOKI_PROVIDER": "openai_chat",
            "LOKI_MODEL": "local-model",
        }
        config = loki.build_config_from_env(env)

        self.assertEqual(config.api_key, "")
        self.assertIsNone(config.credential_env)
        self.assertEqual(config.model, "local-model")
        self.assertNotIn("Authorization", config.headers)
        self.assertNotIn("x-api-key", config.headers)
        self.assertFalse(config.stream)

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

        self.assertTrue(direct.prompt_cache)
        self.assertFalse(compatible.prompt_cache)
        self.assertTrue(opted_in.prompt_cache)
        with self.assertRaisesRegex(ValueError, "LOKI_PROMPT_CACHE must be"):
            loki.build_config_from_env({
                "LOKI_API_BASE":
                    "https://compatible.example/v1/messages",
                "LOKI_PROVIDER": "anthropic_messages",
                "LOKI_MODEL": "compatible-test",
                "LOKI_PROMPT_CACHE": "sometimes",
            })

    def test_no_builtin_connection_exists(self):
        for credential_name in (
                "OPENCODE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            with self.subTest(credential_name=credential_name):
                env = {credential_name: "provider-key"}
                with self.assertRaisesRegex(
                        ValueError,
                        "API endpoint missing"):
                    loki.build_config_from_env(env)

    def test_unrelated_sdk_base_variables_do_not_configure_loki(self):
        for base_name in ("OPENAI_API_BASE", "ANTHROPIC_BASE_URL"):
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
        for credential_name in (
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENCODE_API_KEY"):
            with self.subTest(credential_name=credential_name):
                env = {
                    "LOKI_API_BASE":
                        "https://custom.example.test/v1/chat/completions",
                    credential_name: "must-not-be-sent",
                }
                config = loki.build_config_from_env(env)

                self.assertEqual(config.api_key, "")
                self.assertIsNone(config.credential_env)
                self.assertNotIn("Authorization", config.headers)
                self.assertNotIn("x-api-key", config.headers)

    def test_apply_runtime_config_assigns_runtime_globals(self):
        env = {
            "LOKI_API_BASE": "https://example.test/v1/responses",
            "LOKI_PROVIDER": "openai_responses",
            "LOKI_API_KEY": "test-key",
            "LOKI_MODEL": "gpt-test",
        }
        config = loki.build_config_from_env(env)
        names = ["runtime_config"]
        old_values = save_loki_state(names)

        try:
            loki.apply_runtime_config(config)

            self.assertIs(loki.current_config(), config)
            self.assertEqual(loki.current_model(), "gpt-test")
        finally:
            restore_loki_state(old_values)

    def test_saved_connection_requires_its_exact_credential(self):
        descriptor = ConnectionDescriptor(
            provider_id="openrouter",
            provider_name="OpenRouter",
            model="z-ai/glm",
            chat_url="https://openrouter.ai/api/v1/chat/completions",
            models_url="https://openrouter.ai/api/v1/models",
            protocol=protocols.OPENAI_CHAT,
            credential_env="OPENROUTER_API_KEY",
            model_status="deprecated",
        )
        with self.assertRaisesRegex(
                ValueError, "missing OPENROUTER_API_KEY"):
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
        self.assertEqual(config.url, descriptor.chat_url)
        self.assertEqual(config.api_key, "right-key")
        self.assertEqual(config.model, "override-model")
        self.assertIsNone(config.model_status)
        self.assertEqual(config.provider_id, "openrouter")
        self.assertEqual(config.credential_env, "OPENROUTER_API_KEY")

        restored = loki.config_from_connection_descriptor(
            descriptor,
            CredentialStore({"OPENROUTER_API_KEY": "right-key"}),
        )
        self.assertEqual(restored.model, "z-ai/glm")
        self.assertEqual(restored.model_status, "deprecated")

    def test_saved_credentialless_connection_restores_without_authentication(
            self):
        descriptor = ConnectionDescriptor(
            provider_id=None,
            provider_name="Explicit LOKI_* connection",
            model="local-model",
            chat_url="http://localhost:8000/v1/chat/completions",
            models_url="http://localhost:8000/v1/models",
            protocol=protocols.OPENAI_CHAT,
            credential_env=None,
            stream=True,
        )

        config = loki.config_from_connection_descriptor(
            descriptor, CredentialStore({}))

        self.assertEqual(config.api_key, "")
        self.assertIsNone(config.credential_env)
        self.assertNotIn("Authorization", config.headers)
        self.assertNotIn("x-api-key", config.headers)
        self.assertTrue(config.stream)

    def test_saved_prompt_cache_setting_restores_without_reinference(self):
        descriptor = ConnectionDescriptor(
            provider_id="compatible",
            provider_name="Compatible",
            model="compatible-test",
            chat_url="https://compatible.example/v1/messages",
            models_url="https://compatible.example/v1/models",
            protocol=protocols.ANTHROPIC_MESSAGES,
            credential_env=None,
            prompt_cache=True,
        )

        restored = loki.config_from_connection_descriptor(
            descriptor, CredentialStore({}))
        overridden = loki.config_from_connection_descriptor(
            descriptor, CredentialStore({"LOKI_PROMPT_CACHE": "0"}))

        self.assertTrue(restored.prompt_cache)
        self.assertFalse(overridden.prompt_cache)

    def test_modelsdev_selection_builds_fresh_provider_auth(self):
        provider_entry = {
            "name": "OpenRouter",
            "env": ["OPENROUTER_API_KEY"],
            "api": "https://openrouter.ai/api/v1",
        }
        config = loki.config_from_modelsdev_selection(
            "openrouter",
            provider_entry,
            {"id": "z-ai/glm", "name": "GLM",
             "status": "deprecated"},
            CredentialStore({
                "OPENROUTER_API_KEY": "selected-key",
                "LOKI_API_KEY": "old-key",
                "LOKI_STREAM": "1",
            }),
        )
        self.assertEqual(config.api_key, "selected-key")
        self.assertEqual(
            config.headers["Authorization"], "Bearer selected-key")
        self.assertEqual(config.auth_header, None)
        self.assertEqual(config.provider_id, "openrouter")
        self.assertEqual(config.model, "z-ai/glm")
        self.assertEqual(config.model_status, "deprecated")
        self.assertTrue(config.stream)


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        names = [
            "runtime_config", "CREDENTIALS", "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
        ]
        self.old_values = save_loki_state(names)

    def tearDown(self):
        for name, value in self.old_values.items():
            loki.current_session().__dict__[name] = value

    def test_provider_model_discovery_does_not_select_a_model(self):
        loki.apply_runtime_config(loki.make_runtime_config(
            "https://provider.example.test/v1/chat/completions",
            protocols.OPENAI_CHAT,
            "test-key",
            model="",
            models_url="https://provider.example.test/v1/models",
            credential_env="LOKI_API_KEY",
        ))
        response = {
            "data": [
                {"id": "first-model"},
                {"id": "second-model"},
            ],
        }

        with mock.patch(
                "loki_agent.loki.async_chat_request",
                new=mock.AsyncMock(return_value=response)):
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
        self.assertEqual(loki.current_config().api_key, "")
        self.assertNotIn("Authorization", loki.current_config().headers)

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
                    "loki_agent.terminal_frontend.run_subagent_cli_async",
                    new=runner), contextlib.redirect_stderr(stderr):
            status = asyncio.run(terminal_frontend.async_main(["--headless"]))

        loader.assert_not_awaited()
        runner.assert_not_awaited()
        self.assertEqual(status, 2)
        self.assertIn(
            "Configuration error: model missing; set LOKI_MODEL.",
            stderr.getvalue(),
        )

    def test_headless_startup_accepts_credentialless_explicit_connection(self):
        loki.CREDENTIALS = CredentialStore({
            "LOKI_API_BASE":
                "http://localhost:8000/v1/chat/completions",
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_MODEL": "local-model",
        })
        runner = mock.AsyncMock()

        with mock.patch(
                "loki_agent.terminal_frontend.run_subagent_cli_async",
                new=runner):
            status = asyncio.run(terminal_frontend.async_main(["--headless"]))

        runner.assert_awaited_once()
        self.assertEqual(status, 0)
        self.assertEqual(loki.current_config().api_key, "")
        self.assertNotIn("Authorization", loki.current_config().headers)

    def test_headless_configuration_failure_returns_usage_error(self):
        loki.CREDENTIALS = CredentialStore({})
        runner = mock.AsyncMock()
        stderr = io.StringIO()

        with mock.patch(
                "loki_agent.terminal_frontend.run_subagent_cli_async",
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

    def test_interactive_resume_accepts_saved_credentialless_connection(self):
        loki.CREDENTIALS = CredentialStore({})
        session = ScriptedInputSession([None])
        assistant_markdown = (
            "## Resume heading\n\n"
            "**visible resumed answer** and `code`\n\n"
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
            credential_env=None,
            stream=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-test.json")
            events = loki.initial_transcript_items() + [
                formats.message_item("user", "visible resumed question"),
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
                            "loki_agent.terminals."
                            "_terminal_supports_markdown_ansi",
                            return_value=True), mock.patch(
                            "loki_agent.terminal_frontend.restore_output_area_after_input"), \
                    contextlib.redirect_stdout(stdout):
                status = asyncio.run(
                    terminal_frontend.async_main([f"--resume={path}"]))

        confirm.assert_awaited_once()
        self.assertEqual(status, 0)
        rendered = stdout.getvalue()
        self.assertIn(
            "User: visible resumed question", rendered)
        self.assertIn(
            "local-model: "
            + terminals.render_markdown(assistant_markdown, style=True),
            rendered,
        )
        self.assertIn(
            "\033[42m## Resume heading\033[49m", rendered)
        self.assertIn(
            "\033[1mvisible resumed answer\033[0m", rendered)
        self.assertIn("\033[36mcode\033[0m", rendered)
        self.assertNotIn("**visible resumed answer**", rendered)
        self.assertIn("raw **inside fence**", rendered)
        self.assertTrue(rendered.endswith("----\n"))
        self.assertEqual(loki.current_model(), "local-model")
        self.assertEqual(loki.current_config().api_key, "")
        self.assertIsNone(loki.current_config().credential_env)
        self.assertTrue(loki.current_config().stream)
        self.assertNotIn("Authorization", loki.current_config().headers)

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

        async def load_provider_models():
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
        models_url = "https://catalog.example.test/custom/models"
        loki.CREDENTIALS = CredentialStore({
            "LOKI_API_BASE":
                "https://provider.example.test/v1/chat/completions",
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_API_KEY": "test-key",
            "LOKI_MODELS_URL": models_url,
        })
        session = ScriptedInputSession(["/model", "/quit"])

        async def load_provider_models():
            loki.current_session().models = ["first-model", "selected-model"]

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
                        new=mock.AsyncMock(return_value="selected-model")):
                status = asyncio.run(terminal_frontend.async_main([]))

            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)

        loader.assert_awaited_once()
        self.assertEqual(status, 0)
        self.assertEqual(loki.current_model(), "selected-model")
        self.assertEqual(loki.current_config().model, "selected-model")
        self.assertEqual(
            loki.current_config().chat_provider.models_url, models_url)
        self.assertEqual(
            saved["session_state"]["connection"]["models_url"], models_url)
        self.assertEqual(
            saved["session_state"]["connection"]["model"], "selected-model")

    def test_modelsdev_selection_persists_deprecated_status(self):
        loki.CREDENTIALS = CredentialStore({
            "PROVIDER_API_KEY": "test-key",
        })
        session = ScriptedInputSession(["/model", "/quit"])
        provider_entry = {
            "name": "Provider",
            "env": ["PROVIDER_API_KEY"],
            "api": "https://provider.example.test/v1",
        }
        model_entry = {
            "id": "old-model",
            "name": "Old Model",
            "status": "deprecated",
        }

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
                            new=mock.AsyncMock(return_value=(
                                "provider", provider_entry, model_entry))):
                status = asyncio.run(terminal_frontend.async_main([]))

            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)

        self.assertEqual(status, 0)
        self.assertEqual(loki.current_config().model_status, "deprecated")
        self.assertIn(
            "Model: old-model (deprecated); /model", terminal_frontend.status_text())
        self.assertEqual(
            saved["session_state"]["connection"]["model_status"],
            "deprecated",
        )

    def test_model_can_switch_from_catalog_back_to_explicit_connection(self):
        explicit_url = "http://localhost:8000/v1"
        loki.CREDENTIALS = CredentialStore({
            "LOKI_API_BASE": explicit_url,
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_MODEL": "private-model",
            "CATALOG_API_KEY": "catalog-key",
        })
        session = ScriptedInputSession(["/model", "/model", "/quit"])
        catalog_provider = {
            "name": "Catalog Provider",
            "env": ["CATALOG_API_KEY"],
            "api": "https://catalog.example.test/v1",
        }
        catalog_model = {
            "id": "catalog-model",
            "name": "Catalog Model",
        }
        seen_explicit = []

        async def pick_model(*, input_fn, credentials,
                             explicit_connection=None):
            seen_explicit.append(explicit_connection)
            if len(seen_explicit) == 1:
                return "catalog", catalog_provider, catalog_model
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
                            new=mock.AsyncMock(side_effect=pick_model)):
                status = asyncio.run(terminal_frontend.async_main([]))

            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)

        self.assertEqual(status, 0)
        self.assertEqual(len(seen_explicit), 2)
        self.assertTrue(all(
            isinstance(option, modelsdev.ExplicitConnectionOption)
            for option in seen_explicit))
        self.assertEqual(loki.current_config().url, explicit_url)
        self.assertEqual(loki.current_config().model, "private-model")
        self.assertEqual(
            loki.current_config().provider_name,
            "Explicit LOKI_* connection",
        )
        connection = saved["session_state"]["connection"]
        self.assertEqual(connection["model"], "private-model")
        self.assertEqual(
            connection["chat_url"],
            "http://localhost:8000/v1/chat/completions",
        )
        self.assertIsNone(connection["credential_env"])
        self.assertEqual(loki.current_config().api_key, "")
        self.assertNotIn("Authorization", loki.current_config().headers)

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
                            explicit_connection=None):
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
        self.assertEqual(loki.current_config().url, explicit_url)
        self.assertEqual(loki.current_config().model, "private-model")


class ExitStatusTests(unittest.TestCase):
    def test_executable_entry_points_propagate_headless_failure(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        env = {
            "HOME": os.environ.get("HOME", str(root)),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(root),
            "TERM": "dumb",
        }
        commands = [
            [sys.executable, str(root / "loki.py"), "--headless"],
            [sys.executable, "-m", "loki_agent", "--headless"],
        ]

        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    command,
                    cwd=root,
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
            for async_status, expected_status in [(0, 1), (2, 2)]:
                def finish(coroutine):
                    coroutine.close()
                    return async_status

                with self.subTest(async_status=async_status), mock.patch(
                        "loki_agent.terminal_frontend.CredentialStore.capture",
                        return_value=CredentialStore({})), mock.patch(
                            "loki_agent.terminal_frontend.signal.signal"), mock.patch(
                                "loki_agent.terminal_frontend.signal.pthread_sigmask"
                            ), mock.patch(
                                "loki_agent.terminal_frontend.initialize_terminal_overlay"
                            ), mock.patch(
                                "loki_agent.terminal_frontend.asyncio.run",
                                side_effect=finish), mock.patch(
                                    "loki_agent.terminal_frontend."
                                    "restore_terminal_overlay",
                                    side_effect=OSError("restore failed")
                                ), mock.patch.object(loki.current_session(), "chat_log_path", None
                                ), contextlib.redirect_stderr(stderr):
                    status = terminal_frontend.main()

                self.assertEqual(status, expected_status)
        finally:
            loki.CREDENTIALS = old_credentials

        self.assertIn("Cleanup error: OSError: restore failed",
                      stderr.getvalue())


class StatusTextTests(unittest.TestCase):
    def test_activity_status_redraws_only_for_changed_counts(self):
        activity = terminal_frontend.TerminalActivityStatus()

        with mock.patch(
                "loki_agent.terminal_frontend.terminals.redraw_status_bar"
        ) as redraw:
            activity.set_queued_messages(2)
            activity.set_queued_messages(2)
            activity.set_queued_images(1)
            activity.set_turn_running(True)
            activity.set_turn_running(True)

        self.assertTrue(activity.turn_running)
        self.assertEqual(activity.queued_messages, 2)
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
                url=chat_provider.input_url,
                provider_kind=chat_provider.kind,
                netloc="example.test:8443",
                api_key="secret",
                chat_provider=chat_provider,
                headers={},
                model="model-x"
            )
            # model is derived from runtime_config set above

            text = terminal_frontend.status_text(
                terminal_frontend.TerminalActivityStatus(
                    turn_running=True,
                    queued_messages=2,
                    queued_images=1,
                ))
        finally:
            restore_loki_state(old_values)

        self.assertEqual(
            text,
            "Remote: API: example.test:8443/base/path; Model: model-x; /model\n"
            "Local: turn=running, queued_messages=2, queued_images=1; "
            f"mode={loki.current_agent_mode()}; CWD: {loki.STARTUP_CWD}; "
            "/pwd, /cd DIR, /ps, /image PATH, !foo, /quit",
        )
        self.assertNotIn("user", text)
        self.assertNotIn("pass", text)
        self.assertNotIn("token", text)
        self.assertNotIn("secret", text)

    def test_status_text_marks_a_deprecated_selected_model(self):
        names = ["runtime_config", "shell_cwd"]
        old_values = save_loki_state(names)

        try:
            loki.current_session().shell_cwd = loki.STARTUP_CWD
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://example.test/v1",
                protocols.OPENAI_CHAT,
                "test-key",
                model="old-model",
                credential_env="EXAMPLE_API_KEY",
                model_status="deprecated",
            ))

            text = terminal_frontend.status_text()
        finally:
            restore_loki_state(old_values)

        self.assertIn("Model: old-model (deprecated); /model", text)


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
    def test_resume_renderer_replays_visible_conversation_without_metadata_dump(self):
        items = [
            formats.instruction_item("internal startup instruction"),
            formats.message_item("user", "hello"),
            formats.model_response_event(
                "openai_responses",
                [
                    formats.message_item("assistant", "hi there"),
                    formats.tool_call_item(
                        "call_1", "Read",
                        {"file_path": "README.md"}),
                ],
            ),
            formats.tool_result_item("call_1", "file contents", name="Read"),
        ]

        text = loki.ResumeTranscriptRenderer(assistant_label="Assistant").render(items)

        self.assertEqual(
            text,
            "User: hello\n\n"
            "Assistant: hi there\n\n"
            "Tool call: Read\n"
            "{'file_path': 'README.md'}\n\n"
            "Tool result: Read\n"
            "file contents",
        )
        self.assertNotIn("internal startup instruction", text)
        self.assertNotIn("response_metadata", text)

    def test_resume_renderer_injects_presentation_for_assistant_text_only(self):
        seen = []

        def render_assistant(text):
            seen.append(text)
            return f"<rendered>{text}</rendered>"

        items = [
            formats.message_item("user", "**literal user input**"),
            formats.model_response_event(
                "openai_chat",
                [formats.message_item(
                    "assistant", "**formatted assistant output**")],
                model="model-a",
            ),
        ]

        text = loki.ResumeTranscriptRenderer(
            assistant_label="Assistant",
            assistant_text_renderer=render_assistant,
        ).render(items)

        self.assertEqual(seen, ["**formatted assistant output**"])
        self.assertIn("User: **literal user input**", text)
        self.assertIn(
            "model-a: "
            "<rendered>**formatted assistant output**</rendered>",
            text,
        )

    def test_resume_renderer_uses_response_model_labels(self):
        items = [
            formats.message_item("user", "hello"),
            formats.model_response_event(
                "openai_chat",
                [formats.message_item("assistant", "from first")],
                model="model-a",
            ),
            formats.message_item("user", "again"),
            formats.model_response_event(
                "anthropic_messages",
                [formats.message_item("assistant", "from second")],
                model="model-b",
            ),
        ]

        text = loki.ResumeTranscriptRenderer(
            assistant_label="current").render(items)

        self.assertIn("model-a: from first", text)
        self.assertIn("model-b: from second", text)

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

        text = loki.ResumeTranscriptRenderer(
            assistant_label="Assistant").render(items)

        self.assertIn("Provider tool result", text)
        self.assertIn("Visible result", text)
        self.assertNotIn("\nNone", text)
        self.assertIn("[Model response failed]", text)
        self.assertIn("visible failure", text)


class SessionResponsePersistenceTests(unittest.TestCase):
    def test_saved_note_flushes_stdout_before_writing_stderr(self):
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
            savefiles.report_chat_log_saved("/tmp/session.json")

        self.assertIn("Saved chat log", stderr.getvalue())

    def test_response_boundary_and_toolset_are_saved_without_call_ledger(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
            "session_toolsets", "shell_cwd", "previous_shell_cwd",
        ]
        old_values = {
            name: copy.deepcopy(loki.current_session().__dict__[name])
            for name in names
        }
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                loki.new_chat_log(path)
                loki.current_transcript().append(
                    formats.message_item("user", "hello"))
                turn = formats.DecodedTurn(
                    [formats.message_item("assistant", "world")],
                    {
                        "protocol": "openai_chat",
                        "provider_id": "provider",
                        "model": "model-a",
                        "usage": {"total_tokens": 3},
                    },
                )
                loki.current_transcript().append(turn.to_event())
                loki._remember_session_toolset(loki.TOOLS)
                loki.mark_chat_log_dirty()
                loki.save_chat_log()

                blob = json.loads(pathlib.Path(path).read_text(
                    encoding="utf-8"))
                loki.current_session().session_toolsets = []
                with contextlib.redirect_stdout(io.StringIO()):
                    loki.load_chat_log(path)
                loaded_toolsets = copy.deepcopy(loki.current_toolsets())

            self.assertEqual(
                [item["type"] for item in blob["events"]],
                ["message", "message", "model_response"],
            )
            self.assertNotIn("calls", blob)
            response = blob["events"][2]
            self.assertEqual(response["protocol"], "openai_chat")
            self.assertEqual(response["provider"], "provider")
            self.assertEqual(response["model"], "model-a")
            self.assertEqual(
                response["usage"],
                {"total_tokens": 3},
            )
            self.assertEqual(blob["toolsets"], [loki.TOOLS])
            self.assertEqual(loaded_toolsets, blob["toolsets"])
            self.assertNotIn('"start":', json.dumps(blob))
            self.assertNotIn('"end":', json.dumps(blob))
        finally:
            restore_loki_state(old_values)


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
                request_url, payload, request_headers=None,
                report_errors=False, show_timing=False):
            if not queued:
                raise AssertionError("unexpected provider request")
            captured.append({
                "url": request_url,
                "payload": copy.deepcopy(payload),
                "headers": copy.deepcopy(request_headers),
            })
            return copy.deepcopy(queued.pop(0))

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

    def test_same_protocol_provider_switch_continues_tools_and_resumes(self):
        with self._isolated_runtime(), tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-acceptance.json")
            provider_a = loki.make_runtime_config(
                "https://provider-a.example/v1/responses",
                protocols.OPENAI_RESPONSES,
                "key-a",
                model="model-a",
                provider_id="provider-a",
                provider_name="Provider A",
                credential_env="PROVIDER_A_API_KEY",
            )
            loki.apply_runtime_config(provider_a)
            loki.new_chat_log(path)
            loki.current_transcript().append(
                formats.message_item("user", "read the file"))

            captured_a = []
            requests_a = self._request_sequence([
                {
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {
                            "type": "reasoning",
                            "id": "reasoning_a",
                            "summary": [],
                            "encrypted_content": "provider-a-secret",
                        },
                        {
                            "type": "message",
                            "id": "message_a",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": "Provider A will read it.",
                                "annotations": [],
                            }],
                        },
                        {
                            "type": "function_call",
                            "id": "function_a",
                            "status": "completed",
                            "call_id": "call_a",
                            "name": "Read",
                            "arguments": '{"file_path":"README.md"}',
                        },
                    ],
                },
                {
                    "object": "response",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "id": "message_a_final",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": "Provider A finished.",
                            "annotations": [],
                        }],
                    }],
                },
            ], captured_a)
            dispatched = []

            async def dispatch(
                    name, args, allowed=None, extra_context=None):
                dispatched.append((name, copy.deepcopy(args)))
                return {
                    "ok": True,
                    "content": f"{name} result from provider A",
                }

            with mock.patch(
                    "loki_agent.loki.async_chat_request",
                    new=requests_a), mock.patch(
                        "loki_agent.loki.dispatch_tool_async",
                        new=dispatch):
                result_a = asyncio.run(loki.run_tool_loop_async(
                    loki.current_transcript(),
                    allowed={"Read", "Grep"},
                    max_loops=4,
                ))

            requests_a.assert_exhausted()
            self.assertEqual(result_a, "Provider A finished.")
            self.assertEqual(
                dispatched, [("Read", {"file_path": "README.md"})])
            self.assertEqual(len(captured_a), 2)
            self.assertTrue(all(
                request["url"] == provider_a.chat_provider.chat_url
                for request in captured_a))
            self.assertTrue(all(
                request["headers"]["Authorization"] == "Bearer key-a"
                for request in captured_a))
            self._assert_responses_payload_valid(
                captured_a[1]["payload"])
            serialized_a_continuation = json.dumps(
                captured_a[1]["payload"])
            self.assertIn("provider-a-secret",
                          serialized_a_continuation)
            self.assertIn("call_a", serialized_a_continuation)
            self.assertIn(
                "Read result from provider A",
                serialized_a_continuation,
            )
            loki.mark_chat_log_dirty()
            loki.save_chat_log()

            provider_b = loki.make_runtime_config(
                "https://provider-b.example/v1/responses",
                protocols.OPENAI_RESPONSES,
                "key-b",
                model="model-b",
                provider_id="provider-b",
                provider_name="Provider B",
                credential_env="PROVIDER_B_API_KEY",
            )
            loki.apply_runtime_config(provider_b)
            loki.set_session_connection(
                loki.active_connection_descriptor())
            loki.current_transcript().append(
                formats.message_item("user", "continue with grep"))

            captured_b = []
            requests_b = self._request_sequence([
                {
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {
                            "type": "reasoning",
                            "id": "reasoning_b",
                            "summary": [],
                            "encrypted_content": "provider-b-secret",
                        },
                        {
                            "type": "message",
                            "id": "message_b",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": "Provider B will grep.",
                                "annotations": [],
                            }],
                        },
                        {
                            "type": "function_call",
                            "id": "function_b",
                            "status": "completed",
                            "call_id": "call_b",
                            "name": "Grep",
                            "arguments": '{"pattern":"marker"}',
                        },
                    ],
                },
                {
                    "object": "response",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "id": "message_b_final",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": "Provider B finished.",
                            "annotations": [],
                        }],
                    }],
                },
            ], captured_b)

            async def dispatch_b(
                    name, args, allowed=None, extra_context=None):
                dispatched.append((name, copy.deepcopy(args)))
                return {
                    "ok": True,
                    "content": f"{name} result from provider B",
                }

            with mock.patch(
                    "loki_agent.loki.async_chat_request",
                    new=requests_b), mock.patch(
                        "loki_agent.loki.dispatch_tool_async",
                        new=dispatch_b):
                result_b = asyncio.run(loki.run_tool_loop_async(
                    loki.current_transcript(),
                    allowed={"Read", "Grep"},
                    max_loops=4,
                ))

            requests_b.assert_exhausted()
            self.assertEqual(result_b, "Provider B finished.")
            self.assertEqual(
                dispatched[-1], ("Grep", {"pattern": "marker"}))
            self.assertEqual(len(captured_b), 2)
            self.assertTrue(all(
                request["url"] == provider_b.chat_provider.chat_url
                for request in captured_b))
            self.assertTrue(all(
                request["headers"]["Authorization"] == "Bearer key-b"
                for request in captured_b))
            for captured in captured_b:
                self._assert_responses_payload_valid(
                    captured["payload"])
            first_b = json.dumps(captured_b[0]["payload"])
            second_b = json.dumps(captured_b[1]["payload"])
            self.assertIn("Provider A will read it.", first_b)
            self.assertIn("Read result from provider A", first_b)
            self.assertNotIn("provider-a-secret", first_b)
            self.assertNotIn("reasoning_a", first_b)
            self.assertNotIn("provider-a-secret", second_b)
            self.assertIn("provider-b-secret", second_b)
            self.assertIn("Grep result from provider B", second_b)

            loki.mark_chat_log_dirty()
            loki.save_chat_log()
            loki.current_session().transcript_items = []
            with contextlib.redirect_stdout(io.StringIO()):
                loki.load_chat_log(path)
            descriptor = loki.connection_from_session_state(
                loki.current_state())
            self.assertEqual(descriptor.provider_id, "provider-b")
            self.assertEqual(
                descriptor.chat_url,
                provider_b.chat_provider.chat_url,
            )
            restored_b = loki.config_from_connection_descriptor(
                descriptor,
                CredentialStore({"PROVIDER_B_API_KEY": "key-b"}),
            )
            loki.apply_runtime_config(restored_b)
            loki.current_transcript().append(
                formats.message_item("user", "answer after resume"))
            loki.mark_chat_log_dirty()

            captured_resume = []
            requests_resume = self._request_sequence([
                {
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "id": "message_b_resumed_call",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": "Resumed provider will grep.",
                                "annotations": [],
                            }],
                        },
                        {
                            "type": "function_call",
                            "id": "function_b_resumed",
                            "status": "completed",
                            "call_id": "call_b_resumed",
                            "name": "Grep",
                            "arguments": '{"pattern":"after-resume"}',
                        },
                    ],
                },
                {
                    "object": "response",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "id": "message_b_resumed_final",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": "Resumed on provider B.",
                            "annotations": [],
                        }],
                    }],
                },
            ], captured_resume)
            resumed_dispatches = []

            async def dispatch_resumed(
                    name, args, allowed=None, extra_context=None):
                resumed_dispatches.append(
                    (name, copy.deepcopy(args)))
                return {
                    "ok": True,
                    "content": "tool result after resume",
                }

            with mock.patch(
                    "loki_agent.loki.async_chat_request",
                    new=requests_resume), mock.patch(
                        "loki_agent.loki.dispatch_tool_async",
                        new=dispatch_resumed):
                resumed_result = asyncio.run(
                    loki.run_tool_loop_async(
                        loki.current_transcript(),
                        allowed={"Read", "Grep"},
                        max_loops=4,
                    ))

            requests_resume.assert_exhausted()
            self.assertEqual(
                resumed_result, "Resumed on provider B.")
            self.assertEqual(
                resumed_dispatches,
                [("Grep", {"pattern": "after-resume"})],
            )
            self.assertEqual(len(captured_resume), 2)
            self.assertTrue(all(
                request["url"] == provider_b.chat_provider.chat_url
                for request in captured_resume))
            self.assertTrue(all(
                request["headers"]["Authorization"] == "Bearer key-b"
                for request in captured_resume))
            for request in captured_resume:
                self._assert_responses_payload_valid(
                    request["payload"])
            serialized_resume = json.dumps(
                captured_resume[0]["payload"])
            serialized_resume_continuation = json.dumps(
                captured_resume[1]["payload"])
            self.assertNotIn("provider-a-secret", serialized_resume)
            self.assertIn("provider-b-secret", serialized_resume)
            self.assertIn(
                "Grep result from provider B", serialized_resume)
            self.assertIn(
                "tool result after resume",
                serialized_resume_continuation,
            )
            formats.validate_events(loki.current_transcript())

    def test_incomplete_tool_call_and_media_survive_resume_and_switch(self):
        with self._isolated_runtime(), tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-incomplete.json")
            responses_config = loki.make_runtime_config(
                "https://responses.example/v1/responses",
                protocols.OPENAI_RESPONSES,
                "responses-key",
                model="responses-model",
                provider_id="responses-provider",
                provider_name="Responses Provider",
                credential_env="RESPONSES_API_KEY",
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
                    "loki_agent.loki.async_chat_request",
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
                "chat-key",
                model="chat-model",
                provider_id="chat-provider",
                provider_name="Chat Provider",
                credential_env="CHAT_API_KEY",
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
                    "loki_agent.loki.async_chat_request",
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
                "key-a",
                model="claude-a",
                provider_id="anthropic-a",
                provider_name="Anthropic A",
                credential_env="ANTHROPIC_A_API_KEY",
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
                    "loki_agent.loki.async_chat_request",
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
                request["headers"]["x-api-key"] == "key-a"
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
                "key-b",
                model="claude-b",
                provider_id="anthropic-b",
                provider_name="Anthropic B",
                credential_env="ANTHROPIC_B_API_KEY",
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
                    "loki_agent.loki.async_chat_request",
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
            self.assertEqual(
                captured_b[0]["headers"]["x-api-key"], "key-b")
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

    def test_chat_exact_replay_runs_through_runtime_and_tool_loop(self):
        with self._isolated_runtime(), tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-exact.json")
            config = loki.make_runtime_config(
                "https://chat-origin.example/v1/chat/completions",
                protocols.OPENAI_CHAT,
                "chat-key",
                model="chat-model",
                provider_id="chat-origin",
                provider_name="Chat Origin",
                credential_env="CHAT_ORIGIN_API_KEY",
            )
            loki.apply_runtime_config(config)
            loki.new_chat_log(path)
            loki.current_transcript().append(
                formats.message_item("user", "read exactly"))

            exact_content = [
                {"type": "text", "text": "first block"},
                {"type": "text", "text": "second block"},
            ]
            exact_call = {
                "id": "call_exact",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": '{"file_path":"README.md"}',
                },
            }
            captured = []
            requests = self._request_sequence([
                {
                    "id": "chat_first",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": exact_content,
                            "tool_calls": [exact_call],
                        },
                        "finish_reason": "tool_calls",
                    }],
                },
                {
                    "id": "chat_final",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Exact replay completed.",
                        },
                        "finish_reason": "stop",
                    }],
                },
            ], captured)
            dispatches = []

            async def dispatch(
                    name, args, allowed=None, extra_context=None):
                dispatches.append((name, copy.deepcopy(args)))
                return {"ok": True, "content": "exact tool result"}

            with mock.patch(
                    "loki_agent.loki.async_chat_request",
                    new=requests), mock.patch(
                        "loki_agent.loki.dispatch_tool_async",
                        new=dispatch):
                answer = asyncio.run(loki.run_tool_loop_async(
                    loki.current_transcript(),
                    allowed={"Read"},
                    max_loops=4,
                ))

            requests.assert_exhausted()
            self.assertEqual(answer, "Exact replay completed.")
            self.assertEqual(
                dispatches, [("Read", {"file_path": "README.md"})])
            self.assertEqual(len(captured), 2)
            self.assertTrue(all(
                request["url"] == config.chat_provider.chat_url
                for request in captured))
            self.assertTrue(all(
                request["headers"]["Authorization"]
                == "Bearer chat-key"
                for request in captured))
            continuation = captured[1]["payload"]
            self._assert_chat_payload_valid(continuation)
            historical = next(
                message for message in continuation["messages"]
                if message.get("tool_calls"))
            self.assertEqual(historical["content"], exact_content)
            self.assertEqual(historical["tool_calls"], [exact_call])
            tool_result = next(
                message for message in continuation["messages"]
                if message.get("role") == "tool")
            self.assertEqual(
                tool_result["tool_call_id"], "call_exact")
            self.assertEqual(
                tool_result["content"], "exact tool result")
            formats.validate_events(loki.current_transcript())

    def test_projection_smoke_switches_provider_and_protocol(self):
        names = [
            "runtime_config", "chat_log_path", "session_state",
            "chat_log_dirty", "transcript_items", "session_todos",
            "session_toolsets", "shell_cwd", "previous_shell_cwd",
        ]
        old_values = {
            name: copy.deepcopy(loki.current_session().__dict__[name])
            for name in names
        }
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-primary.json")
                source = loki.make_runtime_config(
                    "https://provider-a.example/v1/messages",
                    protocols.ANTHROPIC_MESSAGES,
                    "key-a",
                    model="model-a",
                    provider_id="provider-a",
                    provider_name="Provider A",
                )
                loki.apply_runtime_config(source)
                loki.new_chat_log(path)
                loki.current_transcript().append(
                    formats.message_item("user", "inspect README"))
                source_turn = formats.anthropic_response_to_items({
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "private thought",
                            "signature": "provider-a-signature",
                        },
                        {
                            "type": "text",
                            "text": "I will inspect it.",
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_a",
                            "name": "Read",
                            "input": {"file_path": "README.md"},
                        },
                    ],
                    "stop_reason": "tool_use",
                })
                source_turn.metadata.update({
                    "provider_id": "provider-a",
                    "endpoint": source.chat_provider.chat_url,
                    "model": "model-a",
                })
                loki.current_transcript().append(source_turn.to_event())
                call = formats.response_tool_calls(source_turn)[0]
                loki.current_transcript().append(
                    formats.tool_result_for_call(
                        call, "README contents"))
                loki.mark_chat_log_dirty()
                loki.save_chat_log()

                loki.current_session().transcript_items = []
                with contextlib.redirect_stdout(io.StringIO()):
                    loki.load_chat_log(path)
                self.assertEqual(
                    loki.current_transcript()[2]["endpoint"],
                    source.chat_provider.chat_url,
                )

                target_b = loki.make_runtime_config(
                    "https://provider-b.example/v1/responses",
                    protocols.OPENAI_RESPONSES,
                    "key-b",
                    model="model-b",
                    provider_id="provider-b",
                    provider_name="Provider B",
                )
                loki.apply_runtime_config(target_b)
                loki.set_session_connection(
                    loki.active_connection_descriptor())
                captured = []

                async def fake_request(
                        request_url, payload, request_headers=None,
                        report_errors=False, show_timing=False):
                    captured.append(copy.deepcopy(payload))
                    return {
                        "object": "response",
                        "status": "completed",
                        "output": [
                            {
                                "type": "reasoning",
                                "id": "reasoning_b",
                                "summary": [],
                                "encrypted_content": "provider-b-only",
                            },
                            {
                                "type": "message",
                                "id": "message_b",
                                "status": "completed",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": "Provider B answer",
                                    "annotations": [],
                                }],
                            },
                            {
                                "type": "function_call",
                                "id": "function_b",
                                "status": "completed",
                                "call_id": "call_b",
                                "name": "Grep",
                                "arguments": '{"pattern":"marker"}',
                            },
                        ],
                    }

                with mock.patch(
                        "loki_agent.loki.async_chat_request",
                        side_effect=fake_request):
                    target_turn = asyncio.run(
                        loki.async_chat_completion(
                            loki.current_transcript(), tools=[]))

                first_switched_payload = json.dumps(captured[0])
                self.assertIn("I will inspect it.", first_switched_payload)
                self.assertIn("README contents", first_switched_payload)
                self.assertNotIn(
                    "provider-a-signature", first_switched_payload)
                self.assertNotIn("private thought", first_switched_payload)
                self.assertEqual(
                    target_turn.metadata["endpoint"],
                    target_b.chat_provider.chat_url,
                )
                loki.current_transcript().append(target_turn.to_event())
                target_call = formats.response_tool_calls(target_turn)[0]
                loki.current_transcript().append(
                    formats.tool_result_for_call(
                        target_call, "grep result"))
                loki.mark_chat_log_dirty()
                loki.save_chat_log()

                loki.current_session().transcript_items = []
                with contextlib.redirect_stdout(io.StringIO()):
                    loki.load_chat_log(path)
                saved_connection = loki.connection_from_session_state(
                    loki.current_state())
                self.assertEqual(
                    saved_connection.provider_id, "provider-b")

                target_c = loki.make_runtime_config(
                    "https://provider-c.example/v1/responses",
                    protocols.OPENAI_RESPONSES,
                    "key-c",
                    model="model-c",
                    provider_id="provider-c",
                    provider_name="Provider C",
                )
                responses_payload = target_c.chat_provider.chat_payload(
                    loki.current_transcript(), [], "model-c")
                rendered_responses = json.dumps(responses_payload)
                self.assertIn("Provider B answer", rendered_responses)
                self.assertIn("grep result", rendered_responses)
                self.assertNotIn(
                    "provider-b-only", rendered_responses)
                self.assertNotIn("reasoning_b", rendered_responses)

                target_d = loki.make_runtime_config(
                    "https://provider-d.example/v1/chat/completions",
                    protocols.OPENAI_CHAT,
                    "key-d",
                    model="model-d",
                    provider_id="provider-d",
                    provider_name="Provider D",
                )
                chat_payload = target_d.chat_provider.chat_payload(
                    loki.current_transcript(), [], "model-d")
                rendered_chat = json.dumps(chat_payload)
                self.assertIn("I will inspect it.", rendered_chat)
                self.assertIn("README contents", rendered_chat)
                self.assertIn("Provider B answer", rendered_chat)
                self.assertIn("grep result", rendered_chat)
                self.assertNotIn(
                    "provider-a-signature", rendered_chat)
                self.assertNotIn("provider-b-only", rendered_chat)
                formats.validate_events(loki.current_transcript())
        finally:
            restore_loki_state(old_values)


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
        self.assertEqual(
            loki.resolve_chat_log_path("./chat-abc.json"),
            os.path.join(loki.STARTUP_CWD, "chat-abc.json"),
        )
        self.assertEqual(
            loki.resolve_chat_log_path("logs/chat-abc.json"),
            os.path.join(loki.STARTUP_CWD, "logs", "chat-abc.json"),
        )

    def test_new_chat_log_path_uses_local_loki_chat_directory(self):
        path = loki.new_chat_log_path()

        self.assertEqual(os.path.dirname(path), loki.CHAT_LOG_DIR)
        self.assertTrue(os.path.basename(path).startswith("chat-"))
        self.assertTrue(path.endswith(".json"))
        self.assertTrue(os.path.isdir(loki.CHAT_LOG_DIR))

    def test_new_chat_log_creates_parent_directory(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
        ]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, ".loki", "chats", "chat-test.json")
                loki.new_chat_log(path)

                self.assertTrue(os.path.isdir(os.path.dirname(path)))
                self.assertEqual(loki.current_chat_log_path(), path)
                self.assertTrue(loki.current_dirty())
                self.assertFalse(os.path.exists(path))
        finally:
            restore_loki_state(old_values)


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
            def __init__(self):
                self.calls = []

            def save_cursor_position(self, *a, **k):
                pass

            def restore_cursor_position(self, *a, **k):
                pass

            def clear_to_end_of_screen(self, *a, **k):
                self.calls.append(("clear_to_end_of_screen",))

            def goto_position(self, *a, **k):
                self.calls.append(("goto_position", *a))

            def flush(self, *a, **k):
                self.calls.append(("flush",))

        terminal_frontend.terminal = _FakeTerminal()

        def restore():
            loki.CHAT_LOG_DIR = saved_log_dir
            terminal_frontend.terminal = saved_terminal

        return restore, session

    def test_picker_selects_by_number(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "aaa", '{"text":"alpha chat"}', mtime=1000)
            self._write_chat(tmpdir, "bbb", '{"text":"beta chat"}', mtime=2000)
            self._write_chat(tmpdir, "ccc", '{"text":"gamma chat"}', mtime=3000)
            restore, session = self._make_picker(tmpdir, ["2"])
            try:
                result = asyncio.run(terminal_frontend.run_session_picker_async(session))
            finally:
                restore()
            # mtime-sorted oldest->newest: aaa(1000), bbb(2000), ccc(3000).
            # "2" selects the middle one = bbb.
            self.assertTrue(result.endswith("chat-bbb.json"))

    def test_picker_prints_saved_sessions_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(
                tmpdir, "aaa", '{"text":"alpha chat"}', mtime=1000)
            restore, session = self._make_picker(tmpdir, ["1"])
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    result = asyncio.run(
                        terminal_frontend.run_session_picker_async(session))
            finally:
                restore()

            self.assertTrue(result.endswith("chat-aaa.json"))
            self.assertTrue(
                output.getvalue().startswith("\nSaved sessions:\n"))

    def test_picker_finishes_clear_before_returning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "aaa", '{"text":"alpha"}', mtime=1000)
            restore, session = self._make_picker(tmpdir, ["1"])
            picker_terminal = terminal_frontend.terminal
            try:
                result = asyncio.run(terminal_frontend.run_session_picker_async(session))
            finally:
                restore()

            self.assertTrue(result.endswith("chat-aaa.json"))
            self.assertEqual(picker_terminal.calls[-3:], [
                ("goto_position", 1, 1),
                ("clear_to_end_of_screen",),
                ("flush",),
            ])

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

    def test_picker_mtime_newest_last(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "old", '{"text":"old session"}', mtime=1000)
            self._write_chat(tmpdir, "mid", '{"text":"mid session"}', mtime=2000)
            self._write_chat(tmpdir, "new", '{"text":"new session"}', mtime=3000)
            restore, session = self._make_picker(tmpdir, ["3"])
            try:
                result = asyncio.run(terminal_frontend.run_session_picker_async(session))
            finally:
                restore()
            # Oldest->newest: old, mid, new. "3" = newest.
            self.assertTrue(result.endswith("chat-new.json"))

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

                self.assertEqual(loki.current_cwd(), tmpdir)
                self.assertEqual(os.getcwd(), process_cwd)
                self.assertEqual(loki._resolve_path("file.txt"), os.path.join(tmpdir, "file.txt"))
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

                result = asyncio.run(loki.run_bash_async("pwd"))
                jobs = list(loki.current_job_manager().jobs.values())
        finally:
            restore_loki_state(old_values)

        self.assertIn("[stdout]\n" + workdir, result)
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

        self.assertEqual(blob["session_state"]["shell_cwd"], cwd)

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
                    "do-not-persist-this",
                    model="z-ai/glm",
                    provider_id="openrouter",
                    provider_name="OpenRouter",
                    credential_env="OPENROUTER_API_KEY",
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
        self.assertEqual(connection["credential_env"], "OPENROUTER_API_KEY")
        self.assertEqual(connection["model_status"], "deprecated")
        self.assertNotIn("api_url", connection)
        self.assertEqual(
            connection["chat_url"],
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertNotIn("do-not-persist-this", text)

    def test_save_chat_log_persists_credentialless_connection(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
            "runtime_config", ]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                loki.apply_runtime_config(loki.make_runtime_config(
                    "http://localhost:8000/v1",
                    protocols.OPENAI_CHAT,
                    "",
                    model="local-model",
                    provider_name="Explicit LOKI_* connection",
                    credential_env=None,
                    stream=True,
                ))
                loki.new_chat_log(path)
                loki.save_chat_log()
                blob = json.loads(
                    pathlib.Path(path).read_text(encoding="utf-8"))
        finally:
            restore_loki_state(old_values)

        connection = blob["session_state"]["connection"]
        self.assertEqual(connection["model"], "local-model")
        self.assertIsNone(connection["credential_env"])
        self.assertTrue(connection["stream"])
        self.assertEqual(
            connection["chat_url"],
            "http://localhost:8000/v1/chat/completions",
        )

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
                    credential_env="OPENROUTER_API_KEY",
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
                self.assertFalse(loki.current_dirty())
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
                    credential_env="OPENROUTER_API_KEY",
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
                    credential_env="SAVED_API_KEY",
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
                    "runtime-only-secret",
                    model="override-model",
                    credential_env="LOKI_API_KEY",
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
                with mock.patch(
                        "loki_agent.loki.os.replace",
                        side_effect=OSError("publish failed")):
                    with self.assertRaisesRegex(OSError, "publish failed"):
                        loki.save_chat_log()

                self.assertEqual(pathlib.Path(path).read_bytes(), original)
                self.assertTrue(loki.current_dirty())
                self.assertEqual(
                    [name for name in os.listdir(tmpdir)
                     if name.endswith(".tmp")],
                    [],
                )
        finally:
            restore_loki_state(old_values)

    def test_successful_model_selection_replaces_resumed_connection(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "runtime_config", "shell_cwd", "previous_shell_cwd",
        ]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                old_descriptor = ConnectionDescriptor(
                    provider_id="old",
                    provider_name="Old",
                    model="old-model",
                    chat_url="https://old.example/v1/chat/completions",
                    models_url="https://old.example/v1/models",
                    protocol=protocols.OPENAI_CHAT,
                    credential_env="OLD_API_KEY",
                )
                blob = formats.new_log_blob(
                    loki.initial_transcript_items(), [])
                blob["session_state"] = {
                    "shell_cwd": tmpdir,
                    "connection": old_descriptor.to_dict(),
                }
                pathlib.Path(path).write_text(
                    json.dumps(blob), encoding="utf-8")
                loki.load_chat_log(path)

                loki.apply_runtime_config(loki.make_runtime_config(
                    "https://new.example/v1",
                    protocols.OPENAI_CHAT,
                    "new-secret",
                    model="new-model",
                    provider_id="new",
                    provider_name="New",
                    credential_env="NEW_API_KEY",
                ))
                new_descriptor = loki.active_connection_descriptor()
                loki.set_session_connection(new_descriptor)
                loki.save_chat_log()

                after = json.loads(
                    pathlib.Path(path).read_text(encoding="utf-8"))
                self.assertEqual(
                    after["session_state"]["connection"],
                    new_descriptor.to_dict(),
                )
        finally:
            restore_loki_state(old_values)

    def test_load_session_state_restores_shell_cwd(self):
        names = ["shell_cwd", "previous_shell_cwd"]
        old_values = save_loki_state(names)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loki.load_session_state({"shell_cwd": tmpdir})

                self.assertEqual(loki.current_cwd(), tmpdir)
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
            credential_env="OPENROUTER_API_KEY",
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
        self.assertIn("Provider: OpenRouter", rendered)
        self.assertIn("Model: z-ai/glm", rendered)
        self.assertIn(
            "Chat endpoint: https://openrouter.ai/api/v1/chat/completions",
            rendered,
        )
        self.assertIn("Credential: OPENROUTER_API_KEY", rendered)

    def test_saved_credentialless_connection_confirmation_identifies_no_auth(
            self):
        descriptor = ConnectionDescriptor(
            provider_id=None,
            provider_name="Explicit LOKI_* connection",
            model="local-model",
            chat_url="http://localhost:8000/v1/chat/completions",
            models_url="http://localhost:8000/v1/models",
            protocol=protocols.OPENAI_CHAT,
            credential_env=None,
            stream=True,
        )

        class FakeSession:
            def modal(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def prompt(self, prompt):
                return "yes"

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            accepted = asyncio.run(
                terminal_frontend.confirm_saved_connection_async(
                    descriptor, FakeSession()))

        self.assertTrue(accepted)
        self.assertIn("Authentication: none", output.getvalue())
        self.assertIn("Streaming: yes", output.getvalue())
        self.assertNotIn("Credential: None", output.getvalue())


class SubagentLaunchTests(unittest.TestCase):
    def test_subagent_launch_uses_current_script_entrypoint(self):
        old_argv = sys.argv[:]
        try:
            sys.argv = ["./loki.py"]

            argv = loki._subagent_argv("Explore", "inspect this")
        finally:
            sys.argv = old_argv

        self.assertEqual(argv, [
            sys.executable,
            os.path.abspath("./loki.py"),
            "--subagent",
            "Explore",
            "--prompt",
            "inspect this",
        ])

    def test_subagent_launch_preserves_module_entrypoint(self):
        old_argv = sys.argv[:]
        try:
            sys.argv = [os.path.abspath("loki_agent/__main__.py")]

            argv = loki._subagent_argv("Explore", "inspect this")
        finally:
            sys.argv = old_argv

        self.assertEqual(argv, [
            sys.executable,
            "-m",
            "loki_agent",
            "--subagent",
            "Explore",
            "--prompt",
            "inspect this",
        ])

    def test_subagent_environment_contains_only_active_normalized_key(self):
        names = ["runtime_config"]
        old_values = save_loki_state(names)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "OPENROUTER_API_KEY": "unrelated-key",
        }
        CredentialStore.capture(env)
        try:
            with mock.patch.dict(os.environ, env, clear=True):
                loki.apply_runtime_config(loki.make_runtime_config(
                    "https://example.test/v1",
                    protocols.OPENAI_CHAT,
                    "active-key",
                    model="active-model",
                    credential_env="EXAMPLE_API_KEY",
                    models_url="https://example.test/custom-models",
                    max_tokens=12345,
                    anthropic_version="2026-01-02",
                    auth_header="X-Custom-Key",
                ))
                child_env = loki._subagent_env()
        finally:
            restore_loki_state(old_values)

        self.assertEqual(child_env["LOKI_API_KEY"], "active-key")
        self.assertEqual(child_env["LOKI_STREAM"], "0")
        self.assertEqual(child_env["LOKI_MAX_TOKENS"], "12345")
        self.assertEqual(
            child_env["LOKI_ANTHROPIC_VERSION"], "2026-01-02")
        self.assertEqual(child_env["LOKI_AUTH_HEADER"], "X-Custom-Key")
        self.assertEqual(
            child_env["LOKI_MODELS_URL"],
            "https://example.test/custom-models",
        )
        self.assertNotIn("OPENROUTER_API_KEY", child_env)

    def test_subagent_environment_preserves_credentialless_connection(self):
        names = ["runtime_config"]
        old_values = save_loki_state(names)
        try:
            with mock.patch.dict(
                    os.environ, {"LOKI_API_KEY": "must-not-leak"},
                    clear=True):
                loki.apply_runtime_config(loki.make_runtime_config(
                    "http://localhost:8000/v1",
                    protocols.OPENAI_CHAT,
                    "",
                    model="local-model",
                    credential_env=None,
                    stream=True,
                ))
                child_env = loki._subagent_env()
        finally:
            restore_loki_state(old_values)

        self.assertEqual(child_env["LOKI_API_BASE"],
                         "http://localhost:8000/v1")
        self.assertEqual(child_env["LOKI_MODEL"], "local-model")
        self.assertEqual(child_env["LOKI_STREAM"], "1")
        self.assertNotIn("LOKI_API_KEY", child_env)


class StreamingCompletionTests(unittest.TestCase):
    def setUp(self):
        self.old_runtime_config = loki.current_config()
        self.old_model = loki.current_model()
        loki.apply_runtime_config(loki.make_runtime_config(
            "http://localhost:8000/v1",
            protocols.OPENAI_CHAT,
            "",
            model="local-model",
            credential_env=None,
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
            "",
            model="other-model",
            credential_env=None,
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

    def test_terminal_stream_writes_one_plain_incremental_line(self):
        old_config = loki.current_session().runtime_config
        output = io.StringIO()
        try:
            loki.current_session().runtime_config = types.SimpleNamespace(model="local-model")
            with contextlib.redirect_stdout(output), \
                    contextlib.redirect_stderr(output):
                terminal_frontend._terminal_agent_event({"type": "assistant_start"})
                terminal_frontend._terminal_agent_event({
                    "type": "assistant_delta",
                    "content": "**hel",
                })
                terminal_frontend._terminal_agent_event({
                    "type": "assistant_delta",
                    "content": "lo**",
                })
                terminal_frontend._terminal_agent_event({
                    "type": "assistant_end",
                    "complete": True,
                })
                terminal_frontend._terminal_agent_event({
                    "type": "response_timing",
                    "elapsed": 1.25,
                })
        finally:
            loki.current_session().runtime_config = old_config

        self.assertEqual(
            output.getvalue(),
            "\nlocal-model: **hello**\n"
            "\n[T]  [LLM Response Time: 1.250s]\n",
        )

    def test_streamed_text_is_not_printed_again_or_duplicated_in_transcript(
            self):
        transcript = [formats.message_item("user", "hello")]
        events = []

        async def chat_fn(items, on_text_delta):
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

        async def chat_fn(items, on_text_delta):
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

        async def chat_fn(items, on_text_delta):
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
    def test_function_call_only_response_executes_tool_and_continues(self):
        transcript = [formats.message_item("user", "read README")]
        seen_inputs = []
        events = []

        async def chat_fn(items):
            seen_inputs.append([item.get("type") for item in items])
            if len(seen_inputs) == 1:
                return formats.DecodedTurn(
                    [formats.tool_call_item(
                        "call_1", "Read",
                        {"file_path": "README.md"})],
                    {
                        "protocol": "openai_responses",
                        "response": {
                            "id": "resp_1",
                            "object": "response",
                            "status": "completed",
                            "model": "gpt-test",
                        },
                    },
                )
            return [formats.message_item("assistant", "done")]

        async def fake_dispatch(fn_name, args, allowed=None, extra_context=None):
            self.assertEqual(fn_name, "Read")
            self.assertEqual(args, {"file_path": "README.md"})
            return {"ok": True, "content": "file contents"}

        old_dispatch = loki.dispatch_tool_async
        try:
            loki.dispatch_tool_async = fake_dispatch
            result = asyncio.run(loki.run_tool_loop_async(
                transcript,
                chat_fn=chat_fn,
                on_event=events.append,
                max_loops=3,
            ))
        finally:
            loki.dispatch_tool_async = old_dispatch

        self.assertEqual(result, "done")
        self.assertEqual(
            [item.get("type") for item in transcript],
            [
                "message", "model_response", "tool_result",
                "model_response",
            ],
        )
        self.assertEqual(transcript[2]["call_id"], "call_1")
        self.assertEqual(formats.item_text(transcript[2]), "file contents")
        self.assertEqual(
            seen_inputs[1],
            ["message", "model_response", "tool_result"],
        )
        self.assertEqual([event.get("type") for event in events], ["tool_call", "tool_result", "assistant_message"])

    def test_autonomous_loop_limit_is_hard_and_closes_pending_call(self):
        transcript = [formats.message_item("user", "keep calling")]
        events = []
        dispatched = []
        response_number = 0

        async def chat_fn(items):
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

        async def chat_fn(items):
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

        async def chat_fn(items):
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

        async def chat_fn(items):
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

        async def chat_fn(items):
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
    def test_allowed_tool_subset_is_the_only_schema_advertised(self):
        seen_tools = []

        async def fake_completion(items, tools, model=None):
            seen_tools.extend(tools)
            return formats.DecodedTurn([
                formats.message_item("assistant", "done"),
            ])

        old_completion = loki.async_chat_completion
        try:
            loki.async_chat_completion = fake_completion
            result = asyncio.run(loki.run_tool_loop_async(
                [formats.message_item("user", "inspect")],
                allowed={"Read", "Grep"},
            ))
        finally:
            loki.async_chat_completion = old_completion

        self.assertEqual(result, "done")
        self.assertEqual(
            {
                tool["function"]["name"]
                for tool in seen_tools
            },
            {"Read", "Grep"},
        )

    def test_toolless_completion_returns_all_assistant_phases(self):
        async def fake_completion(items, tools):
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

        async def chat_fn(items):
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


if __name__ == "__main__":
    unittest.main()
