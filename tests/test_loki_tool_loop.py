import asyncio
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("LOKI_API_KEY", "test-key")
os.environ.setdefault("LOKI_API_BASE", "https://api.openai.com/v1/responses")
os.environ.setdefault("LOKI_PROVIDER", "openai_responses")

from loki_agent import formats
from loki_agent import loki
from loki_agent import protocols
from loki_agent.connections import ConnectionDescriptor
from loki_agent.credentials import CredentialStore
from loki_agent import savefiles


class ScriptedInputSession:
    def __init__(self, messages):
        self.messages = list(messages)
        self.user_messages = self

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
        names = ["RUNTIME_CONFIG", "model"]
        sentinel = object()
        old_values = {name: loki.__dict__.get(name, sentinel) for name in names}

        try:
            loki.apply_runtime_config(loki.build_config_from_env(env))
            old_provider = loki.RUNTIME_CONFIG.chat_provider

            loki.reinstall_provider(model="model-b")

            self.assertEqual(loki.model, "model-b")
            self.assertEqual(loki.RUNTIME_CONFIG.model, "model-b")
            # A fresh Provider object was built and swapped in.
            self.assertIsNot(loki.RUNTIME_CONFIG.chat_provider, old_provider)
            # Everything else carries over from the previous config.
            self.assertEqual(loki.RUNTIME_CONFIG.chat_provider.kind, protocols.OPENAI_RESPONSES)
            self.assertEqual(loki.RUNTIME_CONFIG.chat_provider.chat_url, "https://example.test/v1/responses")
            self.assertEqual(loki.RUNTIME_CONFIG.chat_provider.max_tokens, 512)
            self.assertEqual(loki.RUNTIME_CONFIG.api_key, "test-key")
            self.assertEqual(loki.RUNTIME_CONFIG.headers["Authorization"], "Bearer test-key")
            # The copied headers field was rebuilt too, not left stale.
            self.assertEqual(loki.RUNTIME_CONFIG.headers,
                             loki.RUNTIME_CONFIG.chat_provider.headers)
        finally:
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value

    def test_reinstall_provider_switches_protocol_per_model(self):
        env = {
            "LOKI_API_BASE": "https://example.test/v1/responses",
            "LOKI_PROVIDER": "openai_responses",
            "LOKI_API_KEY": "test-key",
            "LOKI_MODEL": "model-a",
        }
        names = ["RUNTIME_CONFIG", "model"]
        sentinel = object()
        old_values = {name: loki.__dict__.get(name, sentinel) for name in names}

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

            self.assertEqual(loki.model, "claude-model")
            provider = loki.RUNTIME_CONFIG.chat_provider
            self.assertEqual(provider.kind, protocols.ANTHROPIC_MESSAGES)
            self.assertEqual(provider.chat_url, "https://anthropic.example.test/v1/messages")
            self.assertEqual(provider.headers["x-api-key"], "anthropic-key")
            self.assertEqual(loki.RUNTIME_CONFIG.headers["x-api-key"], "anthropic-key")
            self.assertEqual(loki.RUNTIME_CONFIG.provider_kind, protocols.ANTHROPIC_MESSAGES)
            self.assertEqual(loki.RUNTIME_CONFIG.api_key, "anthropic-key")
        finally:
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value

    def test_reinstall_provider_requires_startup_config(self):
        names = ["RUNTIME_CONFIG"]
        sentinel = object()
        old_values = {name: loki.__dict__.get(name, sentinel) for name in names}

        try:
            loki.RUNTIME_CONFIG = None
            with self.assertRaises(RuntimeError):
                loki.reinstall_provider(model="model-a")
        finally:
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value

    def test_reinstall_preserves_status_only_for_the_same_catalog_entry(self):
        names = ["RUNTIME_CONFIG", "model"]
        old_values = {name: loki.__dict__[name] for name in names}

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
                loki.RUNTIME_CONFIG.model_status, "deprecated")

            loki.reinstall_provider(model="new-model")
            self.assertIsNone(loki.RUNTIME_CONFIG.model_status)
        finally:
            for name, value in old_values.items():
                loki.__dict__[name] = value


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

    def test_build_config_rejects_missing_environment_credentials(self):
        env = {
            "LOKI_API_BASE": "https://example.test/v1/chat/completions",
            "LOKI_PROVIDER": "openai_chat",
        }
        with self.assertRaisesRegex(ValueError, "API credential missing"):
            loki.build_config_from_env(env)

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

    def test_custom_connection_requires_loki_credential(self):
        for credential_name in (
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENCODE_API_KEY"):
            with self.subTest(credential_name=credential_name):
                env = {
                    "LOKI_API_BASE":
                        "https://custom.example.test/v1/chat/completions",
                    credential_name: "must-not-be-sent",
                }
                with self.assertRaisesRegex(
                        ValueError, "set one of: LOKI_API_KEY"):
                    loki.build_config_from_env(env)

    def test_apply_runtime_config_assigns_runtime_globals(self):
        env = {
            "LOKI_API_BASE": "https://example.test/v1/responses",
            "LOKI_PROVIDER": "openai_responses",
            "LOKI_API_KEY": "test-key",
            "LOKI_MODEL": "gpt-test",
        }
        config = loki.build_config_from_env(env)
        names = ["RUNTIME_CONFIG", "model"]
        sentinel = object()
        old_values = {name: loki.__dict__.get(name, sentinel) for name in names}

        try:
            loki.apply_runtime_config(config)

            self.assertIs(loki.RUNTIME_CONFIG, config)
            self.assertEqual(loki.model, "gpt-test")
        finally:
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value

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
            }),
        )
        self.assertEqual(config.api_key, "selected-key")
        self.assertEqual(
            config.headers["Authorization"], "Bearer selected-key")
        self.assertEqual(config.auth_header, None)
        self.assertEqual(config.provider_id, "openrouter")
        self.assertEqual(config.model, "z-ai/glm")
        self.assertEqual(config.model_status, "deprecated")


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        names = [
            "RUNTIME_CONFIG", "CREDENTIALS", "model", "models",
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
        ]
        self.old_values = {name: loki.__dict__[name] for name in names}

    def tearDown(self):
        for name, value in self.old_values.items():
            loki.__dict__[name] = value

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
            asyncio.run(loki.load_models_async())

        self.assertEqual(loki.models, ["first-model", "second-model"])
        self.assertEqual(loki.model, "")
        self.assertEqual(loki.RUNTIME_CONFIG.model, "")

    def test_interactive_startup_does_not_fetch_provider_models(self):
        loki.CREDENTIALS = CredentialStore({
            "LOKI_API_BASE":
                "https://provider.example.test/v1/chat/completions",
            "LOKI_PROVIDER": protocols.OPENAI_CHAT,
            "LOKI_API_KEY": "test-key",
            "LOKI_MODEL": "chosen-model",
        })
        session = ScriptedInputSession([None])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-test.json")
            loader = mock.AsyncMock()
            with mock.patch(
                    "loki_agent.loki.input_session",
                    return_value=session), mock.patch(
                        "loki_agent.loki.new_chat_log_path",
                        return_value=path), mock.patch(
                            "loki_agent.loki.restore_output_area_after_input"
                        ), mock.patch(
                            "loki_agent.loki.load_models_async",
                            new=loader):
                status = asyncio.run(loki.async_main([]))

        loader.assert_not_awaited()
        self.assertEqual(status, 0)
        self.assertEqual(loki.model, "chosen-model")

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
                "loki_agent.loki.load_models_async",
                new=loader), mock.patch(
                    "loki_agent.loki.run_subagent_cli_async",
                    new=runner), contextlib.redirect_stderr(stderr):
            status = asyncio.run(loki.async_main(["--headless"]))

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
                "loki_agent.loki.run_subagent_cli_async",
                new=runner), contextlib.redirect_stderr(stderr):
            status = asyncio.run(loki.async_main(["--headless"]))

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
                    "loki_agent.loki.input_session",
                    return_value=session), contextlib.redirect_stderr(stderr):
                status = asyncio.run(
                    loki.async_main([f"--resume={missing}"]))

        self.assertEqual(status, 1)
        self.assertIn("Could not resume chat:", stderr.getvalue())

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
                    "loki_agent.loki.input_session",
                    return_value=session), mock.patch(
                        "loki_agent.loki.new_chat_log_path",
                        return_value=path), mock.patch(
                            "loki_agent.loki.restore_output_area_after_input"
                        ), mock.patch(
                            "loki_agent.loki.run_terminal_turn_async",
                            new=runner), contextlib.redirect_stderr(stderr):
                status = asyncio.run(loki.async_main([]))

        runner.assert_not_awaited()
        self.assertEqual(status, 0)
        self.assertNotIn(
            "do not send this",
            [formats.item_text(item) for item in loki.transcript_items],
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
            loki.models = ["current-model", "other-model"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-test.json")
            loader = mock.AsyncMock(side_effect=load_provider_models)
            with mock.patch(
                    "loki_agent.loki.input_session",
                    return_value=session), mock.patch(
                        "loki_agent.loki.new_chat_log_path",
                        return_value=path), mock.patch(
                            "loki_agent.loki.restore_output_area_after_input"
                        ), mock.patch(
                            "loki_agent.loki.load_models_async",
                            new=loader), mock.patch(
                                "loki_agent.loki.modelsdev."
                                "run_model_picker_async",
                                new=mock.AsyncMock(
                                    side_effect=OSError("offline"))), \
                    mock.patch(
                        "loki_agent.loki.modelsdev."
                        "run_flat_model_picker_async",
                        new=mock.AsyncMock(return_value=None)):
                status = asyncio.run(loki.async_main([]))

        loader.assert_awaited_once()
        self.assertEqual(status, 0)
        self.assertEqual(loki.model, "current-model")
        self.assertEqual(loki.RUNTIME_CONFIG.model, "current-model")

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
            loki.models = ["first-model", "selected-model"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-test.json")
            loader = mock.AsyncMock(side_effect=load_provider_models)
            with mock.patch(
                    "loki_agent.loki.input_session",
                    return_value=session), mock.patch(
                        "loki_agent.loki.new_chat_log_path",
                        return_value=path), mock.patch(
                            "loki_agent.loki.restore_output_area_after_input"
                        ), mock.patch(
                            "loki_agent.loki.load_models_async",
                            new=loader), mock.patch(
                                "loki_agent.loki.modelsdev."
                                "run_model_picker_async",
                                new=mock.AsyncMock(
                                    side_effect=OSError("offline"))), \
                    mock.patch(
                        "loki_agent.loki.modelsdev."
                        "run_flat_model_picker_async",
                        new=mock.AsyncMock(return_value="selected-model")):
                status = asyncio.run(loki.async_main([]))

            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)

        loader.assert_awaited_once()
        self.assertEqual(status, 0)
        self.assertEqual(loki.model, "selected-model")
        self.assertEqual(loki.RUNTIME_CONFIG.model, "selected-model")
        self.assertEqual(
            loki.RUNTIME_CONFIG.chat_provider.models_url, models_url)
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
                    "loki_agent.loki.input_session",
                    return_value=session), mock.patch(
                        "loki_agent.loki.new_chat_log_path",
                        return_value=path), mock.patch(
                            "loki_agent.loki.restore_output_area_after_input"
                        ), mock.patch(
                            "loki_agent.loki.modelsdev."
                            "run_model_picker_async",
                            new=mock.AsyncMock(return_value=(
                                "provider", provider_entry, model_entry))):
                status = asyncio.run(loki.async_main([]))

            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)

        self.assertEqual(status, 0)
        self.assertEqual(loki.RUNTIME_CONFIG.model_status, "deprecated")
        self.assertIn(
            "Model: old-model (deprecated); /model", loki.status_text())
        self.assertEqual(
            saved["session_state"]["connection"]["model_status"],
            "deprecated",
        )


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
                        "loki_agent.loki.CredentialStore.capture",
                        return_value=CredentialStore({})), mock.patch(
                            "loki_agent.loki.signal.signal"), mock.patch(
                                "loki_agent.loki.signal.pthread_sigmask"
                            ), mock.patch(
                                "loki_agent.loki.initialize_terminal_overlay"
                            ), mock.patch(
                                "loki_agent.loki.asyncio.run",
                                side_effect=finish), mock.patch(
                                    "loki_agent.loki."
                                    "restore_terminal_overlay",
                                    side_effect=OSError("restore failed")
                                ), mock.patch.object(
                                    loki, "chat_log_path", None
                                ), contextlib.redirect_stderr(stderr):
                    status = loki.main()

                self.assertEqual(status, expected_status)
        finally:
            loki.CREDENTIALS = old_credentials

        self.assertIn("Cleanup error: OSError: restore failed",
                      stderr.getvalue())


class StatusTextTests(unittest.TestCase):
    def test_status_text_includes_short_api_base_before_model_without_url_secrets(self):
        names = ["RUNTIME_CONFIG", "model", "shell_cwd"]
        sentinel = object()
        old_values = {name: loki.__dict__.get(name, sentinel) for name in names}

        try:
            loki.shell_cwd = loki.STARTUP_CWD
            chat_provider = protocols.Provider(
                kind=protocols.OPENAI_CHAT,
                input_url="https://user:pass@example.test:8443/base/path/v1/chat/completions?token=secret#fragment",
                chat_url="https://example.test:8443/base/path/chat/completions",
                models_url=None,
                model_urls=[],
                headers={},
                max_tokens=4096,
            )
            loki.RUNTIME_CONFIG = loki.RuntimeConfig(
                url=chat_provider.input_url,
                provider_kind=chat_provider.kind,
                netloc="example.test:8443",
                api_key="secret",
                chat_provider=chat_provider,
                headers={},
                model="model-x"
            )
            loki.model = "model-x"

            text = loki.status_text()
        finally:
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value

        self.assertEqual(
            text,
            "Remote: API: example.test:8443/base/path; Model: model-x; /model\n"
            f"Local: mode={loki.AGENT_MODE}; CWD: {loki.STARTUP_CWD}; /pwd, /cd DIR, /ps, !foo, /quit",
        )
        self.assertNotIn("user", text)
        self.assertNotIn("pass", text)
        self.assertNotIn("token", text)
        self.assertNotIn("secret", text)

    def test_status_text_marks_a_deprecated_selected_model(self):
        names = ["RUNTIME_CONFIG", "model", "shell_cwd"]
        old_values = {name: loki.__dict__[name] for name in names}

        try:
            loki.shell_cwd = loki.STARTUP_CWD
            loki.apply_runtime_config(loki.make_runtime_config(
                "https://example.test/v1",
                protocols.OPENAI_CHAT,
                "test-key",
                model="old-model",
                credential_env="EXAMPLE_API_KEY",
                model_status="deprecated",
            ))

            text = loki.status_text()
        finally:
            for name, value in old_values.items():
                loki.__dict__[name] = value

        self.assertIn("Model: old-model (deprecated); /model", text)


class TerminalOverlayLifecycleTests(unittest.TestCase):
    class RecordingTerminal:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return lambda *args: self.calls.append((name, *args))

    def test_initialize_clears_only_from_cursor_to_end(self):
        terminal = self.RecordingTerminal()

        loki.initialize_terminal_overlay(terminal)

        self.assertEqual(terminal.calls, [
            ("enable_bracketed_paste_mode",),
            ("enable_origin_mode",),
            ("clear_to_end_of_screen",),
            ("reset_colors_and_flags",),
            ("set_clipping_region", *loki.terminals.output_area),
            ("goto_position", 1, 1),
            ("flush",),
        ])
        self.assertNotIn(("clear_screen",), terminal.calls)

    def test_restore_resets_scroll_region_then_clears_to_end(self):
        terminal = self.RecordingTerminal()

        loki.restore_terminal_overlay(terminal)

        self.assertEqual(terminal.calls, [
            ("disable_bracketed_paste_mode",),
            ("disable_clipping_regions",),
            ("disable_origin_mode",),
            ("reset_colors_and_flags",),
            ("goto_position", loki.terminals.input_area[0], 1),
            ("clear_to_end_of_screen",),
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
            formats.response_metadata_item(
                "openai",
                "openai_chat",
                {"id": "resp_1", "model": "glm-test", "status": "completed"},
            ),
            formats.message_item("assistant", "hi there"),
            formats.tool_call_item("call_1", "Read", {"file_path": "README.md"}),
            formats.tool_result_item("call_1", "file contents", name="Read"),
        ]

        text = loki.ResumeTranscriptRenderer(assistant_label="Assistant").render(items)

        self.assertEqual(
            text,
            "User: hello\n\n"
            "glm-test: hi there\n\n"
            "Tool call: Read\n"
            "{'file_path': 'README.md'}\n\n"
            "Tool result: Read\n"
            "file contents",
        )
        self.assertNotIn("internal startup instruction", text)
        self.assertNotIn("response_metadata", text)
        self.assertNotIn("provider_raw", text)


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
        sentinel = object()
        old_values = {name: loki.__dict__.get(name, sentinel) for name in names}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, ".loki", "chats", "chat-test.json")
                loki.new_chat_log(path)

                self.assertTrue(os.path.isdir(os.path.dirname(path)))
                self.assertEqual(loki.chat_log_path, path)
                self.assertTrue(loki.chat_log_dirty)
                self.assertFalse(os.path.exists(path))
        finally:
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value


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
        saved_terminal = loki.terminal

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

        loki.terminal = _FakeTerminal()

        def restore():
            loki.CHAT_LOG_DIR = saved_log_dir
            loki.terminal = saved_terminal

        return restore, session

    def test_picker_selects_by_number(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "aaa", '{"text":"alpha chat"}', mtime=1000)
            self._write_chat(tmpdir, "bbb", '{"text":"beta chat"}', mtime=2000)
            self._write_chat(tmpdir, "ccc", '{"text":"gamma chat"}', mtime=3000)
            restore, session = self._make_picker(tmpdir, ["2"])
            try:
                result = asyncio.run(loki.run_session_picker_async(session))
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
                        loki.run_session_picker_async(session))
            finally:
                restore()

            self.assertTrue(result.endswith("chat-aaa.json"))
            self.assertTrue(
                output.getvalue().startswith("\nSaved sessions:\n"))

    def test_picker_finishes_clear_before_returning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "aaa", '{"text":"alpha"}', mtime=1000)
            restore, session = self._make_picker(tmpdir, ["1"])
            picker_terminal = loki.terminal
            try:
                result = asyncio.run(loki.run_session_picker_async(session))
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
                result = asyncio.run(loki.run_session_picker_async(session))
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
                result = asyncio.run(loki.run_session_picker_async(session))
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
                result = asyncio.run(loki.run_session_picker_async(session))
            finally:
                restore()
            # After clearing, both visible; "2" = bbb (newest last).
            self.assertTrue(result.endswith("chat-bbb.json"))

    def test_picker_empty_input_cancels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chat(tmpdir, "aaa", '{"text":"alpha"}', mtime=1000)
            restore, session = self._make_picker(tmpdir, [""])
            try:
                result = asyncio.run(loki.run_session_picker_async(session))
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
                result = asyncio.run(loki.run_session_picker_async(session))
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
                result = asyncio.run(loki.run_session_picker_async(session))
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
                result = asyncio.run(loki.run_session_picker_async(session))
            finally:
                restore()
            self.assertTrue(result.endswith("chat-aaa.json"))


class ShellCwdTests(unittest.TestCase):
    def test_change_shell_cwd_does_not_change_process_cwd(self):
        names = ["shell_cwd", "previous_shell_cwd"]
        old_values = {name: loki.__dict__[name] for name in names}
        process_cwd = os.getcwd()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loki.change_shell_cwd(tmpdir)

                self.assertEqual(loki.shell_cwd, tmpdir)
                self.assertEqual(os.getcwd(), process_cwd)
                self.assertEqual(loki._resolve_path("file.txt"), os.path.join(tmpdir, "file.txt"))
        finally:
            for name, value in old_values.items():
                loki.__dict__[name] = value

    def test_bash_runs_in_shell_cwd(self):
        names = ["shell_cwd", "previous_shell_cwd", "job_manager"]
        old_values = {name: loki.__dict__[name] for name in names}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                workdir = os.path.join(tmpdir, "work")
                os.mkdir(workdir)
                loki.job_manager = loki.JobManager(os.path.join(tmpdir, "jobs"))
                loki.change_shell_cwd(workdir)

                result = asyncio.run(loki.run_bash_async("pwd"))
                jobs = list(loki.job_manager.jobs.values())
        finally:
            for name, value in old_values.items():
                loki.__dict__[name] = value

        self.assertIn("[stdout]\n" + workdir, result)
        self.assertEqual(os.path.basename(jobs[0].stdout_path), "stdout.log")
        self.assertEqual(os.path.basename(jobs[0].stderr_path), "stderr.log")

    def test_save_chat_log_persists_shell_cwd(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "shell_cwd",
            "previous_shell_cwd",
        ]
        sentinel = object()
        old_values = {name: loki.__dict__.get(name, sentinel) for name in names}

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
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value

        self.assertEqual(blob["session_state"]["shell_cwd"], cwd)

    def test_save_chat_log_persists_connection_without_credential_value(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
            "RUNTIME_CONFIG", "model",
        ]
        sentinel = object()
        old_values = {
            name: loki.__dict__.get(name, sentinel) for name in names}

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
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value

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

    def test_loading_and_clean_cleanup_leave_chat_bytes_unchanged(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "RUNTIME_CONFIG",
            "shell_cwd", "previous_shell_cwd",
        ]
        old_values = {name: loki.__dict__[name] for name in names}

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
                loki.RUNTIME_CONFIG = None

                loki.load_chat_log(path)
                saved = loki.save_chat_log()

                self.assertFalse(saved)
                self.assertFalse(loki.chat_log_dirty)
                self.assertEqual(pathlib.Path(path).read_bytes(), original)
                self.assertEqual(
                    loki.session_state["connection"], descriptor.to_dict())
                self.assertEqual(
                    loki.session_state["future_field"], {"keep": True})
        finally:
            for name, value in old_values.items():
                loki.__dict__[name] = value

    def test_later_save_preserves_unavailable_loaded_connection(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "RUNTIME_CONFIG",
            "shell_cwd", "previous_shell_cwd",
        ]
        old_values = {name: loki.__dict__[name] for name in names}

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
                loki.RUNTIME_CONFIG = None

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
            for name, value in old_values.items():
                loki.__dict__[name] = value

    def test_resumed_chat_does_not_adopt_explicit_runtime_connection(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "RUNTIME_CONFIG", "model",
            "shell_cwd", "previous_shell_cwd",
        ]
        old_values = {name: loki.__dict__[name] for name in names}

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
            for name, value in old_values.items():
                loki.__dict__[name] = value

    def test_chat_save_atomically_replaces_existing_snapshot(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
        ]
        old_values = {name: loki.__dict__[name] for name in names}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                loki.new_chat_log(path)
                loki.save_chat_log()
                first_inode = os.stat(path).st_ino

                loki.transcript_items.append(
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
            for name, value in old_values.items():
                loki.__dict__[name] = value

    def test_failed_atomic_publish_preserves_previous_snapshot(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos",
        ]
        old_values = {name: loki.__dict__[name] for name in names}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "chat-test.json")
                loki.new_chat_log(path)
                loki.save_chat_log()
                original = pathlib.Path(path).read_bytes()

                loki.transcript_items.append(
                    formats.message_item("user", "must not publish"))
                loki.mark_chat_log_dirty()
                with mock.patch(
                        "loki_agent.loki.os.replace",
                        side_effect=OSError("publish failed")):
                    with self.assertRaisesRegex(OSError, "publish failed"):
                        loki.save_chat_log()

                self.assertEqual(pathlib.Path(path).read_bytes(), original)
                self.assertTrue(loki.chat_log_dirty)
                self.assertEqual(
                    [name for name in os.listdir(tmpdir)
                     if name.endswith(".tmp")],
                    [],
                )
        finally:
            for name, value in old_values.items():
                loki.__dict__[name] = value

    def test_successful_model_selection_replaces_resumed_connection(self):
        names = [
            "chat_log_path", "session_state", "chat_log_dirty",
            "transcript_items", "session_todos", "RUNTIME_CONFIG", "model",
            "shell_cwd", "previous_shell_cwd",
        ]
        old_values = {name: loki.__dict__[name] for name in names}

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
            for name, value in old_values.items():
                loki.__dict__[name] = value

    def test_load_session_state_restores_shell_cwd(self):
        names = ["shell_cwd", "previous_shell_cwd"]
        old_values = {name: loki.__dict__[name] for name in names}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loki.load_session_state({"shell_cwd": tmpdir})

                self.assertEqual(loki.shell_cwd, tmpdir)
        finally:
            for name, value in old_values.items():
                loki.__dict__[name] = value

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
                loki.confirm_saved_connection_async(descriptor, no_session))
            accepted = asyncio.run(
                loki.confirm_saved_connection_async(descriptor, yes_session))

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
        names = ["RUNTIME_CONFIG", "model"]
        old_values = {name: loki.__dict__[name] for name in names}
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
                ))
                child_env = loki._subagent_env()
        finally:
            for name, value in old_values.items():
                loki.__dict__[name] = value

        self.assertEqual(child_env["LOKI_API_KEY"], "active-key")
        self.assertNotIn("OPENROUTER_API_KEY", child_env)


class ResponsesToolLoopTests(unittest.TestCase):
    def test_function_call_only_response_executes_tool_and_continues(self):
        transcript = [formats.message_item("user", "read README")]
        seen_inputs = []
        events = []

        async def chat_fn(items):
            seen_inputs.append([item.get("type") for item in items])
            if len(seen_inputs) == 1:
                return [
                    formats.response_metadata_item(
                        "openai",
                        "openai_responses",
                        {"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-test"},
                    ),
                    formats.tool_call_item("call_1", "Read", {"file_path": "README.md"}),
                ]
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
            ["message", "response_metadata", "tool_call", "tool_result", "message"],
        )
        self.assertEqual(transcript[3]["tool_call_id"], "call_1")
        self.assertEqual(formats.item_text(transcript[3]), "file contents")
        self.assertEqual(seen_inputs[1], ["message", "response_metadata", "tool_call", "tool_result"])
        self.assertEqual([event.get("type") for event in events], ["tool_call", "assistant_message"])


if __name__ == "__main__":
    unittest.main()
