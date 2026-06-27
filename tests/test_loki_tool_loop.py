import asyncio
import json
import os
import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("LOKI_API_KEY", "test-key")
os.environ.setdefault("LOKI_API_BASE", "https://api.openai.com/v1/responses")
os.environ.setdefault("LOKI_PROVIDER", "openai_responses")

from loki_agent import formats
from loki_agent import loki
from loki_agent import protocols


class RuntimeConfigTests(unittest.TestCase):
    def test_build_config_uses_explicit_env_key_without_secret_lookup(self):
        env = {
            "LOKI_API_BASE": "https://api.deepseek.com/anthropic",
            "LOKI_PROVIDER": "anthropic_messages",
            "LOKI_API_KEY": "loki-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "OPENAI_API_KEY": "openai-key",
            "LOKI_MODEL": "deepseek-test",
            "LOKI_MAX_TOKENS": "123",
            "ANTHROPIC_VERSION": "2024-01-01",
        }

        def secret_lookup(domain):
            raise AssertionError(f"secret lookup should not be called for {domain}")

        config = loki.build_config_from_env(env, secret_lookup)

        self.assertEqual(config.url, "https://api.deepseek.com/anthropic")
        self.assertEqual(config.provider_kind, protocols.ANTHROPIC_MESSAGES)
        self.assertEqual(config.netloc, "api.deepseek.com")
        self.assertEqual(config.api_key, "loki-key")
        self.assertEqual(config.model, "deepseek-test")
        self.assertEqual(config.chat_provider.max_tokens, 123)
        self.assertEqual(config.headers["x-api-key"], "loki-key")
        self.assertEqual(config.headers["anthropic-version"], "2024-01-01")
        self.assertNotIn("LOKI_API_KEY", env)
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("OPENAI_API_KEY", env)

    def test_build_config_uses_secret_lookup_when_env_keys_are_absent(self):
        env = {
            "LOKI_API_BASE": "https://example.test/v1/chat/completions",
            "LOKI_PROVIDER": "openai_chat",
        }
        calls = []

        def secret_lookup(domain):
            calls.append(domain)
            return "secret-key"

        config = loki.build_config_from_env(env, secret_lookup)

        self.assertEqual(calls, ["example.test"])
        self.assertEqual(config.api_key, "secret-key")
        self.assertEqual(config.provider_kind, protocols.OPENAI_CHAT)
        self.assertEqual(config.headers["Authorization"], "Bearer secret-key")

    def test_apply_runtime_config_assigns_runtime_globals(self):
        env = {
            "LOKI_API_BASE": "https://example.test/v1/responses",
            "LOKI_PROVIDER": "openai_responses",
            "LOKI_API_KEY": "test-key",
            "LOKI_MODEL": "gpt-test",
        }
        config = loki.build_config_from_env(env, lambda domain: "")
        names = ["url", "provider_kind", "netloc", "api_key", "chat_provider", "headers", "model"]
        sentinel = object()
        old_values = {name: loki.__dict__.get(name, sentinel) for name in names}

        try:
            loki.apply_runtime_config(config)

            self.assertEqual(loki.url, "https://example.test/v1/responses")
            self.assertEqual(loki.provider_kind, protocols.OPENAI_RESPONSES)
            self.assertEqual(loki.netloc, "example.test")
            self.assertEqual(loki.api_key, "test-key")
            self.assertIs(loki.chat_provider, config.chat_provider)
            self.assertEqual(loki.headers["Authorization"], "Bearer test-key")
            self.assertEqual(loki.model, "gpt-test")
        finally:
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value


class StatusTextTests(unittest.TestCase):
    def test_status_text_includes_short_api_base_before_model_without_url_secrets(self):
        names = ["chat_provider", "model", "shell_cwd"]
        sentinel = object()
        old_values = {name: loki.__dict__.get(name, sentinel) for name in names}

        try:
            loki.shell_cwd = loki.STARTUP_CWD
            loki.chat_provider = protocols.Provider(
                kind=protocols.OPENAI_CHAT,
                input_url="https://user:pass@example.test:8443/base/path/v1/chat/completions?token=secret#fragment",
                chat_url="https://example.test:8443/base/path/chat/completions",
                models_url=None,
                model_urls=[],
                headers={},
                max_tokens=4096,
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
            f"Local: CWD: {loki.STARTUP_CWD}; /pwd, /cd DIR, !foo, /quit",
        )
        self.assertNotIn("user", text)
        self.assertNotIn("pass", text)
        self.assertNotIn("token", text)
        self.assertNotIn("secret", text)


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
        names = ["chat_log", "transcript_items", "session_todos"]
        sentinel = object()
        old_values = {name: loki.__dict__.get(name, sentinel) for name in names}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, ".loki", "chats", "chat-test.json")
                loki.new_chat_log(path)

                self.assertTrue(os.path.isdir(os.path.dirname(path)))
                self.assertEqual(loki.chat_log.name, path)
        finally:
            if "chat_log" in loki.__dict__:
                try:
                    loki.chat_log.close()
                except OSError:
                    pass
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value


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
        names = ["chat_log", "transcript_items", "session_todos", "shell_cwd", "previous_shell_cwd"]
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
                loki.chat_log.close()

                with open(path, "r", encoding="utf-8") as f:
                    blob = json.load(f)
        finally:
            if "chat_log" in loki.__dict__:
                try:
                    loki.chat_log.close()
                except OSError:
                    pass
            for name, value in old_values.items():
                if value is sentinel:
                    loki.__dict__.pop(name, None)
                else:
                    loki.__dict__[name] = value

        self.assertEqual(blob["session_state"]["shell_cwd"], cwd)

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

        async def fake_dispatch(fn_name, args, allowed=None):
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
