import asyncio
import os
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("LOKI_API_KEY", "test-key")
os.environ.setdefault("LOKI_API_BASE", "https://api.openai.com/v1/responses")
os.environ.setdefault("LOKI_PROVIDER", "openai_responses")

from day_agent import formats
from day_agent import loki
from day_agent import protocols


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
            sys.argv = [os.path.abspath("day_agent/__main__.py")]

            argv = loki._subagent_argv("Explore", "inspect this")
        finally:
            sys.argv = old_argv

        self.assertEqual(argv, [
            sys.executable,
            "-m",
            "day_agent",
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
