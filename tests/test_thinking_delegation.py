"""One computation snapshot crosses helper/process boundaries without authority."""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import formats, loki, models, protocols, subagents
from loki_agent.credentials import CredentialStore
from loki_agent.sessions import Session


class ThinkingDelegationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = Session(runtime_config=loki.make_runtime_config(
            "https://api.anthropic.com/v1", protocols.ANTHROPIC_MESSAGES, model="claude-opus-4-5",
            reasoning_capabilities=models.ReasoningCapabilities(True, ["budget_tokens"], 1024, 6000)))
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)

    async def test_envelope_round_trip_preserves_computation_not_presentation(self):
        loki.thinking_command("mode manual budget 2048")
        self.session.reasoning_traces = "on"
        snapshot = loki.capture_turn_settings()
        self.session.thinking_mode = "off"
        env = loki._subagent_env(thinking=snapshot, environ={
            "LOKI_TURN_THINKING": "stale", "LOKI_REASONING_TRACES": "on", "OPENAI_API_KEY": "secret"})
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("LOKI_REASONING_TRACES", env)
        payload = json.loads(env["LOKI_TURN_THINKING"])
        self.assertNotIn("traces", payload["settings"])
        config = self.session.runtime_config
        restored = loki.delegated_turn_settings(CredentialStore.capture(env))
        self.assertEqual(restored.mode, "manual")
        self.assertEqual(restored.budget, 2048)
        self.assertEqual(restored.traces, "off")
        self.assertEqual(self.session.runtime_config.reasoning_capabilities, config.reasoning_capabilities)
        self.assertIs(self.session.runtime_config.auth_spec, config.auth_spec)
        self.assertEqual(self.session.runtime_config.chat_provider.chat_url, config.chat_provider.chat_url)
        recursive = loki._subagent_env(thinking=restored, environ={})
        self.assertEqual(json.loads(recursive["LOKI_TURN_THINKING"])["settings"], payload["settings"])

    async def test_profileless_provider_identity_survives_environment_round_trip(self):
        self.session.runtime_config = loki.make_runtime_config(
            "https://api.z.ai/api/paas/v4", protocols.OPENAI_CHAT,
            model="glm-5.2", provider_id="zai")
        env = loki._subagent_env(thinking=loki.capture_turn_settings(), environ={})
        self.assertEqual(env["LOKI_PROVIDER_ID"], "zai")
        self.assertNotIn("LOKI_REASONING_EFFORT_PROFILE", env)
        child = loki.build_config_from_env(credentials=CredentialStore.capture(env))
        self.assertEqual(child.chat_provider.provider_id, "zai")
        self.assertIsNone(child.reasoning_effort_profile)

    async def test_effort_selection_has_one_channel_and_profile_remains_model_data(self):
        profile = models.ReasoningEffortProfile(["low", "high"])
        self.session.runtime_config = loki.make_runtime_config(
            "https://api.anthropic.com/v1", protocols.ANTHROPIC_MESSAGES,
            model="claude-opus-4-5", reasoning_effort_profile=profile)
        snapshot = loki.TurnThinkingSettings(effort="high")
        env = loki._subagent_env(thinking=snapshot, environ={"LOKI_REASONING_EFFORT": "low"})
        self.assertNotIn("LOKI_REASONING_EFFORT", env)
        self.assertEqual(json.loads(env["LOKI_TURN_THINKING"])["settings"]["effort"], "high")
        self.assertEqual(json.loads(env["LOKI_REASONING_EFFORT_PROFILE"]), profile.to_dict())
        restored = loki.delegated_turn_settings(CredentialStore.capture(env))
        self.assertEqual(restored.effort, "high")
        self.assertIsNone(self.session.reasoning_effort_preference)

    async def test_binding_and_trial_permission_are_validated(self):
        self.session.runtime_config = loki.make_runtime_config(
            "https://relay.example/v1/messages", protocols.ANTHROPIC_MESSAGES, model="custom")
        loki.thinking_command("mode adaptive")
        snapshot = loki.capture_turn_settings()
        env = loki._subagent_env(thinking=snapshot, environ={})
        restored = loki.delegated_turn_settings(CredentialStore.capture(env))
        self.assertEqual(restored.mode, "adaptive")
        self.assertIsNotNone(restored.trial)
        payload = json.loads(env["LOKI_TURN_THINKING"])
        payload["settings"]["trial"] = False
        env["LOKI_TURN_THINKING"] = json.dumps(payload)
        with self.assertRaisesRegex(ValueError, "parent"):
            loki.delegated_turn_settings(CredentialStore.capture(env))
        payload["settings"]["trial"] = True
        payload["connection"]["endpoint"] = "https://other.example/v1/messages"
        env["LOKI_TURN_THINKING"] = json.dumps(payload)
        with self.assertRaisesRegex(ValueError, "connection"):
            loki.delegated_turn_settings(CredentialStore.capture(env))

    async def test_foreground_and_background_use_captured_snapshot_and_exact_executable(self):
        loki.thinking_command("mode manual budget 2048")
        snapshot = loki.capture_turn_settings()
        self.session.thinking_mode = "off"
        executable = os.path.abspath("loki.py")
        job = mock.Mock(id="j", cwd="cwd", pid=1, pgid=1, status="running", exit_code=0,
                        stdout_path="out", stderr_path="err")
        manager = mock.Mock(run_exec=mock.AsyncMock(return_value=[job, "completed", "answer", ""]),
                            run_background_exec=mock.AsyncMock(return_value=job))
        with mock.patch.object(loki, "current_job_manager", return_value=manager), mock.patch.object(sys, "argv", [executable]):
            await loki.run_agent_async("task", "prompt", thinking=snapshot)
            await loki.run_agent_async("task", "prompt", run_in_background=True, thinking=snapshot)
        for call in [manager.run_exec.await_args, manager.run_background_exec.await_args]:
            self.assertEqual(call.args[0][0], executable)
            data = json.loads(call.kwargs["env"]["LOKI_TURN_THINKING"])
            self.assertEqual(data["settings"]["mode"], "manual")
            self.assertEqual(data["settings"]["budget"], 2048)
            self.assertTrue(call.kwargs["session_owned"])

    async def test_tool_context_reaches_webfetch_and_recursive_agent(self):
        loki.thinking_command("mode manual budget 2048")
        snapshot = loki.capture_turn_settings()
        context = {"thinking": snapshot, "reasoning_effort": snapshot.effort}
        agent = mock.AsyncMock(return_value="answer")
        fetch = mock.AsyncMock(return_value="answer")
        with mock.patch.object(loki, "run_agent_async", agent), mock.patch.object(loki, "run_webfetch_async", fetch):
            await loki._handle_agent_async({"prompt": "search"}, context)
            await loki._handle_webfetch_async({"url": "https://example.test", "prompt": "read"}, context)
        self.assertIs(agent.await_args.kwargs["thinking"], snapshot)
        self.assertIs(fetch.await_args.kwargs["thinking"], snapshot)

    async def test_toolless_helper_keeps_settings_and_returns_answer_only(self):
        snapshot = loki.TurnThinkingSettings(mode="manual", budget=2048, traces="on")
        response = formats.DecodedTurn([
            {"type": "anthropic_thinking", "thinking": "hidden", "signature": "opaque"},
            formats.message_item("assistant", "answer")])
        completion = mock.AsyncMock(return_value=response)
        with mock.patch.object(loki, "async_chat_completion", completion):
            result = await loki.run_toolless_completion_async([], thinking=snapshot)
        forwarded = completion.await_args.kwargs["thinking"]
        self.assertEqual(forwarded.mode, "manual")
        self.assertEqual(forwarded.budget, 2048)
        self.assertEqual(forwarded.traces, "off")
        self.assertEqual(snapshot.traces, "on")
        self.assertEqual(result, "answer")

    async def test_real_sanctioned_child_receives_one_turn_computation(self):
        requests = []

        async def server_request(reader, writer):
            try:
                header = await reader.readuntil(b"\r\n\r\n")
                length = next(int(line.split(b":", 1)[1]) for line in header.split(b"\r\n")
                              if line.lower().startswith(b"content-length:"))
                requests.append(json.loads(await reader.readexactly(length)))
                response = json.dumps({
                    "type": "message", "role": "assistant", "model": "custom",
                    "content": [{"type": "text", "text": "child answer"}], "stop_reason": "end_turn"}).encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                             + str(len(response)).encode() + b"\r\nConnection: close\r\n\r\n" + response)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(server_request, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            self.session.runtime_config = loki.make_runtime_config(
                f"http://127.0.0.1:{port}/v1/messages", protocols.ANTHROPIC_MESSAGES, model="custom",
                reasoning_effort_profile=models.ReasoningEffortProfile(["low", "high"]))
            loki.thinking_command("effort high mode adaptive")
            self.session.reasoning_traces = "on"
            snapshot = loki.capture_turn_settings()
            self.session.credential_authority = loki.authentications.CredentialBroker()
            with tempfile.TemporaryDirectory() as directory:
                manager = loki.JobManager(directory)
                self.session.job_manager = manager
                try:
                    export_environment = loki._subagent_env

                    def conflicting_ambient_preference(*args, **kwargs):
                        env = export_environment(*args, **kwargs)
                        env["LOKI_REASONING_EFFORT"] = "not-a-supported-effort"
                        return env

                    with mock.patch.object(sys, "argv", [os.path.abspath("loki.py")]), mock.patch.object(
                            loki, "_subagent_env", conflicting_ambient_preference):
                        result = await loki.run_agent_async("task", "answer briefly", thinking=snapshot)
                    self.assertIn("child answer", result)
                finally:
                    await manager.close_session_owned()
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]["thinking"], {"type": "adaptive"})
            self.assertEqual(requests[0]["output_config"], {"effort": "high"})
            self.assertNotIn("display", requests[0]["thinking"])
        finally:
            server.close()
            await server.wait_closed()

    async def test_answer_only_explore_captures_once_then_reuses_explicit_settings(self):
        snapshot = loki.TurnThinkingSettings(mode="manual", budget=2048, traces="on")
        loop = mock.AsyncMock(return_value="answer")
        with mock.patch.object(loki, "capture_turn_settings", return_value=snapshot) as capture, mock.patch.object(
                loki, "run_tool_loop_async", loop):
            self.assertEqual(await subagents.run_prompt_async("Explore", "search"), "answer")
        capture.assert_called_once_with()
        self.assertEqual(loop.await_args.kwargs["thinking"].mode, "manual")
        self.assertEqual(loop.await_args.kwargs["thinking"].traces, "off")
