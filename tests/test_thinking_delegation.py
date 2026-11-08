"""One computation snapshot crosses helper/process boundaries without authority."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from loki_entrypoints import child_environment, configure_container, loki_command
from test_acps import _ACPFrontFixture
from loki_agent import formats, loki, models, protocols, subagents
from loki_agent.credentials import CredentialStore
from loki_agent.sessions import Session
from loki_agent import connections
from loki_agent import model_requests


class ThinkingDelegationTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = Session(runtime_config=connections.make_runtime_config(
            "https://api.anthropic.com/v1", protocols.ANTHROPIC_MESSAGES, model="claude-opus-4-5",
            reasoning_capabilities=models.ReasoningCapabilities(True, ["budget_tokens"], 1024, 6000)))
        self.addCleanup(self.session.response_headers.close)
        self.enterContext(mock.patch.object(loki, "_DEFAULT_SESSION", self.session))

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
        self.session.runtime_config = connections.make_runtime_config(
            "https://api.z.ai/api/paas/v4", protocols.OPENAI_CHAT,
            model="glm-5.2", provider_id="zai")
        env = loki._subagent_env(thinking=loki.capture_turn_settings(), environ={})
        self.assertEqual(env["LOKI_PROVIDER_ID"], "zai")
        self.assertNotIn("LOKI_REASONING_EFFORT_PROFILE", env)
        child = connections.build_config_from_env(credentials=CredentialStore.capture(env))
        self.assertEqual(child.chat_provider.provider_id, "zai")
        self.assertIsNone(child.reasoning_effort_profile)

    async def test_effort_selection_has_one_channel_and_profile_remains_model_data(self):
        profile = models.ReasoningEffortProfile(["low", "high"])
        self.session.runtime_config = connections.make_runtime_config(
            "https://api.anthropic.com/v1", protocols.ANTHROPIC_MESSAGES,
            model="claude-opus-4-5", reasoning_effort_profile=profile)
        snapshot = model_requests.TurnThinkingSettings(effort="high")
        env = loki._subagent_env(thinking=snapshot, environ={"LOKI_REASONING_EFFORT": "low"})
        self.assertNotIn("LOKI_REASONING_EFFORT", env)
        self.assertEqual(json.loads(env["LOKI_TURN_THINKING"])["settings"]["effort"], "high")
        self.assertEqual(json.loads(env["LOKI_REASONING_EFFORT_PROFILE"]), profile.to_dict())
        env["LOKI_REASONING_EFFORT"] = "not-a-supported-effort"
        restored = loki.delegated_turn_settings(CredentialStore.capture(env))
        self.assertEqual(restored.effort, "high")
        self.assertIsNone(self.session.reasoning_effort_preference)

    async def test_binding_and_connection_are_validated(self):
        self.session.runtime_config = connections.make_runtime_config(
            "https://relay.example/v1/messages", protocols.ANTHROPIC_MESSAGES, model="custom")
        loki.thinking_command("mode adaptive")
        snapshot = loki.capture_turn_settings()
        env = loki._subagent_env(thinking=snapshot, environ={})
        restored = loki.delegated_turn_settings(CredentialStore.capture(env))
        self.assertEqual(restored.mode, "adaptive")
        self.assertFalse(hasattr(restored, "trial"))
        payload = json.loads(env["LOKI_TURN_THINKING"])
        self.assertNotIn("trial", payload["settings"])
        payload["settings"]["trial"] = True
        env["LOKI_TURN_THINKING"] = json.dumps(payload)
        with self.assertRaisesRegex(ValueError, "settings"):
            loki.delegated_turn_settings(CredentialStore.capture(env))
        del payload["settings"]["trial"]
        payload["connection"]["endpoint"] = "https://other.example/v1/messages"
        env["LOKI_TURN_THINKING"] = json.dumps(payload)
        with self.assertRaisesRegex(ValueError, "connection"):
            loki.delegated_turn_settings(CredentialStore.capture(env))

    async def test_foreground_and_background_use_captured_snapshot_and_exact_executable(self):
        loki.thinking_command("mode manual budget 2048")
        snapshot = loki.capture_turn_settings()
        self.session.thinking_mode = "off"
        command = loki_command()
        job = mock.Mock(id="j", cwd="cwd", pid=1, pgid=1, status="running", exit_code=0,
                        stdout_path="out", stderr_path="err")
        manager = mock.Mock(run_exec=mock.AsyncMock(return_value=[job, "completed", "answer", ""]),
                            run_background_exec=mock.AsyncMock(return_value=job))
        with mock.patch.object(loki, "current_job_manager", return_value=manager), mock.patch.object(sys, "argv", [command[-1]]):
            await loki.run_agent_async("task", "prompt", thinking=snapshot)
            await loki.run_agent_async("task", "prompt", run_in_background=True, thinking=snapshot)
        for call in [manager.run_exec.await_args, manager.run_background_exec.await_args]:
            self.assertEqual(call.args[0][:len(command)], command)
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
        snapshot = model_requests.TurnThinkingSettings(mode="manual", budget=2048, traces="on")
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
        async def respond(request):
            self.assertLessEqual(len(requests), 3)
            if len(requests) == 1:
                return [{"type": "tool_use", "id": "delegate", "name": "Agent", "input": {
                    "description": "task", "prompt": "answer briefly", "subagent_type": "Explore"}}]
            return [{"type": "text", "text": "child answer"}]

        endpoint, requests, _authorizations = await self._provider(
            respond, protocol=protocols.ANTHROPIC_MESSAGES)
        temporary = tempfile.TemporaryDirectory(prefix='loki-thinking-delegation-')
        self.addCleanup(temporary.cleanup)
        workspace = os.path.join(temporary.name, 'workspace')
        os.mkdir(workspace)
        environment = child_environment(
            PATH=os.environ.get('PATH', ''), HOME=temporary.name, TERM='dumb',
            XDG_CONFIG_HOME=os.path.join(temporary.name, 'config'),
            XDG_STATE_HOME=os.path.join(temporary.name, 'state'),
            LOKI_PROVIDER='anthropic', LOKI_API_BASE=endpoint, LOKI_MODEL='custom', LOKI_STREAM='0',
            LOKI_REASONING_EFFORT='low',
            LOKI_REASONING_EFFORT_PROFILE=json.dumps(models.ReasoningEffortProfile(['low', 'high']).to_dict()))
        configure_container(environment, workspace)
        # A unittest dispatcher is not an owning AppContainer runtime. The real
        # front must create its worker before that worker delegates to Explore.
        front, diagnostics = await self._front(environment, workspace, cleanup=self.addAsyncCleanup)
        await self._request(front, 1, 'initialize', {'protocolVersion': 1})
        opened, _ = await self._request(front, 2, 'session/new', {'cwd': workspace})
        session_id = opened['sessionId']
        for request_id, config_id, value in [
                [3, 'reasoning_effort', 'effort:high'], [4, 'reasoning_traces', 'on']]:
            configured, _ = await self._request(front, request_id, 'session/set_config_option', {
                'sessionId': session_id, 'configId': config_id, 'value': value})
            self.assertEqual(next(option['currentValue'] for option in configured['configOptions']
                                  if option['id'] == config_id), value)
        # A custom model has no advertised native mode menu. Its explicit
        # /thinking command is the supported input, not an invented menu choice.
        local, _ = await self._request(front, 5, 'session/prompt', {
            'sessionId': session_id, 'prompt': [{'type': 'text', 'text': '/thinking mode adaptive'}]})
        self.assertEqual(local, {'stopReason': 'end_turn'})
        self.assertEqual(requests, [])
        result, messages = await self._request(front, 6, 'session/prompt', {
            'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'delegate once'}]})
        self.assertEqual(result, {'stopReason': 'end_turn'})
        self.assertIn('child answer', json.dumps(messages))
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[1]['thinking'], {'type': 'adaptive'})
        self.assertEqual(requests[1]['output_config'], {'effort': 'high'})
        self.assertNotIn('display', requests[1]['thinking'])
        await self._finish(front, diagnostics, session_id)

    async def test_answer_only_explore_captures_once_then_reuses_explicit_settings(self):
        snapshot = model_requests.TurnThinkingSettings(mode="manual", budget=2048, traces="on")
        loop = mock.AsyncMock(return_value="answer")
        with mock.patch.object(loki, "capture_turn_settings", return_value=snapshot) as capture, mock.patch.object(
                loki, "run_tool_loop_async", loop):
            self.assertEqual(await subagents.run_prompt_async("Explore", "search"), "answer")
        capture.assert_called_once_with()
        self.assertEqual(loop.await_args.kwargs["thinking"].mode, "manual")
        self.assertEqual(loop.await_args.kwargs["thinking"].traces, "off")
