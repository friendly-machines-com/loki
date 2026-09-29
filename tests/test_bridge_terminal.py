import asyncio
import contextlib
import getopt
import io
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from loki_agent import model_commands, models, terminal_frontend as front, terminals
from loki_agent.bridge_sessions import BridgeSession
from loki_agent.credentials import CredentialStore
from loki_agent.submissions import Submission
from loki_entrypoints import child_environment, configure_container, entrypoint


class ModelCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.credentials = CredentialStore({"EXAMPLE_API_KEY": "secret"})
        self.catalog = {
            pid: {"id": pid, "name": pid,
                  "npm": "@ai-sdk/openai-compatible",
                  "api": f"https://{pid}.invalid/v1",
                  "env": ["EXAMPLE_API_KEY"],
                  "models": {model_id: {"id": model_id, "name": "Commodity"}}}
            for pid, model_id in (("first", "model-a"), ("second", "model-b"))}
        self.index = mock.patch.object(
            models, "ensure_index", mock.AsyncMock(return_value=(
                self.catalog, models.build_groups(self.catalog))))
        self.index.start()
        self.addCleanup(self.index.stop)

    async def run_command(self, text, **kwargs):
        return await model_commands.run(
            text, credentials=self.credentials, **kwargs)

    async def test_model_first_provider_second_and_no_automatic_choice(self):
        listing = await self.run_command("/models comm")
        self.assertIn("Commodity", listing.text)
        providers = await self.run_command('/providers "Commodity"')
        self.assertIn("first: model-a", providers.text)
        self.assertIn("second: model-b", providers.text)
        with self.assertRaisesRegex(ValueError, "--provider"):
            await self.run_command('/model "Commodity"')
        selection = await self.run_command(
            '/model "Commodity" --provider second')
        self.assertEqual(selection.selection[0], "second")
        self.assertEqual(selection.selection[2]["id"], "model-b")

    async def test_exact_provider_model_ids_and_invalid_choices(self):
        selection = await self.run_command('/model model-b --provider second')
        self.assertEqual(selection.selection[0], "second")
        with self.assertRaises(ValueError):
            await self.run_command('/model model-b --provider first')
        with self.assertRaises(ValueError):
            await self.run_command('/providers mod')
        with self.assertRaises(ValueError):
            await self.run_command('/model "unterminated')

    async def test_explicit_connection_selection_does_not_need_catalog(self):
        explicit = models.ExplicitConnectionOption(
            "private-model", "http://localhost/v1", "openai")
        with mock.patch.object(models, "ensure_index", mock.AsyncMock()) as index:
            result = await self.run_command(
                '/model private-model --provider explicit',
                explicit_connection=explicit)
        self.assertIs(result.selection, explicit)
        index.assert_not_awaited()

    async def test_direct_selection_preserves_endpoint_approval(self):
        with (
            mock.patch.object(front._core, "CREDENTIALS", self.credentials),
            mock.patch.object(front._core.endpoint_pins, "status",
                              return_value=("new", None)),
            mock.patch.object(front, "apply_runtime_config") as apply,
            self.assertRaisesRegex(ValueError, "approved"),
        ):
            picked = await self.run_command('/model model-a --provider first')
            front._apply_model_selection(picked.selection)
        apply.assert_not_called()


class TerminalBridgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.transcript = []
        self.input = SimpleNamespace(
            user_messages=terminals.UserMessageQueue(),
            reader=SimpleNamespace(cancel_requested=False,
                                   cancel_event=asyncio.Event()),
            modal=mock.Mock(side_effect=AssertionError("unexpected picker")))
        self.bridge = BridgeSession(
            "conversation", frontend="terminal", path="/unused.sock", peer_uid=0,
            enqueue=self.input.user_messages.put_nowait)
        self.config = SimpleNamespace(chat_provider=SimpleNamespace(
            provider_id="provider"))
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for name, value in (
                ("restore_output_area_after_input", mock.Mock()),
                ("current_config", mock.Mock(return_value=self.config)),
                ("current_model", mock.Mock(return_value="model")),
                ("current_transcript", mock.Mock(return_value=self.transcript)),
                ("record_agent_mode_instruction", mock.Mock()),
                ("mark_chat_log_dirty", mock.Mock()),
                ("save_chat_log", mock.Mock())):
            stack.enter_context(mock.patch.object(front, name, value))
        stack.enter_context(mock.patch.object(front, "terminal", mock.Mock()))
        stack.enter_context(mock.patch.object(front, "_terminal_activity", mock.Mock()))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        stack.enter_context(mock.patch.object(
            front._core, "TOOL_HOOK_PIPELINE", SimpleNamespace(turn_end_hooks=[])))

    def submit(self, input_id, text):
        self.bridge.receive({"type": "submit_prompt",
                             "instance_id": self.bridge.instance_id,
                             "input_id": input_id, "text": text})

    async def test_mixed_inputs_and_skills_execute_fifo_without_hooks(self):
        self.input.user_messages.put_nowait("keyboard-one")
        self.submit("remote", "/a-skill argument")
        self.input.user_messages.put_nowait("keyboard-two")
        self.input.user_messages.put_nowait(None)
        texts = []

        async def turn(transcript, **kwargs):
            text = transcript[-1]["content"][0]["text"]
            texts.append(text)
            return "reply " + text

        with mock.patch.object(front, "run_terminal_turn_async", turn):
            await front._consume_terminal_submissions(self.input, self.bridge)
        self.assertEqual(texts, ["keyboard-one", "/a-skill argument", "keyboard-two"])
        finished = [event for event, _size in self.bridge.journal
                    if event["type"] == "turn_finished"]
        self.assertEqual([event["origin"] for event in finished],
                         ["keyboard", "bridge", "keyboard"])
        self.assertEqual(self.bridge.inputs["remote"].status, "finished")

    async def test_remote_model_is_noninteractive_and_returns_command_result(self):
        self.submit("models", "/model")
        self.input.user_messages.put_nowait(None)
        with (
            mock.patch.object(front, "_run_direct_model_command",
                              mock.AsyncMock(return_value="Commodity")) as command,
            mock.patch.object(front, "run_terminal_turn_async") as turn,
        ):
            await front._consume_terminal_submissions(self.input, self.bridge)
        command.assert_awaited_once_with("/model")
        turn.assert_not_called()
        self.assertEqual(self.bridge.journal[-1][0]["type"], "command_finished")
        self.assertEqual(self.bridge.journal[-2][0]["text"], "Commodity")

    async def test_socket_input_waits_while_keyboard_picker_owns_input(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        @contextlib.asynccontextmanager
        async def modal():
            entered.set()
            yield SimpleNamespace(prompt=mock.AsyncMock())

        self.input.modal = modal
        self.input.user_messages.put_nowait('/model')

        async def picker(**kwargs):
            await release.wait()
            return None

        with (
            mock.patch.object(front.modelsdev, 'run_model_picker_async', picker),
            mock.patch.object(front, 'explicit_connection_option', return_value=None),
            mock.patch.object(front, 'run_terminal_turn_async',
                              mock.AsyncMock(return_value='remote reply')) as turn,
        ):
            consumer = asyncio.create_task(
                front._consume_terminal_submissions(self.input, self.bridge))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                self.submit('remote', 'queued-during-picker')
                self.assertEqual(self.bridge.inputs['remote'].status, 'queued')
                turn.assert_not_awaited()
                self.input.user_messages.put_nowait(None)
                release.set()
                await asyncio.wait_for(consumer, 1)
            finally:
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
        turn.assert_awaited_once()

    async def test_unsupported_controls_never_open_modals_or_execute_shell(self):
        for index, text in enumerate(("/quit", "/effort", "/account",
                                      "/image x", "!touch x")):
            self.submit(str(index), text)
        self.input.user_messages.put_nowait(None)
        with mock.patch.object(front, "run_bash_async") as bash:
            await front._consume_terminal_submissions(self.input, self.bridge)
        bash.assert_not_called()
        self.assertTrue(all(record.outcome == "error"
                            for record in self.bridge.inputs.values()))
        self.assertEqual(self.transcript, [])

    async def test_remote_turn_does_not_consume_keyboard_images(self):
        image = SimpleNamespace(content_block=lambda: {"type": "image", "data": "x"})
        pending = [image]
        with mock.patch.object(front, "run_terminal_turn_async",
                               mock.AsyncMock(return_value="done")):
            await front._run_terminal_submission(
                Submission("remote", "bridge", "remote"), self.input,
                pending, front.SubmissionResult(), self.bridge)
            self.assertEqual(pending, [image])
            self.assertEqual(len(self.transcript[-1]["content"]), 1)
            await front._run_terminal_submission(
                Submission("local"), self.input, pending, front.SubmissionResult())
        self.assertEqual(pending, [])
        self.assertEqual(len(self.transcript[-1]["content"]), 2)

    async def test_cancel_skips_queued_remote_work_until_explicit_resume(self):
        self.input.user_messages.put_nowait("cancel-me")
        self.submit("skip", "do-not-run")
        self.input.user_messages.put_nowait("/bridge resume")
        self.submit("run", "run-after-resume")
        self.input.user_messages.put_nowait(None)
        calls = []

        async def turn(transcript, **kwargs):
            calls.append(transcript[-1]["content"][0]["text"])
            if len(calls) == 1:
                kwargs["turn_events"].append({"type": "response_cancelled"})
            return "done"

        with mock.patch.object(front, "run_terminal_turn_async", turn):
            await front._consume_terminal_submissions(self.input, self.bridge)
        self.assertEqual(calls, ["cancel-me", "run-after-resume"])
        self.assertEqual(self.bridge.inputs["skip"].outcome, "unexecuted")
        self.assertFalse(self.bridge.paused)

    async def test_completion_follows_save_and_failed_hook_without_changing_outcome(self):
        self.submit('hook', 'hello')
        self.input.user_messages.put_nowait(None)
        order = []
        pipeline = front.tool_runtime.ToolHookPipeline()
        pipeline.turn_end_hooks.append(SimpleNamespace(
            command=['notify'], timeout_ms=10, stderr_reporter=None,
            hook_id='notify'))

        async def hook(*args, **kwargs):
            order.append('hook')
            raise OSError('delivery failed')

        finished = self.bridge.turn_finished

        def finish(*args):
            order.append('finish')
            finished(*args)

        with (
            mock.patch.object(front._core, 'TOOL_HOOK_PIPELINE', pipeline),
            mock.patch.object(front.tool_runtime, '_run_hook_command', hook),
            mock.patch.object(front, 'save_chat_log',
                              side_effect=lambda: order.append('save')),
            mock.patch.object(front, 'run_terminal_turn_async',
                              mock.AsyncMock(return_value='done')),
            mock.patch.object(self.bridge, 'turn_finished', finish),
        ):
            await front._consume_terminal_submissions(self.input, self.bridge)
        self.assertEqual(order, ['save', 'hook', 'finish'])
        self.assertEqual(self.bridge.inputs['hook'].outcome, 'completed')

    async def test_api_failure_is_reported_and_execution_exception_cleans_activity(self):
        self.submit("failure", "hello")
        self.input.user_messages.put_nowait(None)

        async def turn(transcript, **kwargs):
            kwargs["turn_events"].append({"type": "network_error", "error": "offline"})
            return ""

        with mock.patch.object(front, "run_terminal_turn_async", turn):
            await front._consume_terminal_submissions(self.input, self.bridge)
        self.assertEqual(self.bridge.inputs["failure"].outcome, "error")
        self.assertEqual(self.bridge.journal[-2][0]["text"], "offline")
        front._terminal_activity.set_turn_running.assert_called_with(False)


class BridgeCliTests(unittest.TestCase):
    def test_bridge_validation_precedes_terminal_setup(self):
        for args in (["--bridge-socket", "relative"],
                     ["--bridge-socket", "/socket", "--headless"],
                     ["--bridge-peer-uid", "1"],
                     ["--bridge-socket", "/socket", "--bridge-peer-uid", "-1"]):
            with self.subTest(args=args), self.assertRaises(getopt.GetoptError):
                front.parse_cli_args(args)

    @unittest.skipUnless(hasattr(socket, "SO_PEERCRED"), "SO_PEERCRED required")
    def test_optional_bridge_options(self):
        options, _args = front.parse_cli_args(
            ["--bridge-socket", "/socket", "--bridge-peer-uid", "123"])
        self.assertEqual(dict(options)["--bridge-peer-uid"], "123")


@unittest.skipUnless(os.name == "posix" and hasattr(socket, "AF_UNIX"),
                     "POSIX terminal and Unix socket required")
class BridgeEntrypointTests(unittest.TestCase):
    def test_real_terminal_entrypoint_receives_prompt_and_command(self):
        from loki_agent import pty_backend

        with tempfile.TemporaryDirectory(prefix="loki-bridge-test-") as root:
            workspace = os.path.join(root, "workspace")
            os.mkdir(workspace)
            path = os.path.join(root, "bridge.sock")
            env = child_environment(
                HOME=root, XDG_CONFIG_HOME=os.path.join(root, "config"),
                XDG_STATE_HOME=os.path.join(root, "state"),
                PATH=os.environ.get("PATH", ""), TERM="xterm",
                LOKI_PROVIDER="dummy", LOKI_API_BASE="http://dummy.invalid/v1",
                LOKI_DUMMY_REPLY="bridge-reply")
            configure_container(env, workspace)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(path)
            listener.listen()
            listener.settimeout(10)
            handle = None
            peer = None
            stream = None
            try:
                handle = pty_backend.spawn_pty(
                    [entrypoint("loki"), "--bridge-socket", path],
                    env=env, cwd=workspace)
                peer, _address = listener.accept()
                peer.settimeout(10)
                stream = peer.makefile("rb")
                self.assertEqual(json.loads(stream.readline())["type"], "hello")
                peer.sendall(b'{"type":"hello","version":1}\n')
                registration = json.loads(stream.readline())
                instance = registration["instance_id"]
                for input_id, text in (("command", "/pwd"), ("prompt", "hello")):
                    peer.sendall((json.dumps({
                        "type": "submit_prompt", "instance_id": instance,
                        "input_id": input_id, "text": text}) + "\n").encode())
                    events = []
                    while True:
                        event = json.loads(stream.readline())
                        events.append(event)
                        if event["type"] in ("command_finished", "turn_finished"):
                            break
                    self.assertEqual(events[-1]["input_id"], input_id)
                    self.assertEqual(events[-1]["outcome"], "completed")
                    output = "".join(event["text"] for event in events
                                     if event["type"] == "output_chunk")
                    self.assertIn(workspace if input_id == "command" else "bridge-reply",
                                  output)
                handle.write(b"/quit\r")
                # Drain the real terminal so its output pipe cannot stall exit.
                for _ in range(20):
                    handle.read(65536, 0.05)
                logs = list(Path(workspace).glob("chat-*.json"))
                if not logs:
                    logs = list(Path(workspace).glob(".loki/**/chat-*.json"))
                self.assertTrue(logs)
            finally:
                if handle is not None:
                    handle.terminate()
                    handle.close()
                if stream is not None:
                    stream.close()
                if peer is not None:
                    peer.close()
                listener.close()
