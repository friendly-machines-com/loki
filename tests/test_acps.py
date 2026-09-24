"""ACP transport and front/worker tests.

The end-to-end test runs the real front process and its spawned worker
over pipes with the dummy provider (no network): initialize, session/new
(real subprocess spawn), session/prompt, and the reply's stopReason.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from loki_entrypoints import configure_container, loki_acp_command
from response_header_fixtures import setUpModule  # noqa: F401 - unittest hook

from loki_agent import (
    __version__,
    acp,
    acp_main,
    acps,
    authentications,
    models,
)
from loki_agent.credentials import CredentialStore, is_credential_name
from loki_endpoints import assume_endpoints_approved

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _configured_workspace(tmpdir):
    """The session cwd a test must request: the registered container dir.

    Every front-spawning fixture here configures ``<tmpdir>/workspace`` with
    the real ``loki-setup``, and the Windows front refuses a session whose
    cwd is not a configured container.  On POSIX this is inert.
    """
    return os.path.join(tmpdir, "workspace")


def _close_process_streams(process):
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, name, None)
        if stream is not None and not stream.closed:
            stream.close()


class FramingTests(unittest.TestCase):
    def test_response_and_notification_shapes(self):
        self.assertEqual(
            acps.response(7, result={"a": 1}),
            {"jsonrpc": "2.0", "id": 7, "result": {"a": 1}})
        self.assertEqual(
            acps.response(7, error={"code": -32601, "message": "x"})["error"],
            {"code": -32601, "message": "x"})
        self.assertNotIn("result", acps.response(
            7, error={"code": -1, "message": "x"}))
        note = acps.notification("session/update", {"sessionId": "s"})
        self.assertNotIn("id", note)
        self.assertEqual(note["method"], "session/update")


class AsyncFdLineReaderTests(unittest.TestCase):
    def test_reads_buffered_lines_and_eof_without_changing_fd_flags(self):
        read_fd, write_fd = os.pipe()
        blocking_before = os.get_blocking(read_fd)
        try:
            os.write(write_fd, b"first\nsecond\npartial")
            os.close(write_fd)
            write_fd = None

            async def scenario():
                reader = acps.AsyncFdLineReader(read_fd, chunk_size=7)
                return [line async for line in reader]

            lines = asyncio.run(scenario())
            self.assertEqual(lines, [b"first\n", b"second\n", b"partial"])
            self.assertEqual(os.get_blocking(read_fd), blocking_before)
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def test_cancelled_read_unregisters_cleanly(self):
        read_fd, write_fd = os.pipe()
        try:
            async def scenario():
                reader = acps.AsyncFdLineReader(read_fd)
                pending = asyncio.create_task(reader.readline())
                await asyncio.sleep(0)
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                os.write(write_fd, b"after cancellation\n")
                return await asyncio.wait_for(reader.readline(), timeout=1)

            self.assertEqual(
                asyncio.run(scenario()), b"after cancellation\n")
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_reads_through_event_loop_readiness_without_an_executor(self):
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"message\n")

            async def scenario():
                reader = acps.AsyncFdLineReader(read_fd)
                with mock.patch.object(
                        asyncio.BaseEventLoop,
                        "run_in_executor",
                        side_effect=AssertionError(
                            "ACP stdin used an executor")):
                    return await reader.readline()

            self.assertEqual(asyncio.run(scenario()), b"message\n")
        finally:
            os.close(read_fd)
            os.close(write_fd)


class WorkerChannelLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_exit_awaits_credential_capability_cleanup(self):
        class Output:
            async def readline(self):
                return b""

        class Process:
            stdout = Output()
            stdin = None
            returncode = 0

            async def wait(self):
                return 0

        delegation = mock.Mock()
        delegation.revoke_now = mock.Mock()
        delegation.close = mock.AsyncMock()
        channel = acp.WorkerChannel(
            "session", Process(), lambda message: None, delegation)

        await channel._reader_task

        delegation.revoke_now.assert_called_once_with()
        delegation.close.assert_awaited_once_with()
        self.assertIsNone(channel.credential_delegation)

    async def test_close_releases_the_platform_process(self):
        from process_lifecycle_fixtures import process_lifecycle

        async with process_lifecycle('loki-acp') as fixture:
            delegation = await fixture.supervisor.delegate()
            process = fixture.record_process(await acp.runtime_isolation.start_worker(
                fixture.workspace, fixture.supervisor.environment, delegation))
            delegation.child_spawned()
            channel = acp.WorkerChannel('session', process, lambda message: None, delegation)
            try:
                # A reply proves that the real worker passed its startup gate
                # and established its delegated runtime, not just that it spawned.
                self.assertEqual(await asyncio.wait_for(
                    channel.request('session/cancel', {}), 10), {})
            finally:
                await asyncio.wait_for(channel.close(), 10)
            self.assertEqual(process.returncode, 0)
            self.assertTrue(channel._reader_task.done())
            self.assertTrue(process.stdout.at_eof())
            self.assertTrue(process.stdin.is_closing())
            self.assertIsNone(channel.credential_delegation)
            await fixture.assert_released(self)
            # Repeated caller cleanup must not compete with transport ownership.
            await asyncio.wait_for(channel.close(), 10)
            await fixture.assert_released(self)

    async def test_a_dead_worker_fails_the_pending_request(self):
        # A worker that exits without answering must fail the request that
        # waits on it, not leave the caller waiting for a reply that cannot
        # arrive.
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "pass",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        channel = acp.WorkerChannel("dying", process, lambda message: None)
        try:
            with self.assertRaises(acps.TransportError):
                await asyncio.wait_for(channel.request("session/new", {}), 10)
        finally:
            await channel.close()


class WorkerSpawnGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_denied_workspace_fails_closed_and_releases_delegation(self):
        # A session whose workspace has no configured container must not get
        # an unprotected worker: the spawn seam's refusal surfaces as a
        # transport error and the delegation it created is closed.
        front = acp.Front(
            lambda: None, lambda message: None, CredentialStore({}))
        delegation = mock.Mock()
        delegation.close = mock.AsyncMock()
        spawned = mock.AsyncMock(
            side_effect=acp.RuntimeIsolationError(
                "No Windows container configured; run loki-setup"))
        with mock.patch.object(
                front.credential_supervisor,
                "delegate",
                new=mock.AsyncMock(return_value=delegation)), \
                mock.patch.object(
                    acp.runtime_isolation, "start_worker", new=spawned):
            with self.assertRaisesRegex(
                    acps.TransportError, "could not start worker"):
                await front._open_worker(
                    cwd=ROOT, open_method="session/new")
        spawned.assert_awaited_once_with(
            ROOT, front.environment, delegation)
        delegation.close.assert_awaited_once_with()
        self.assertFalse(front.workers)
        self.assertFalse(front._opening_sessions)


class FrontPromptOrderingTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffered_config_cannot_overtake_prompt(self):
        order = []
        responses = []
        prompt_reply = asyncio.get_running_loop().create_future()

        class Channel:
            async def request(self, method, params, forwarded=None):
                order.append(method)
                if forwarded is not None:
                    forwarded.set()
                if method == "session/prompt":
                    return await prompt_reply
                if method == "session/describe_config_selection":
                    # Nothing to approve for this change; the front's read-only
                    # question is part of forwarding a config option now.
                    return {}
                return {"configOptions": []}

        front = acp.Front(
            lambda: None, responses.append, CredentialStore({}))
        front.workers["session"] = Channel()

        await front.handle({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/prompt",
            "params": {
                "sessionId": "session",
                "prompt": [{"type": "text", "text": "hello"}],
            },
        })
        await front.handle({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/set_config_option",
            "params": {
                "sessionId": "session",
                "configId": "reasoning_effort",
                "value": "effort:low",
            },
        })

        self.assertEqual(order, [
            "session/prompt",
            "session/describe_config_selection",
            "session/set_config_option",
        ])
        prompt_reply.set_result({"stopReason": "end_turn"})
        await asyncio.gather(*front._tasks)
        self.assertEqual(
            {message["id"] for message in responses}, {1, 2})


class SavedConnectionAuthorizationTests(
        unittest.IsolatedAsyncioTestCase):
    def _descriptor(self):
        from loki_agent import protocols
        from loki_agent.connections import ConnectionDescriptor

        return ConnectionDescriptor(
            provider_id=None,
            provider_name="Saved Provider",
            model="saved-model",
            chat_url="https://saved.example/v1/chat/completions",
            models_url="https://saved.example/v1/models",
            protocol=protocols.OPENAI_CHAT,
        )

    async def _restore(
            self, method, action, *, advertise=True,
            authorization_connection=True):
        messages = []
        requests = []
        descriptor = self._descriptor()
        front = None
        test_case = self

        class Delegation:
            def child_arguments(self):
                return ()

            def child_spawn_kwargs(self):
                return {}

            def child_spawned(self):
                return None

            async def close(self):
                return None

        class Channel:
            def __init__(self, session_id, process, forward, delegation):
                self.session_id = session_id
                self.closed = False

            async def request(self, method, params):
                requests.append((method, params))
                if method == "session/prepare_open":
                    return (
                        {
                            "authorizationConnection":
                                descriptor.to_dict(),
                        }
                        if authorization_connection else {})
                if method == "session/commit_open":
                    test_case.assertNotIn(
                        self.session_id, front.workers)
                    return {"configOptions": []}
                raise AssertionError(method)

            async def close(self):
                self.closed = True

        def write(message):
            messages.append(message)
            if message.get("method") == "elicitation/create":
                async def respond():
                    await front.handle(acps.response(
                        message["id"],
                        result=(
                            {
                                "action": "accept",
                                "content": {"authorize": True},
                            }
                            if action == "accept"
                            else {"action": action}
                        ),
                    ))

                asyncio.create_task(respond())

        front = acp.Front(
            lambda: None, write, CredentialStore({}))
        if advertise:
            front.initialize({
                "clientCapabilities": {
                    "elicitation": {"form": {}},
                },
            })
        delegation = Delegation()
        with mock.patch.object(
                front.credential_supervisor,
                "delegate",
                new=mock.AsyncMock(return_value=delegation)), \
                mock.patch.object(
                    acp.runtime_isolation,
                    "start_worker",
                    new=mock.AsyncMock(return_value=object())), \
                mock.patch.object(acp, "WorkerChannel", Channel):
            try:
                result = await front.restore_session(
                    method,
                    {"sessionId": "saved", "cwd": ROOT},
                    request_id=73,
                )
            except Exception as error:
                return front, requests, messages, error
        return front, requests, messages, result

    async def test_accepts_exact_connection_before_publishing_worker(self):
        for method in acp.RESTORE_METHODS:
            with self.subTest(method=method):
                front, requests, messages, result = await self._restore(
                    method, "accept")

                self.assertEqual(result, {})
                self.assertIn("saved", front.workers)
                self.assertEqual(
                    [name for name, _params in requests],
                    ["session/prepare_open", "session/commit_open"],
                )
                self.assertEqual(requests[0][1]["openMethod"], method)
                elicitation = next(
                    message for message in messages
                    if message.get("method") == "elicitation/create")
                self.assertEqual(elicitation["params"]["requestId"], 73)
                self.assertEqual(elicitation["params"]["mode"], "form")
                self.assertIs(
                    elicitation["params"]["requestedSchema"]["properties"]
                    ["authorize"]["default"],
                    False,
                )
                self.assertIn(
                    '"https://saved.example/v1/chat/completions"',
                    elicitation["params"]["message"],
                )
                self.assertIn(
                    "Working directory:",
                    elicitation["params"]["message"],
                )
                self.assertIn(
                    ROOT,
                    elicitation["params"]["message"],
                )

    async def test_decline_closes_provisional_worker_without_commit(self):
        for method in acp.RESTORE_METHODS:
            with self.subTest(method=method):
                front, requests, _messages, error = await self._restore(
                    method, "decline")

                self.assertIsInstance(error, acps.TransportError)
                self.assertEqual(
                    [name for name, _params in requests],
                    ["session/prepare_open"],
                )
                self.assertNotIn("saved", front.workers)
                self.assertNotIn("saved", front._opening_sessions)

    async def test_restore_fails_closed_without_form_elicitation(self):
        for method in acp.RESTORE_METHODS:
            with self.subTest(method=method):
                front, requests, messages, error = await self._restore(
                    method, "accept", advertise=False)

                self.assertIsInstance(error, acps.TransportError)
                self.assertEqual(
                    [name for name, _params in requests],
                    ["session/prepare_open"],
                )
                self.assertFalse(any(
                    message.get("method") == "elicitation/create"
                    for message in messages))
                self.assertNotIn("saved", front.workers)

    async def test_explicit_startup_connection_needs_no_saved_approval(self):
        for method in acp.RESTORE_METHODS:
            with self.subTest(method=method):
                front, requests, messages, result = await self._restore(
                    method, "accept", authorization_connection=False)

                self.assertEqual(result, {})
                self.assertIn("saved", front.workers)
                self.assertEqual(
                    [name for name, _params in requests],
                    ["session/prepare_open", "session/commit_open"],
                )
                self.assertFalse(any(
                    message.get("method") == "elicitation/create"
                    for message in messages))

    async def test_session_lifecycle_requires_explicit_cwd(self):
        front = acp.Front(
            lambda: None, lambda _message: None, CredentialStore({}))
        with self.assertRaisesRegex(
                acps.TransportError, "cwd must be an absolute path"):
            await front.new_session({})
        for method in acp.RESTORE_METHODS:
            with self.subTest(method=method):
                with self.assertRaisesRegex(
                        acps.TransportError,
                        "cwd must be an absolute path"):
                    await front.restore_session(
                        method, {"sessionId": "saved"}, request_id=73)


class QuarantineTests(unittest.TestCase):
    def test_stdout_is_reserved_for_protocol(self):
        code = (
            "import sys, os, json\n"
            "from loki_agent import acps\n"
            "saved = os.dup(1)\n"
            "acps.quarantine_stdout()\n"
            "write = acps.make_writer(saved)\n"
            "write(acps.response(1, result={'ok': True}))\n"
            "print('stray output')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=ROOT)
        lines = [
            line for line in proc.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]),
                         {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})

        # A contained worker is given the null descriptor to redirect to
        # instead of opening the device itself, which its container refuses.
        # The borrowed descriptor stays the caller's, the protocol keeps its
        # own, and stray output is still discarded.
        borrowed = (
            "import sys, os, json\n"
            "from loki_agent import acps\n"
            "null = os.open(os.devnull, os.O_WRONLY)\n"
            "saved = os.dup(1)\n"
            "acps.quarantine_stdout(null)\n"
            "write = acps.make_writer(saved)\n"
            "write(acps.response(1, result={'ok': True}))\n"
            "print('stray output')\n"
            "os.fstat(null)\n"
            "os.close(null)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", borrowed],
            capture_output=True, text=True, cwd=ROOT)
        lines = [
            line for line in proc.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]),
                         {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})


class EntrypointTests(unittest.TestCase):
    def test_subagent_uses_inherited_worker_authority(self):
        with mock.patch.object(
                acp_main,
                "capture_process_credentials") as capture, \
                mock.patch.object(
                    acp_main,
                    "protect_credential_process") as protect, \
                mock.patch(
                    "loki_agent.subagents.main",
                    return_value=31) as subagent_main, \
                mock.patch.object(sys, "argv", [
                    "/installed/bin/loki-acp",
                    "--subagent",
                    "Explore",
                    "--session-owner-fd", "7",
                    "--credential-capability-fd", "8",
                ]):
            status = acp_main.main()

        self.assertEqual(status, 31)
        capture.assert_not_called()
        protect.assert_called_once_with()
        subagent_main.assert_called_once_with([
            "Explore",
            "--session-owner-fd", "7",
            "--credential-capability-fd", "8",
        ])


class FrontWorkerTests(unittest.TestCase):
    def _front_env(self, tmpdir):
        env = dict(os.environ)
        env.update({
            "HOME": tmpdir,
            "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
            "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
            "TERM": "dumb",
            "LOKI_PROVIDER": "dummy",
            "LOKI_API_BASE": "http://dummy.invalid/v1",
            "LOKI_MODEL": "dummy-model",
            "LOKI_DUMMY_REPLY": "acp reply text",
        })
        workspace = os.path.join(tmpdir, "workspace")
        os.makedirs(workspace, exist_ok=True)
        configure_container(env, workspace)
        return env

    def test_ini_logging_reaches_the_front_not_the_contained_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            config = os.path.join(directory, "logging.ini")
            # The handler arg is a Python literal that fileConfig evals, so the
            # path must be repr'd: a raw Windows path would be consumed by
            # backslash escapes before that eval ran, and the front would exit
            # before opening its protocol loop.
            trace_prefix = os.path.join(directory, "trace-")
            with open(config, "w") as stream:
                stream.write("""[loggers]
keys=root
[handlers]
keys=trace
[formatters]
keys=
[logger_root]
level=DEBUG
handlers=trace
[handler_trace]
class=FileHandler
args=(%r + str(__import__('os').getpid()), 'a')
""" % trace_prefix)
            front_env = self._front_env

            def relative_config_env(cwd):
                env = front_env(cwd)
                env["LOKI_LOG_CONFIG"] = os.path.relpath(
                    config, os.path.join(cwd, "workspace"))
                return env

            with mock.patch.object(
                    self, "_front_env", side_effect=relative_config_env):
                self.test_initialize_new_session_prompt_roundtrip()
            # The front is uncontained and loads the INI.  The separately
            # execed worker is contained: it refuses any configuration the
            # invoker names (it cannot be assumed able to read it or write its
            # handlers' targets) and logs to stderr instead, so only the front
            # opens a trace file.
            traces = [name for name in os.listdir(directory)
                      if name.startswith("trace-")]
            self.assertEqual(len(traces), 1)

    def test_initialize_new_session_prompt_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._front_env(tmpdir)
            front = subprocess.Popen(
                loki_acp_command(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=env, cwd=os.path.join(tmpdir, "workspace"))
            self.addCleanup(_close_process_streams, front)
            try:
                def send(message):
                    front.stdin.write(json.dumps(message) + "\n")
                    front.stdin.flush()

                def recv():
                    line = front.stdout.readline()
                    self.assertTrue(line, "front produced no message")
                    return json.loads(line)

                def recv_reply(reply_id):
                    while True:
                        message = recv()
                        if message.get("id") == reply_id:
                            return message

                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1}})
                reply = recv_reply(1)
                self.assertEqual(reply["result"]["protocolVersion"], 1)
                self.assertEqual(
                    reply["result"]["agentInfo"]["name"], "loki")
                self.assertEqual(
                    reply["result"]["agentInfo"]["version"], __version__)
                capabilities = reply["result"]["agentCapabilities"]
                self.assertEqual(
                    capabilities["sessionCapabilities"]["close"], {})
                self.assertEqual(
                    capabilities["sessionCapabilities"]["list"], {})
                self.assertEqual(
                    capabilities["sessionCapabilities"]["resume"], {})
                self.assertFalse(
                    capabilities["promptCapabilities"]["image"])

                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": _configured_workspace(tmpdir)}})
                reply = recv_reply(2)
                session_id = reply["result"]["sessionId"]
                self.assertTrue(session_id)

                send({"jsonrpc": "2.0", "id": 3,
                      "method": "session/prompt",
                      "params": {
                          "sessionId": session_id,
                          "prompt": [{"type": "text",
                                      "text": "hello acp"}]}})
                updates = []
                while True:
                    message = recv()
                    if message.get("id") == 3:
                        reply = message
                        break
                    updates.append(message)
                self.assertEqual(reply["result"]["stopReason"], "end_turn")
                # The turn's assistant text must have streamed as a
                # session/update before the reply landed.
                self.assertTrue(any(
                    m.get("method") == "session/update"
                    and m["params"]["update"]["sessionUpdate"]
                    == "agent_message_chunk"
                    for m in updates))

                send({"jsonrpc": "2.0", "id": 4,
                      "method": "session/close",
                      "params": {"sessionId": session_id}})
                self.assertEqual(recv_reply(4)["result"], {})
                send({"jsonrpc": "2.0", "id": 5,
                      "method": "session/prompt",
                      "params": {
                          "sessionId": session_id,
                          "prompt": [{"type": "text", "text": "closed"}]}})
                self.assertIn("error", recv_reply(5))
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()

    def test_advertises_commands_and_serves_slash_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._front_env(tmpdir)
            front = subprocess.Popen(
                loki_acp_command(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=env,
                cwd=os.path.join(tmpdir, "workspace"))
            self.addCleanup(_close_process_streams, front)
            try:
                def send(message):
                    front.stdin.write(json.dumps(message) + "\n")
                    front.stdin.flush()

                def recv():
                    line = front.stdout.readline()
                    self.assertTrue(line, "front produced no message")
                    return json.loads(line)

                def recv_reply(reply_id):
                    while True:
                        message = recv()
                        if message.get("id") == reply_id:
                            return message

                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1}})
                recv_reply(1)

                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": _configured_workspace(tmpdir)}})
                reply = recv_reply(2)
                session_id = reply["result"]["sessionId"]

                # The advertisement follows the session reply, so the client
                # already knows the session it describes.
                message = recv()
                self.assertEqual(message["method"], "session/update")
                self.assertEqual(message["params"]["sessionId"], session_id)
                update = message["params"]["update"]
                self.assertEqual(
                    update["sessionUpdate"], "available_commands_update")
                self.assertIn(
                    "pwd",
                    [command["name"] for command in update["availableCommands"]])

                # /pwd is answered locally: no model turn, no dummy reply.
                send({"jsonrpc": "2.0", "id": 3,
                      "method": "session/prompt",
                      "params": {
                          "sessionId": session_id,
                          "prompt": [{"type": "text", "text": "/pwd"}]}})
                chunks = []
                while True:
                    message = recv()
                    if message.get("id") == 3:
                        reply = message
                        break
                    if (message.get("method") == "session/update"
                            and message["params"]["update"]["sessionUpdate"]
                            == "agent_message_chunk"):
                        chunks.append(
                            message["params"]["update"]["content"]["text"])
                self.assertEqual(reply["result"]["stopReason"], "end_turn")
                text = "".join(chunks)
                self.assertIn("cwd:", text)
                self.assertNotIn("acp reply text", text)

                # /image stages a snapshot for the next prompt.
                with open(os.path.join(
                        _configured_workspace(tmpdir), "shot.png"), "wb") as stream:
                    stream.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
                send({"jsonrpc": "2.0", "id": 6,
                      "method": "session/prompt",
                      "params": {
                          "sessionId": session_id,
                          "prompt": [{"type": "text",
                                      "text": "/image shot.png"}]}})
                chunks = []
                while True:
                    message = recv()
                    if message.get("id") == 6:
                        reply = message
                        break
                    if (message.get("method") == "session/update"
                            and message["params"]["update"]["sessionUpdate"]
                            == "agent_message_chunk"):
                        chunks.append(
                            message["params"]["update"]["content"]["text"])
                self.assertEqual(reply["result"]["stopReason"], "end_turn")
                self.assertIn("Attached image", "".join(chunks))

                # A plain prompt still reaches the provider.
                send({"jsonrpc": "2.0", "id": 4,
                      "method": "session/prompt",
                      "params": {
                          "sessionId": session_id,
                          "prompt": [{"type": "text", "text": "hello"}]}})
                chunks = []
                while True:
                    message = recv()
                    if message.get("id") == 4:
                        break
                    if (message.get("method") == "session/update"
                            and message["params"]["update"]["sessionUpdate"]
                            == "agent_message_chunk"):
                        chunks.append(
                            message["params"]["update"]["content"]["text"])
                self.assertIn("acp reply text", "".join(chunks))

                send({"jsonrpc": "2.0", "id": 5,
                      "method": "session/close",
                      "params": {"sessionId": session_id}})
                self.assertEqual(recv_reply(5)["result"], {})
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()

    def test_front_delegates_credentials_without_worker_environment_values(self):
        credential_name = "LOKI_ACP_BOOTSTRAP_TEST_TOKEN"
        credential_value = "secret-" + ("v" * 137)
        store = CredentialStore({
            "LOKI_API_BASE": "http://dummy.invalid/v1",
            "LOKI_PROVIDER": "dummy",
            "LOKI_MODEL": "dummy-model",
            credential_name: credential_value,
        })
        front = acp.Front(
            lambda: None, lambda message: None, store)
        spawned = {}

        class FakeProcess:
            pass

        class FakeChannel:
            def __init__(
                    self, session_id, process, forward,
                    credential_delegation):
                self.session_id = session_id
                self.credential_delegation = credential_delegation

            async def request(self, method, params):
                return {}

            async def close(self):
                await self.credential_delegation.close()

        async def fake_spawn(cwd, environment, delegation):
            spawned["cwd"] = cwd
            spawned["environment"] = environment
            spawned["delegation"] = delegation
            return FakeProcess()

        async def scenario():
            with mock.patch.object(
                    acp.runtime_isolation, "start_worker",
                    new=fake_spawn), mock.patch.object(
                        acp, "WorkerChannel", FakeChannel):
                session_id, _reply = await front._open_worker(
                    cwd=ROOT, open_method="session/new")
                await front.workers.pop(session_id).close()

        asyncio.run(scenario())

        # The spawn seam receives the session's workspace and the sanitized
        # environment -- the spawn-specific kwargs (pipes, descriptors or
        # handle lists) are the seam's own contract, pinned in
        # test_runtime_isolations and test_host_ipc.
        self.assertEqual(spawned["cwd"], ROOT)
        child_environment = spawned["environment"]
        self.assertNotIn(credential_name, child_environment)
        self.assertNotIn(credential_value, repr(child_environment))
        self.assertIsNotNone(spawned["delegation"])
        credential = authentications.CredentialRef.environment(
            credential_name)
        self.assertTrue(front.credentials.has_ref(credential))
        self.assertEqual(front.credentials.get(credential_name), "")
        lease = asyncio.run(front.credential_broker.lease(credential))
        self.assertEqual(lease.value, credential_value)

    def test_real_front_scrubs_and_real_worker_never_receives_credential(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                name: value
                for name, value in self._front_env(tmpdir).items()
                if not is_credential_name(name)
            }
            credential_name = "LOKI_ACP_BOOTSTRAP_TEST_TOKEN"
            credential_value = "secret-" + ("v" * 137)
            env[credential_name] = credential_value
            env["LOKI_ACP_AFTER"] = "visible"

            observer_dir = os.path.join(tmpdir, "observer")
            report_dir = os.path.join(tmpdir, "reports")
            os.makedirs(observer_dir)
            os.makedirs(report_dir)
            sitecustomize = os.path.join(observer_dir, "sitecustomize.py")
            observer = r'''
import ctypes
import json
import os
import sys

from loki_agent import credentials

_capture = credentials.capture_process_credentials


def native_environment():
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        get = kernel.GetEnvironmentStringsW
        get.restype = ctypes.c_void_p
        free = kernel.FreeEnvironmentStringsW
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_int
        pointer = get()
        if not pointer:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entries, address = [], pointer
            while True:
                text = ctypes.wstring_at(address)
                if not text:
                    break
                entries.append(text)
                address += (len(text) + 1) * ctypes.sizeof(ctypes.c_wchar)
            return b"\0".join(entry.encode("utf-8") for entry in entries)
        finally:
            free(pointer)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        libc.sysctl.argtypes = (
            ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t)
        libc.sysctl.restype = ctypes.c_int
        mib = (ctypes.c_int * 3)(1, 49, os.getpid())
        size = ctypes.c_size_t(os.sysconf("SC_ARG_MAX"))
        buffer = ctypes.create_string_buffer(size.value)
        if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), "sysctl failed")
        return buffer.raw[:size.value]
    with open("/proc/self/environ", "rb") as source:
        return source.read()


def report_environment():
    raw = native_environment()
    name = __NAME__.encode("ascii")
    value = __VALUE__.encode("ascii")
    filler = b"x" * len(name + b"=" + value)
    after = b"LOKI_ACP_AFTER=visible\0"
    report = {
        "pid": os.getpid(),
        "worker": "--worker" in sys.argv,
        "name_present": name + b"=" in raw,
        "value_present": value in raw,
        "after_present": after in raw,
    }
    if sys.platform.startswith("linux"):
        # Linux overwrites the original records in place, so the filler and the
        # record that follows it are observable.  Windows removes them instead.
        report["filler_present"] = filler in raw.split(b"\0")
        report["after_follows_filler"] = (
            raw.find(after) > raw.find(filler + b"\0")
            if filler + b"\0" in raw else False)
    path = os.path.join(os.environ["LOKI_TEST_REPORT_DIR"],
                        "%d.json" % os.getpid())
    with open(path, "w", encoding="ascii") as output:
        json.dump(report, output)


def capture_process_credentials():
    store = _capture()
    report_environment()
    return store


credentials.capture_process_credentials = capture_process_credentials
if "--worker" in sys.argv:
    report_environment()
'''
            observer = observer.replace("__NAME__", repr(credential_name))
            observer = observer.replace("__VALUE__", repr(credential_value))
            with open(sitecustomize, "w", encoding="ascii") as stream:
                stream.write(observer)
            env["LOKI_TEST_REPORT_DIR"] = report_dir
            env["PYTHONPATH"] = os.pathsep.join(
                [observer_dir, ROOT])

            # Drive the front the way the product drives a worker channel:
            # an asyncio subprocess read line by line (WorkerChannel), under
            # the default event-loop policy the entrypoints themselves use.
            # stderr stays inherited, so a failing front or worker explains
            # itself in the runner log.
            async def converse(process):
                async def send(message):
                    process.stdin.write(
                        (json.dumps(message) + "\n").encode("utf-8"))
                    await process.stdin.drain()

                async def reply(reply_id):
                    while True:
                        line = await asyncio.wait_for(
                            process.stdout.readline(), 30)
                        self.assertTrue(
                            line,
                            "front closed its protocol output",
                        )
                        message = json.loads(line.decode("utf-8"))
                        if message.get("id") == reply_id:
                            return message

                await send({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": 1},
                })
                self.assertIn("result", await reply(1))
                await send({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/new",
                    "params": {"cwd": _configured_workspace(tmpdir)},
                })
                self.assertIn("result", await reply(2))

            async def run():
                process = await asyncio.create_subprocess_exec(
                    *loki_acp_command(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=None,
                    env=env,
                    cwd=os.path.join(tmpdir, "workspace"),
                )
                try:
                    await converse(process)
                    # Read the observer reports before tearing the
                    # front down: the Linux check below needs the
                    # worker's live /proc entry.
                    report_names = os.listdir(report_dir)
                    self.assertEqual(len(report_names), 2)
                    reports = []
                    for name in report_names:
                        with open(
                                os.path.join(report_dir, name),
                                encoding="ascii") as stream:
                            reports.append(json.load(stream))
                    front_report = next(
                        report for report in reports if not report["worker"])
                    worker_report = next(
                        report for report in reports if report["worker"])

                    for report in reports:
                        self.assertFalse(report["name_present"])
                        self.assertFalse(report["value_present"])
                        self.assertTrue(report["after_present"])
                    if sys.platform.startswith("linux"):
                        # Linux overwrites the credential record in place and
                        # covers the directory with a tmpfs; both are Linux
                        # mechanisms.  The tmpfs cover is containment, whose
                        # Windows counterpart is the AppContainer gate exercised
                        # by test_windows_runtime and test_runtime_gate.  The
                        # in-place overwrite is a scrub-in-place detail: Windows
                        # removes the record instead (see report_environment),
                        # so it has no counterpart assertion here.
                        self.assertTrue(front_report["filler_present"])
                        self.assertTrue(front_report["after_follows_filler"])
                        self.assertFalse(worker_report["filler_present"])
                        self.assertFalse(worker_report["after_follows_filler"])
                        credential_dir = os.path.join(
                            tmpdir, "config", "loki", "credentials")
                        with open(
                                f"/proc/{worker_report['pid']}/mountinfo",
                                encoding="ascii") as stream:
                            worker_mounts = stream.read()
                        self.assertTrue(any(
                            f" {credential_dir} " in line
                            and " - tmpfs " in line
                            for line in worker_mounts.splitlines()
                        ), worker_mounts)
                finally:
                    process.stdin.close()
                    try:
                        await asyncio.wait_for(process.wait(), 5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()

            asyncio.run(run())

    def test_unknown_session_is_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._front_env(tmpdir)
            front = subprocess.Popen(
                loki_acp_command(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=env, cwd=os.path.join(tmpdir, "workspace"))
            self.addCleanup(_close_process_streams, front)
            try:
                front.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": 9,
                    "method": "session/prompt",
                    "params": {"sessionId": "nope",
                               "prompt": [{"type": "text", "text": "x"}]},
                }) + "\n")
                front.stdin.flush()
                line = front.stdout.readline()
                reply = json.loads(line)
                self.assertEqual(reply["id"], 9)
                self.assertIn("error", reply)
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()


class EventMapperTests(unittest.TestCase):
    def test_assistant_delta_streams_chunk(self):
        from loki_agent import acp_events
        updates = acp_events.map_event("s", {"type": "assistant_delta",
                                             "content": "hi"}, {})
        self.assertEqual(len(updates), 1)
        self.assertEqual(
            updates[0]["update"]["sessionUpdate"], "agent_message_chunk")
        self.assertEqual(updates[0]["update"]["content"]["text"], "hi")

    def test_tool_call_then_result_pair(self):
        from loki_agent import acp_events
        state = {}
        call = acp_events.map_event("s", {"type": "tool_call",
                                          "name": "Bash",
                                          "call_id": "call_1",
                                          "args": {"command": "ls"}}, state)
        self.assertEqual(call[0]["update"]["sessionUpdate"], "tool_call")
        self.assertEqual(call[0]["update"]["kind"], "execute")
        self.assertIn("ls", call[0]["update"]["title"])
        result = acp_events.map_event("s", {"type": "tool_result",
                                            "name": "Bash",
                                            "call_id": "call_1",
                                            "content": "a\nb",
                                            "is_error": False}, state)
        self.assertEqual(result[0]["update"]["sessionUpdate"],
                         "tool_call_update")
        self.assertEqual(result[0]["update"]["toolCallId"], "call_1")
        self.assertNotIn("status", result[0]["update"])
        self.assertEqual(
            result[0]["update"]["content"],
            [{
                "type": "content",
                "content": {"type": "text", "text": "a\nb"},
            }],
        )

    def test_rejected_tool_uses_its_real_call_id(self):
        from loki_agent import acp_events
        updates = acp_events.map_event(
            "s",
            {
                "type": "tool_rejected",
                "name": "Write",
                "call_id": "provider-call-77",
                "args": {"file_path": "x"},
            },
            {},
        )
        self.assertEqual(
            updates[0]["update"]["sessionUpdate"], "tool_call")
        self.assertEqual(
            updates[0]["update"]["toolCallId"], "provider-call-77")
        self.assertEqual(updates[0]["update"]["status"], "failed")

    def test_tool_result_error_is_failed(self):
        from loki_agent import acp_events
        updates = acp_events.map_event("s", {"type": "tool_result",
                                             "content": "boom",
                                             "is_error": True},
                                       {"pending_call_id": "call-2"})
        self.assertEqual(updates[0]["update"]["status"], "failed")

    def test_ignored_events_map_to_nothing(self):
        from loki_agent import acp_events
        for kind in ("assistant_end", "response_timing", "max_loops",
                     "assistant_start", "provider_notice"):
            self.assertEqual(
                acp_events.map_event("s", {"type": kind}, {}), [])


class UpdateStreamingTests(unittest.TestCase):
    """A tool-call turn must stream session/update notifications."""

    def test_prompt_emits_updates_and_stop_reason(self):
        # The dummy provider replies with tool calls when the user text
        # starts with "tool:" -- reply is JSON naming the call.
        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env.update({
                "HOME": tmpdir,
                "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
                "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
                "TERM": "dumb",
                "LOKI_PROVIDER": "dummy",
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_MODEL": "dummy-model",
                "LOKI_DUMMY_REPLY": "plain answer",
            })
            workspace = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace, exist_ok=True)
            configure_container(env, workspace)
            front = subprocess.Popen(
                loki_acp_command(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, env=env, cwd=os.path.join(tmpdir, "workspace"))
            self.addCleanup(_close_process_streams, front)
            try:
                def send(message):
                    front.stdin.write(json.dumps(message) + "\n")
                    front.stdin.flush()

                def recv():
                    line = front.stdout.readline()
                    self.assertTrue(line, "front produced no message")
                    return json.loads(line)

                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1}})
                recv()
                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": workspace}})
                session_id = recv()["result"]["sessionId"]
                send({"jsonrpc": "2.0", "id": 3,
                      "method": "session/prompt",
                      "params": {"sessionId": session_id,
                                 "prompt": [{"type": "text",
                                             "text": "hello"}]}})
                messages = []
                while True:
                    reply = recv()
                    if reply.get("id") == 3:
                        break
                    messages.append(reply)
                # Plain reply: at least the assistant message chunk arrived
                # as a session/update notification before the reply.
                self.assertTrue(
                    any(m.get("method") == "session/update"
                        and m["params"]["update"]["sessionUpdate"]
                        == "agent_message_chunk"
                        for m in messages),
                    f"no agent_message_chunk in {[m.get('method') for m in messages]}")
                self.assertEqual(reply["result"]["stopReason"], "end_turn")
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()


class CancelEndToEndTests(unittest.TestCase):
    """session/cancel mid-turn must yield stopReason "cancelled"."""

    def test_cancel_during_streaming_turn(self):
        import time as _time
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = os.path.join(tmpdir, "release")
            env = dict(os.environ)
            env.update({
                "HOME": tmpdir,
                "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
                "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
                "TERM": "dumb",
                "LOKI_PROVIDER": "dummy",
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_MODEL": "dummy-model",
                "LOKI_DUMMY_REPLY": "chunked answer",
                "LOKI_STREAM": "1",
                "LOKI_DUMMY_STREAM_CHUNKS":
                    '["first ", "second part"]',
                "LOKI_DUMMY_STREAM_GATE": gate,
            })
            workspace = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace, exist_ok=True)
            configure_container(env, workspace)
            front = subprocess.Popen(
                loki_acp_command(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, env=env, cwd=os.path.join(tmpdir, "workspace"))
            self.addCleanup(_close_process_streams, front)
            try:
                def send(message):
                    front.stdin.write(json.dumps(message) + "\n")
                    front.stdin.flush()

                def recv():
                    line = front.stdout.readline()
                    self.assertTrue(line, "front produced no message")
                    return json.loads(line)

                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1}})
                while recv().get("id") != 1:
                    pass
                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": workspace}})
                while True:
                    reply = recv()
                    if reply.get("id") == 2:
                        break
                session_id = reply["result"]["sessionId"]

                send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                      "params": {"sessionId": session_id,
                                 "prompt": [{"type": "text",
                                             "text": "hello"}]}})
                # Wait for the first delta to stream: the turn is now
                # in flight and blocked on the gate.
                deadline = _time.monotonic() + 5
                saw_first_chunk = False
                while _time.monotonic() < deadline:
                    message = recv()
                    if (message.get("method") == "session/update"
                            and message["params"]["update"].get(
                                "sessionUpdate") == "agent_message_chunk"):
                        saw_first_chunk = True
                        break
                self.assertTrue(saw_first_chunk, "no delta streamed")

                send({"jsonrpc": "2.0", "method": "session/cancel",
                      "params": {"sessionId": session_id}})

                after_cancel = []
                while True:
                    reply = recv()
                    if reply.get("id") == 3:
                        break
                    after_cancel.append(reply)
                self.assertEqual(reply["result"]["stopReason"],
                                 "cancelled")
                self.assertTrue(all(
                    message.get("id") is None
                    for message in after_cancel
                ), after_cancel)
                # The gate was never released: the cancel, not the gate,
                # ended the turn.
                self.assertFalse(os.path.exists(gate))
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()


class SessionRestoreTests(unittest.TestCase):
    def _front(self, env, cwd):
        process = subprocess.Popen(
            loki_acp_command(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, env=env, cwd=cwd)
        self.addCleanup(_close_process_streams, process)
        return process

    def _env(self, tmpdir, reply="loadable answer"):
        env = dict(os.environ)
        env.update({
            "HOME": tmpdir,
            "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
            "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
            "TERM": "dumb",
            "LOKI_PROVIDER": "dummy",
            "LOKI_API_BASE": "http://dummy.invalid/v1",
            "LOKI_MODEL": "dummy-model",
            "LOKI_DUMMY_REPLY": reply,
        })
        workspace = os.path.join(tmpdir, "workspace")
        os.makedirs(workspace, exist_ok=True)
        configure_container(env, workspace)
        return env

    def _send(self, front, request_id, method, params):
        front.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }) + "\n")
        front.stdin.flush()

    def _response(self, front, request_id):
        preceding = []
        while True:
            line = front.stdout.readline()
            self.assertTrue(line, "ACP front produced no response")
            message = json.loads(line)
            if message.get("id") == request_id:
                return message, preceding
            preceding.append(message)

    def _initialize(self, front, *, agent_shell=False):
        params = {"protocolVersion": 1}
        if agent_shell:
            params["clientInfo"] = {
                "name": "agent-shell",
                "title": "Emacs Agent Shell",
                "version": "test",
            }
        self._send(front, 1, "initialize", params)
        return self._response(front, 1)[0]["result"]

    def _create_saved_session(self, env, tmpdir):
        workspace = _configured_workspace(tmpdir)
        front = self._front(env, tmpdir)
        try:
            self._initialize(front)
            self._send(front, 2, "session/new", {"cwd": workspace})
            session_id = self._response(front, 2)[0]["result"]["sessionId"]
            self._send(front, 3, "session/prompt", {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": "remember this"}],
            })
            response, _updates = self._response(front, 3)
            self.assertEqual(response["result"]["stopReason"], "end_turn")
            return session_id
        finally:
            front.stdin.close()
            front.wait(timeout=5)

    def test_load_always_replays_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._env(tmpdir)
            saved_id = self._create_saved_session(env, tmpdir)
            front = self._front(env, tmpdir)
            try:
                self._initialize(front)
                self._send(front, 2, "session/load", {
                    "sessionId": saved_id,
                    "cwd": _configured_workspace(tmpdir),
                    "mcpServers": [],
                    # This former Loki extension cannot suppress ACP's
                    # mandatory load replay.
                    "replay": False,
                })
                response, replayed = self._response(front, 2)
                self.assertNotIn("sessionId", response["result"])
                self.assertIn("configOptions", response["result"])
                kinds = [
                    message["params"]["update"]["sessionUpdate"]
                    for message in replayed
                    if message.get("method") == "session/update"
                ]
                self.assertIn("user_message_chunk", kinds)
                self.assertIn("agent_message_chunk", kinds)
                replayed_text = " ".join(
                    message["params"]["update"]["content"]["text"]
                    for message in replayed
                    if message.get("method") == "session/update"
                    and message["params"]["update"]["sessionUpdate"]
                    in ("user_message_chunk", "agent_message_chunk"))
                self.assertIn("remember this", replayed_text)
                self.assertIn("loadable answer", replayed_text)
            finally:
                front.stdin.close()
                front.wait(timeout=5)

    def test_resume_continues_without_replaying_history(self):
        from loki_agent import loki

        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._env(tmpdir)
            saved_id = self._create_saved_session(env, tmpdir)
            front = self._front(env, tmpdir)
            try:
                initialized = self._initialize(front, agent_shell=True)
                self.assertEqual(
                    initialized["agentCapabilities"]
                    ["sessionCapabilities"]["resume"],
                    {},
                )
                self._send(front, 2, "session/list",
                           {"cwd": _configured_workspace(tmpdir)})
                listed = self._response(front, 2)[0]
                self.assertIn(
                    saved_id,
                    [entry["sessionId"]
                     for entry in listed["result"]["sessions"]],
                )
                self._send(front, 3, "session/resume", {
                    "sessionId": saved_id,
                    "cwd": _configured_workspace(tmpdir),
                    "mcpServers": [],
                    # Nor can the old extension make resume replay.
                    "replay": True,
                })
                resumed, preceding = self._response(front, 3)
                self.assertIn("configOptions", resumed["result"])
                self.assertFalse(any(
                    message.get("method") == "session/update"
                    for message in preceding
                ), preceding)

                self._send(front, 4, "session/prompt", {
                    "sessionId": saved_id,
                    "prompt": [{
                        "type": "text",
                        "text": "after minimal resume",
                    }],
                })
                continued, updates = self._response(front, 4)
                self.assertEqual(
                    continued["result"]["stopReason"], "end_turn")
                self.assertTrue(any(
                    message.get("method") == "session/update"
                    and message["params"]["update"]["sessionUpdate"]
                    == "agent_message_chunk"
                    for message in updates
                ), updates)

                self._send(
                    front, 5, "session/close", {"sessionId": saved_id})
                self.assertEqual(self._response(front, 5)[0]["result"], {})
            finally:
                front.stdin.close()
                front.wait(timeout=5)

            saved_path = os.path.join(
                loki.chat_log_dir_for(_configured_workspace(tmpdir)),
                f"chat-{saved_id}.json")
            with open(saved_path, encoding="utf-8") as stream:
                persisted = stream.read()
            self.assertIn("remember this", persisted)
            self.assertIn("after minimal resume", persisted)


class SessionListTests(unittest.TestCase):
    def test_list_reports_saved_sessions_with_cwd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env.update({
                "HOME": tmpdir,
                "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
                "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
                "TERM": "dumb",
                "LOKI_PROVIDER": "dummy",
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_MODEL": "dummy-model",
                "LOKI_DUMMY_REPLY": "one",
            })
            workspace = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace, exist_ok=True)
            configure_container(env, workspace)
            front = subprocess.Popen(
                loki_acp_command(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, env=env, cwd=os.path.join(tmpdir, "workspace"))
            self.addCleanup(_close_process_streams, front)
            try:
                def send(m):
                    front.stdin.write(json.dumps(m) + "\n")
                    front.stdin.flush()

                def recv():
                    line = front.stdout.readline()
                    self.assertTrue(line)
                    return json.loads(line)

                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1}})
                while recv().get("id") != 1:
                    pass
                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": workspace}})
                while True:
                    m = recv()
                    if m.get("id") == 2:
                        break
                session_id = m["result"]["sessionId"]
                send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                      "params": {"sessionId": session_id,
                                 "prompt": [{"type": "text",
                                             "text": "hi"}]}})
                while True:
                    m = recv()
                    if m.get("id") == 3:
                        break
            finally:
                front.stdin.close()
                front.wait(timeout=5)

            # A second front process sees the saved conversation listed.
            front2 = subprocess.Popen(
                loki_acp_command(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, env=env, cwd=os.path.join(tmpdir, "workspace"))
            self.addCleanup(_close_process_streams, front2)
            try:
                front2.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": 1}}) + "\n")
                front2.stdin.flush()
                while True:
                    m = json.loads(front2.stdout.readline())
                    if m.get("id") == 1:
                        break
                front2.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": 2,
                    "method": "session/list",
                    "params": {}}) + "\n")
                front2.stdin.flush()
                m = json.loads(front2.stdout.readline())
                self.assertEqual(m["id"], 2)
                sessions = m["result"]["sessions"]
                self.assertEqual(len(sessions), 1, sessions)
                entry = sessions[0]
                self.assertEqual(entry["sessionId"], session_id)
                self.assertEqual(entry["cwd"], workspace)
                self.assertIn("updatedAt", entry)
            finally:
                front2.stdin.close()
                front2.wait(timeout=5)


class ConfigOptionTests(unittest.TestCase):
    def test_session_new_returns_model_options(self):
        # The dummy env has no usable catalog credentials, so the option
        # list carries only the explicit LOKI_* connection -- which is
        # enough to prove the option plumbing flows and is settable.
        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env.update({
                "HOME": tmpdir,
                "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
                "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
                "TERM": "dumb",
                "LOKI_PROVIDER": "dummy",
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_MODEL": "dummy-model",
                "LOKI_DUMMY_REPLY": "x",
            })
            workspace = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace, exist_ok=True)
            configure_container(env, workspace)
            front = subprocess.Popen(
                loki_acp_command(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, env=env, cwd=os.path.join(tmpdir, "workspace"))
            self.addCleanup(_close_process_streams, front)
            try:
                def send(m):
                    front.stdin.write(json.dumps(m) + "\n")
                    front.stdin.flush()

                def recv():
                    line = front.stdout.readline()
                    self.assertTrue(line)
                    return json.loads(line)

                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1}})
                while recv().get("id") != 1:
                    pass
                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": workspace}})
                while True:
                    m = recv()
                    if m.get("id") == 2:
                        break
                session_id = m["result"]["sessionId"]
                options = m["result"]["configOptions"]
                self.assertEqual(len(options), 1)
                model_option = options[0]
                self.assertEqual(model_option["id"], "model")
                self.assertEqual(model_option["category"], "model")
                self.assertEqual(model_option["type"], "select")
                self.assertEqual(model_option["currentValue"],
                                 "loki-explicit")
                values = [o["value"] for o in model_option["options"]]
                self.assertEqual(values, ["loki-explicit"])

                # Setting the value round-trips and echoes full state.
                send({"jsonrpc": "2.0", "id": 3,
                      "method": "session/set_config_option",
                      "params": {"sessionId": session_id,
                                 "configId": "model",
                                 "value": "loki-explicit"}})
                while True:
                    m = recv()
                    if m.get("id") == 3:
                        break
                self.assertIn("configOptions", m["result"])
                self.assertEqual(
                    m["result"]["configOptions"][0]["currentValue"],
                    "loki-explicit")
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()


class WorkerReasoningConfigTests(unittest.TestCase):
    def setUp(self):
        assume_endpoints_approved(self)

    @staticmethod
    def _profile(*values):
        return models.ReasoningEffortProfile(tuple(values))

    def test_model_changes_return_dependent_agent_shell_option(self):
        from loki_agent import loki
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        old_credentials = loki.CREDENTIALS
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                loki._DEFAULT_SESSION = session
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
                high_model = {
                    "id": "high-model",
                    "name": "High model",
                    "reasoning_options": [{
                        "type": "effort",
                        "values": ["low", "high"],
                    }],
                }
                low_model = {
                    "id": "low-model",
                    "name": "Low model",
                    "reasoning_options": [{
                        "type": "effort",
                        "values": ["low"],
                    }],
                }
                no_effort_model = {
                    "id": "plain-model",
                    "name": "Plain model",
                }
                loki.apply_runtime_config(
                    loki.config_from_modelsdev_selection(
                        "openrouter",
                        provider,
                        high_model,
                        loki.CREDENTIALS,
                    ))
                loki.new_chat_log(
                    os.path.join(tmpdir, "chat-reasoning.json"))
                worker = Worker(session, lambda _message: None, "session")
                worker._set_choices([
                    ({
                        "value": "openrouter/high-model",
                        "name": "High model",
                    }, ("openrouter", provider, high_model)),
                    ({
                        "value": "openrouter/low-model",
                        "name": "Low model",
                    }, ("openrouter", provider, low_model)),
                    ({
                        "value": "openrouter/plain-model",
                        "name": "Plain model",
                    }, ("openrouter", provider, no_effort_model)),
                ], "openrouter/high-model")

                initial = worker.config_options()
                thought = initial[1]
                self.assertEqual(
                    [option["category"] for option in initial],
                    ["model", "thought_level"],
                )
                self.assertEqual(thought["id"], "reasoning_effort")
                self.assertEqual(thought["currentValue"], "default")
                self.assertEqual(
                    [value["value"] for value in thought["options"]],
                    ["default", "effort:low", "effort:high"],
                )

                selected = worker.set_config_option({
                    "configId": "reasoning_effort",
                    "value": "effort:high",
                })
                self.assertEqual(
                    selected["configOptions"][1]["currentValue"],
                    "effort:high")

                plain = worker.set_config_option({
                    "configId": "model",
                    "value": "openrouter/plain-model",
                })
                self.assertEqual(len(plain["configOptions"]), 1)
                self.assertEqual(
                    loki.current_reasoning_effort_preference(), "high")
                restored = worker.set_config_option({
                    "configId": "model",
                    "value": "openrouter/high-model",
                })
                self.assertEqual(
                    restored["configOptions"][1]["currentValue"],
                    "effort:high")

                narrowed = worker.set_config_option({
                    "configId": "model",
                    "value": "openrouter/low-model",
                })
                narrowed_thought = narrowed["configOptions"][1]
                self.assertEqual(
                    narrowed_thought["currentValue"], "default")
                self.assertEqual(
                    [value["value"]
                     for value in narrowed_thought["options"]],
                    ["default", "effort:low"],
                )
                self.assertIn(
                    "preferred high is unavailable",
                    narrowed_thought["options"][0]["name"],
                )
                self.assertEqual(
                    loki.current_reasoning_effort_preference(), "high")
                self.assertIsNone(loki.effective_reasoning_effort())
                with self.assertRaises(acps.TransportError) as caught:
                    worker.set_config_option({
                        "configId": "reasoning_effort",
                        "value": "effort:max",
                    })
                self.assertEqual(
                    caught.exception.code, acps.INVALID_PARAMS)
                self.assertEqual(
                    loki.current_reasoning_effort_preference(), "high")

                restored = worker.set_config_option({
                    "configId": "model",
                    "value": "openrouter/high-model",
                })
                self.assertEqual(
                    restored["configOptions"][1]["currentValue"],
                    "effort:high")
                self.assertEqual(loki.effective_reasoning_effort(), "high")
        finally:
            loki._DEFAULT_SESSION = old_session
            loki.CREDENTIALS = old_credentials

    def test_effort_change_during_prompt_applies_to_next_turn(self):
        from loki_agent import loki, protocols
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        try:
            session = Session(shell_cwd=ROOT)
            session.runtime_config = loki.make_runtime_config(
                "https://api.openai.com/v1/responses",
                protocols.OPENAI_RESPONSES,
                model="gpt-test",
                provider_id="openai",
                reasoning_effort_profile=self._profile("low", "high"),
            )
            session.reasoning_effort_preference = "high"
            session.session_state = {"reasoning_effort": "high"}
            loki._DEFAULT_SESSION = session
            written = []
            worker = Worker(session, written.append, "session")
            worker._set_choices([
                ({
                    "value": "model",
                    "name": "Model",
                }, object()),
            ], "model")
            started = asyncio.Event()
            release = asyncio.Event()
            snapshots = []

            async def run_turn(_on_event, reasoning_effort):
                snapshots.append(reasoning_effort)
                started.set()
                await release.wait()

            worker._run_turn = run_turn

            async def scenario():
                await worker.handle({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": "session",
                        "prompt": [{
                            "type": "text",
                            "text": "hello",
                        }],
                    },
                }, concurrent=True)
                await started.wait()
                await worker.handle({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/set_config_option",
                    "params": {
                        "sessionId": "session",
                        "configId": "reasoning_effort",
                        "value": "effort:low",
                    },
                }, concurrent=True)
                release.set()
                await worker._prompt_task

            asyncio.run(scenario())

            self.assertEqual(snapshots, ["high"])
            self.assertEqual(
                loki.current_reasoning_effort_preference(), "low")
            response = next(
                message for message in written
                if message.get("id") == 2)
            self.assertEqual(
                response["result"]["configOptions"][1]["currentValue"],
                "effort:low",
            )
        finally:
            loki._DEFAULT_SESSION = old_session


class TtyStdinTests(unittest.TestCase):
    """The front must work when stdin is a tty, not just a pipe.

    ACP owns its protocol stdin and reads fd 0 directly. This test gives
    the shipped front executable a real controlling pty, exactly like an
    interactive manual run.
    """

    def test_front_answers_initialize_with_tty_stdin(self):
        import fcntl
        import pty
        import termios
        import time as _time
        master, slave = pty.openpty()

        def child_setup():
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env.update({
                "HOME": tmpdir,
                "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
                "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
                "TERM": "dumb",
                "LOKI_PROVIDER": "dummy",
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_MODEL": "dummy-model",
                "LOKI_DUMMY_REPLY": "x",
            })
            os.makedirs(os.path.join(tmpdir, "workspace"), exist_ok=True)
            proc = subprocess.Popen(
                loki_acp_command(),
                stdin=slave, stdout=subprocess.PIPE,
                env=env, cwd=os.path.join(tmpdir, "workspace"),
                preexec_fn=child_setup, text=True)
            self.addCleanup(_close_process_streams, proc)
            os.close(slave)
            try:
                os.write(master, (json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": 1},
                }) + "\n").encode())
                reply = None
                deadline = _time.monotonic() + 8
                while _time.monotonic() < deadline:
                    line = proc.stdout.readline()
                    if line:
                        reply = json.loads(line)
                        break
                self.assertIsNotNone(reply, "no reply to initialize")
                self.assertEqual(reply["id"], 1)
                self.assertEqual(
                    reply["result"]["agentInfo"]["name"], "loki")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                os.close(master)


class WireCwdTests(unittest.TestCase):
    """session/open's cwd lands in the session's virtual shell_cwd.

    Workers inherit the front process's cwd; the conversation's working
    directory arrives over the wire, so a tool call with a relative path
    resolves inside the session directory, not the worker's.
    """

    def test_open_sets_shell_cwd_and_tools_resolve(self):
        from loki_agent.acp_worker import Worker
        from loki_agent.credentials import CredentialStore
        from loki_agent.sessions import Session
        from loki_agent import loki

        async def run():
            session = Session(shell_cwd="/")  # worker cwd, deliberately wrong
            worker = Worker(session, lambda message: None)
            old_credentials = loki.CREDENTIALS
            old_session = loki._DEFAULT_SESSION
            try:
                loki.CREDENTIALS = CredentialStore({})
                loki._DEFAULT_SESSION = session
                await worker.prepare_open({
                    "sessionId": "w",
                    "cwd": ROOT,
                    "openMethod": "session/new",
                })
                worker.commit_open()
                result = await loki.dispatch_tool_async(
                    "Bash", {"command": "cat tests/test_acps.py",
                             "description": "probe"})
                return session.shell_cwd, result
            finally:
                loki.CREDENTIALS = old_credentials
                loki._DEFAULT_SESSION = old_session

        shell_cwd, result = asyncio.run(run())
        self.assertEqual(shell_cwd, ROOT)
        self.assertTrue(result["ok"])
        self.assertIn("WireCwdTests", result["content"])


class WorkerSessionContractTests(unittest.TestCase):
    def setUp(self):
        assume_endpoints_approved(self)

    def test_worker_close_reaps_session_owned_jobs(self):
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        session = Session(shell_cwd="/tmp")
        manager = mock.Mock()
        manager.close_session_owned = mock.AsyncMock()
        session.job_manager = manager
        worker = Worker(session, lambda message: None, "session")

        asyncio.run(worker.close())

        manager.close_session_owned.assert_awaited_once_with()

    def test_worker_rejects_unknown_open_method(self):
        from loki_agent import loki
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        try:
            session = Session(shell_cwd=ROOT)
            loki._DEFAULT_SESSION = session
            worker = Worker(session, lambda _message: None)
            with self.assertRaisesRegex(
                    acps.TransportError,
                    "unsupported session opening method"):
                asyncio.run(worker.prepare_open({
                    "sessionId": "saved",
                    "cwd": ROOT,
                    "openMethod": "session/unknown",
                }))
            self.assertIsNone(worker._pending_open)
        finally:
            loki._DEFAULT_SESSION = old_session

    def test_restore_rejects_a_different_saved_cwd(self):
        from loki_agent import formats, loki
        from loki_agent.acp_worker import Worker
        from loki_agent.credentials import CredentialStore
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        old_credentials = loki.CREDENTIALS
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                saved_cwd = os.path.join(tmpdir, "saved-cwd")
                requested_cwd = os.path.join(tmpdir, "requested-cwd")
                # The saved log lives in the workspace the client names.
                chat_dir = loki.chat_log_dir_for(requested_cwd)
                os.mkdir(saved_cwd)
                os.mkdir(requested_cwd)
                os.makedirs(chat_dir)
                loki.CREDENTIALS = CredentialStore({})

                for stored_cwd in (saved_cwd, "invalid\x00cwd"):
                    blob = formats.new_log_blob(
                        loki.initial_transcript_items(), [])
                    blob["session_state"] = {"shell_cwd": stored_cwd}
                    with open(
                            os.path.join(chat_dir, "chat-saved.json"),
                            "w", encoding="utf-8") as stream:
                        json.dump(blob, stream)
                    for method in acp.RESTORE_METHODS:
                        with self.subTest(
                                method=method, stored_cwd=stored_cwd):
                            session = Session(shell_cwd="/")
                            loki._DEFAULT_SESSION = session
                            worker = Worker(session, lambda _message: None)
                            with self.assertRaisesRegex(
                                    acps.TransportError,
                                    ("cwd does not match"
                                     if stored_cwd == saved_cwd
                                     else "could not load saved session")
                            ) as caught:
                                asyncio.run(worker.prepare_open({
                                    "sessionId": "saved",
                                    "cwd": requested_cwd,
                                    "openMethod": method,
                                }))
                            self.assertEqual(
                                caught.exception.code, acps.INVALID_PARAMS)
                            self.assertIsNone(worker._pending_open)
        finally:
            loki._DEFAULT_SESSION = old_session
            loki.CREDENTIALS = old_credentials

    def test_subscription_option_and_resume_use_brokered_credential(self):
        from loki_agent import formats, http_client, loki, models
        from loki_agent.acp_worker import Worker
        from loki_agent.credentials import CredentialInventory
        from loki_agent.sessions import Session

        credential = (
            authentications.CredentialRef.openai_subscription())
        catalog = models.add_openai_subscription_catalog(
            models.normalize_catalog({
                "openai": {
                    "id": "openai",
                    "name": "OpenAI",
                    "npm": "@ai-sdk/openai",
                    "env": ["OPENAI_API_KEY"],
                    "models": {},
                },
            }),
            {
                "models": [{
                    "slug": "gpt-test",
                    "display_name": "GPT Test",
                    "visibility": "list",
                    "input_modalities": ["text"],
                    "supported_reasoning_levels": [
                        {"effort": "low", "description": "Low"},
                    ],
                    "default_reasoning_level": "low",
                    "supports_reasoning_summaries": True,
                    "default_reasoning_summary": "none",
                    "support_verbosity": True,
                    "default_verbosity": "low",
                    "supports_parallel_tool_calls": False,
                    "use_responses_lite": True,
                    "tool_mode": "code_mode_only",
                    "context_window": 200000,
                    "base_instructions":
                        "must not be copied into the session log",
                }],
            },
        )
        groups = models.build_groups(catalog)
        refreshed_catalog = models.add_openai_subscription_catalog(
            {},
            {
                "models": [{
                    "slug": "gpt-test",
                    "display_name": "GPT Test",
                    "visibility": "list",
                    "input_modalities": ["text"],
                    "supported_reasoning_levels": [
                        {"effort": "high", "description": "High"},
                    ],
                    "default_reasoning_level": "high",
                    "supports_reasoning_summaries": True,
                    "default_reasoning_summary": "detailed",
                    "support_verbosity": True,
                    "default_verbosity": "medium",
                    "supports_parallel_tool_calls": True,
                    "use_responses_lite": False,
                }],
            },
        )
        refreshed_groups = models.build_groups(refreshed_catalog)
        broker = authentications.CredentialBroker()
        broker.install_openai_subscription(
            authentications.OpenAITokenSet(
                access_token="access-secret",
                refresh_token="refresh-secret",
                expires_at=10**12,
            ))
        old_session = loki._DEFAULT_SESSION
        old_credentials = loki.CREDENTIALS
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loki.CREDENTIALS = CredentialInventory(
                    {}, {credential})
                requests = []

                async def request(method, url, **kwargs):
                    requests.append((
                        method,
                        url,
                        dict(kwargs["headers_in"]),
                    ))
                    return http_client.HttpResponse(
                        url, 200, "OK", {}, b"{}")

                async def scenario():
                    first_session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                    first_session.credential_authority = broker
                    loki._DEFAULT_SESSION = first_session
                    first_worker = Worker(
                        first_session,
                        lambda _message: None,
                        "first",
                    )
                    with mock.patch.object(
                            models,
                            "ensure_index",
                            new=mock.AsyncMock(
                                return_value=(catalog, groups))):
                        await first_worker.prepare_open({
                            "sessionId": "first",
                            "cwd": tmpdir,
                            "openMethod": "session/new",
                        })
                        opened = first_worker.commit_open()
                    options = opened["configOptions"][0]["options"]
                    value = "openai-subscription/gpt-test"
                    self.assertIn(
                        value,
                        [option["value"] for option in options],
                    )
                    first_worker.set_config_option({
                        "configId": "model",
                        "value": value,
                    })
                    first_provider = (
                        loki.current_config().chat_provider)
                    first_profile = first_provider.openai_request_profile
                    first_payload = (
                        loki.current_config().chat_provider.chat_payload(
                            [formats.message_item("user", "hello")],
                            [],
                            "gpt-test",
                            prompt_cache_key=(
                                first_session.conversation_id),
                        ))
                    with mock.patch.object(
                            http_client,
                            "async_http_request",
                            new=request):
                        await loki.async_provider_request(
                            "POST",
                            loki.current_config().chat_provider.input_url,
                            {})
                    saved_id = "first"
                    with open(
                            first_session.chat_log_path,
                            "r", encoding="utf-8") as stream:
                        saved_text = stream.read()
                    await first_worker.close()

                    resumed_session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                    resumed_session.credential_authority = broker
                    loki._DEFAULT_SESSION = resumed_session
                    resumed_worker = Worker(
                        resumed_session,
                        lambda _message: None,
                        "resumed",
                    )
                    with mock.patch.object(
                            models,
                            "ensure_index",
                            new=mock.AsyncMock(
                                return_value=(
                                    refreshed_catalog,
                                    refreshed_groups,
                                ))):
                        prepared = await resumed_worker.prepare_open({
                            "sessionId": saved_id,
                            "cwd": tmpdir,
                            "openMethod": "session/resume",
                        })
                        self.assertIsNone(resumed_session.runtime_config)
                        self.assertEqual(
                            prepared["authorizationConnection"]["model"],
                            "gpt-test",
                        )
                        with open(
                                resumed_session.chat_log_path,
                                "r", encoding="utf-8") as stream:
                            self.assertEqual(stream.read(), saved_text)
                        resumed = resumed_worker.commit_open()
                    with open(
                            resumed_session.chat_log_path,
                            "r", encoding="utf-8") as stream:
                        committed_text = stream.read()
                    self.assertNotEqual(committed_text, saved_text)
                    self.assertEqual(
                        resumed["configOptions"][0]["currentValue"],
                        "loki-saved",
                    )
                    resumed_provider = (
                        loki.current_config().chat_provider)
                    resumed_profile = (
                        resumed_provider.openai_request_profile)
                    resumed_payload = (
                        loki.current_config().chat_provider.chat_payload(
                            [formats.message_item("user", "hello")],
                            [],
                            "gpt-test",
                            prompt_cache_key=(
                                resumed_session.conversation_id),
                        ))
                    with mock.patch.object(
                            http_client,
                            "async_http_request",
                            new=request):
                        await loki.async_provider_request(
                            "POST",
                            loki.current_config().chat_provider.input_url,
                            {})
                    await resumed_worker.close()
                    return (
                        first_profile,
                        resumed_profile,
                        first_payload,
                        resumed_payload,
                        saved_text,
                    )

                (
                    first_profile,
                    resumed_profile,
                    first_payload,
                    resumed_payload,
                    saved_text,
                ) = asyncio.run(scenario())

                self.assertNotEqual(first_profile, resumed_profile)
                self.assertFalse(hasattr(first_profile, "tool_mode"))
                self.assertFalse(hasattr(first_profile, "context_window"))
                self.assertEqual(
                    first_payload["reasoning"],
                    {"effort": "low", "context": "all_turns"},
                )
                self.assertEqual(
                    first_payload["text"], {"verbosity": "low"})
                self.assertFalse(first_payload["parallel_tool_calls"])
                self.assertEqual(
                    resumed_payload["reasoning"],
                    {"effort": "high", "summary": "detailed"},
                )
                self.assertEqual(
                    resumed_payload["text"], {"verbosity": "medium"})
                self.assertTrue(resumed_payload["parallel_tool_calls"])
                self.assertNotIn("context", resumed_payload["reasoning"])
                self.assertNotIn(
                    "must not be copied into the session log", saved_text)
                self.assertEqual(len(requests), 2)
                for _method, url, headers in requests:
                    self.assertEqual(
                        url,
                        authentications.OPENAI_CHATGPT_RESPONSES_URL,
                    )
                    self.assertEqual(
                        headers["Authorization"],
                        "Bearer access-secret",
                    )
        finally:
            loki._DEFAULT_SESSION = old_session
            loki.CREDENTIALS = old_credentials

    def test_restore_accepts_equivalent_client_cwd(self):
        from unittest import mock
        from loki_agent import formats, loki, protocols
        from loki_agent.acp_worker import Worker
        from loki_agent.connections import ConnectionDescriptor
        from loki_agent.credentials import CredentialStore
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        old_credentials = loki.CREDENTIALS
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                saved_cwd = os.path.join(tmpdir, "saved-cwd")
                client_cwd = os.path.join(tmpdir, "client-cwd")
                os.mkdir(saved_cwd)
                os.symlink(saved_cwd, client_cwd)
                # The saved log lives in the workspace the client names -- here
                # a symlink to the saved directory.
                chat_dir = loki.chat_log_dir_for(client_cwd)
                os.makedirs(chat_dir)
                descriptor = ConnectionDescriptor(
                    provider_id="saved-provider",
                    provider_name="Saved Provider",
                    model="saved-model",
                    chat_url=(
                        "https://saved.example/v1/chat/completions"),
                    models_url="https://saved.example/v1/models",
                    protocol=protocols.OPENAI_CHAT,
                    credential_ref=(
                        authentications.CredentialRef.environment(
                            "SAVED_API_KEY")),
                    max_tokens=777,
                )
                blob = formats.new_log_blob(
                    loki.initial_transcript_items(), [])
                blob["session_state"] = {
                    "shell_cwd": saved_cwd,
                    "connection": descriptor.to_dict(),
                }
                saved_name = "chat-saved.json"
                with open(
                        os.path.join(chat_dir, saved_name),
                        "w", encoding="utf-8") as stream:
                    json.dump(blob, stream)

                session = Session(shell_cwd="/")
                loki._DEFAULT_SESSION = session
                loki.CREDENTIALS = CredentialStore({
                    "SAVED_API_KEY": "secret",
                })
                worker = Worker(session, lambda message: None)
                with mock.patch(
                        "loki_agent.acp_worker.modelsdev.ensure_index",
                        new=mock.AsyncMock(return_value=({}, {}))):
                    async def open_worker():
                        await worker.prepare_open({
                            "sessionId": "saved",
                            "cwd": client_cwd,
                            "openMethod": "session/resume",
                        })
                        return worker.commit_open()

                    result = asyncio.run(open_worker())

                self.assertEqual(session.shell_cwd, client_cwd)
                self.assertEqual(session.model, "saved-model")
                self.assertEqual(
                    session.runtime_config.chat_provider.max_tokens, 777)
                self.assertEqual(
                    result["configOptions"][0]["currentValue"],
                    "loki-saved",
                )
        finally:
            loki._DEFAULT_SESSION = old_session
            loki.CREDENTIALS = old_credentials

    def test_new_sessions_get_distinct_persistent_logs_with_empty_catalog(self):
        from unittest import mock
        from loki_agent import loki, models
        from loki_agent.acp_worker import Worker
        from loki_agent.credentials import CredentialStore
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        old_credentials = loki.CREDENTIALS
        paths = []
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loki.CREDENTIALS = CredentialStore({})
                for number in range(2):
                    session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                    loki._DEFAULT_SESSION = session
                    worker = Worker(session, lambda message: None)
                    with mock.patch.object(
                            models, "ensure_index",
                            new=mock.AsyncMock(return_value=({}, {}))):
                        async def open_worker():
                            await worker.prepare_open({
                                "sessionId": f"live-{number}",
                                "cwd": tmpdir,
                                "openMethod": "session/new",
                            })
                            return worker.commit_open()

                        result = asyncio.run(open_worker())
                    paths.append(session.chat_log_path)
                    option = result["configOptions"][0]
                    self.assertEqual(
                        option["currentValue"], "loki-disconnected")
                    self.assertIn(
                        option["currentValue"],
                        [entry["value"] for entry in option["options"]],
                    )
                self.assertNotEqual(paths[0], paths[1])
                # Session paths are stored through os.path.realpath; compare
                # resolved directories, or a differently-spelled temp path
                # (Windows short names) fails on spelling alone.
                log_dir = os.path.realpath(loki.chat_log_dir_for(tmpdir))
                self.assertTrue(all(
                    os.path.realpath(os.path.dirname(path)) == log_dir
                    for path in paths))
        finally:
            loki._DEFAULT_SESSION = old_session
            loki.CREDENTIALS = old_credentials

    def test_provider_failure_is_an_acp_request_failure(self):
        from loki_agent import formats, loki, protocols
        from loki_agent.acp_worker import TurnFailure, Worker
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                session.transcript_items = [
                    formats.instruction_item("system"),
                ]
                loki._DEFAULT_SESSION = session
                worker = Worker(session, lambda message: None, "s")

                async def failed_turn(on_event, _reasoning_effort):
                    on_event({
                        "type": "provider_error",
                        "error": protocols.ProtocolError(
                            "invalid provider JSON"),
                    })

                worker._run_turn = failed_turn
                with self.assertRaisesRegex(
                        TurnFailure, "invalid provider JSON"):
                    asyncio.run(worker.prompt({
                        "sessionId": "s",
                        "prompt": [{"type": "text", "text": "hello"}],
                    }))
        finally:
            loki._DEFAULT_SESSION = old_session

    def test_stop_reasons_cover_protocol_terminal_states(self):
        from loki_agent.acp_worker import Worker

        self.assertEqual(
            Worker._stop_reason([{"type": "response_cancelled"}]),
            "cancelled",
        )
        self.assertEqual(
            Worker._stop_reason([{"type": "max_loops"}]),
            "max_turn_requests",
        )
        self.assertEqual(
            Worker._stop_reason([{"type": "response_incomplete"}]),
            "max_tokens",
        )
        self.assertEqual(
            Worker._stop_reason([{"type": "response_refusal"}]),
            "refusal",
        )

    def test_cancelled_prompt_returns_cancelled_even_if_operation_raises(self):
        from loki_agent import loki
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                loki._DEFAULT_SESSION = session
                worker = Worker(session, lambda message: None, "s")

                async def cancelled_failure(
                        on_event, _reasoning_effort):
                    worker.cancel_event.set()
                    raise RuntimeError("underlying close race")

                worker._run_turn = cancelled_failure
                result = asyncio.run(worker.prompt({
                    "sessionId": "s",
                    "prompt": [{"type": "text", "text": "hello"}],
                }))
                self.assertEqual(result["stopReason"], "cancelled")
        finally:
            loki._DEFAULT_SESSION = old_session

    def test_prompt_supports_baseline_resource_links(self):
        from loki_agent import formats, loki
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                session = Session(shell_cwd=os.path.join(tmpdir, "workspace"))
                loki._DEFAULT_SESSION = session
                worker = Worker(session, lambda message: None, "s")

                async def no_turn(on_event, _reasoning_effort):
                    return None

                worker._run_turn = no_turn
                result = asyncio.run(worker.prompt({
                    "sessionId": "s",
                    "prompt": [
                        {"type": "text", "text": "inspect this"},
                        {
                            "type": "resource_link",
                            "name": "source.py",
                            "uri": "file:///tmp/source.py",
                            "mimeType": "text/x-python",
                        },
                    ],
                }))
                text = formats.item_text(
                    session.transcript_items[-1])
                self.assertIn("inspect this", text)
                self.assertIn("file:///tmp/source.py", text)
                self.assertEqual(result["stopReason"], "end_turn")
        finally:
            loki._DEFAULT_SESSION = old_session

    def test_refused_prompt_is_retained_persisted_replayed_and_reused(self):
        import copy
        from unittest import mock
        from loki_agent import formats, loki, protocols, savefiles
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                initial = [formats.instruction_item("system")]
                chat_path = os.path.join(tmpdir, "chat-refusal.json")
                session = Session(
                    shell_cwd=os.path.join(tmpdir, "workspace"),
                    transcript_items=list(initial),
                    chat_log_path=chat_path,
                )
                loki._DEFAULT_SESSION = session
                messages = []
                worker = Worker(session, messages.append, "s")
                turns = [
                    formats.DecodedTurn(
                        [formats.message_item(
                            "assistant",
                            [{"type": "refusal", "text": "No"}],
                        )],
                        metadata={
                            "protocol": protocols.OPENAI_CHAT,
                            "stop_reason": "refusal",
                        },
                    ),
                    formats.DecodedTurn(
                        [formats.message_item("assistant", "Continued")],
                        metadata={
                            "protocol": protocols.OPENAI_CHAT,
                            "stop_reason": "end_turn",
                        },
                    ),
                ]
                requests = []

                async def completion(items, *_args, **_kwargs):
                    requests.append(copy.deepcopy(items))
                    return turns.pop(0)

                async def scenario():
                    with mock.patch.object(
                            loki, "current_model", return_value="model"
                    ), mock.patch.object(
                            loki, "async_chat_completion", new=completion
                    ):
                        refused = await worker.prompt({
                            "sessionId": "s",
                            "prompt": [
                                {"type": "text", "text": "unsafe"},
                            ],
                        })
                        with open(
                                chat_path, encoding="utf-8") as chat_file:
                            persisted = savefiles.read_chat_log(chat_file)[0]
                        messages.clear()
                        worker._replay_transcript()
                        replayed = list(messages)
                        messages.clear()
                        continued = await worker.prompt({
                            "sessionId": "s",
                            "prompt": [
                                {"type": "text", "text": "why?"},
                            ],
                        })
                        return refused, continued, persisted, replayed

                refused, continued, persisted, replayed = asyncio.run(
                    scenario())

                self.assertEqual(refused["stopReason"], "refusal")
                self.assertEqual(continued["stopReason"], "end_turn")
                self.assertEqual(
                    [event["type"] for event in persisted],
                    ["message", "message", "model_response"],
                )
                self.assertEqual(formats.item_text(persisted[1]), "unsafe")
                self.assertEqual(formats.item_text(persisted[2]), "No")
                self.assertEqual(
                    [formats.item_text(event) for event in requests[1]],
                    ["system", "unsafe", "No", "why?"],
                )
                replay_updates = [
                    message["params"]["update"]
                    for message in replayed
                    if message.get("method") == "session/update"
                ]
                replay_text = [
                    update["content"]["text"]
                    for update in replay_updates
                    if update.get("sessionUpdate") in (
                        "user_message_chunk", "agent_message_chunk")
                ]
                self.assertIn("unsafe", replay_text)
                self.assertIn("No", replay_text)
        finally:
            loki._DEFAULT_SESSION = old_session

    def test_refusal_does_not_erase_prior_tool_activity_in_prompt(self):
        from loki_agent import formats, loki
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        old_session = loki._DEFAULT_SESSION
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                session = Session(
                    shell_cwd=os.path.join(tmpdir, "workspace"),
                    transcript_items=[formats.instruction_item("system")],
                )
                loki._DEFAULT_SESSION = session
                worker = Worker(session, lambda message: None, "s")
                call = {
                    "type": "function_call",
                    "name": "Read",
                    "call_id": "call-1",
                    "arguments": {"file_path": "source.py"},
                }

                async def refused_after_tool(
                        on_event, _reasoning_effort):
                    session.transcript_items.append(
                        formats.DecodedTurn([call]).to_event())
                    session.transcript_items.append(
                        formats.tool_result_for_call(call, "file contents"))
                    session.transcript_items.append(
                        formats.DecodedTurn([
                            formats.message_item(
                                "assistant",
                                [{"type": "refusal", "text": "No"}],
                            ),
                        ]).to_event())
                    on_event({"type": "response_refusal"})

                worker._run_turn = refused_after_tool
                result = asyncio.run(worker.prompt({
                    "sessionId": "s",
                    "prompt": [{"type": "text", "text": "inspect"}],
                }))

                self.assertEqual(result["stopReason"], "refusal")
                self.assertEqual(
                    [event["type"] for event in session.transcript_items],
                    [
                        "message",
                        "message",
                        "model_response",
                        "tool_result",
                        "model_response",
                    ],
                )
                formats.validate_events(session.transcript_items)
        finally:
            loki._DEFAULT_SESSION = old_session


class ConfigEndpointApprovalTests(unittest.IsolatedAsyncioTestCase):
    """The ACP front asks before a catalog endpoint receives a credential."""

    PAIR = {
        "providerId": "acme",
        "endpoint": "https://acme.invalid/v1",
        "credential": "env:ACME_API_KEY",
        "changed": False,
        "approvedEndpoint": None,
        "approvedCredential": None,
    }

    def _front(self, *, supports_elicitation=True):
        front = acp.Front(
            lambda: None, lambda message: None, CredentialStore({}))
        front._client_supports_form_elicitation = supports_elicitation
        return front

    def _channel(self, selection):
        channel = mock.Mock()
        channel.session_id = "s"
        channel.request = mock.AsyncMock(return_value=selection)
        return channel

    async def test_accepted_approval_records_the_pair_and_asks(self):
        front = self._front()
        channel = self._channel(dict(self.PAIR))
        front._request_client = mock.AsyncMock(
            return_value={"action": "accept", "content": {"approve": True}})

        with mock.patch.object(acp.endpoint_pins, "record") as record:
            await front._approve_config_endpoint(
                channel, {"sessionId": "s"}, 9)

        params = front._request_client.await_args.args[1]
        self.assertEqual(params["requestId"], 9)
        self.assertEqual(params["mode"], "form")
        self.assertIn("https://acme.invalid/v1", params["message"])
        self.assertIn("env:ACME_API_KEY", params["message"])
        record.assert_called_once_with(
            "acme", "https://acme.invalid/v1", "env:ACME_API_KEY")

    async def test_declined_approval_fails_the_switch(self):
        front = self._front()
        channel = self._channel(dict(self.PAIR))
        front._request_client = mock.AsyncMock(
            return_value={"action": "decline"})

        with mock.patch.object(acp.endpoint_pins, "record") as record:
            with self.assertRaises(acps.TransportError):
                await front._approve_config_endpoint(
                    channel, {"sessionId": "s"}, 9)

        record.assert_not_called()

    async def test_changed_pair_shows_the_approved_values(self):
        front = self._front()
        selection = dict(self.PAIR, changed=True,
                         approvedEndpoint="https://old.invalid/v1",
                         approvedCredential="env:OLD_KEY")
        channel = self._channel(selection)
        front._request_client = mock.AsyncMock(
            return_value={"action": "accept", "content": {"approve": True}})

        with mock.patch.object(acp.endpoint_pins, "record"):
            await front._approve_config_endpoint(
                channel, {"sessionId": "s"}, 9)

        message = front._request_client.await_args.args[1]["message"]
        self.assertIn("https://old.invalid/v1", message)
        self.assertIn("env:OLD_KEY", message)
        self.assertIn("https://acme.invalid/v1", message)

    async def test_nothing_to_approve_asks_nothing(self):
        front = self._front()
        channel = self._channel({})
        front._request_client = mock.AsyncMock()

        with mock.patch.object(acp.endpoint_pins, "record") as record:
            await front._approve_config_endpoint(
                channel, {"sessionId": "s"}, 9)

        front._request_client.assert_not_awaited()
        record.assert_not_called()

    async def test_client_without_form_elicitation_fails_closed(self):
        front = self._front(supports_elicitation=False)
        channel = self._channel(dict(self.PAIR))
        front._request_client = mock.AsyncMock()

        with self.assertRaises(acps.TransportError):
            await front._approve_config_endpoint(
                channel, {"sessionId": "s"}, 9)

        front._request_client.assert_not_awaited()

    async def test_config_option_change_goes_through_the_approval(self):
        front = self._front()
        channel = self._channel({})
        front.workers["s"] = channel
        front._approve_config_endpoint = mock.AsyncMock()

        await front.forward_to_worker(
            "session/set_config_option", {"sessionId": "s", "value": "v"},
            request_id=9)

        front._approve_config_endpoint.assert_awaited_once()
        self.assertEqual(
            channel.request.await_args.args[0],
            "session/set_config_option")


if __name__ == "__main__":
    unittest.main()
