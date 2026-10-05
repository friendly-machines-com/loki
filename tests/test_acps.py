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
from contextlib import ExitStack, asynccontextmanager
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


class _FrontInput:
    """One serial test input stream, not an application prompt queue.

    A send acknowledges routing by Front.run, never completion of the request.
    This catches handlers which starve their own reverse responses or EOF.
    """

    def __init__(self, front):
        self.front = front
        self.messages = asyncio.Queue()
        front.read = self.read
        self.task = asyncio.create_task(front.run())

    async def read(self):
        while True:
            message, routed = await self.messages.get()
            if message is None:
                routed.set_result(None)
                return
            yield message
            routed.set_result(None)

    async def send(self, message):
        await self.send_many(message)

    async def send_many(self, *messages):
        routed = []
        for message in messages:
            future = asyncio.get_running_loop().create_future()
            self.messages.put_nowait((message, future))
            routed.append(future)
        await asyncio.wait_for(asyncio.gather(*routed), 3)

    async def finish(self):
        await self.send(None)
        await asyncio.wait_for(self.task, 5)

    async def close(self):
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


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
        self.assertFalse(any(owner.state == "opening"
                             for owner in front._sessions.values()))


class FrontPromptOrderingTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffered_restore_close_settles_unstarted_request(self):
        responses = []
        changed = asyncio.Event()

        def write(message):
            responses.append(message)
            changed.set()

        front = acp.Front(lambda: None, write, CredentialStore({}))
        source = _FrontInput(front)
        self.addAsyncCleanup(source.close)
        with mock.patch.object(front.credential_supervisor, 'delegate',
                               new=mock.AsyncMock()) as delegate:
            await source.send_many(
                acps.request(1, 'session/resume', {'sessionId': 'saved', 'cwd': ROOT}),
                acps.request(2, 'session/close', {'sessionId': 'saved'}))
            async with asyncio.timeout(3):
                while len(responses) < 2:
                    changed.clear()
                    await changed.wait()
            by_id = {message['id']: message for message in responses}
            self.assertEqual(by_id[1]['error']['message'], 'session operation was closed')
            self.assertEqual(by_id[2], acps.response(2, result={}))
            self.assertEqual(len(responses), 2)
            delegate.assert_not_awaited()
            self.assertFalse(front._sessions)
            await source.finish()
            self.assertFalse(front._tasks)

    async def test_buffered_config_cannot_overtake_prompt(self):
        order = []
        responses = []
        prompt_reply = asyncio.get_running_loop().create_future()
        forwarding = asyncio.Event()
        entered = asyncio.Event()
        changed = asyncio.Event()

        def write(message):
            responses.append(message)
            changed.set()

        class Channel:
            _closed = False
            process = mock.Mock(returncode=None)

            async def close(self):
                pass

            async def request(self, method, params, forwarded=None):
                order.append(method)
                changed.set()
                if method == "session/prompt":
                    entered.set()
                    await forwarding.wait()
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
            lambda: None, write, CredentialStore({}))
        owner = front._reserve_session("session")
        owner.channel = Channel()
        owner.state = "active"
        source = _FrontInput(front)
        self.addAsyncCleanup(source.close)

        await source.send_many({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/prompt",
            "params": {
                "sessionId": "session",
                "prompt": [{"type": "text", "text": "hello"}],
            },
        }, {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/set_config_option",
            "params": {
                "sessionId": "session",
                "configId": "reasoning_effort",
                "value": "effort:low",
            },
        })

        await asyncio.wait_for(entered.wait(), 3)
        # Even a worker-pipe forwarding wait belongs outside the sole reader.
        await source.send(acps.request(3, 'unknown-method', {}))
        async with asyncio.timeout(3):
            while not any(message.get('id') == 3 for message in responses):
                changed.clear()
                await changed.wait()
        self.assertEqual(order, ['session/prompt'])
        self.assertEqual(next(m for m in responses if m.get('id') == 3)['error']['code'],
                         acps.METHOD_NOT_FOUND)
        forwarding.set()
        async with asyncio.timeout(3):
            while len(order) < 3:
                changed.clear()
                await changed.wait()
        self.assertEqual(order, [
            "session/prompt",
            "session/describe_config_selection",
            "session/set_config_option",
        ])
        prompt_reply.set_result({"stopReason": "end_turn"})
        await asyncio.gather(*front._tasks)
        self.assertEqual(
            {message["id"] for message in responses}, {1, 2, 3})
        await source.finish()


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

            def revoke_now(self):
                return None

            async def close(self):
                return None

        class Channel:
            def __init__(self, session_id, process, forward, delegation):
                self.session_id = session_id
                self.closed = False
                self._closed = False
                self.process = mock.Mock(returncode=None)
                self._reader_task = asyncio.get_running_loop().create_future()

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
                self._closed = True
                if not self._reader_task.done():
                    self._reader_task.set_result(None)

        def write(message):
            messages.append(message)
            if message.get("method") == "elicitation/create":
                async def respond():
                    front.handle(acps.response(
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
                self.assertNotIn("saved", front._sessions)

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
            "identity = os.fstat(null)\n"
            "saved = os.dup(1)\n"
            "acps.quarantine_stdout(null)\n"
            # quarantine_stdout only dup2s the descriptor; the caller's fd must
            # still be the same object, or the close below would hit another.
            "assert os.path.samestat(os.fstat(null), identity)\n"
            "write = acps.make_writer(saved)\n"
            "write(acps.response(1, result={'ok': True}))\n"
            "print('stray output')\n"
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


class _ACPFrontFixture:
    async def _front(self, env, workspace):
        process = await asyncio.create_subprocess_exec(
            *loki_acp_command(), env=env, cwd=workspace,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        # Drain diagnostics while the child runs; a full stderr pipe must not
        # stall the protocol. Register the backstop before readiness assertions.
        diagnostics = asyncio.create_task(process.stderr.read())

        async def cleanup():
            if process.returncode is None:
                process.kill()
            await asyncio.wait_for(process.wait(), 5)
            await asyncio.wait_for(diagnostics, 5)
        self.addAsyncCleanup(cleanup)
        return process, diagnostics

    async def _response(self, front, request_id):
        preceding = []
        deadline = asyncio.get_running_loop().time() + 15
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            line = await asyncio.wait_for(front.stdout.readline(), remaining)
            self.assertTrue(line, "ACP front exited before its response")
            message = json.loads(line)
            if message.get("id") == request_id:
                self.assertNotIn("error", message, message)
                return message["result"], preceding
            preceding.append(message)

    async def _request(self, front, request_id, method, params):
        front.stdin.write((json.dumps(acps.request(
            request_id, method, params)) + "\n").encode())
        await asyncio.wait_for(front.stdin.drain(), 5)
        return await self._response(front, request_id)

    async def _finish(self, front, diagnostics, session_id):
        result, _ = await self._request(
            front, 90, "session/close", {"sessionId": session_id})
        self.assertEqual(result, {})
        # The original owner must no longer route requests to the closed worker.
        front.stdin.write((json.dumps(acps.request(
            91, "session/prompt", {"sessionId": session_id,
                                   "prompt": [{"type": "text", "text": "forbidden"}]}))
                           + "\n").encode())
        await asyncio.wait_for(front.stdin.drain(), 5)
        async with asyncio.timeout(5):
            while True:
                line = await front.stdout.readline()
                self.assertTrue(line, "front exited before rejecting closed session")
                message = json.loads(line)
                if message.get("id") == 91:
                    self.assertIn("unknown session", message["error"]["message"])
                    break
        front.stdin.close()
        await asyncio.wait_for(front.wait(), 5)
        stderr = await asyncio.wait_for(diagnostics, 5)
        self.assertEqual(front.returncode, 0, stderr.decode(errors="replace"))
        self.assertEqual(await asyncio.wait_for(front.stdout.read(), 5), b"")


class FrontWorkerTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
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

    async def test_ini_logging_reaches_the_front_not_the_contained_worker(self):
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
            env = self._front_env(directory)
            workspace = _configured_workspace(directory)
            env["LOKI_LOG_CONFIG"] = os.path.relpath(config, workspace)
            front, diagnostics = await self._front(env, workspace)
            await self._request(front, 1, "initialize", {"protocolVersion": 1})
            opened, _ = await self._request(front, 2, "session/new", {"cwd": workspace})
            session_id = opened["sessionId"]
            reply, updates = await self._request(front, 3, "session/prompt", {
                "sessionId": session_id, "prompt": [{"type": "text", "text": "logging probe"}],
            })
            self.assertEqual(reply, {"stopReason": "end_turn"})
            self.assertEqual(_tool_updates(updates, session_id), _assistant_chunks("acp reply text"))
            await self._finish(front, diagnostics, session_id)
            # The front is uncontained and loads the INI.  The separately
            # execed worker is contained: it refuses any configuration the
            # invoker names (it cannot be assumed able to read it or write its
            # handlers' targets) and logs to stderr instead, so only the front
            # opens a trace file.
            traces = [name for name in os.listdir(directory)
                      if name.startswith("trace-")]
            self.assertEqual(len(traces), 1)

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
            returncode = None

        class FakeChannel:
            def __init__(
                    self, session_id, process, forward,
                    credential_delegation):
                self.session_id = session_id
                self.credential_delegation = credential_delegation
                self.process = process
                self._closed = False
                self._reader_task = asyncio.get_running_loop().create_future()

            async def request(self, method, params):
                return {}

            async def close(self):
                await self.credential_delegation.close()
                self._closed = True
                if not self._reader_task.done():
                    self._reader_task.set_result(None)

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
                await front.close_session({'sessionId': session_id})

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


def _conversation_pairs(blob):
    """Inspect canonical durable messages, without a production replay oracle."""
    items = []
    for event in blob["events"]:
        if event["type"] == "model_response":
            items.extend(event["items"])
        else:
            items.append(event)
    return [(item["role"], item["content"]) for item in items
            if item["type"] == "message"
            and item["role"] in ("user", "assistant")]


def _expected_pairs(*turns):
    return [
        (role, [{"type": "text", "text": text}])
        for prompt, answer in turns
        for role, text in (("user", prompt), ("assistant", answer))
    ]


def _historical_chunks(messages, session_id):
    chunks = []
    for message in messages:
        if message.get("method") != "session/update":
            continue
        params = message["params"]
        update = params["update"]
        kind = update["sessionUpdate"]
        if kind in ("user_message_chunk", "agent_message_chunk"):
            if params["sessionId"] != session_id:
                raise AssertionError("replay belongs to a different session")
            chunks.append((kind, update["content"]))
    return chunks


class SessionRestoreTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    async def test_saved_session_list_load_resume_durable_journey(self):
        import datetime
        from loki_agent import formats, loki

        with tempfile.TemporaryDirectory() as root:
            workspace = _configured_workspace(root)
            os.mkdir(workspace)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("LOKI_")
                   and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
            env.update({
                "HOME": root, "XDG_CONFIG_HOME": os.path.join(root, "config"),
                "XDG_STATE_HOME": os.path.join(root, "state"), "TERM": "dumb",
                "LOKI_PROVIDER": "dummy", "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_MODEL": "dummy-model", "LOKI_DUMMY_REPLY": "first answer",
            })
            configure_container(env, workspace)
            front, diagnostics = await self._front(env, workspace)
            await self._request(front, 1, "initialize", {"protocolVersion": 1})
            opened, _ = await self._request(front, 2, "session/new", {"cwd": workspace})
            session_id = opened["sessionId"]
            result, _ = await self._request(front, 3, "session/prompt", {
                "sessionId": session_id, "prompt": [{"type": "text", "text": "first prompt"}]})
            self.assertEqual(result["stopReason"], "end_turn")
            await self._finish(front, diagnostics, session_id)
            path = os.path.join(loki.chat_log_dir_for(workspace), f"chat-{session_id}.json")

            def read_saved(turns):
                with open(path, encoding="utf-8") as stream:
                    blob = json.load(stream)
                formats.validate_events(blob["events"])
                self.assertEqual(_conversation_pairs(blob), _expected_pairs(*turns))
                self.assertEqual(blob["session_state"]["shell_cwd"], workspace)
                return blob

            turns = [("first prompt", "first answer")]
            read_saved(turns)
            for method, extension, prompt, answer in (
                    ("session/load", False, "loaded prompt", "loaded answer"),
                    ("session/resume", True, "resumed prompt", "resumed answer")):
                env["LOKI_DUMMY_REPLY"] = answer
                front, diagnostics = await self._front(env, workspace)
                initialized, _ = await self._request(front, 1, "initialize", {
                    "protocolVersion": 1,
                    "clientInfo": {
                        "name": "agent-shell", "title": "Emacs Agent Shell",
                        "version": "test",
                    },
                })
                self.assertEqual(initialized["agentCapabilities"]["sessionCapabilities"]["resume"], {})
                for params in ({}, {"cwd": workspace}):
                    listed, _ = await self._request(front, 2, "session/list", params)
                    self.assertEqual(len(listed["sessions"]), 1)
                    entry = listed["sessions"][0]
                    self.assertEqual(entry["sessionId"], session_id)
                    self.assertEqual(entry["cwd"], workspace)
                    self.assertEqual(datetime.datetime.fromisoformat(entry["updatedAt"]),
                                     datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc))
                restored, replay = await self._request(front, 3, method, {
                    "sessionId": session_id, "cwd": workspace, "mcpServers": [], "replay": extension})
                self.assertNotIn("sessionId", restored)
                self.assertIn("configOptions", restored)
                expected_replay = [
                    (kind, {"type": "text", "text": text})
                    for user, assistant in turns
                    for kind, text in (("user_message_chunk", user), ("agent_message_chunk", assistant))
                ] if method == "session/load" else []
                self.assertEqual(_historical_chunks(replay, session_id), expected_replay)
                if method == "session/resume":
                    self.assertFalse(any(m.get("method") == "session/update" for m in replay), replay)
                result, updates = await self._request(front, 4, "session/prompt", {
                    "sessionId": session_id, "prompt": [{"type": "text", "text": prompt}]})
                self.assertEqual(result["stopReason"], "end_turn")
                self.assertEqual(_historical_chunks(updates, session_id), [
                    ("agent_message_chunk", {"type": "text", "text": answer})])
                await self._finish(front, diagnostics, session_id)
                turns.append((prompt, answer))
                read_saved(turns)


class SavedApprovalStdioTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    async def test_provisional_approval_rejection_close_and_eof_over_real_pipes(self):
        from loki_agent import loki, protocols
        from loki_agent.connections import ConnectionDescriptor

        with tempfile.TemporaryDirectory() as root:
            workspace = _configured_workspace(root)
            os.mkdir(workspace)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith('LOKI_')
                   and not key.endswith(('_KEY', '_TOKEN', '_PAT'))}
            env.update(HOME=root, XDG_CONFIG_HOME=os.path.join(root, 'config'),
                       XDG_STATE_HOME=os.path.join(root, 'state'), TERM='dumb',
                       LOKI_PROVIDER='dummy', LOKI_API_BASE='http://dummy.invalid/v1',
                       LOKI_MODEL='dummy-model')
            configure_container(env, workspace)
            front, diagnostics = await self._front(env, workspace)
            await self._request(front, 1, 'initialize', {'protocolVersion': 1})
            opened, _ = await self._request(front, 2, 'session/new', {'cwd': workspace})
            session_id = opened['sessionId']
            await self._request(front, 3, 'session/prompt', {
                'sessionId': session_id,
                'prompt': [{'type': 'text', 'text': 'seed conversation'}]})
            await self._finish(front, diagnostics, session_id)
            path = os.path.join(loki.chat_log_dir_for(workspace), f'chat-{session_id}.json')
            with open(path, encoding='utf-8') as stream:
                saved = json.load(stream)
            saved['session_state']['connection'] = ConnectionDescriptor(
                provider_id=None, provider_name='Saved connection',
                model='untrusted-saved-model', protocol=protocols.OPENAI_CHAT,
                chat_url='https://saved.invalid/v1/chat/completions',
                models_url='https://saved.invalid/v1/models').to_dict()
            with open(path, 'w', encoding='utf-8') as stream:
                json.dump(saved, stream)
            with open(path, 'rb') as stream:
                original = stream.read()
            for name in ('LOKI_PROVIDER', 'LOKI_API_BASE', 'LOKI_MODEL'):
                del env[name]

            for decision in ('accept', 'decline', 'close then accept', 'eof'):
                with self.subTest(decision=decision):
                    front, diagnostics = await self._front(env, workspace)
                    await self._request(front, 1, 'initialize', {
                        'protocolVersion': 1,
                        'clientCapabilities': {'elicitation': {'form': {}}}})
                    messages = []

                    async def receive(predicate):
                        async with asyncio.timeout(15):
                            while True:
                                raw = await front.stdout.readline()
                                self.assertTrue(raw, 'front exited before expected response')
                                message = json.loads(raw)
                                messages.append(message)
                                if predicate(message):
                                    return message

                    async def send(*messages):
                        front.stdin.write(b''.join(
                            (json.dumps(message) + '\n').encode() for message in messages))
                        await asyncio.wait_for(front.stdin.drain(), 5)

                    await send(acps.request(3, 'session/resume', {
                        'sessionId': session_id, 'cwd': workspace}))
                    ask = await receive(lambda m: m.get('method') == 'elicitation/create')
                    self.assertEqual(ask['params']['requestId'], 3)
                    await send(acps.request(4, 'session/prompt', {
                        'sessionId': session_id,
                        'prompt': [{'type': 'text', 'text': 'must not execute'}]}))
                    rejected = await receive(lambda m: m.get('id') == 4)
                    self.assertIn('request was not executed', rejected['error']['message'])
                    answer = acps.response(ask['id'], result={
                        'action': 'accept', 'content': {'authorize': True}})
                    if decision == 'accept':
                        await send(answer)
                        restored = await receive(lambda m: m.get('id') == 3)
                        self.assertIn('configOptions', restored['result'])
                        await self._finish(front, diagnostics, session_id)
                    else:
                        if decision == 'decline':
                            await send(acps.response(ask['id'], result={'action': 'decline'}))
                            self.assertIn('error', await receive(lambda m: m.get('id') == 3))
                        elif decision == 'close then accept':
                            await send(acps.request(5, 'session/close', {'sessionId': session_id}), answer)
                            await receive(lambda m: m.get('id') == 5)
                            self.assertEqual(next(m for m in messages if m.get('id') == 5)['result'], {})
                            self.assertIn('error', next(m for m in messages if m.get('id') == 3))
                        front.stdin.close()
                        await asyncio.wait_for(front.wait(), 5)
                        stderr = await asyncio.wait_for(diagnostics, 5)
                        self.assertEqual(front.returncode, 0, stderr.decode(errors='replace'))
                        remaining = await asyncio.wait_for(front.stdout.read(), 5)
                        if decision == 'eof':
                            self.assertNotIn(b'"result"', remaining)
                    with open(path, 'rb') as stream:
                        self.assertEqual(stream.read(), original)
                    self.assertFalse(any(m.get('method') == 'session/update' for m in messages))


def _tool_fixture(root):
    workspace = os.path.join(root, "workspace")
    os.mkdir(workspace)
    payload = b"ACP-W3 unique file output"
    with open(os.path.join(workspace, "payload.txt"), "wb") as stream:
        stream.write(payload)
    # A wrong cwd cannot accidentally return the same bytes.
    with open(os.path.join(root, "payload.txt"), "wb") as stream:
        stream.write(b"wrong directory")
    read = "type" if os.name == "nt" else "cat"
    command = f"{read} payload.txt >> executed.txt && {read} executed.txt"
    result = "status: completed\nexit_code: 0\n[stdout]\n" + payload.decode()
    return workspace, payload, command, result


def _tool_updates(messages, session_id):
    updates = []
    for message in messages:
        if message.get("method") != "session/update":
            continue
        params = message["params"]
        if params["sessionId"] != session_id:
            raise AssertionError("tool update belongs to another session")
        update = params["update"]
        if update["sessionUpdate"] in (
                "agent_message_chunk", "user_message_chunk",
                "tool_call", "tool_call_update"):
            updates.append(update)
    return updates


def _call_and_result(call_id, command, workspace, result):
    # Independent ACP wire oracle, including successful-status omission.
    return [
        {"sessionUpdate": "tool_call", "toolCallId": call_id,
         "title": f"Bash: {command} (cwd: {workspace})",
         "kind": "execute", "status": "in_progress"},
        {"sessionUpdate": "tool_call_update", "toolCallId": call_id,
         "content": [{"type": "content", "content": {
             "type": "text", "text": result}}]},
    ]


def _assistant_chunks(*texts):
    return [{"sessionUpdate": "agent_message_chunk",
             "content": {"type": "text", "text": text}} for text in texts]


class ToolStreamingJourneyTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    def _saved_tool_turn(self, path, session_id, user, call_id, args, result, answers):
        from loki_agent import formats

        with open(path, encoding="utf-8") as stream:
            blob = json.load(stream)
        formats.validate_events(blob["events"])
        events = blob["events"]
        user_index = next(i for i, event in enumerate(events)
                          if event.get("role") == "user")
        turn = events[user_index:]
        self.assertEqual([event["type"] for event in turn],
                         ["message", "model_response", "tool_result", "model_response"])
        self.assertEqual(_conversation_pairs(blob), [
            ("user", [{"type": "text", "text": user}]),
            *[("assistant", [{"type": "text", "text": answer}]) for answer in answers],
        ])
        calls = [item for event in turn if event["type"] == "model_response"
                 for item in event["items"] if item["type"] == "function_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["call_id"], call_id)
        self.assertEqual(calls[0]["name"], "Bash")
        self.assertEqual(json.loads(calls[0]["arguments"]), args)
        tool_result = turn[2]
        self.assertEqual(tool_result["call_id"], call_id)
        self.assertEqual(tool_result["name"], "Bash")
        self.assertIs(tool_result["is_error"], False)
        self.assertEqual(tool_result["content"], [{"type": "text", "text": result}])
        self.assertEqual(os.path.basename(path), f"chat-{session_id}.json")
        return blob

    async def test_shipped_front_real_tool_updates_save_and_close(self):
        from loki_agent import loki

        with tempfile.TemporaryDirectory() as root:
            workspace, payload, command, result = _tool_fixture(root)
            args = {"command": command, "description": "Read the relative fixture", "timeout": 5000}
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("LOKI_")
                           and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
            environment.update({
                "HOME": root, "XDG_CONFIG_HOME": os.path.join(root, "config"),
                "XDG_STATE_HOME": os.path.join(root, "state"), "TERM": "dumb",
                "LOKI_PROVIDER": "dummy", "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_MODEL": "dummy-model", "LOKI_DUMMY_REPLY": "Tool complete.",
                "LOKI_DUMMY_TOOL_CALL": json.dumps({"name": "Bash", "arguments": args}),
            })
            configure_container(environment, workspace)
            front, diagnostics = await self._front(environment, workspace)
            await self._request(front, 1, "initialize", {"protocolVersion": 1})
            opened, _ = await self._request(front, 2, "session/new", {"cwd": workspace})
            session_id = opened["sessionId"]
            reply, messages = await self._request(front, 3, "session/prompt", {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": "Run the real file proof"}],
            })
            self.assertEqual(reply, {"stopReason": "end_turn"})
            self.assertEqual(_tool_updates(messages, session_id),
                             _call_and_result("dummy-tool-call", command, workspace, result)
                             + _assistant_chunks("Tool complete."))
            # The append-only witness distinguishes skipped/fabricated dispatch
            # and duplicate execution; it is read before shutdown/backstops.
            with open(os.path.join(workspace, "executed.txt"), "rb") as stream:
                self.assertEqual(stream.read(), payload)
            await self._finish(front, diagnostics, session_id)
            path = os.path.join(loki.chat_log_dir_for(workspace), f"chat-{session_id}.json")
            self._saved_tool_turn(path, session_id, "Run the real file proof",
                                  "dummy-tool-call", args, result, ["Tool complete."])

    async def test_resource_real_tool_continuation_save_and_fresh_replay(self):
        import copy
        from contextlib import ExitStack
        from pathlib import Path
        from loki_agent import formats, loki, protocols
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            workspace, payload, command, result = _tool_fixture(root)
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("LOKI_")
                           and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
            environment.update({"HOME": root, "XDG_CONFIG_HOME": os.path.join(root, "config"),
                                "XDG_STATE_HOME": os.path.join(root, "state")})
            stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
            credentials = CredentialStore({
                "LOKI_PROVIDER": "openai", "LOKI_API_BASE": "https://provider.invalid/v1",
                "LOKI_MODEL": "tool-model", "LOKI_STREAM": "1",
            })
            stack.enter_context(mock.patch.object(loki, "CREDENTIALS", credentials))
            session = Session(shell_cwd=workspace)
            stack.enter_context(mock.patch.object(loki, "_DEFAULT_SESSION", session))
            stack.enter_context(mock.patch.object(loki, "LOKI_JOB_STATE_DIR", os.path.join(root, "jobs")))
            stack.enter_context(mock.patch.object(models, "ensure_index", new=mock.AsyncMock(return_value=({}, {}))))
            loki.apply_runtime_config(loki.build_config_from_env(credentials=credentials))
            messages = []
            session_id = "tool-workflow"
            worker = Worker(session, messages.append, session_id)
            workers = [worker]
            args = {"command": command, "description": "Read the relative fixture", "timeout": 5000}
            call_id = "provider-call-73"
            resource = {"name": "payload.txt", "uri": Path(workspace, "payload.txt").as_uri(),
                        "title": "Source \u03b1", "description": "A uniquely populated local file",
                        "mimeType": "text/plain", "size": len(payload)}
            user = "inspect this\n[ACP resource link]\n" + json.dumps(resource, ensure_ascii=False)
            requests = []

            async def completion(items, _tools, _verbose, _timing, **kwargs):
                self.assertFalse(kwargs["cancel_check"]())
                requests.append(copy.deepcopy(items))
                self.assertLessEqual(len(requests), 2, "unexpected additional provider request")
                if len(requests) == 1:
                    for chunk in ("Inspecting ", "resource."):
                        kwargs["on_text_delta"](chunk)
                    return formats.DecodedTurn([
                        formats.message_item("assistant", "Inspecting resource."),
                        formats.tool_call_item(call_id, "Bash", args),
                    ], {"protocol": protocols.OPENAI_CHAT})
                for chunk in ("Tool ", "complete."):
                    kwargs["on_text_delta"](chunk)
                return formats.DecodedTurn([
                    formats.message_item("assistant", "Tool complete."),
                ], {"protocol": protocols.OPENAI_CHAT})

            stack.enter_context(mock.patch.object(loki, "async_chat_completion", new=completion))
            try:
                await asyncio.wait_for(worker.prepare_open({
                    "sessionId": session_id, "cwd": workspace, "openMethod": "session/new"}), 3)
                worker.commit_open()
                messages.clear()
                reply = await asyncio.wait_for(worker.prompt({
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "inspect this"},
                               {"type": "resource_link", **resource, "unadvertised": "must not leak"}],
                }), 10)
                self.assertEqual(reply, {"stopReason": "end_turn"})
                self.assertEqual(len(requests), 2)
                for request in requests:
                    self.assertEqual([item["content"] for item in request
                                      if item.get("role") == "user"],
                                     [[{"type": "text", "text": user}]])
                followup = requests[1]
                self.assertEqual([item["type"] for item in followup[-3:]],
                                 ["message", "model_response", "tool_result"])
                self.assertEqual(followup[-2]["items"][0], {
                    "type": "message", "role": "assistant",
                    "content": [{"type": "text", "text": "Inspecting resource."}],
                })
                continued_call = followup[-2]["items"][1]
                self.assertEqual(continued_call["call_id"], call_id)
                self.assertEqual(continued_call["name"], "Bash")
                self.assertEqual(json.loads(continued_call["arguments"]), args)
                self.assertEqual(len(followup[-2]["items"]), 2)
                self.assertEqual(followup[-1]["call_id"], call_id)
                self.assertEqual(followup[-1]["content"], [{"type": "text", "text": result}])
                self.assertIs(followup[-1]["is_error"], False)
                self.assertEqual(_tool_updates(messages, session_id),
                                 _assistant_chunks("Inspecting ", "resource.")
                                 + _call_and_result(call_id, command, workspace, result)
                                 + _assistant_chunks("Tool ", "complete."))
                with open(os.path.join(workspace, "executed.txt"), "rb") as stream:
                    self.assertEqual(stream.read(), payload)
                self.assertIsNotNone(session.job_manager)
                self.assertEqual(len(session.job_manager.jobs), 1)
                job = next(iter(session.job_manager.jobs.values()))
                self.assertEqual(job.cwd, workspace)
                self.assertEqual(job.status, "exited")
                self.assertEqual(job.exit_code, 0)
                self.assertEqual(job.process.returncode, 0)
                with open(job.stdout_path, encoding="utf-8") as stream:
                    self.assertEqual(stream.read(), payload.decode())
                with open(job.stderr_path, encoding="utf-8") as stream:
                    self.assertEqual(stream.read(), "")
                path = session.chat_log_path
                saved = self._saved_tool_turn(path, session_id, user, call_id, args, result,
                                              ["Inspecting resource.", "Tool complete."])
                await asyncio.wait_for(worker.close(), 3)
                restored_session = Session(shell_cwd=workspace)
                loki._DEFAULT_SESSION = restored_session
                restored = Worker(restored_session, messages.append, session_id)
                workers.append(restored)
                messages.clear()
                await asyncio.wait_for(restored.prepare_open({
                    "sessionId": session_id, "cwd": workspace, "openMethod": "session/load",
                    "replay": False}), 3)
                self.assertEqual(restored_session.transcript_items, saved["events"])
                self.assertEqual(_tool_updates(messages, session_id), [])
                restored.commit_open()
                self.assertEqual(_tool_updates(messages, session_id), [
                    {"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": user}},
                    *_assistant_chunks("Inspecting resource."),
                    {"sessionUpdate": "tool_call", "toolCallId": call_id, "title": "Bash",
                     "kind": "other", "status": "completed"},
                    {"sessionUpdate": "tool_call", "toolCallId": call_id, "title": "Tool result: Bash",
                     "kind": "other", "status": "completed"},
                    *_assistant_chunks("Tool complete."),
                ])
                self.assertEqual(len(requests), 2, "load must not perform inference")
                with open(os.path.join(workspace, "executed.txt"), "rb") as stream:
                    self.assertEqual(stream.read(), payload, "load must not execute historical calls")
                await asyncio.wait_for(restored.close(), 3)
            finally:
                # Backstop only: owned job exit/output was asserted above.
                for owner in workers:
                    await asyncio.wait_for(owner.close(), 3)


def _command_fixture(root):
    import base64
    workspace = os.path.join(root, "workspace")
    destination = os.path.join(workspace, "changed cwd")
    os.makedirs(destination)
    payload = b"ACP-W1 destination bytes"
    for directory, content in ((workspace, b"wrong initial cwd"), (destination, payload)):
        with open(os.path.join(directory, "payload.txt"), "wb") as stream:
            stream.write(content)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
    with open(os.path.join(destination, "shot.png"), "wb") as stream:
        stream.write(png)
    read = "type" if os.name == "nt" else "cat"
    command = f"{read} payload.txt >> executed.txt && {read} executed.txt"
    output = "status: completed\nexit_code: 0\n[stdout]\n" + payload.decode()
    model_text = f"I ran the local command `{command}`.\nOutput:\n```\n{output}\n```"
    image = {"type": "image", "source": {
        "type": "base64", "media_type": "image/png",
        "data": base64.b64encode(png).decode("ascii"),
    }}
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("LOKI_")
                   and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
    environment.update({
        "HOME": root, "XDG_CONFIG_HOME": os.path.join(root, "config"),
        "XDG_STATE_HOME": os.path.join(root, "state"), "TERM": "dumb",
        "LOKI_PROVIDER": "dummy", "LOKI_API_BASE": "http://dummy.invalid/v1",
        "LOKI_MODEL": "dummy-model", "LOKI_DUMMY_REPLY": "Local command answer.",
    })
    return workspace, os.path.realpath(destination), payload, png, command, output, model_text, image, environment


async def _command_status(environment):
    from loki_agent import paths, response_headers
    path = os.path.join(paths.loki_state_dir(environment), "response-headers.json")
    store = response_headers.Store(path)
    store.observer("https://unrelated.example/v1/chat/completions", None, "observed-model")(
        200, {"x-remaining": "7"})
    await store.save()
    with open(path, "rb") as stream:
        original = stream.read()
    return store, path, original


class LocalCommandJourneyTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    def _advertisement(self, commands):
        names = [command["name"] for command in commands]
        self.assertEqual(set(names), {"status", "account", "pwd", "cd", "image", "thinking", "trace"})
        self.assertEqual(len(names), 7)
        self.assertNotIn("ps", names)
        self.assertTrue(all(command["description"] for command in commands))
        self.assertNotIn("model", names)
        self.assertNotIn("effort", names)

    def _status(self, messages, session_id, original):
        updates = _tool_updates(messages, session_id)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["sessionUpdate"], "agent_message_chunk")
        document = json.loads(updates[0]["content"]["text"])
        self.assertEqual(document, json.loads(original))
        self.assertEqual(document["version"], 1)
        self.assertEqual(len(document["endpoints"]), 1)
        entry = document["endpoints"][0]
        self.assertEqual(entry["endpoint"], "https://unrelated.example/v1/chat/completions")
        self.assertEqual(entry["headers"]["x-remaining"]["value"], "7")
        self.assertEqual(entry["latest"]["model"], "observed-model")
        self.assertEqual(entry["latest"]["status"], 200)

    def _saved(self, path, destination, model_text, image, answers):
        from loki_agent import formats
        with open(path, encoding="utf-8") as stream:
            blob = json.load(stream)
        formats.validate_events(blob["events"])
        self.assertEqual(_conversation_pairs(blob), [
            ("user", [{"type": "text", "text": model_text}, image]),
            ("assistant", [{"type": "text", "text": answers[0]}]),
            ("user", [{"type": "text", "text": "Second prompt"}]),
            ("assistant", [{"type": "text", "text": answers[1]}]),
        ])
        start = next(i for i, event in enumerate(blob["events"]) if event.get("role") == "user")
        self.assertEqual([event["type"] for event in blob["events"][start:]],
                         ["message", "model_response", "message", "model_response"])
        self.assertEqual(blob["session_state"]["shell_cwd"], destination)

    async def test_shipped_commands_snapshot_bang_two_turns_and_close(self):
        import socket
        from loki_agent import loki
        with tempfile.TemporaryDirectory() as root:
            workspace, destination, payload, png, command, output, model_text, image, env = _command_fixture(root)
            _store, status_path, status_bytes = await _command_status(env)
            configure_container(env, workspace)
            # The new virtual cwd is also a supported configured workspace on
            # Windows; no alternative launcher or containment route is invented.
            configure_container(env, destination)
            front, diagnostics = await self._front(env, workspace)
            initialized, _ = await self._request(front, 1, "initialize", {"protocolVersion": 1})
            self.assertEqual(initialized["protocolVersion"], 1)
            self.assertEqual(initialized["agentInfo"]["name"], "loki")
            self.assertEqual(initialized["agentInfo"]["version"], __version__)
            for capability in ("close", "list", "resume"):
                self.assertEqual(initialized["agentCapabilities"]["sessionCapabilities"][capability], {})
            self.assertIs(initialized["agentCapabilities"]["promptCapabilities"]["image"], False)
            opened, preceding = await self._request(front, 2, "session/new", {"cwd": workspace})
            session_id = opened["sessionId"]
            self.assertTrue(session_id)
            self.assertFalse(any(message.get("method") == "session/update" for message in preceding))
            advertised = json.loads(await asyncio.wait_for(front.stdout.readline(), 5))
            self.assertEqual(advertised["method"], "session/update")
            self.assertEqual(advertised["params"]["sessionId"], session_id)
            self.assertEqual(advertised["params"]["update"]["sessionUpdate"], "available_commands_update")
            self._advertisement(advertised["params"]["update"]["availableCommands"])
            path = os.path.join(loki.chat_log_dir_for(workspace), f"chat-{session_id}.json")

            async def prompt(request_id, text):
                reply, messages = await self._request(front, request_id, "session/prompt", {
                    "sessionId": session_id, "prompt": [{"type": "text", "text": text}],
                })
                self.assertEqual(reply, {"stopReason": "end_turn"})
                return messages

            for request_id, text, expected in (
                    (3, "/pwd", f"cwd: {workspace}"),
                    (4, f"/cd {destination}", f"cwd: {destination}"),
                    (5, "/pwd", f"cwd: {destination}"),
                    (6, "/image shot.png", f"Attached image for next prompt: {os.path.join(destination, 'shot.png')} (image/png, {len(png)} bytes)")):
                messages = await prompt(request_id, text)
                self.assertEqual(_tool_updates(messages, session_id), _assistant_chunks(expected))
            self._status(await prompt(7, "/status all --json"), session_id, status_bytes)
            self.assertFalse(os.path.exists(path), "local commands must not save a model turn")
            with open(status_path, "rb") as stream:
                self.assertEqual(stream.read(), status_bytes)
            with open(os.path.join(destination, "shot.png"), "wb") as stream:
                stream.write(b"\xff\xd8\xffreplacement source")
            messages = await prompt(8, "!" + command)
            acknowledgement = f"{socket.gethostname()}: [Running local command: {command}]\n{output}"
            self.assertEqual(_tool_updates(messages, session_id),
                             _assistant_chunks(acknowledgement, "Local command answer."))
            self.assertEqual(_tool_updates(await prompt(9, "Second prompt"), session_id),
                             _assistant_chunks("Local command answer."))
            with open(os.path.join(destination, "executed.txt"), "rb") as stream:
                self.assertEqual(stream.read(), payload)
            self.assertFalse(os.path.exists(os.path.join(workspace, "executed.txt")))
            await self._finish(front, diagnostics, session_id)
            self._saved(path, destination, model_text, image,
                        ("Local command answer.", "Local command answer."))

    async def test_wire_commands_snapshot_bang_and_durable_two_turns(self):
        import copy
        import socket
        from contextlib import ExitStack
        from loki_agent import formats, loki, paths, protocols
        from loki_agent.sessions import Session
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            workspace, destination, payload, png, command, output, model_text, image, env = _command_fixture(root)
            store, status_path, status_bytes = await _command_status(env)
            stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
            stack.enter_context(mock.patch.object(loki, "CREDENTIALS", CredentialStore(env)))
            stack.enter_context(mock.patch.object(loki, "LOKI_CONFIG_DIR", paths.loki_config_dir(env)))
            stack.enter_context(mock.patch.object(loki, "LOKI_JOB_STATE_DIR", os.path.join(root, "jobs")))
            # Start at the wrong cwd. Only the actual prepare wire request may
            # install the conversation's initial workspace.
            session = Session(shell_cwd=root, response_headers=store)
            stack.enter_context(mock.patch.object(loki, "_DEFAULT_SESSION", session))
            stack.enter_context(mock.patch.object(models, "ensure_index", new=mock.AsyncMock(return_value=({}, {}))))
            stack.enter_context(mock.patch.object(acp.runtime_isolation, "close_runtime_process", new=lambda process: None))
            loki.apply_runtime_config(loki.build_config_from_env(credentials=loki.CREDENTIALS))
            messages = []
            requests = []
            session_id = "local-commands"
            process = _LocalWorkerProcess(session, session_id, lambda message: None)
            channel = acp.WorkerChannel(session_id, process, messages.append)
            process_cwd = os.getcwd()

            async def completion(items, _tools, _verbose, _timing, **kwargs):
                self.assertFalse(kwargs["cancel_check"]())
                requests.append(copy.deepcopy(items))
                self.assertLessEqual(len(requests), 2)
                answer = "Image accepted." if len(requests) == 1 else "Second answer."
                return formats.DecodedTurn(
                    [formats.message_item("assistant", answer)],
                    {"protocol": protocols.OPENAI_CHAT})

            stack.enter_context(mock.patch.object(loki, "async_chat_completion", new=completion))

            async def prompt(text):
                messages.clear()
                reply = await asyncio.wait_for(channel.request("session/prompt", {
                    "sessionId": session_id, "prompt": [{"type": "text", "text": text}],
                }), 10)
                self.assertEqual(reply, {"stopReason": "end_turn"})
                return list(messages)

            try:
                await asyncio.wait_for(channel.request("session/prepare_open", {
                    "sessionId": session_id, "cwd": workspace, "openMethod": "session/new"}), 3)
                opened = await asyncio.wait_for(channel.request("session/commit_open", {}), 3)
                self.assertEqual(session.shell_cwd, workspace)
                self._advertisement(opened["lokiCommands"])
                initial = copy.deepcopy(session.transcript_items)
                for text, expected in (
                        ("/pwd", f"cwd: {workspace}"),
                        (f"/cd {destination}", f"cwd: {destination}"),
                        ("/pwd", f"cwd: {destination}"),
                        ("/image shot.png", f"Attached image for next prompt: {os.path.join(destination, 'shot.png')} (image/png, {len(png)} bytes)")):
                    self.assertEqual(_tool_updates(await prompt(text), session_id), _assistant_chunks(expected))
                    self.assertEqual(requests, [])
                    if text.startswith("/cd "):
                        initial.append({
                            "type": "message", "role": "system", "content": [{
                                "type": "text",
                                "text": f"Current Loki cwd changed to: {destination}. Relative tool paths and Bash commands now run from this directory.",
                            }],
                        })
                    self.assertEqual(session.transcript_items, initial)
                self._status(await prompt("/status all --json"), session_id, status_bytes)
                self.assertEqual(requests, [])
                self.assertEqual(session.transcript_items, initial)
                with open(status_path, "rb") as stream:
                    self.assertEqual(stream.read(), status_bytes)
                with open(os.path.join(destination, "shot.png"), "wb") as stream:
                    stream.write(b"\xff\xd8\xffreplacement source")
                acknowledgement = f"{socket.gethostname()}: [Running local command: {command}]\n{output}"
                self.assertEqual(_tool_updates(await prompt("!" + command), session_id),
                                 _assistant_chunks(acknowledgement, "Image accepted."))
                self.assertEqual([item["content"] for item in requests[0] if item.get("role") == "user"],
                                 [[{"type": "text", "text": model_text}, image]])
                self.assertEqual(_tool_updates(await prompt("Second prompt"), session_id),
                                 _assistant_chunks("Second answer."))
                self.assertEqual(len(requests), 2)
                self.assertEqual([item["content"] for item in requests[1] if item.get("role") == "user"],
                                 [[{"type": "text", "text": model_text}, image],
                                  [{"type": "text", "text": "Second prompt"}]])
                self.assertEqual(os.getcwd(), process_cwd)
                self.assertEqual(session.shell_cwd, destination)
                with open(os.path.join(destination, "executed.txt"), "rb") as stream:
                    self.assertEqual(stream.read(), payload)
                self.assertFalse(os.path.exists(os.path.join(workspace, "executed.txt")))
                self.assertEqual(len(session.job_manager.jobs), 1)
                job = next(iter(session.job_manager.jobs.values()))
                self.assertEqual(job.cwd, destination)
                self.assertEqual(job.status, "exited")
                self.assertEqual(job.exit_code, 0)
                self.assertEqual(job.process.returncode, 0)
                for path, expected in ((job.stdout_path, payload.decode()), (job.stderr_path, "")):
                    with open(path, encoding="utf-8") as stream:
                        self.assertEqual(stream.read(), expected)
                self._saved(session.chat_log_path, destination, model_text, image,
                            ("Image accepted.", "Second answer."))
                await asyncio.wait_for(channel.close(), 3)
                self.assertEqual(process.returncode, 0)
                self.assertTrue(process.close_task.done())
                self.assertIsNone(process.close_task.exception())
                self.assertTrue(channel._reader_task.done())
                self.assertFalse(channel._pending)
                self._saved(session.chat_log_path, destination, model_text, image,
                            ("Image accepted.", "Second answer."))
            finally:
                process.close()
                await asyncio.wait_for(process.wait(), 3)
                await asyncio.wait_for(channel.close(), 3)


class _LocalWorkerProcess:
    """Byte-transport seam only: real Worker and WorkerChannel, no OS child.

    Worker.handle generates every reply and update. EOF drives Worker.close;
    wait observes that operation, rather than supplying cleanup in assertions.
    """
    def __init__(self, session, session_id, before_request):
        from loki_agent.acp_worker import Worker
        self.before_request = before_request
        self.stdout = asyncio.StreamReader()
        self.stdin = self
        self.returncode = None
        self.buffer = bytearray()
        self.finished = asyncio.Event()
        self.close_task = None
        self.methods = []
        self.worker = Worker(session, lambda message: self.stdout.feed_data(
            (json.dumps(message) + "\n").encode()), session_id)

    def write(self, data):
        self.buffer.extend(data)

    async def drain(self):
        while b"\n" in self.buffer:
            line, _, rest = self.buffer.partition(b"\n")
            self.buffer = bytearray(rest)
            message = json.loads(line)
            self.methods.append(message["method"])
            self.before_request(message)
            # Match the worker entrypoint: prompt replies may be long-running,
            # while ordered config/cancel messages continue to be handled.
            await self.worker.handle(message, concurrent=True)

    def close(self):
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._finish())

    async def _finish(self):
        await self.worker.close()
        self.returncode = 0
        self.stdout.feed_eof()
        self.finished.set()

    async def wait_closed(self):
        await self.wait()

    async def wait(self):
        await self.finished.wait()
        return self.returncode


class WorkerPromptOwnershipTests(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def _channel(self, root, messages, completion):
        from loki_agent import formats, loki
        from loki_agent.sessions import Session
        session = Session(
            shell_cwd=root, chat_log_path=os.path.join(root, "chat.json"),
            transcript_items=[formats.instruction_item("system")])
        tasks_before = asyncio.all_tasks()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(loki, "_DEFAULT_SESSION", session))
            stack.enter_context(mock.patch.object(loki, "CREDENTIALS", CredentialStore({})))
            stack.enter_context(mock.patch.object(loki, "current_model", return_value="model"))
            stack.enter_context(mock.patch.object(loki, "async_chat_completion", new=completion))
            stack.enter_context(mock.patch.object(loki, "LOKI_JOB_STATE_DIR", os.path.join(root, "jobs")))
            # The process seam owns byte transport, not an OS process.
            stack.enter_context(mock.patch.object(
                acp.runtime_isolation, "close_runtime_process", new=lambda process: None))
            process = _LocalWorkerProcess(session, "prompt-owner", lambda message: None)
            channel = acp.WorkerChannel("prompt-owner", process, messages.append)
            try:
                yield channel
                await asyncio.wait_for(channel.close(), 5)
                self.assertEqual(process.returncode, 0)
                self.assertTrue(process.close_task.done())
                self.assertIsNone(process.close_task.exception())
                self.assertFalse(channel._pending)
                self.assertTrue(channel._reader_task.done())
                self.assertFalse(asyncio.all_tasks() - tasks_before)
            finally:
                process.close()
                await asyncio.wait_for(process.wait(), 5)
                await asyncio.wait_for(channel.close(), 5)

    def _params(self, text):
        return {"sessionId": "prompt-owner", "prompt": [{"type": "text", "text": text}]}

    async def test_wire_cancellation_before_prompt_task_starts_and_next_turn(self):
        from loki_agent import formats, protocols
        for text in ["ordinary prompt", "!echo should-not-run > executed.txt"]:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as root:
                messages = []
                completion = mock.AsyncMock(return_value=formats.DecodedTurn(
                    [formats.message_item("assistant", "next answer")],
                    {"protocol": protocols.OPENAI_CHAT}))
                async with self._channel(root, messages, completion) as channel:
                    process = channel.process
                    worker = process.worker

                    def before_request(message):
                        if message["method"] == "session/prompt":
                            # Both byte-framed requests are admitted before the
                            # reader yields to the new prompt task.
                            process.write((json.dumps(acps.request(
                                "cancel-before-start", "session/cancel", {})) + "\n").encode())
                            process.before_request = lambda message: None

                    process.before_request = before_request
                    reply = await asyncio.wait_for(
                        channel.request("session/prompt", self._params(text)), 5)
                    self.assertEqual(reply, {"stopReason": "cancelled"})
                    self.assertEqual(completion.await_count, 0)
                    self.assertTrue(worker.cancel_event.is_set())
                    self.assertIsNone(worker.session.job_manager)
                    self.assertFalse(os.path.exists(os.path.join(root, "executed.txt")))
                    self.assertEqual(_tool_updates(messages, "prompt-owner"),
                                     _assistant_chunks("[turn cancelled by user]"))
                    with open(worker.session.chat_log_path, encoding="utf-8") as stream:
                        self.assertEqual(_conversation_pairs(json.load(stream)),
                                         [("user", [{"type": "text", "text": text}])])
                    messages.clear()
                    reply = await asyncio.wait_for(
                        channel.request("session/prompt", self._params("next prompt")), 5)
                    self.assertEqual(reply, {"stopReason": "end_turn"})
                    self.assertFalse(worker.cancel_event.is_set())
                    self.assertEqual(completion.await_count, 1)
                    self.assertEqual(_tool_updates(messages, "prompt-owner"),
                                     _assistant_chunks("next answer"))
                    with open(worker.session.chat_log_path, encoding="utf-8") as stream:
                        self.assertEqual(_conversation_pairs(json.load(stream)), [
                            ("user", [{"type": "text", "text": text}]),
                            ("user", [{"type": "text", "text": "next prompt"}]),
                            ("assistant", [{"type": "text", "text": "next answer"}]),
                        ])

    async def test_local_commands_do_not_reset_cancellation_or_enter_history(self):
        import copy
        messages = []
        completion = mock.AsyncMock(side_effect=AssertionError("local command requested inference"))
        with tempfile.TemporaryDirectory() as root:
            async with self._channel(root, messages, completion) as channel:
                worker = channel.process.worker
                initial = copy.deepcopy(worker.session.transcript_items)
                worker.cancel_event.set()
                for text in ["/status --json", "/pwd"]:
                    messages.clear()
                    reply = await asyncio.wait_for(
                        channel.request("session/prompt", self._params(text)), 5)
                    self.assertEqual(reply, {"stopReason": "end_turn"})
                    self.assertTrue(worker.cancel_event.is_set())
                    self.assertEqual(worker.session.transcript_items, initial)
                    self.assertEqual(len(_tool_updates(messages, "prompt-owner")), 1)
                self.assertEqual(await worker.prompt(self._params("/pwd")),
                                 {"stopReason": "end_turn"})
                self.assertTrue(worker.cancel_event.is_set())
                self.assertIsNone(worker._prompt_task)
                self.assertFalse(os.path.exists(worker.session.chat_log_path))
                completion.assert_not_awaited()

    async def test_ps_is_ordinary_text_without_local_control_or_fake_tool_updates(self):
        from loki_agent import formats, loki, protocols
        messages = []
        completion = mock.AsyncMock(return_value=formats.DecodedTurn(
            [formats.message_item("assistant", "ordinary text response")],
            {"protocol": protocols.OPENAI_CHAT}))
        with tempfile.TemporaryDirectory() as root:
            async with self._channel(root, messages, completion) as channel:
                worker = channel.process.worker
                with mock.patch.object(loki, "run_ps") as run_ps:
                    result = await asyncio.wait_for(channel.request(
                        "session/prompt", self._params("/ps all")), 3)
                run_ps.assert_not_called()
                self.assertEqual(result, {"stopReason": "end_turn"})
                completion.assert_awaited_once()
                self.assertEqual(_tool_updates(messages, "prompt-owner"),
                                 _assistant_chunks("ordinary text response"))
                with open(worker.session.chat_log_path, encoding="utf-8") as stream:
                    self.assertEqual(_conversation_pairs(json.load(stream)), [
                        ("user", [{"type": "text", "text": "/ps all"}]),
                        ("assistant", [{"type": "text", "text": "ordinary text response"}]),
                    ])

    async def test_all_prompt_entry_paths_preserve_active_owner_and_updates(self):
        import copy
        from loki_agent import formats, protocols
        for origin in ["wire", "inline"]:
            with self.subTest(origin=origin), tempfile.TemporaryDirectory() as root:
                entered = asyncio.Event()
                release = asyncio.Event()
                messages = []

                async def completion(*args, **kwargs):
                    entered.set()
                    await worker.cancel_event.wait()
                    await release.wait()
                    return formats.DecodedTurn(
                        [formats.message_item("assistant", "must not appear")],
                        {"protocol": protocols.OPENAI_CHAT})

                async with self._channel(root, messages, completion) as channel:
                    worker = channel.process.worker
                    active = asyncio.create_task(
                        channel.request("session/prompt", self._params("active"))
                        if origin == "wire" else worker.prompt(self._params("active")))
                    try:
                        await asyncio.wait_for(entered.wait(), 3)
                        owner = worker._prompt_task
                        self.assertIsNotNone(owner)
                        initial = copy.deepcopy(worker.session.transcript_items)
                        await asyncio.wait_for(channel.request("session/cancel", {}), 3)
                        for text in ["/ps", "/ps stop 1", "/status", "second prompt"]:
                            with self.assertRaisesRegex(acps.TransportError, "already running"):
                                await asyncio.wait_for(
                                    channel.request("session/prompt", self._params(text)), 3)
                            with self.assertRaisesRegex(acps.TransportError, "already running"):
                                await worker.prompt(self._params(text))
                            replies = []
                            write = worker.write

                            def record(message):
                                replies.append(message)
                                write(message)

                            with mock.patch.object(worker, "write", new=record):
                                await worker.handle(acps.request(
                                    "inline-rejected", "session/prompt", self._params(text)))
                            self.assertEqual(replies, [acps.response("inline-rejected", error={
                                "code": acps.INVALID_PARAMS,
                                "message": "a prompt is already running for this session",
                            })])
                            self.assertIs(worker._prompt_task, owner)
                            self.assertFalse(owner.done())
                            self.assertTrue(worker.cancel_event.is_set())
                            self.assertEqual(worker.session.transcript_items, initial)
                            self.assertEqual(messages, [])
                        # Close must still wait for the original prompt, not a
                        # rejected command or a substituted task.
                        close = asyncio.create_task(worker.close())
                        await asyncio.sleep(0)
                        self.assertFalse(close.done())
                        release.set()
                        self.assertEqual(await asyncio.wait_for(active, 3),
                                         {"stopReason": "cancelled"})
                        await asyncio.wait_for(close, 3)
                        await asyncio.sleep(0)
                        self.assertEqual(_tool_updates(messages, "prompt-owner"),
                                         _assistant_chunks("[turn cancelled by user]"))
                    finally:
                        worker.cancel_event.set()
                        release.set()
                        await asyncio.wait_for(asyncio.gather(active, return_exceptions=True), 3)

    async def test_bang_cancellation_reaps_command_without_requesting_inference(self):
        import shlex
        from loki_agent import loki
        from process_lifecycle_fixtures import ProcessResources
        messages = []
        completion = mock.AsyncMock(side_effect=AssertionError("cancelled bang requested inference"))
        command_parts = [sys.executable, "-c",
                         "import time; print('bang-ready', flush=True); time.sleep(60)"]
        command = (subprocess.list2cmdline(command_parts)
                   if os.name == "nt" else shlex.join(command_parts))
        with tempfile.TemporaryDirectory() as root:
            async with self._channel(root, messages, completion) as channel:
                worker = channel.process.worker
                active = asyncio.create_task(channel.request(
                    "session/prompt", self._params("!" + command)))
                resources = None
                try:
                    deadline = asyncio.get_running_loop().time() + 5
                    while True:
                        manager = worker.session.job_manager
                        job = next(iter(manager.jobs.values()), None) if manager is not None else None
                        if job is not None and job.process is not None:
                            self.assertIsNone(job.process.returncode)
                            if "bang-ready" in loki._read_spool_tail(job.stdout_path):
                                break
                        if asyncio.get_running_loop().time() >= deadline:
                            self.fail("bang command did not become ready")
                        await asyncio.sleep(.01)
                    resources = ProcessResources(job.process)
                    await asyncio.wait_for(channel.request("session/cancel", {}), 3)
                    done, _pending = await asyncio.wait({active}, timeout=5)
                    self.assertTrue(done, "cancellation did not finish the bang command")
                    self.assertEqual(active.result(), {"stopReason": "cancelled"})
                    self.assertEqual(job.status, "cancelled")
                    self.assertIsNotNone(job.process.returncode)
                    with open(job.metadata_path, encoding="utf-8") as stream:
                        self.assertEqual(json.load(stream)["status"], "cancelled")
                    completion.assert_not_awaited()
                    updates = _tool_updates(messages, "prompt-owner")
                    self.assertEqual(updates[-1], _assistant_chunks("[turn cancelled by user]")[0])
                    self.assertIn("bang-ready", updates[0]["content"]["text"])
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    resources.assert_released(self)
                finally:
                    if resources is not None:
                        if job.process.returncode is None:
                            loki.host_process.signal_group(
                                job.process, job.pgid, loki.host_process.FORCE)
                            await asyncio.wait_for(job.process.wait(), 3)
                        await resources.cleanup()
                    if not active.done():
                        active.cancel()
                    await asyncio.wait_for(asyncio.gather(active, return_exceptions=True), 3)


class SavedSessionJourneyTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_restore_continues_and_saves(self):
        from contextlib import ExitStack
        from test_http_client import FakeConnector
        from loki_agent import formats, http_client, loki
        from loki_agent.sessions import Session

        for explicit in (False, True):
            for method in acp.RESTORE_METHODS:
                with self.subTest(explicit=explicit, method=method), tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                    workspace = os.path.join(root, 'workspace with spaces')
                    os.mkdir(workspace)
                    environment = {key: value for key, value in os.environ.items()
                                   if not key.startswith("LOKI_")
                                   and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
                    environment.update({"HOME": root, "XDG_CONFIG_HOME": os.path.join(root, "config"),
                                        "XDG_STATE_HOME": os.path.join(root, "state")})
                    stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
                    stack.enter_context(mock.patch.object(models, "ensure_index", new=mock.AsyncMock(return_value=({}, {}))))
                    responses = []
                    for answer in ("original answer", "continued answer"):
                        body = json.dumps({
                            "id": answer,
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": answer},
                                "finish_reason": "stop",
                            }],
                        }).encode()
                        responses.append(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                                         + str(len(body)).encode() + b"\r\n\r\n" + body)
                    connector = FakeConnector(responses)
                    open_connection = asyncio.open_connection

                    async def connect(*args, **kwargs):
                        if "sock" in kwargs:
                            return await open_connection(*args, **kwargs)
                        return await connector.open_connection(*args, **kwargs)

                    stack.enter_context(mock.patch.object(
                        http_client.asyncio, "open_connection", new=connect))
                    stack.enter_context(mock.patch.object(loki, "_DEFAULT_SESSION", Session(shell_cwd=workspace)))
                    startup = {"LOKI_PROVIDER": "openai", "LOKI_API_BASE": "https://saved.example/v1",
                               "LOKI_MODEL": "saved-model", "LOKI_STREAM": "0"}
                    stack.enter_context(mock.patch.object(loki, "CREDENTIALS", CredentialStore(startup)))
                    processes = []
                    delegations = []
                    channels = []
                    messages = []
                    changed = asyncio.Event()
                    front = None
                    sources = []

                    def write(message):
                        messages.append(message)
                        changed.set()

                    async def launch(cwd, environment, delegation):
                        session = Session(shell_cwd=cwd)
                        session.credential_authority = front.credential_broker
                        loki._DEFAULT_SESSION = session
                        # Match the worker entrypoint's explicit startup config;
                        # prepare_open itself deliberately does not invent it.
                        if loki.explicit_connection_option(loki.CREDENTIALS):
                            loki.apply_runtime_config(loki.build_config_from_env(
                                credentials=loki.CREDENTIALS))

                        def before_request(message):
                            if message["method"] == "session/commit_open":
                                self.assertNotIn(process.worker.session_id, front.workers)

                        process = _LocalWorkerProcess(session, "provisional", before_request)
                        processes.append(process)
                        delegations.append(delegation)
                        return process

                    channel_type = acp.WorkerChannel

                    def channel(*args):
                        result = channel_type(*args)
                        channels.append(result)
                        return result

                    stack.enter_context(mock.patch.object(acp.runtime_isolation, "start_worker", new=launch))
                    stack.enter_context(mock.patch.object(acp, "WorkerChannel", new=channel))
                    # OS process release is outside this byte-transport lane.
                    stack.enter_context(mock.patch.object(acp.runtime_isolation, "close_runtime_process", new=lambda process: None))

                    async def backstop():
                        for source in sources:
                            await source.close()
                        for process in processes:
                            process.close()
                            await asyncio.wait_for(process.wait(), 1)
                        for channel in channels:
                            await asyncio.wait_for(channel.close(), 1)
                        if front is not None:
                            for task in list(front._tasks):
                                task.cancel()
                            await asyncio.gather(*front._tasks, return_exceptions=True)

                    # Runs before restoring patched globals, including on failure.
                    async def response(request_id):
                        async with asyncio.timeout(3):
                            while True:
                                changed.clear()
                                for message in messages:
                                    if message.get("id") == request_id:
                                        self.assertNotIn("error", message, message)
                                        return message["result"]
                                await changed.wait()

                    async def request(request_id, operation, params):
                        await source.send(acps.request(request_id, operation, params))
                        return await response(request_id)

                    try:
                        front = acp.Front(lambda: None, write, loki.CREDENTIALS)
                        source = _FrontInput(front)
                        sources.append(source)
                        await source.send(acps.request(0, 'initialize', {
                            'clientCapabilities': {'elicitation': {'form': {}}}}))
                        opened = await request(1, "session/new", {"cwd": workspace})
                        session_id = opened["sessionId"]
                        result = await request(2, "session/prompt", {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "original prompt"}]})
                        self.assertEqual(result["stopReason"], "end_turn")
                        path = processes[0].worker.session.chat_log_path
                        self.assertEqual(await request(3, "session/close", {"sessionId": session_id}), {})
                        await source.finish()
                        self.assertEqual(processes[0].returncode, 0)
                        self.assertTrue(channels[0]._reader_task.done())
                        with open(path, "rb") as stream:
                            original = stream.read()
                        self.assertEqual(_conversation_pairs(json.loads(original)),
                                         _expected_pairs(("original prompt", "original answer")))
                        credentials = CredentialStore({**startup, "LOKI_API_BASE": "https://explicit.example/v1",
                                                       "LOKI_MODEL": "explicit-model"} if explicit else {})
                        loki.CREDENTIALS = credentials
                        front = acp.Front(lambda: None, write, credentials)
                        source = _FrontInput(front)
                        sources.append(source)
                        await source.send(acps.request(0, 'initialize', {
                            'clientCapabilities': {'elicitation': {'form': {}}}}))
                        messages.clear()
                        await source.send(acps.request(73, method, {
                            "sessionId": session_id, "cwd": workspace,
                            "replay": method == "session/resume"}))
                        if not explicit:
                            async with asyncio.timeout(3):
                                while not any(m.get("method") == "elicitation/create" for m in messages):
                                    changed.clear()
                                    await changed.wait()
                            elicitation = next(m for m in messages if m.get("method") == "elicitation/create")
                            params = elicitation["params"]
                            self.assertEqual(params["requestId"], 73)
                            self.assertEqual(params["mode"], "form")
                            self.assertIs(params["requestedSchema"]["properties"]["authorize"]["default"], False)
                            self.assertIn('"https://saved.example/v1/chat/completions"', params["message"])
                            self.assertIn("Working directory:", params["message"])
                            self.assertIn(json.dumps(workspace, ensure_ascii=True), params["message"])
                            self.assertNotIn(session_id, front.workers)
                            self.assertEqual(front._sessions[session_id].state, "opening")
                            self.assertEqual(processes[-1].methods, ["session/prepare_open"])
                            self.assertIsNone(processes[-1].worker.session.runtime_config)
                            self.assertIsNotNone(processes[-1].worker._pending_open)
                            self.assertEqual(len(connector.writers), 1)
                            self.assertEqual(_historical_chunks(messages, session_id), [])
                            with open(path, "rb") as stream:
                                self.assertEqual(stream.read(), original)
                            await source.send(acps.response(elicitation["id"], result={
                                "action": "accept", "content": {"authorize": True}}))
                        restored = await response(73)
                        self.assertNotIn("sessionId", restored)
                        self.assertIn("configOptions", restored)
                        self.assertIn(session_id, front.workers)
                        self.assertTrue(all(owner.state != "opening"
                                            for owner in front._sessions.values()))
                        self.assertFalse(front._client_requests)
                        self.assertEqual(processes[-1].methods, ["session/prepare_open", "session/commit_open"])
                        if explicit:
                            self.assertFalse(any(m.get("method") == "elicitation/create" for m in messages))
                        expected = [("user_message_chunk", {"type": "text", "text": "original prompt"}),
                                    ("agent_message_chunk", {"type": "text", "text": "original answer"})]
                        self.assertEqual(_historical_chunks(messages, session_id), expected if method == "session/load" else [])
                        continued_start = len(messages)
                        result = await request(74, "session/prompt", {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "continued prompt"}]})
                        self.assertEqual(result["stopReason"], "end_turn")
                        self.assertEqual(_historical_chunks(messages[continued_start:], session_id), [
                            ("agent_message_chunk", {"type": "text", "text": "continued answer"})])
                        self.assertEqual(len(connector.writers), 2)
                        self.assertEqual(connector.responses, [])
                        self.assertEqual([call["host"] for call in connector.calls],
                                         ["saved.example", "explicit.example" if explicit else "saved.example"])
                        packet = bytes(connector.writers[-1].data)
                        headers, body = packet.split(b"\r\n\r\n", 1)
                        host = b"explicit.example" if explicit else b"saved.example"
                        self.assertIn(b"POST /v1/chat/completions HTTP/1.1\r\n", headers)
                        self.assertIn(b"Host: " + host, headers)
                        payload = json.loads(body)
                        self.assertEqual(payload["model"], "explicit-model" if explicit else "saved-model")
                        self.assertEqual([(m["role"], m["content"]) for m in payload["messages"]
                                          if m["role"] in ("user", "assistant")],
                                         [("user", "original prompt"), ("assistant", "original answer"),
                                          ("user", "continued prompt")])
                        for writer in connector.writers:
                            self.assertTrue(writer.closed)
                            self.assertTrue(writer.wait_closed_called)
                        self.assertEqual(await request(75, "session/close", {"sessionId": session_id}), {})
                        await source.finish()
                        self.assertFalse(front.workers)
                        self.assertEqual(processes[-1].returncode, 0)
                        self.assertTrue(processes[-1].close_task.done())
                        self.assertIsNone(processes[-1].close_task.exception())
                        self.assertTrue(channels[-1]._reader_task.done())
                        for delegation in delegations:
                            self.assertIsNone(delegation.owner_parent)
                            self.assertIsNone(delegation.owner_child)
                            self.assertIsNone(delegation.credential_child)
                            server = delegation.credential_server
                            self.assertTrue(server._reader_task.done())
                            self.assertTrue(server._writer_close_task.done())
                            self.assertIsNone(server._writer_close_task.exception())
                        self.assertIsNone(channels[-1].credential_delegation)
                        self.assertFalse(channels[-1]._pending)
                        with open(path, encoding="utf-8") as stream:
                            blob = json.load(stream)
                        formats.validate_events(blob["events"])
                        self.assertEqual(_conversation_pairs(blob), _expected_pairs(
                            ("original prompt", "original answer"), ("continued prompt", "continued answer")))
                    finally:
                        await backstop()


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
                self.assertEqual([option["id"] for option in options], ["model", "reasoning_traces"])
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
        return models.ReasoningEffortProfile(list(values))

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
                    ["model", "thought_level", "other"],
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
                self.assertEqual([option["id"] for option in plain["configOptions"]],
                                 ["model", "reasoning_traces"])
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
        from loki_agent import formats, loki, protocols
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, 'read.txt')
            with open(target, 'w', encoding='utf-8') as stream:
                stream.write('real tool continuation')
            session = Session(shell_cwd=directory)
            session.chat_log_path = os.path.join(directory, 'chat.json')
            session.runtime_config = loki.make_runtime_config(
                'https://api.openai.com/v1/responses',
                protocols.OPENAI_RESPONSES, model='gpt-test',
                provider_id='openai',
                reasoning_effort_profile=self._profile('low', 'high'))
            session.reasoning_effort_preference = 'high'
            session.session_state = {'reasoning_effort': 'high'}
            written, snapshots = [], []
            worker = Worker(session, written.append, 'session')
            worker._set_choices([
                ({'value': 'model', 'name': 'Model'}, object())], 'model')
            started, release = asyncio.Event(), asyncio.Event()

            async def completion(items, tools, *args, thinking, **kwargs):
                snapshots.append(thinking.effort)
                if len(snapshots) == 1:
                    started.set()
                    await release.wait()
                    return formats.DecodedTurn([
                        formats.tool_call_item('read', 'Read',
                                               {'file_path': target})])
                if len(snapshots) == 2:
                    result = next(item for item in items
                                  if item.get('type') == 'tool_result')
                    self.assertEqual(result['call_id'], 'read')
                    self.assertFalse(result['is_error'])
                    self.assertIn('real tool continuation',
                                  result['content'][0]['text'])
                else:
                    self.assertEqual(len(snapshots), 3)
                return formats.DecodedTurn([
                    formats.message_item('assistant',
                                         f'answer-{len(snapshots)}')])

            async def prompt(request_id):
                await worker.handle({
                    'jsonrpc': '2.0', 'id': request_id,
                    'method': 'session/prompt', 'params': {
                        'sessionId': 'session',
                        'prompt': [{'type': 'text', 'text': 'continue'}]}},
                    concurrent=True)

            async def scenario():
                try:
                    await prompt(1)
                    await asyncio.wait_for(started.wait(), 2)
                    await worker.handle({
                        'jsonrpc': '2.0', 'id': 2,
                        'method': 'session/set_config_option', 'params': {
                            'sessionId': 'session',
                            'configId': 'reasoning_effort',
                            'value': 'effort:low'}}, concurrent=True)
                    release.set()
                    await asyncio.wait_for(worker._prompt_task, 2)
                    self.assertEqual(snapshots, ['high', 'high'])
                    await prompt(3)
                    await asyncio.wait_for(worker._prompt_task, 2)
                finally:
                    release.set()
                    await asyncio.wait_for(worker.close(), 2)

            with mock.patch.object(loki, '_DEFAULT_SESSION', session), \
                    mock.patch.object(loki, 'file_state', {}), \
                    mock.patch.object(loki, 'async_chat_completion', completion):
                asyncio.run(scenario())
            self.assertEqual(snapshots, ['high', 'high', 'low'])
            response = next(message for message in written
                            if message.get('id') == 2)
            self.assertEqual(response['result']['configOptions'][1]
                             ['currentValue'], 'effort:low')
            for request_id in (1, 3):
                response = next(message for message in written
                                if message.get('id') == request_id)
                self.assertEqual(response['result']['stopReason'], 'end_turn')
            with open(session.chat_log_path, encoding='utf-8') as stream:
                saved = json.load(stream)
            self.assertEqual(saved['session_state']['reasoning_effort'], 'low')
            results = [item for item in saved['events']
                       if item.get('type') == 'tool_result']
            self.assertEqual([item['call_id'] for item in results], ['read'])
            self.assertFalse(results[0]['is_error'])
            self.assertIn('real tool continuation', results[0]['content'][0]['text'])
            self.assertIn('answer-2', json.dumps(saved['events']))
            self.assertIn('answer-3', json.dumps(saved['events']))


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


class WorkerSessionContractTests(unittest.TestCase):
    def setUp(self):
        assume_endpoints_approved(self)

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

    def _channel(self, front, selection):
        channel = mock.Mock()
        channel.session_id = "s"
        channel._closed = False
        channel.process.returncode = None
        channel.request = mock.AsyncMock(return_value=selection)
        owner = front._reserve_session('s')
        owner.channel = channel
        owner.state = 'active'
        return channel

    @asynccontextmanager
    async def _approval_journey(self):
        """Real application components; only catalog/HTTP and OS-child seams.

        All client traffic uses one serial Front.run input stream. This is
        component qualification, not a shipped-process catalog/HTTP check.
        """
        from types import SimpleNamespace
        from test_http_client import FakeConnector
        from loki_agent import endpoint_pins, http_client, loki
        from loki_agent.sessions import Session

        async with asyncio.timeout(20):
            with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                workspace = os.path.join(root, 'workspace')
                os.mkdir(workspace)
                environment = {key: value for key, value in os.environ.items()
                               if not key.startswith('LOKI_')
                               and not key.endswith(('_KEY', '_TOKEN', '_PAT'))}
                environment.update(HOME=root, XDG_CONFIG_HOME=os.path.join(root, 'config'),
                                   XDG_STATE_HOME=os.path.join(root, 'state'))
                stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
                selected, unrelated, previous = (
                    'approval-selected-secret', 'approval-unrelated-secret',
                    'approval-previous-secret')
                credentials = CredentialStore({
                    'ACME_API_KEY': selected, 'UNRELATED_API_KEY': unrelated,
                    'LOKI_API_KEY': previous, 'LOKI_PROVIDER': 'openai',
                    'LOKI_API_BASE': 'https://previous.example/v1',
                    'LOKI_MODEL': 'previous-model', 'LOKI_STREAM': '0'})
                endpoint = 'https://approved.example/override/v1'
                catalog = {'acme': {
                    'id': 'acme', 'name': 'Acme', 'env': ['ACME_API_KEY'],
                    'npm': '@ai-sdk/openai-compatible', 'api': 'https://provider.example/base/v1',
                    'models': {'chosen-model': {
                        'id': 'chosen-model', 'name': 'Chosen Model',
                        'provider': {'api': endpoint, 'npm': '@ai-sdk/openai-compatible'}}}}}
                groups = models.build_groups(catalog)
                stack.enter_context(mock.patch.object(
                    models, 'ensure_index',
                    new=mock.AsyncMock(return_value=(catalog, groups))))
                packets = []
                for answer in ('approved answer', 'pinned answer'):
                    body = json.dumps({'id': answer, 'choices': [{
                        'index': 0, 'message': {'role': 'assistant', 'content': answer},
                        'finish_reason': 'stop'}]}).encode()
                    packets.append(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '
                                   + str(len(body)).encode() + b'\r\n\r\n' + body)
                f = SimpleNamespace(messages=[], changed=asyncio.Event(), tasks=[],
                                    processes=[], delegations=[], connector=FakeConnector(packets),
                                    trace=[], endpoint=endpoint, workspace=workspace,
                                    secrets=(selected, unrelated, previous),
                                    tasks_before=asyncio.all_tasks())

                def write(message):
                    f.messages.append(message)
                    if message.get('method') == 'elicitation/create':
                        f.trace.append('elicitation')
                    f.changed.set()

                f.front = acp.Front(lambda: None, write, credentials)
                f.source = _FrontInput(f.front)
                await f.source.send(acps.request(0, 'initialize', {
                    'clientCapabilities': {'elicitation': {'form': {}}}}))
                stack.enter_context(mock.patch.object(loki, 'CREDENTIALS', f.front.credentials))
                stack.enter_context(mock.patch.object(loki, '_DEFAULT_SESSION', Session(shell_cwd=workspace)))
                real_connect = asyncio.open_connection

                async def connect(*args, **kwargs):
                    if 'sock' in kwargs:
                        return await real_connect(*args, **kwargs)
                    f.trace.append('http')
                    return await f.connector.open_connection(*args, **kwargs)

                stack.enter_context(mock.patch.object(http_client.asyncio, 'open_connection', new=connect))
                real_lease = f.front.credential_broker.lease
                f.leases = []

                async def lease(ref, **kwargs):
                    f.leases.append(ref.encode())
                    f.trace.append('lease')
                    return await real_lease(ref, **kwargs)

                stack.enter_context(mock.patch.object(f.front.credential_broker, 'lease', new=lease))
                endpoint_pins.record('untouched', 'https://untouched.example/v1', 'env:UNRELATED_API_KEY')
                f.pins_path = endpoint_pins._path()
                with open(f.pins_path, 'rb') as stream:
                    f.original_pins = stream.read()
                f.expected_pins = {
                    'untouched': {'api': 'https://untouched.example/v1',
                                  'credential': 'env:UNRELATED_API_KEY'},
                    'acme': {'api': endpoint, 'credential': 'env:ACME_API_KEY'}}
                real_record = endpoint_pins.record

                def record(*args):
                    real_record(*args)
                    f.trace.append('pin')

                stack.enter_context(mock.patch.object(endpoint_pins, 'record', new=record))

                async def launch(cwd, environment, delegation):
                    session = Session(shell_cwd=cwd)
                    # Same-process authority, as in the existing selection
                    # journey; no worker credential-IPC claim is made here.
                    session.credential_authority = f.front.credential_broker
                    loki._DEFAULT_SESSION = session
                    loki.apply_runtime_config(loki.build_config_from_env(credentials=loki.CREDENTIALS))

                    def before_request(message):
                        if message['method'] == 'session/describe_config_selection':
                            f.trace.append('describe')
                        if message['method'] == 'session/set_config_option':
                            f.trace.append('switch')
                            with open(f.pins_path, encoding='utf-8') as stream:
                                self.assertEqual(json.load(stream), f.expected_pins)

                    process = _LocalWorkerProcess(session, 'provisional', before_request)
                    f.processes.append(process)
                    f.delegations.append(delegation)
                    f.session = session
                    return process

                stack.enter_context(mock.patch.object(acp.runtime_isolation, 'start_worker', new=launch))
                stack.enter_context(mock.patch.object(acp.runtime_isolation, 'close_runtime_process', new=lambda process: None))

                async def wait_message(predicate):
                    async with asyncio.timeout(3):
                        while True:
                            f.changed.clear()
                            for message in f.messages:
                                if predicate(message):
                                    return message
                            await f.changed.wait()

                async def request(request_id, method, params):
                    await f.source.send(acps.request(request_id, method, params))
                    message = await wait_message(lambda message: message.get('id') == request_id)
                    self.assertNotIn('error', message, message)
                    return message['result']

                f.wait_message, f.request = wait_message, request
                try:
                    opened = await request(1, 'session/new', {'cwd': workspace})
                    f.session_id = opened['sessionId']
                    f.channel = f.front.workers[f.session_id]
                    choice, = [option for config in opened['configOptions'] if config['id'] == 'model'
                               for option in config['options'] if option['value'] == 'acme/chosen-model']
                    f.value = choice['value']
                    f.params = {'sessionId': f.session_id, 'configId': 'model', 'value': f.value}
                    f.original_config = f.session.runtime_config
                    f.path = f.session.chat_log_path
                    self.assertTrue(loki.save_chat_log())
                    with open(f.path, 'rb') as stream:
                        f.original_chat = stream.read()
                    f.trace.clear()
                    self.assertEqual(f.leases, [])
                    yield f
                finally:
                    for task in f.tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*f.tasks, return_exceptions=True)
                    await f.source.close()
                    # Failure backstop, not release evidence: assertions above
                    # must pass before this fallback can repair a leaked owner.
                    await f.front.shutdown()
                    for delegation in f.delegations:
                        await delegation.close()
                    for task in list(f.front._tasks):
                        task.cancel()
                    await asyncio.gather(*f.front._tasks, return_exceptions=True)

    async def _pending_approval(self, f, request_id):
        task = asyncio.create_task(f.wait_message(
            lambda message: message.get('id') == request_id))
        f.tasks.append(task)
        await f.source.send(acps.request(
            request_id, 'session/set_config_option', f.params))
        elicitation = await f.wait_message(
            lambda message: message.get('method') == 'elicitation/create'
            and message['params'].get('requestId') == request_id)
        self.assertFalse(task.done())
        self.assertIn(elicitation['id'], f.front._client_requests)
        self.assertEqual(elicitation['jsonrpc'], '2.0')
        params = elicitation['params']
        self.assertEqual(params['mode'], 'form')
        self.assertEqual(params['message'], 'Send this credential to this endpoint?\n'
                         f'Endpoint: "{f.endpoint}"\nCredential: "env:ACME_API_KEY"')
        self.assertEqual(params['requestedSchema']['type'], 'object')
        self.assertEqual(params['requestedSchema']['required'], ['approve'])
        field = params['requestedSchema']['properties']['approve']
        self.assertEqual(field['type'], 'boolean')
        self.assertIs(field['default'], False)
        self.assertEqual(f.trace[-2:], ['describe', 'elicitation'])
        self.assertIs(f.session.runtime_config, f.original_config)
        self.assertEqual(f.session.session_state['connection'],
                         json.loads(f.original_chat)['session_state']['connection'])
        self.assertFalse(f.session.chat_log_dirty)
        self.assertEqual(f.leases, [])
        self.assertEqual(f.connector.calls, [])
        with open(f.pins_path, 'rb') as stream:
            self.assertEqual(stream.read(), f.original_pins)
        with open(f.path, 'rb') as stream:
            self.assertEqual(stream.read(), f.original_chat)
        return task, elicitation

    async def _close_approval_journey(self, f):
        self.assertEqual(await f.request(90, 'session/close', {'sessionId': f.session_id}), {})
        await f.source.finish()
        await self._assert_approval_released(f)

    async def _assert_approval_released(self, f):
        self.assertFalse(f.front._sessions)
        self.assertFalse(f.front.workers)
        self.assertFalse(f.front._client_requests)
        self.assertFalse(f.channel._pending)
        self.assertTrue(f.channel._reader_task.done())
        self.assertIsNone(f.channel.credential_delegation)
        for process in f.processes:
            self.assertEqual(process.returncode, 0)
            self.assertTrue(process.close_task.done())
            self.assertIsNone(process.close_task.exception())
        for delegation in f.delegations:
            self.assertIsNone(delegation.owner_parent)
            self.assertIsNone(delegation.owner_child)
            self.assertIsNone(delegation.credential_child)
            server = delegation.credential_server
            self.assertTrue(server._reader_task.done())
            self.assertTrue(server._writer_close_task.done())
            self.assertIsNone(server._writer_close_task.exception())
        for writer in f.connector.writers:
            self.assertTrue(writer.closed)
            self.assertTrue(writer.wait_closed_called)
        await asyncio.sleep(0)
        self.assertFalse(f.front._tasks)
        if f.session.job_manager is not None:
            self.assertEqual(f.session.job_manager.jobs, {})
        self.assertFalse(asyncio.all_tasks() - f.tasks_before)
        with open(f.path, 'rb') as stream:
            saved = stream.read()
        with open(f.pins_path, 'rb') as stream:
            pins = stream.read()
        for secret in f.secrets:
            self.assertNotIn(secret.encode(), saved + pins + json.dumps(f.messages).encode())

    def _assert_approved_descriptor(self, f, blob):
        descriptor = blob['session_state']['connection']
        self.assertEqual(descriptor['provider_id'], 'acme')
        self.assertEqual(descriptor['model'], 'chosen-model')
        self.assertEqual(descriptor['chat_url'], f.endpoint + '/chat/completions')
        self.assertEqual(descriptor['credential'], {'kind': 'env', 'name': 'ACME_API_KEY'})
        self.assertEqual(descriptor['protocol'], 'openai_chat')
        self.assertIs(descriptor['stream'], False)

    async def test_approval_routes_pins_switches_infers_and_reuses(self):
        from loki_agent import endpoint_pins, formats

        async with self._approval_journey() as f:
            task, elicitation = await self._pending_approval(f, 9)
            # A foreign response cannot resolve this approval.
            await f.source.send(acps.response('unrelated-id', result={
                'action': 'accept', 'content': {'approve': True}}))
            self.assertFalse(f.front._client_requests[elicitation['id']].done())
            self.assertFalse(task.done())
            await f.source.send(acps.response(elicitation['id'], result={
                'action': 'accept', 'content': {'approve': True}}))
            await asyncio.wait_for(task, 3)
            reply = await f.wait_message(lambda message: message.get('id') == 9)
            self.assertNotIn('error', reply, reply)
            self.assertEqual(f.trace, ['describe', 'elicitation', 'pin', 'switch'])
            model_option, = [option for option in reply['result']['configOptions'] if option['id'] == 'model']
            self.assertEqual(model_option['currentValue'], f.value)
            self.assertFalse(f.front._client_requests)
            self.assertEqual(f.session.runtime_config.model, 'chosen-model')
            self.assertEqual(f.session.runtime_config.chat_provider.chat_url, f.endpoint + '/chat/completions')
            with open(f.path, 'rb') as stream:
                switched = json.load(stream)
            self._assert_approved_descriptor(f, switched)
            self.assertEqual(_conversation_pairs(switched), [])
            self.assertEqual(endpoint_pins.load(), f.expected_pins)
            self.assertEqual(endpoint_pins.status('acme', f.endpoint, 'env:ACME_API_KEY'),
                             (endpoint_pins.PINNED, f.expected_pins['acme']))
            expected_pairs = []
            for index, (prompt, answer) in enumerate((('approved prompt', 'approved answer'),
                                                      ('pinned prompt', 'pinned answer'))):
                if index:
                    with open(f.pins_path, 'rb') as stream:
                        pinned_bytes = stream.read()
                    await f.request(12, 'session/set_config_option', f.params)
                    self.assertEqual(sum(message.get('method') == 'elicitation/create'
                                         for message in f.messages), 1)
                    with open(f.pins_path, 'rb') as stream:
                        self.assertEqual(stream.read(), pinned_bytes)
                start = len(f.messages)
                result = await f.request(20 + index, 'session/prompt', {
                    'sessionId': f.session_id, 'prompt': [{'type': 'text', 'text': prompt}]})
                self.assertEqual(result, {'stopReason': 'end_turn'})
                self.assertEqual(_historical_chunks(f.messages[start:], f.session_id), [
                    ('agent_message_chunk', {'type': 'text', 'text': answer})])
                updates = [message for message in f.messages[start:] if message.get('method') == 'session/update']
                self.assertTrue(updates)
                self.assertTrue(all(message['params']['sessionId'] == f.session_id for message in updates))
                packet = bytes(f.connector.writers[index].data)
                headers, body = packet.split(b'\r\n\r\n', 1)
                self.assertTrue(headers.startswith(b'POST /override/v1/chat/completions HTTP/1.1\r\n'))
                self.assertIn(b'Host: approved.example', headers)
                authorization = [line for line in headers.split(b'\r\n')
                                 if line.lower().startswith(b'authorization:')]
                self.assertEqual(authorization, [b'Authorization: Bearer ' + f.secrets[0].encode()])
                for secret in f.secrets[1:]:
                    self.assertNotIn(secret.encode(), packet)
                payload = json.loads(body)
                self.assertEqual(payload['model'], 'chosen-model')
                self.assertNotIn('stream', payload)
                expected_context = expected_pairs + [('user', prompt)]
                self.assertEqual([(message['role'], message['content']) for message in payload['messages']
                                  if message['role'] in ('user', 'assistant')], expected_context)
                expected_pairs.extend([('user', prompt), ('assistant', answer)])
                with open(f.path, 'rb') as stream:
                    saved = stream.read()
                blob = json.loads(saved)
                formats.validate_events(blob['events'])
                self.assertEqual(_conversation_pairs(blob), [
                    (role, [{'type': 'text', 'text': text}])
                    for role, text in expected_pairs])
                self._assert_approved_descriptor(f, blob)
                with open(f.pins_path, 'rb') as stream:
                    pins = stream.read()
                for secret in f.secrets:
                    self.assertNotIn(secret.encode(), saved + pins + json.dumps(f.messages).encode())
            self.assertEqual([call['host'] for call in f.connector.calls], ['approved.example'] * 2)
            self.assertEqual(f.leases, ['env:ACME_API_KEY'] * 2)
            self.assertEqual(f.connector.responses, [])
            self.assertEqual(f.trace, ['describe', 'elicitation', 'pin', 'switch', 'lease', 'http',
                                       'describe', 'switch', 'lease', 'http'])
            await self._close_approval_journey(f)

    async def test_declined_approval_fails_the_switch(self):
        refusals = ({'action': 'decline'}, {'action': 'cancel'}, {}, [],
                    {'action': 'accept', 'content': {'approve': False}},
                    {'action': 'accept', 'content': {}},
                    {'action': 'accept', 'content': {'approve': 1}},
                    {'action': 'accept', 'content': {'approve': 'true'}})
        for refusal in refusals:
            with self.subTest(refusal=refusal):
                async with self._approval_journey() as f:
                    for request_id in (9, 10):
                        task, elicitation = await self._pending_approval(f, request_id)
                        await f.source.send(acps.response(elicitation['id'], result=refusal))
                        await asyncio.wait_for(task, 3)
                        reply = await f.wait_message(lambda message: message.get('id') == request_id)
                        self.assertIn('error', reply, reply)
                        self.assertEqual(reply['error'], {
                            'code': acps.INVALID_PARAMS,
                            'message': 'the provider endpoint was not approved'})
                        self.assertIs(f.session.runtime_config, f.original_config)
                        self.assertEqual(f.session.session_state['connection'],
                                         json.loads(f.original_chat)['session_state']['connection'])
                        self.assertFalse(f.session.chat_log_dirty)
                        with open(f.path, 'rb') as stream:
                            self.assertEqual(stream.read(), f.original_chat)
                        with open(f.pins_path, 'rb') as stream:
                            self.assertEqual(stream.read(), f.original_pins)
                        self.assertEqual(f.leases, [])
                        self.assertEqual(f.connector.calls, [])
                        self.assertFalse(f.front._client_requests)
                        self.assertNotIn('switch', f.trace)
                        self.assertNotIn('pin', f.trace)
                    await self._close_approval_journey(f)

    def _assert_no_pin_or_inference(self, f):
        with open(f.pins_path, 'rb') as stream:
            self.assertEqual(stream.read(), f.original_pins)
        self.assertEqual(f.leases, [])
        self.assertEqual(f.connector.calls, [])
        self.assertNotIn('pin', f.trace)
        self.assertNotIn('switch', f.trace)

    async def test_pending_approval_rejects_buffered_work_only_in_its_session(self):
        async with self._approval_journey() as f:
            calls = []

            class OtherChannel:
                _closed = False
                process = mock.Mock(returncode=None)

                async def request(self, method, params, forwarded=None):
                    calls.append(method)
                    if forwarded is not None:
                        forwarded.set()
                    return {'stopReason': 'end_turn'}

                async def close(self):
                    pass

            other = f.front._reserve_session('other')
            other.channel = OtherChannel()
            other.state = 'active'
            await f.source.send_many(
                acps.request(9, 'session/set_config_option', f.params),
                acps.request(10, 'session/prompt', {
                    'sessionId': f.session_id,
                    'prompt': [{'type': 'text', 'text': 'must not run'}]}),
                acps.request(11, 'session/set_config_option', f.params),
                acps.request(12, 'session/prompt', {
                    'sessionId': 'other',
                    'prompt': [{'type': 'text', 'text': 'independent'}]}))
            for request_id in (10, 11):
                reply = await f.wait_message(lambda m: m.get('id') == request_id)
                self.assertEqual(reply['error']['code'], acps.INVALID_PARAMS)
                self.assertIn('request was not executed', reply['error']['message'])
            self.assertEqual(await f.wait_message(lambda m: m.get('id') == 12),
                             acps.response(12, result={'stopReason': 'end_turn'}))
            self.assertEqual(calls, ['session/prompt'])
            ask = await f.wait_message(lambda m: m.get('method') == 'elicitation/create')
            self.assertEqual(ask['params']['requestId'], 9)
            self.assertEqual(f.processes[0].methods[-1], 'session/describe_config_selection')
            self._assert_no_pin_or_inference(f)
            with open(f.path, 'rb') as stream:
                self.assertEqual(stream.read(), f.original_chat)
            await f.source.send(acps.response(ask['id'], result={'action': 'decline'}))
            reply = await f.wait_message(lambda m: m.get('id') == 9)
            self.assertIn('not approved', reply['error']['message'])
            self.assertEqual(await f.request(80, 'session/close', {'sessionId': 'other'}), {})
            await self._close_approval_journey(f)

    async def test_close_invalidates_approval_before_accept_and_after_reopen(self):
        async with self._approval_journey() as f:
            task, ask = await self._pending_approval(f, 9)
            old_channel = f.channel
            await f.source.send_many(
                acps.request(10, 'session/close', {'sessionId': f.session_id}),
                acps.response(ask['id'], result={
                    'action': 'accept', 'content': {'approve': True}}))
            self.assertIn('error', await asyncio.wait_for(task, 3))
            self.assertEqual(await f.wait_message(lambda m: m.get('id') == 10),
                             acps.response(10, result={}))
            self._assert_no_pin_or_inference(f)
            await f.request(11, 'session/resume', {
                'sessionId': f.session_id, 'cwd': f.workspace})
            f.channel = f.front.workers[f.session_id]
            self.assertIsNot(f.channel, old_channel)
            await f.source.send(acps.request(12, 'session/set_config_option', f.params))
            new_ask = await f.wait_message(
                lambda m: m.get('method') == 'elicitation/create'
                and m['params']['requestId'] == 12)
            self.assertNotEqual(new_ask['id'], ask['id'])
            await f.source.send(acps.response(ask['id'], result={
                'action': 'accept', 'content': {'approve': True}}))
            self.assertFalse(f.front._client_requests[new_ask['id']].done())
            self._assert_no_pin_or_inference(f)
            await f.source.send(acps.response(new_ask['id'], result={'action': 'cancel'}))
            self.assertIn('error', await f.wait_message(lambda m: m.get('id') == 12))
            await self._close_approval_journey(f)

    async def test_pending_approval_eof_and_worker_exit_release_authority(self):
        for ending in ('eof', 'worker exit'):
            with self.subTest(ending=ending):
                async with self._approval_journey() as f:
                    task, _ask = await self._pending_approval(f, 9)
                    if ending == 'worker exit':
                        f.processes[0].close()
                        self.assertIn('error', await asyncio.wait_for(task, 3))
                        await f.source.finish()
                    else:
                        # No response is owed over a disconnected transport.
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        await f.source.finish()
                        self.assertFalse(any(m.get('id') == 9 for m in f.messages))
                    self._assert_no_pin_or_inference(f)
                    await self._assert_approval_released(f)

    async def test_active_prompt_rejects_model_change_before_asking(self):
        async with self._approval_journey() as f:
            entered = asyncio.Event()
            worker = f.processes[0].worker

            async def turn(on_event, reasoning_effort):
                entered.set()
                await worker.cancel_event.wait()
                on_event({'type': 'response_cancelled'})
                return ''

            with mock.patch.object(worker, '_run_turn', side_effect=turn):
                await f.source.send(acps.request(9, 'session/prompt', {
                    'sessionId': f.session_id,
                    'prompt': [{'type': 'text', 'text': 'running turn'}]}))
                await asyncio.wait_for(entered.wait(), 3)
                await f.source.send(acps.request(10, 'session/set_config_option', f.params))
                reply = await f.wait_message(lambda m: m.get('id') == 10)
                self.assertIn('prompt is running', reply['error']['message'])
                self.assertFalse(any(m.get('method') == 'elicitation/create' for m in f.messages))
                self._assert_no_pin_or_inference(f)
                await f.source.send(acps.notification('session/cancel', {'sessionId': f.session_id}))
                reply = await f.wait_message(lambda m: m.get('id') == 9)
                self.assertEqual(reply['result'], {'stopReason': 'cancelled'})
            await self._close_approval_journey(f)

    async def test_client_error_and_duplicate_response_do_not_authorize(self):
        valid_error = {'code': -32800, 'message': 'dismissed'}
        for error, expected in ((valid_error, valid_error), (['invalid error'], {
                'code': acps.INVALID_PARAMS,
                'message': 'ACP client response has an invalid error object'})):
            with self.subTest(error=error):
                async with self._approval_journey() as f:
                    task, ask = await self._pending_approval(f, 9)
                    await f.source.send_many(
                        acps.response(ask['id'], error=error),
                        acps.response(ask['id'], result={
                            'action': 'accept', 'content': {'approve': True}}))
                    reply = await asyncio.wait_for(task, 3)
                    self.assertEqual(reply['error'], expected)
                    self._assert_no_pin_or_inference(f)
                    await self._close_approval_journey(f)

    async def test_changed_pair_shows_the_approved_values(self):
        front = self._front()
        selection = dict(self.PAIR, changed=True,
                         approvedEndpoint="https://old.invalid/v1",
                         approvedCredential="env:OLD_KEY")
        channel = self._channel(front, selection)
        front._request_client = mock.AsyncMock(
            return_value={"action": "accept", "content": {"approve": True}})

        with mock.patch.object(acp.endpoint_pins, "record"):
            await front._approve_config_endpoint(
                channel, {"sessionId": "s"}, 9, owner=front._sessions['s'])

        message = front._request_client.await_args.args[1]["message"]
        self.assertIn("https://old.invalid/v1", message)
        self.assertIn("env:OLD_KEY", message)
        self.assertIn("https://acme.invalid/v1", message)

    async def test_nothing_to_approve_asks_nothing(self):
        front = self._front()
        channel = self._channel(front, {})
        front._request_client = mock.AsyncMock()

        with mock.patch.object(acp.endpoint_pins, "record") as record:
            await front._approve_config_endpoint(
                channel, {"sessionId": "s"}, 9, owner=front._sessions['s'])

        front._request_client.assert_not_awaited()
        record.assert_not_called()

    async def test_client_without_form_elicitation_fails_closed(self):
        front = self._front(supports_elicitation=False)
        channel = self._channel(front, dict(self.PAIR))
        front._request_client = mock.AsyncMock()

        with self.assertRaises(acps.TransportError):
            await front._approve_config_endpoint(
                channel, {"sessionId": "s"}, 9, owner=front._sessions['s'])

        front._request_client.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
