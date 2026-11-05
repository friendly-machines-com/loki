"""One policy across prepared replay, history continuation and finalization."""

import asyncio
import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import unittest
from unittest import mock

import test_acps as acp_fixtures
from loki_entrypoints import child_environment, configure_container
from response_header_fixtures import setUpModule  # noqa: F401 - unittest hook
from settings_fixtures import write_user_settings
from loki_agent import acps, connections, formats, http_client, loki, model_requests
from loki_agent import protocols, recoveries, settings, terminal_frontend, tool_runtime
from loki_agent.acp_worker import Worker
from loki_agent.sessions import Session


def response(text="done", status="completed"):
    value = {
        "object": "response", "id": "r", "model": "test", "status": status,
        "output": [{"type": "message", "id": "m", "role": "assistant", "status": status,
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
    }
    if status == "incomplete":
        value["incomplete_details"] = {"reason": "max_output_tokens"}
    return value


def answer(text="done"):
    return formats.openai_responses_response_to_items(response(text))


def sse_response(event, value):
    return ("data: " + json.dumps({"type": event, "response": value}) + "\n\n").encode()


class TurnRecoveryTests(acp_fixtures._ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.session = Session(shell_cwd=directory.name)
        self.addCleanup(self.session.response_headers.close)
        self.enterContext(mock.patch.object(loki, "_DEFAULT_SESSION", self.session))
        self.stderr = self.enterContext(mock.patch("sys.stderr", new=io.StringIO()))
        # The root test driver supplies this lookup context. Read the shipped
        # standalone policy by its literal key, then test an explicitly supplied
        # temporary copy: no root discovery, rebasing, normalization or joins.
        with open("contrib/hooks/retries.py", encoding="utf-8") as source:
            self.script = self.make_file(source.read())
        # Integration cases exercise retry execution without waiting for either
        # backoff schedule; separate policy tests check the shipped delays.
        self.pipeline = self.policy([
            "--initial-delay", "0", "--overload-initial-delay", "0"])
        self.enterContext(mock.patch.object(loki, "TOOL_HOOK_PIPELINE", self.pipeline))
        self.items = self.session.transcript_items
        self.items.append(formats.message_item("user", "work"))

    def make_file(self, text="", *, directory=None):
        # The resource creator supplies the filename in the selected directory.
        # Close before subprocess use (also on Windows), then remove exactly
        # that supplied key; wire fixtures keep hook code outside the workspace.
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, delete=False) as resource:
            resource.write(text)
            name = resource.name
        self.addCleanup(os.unlink, name)
        return name

    def config(self, value):
        # Fixtures stay dict-shaped for readability; this helper emits the
        # settings-file INI: one section per hook named <phase>.<id>, with
        # indexed options for lists. Hook configuration entries carry a
        # mandatory cwd; tests that do not care about location run in the
        # session directory, and the per-phase location test supplies a
        # distinct directory explicitly.
        for section in ["pre_tool_call", "pre_tool_gate", "post_tool_call",
                        "turn_end", "turn_error"]:
            entries = value.get(section, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                entry.setdefault("cwd", self.session.shell_cwd)
        lines = []
        for section, entries in value.items():
            for entry in entries:
                lines.append(f"[hook.{section}.{entry['id']}]")
                for key, item in entry.items():
                    if key == "id":
                        continue
                    if isinstance(item, list):
                        for index, part in enumerate(item):
                            lines.append(f"{key}.{index} = {part}")
                    elif isinstance(item, bool):
                        lines.append(f"{key} = {'yes' if item else 'no'}")
                    else:
                        lines.append(f"{key} = {item}")
                lines.append("")
        configuration = settings.from_parser(
            settings.read_ini(self.make_file("\n".join(lines))))
        return tool_runtime.build_hook_pipeline(configuration.hooks)

    def policy(self, arguments: list):
        command = [sys.executable, self.script]
        command.extend(arguments)
        return self.config({"turn_error": [{"id": "retry", "command": command}]})

    async def run_loop(self, chat=None, **kwargs):
        event = kwargs.pop("cancel_event", None)
        if event is None:
            event = asyncio.Event()
        recovery = loki.make_turn_recovery(event)
        return await loki.run_tool_loop_async(
            self.items, chat_fn=chat, recovery=recovery, cancel_event=event, **kwargs)

    def observe_end(self):
        self.pipeline.turn_end_hooks.append(object())
        return self.enterContext(mock.patch.object(self.pipeline, "finish_turn", new=mock.AsyncMock()))

    async def serve(self, replies, *, stream=False):
        """Count real generation POSTs/keys; mocks alone cannot prove nesting.

        The fixture owns its listener and connection tasks. Requests contain
        only local test data; all URLs are supplied to the existing config API.
        """
        requests = []
        tasks = set()

        async def handle(reader, writer):
            task = asyncio.current_task()
            tasks.add(task)
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                fields = {}
                for line in headers.splitlines()[1:]:
                    parts = line.partition(b":")
                    fields[parts[0].lower()] = parts[2].strip()
                body = await reader.readexactly(int(fields.get(b"content-length", b"0")))
                requests.append({"headers": fields, "body": json.loads(body)})
                reply = replies[min(len(requests) - 1, len(replies) - 1)]
                # Bind pending-prompt assertions to a request the real worker
                # actually sent, then let the test release its response. This
                # is a controlled peer, not an elapsed quiet-read assumption.
                if reply.get("received") is not None:
                    reply["received"].set()
                if reply.get("release") is not None:
                    await reply["release"].wait()
                content = reply["body"]
                if not isinstance(content, bytes):
                    content = json.dumps(content).encode()
                status = reply.get("status", 200)
                kind = "text/event-stream" if stream else "application/json"
                writer.write((f"HTTP/1.1 {status} Test\r\nContent-Type: {kind}\r\n"
                              f"Content-Length: {len(content)}\r\nConnection: close\r\n\r\n").encode() + content)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                tasks.discard(task)

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        try:
            server = await asyncio.start_server(handle, sock=listener)
        except BaseException:
            listener.close()
            raise

        async def stop():
            server.close()
            await server.wait_closed()
            pending = list(tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        self.addAsyncCleanup(stop)
        self.session.runtime_config = connections.make_runtime_config(
            f"http://127.0.0.1:{port}/responses", protocols.OPENAI_RESPONSES,
            model="test", stream=stream)
        return requests

    def test_policy_does_not_indirectly_parse_its_selected_filename_for_help(self):
        with open(self.script, encoding="utf-8") as source:
            program = source.read()
        output = io.StringIO()
        # A spelling-only scan misses argparse's default basename(argv[0]).
        # Block that operation rather than replacing it with an equivalent one.
        with mock.patch.object(sys, "argv", ["opaque/../selected-policy"]), \
                mock.patch.object(sys, "stdin", io.StringIO('{"event":"turn_error","retry_count":0}')), \
                mock.patch.object(os.path, "basename", side_effect=AssertionError("unexpected pathname parsing")), \
                contextlib.redirect_stdout(output):
            exec(program, {"__name__": "__main__"})
        self.assertEqual(json.loads(output.getvalue()), {"action": "retry", "delay_seconds": 5})

    async def test_shipped_script_backoff_cap_limit_and_fixed_delay(self):
        for arguments, expected in [
                [[], [5, 10, 20, 40, 80, 160, 320, 640, 1280, 1800, 1800]],
                [["--max-retries", "3"], [5, 10, 20, None]],
                [["--max-delay", "60", "--max-retries", "7"],
                 [5, 10, 20, 40, 60, 60, 60, None]],
                [["--multiplier", "1"], [5, 5, 5, 5]]]:
            pipeline = self.policy(arguments)
            self.assertIsNone(pipeline.turn_error_hooks[0].config.timeout_ms)
            for count, delay in enumerate(expected):
                decision = await pipeline.recover_turn({
                    "event": "turn_error", "cwd": self.session.shell_cwd,
                    "retry_count": count, "error_type": "response_incomplete",
                    "incomplete_reason": "max_output_tokens",
                }, cancel_event=asyncio.Event())
                self.assertEqual(decision, recoveries.RetryDecision(delay))
        decision = await self.policy([]).recover_turn({
            "event": "turn_error", "cwd": self.session.shell_cwd,
            "retry_count": 10_000, "error_type": "network_error",
            "error_message": "Temporary failure in name resolution",
        }, cancel_event=asyncio.Event())
        self.assertEqual(decision, recoveries.RetryDecision(1800))

    async def test_shipped_script_overload_backoff_keeps_retrying(self):
        failure = model_requests.failure_event(protocols.ResponseApiError(
            "Our servers are currently overloaded. Please try again later.",
            code="server_is_overloaded", retryable=False))
        fields = recoveries.failure_fields(failure)
        for arguments, expected in [
                [[], [600, 1200, 1800, 1800]],
                [["--overload-initial-delay", "120", "--max-delay", "600"],
                 [120, 240, 480, 600]]]:
            pipeline = self.policy(arguments)
            for count in [0, 1, 2, 3, 10_000]:
                decision = await pipeline.recover_turn({
                    "event": "turn_error", "cwd": self.session.shell_cwd,
                    "retry_count": count, **fields,
                }, cancel_event=asyncio.Event())
                self.assertEqual(decision, recoveries.RetryDecision(
                    expected[min(count, len(expected) - 1)]))

    async def test_intermittent_start_failures_retry_until_success(self):
        request_url = "https://example.test/responses"
        for stream in [False, True]:
            with self.subTest(stream=stream):
                self.session.runtime_config = connections.make_runtime_config(
                    request_url, protocols.OPENAI_RESPONSES, model="test", stream=stream)
                errors = [
                    socket.gaierror(-3, "Temporary failure in name resolution"),
                    ConnectionRefusedError(111, "Connection refused"),
                    ConnectionResetError(104, "Connection reset"),
                    OSError(110, "Connection timed out"),
                ]
                reply = (protocols.ProviderResponse(response()) if stream else
                         http_client.HttpResponse(
                             request_url, 200, "OK", {}, json.dumps(response()).encode()))
                owner = model_requests if stream else http_client
                seam = "_async_chat_stream_request_once" if stream else "_async_http_request_once"
                request = mock.AsyncMock(side_effect=[*errors, reply])
                with mock.patch.object(owner, seam, new=request):
                    self.assertEqual(await self.run_loop(stream_chat=stream), "done")
                self.assertEqual(request.await_count, len(errors) + 1)

    async def test_policy_order_invalid_output_and_failure_finish_original_outcome(self):
        hooks = [{"id": str(index), "command": ["policy"]} for index in range(3)]
        pipeline = self.config({"turn_error": hooks})
        with mock.patch.object(tool_runtime, "_run_hook_command", new=mock.AsyncMock(
                side_effect=[{}, {"action": "retry", "delay_seconds": 0}])) as run:
            self.assertEqual(await pipeline.recover_turn({"cwd": os.getcwd()}, cancel_event=asyncio.Event()), recoveries.RetryDecision(0))
            self.assertEqual(run.await_count, 2)
        invalid = [
            {"action": "retry"}, {"action": "unknown"},
            {"action": "finish", "delay_seconds": 1},
            *[{"action": "retry", "delay_seconds": value}
              for value in [True, "5", -1, float("nan"), float("inf"), 10 ** 400]],
        ]
        for result in [{"action": "finish"}, *invalid]:
            with self.subTest(result=result), mock.patch.object(
                    tool_runtime, "_run_hook_command", new=mock.AsyncMock(return_value=result)) as run:
                self.assertEqual(await pipeline.recover_turn({"cwd": os.getcwd()}, cancel_event=asyncio.Event()), recoveries.RetryDecision(None))
                run.assert_awaited_once()
        with mock.patch.object(tool_runtime, "_run_hook_command", new=mock.AsyncMock(
                side_effect=tool_runtime.HookExecutionError("broken"))):
            self.assertEqual(await pipeline.recover_turn({"cwd": os.getcwd()}, cancel_event=asyncio.Event()), recoveries.RetryDecision(None))
        self.assertIn("broken", self.stderr.getvalue())

    def test_configuration_reuses_turn_hook_restrictions(self):
        for field in ["tools", "on_error", "workspace_side_effects"]:
            with self.subTest(field=field), self.assertRaises(settings.HookConfigurationError):
                self.config({"turn_error": [{"id": "retry", "command": ["policy"], field: True}]})
        for value in [
                {"turn_error": [{"id": "same", "command": ["policy"]}],
                 "turn_end": [{"id": "same", "command": ["notify"]}]}]:
            with self.assertRaises(settings.HookConfigurationError):
                self.config(value)

    async def test_all_hook_phases_run_in_configured_cwd_and_do_not_inherit_secrets(self):
        with tempfile.TemporaryDirectory() as configured:
            for phase in ["pre_tool_call", "pre_tool_gate", "post_tool_call", "turn_end", "turn_error"]:
                with self.subTest(phase=phase):
                    record = self.make_file()
                    command = [sys.executable, "-c", (
                        "import json,os,sys; p=json.load(sys.stdin); i=os.stat('.'); "
                        "p['process_cwd']=[i.st_dev,i.st_ino]; p['secret']=os.getenv('LOKI_API_KEY'); "
                        "json.dump(p,open(sys.argv[1],'w')); print('{}')"), record]
                    pipeline = self.config({phase: [{"id": "record", "command": command,
                                                     "cwd": configured}]})
                    invocation = tool_runtime.ToolInvocation(
                        "call", "Example", {}, {}, {}, {"type": "object"}, cwd=self.session.shell_cwd)
                    expected = os.stat(configured)
                    with mock.patch.dict(os.environ, {"LOKI_API_KEY": "must-not-leak"}):
                        if phase == "turn_end":
                            await pipeline.finish_turn({"event": phase, "cwd": self.session.shell_cwd})
                        elif phase == "turn_error":
                            await pipeline.recover_turn({"event": phase, "cwd": self.session.shell_cwd},
                                                        cancel_event=asyncio.Event())
                        elif phase == "post_tool_call":
                            await pipeline.finish(invocation, tool_runtime.ToolOutcome("success", True, True, "done"))
                        else:
                            await pipeline.prepare(invocation)
                    with open(record, encoding="utf-8") as source:
                        payload = json.load(source)
                    self.assertEqual(payload["process_cwd"], [expected.st_dev, expected.st_ino])
                    self.assertIsNone(payload["secret"])
                    context = payload.get("invocation", payload)
                    self.assertEqual(context["cwd"], self.session.shell_cwd)

    async def test_real_buffered_503_replays_prepared_body_and_key(self):
        requests = await self.serve([
            {"status": 503, "body": {"error": "unavailable"}},
            {"status": 503, "body": {"error": "unavailable"}},
            {"body": response()},
        ])
        self.assertEqual(await self.run_loop(), "done")
        self.assertEqual(len(requests), 3)
        keys = [request["headers"][b"idempotency-key"] for request in requests]
        self.assertEqual(keys, [keys[0]] * 3)
        self.assertEqual([request["body"] for request in requests], [requests[0]["body"]] * 3)

    async def test_real_buffered_invalid_json_replays_prepared_body_and_key(self):
        requests = await self.serve([
            {"body": b"not JSON"},
            {"body": response()},
        ])
        events = []
        self.assertEqual(await self.run_loop(on_event=events.append), "done")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["headers"][b"idempotency-key"],
                         requests[1]["headers"][b"idempotency-key"])
        self.assertEqual(requests[0]["body"], requests[1]["body"])
        self.assertEqual(sum(item.get("type") == "model_response" for item in self.items), 1)
        self.assertFalse(any(event["type"] == "provider_error" for event in events))

    async def test_buffered_invalid_json_limit_is_not_offered_twice(self):
        requests = await self.serve([{"body": b"not JSON"}])
        policy = mock.AsyncMock(side_effect=[
            recoveries.RetryDecision(0), recoveries.RetryDecision(None)])
        events = []
        with mock.patch.object(self.pipeline, "recover_turn", new=policy):
            self.assertEqual(await self.run_loop(on_event=events.append), "")
        self.assertEqual(len(requests), 2)
        self.assertEqual([call.args[0]["retry_count"] for call in policy.await_args_list], [0, 1])
        self.assertEqual(requests[0]["headers"][b"idempotency-key"],
                         requests[1]["headers"][b"idempotency-key"])
        self.assertEqual([event["type"] for event in events], ["provider_error"])
        self.assertIsInstance(events[0]["error"], protocols.ProtocolError)
        self.assertIsInstance(events[0]["error"].__cause__, json.JSONDecodeError)
        self.assertFalse(any(item.get("type") == "model_response" for item in self.items))

    async def test_local_request_error_does_not_run_policy(self):
        for stream in [False, True]:
            with self.subTest(stream=stream):
                self.session.runtime_config = connections.make_runtime_config(
                    "https://example.test/responses", protocols.OPENAI_RESPONSES,
                    model="test", stream=stream)
                owner = model_requests if stream else http_client
                seam = "_async_chat_stream_request_once" if stream else "async_http_request"
                policy = mock.AsyncMock(return_value=recoveries.RetryDecision(0))
                with mock.patch.object(owner, seam, new=mock.AsyncMock(
                        side_effect=RuntimeError("local request bug"))), \
                        mock.patch.object(self.pipeline, "recover_turn", new=policy):
                    with self.assertRaisesRegex(RuntimeError, "local request bug"):
                        await self.run_loop(stream_chat=stream)
                policy.assert_not_awaited()

    async def test_stream_replay_and_incomplete_continuation_share_policy_counter(self):
        failed = {"status": "failed", "error": {"code": "server_is_overloaded", "message": "overloaded"}}
        requests = await self.serve([
            {"body": sse_response("response.failed", failed)},
            {"body": sse_response("response.incomplete", response("partial", "incomplete"))},
            {"body": sse_response("response.completed", response())},
        ], stream=True)
        observed = []
        policy = self.pipeline.recover_turn

        async def observe(payload, *, cancel_event):
            observed.append(payload)
            return await policy(payload, cancel_event=cancel_event)

        end = self.observe_end()
        events = []
        with mock.patch.object(self.pipeline, "recover_turn", new=observe):
            text = await self.run_loop(stream_chat=True, on_event=events.append)
        self.assertEqual(text, "done")
        self.assertEqual([entry["retry_count"] for entry in observed], [0, 1])
        self.assertEqual([entry["error_type"] for entry in observed], ["api_error", "response_incomplete"])
        self.assertEqual(observed[1]["incomplete_reason"], "max_output_tokens")
        self.assertEqual(len(requests), 3)
        keys = [request["headers"][b"idempotency-key"] for request in requests]
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[1], keys[2])
        self.assertEqual(requests[0]["body"], requests[1]["body"])
        self.assertIn("partial", json.dumps(requests[2]["body"]))
        await loki.run_turn_end_hooks_async(events, text)
        end.assert_awaited_once()
        self.assertEqual(end.await_args.args[0]["reason"], "completed")

    async def test_transport_budget_is_not_multiplied_and_decline_is_not_offered_twice(self):
        # Exhaustion requires an explicit bound now that long-running tasks have
        # no default retry limit. Do not make the shipped policy stop for a test.
        self.pipeline = self.policy([
            "--initial-delay", "0", "--overload-initial-delay", "0",
            "--max-retries", "3"])
        self.enterContext(mock.patch.object(loki, "TOOL_HOOK_PIPELINE", self.pipeline))
        for stream in [False, True]:
            with self.subTest(stream=stream):
                await self.serve([{"body": response()}], stream=stream)
                events = []
                counts = []
                policy = self.pipeline.recover_turn

                async def observe(payload, *, cancel_event):
                    counts.append(payload["retry_count"])
                    return await policy(payload, cancel_event=cancel_event)

                seam = "_async_chat_stream_request_once" if stream else "_async_http_request_once"
                owner = model_requests if stream else http_client
                request = mock.AsyncMock(side_effect=ConnectionResetError(104, "reset"))
                with mock.patch.object(owner, seam, new=request), \
                        mock.patch.object(self.pipeline, "recover_turn", new=observe):
                    await self.run_loop(stream_chat=stream, on_event=events.append)
                self.assertEqual(request.await_count, 4)
                self.assertEqual(counts, [0, 1, 2, 3])
                headers = [call.args[2] if stream else call.kwargs["headers_in"]
                           for call in request.await_args_list]
                self.assertEqual([value["Idempotency-Key"] for value in headers],
                                 [headers[0]["Idempotency-Key"]] * 4)
                self.assertEqual([event["type"] for event in events], ["network_error"])

    async def test_raw_stream_failure_starts_new_sampling_without_fabricating_history(self):
        self.session.runtime_config = connections.make_runtime_config(
            "https://example.test/responses", protocols.OPENAI_RESPONSES, model="test", stream=True)
        calls = []

        async def request_once(url, payload, headers, on_text_delta, cancel_check, **kwargs):
            calls.append({"headers": dict(headers), "body": payload})
            if len(calls) == 1:
                on_text_delta("uncommitted transport fragment")
                raise protocols.StreamProtocolError("EOF after output")
            return protocols.ProviderResponse(response())

        with mock.patch.object(model_requests, "_async_chat_stream_request_once", new=request_once):
            self.assertEqual(await self.run_loop(stream_chat=True), "done")
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]["headers"]["Idempotency-Key"], calls[1]["headers"]["Idempotency-Key"])
        self.assertEqual(calls[0]["body"], calls[1]["body"])
        self.assertNotIn("uncommitted transport fragment", json.dumps(self.items))

    async def test_token_caps_are_incomplete_in_each_owning_adapter(self):
        turns = [
            [formats.openai_chat_response_to_items({"choices": [{
                "message": {"role": "assistant", "content": "partial"}, "finish_reason": "length",
            }]}), "length"],
            [formats.anthropic_response_to_items({
                "id": "a", "type": "message", "role": "assistant", "model": "test",
                "stop_reason": "max_tokens", "content": [{"type": "text", "text": "partial"}],
            }), "max_tokens"],
            [formats.openai_responses_response_to_items(response("partial", "incomplete")), "max_output_tokens"],
        ]
        observed = []
        policy = self.pipeline.recover_turn

        async def observe(payload, *, cancel_event):
            observed.append(payload)
            return await policy(payload, cancel_event=cancel_event)

        with mock.patch.object(self.pipeline, "recover_turn", new=observe):
            for turn, reason in turns:
                with self.subTest(reason=reason):
                    self.assertFalse(turn.complete)
                    self.assertEqual(turn.metadata["status"], "incomplete")
                    self.assertEqual(await self.run_loop(mock.AsyncMock(side_effect=[turn, answer()])), "done")
                    self.assertEqual(observed[-1]["incomplete_reason"], reason)
        self.assertEqual([entry["retry_count"] for entry in observed], [0, 0, 0])

    async def test_incomplete_tools_are_closed_without_execution_and_limits_do_not_reset(self):
        call = formats.tool_call_item("unfinished", "Write", raw_arguments='{"file_path":')
        partial = formats.DecodedTurn(
            [formats.message_item("assistant", "partial"), call],
            {"status": "incomplete", "stop_reason": "max_tokens"}, complete=False)
        with mock.patch.object(loki, "execute_tool_call_async", new=mock.AsyncMock()) as execute:
            self.assertEqual(await self.run_loop(mock.AsyncMock(side_effect=[partial, answer()])), "done")
            execute.assert_not_awaited()
        self.assertIn(partial.to_event(), self.items)
        self.assertFalse(formats.pending_tool_calls(self.items))
        for limit in [1, 2]:
            events = []
            chat = mock.AsyncMock(side_effect=OSError("DNS"))
            await self.run_loop(chat, max_loops=limit, on_event=events.append)
            self.assertEqual(chat.await_count, limit)
            self.assertEqual([event["type"] for event in events], ["network_error"])

    async def test_completed_tools_and_retry_count_are_not_reset_by_intermediate_success(self):
        call = formats.tool_call_item("once", "Bash", {"command": "work"})
        chat = mock.AsyncMock(side_effect=[OSError("DNS"), formats.DecodedTurn([call]), OSError("DNS"), answer()])
        seen = []

        async def policy(payload, *, cancel_event):
            seen.append(payload["retry_count"])
            return recoveries.RetryDecision(0)

        with mock.patch.object(self.pipeline, "recover_turn", new=policy), \
                mock.patch.object(loki, "execute_tool_call_async", new=mock.AsyncMock(
                    return_value=[{"ok": True, "content": "ran"}, None])) as execute:
            await self.run_loop(chat)
            execute.assert_awaited_once()
            await self.run_loop(mock.AsyncMock(side_effect=[OSError("DNS"), answer()]))
        self.assertEqual(seen, [0, 1, 0])
        self.assertEqual(sum(item.get("role") == "user" for item in self.items), 1)
        states = [call.kwargs["codex_turn_state"] for call in chat.await_args_list]
        self.assertTrue(all(state is states[0] for state in states))

    async def test_one_cancellation_authority_with_and_without_policy(self):
        for enabled in [False, True]:
            event = asyncio.Event()
            event.set()
            pipeline = self.pipeline if enabled else tool_runtime.ToolHookPipeline()
            chat = mock.AsyncMock(return_value=answer())
            events = []
            with mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline):
                await self.run_loop(chat, cancel_event=event, cancel_check=lambda: False, on_event=events.append)
            chat.assert_not_awaited()
            self.assertEqual([entry["type"] for entry in events], ["response_cancelled"])
        with self.assertRaises(TypeError):
            await loki.run_tool_loop_async(self.items, cancel_event=mock.Mock())

    async def test_cancel_during_backoff_and_simultaneous_completion(self):
        for simultaneous in [False, True]:
            event = asyncio.Event()
            entered = asyncio.Event()
            events = []

            async def policy(payload, *, cancel_event):
                entered.set()
                if simultaneous:
                    event.set()
                return recoveries.RetryDecision(0 if simultaneous else 3600)

            with mock.patch.object(self.pipeline, "recover_turn", new=policy):
                chat = mock.AsyncMock(side_effect=OSError("DNS"))
                task = asyncio.create_task(self.run_loop(chat, cancel_event=event, on_event=events.append))
                await entered.wait()
                event.set()
                await task
            chat.assert_awaited_once()
            self.assertEqual([entry["type"] for entry in events], ["response_cancelled"])

    def test_decoded_failure_reuses_the_stream_classifier_and_wait_hint(self):
        turn = formats.openai_responses_response_to_items({
            "object": "response", "id": "failed", "status": "failed", "output": [], "error": {
                "code": "rate_limit_exceeded", "message": "Try again in 125 ms",
            },
        })
        failure = model_requests.failure_event(turn)
        self.assertEqual(failure["type"], "response_failed")
        self.assertEqual(failure["error"].code, "rate_limit_exceeded")
        self.assertEqual(recoveries.failure_fields(failure)["retry_after_seconds"], .125)

    async def test_provider_minimum_is_not_shortened_by_generated_delay_cap(self):
        event = asyncio.Event()
        ready = asyncio.Event()
        minimum = protocols.ResponseApiError("rate limited", retry_after=3600)
        original_wait = asyncio.wait_for
        delays = []

        async def wait(operation, timeout):
            delays.append(timeout)
            ready.set()
            return await original_wait(operation, timeout)

        recovery = recoveries.TurnRecovery(mock.AsyncMock(return_value=recoveries.RetryDecision(0)), event)
        with mock.patch.object(recoveries.asyncio, "wait_for", new=wait):
            task = asyncio.create_task(recovery.retry({"type": "api_error", "error": minimum}))
            await ready.wait()
            event.set()
            with self.assertRaises(recoveries.Cancelled):
                await task
        self.assertEqual(delays, [3600])

    async def test_real_hook_is_reaped_when_cancelled(self):
        started = asyncio.Event()
        processes = []
        spawn = tool_runtime.host_process.create_exec
        pipeline = self.config({"turn_error": [{"id": "blocked", "command": [
            sys.executable, "-c", "import time; time.sleep(3600)"],
        }]})

        async def capture(executable, option, source, **options):
            process = await spawn(executable, option, source, **options)
            processes.append(process)
            started.set()
            return process

        event = asyncio.Event()
        events = []
        with mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline), \
                mock.patch.object(tool_runtime.host_process, "create_exec", new=capture):
            task = asyncio.create_task(self.run_loop(
                mock.AsyncMock(side_effect=OSError("DNS")), cancel_event=event, on_event=events.append))
            try:
                await started.wait()
                event.set()
                await task
                self.assertEqual([entry["type"] for entry in events], ["response_cancelled"])
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)

    async def test_local_failures_refusal_and_ordinary_tool_errors_do_not_run_policy(self):
        policy = mock.AsyncMock(return_value=recoveries.RetryDecision(0))
        with mock.patch.object(self.pipeline, "recover_turn", new=policy):
            for error in [formats.TranscriptFormatError("local"), RuntimeError("bug")]:
                if isinstance(error, RuntimeError):
                    with self.assertRaises(RuntimeError):
                        await self.run_loop(mock.AsyncMock(side_effect=error))
                else:
                    await self.run_loop(mock.AsyncMock(side_effect=error))
            await self.run_loop(mock.AsyncMock(return_value=formats.DecodedTurn([], {"stop_reason": "refusal"})))
            call = formats.tool_call_item("bad-tool", "Bash", {"command": "work"})
            with mock.patch.object(loki, "execute_tool_call_async", new=mock.AsyncMock(
                    return_value=[{"ok": False, "content": "tool failed"}, None])):
                await self.run_loop(mock.AsyncMock(side_effect=[formats.DecodedTurn([call]), answer()]))
        policy.assert_not_awaited()

    async def acp_wire_recovery(self, *, stream, cancel):
        # Reuse the sanctioned frontend/worker fixture and the real provider
        # server. A peer-held response (success) or observed backoff (cancel)
        # supplies the milestone; a busy-prompt round trip proves the original
        # prompt is still admitted without an elapsed quiet-read assumption.
        received = asyncio.Event()
        release = asyncio.Event()
        completed = (sse_response("response.completed", response())
                     if stream else response())
        requests = await self.serve([
            {"status": 503, "body": {"error": "temporarily unavailable"}},
            {"body": completed, "received": received, "release": release},
        ], stream=stream)
        config = tempfile.TemporaryDirectory()
        state = tempfile.TemporaryDirectory()
        self.addCleanup(config.cleanup)
        self.addCleanup(state.cleanup)
        workspace = self.session.shell_cwd
        witness = self.make_file(directory=workspace)
        with open(self.script, encoding="utf-8") as source:
            script = self.make_file(source.read(), directory=config.name)
        observer = self.make_file(
            "import json,sys\n"
            "payload=json.load(sys.stdin)\n"
            "with open(sys.argv[1], 'a', encoding='utf-8') as output:\n"
            "    output.write(json.dumps(payload) + '\\n')\n"
            "print('{}')\n", directory=config.name)
        observe_command = [sys.executable, observer, witness]
        delay = "1800" if cancel else "0"

        def hook_section(name, command):
            lines = [f"[{name}]"]
            lines.extend(f"command.{index} = {part}"
                         for index, part in enumerate(command))
            lines.append(f"cwd = {config.name}")
            return "\n".join(lines) + "\n"

        configuration_text = "".join([
            hook_section("hook.turn_error.observe-error", observe_command),
            hook_section("hook.turn_error.retry", [
                sys.executable, script, "--initial-delay", delay,
                "--overload-initial-delay", delay]),
            hook_section("hook.turn_end.observe-end", observe_command),
        ])
        write_user_settings(config.name, configuration_text)
        environment = child_environment(
            HOME=config.name, XDG_CONFIG_HOME=config.name,
            XDG_STATE_HOME=state.name, PATH=os.environ.get("PATH", ""),
            TERM="dumb", LOKI_PROVIDER=protocols.OPENAI_RESPONSES,
            LOKI_API_BASE=self.session.runtime_config.chat_provider.chat_url,
            LOKI_MODEL="test", LOKI_STREAM="1" if stream else "0")
        configure_container(environment, workspace)
        backoff = asyncio.Event()

        def on_diagnostics(data):
            if b"Retrying in 1800 seconds." in data:
                backoff.set()

        front, diagnostics = await self._front(
            environment, workspace, on_diagnostics=on_diagnostics)
        await self._request(front, 1, "initialize", {"protocolVersion": 1})
        opened, _ = await self._request(front, 2, "session/new", {"cwd": workspace})
        session_id = opened["sessionId"]
        messages = []
        loop = asyncio.get_running_loop()
        prompt_reply = loop.create_future()
        control_reply = loop.create_future()

        async def read_stdout():
            # Sole stdout reader from prompt admission through EOF, retaining
            # errors and late replies. Readiness is raced against this actual
            # producer and an early final reply, not an unowned Event alone.
            async for line in front.stdout:
                message = json.loads(line)
                messages.append(message)
                if message.get("id") == 3 and not prompt_reply.done():
                    prompt_reply.set_result(message)
                elif message.get("id") == 4 and not control_reply.done():
                    control_reply.set_result(message)

        reader = asyncio.create_task(read_stdout())
        milestone = asyncio.create_task(backoff.wait() if cancel else received.wait())

        async def reply(pending):
            await asyncio.wait([pending, reader], return_when=asyncio.FIRST_COMPLETED)
            if not pending.done():
                await reader
                self.fail("ACP frontend exited before replying")
            return pending.result()

        params = {"sessionId": session_id,
                  "prompt": [{"type": "text", "text": "retry wire probe"}]}
        try:
            front.stdin.write((json.dumps(acps.request(3, "session/prompt", params)) + "\n").encode())
            await front.stdin.drain()
            await asyncio.wait([milestone, prompt_reply, reader], return_when=asyncio.FIRST_COMPLETED)
            if reader.done():
                await reader
                self.fail("ACP frontend exited while recovery was pending")
            self.assertFalse(prompt_reply.done(), messages)
            self.assertTrue(milestone.done())
            front.stdin.write((json.dumps(acps.request(4, "session/prompt", params)) + "\n").encode())
            await front.stdin.drain()
            busy = await reply(control_reply)
            self.assertEqual(busy["error"]["code"], acps.INVALID_PARAMS)
            self.assertIn("already running", busy["error"]["message"])
            self.assertFalse(prompt_reply.done(), messages)
            if cancel:
                front.stdin.write((json.dumps(acps.notification(
                    "session/cancel", {"sessionId": session_id})) + "\n").encode())
                await front.stdin.drain()
            else:
                release.set()
            final = await reply(prompt_reply)
            self.assertEqual(final["result"]["stopReason"], "cancelled" if cancel else "end_turn")
            front.stdin.close()
            await front.stdin.wait_closed()
            await reader
            await front.wait()
            errors = await diagnostics
            self.assertEqual(front.returncode, 0, errors.decode(errors="replace"))
        finally:
            release.set()
            for task in [milestone, reader]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(milestone, reader, return_exceptions=True)
            for pending in [prompt_reply, control_reply]:
                if not pending.done():
                    pending.cancel()
        self.assertEqual(sum(message.get("id") == 3 for message in messages), 1)
        self.assertEqual(len(requests), 1 if cancel else 2)
        if not cancel:
            self.assertEqual(requests[0]["headers"][b"idempotency-key"],
                             requests[1]["headers"][b"idempotency-key"])
            self.assertEqual(requests[0]["body"], requests[1]["body"])
        with open(witness, encoding="utf-8") as source:
            observations = [json.loads(line) for line in source]
        self.assertEqual([item["event"] for item in observations], ["turn_error", "turn_end"])
        self.assertEqual(observations[0]["retry_count"], 0)
        self.assertEqual(observations[1]["reason"], "cancelled" if cancel else "completed")

    async def test_acp_wire_buffered_retry_finishes_once(self):
        await self.acp_wire_recovery(stream=False, cancel=False)

    async def test_acp_wire_streamed_retry_finishes_once(self):
        await self.acp_wire_recovery(stream=True, cancel=False)

    async def test_acp_wire_buffered_cancel_during_backoff_finishes_once(self):
        await self.acp_wire_recovery(stream=False, cancel=True)

    async def test_acp_wire_streamed_cancel_during_backoff_finishes_once(self):
        await self.acp_wire_recovery(stream=True, cancel=True)

    async def test_terminal_and_acp_use_same_owner_and_notify_final_outcome_once(self):
        end = self.observe_end()
        events = []
        with mock.patch.object(terminal_frontend, "async_chat_completion", new=mock.AsyncMock(
                side_effect=[OSError("DNS"), answer()])), \
                mock.patch.object(terminal_frontend, "_redraw_status"), \
                mock.patch.object(terminal_frontend, "_terminal_agent_event"):
            text = await terminal_frontend.run_terminal_turn_async(
                self.items, cancel_event=asyncio.Event(), turn_events=events)
        await loki.run_turn_end_hooks_async(events, text)
        end.assert_awaited_once()
        self.assertEqual(end.await_args.args[0]["reason"], "completed")
        for cancel in [False, True]:
            with self.subTest(cancel=cancel):
                end.reset_mock()
                entered = asyncio.Event()
                messages = []
                worker = Worker(self.session, messages.append, "s")

                async def policy(payload, *, cancel_event):
                    entered.set()
                    return recoveries.RetryDecision(3600 if cancel else 0)

                with mock.patch.object(self.pipeline, "recover_turn", new=policy), \
                        mock.patch.object(loki, "current_model", return_value="test"), \
                        mock.patch.object(loki, "async_chat_completion", new=mock.AsyncMock(
                            side_effect=[OSError("DNS"), answer()])), \
                        mock.patch.object(worker, "_ask_advertised_tools", return_value=[[], None]), \
                        mock.patch.object(worker, "_publish_context_usage"), mock.patch.object(loki, "save_chat_log"):
                    params = {"sessionId": "s", "prompt": [{"type": "text", "text": "work"}]}
                    task = asyncio.create_task(worker._answer({"id": 7, "method": "session/prompt", "params": params}))
                    await entered.wait()
                    self.assertFalse(any(message.get("id") == 7 for message in messages))
                    if cancel:
                        with self.assertRaisesRegex(acps.TransportError, "already running"):
                            await worker.prompt(params)
                        await worker.dispatch("session/cancel", {"sessionId": "s"})
                    await task
                replies = [message for message in messages if message.get("id") == 7]
                self.assertEqual(len(replies), 1)
                self.assertEqual(replies[0]["result"]["stopReason"], "cancelled" if cancel else "end_turn")
                end.assert_awaited_once()
                self.assertEqual(end.await_args.args[0]["reason"], "cancelled" if cancel else "completed")


if __name__ == "__main__":
    unittest.main()
