"""Negotiated ACP dialects, insertion ownership, and real v2 wire journeys."""

import asyncio
import copy
import json
import os
import tempfile
import unittest
from unittest import mock

import test_acps as fixtures
from loki_entrypoints import child_environment, configure_container
from loki_agent import acp, acp_events, acp_worker, acps
from loki_agent.credentials import CredentialStore
from loki_agent.sessions import Session
from process_lifecycle_fixtures import observe_fixture


setUpModule = fixtures.setUpModule


class NegotiationTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_dialect_and_failed_reinitialization_preserve_live_capabilities(self):
        front = acp.Front(None, mock.Mock(), CredentialStore({}))
        self.addAsyncCleanup(front.shutdown)
        result = front.initialize({"protocolVersion": 2, "info": {"name": "client", "version": "test"},
                                   "capabilities": {"elicitation": {"form": {}}}})
        self.assertEqual(result["protocolVersion"], 2)
        self.assertEqual(result["capabilities"], {"session": {}})
        self.assertIn("info", result)
        self.assertNotIn("agentCapabilities", result)
        self.assertTrue(front._client_supports_form_elicitation)
        self.assertTrue(front._client_supports_compaction)
        for info in [None, {}, {"name": "client"}, {"name": "client", "version": 1}]:
            with self.subTest(info=info), self.assertRaises(acps.TransportError) as invalid:
                front.initialize({"protocolVersion": 2, "info": info})
            self.assertEqual(invalid.exception.code, acps.INVALID_PARAMS)
            self.assertEqual(front.protocol_version, 2)
            self.assertTrue(front._client_supports_form_elicitation)
        # Unknown future-version fields cannot prevent a valid downgrade.
        self.assertEqual(front.initialize({"protocolVersion": 3, "info": []})["protocolVersion"], 2)
        result = front.initialize({"protocolVersion": 1})
        self.assertIn("agentInfo", result)
        self.assertNotIn("info", result)
        self.assertFalse(front._client_supports_compaction)

    async def test_wire_requires_initialization_and_rejects_invalid_v2_controls_before_dispatch(self):
        messages = []
        front = acp.Front(None, messages.append, CredentialStore({}))
        source = fixtures._FrontInput(front)
        self.addAsyncCleanup(source.close)
        with mock.patch.object(front.credential_supervisor, "delegate", new=mock.AsyncMock()) as delegate:
            await source.send(acps.request(1, "session/new", {"cwd": os.getcwd()}))
            self.assertEqual(messages[-1]["error"]["code"], acps.INVALID_REQUEST)
            delegate.assert_not_awaited()
            await source.send(acps.request(2, "initialize", {
                "protocolVersion": 2, "info": {"name": "client", "version": "test"}}))
            for identifier, method, params, code in [
                [None, "session/new", {}, acps.INVALID_REQUEST],
                [3, "session/load", {}, acps.METHOD_NOT_FOUND],
                [4, "session/resume", {"replayFrom": {"type": "unknown"}}, acps.INVALID_PARAMS],
                [5, "session/set_config_option", {"type": "boolean", "value": True}, acps.INVALID_PARAMS],
                [6, "session/prompt", {"sessionId": "unknown"}, -32002],
            ]:
                await source.send(acps.request(identifier, method, params))
                self.assertEqual(messages[-1]["error"]["code"], code)
            self.assertFalse(front._sessions)
            delegate.assert_not_awaited()
        await source.finish()


class FrontCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_withdrawal_and_session_cancel_reject_only_matching_uninserted_prompts(self):
        for method in ["$/cancel_request", "session/cancel"]:
            with self.subTest(method=method):
                messages = []
                front = acp.Front(None, messages.append, CredentialStore({}))
                self.addAsyncCleanup(front.shutdown)
                front.initialize({"protocolVersion": 2, "info": {"name": "client", "version": "test"}})
                owners = []
                for name in ["one", "two"]:
                    owner = front._reserve_session(name)
                    owner.state = "active"

                    async def request(method, params, forwarded=None, **options):
                        if forwarded is not None:
                            forwarded.set()
                        return {"messageId": params["sessionId"]} if method == "session/prompt" else {}

                    owner.channel = mock.Mock(_closed=False, process=mock.Mock(returncode=None),
                                              request=mock.AsyncMock(side_effect=request), close=mock.AsyncMock())
                    owner.forwarded.clear()
                    owners.append([owner, owner.forwarded])
                for identifier, name in [[1, "one"], [2, "two"]]:
                    front.handle(acps.request(identifier, "session/prompt", {
                        "sessionId": name, "prompt": [{"type": "text", "text": "work"}]}))
                front.handle(acps.notification(method, {"requestId": 1} if method == "$/cancel_request" else {"sessionId": "one"}))
                for owner, blocked in owners:
                    blocked.set()
                await asyncio.gather(*list(front._tasks))
                replies = {message["id"]: message for message in messages if "id" in message}
                self.assertEqual(replies[1]["error"]["code"], -32800)
                self.assertEqual(replies[2]["result"], {"messageId": "two"})
                front.handle(acps.request(3, "session/prompt", {
                    "sessionId": "one", "prompt": [{"type": "text", "text": "later"}]}))
                await asyncio.gather(*list(front._tasks))
                self.assertEqual(messages[-1]["result"], {"messageId": "one"})
                self.assertFalse(front._v2_prompt_cancellations)
                await front.shutdown()


class EventIdentityTests(unittest.TestCase):
    def test_stream_ids_follow_message_boundaries_and_completed_tools_are_explicit(self):
        state = {}

        def update(event):
            packets = acp_events.map_event("s", event, state, protocol_version=2)
            return acp_events.update_v2(packets[0]["update"]) if packets else None

        update({"type": "assistant_start"})
        first = update({"type": "assistant_delta", "content": "one"})
        self.assertEqual(first["messageId"], update({"type": "assistant_delta", "content": "two"})["messageId"])
        update({"type": "assistant_end"})
        update({"type": "assistant_start"})
        self.assertNotEqual(first["messageId"], update({"type": "assistant_delta", "content": "next"})["messageId"])
        update({"type": "reasoning_start"})
        thought = update({"type": "reasoning_delta", "text": "think"})
        self.assertEqual(thought["messageId"], update({"type": "reasoning_delta", "text": "more"})["messageId"])
        update({"type": "reasoning_start"})
        self.assertNotEqual(thought["messageId"], update({"type": "reasoning_delta", "text": "new"})["messageId"])
        started = update({"type": "tool_call", "name": "Read", "call_id": "call"})
        completed = update({"type": "tool_result", "call_id": "call", "content": "read"})
        self.assertEqual(started["sessionUpdate"], "tool_call_update")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(started["toolCallId"], completed["toolCallId"])

    def test_dialect_projection_does_not_mutate_shared_facts(self):
        options = [{"id": "model", "options": [{"group": "models", "options": []}]}]
        original = copy.deepcopy(options)
        projected = acp_events.config_options_v2(options)
        self.assertEqual(projected, [{"configId": "model", "options": [{"groupId": "models", "options": []}]}])
        self.assertEqual(options, original)
        plan = acp_events.update_v2(acp_events.plan("s", [])["update"])
        self.assertEqual(plan, {"sessionUpdate": "plan_update", "plan": {
            "type": "items", "planId": "loki-plan", "entries": []}})
        commands = [{"name": "compact", "description": "Compact", "input": {"hint": "text"}}]
        self.assertNotIn("type", commands[0])
        self.assertEqual(acp_events.commands_v2(commands)[0]["input"]["type"], "text")
        self.assertNotIn("type", commands[0]["input"])


class ChannelInsertionTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_before_foreground_task_starts_publishes_idle_cancelled(self):
        session = Session()
        self.addAsyncCleanup(session.close_state)
        messages = []
        worker = acp_worker.Worker(session, messages.append)
        worker.protocol_version = 2
        await worker.handle(acps.request(1, "session/prompt", {
            "prompt": [{"type": "text", "text": "/compact"}]}))
        await worker.close()
        updates = [message["params"]["update"] for message in messages if message.get("method") == "session/update"]
        self.assertEqual(updates[-1], {"sessionUpdate": "state_update", "state": "idle", "stopReason": "cancelled"})
        self.assertEqual(len([message for message in messages if "result" in message]), 1)
        self.assertIsNone(worker.compaction_requests.pending)

    def channel(self):
        session = Session()
        self.addAsyncCleanup(session.close_state)
        process = fixtures._LocalWorkerProcess(session, "insertion", lambda message: None)
        process.worker.protocol_version = 2
        self.enterContext(mock.patch.object(acp.runtime_isolation, "close_runtime_process", new=lambda process: None))
        channel = acp.WorkerChannel("insertion", process, lambda message: None)
        self.addAsyncCleanup(channel.close)
        return channel

    async def test_withdrawal_while_waiting_to_write_never_inserts(self):
        channel = self.channel()
        await channel._write_lock.acquire()
        entered = asyncio.Event()

        async def send():
            entered.set()
            return await channel.request("session/prompt", {"prompt": [{"type": "text", "text": "/pwd"}]}, cancellation_id=7)

        task = asyncio.create_task(send())
        try:
            await observe_fixture(entered.wait(), task)
            channel.cancel_prompt(7)
        finally:
            channel._write_lock.release()
        with self.assertRaises(acps.TransportError) as cancelled:
            await task
        self.assertEqual(cancelled.exception.code, -32800)
        self.assertEqual(channel.process.methods, [])
        self.assertFalse(channel._cancellable_prompts)

    async def test_close_during_drain_preserves_inserted_prompt_ack(self):
        channel = self.channel()
        process = channel.process
        entered, release = asyncio.Event(), asyncio.Event()
        drain = process.drain

        async def held_drain():
            await drain()
            entered.set()
            await release.wait()

        with mock.patch.object(process, "drain", new=held_drain):
            task = asyncio.create_task(channel.request("session/prompt", {
                "prompt": [{"type": "text", "text": "/pwd"}]}, cancellation_id=7))
            try:
                await observe_fixture(entered.wait(), task)
                process.close()
                task.cancel()
                reply = await task
                self.assertIsInstance(reply["messageId"], str)
                self.assertNotIn("stopReason", reply)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(channel._cancellable_prompts)


class WireV2Tests(fixtures._ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    async def test_real_entrypoint_ack_lifecycle_error_replay_and_v1_selection(self):
        workspace = self.enterContext(tempfile.TemporaryDirectory())
        config = self.enterContext(tempfile.TemporaryDirectory())
        state = self.enterContext(tempfile.TemporaryDirectory())
        env = child_environment(HOME=workspace, PATH=os.environ.get("PATH", ""), TERM="dumb",
                                XDG_CONFIG_HOME=config, XDG_STATE_HOME=state,
                                LOKI_PROVIDER="dummy", LOKI_API_BASE="http://dummy.invalid/v1", LOKI_MODEL="dummy")
        configure_container(env, workspace)
        front, diagnostics = await self._front(env, workspace, cleanup=self.addAsyncCleanup)
        initialized, _ = await self._request(front, 1, "initialize", {
            "protocolVersion": 2, "info": {"name": "wire-test", "version": "test"}, "capabilities": {}})
        self.assertEqual(initialized["protocolVersion"], 2)
        self.assertIn("capabilities", initialized)
        opened, _ = await self._request(front, 2, "session/new", {"cwd": workspace})
        session_id = opened["sessionId"]
        self.assertTrue(all("configId" in option and "id" not in option for option in opened["configOptions"]))
        self.assertTrue(all(command["type"] == "command" for command in opened["availableCommands"]))
        packets = []

        async def idle(preceding):
            seen = list(preceding)
            while True:
                for message in seen:
                    update = message.get("params", {}).get("update", {})
                    if update.get("sessionUpdate") == "state_update" and update.get("state") == "idle" and "stopReason" in update:
                        packets.extend(seen)
                        return update
                line = await front.stdout.readline()
                self.assertTrue(line, "front exited before idle")
                seen.append(json.loads(line))

        identifiers = []
        for request_id, text, reason in [[3, "/compact", "end_turn"], [4, "work", "end_turn"],
                                         [5, "/compact", "error"], [6, "after failure", "end_turn"]]:
            reply, preceding = await self._request(front, request_id, "session/prompt", {
                "sessionId": session_id, "prompt": [{"type": "text", "text": text}]})
            self.assertEqual(list(reply), ["messageId"])
            identifiers.append(reply["messageId"])
            ended = await idle(preceding)
            self.assertEqual(ended["stopReason"], reason)
            if reason == "error":
                self.assertIn("error", ended)
        self.assertEqual(len(set(identifiers)), 4)
        live_users = {packet.get("params", {}).get("update", {}).get("messageId") for packet in packets
                      if packet.get("params", {}).get("update", {}).get("sessionUpdate") == "user_message_chunk"}
        self.assertEqual(live_users, set(identifiers))
        await self._request(front, 7, "session/close", {"sessionId": session_id})
        _, replay = await self._request(front, 8, "session/resume", {
            "sessionId": session_id, "cwd": workspace, "replayFrom": {"type": "start"}})
        replay_updates = [message.get("params", {}).get("update", {}) for message in replay]
        retained = [update for update in replay_updates if update.get("sessionUpdate") == "user_message"]
        self.assertEqual({update["messageId"] for update in retained}, {identifiers[1], identifiers[3]})
        self.assertTrue(all(update["content"] == [] for update in retained))
        self.assertEqual(replay_updates[-1], {"sessionUpdate": "state_update", "state": "idle"})
        await self._finish(front, diagnostics, session_id)
        # An unchanged v1 client never gets upgraded by the dual-version agent.
        v1, diagnostics = await self._front(env, workspace, cleanup=self.addAsyncCleanup)
        initialized, _ = await self._request(v1, 1, "initialize", {"protocolVersion": 1})
        self.assertEqual(initialized["protocolVersion"], 1)
        self.assertIn("agentCapabilities", initialized)
        v1.stdin.close()
        await v1.wait()
        self.assertEqual(v1.returncode, 0, await diagnostics)
