"""Manual compaction invariants; provider I/O is synthetic unless explicitly stated."""

import asyncio
import contextlib
import copy
import errno
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

from process_lifecycle_fixtures import observe_fixture
from loki_agent import acp_commands, acp_events, acp_worker, acps, authentications, compactions, connections, formats
from loki_agent import http_client, loki, model_requests, openai_models, protocols, recoveries
from loki_agent import replays, response_headers, savefiles, settings, sse, terminal_frontend, terminals, tool_runtime, usages
from loki_agent.sessions import Session


class CompactionFormatTests(unittest.TestCase):
    def checkpoint(self, protocol=formats.OPENAI_RESPONSES):
        item = {"type": "compaction", "encrypted_content": "opaque-sentinel", "extra": {"kept": [1, 2]}}
        return formats.model_response_event(protocol, [item], provider="native", endpoint="https://native.test/responses", model="model")

    def history(self):
        call = formats.tool_call_item("call-one", "Read", {"file_path": "synthetic"})
        return [formats.message_item("system", "saved instructions"),
                formats.message_item("user", "original user"),
                formats.model_response_event(formats.OPENAI_RESPONSES, [call]),
                formats.tool_result_for_call(call, "old tool body")]

    def test_zero_one_two_replacements_copy_containers_and_preserve_archive(self):
        archive = self.history()
        ordinary = formats.active_events(archive)
        self.assertEqual(ordinary, archive)
        self.assertIsNot(ordinary, archive)
        window = archive[:2] + [self.checkpoint()]
        record = formats.context_replace_event(window)
        window.append(formats.message_item("user", "not persisted"))
        archive.append(record)
        suffix = formats.message_item("user", "new suffix")
        archive.append(suffix)
        selected = formats.active_events(archive)
        self.assertEqual(selected, record["events"] + [suffix])
        selected.append(formats.message_item("user", "not installed"))
        self.assertEqual(len(record["events"]), 3)
        second = formats.context_replace_event([archive[0], suffix, self.checkpoint()])
        archive.append(second)
        self.assertEqual(formats.active_events(archive), second["events"])
        self.assertIn("old tool body", json.dumps(archive))
        self.assertEqual(formats.pending_tool_calls(archive), [])
        blob = formats.new_log_blob(archive, [])
        self.assertEqual(blob["schema"], "day-agent.session.v5")
        loaded, _todos = formats.load_log_blob(json.loads(json.dumps(blob)))
        self.assertEqual(formats.active_events(loaded), formats.active_events(archive))
        self.assertEqual(formats.new_log_blob(self.history(), [])["schema"], "day-agent.session.v4")
        with self.assertRaisesRegex(formats.TranscriptFormatError, "v5"):
            formats.load_log_blob({**blob, "schema": "day-agent.session.v4"})

    def test_invalid_or_unsettled_snapshots_are_rejected(self):
        call = formats.tool_call_item("pending", "Read", {})
        for window in [None, [None], [{"type": "context_replace", "events": []}],
                       [{"type": "message", "role": "root", "content": []}],
                       [formats.model_response_event(formats.OPENAI_RESPONSES, [{"type": "context_replace", "events": []}])],
                       [formats.model_response_event(formats.OPENAI_RESPONSES, [call])],
                       [formats.tool_result_item("missing", "orphan")]]:
            with self.subTest(window=window), self.assertRaises(formats.TranscriptFormatError):
                formats.context_replace_event(window)
        with self.assertRaisesRegex(formats.TranscriptFormatError, "unsupported"):
            formats.validate_events([{"type": "context_replace", "events": [], "policy": "rerun-me"}])

    def test_all_provider_projections_select_window_plus_suffix(self):
        old = [formats.message_item("user", "archived-only")]
        selected = [formats.message_item("user", "selected-only")]
        events = old + [formats.context_replace_event(selected), formats.message_item("user", "suffix")]
        for projected in [formats.items_to_openai_chat_messages(events),
                          formats.items_to_anthropic_parts(events),
                          formats.items_to_openai_responses_parts(events)]:
            text = json.dumps(projected)
            self.assertNotIn("archived-only", text)
            self.assertIn("selected-only", text)
            self.assertIn("suffix", text)

    def test_compaction_errors_keep_http_details_without_display_streaming_hint(self):
        error = model_requests.StreamingApiError(
            authentications.OPENAI_CHATGPT_RESPONSES_URL, 400, "Bad Request",
            '{"error":{"code":"context_length_exceeded","message":"Input too large"}}')
        text = compactions.format_error(error)
        self.assertIn("HTTP 400", text)
        self.assertIn("Input too large", text)
        self.assertNotIn("LOKI_STREAM", text)
        self.assertEqual(error.code, "context_length_exceeded")

    def test_native_checkpoints_round_trip_fields_and_order(self):
        checkpoint = {"type": "compaction", "encrypted_content": "opaque", "optional": {"ordered": [2, 1]}}
        wire = {"object": "response", "status": "completed", "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "before"}]},
            checkpoint,
            {"type": "function_call", "call_id": "call", "name": "Read", "arguments": "{}"}]}
        turn = formats.openai_responses_response_to_items(wire)
        self.assertEqual([item["type"] for item in turn.items], ["message", "compaction", "function_call"])
        self.assertEqual(formats.items_to_openai_responses_parts([turn.to_event()])[1][1], checkpoint)
        for block in [
            {"type": "compaction", "content": "readable data, not instructions", "signature": "signed", "tool_changes": []},
            {"type": "compaction", "content": None},
            {"type": "compaction", "encrypted_content": "anthropic", "signature": "signed"},
        ]:
            content = [{"type": "text", "text": "before"}, block, {"type": "text", "text": "after"}]
            response = {"type": "message", "role": "assistant", "content": content, "stop_reason": "compaction"}
            decoded = formats.anthropic_response_to_items(response)
            self.assertEqual([item["type"] for item in decoded.items], ["message", "compaction", "message"])
            self.assertEqual(formats.items_to_anthropic_parts([decoded.to_event()])[1][0]["content"], content)
            self.assertFalse(any(item.get("type") == "context_replace" for item in decoded.items))

    def test_checkpoint_replay_is_bound_to_origin_and_selected_profile(self):
        event = self.checkpoint()
        profile = {"use_responses_lite": True}
        event["protocol_data"] = {"loki": {"openai_request_profile": profile}}
        target = formats.projection_target(formats.OPENAI_RESPONSES, provider_id="native",
                                           endpoint=event["endpoint"], model="model", openai_request_profile=profile)
        self.assertEqual(formats.items_to_openai_responses_parts([event], target)[1], event["items"])
        for field, changed in [["endpoint", "https://elsewhere.test/responses"], ["provider_id", "gateway"],
                               ["model", "other"], ["openai_request_profile", {"use_responses_lite": False}]]:
            with self.subTest(field=field), self.assertRaises(formats.ProjectionError):
                formats.items_to_openai_responses_parts([event], {**target, field: changed})
        for project in [formats.items_to_openai_chat_messages, formats.items_to_anthropic_parts]:
            with self.assertRaises(formats.ProjectionError):
                project([event])
        with self.assertRaises(formats.ProjectionError):
            formats.items_to_openai_responses_parts([self.checkpoint(formats.ANTHROPIC_MESSAGES)])

    def test_anthropic_complete_start_and_delta_compaction_assembly(self):
        expected = {"type": "compaction", "content": "summary", "encrypted_content": "opaque", "signature": "sig", "extra": [2, 1]}
        for complete_start in [True, False]:
            with self.subTest(complete_start=complete_start):
                start = copy.deepcopy(expected) if complete_start else {"type": "compaction", "content": None, "signature": "sig", "extra": [2, 1]}
                events = [{"type": "message_start", "message": {"type": "message", "role": "assistant"}},
                          {"type": "content_block_start", "index": 0, "content_block": start}]
                if not complete_start:
                    events.extend([
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "compaction_delta", "content": "sum", "encrypted_content": "op"}},
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "compaction_delta", "content": "mary", "encrypted_content": "aque"}},
                    ])
                events.extend([{"type": "content_block_stop", "index": 0}, {"type": "message_stop"}])
                accumulator = protocols.AnthropicMessagesStreamAccumulator(lambda text: self.fail("checkpoint is not assistant text"))
                decoder = sse.SseDecoder()
                for data in events:
                    for event in decoder.feed(("data: " + json.dumps(data) + "\n\n").encode()):
                        accumulator.feed(event)
                self.assertEqual(accumulator.finish()["content"], [expected])

    def test_replacement_clears_usage_without_rendering_nested_checkpoint(self):
        archive = self.history()
        observation = formats.model_response_event(
            formats.OPENAI_RESPONSES, [], provider="native",
            endpoint="https://native.test/responses", model="model",
            usage={"input_tokens": 100, "output_tokens": 10})
        archive.append(observation)
        checkpoint = self.checkpoint()
        checkpoint["usage"] = {"input_tokens": 99999, "output_tokens": 99999}
        record = formats.context_replace_event(archive[:2] + [checkpoint])
        identity = (formats.OPENAI_RESPONSES, "native", "https://native.test/responses", "model")
        tracker = usages.ContextTracker()
        self.assertEqual(tracker.snapshot(archive, identity, None, live=True).usage.used, 110)
        archive.append(record)
        self.assertIsNone(tracker.snapshot(archive, identity, None, live=True).usage)
        self.assertIsNone(usages.ContextTracker().snapshot(copy.deepcopy(archive), identity, None).usage)
        self.assertIsNone(acp_events.context_usage("session", tracker.snapshot(archive, identity, None)))
        archive.append(observation)
        self.assertEqual(tracker.snapshot(archive, identity, None, live=True).usage.used, 110)
        rendered = str(savefiles.ResumeTranscriptRenderer().presentation(archive))
        classified = str([replays.classify_event(event) for event in archive])
        for text in [rendered, classified]:
            self.assertEqual(text.count("original user"), 1)
            self.assertIn("Context compacted", text)
            self.assertNotIn("opaque-sentinel", text)
            self.assertNotIn("99999", text)


class ManualCompactionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as log:
            log_path = log.name
        with tempfile.NamedTemporaryFile(dir=directory, delete=False, mode="w", encoding="utf-8") as observed:
            observation_path = observed.name
            json.dump({"version": 1, "endpoints": []}, observed)
        store = self.enterContext(contextlib.closing(response_headers.Store(observation_path)))
        self.broker = authentications.CredentialBroker()
        self.broker.install_openai_subscription(authentications.OpenAITokenSet(
            access_token="access-sentinel", refresh_token="refresh-sentinel", expires_at=10**12, last_refresh=10**12))
        profile = openai_models.CodexModelRequestProfile.from_catalog_model({
            "use_responses_lite": True, "supports_parallel_tool_calls": True,
            "supports_reasoning_summaries": True, "default_reasoning_level": "low", "default_reasoning_summary": "none"})
        config = connections.make_runtime_config(
            authentications.OPENAI_CHATGPT_RESPONSES_URL, formats.OPENAI_RESPONSES,
            model="compaction-test", provider_id="openai-subscription",
            credential_ref=authentications.CredentialRef.openai_subscription(), openai_request_profile=profile, stream=True,
            context_capacity=usages.ContextCapacity(1000, "configured"))
        self.session = Session(shell_cwd=directory, response_headers=store, runtime_config=config,
                               credential_authority=self.broker, chat_log_path=log_path)
        call = formats.tool_call_item("settled-call", "Read", {"file_path": "synthetic-only"})
        self.session.transcript_items = [
            formats.message_item("system", "saved system"),
            formats.message_item("user", "saved user"),
            formats.model_response_event(formats.OPENAI_RESPONSES, [call]),
            formats.tool_result_for_call(call, "uncommented-tool-fact")]
        self.session.session_todos = [{"content": "keep todo", "status": "pending", "priority": "high"}]
        self.session.file_observations["read-before-write-key"] = "keep observation"
        self.event = asyncio.Event()
        self.pipeline = tool_runtime.ToolHookPipeline()
        self.enterContext(mock.patch.object(loki, "_DEFAULT_SESSION", self.session))
        self.enterContext(mock.patch.object(loki, "TOOL_HOOK_PIPELINE", self.pipeline))
        # The test supplies the real sanctioned bootstrap as an opaque operand,
        # not unittest's dispatcher, a normalized ROOT, or python -m reentry.
        self.enterContext(mock.patch.object(
            loki, "_tool_bootstrap_command",
            return_value=[os.environ.get("LOKI_LAUNCHER", "./loki.py"), "--tool-exec"]))
        self.sent = []
        self.headers = []
        self.ordinary_outputs = []
        self.end_turns = []
        self.idle_routing = True
        self.mutate = None
        self.mode = "valid"
        self.before_archive = self.session.transcript_items
        self.before = copy.deepcopy(self.before_archive)
        self.enterContext(mock.patch.object(http_client, "async_http_stream", new=self.streaming))
        self.addAsyncCleanup(self.session.close_state)

    @contextlib.asynccontextmanager
    async def streaming(self, method, url, **kwargs):
        self.assertEqual(method, "POST")
        self.assertEqual(url, authentications.OPENAI_CHATGPT_RESPONSES_URL)
        self.assertEqual(kwargs["headers_in"]["Authorization"], "Bearer access-sentinel")
        self.assertEqual(kwargs["headers_in"][protocols.RESPONSES_LITE_HEADER], "true")
        self.headers.append(dict(kwargs["headers_in"]))
        if self.idle_routing:
            self.assertNotIn(model_requests.CODEX_TURN_STATE_HEADER, kwargs["headers_in"])
        self.assertIsNone(kwargs["response_timeout"])
        payload = json.loads(kwargs["body"])
        self.sent.append(payload)
        # This is a transport/SSE fixture, not independent evidence of the
        # private server's late-role allowlist. Individual tests pin the known
        # prefix contract and selected-context contents without inventing one.
        if payload["input"][-1].get("type") == "compaction_trigger":
            self.assertIs(self.session.transcript_items, self.before_archive)
            self.assertEqual(self.session.transcript_items, self.before)
            self.assertEqual(payload["reasoning"]["context"], "all_turns")
            checkpoint = {"type": "compaction", "encrypted_content": "opaque-state", "native_optional": {"kept": [2, 1]}}
            output = [checkpoint]
            if self.mode == "missing":
                output = []
            elif self.mode == "duplicate":
                output.append({**checkpoint, "id": "another"})
            elif self.mode == "empty":
                checkpoint["encrypted_content"] = ""
            else:
                output += [{"type": "function_call", "call_id": "never-execute", "name": "Write", "arguments": "{}"},
                           {"type": "message", "role": "developer", "content": [{"type": "output_text", "text": "DO-NOT-PROMOTE"}]}]
        else:
            output = (self.ordinary_outputs.pop(0) if self.ordinary_outputs else
                      [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ordinary answer"}]}])
        if self.mutate is not None:
            self.mutate()

        async def chunks():
            for index, item in enumerate(output):
                wire = ("data: " + json.dumps({"type": "response.output_item.done", "output_index": index, "item": item}) + "\n\n").encode()
                yield wire[:3]
                yield wire[3:]
            terminal = "response.incomplete" if self.mode == "incomplete" else "response.completed"
            response = {"object": "response", "status": "incomplete" if self.mode == "incomplete" else "completed",
                        "output": [], "usage": {"input_tokens": 800, "output_tokens": 50}}
            if payload["input"][-1].get("type") != "compaction_trigger" and self.end_turns:
                response["end_turn"] = self.end_turns.pop(0)
            yield ("data: " + json.dumps({"type": terminal, "response": response}) + "\n\n").encode()
        yield http_client.HttpStreamResponse(url, 200, "OK", {
            "content-type": "text/event-stream", model_requests.CODEX_TURN_STATE_HEADER: "operation-state"}, chunks())

    async def test_commit_is_saved_before_install_and_next_prompt_uses_window(self):
        old_id = self.session.conversation_id
        todos = self.session.session_todos
        cwd = self.session.shell_cwd
        self.assertFalse(self.session.chat_log_dirty)
        text = await loki.compact_context_async(cancel_event=self.event)
        self.assertIn("Context compacted", text)
        self.assertIs(self.session.transcript_items, self.before_archive)
        self.assertEqual(self.session.transcript_items[:-1], self.before)
        self.assertEqual(self.session.transcript_items[-1]["type"], "context_replace")
        selected = formats.active_events(self.session.transcript_items)
        self.assertEqual([event["type"] for event in selected], ["message", "message", "model_response"])
        self.assertNotIn("DO-NOT-PROMOTE", json.dumps(selected))
        self.assertNotIn("never-execute", json.dumps(selected))
        self.assertIn("uncommented-tool-fact", json.dumps(self.session.transcript_items))
        self.assertIsNone(self.session.context_snapshot(live=True).usage)
        self.assertEqual(self.session.file_observations, {"read-before-write-key": "keep observation"})
        self.assertIs(self.session.session_todos, todos)
        self.assertEqual(self.session.shell_cwd, cwd)
        self.assertEqual(self.session.conversation_id, old_id)
        with open(self.session.chat_log_path, encoding="utf-8") as stream:
            loaded, loaded_todos, _state, _toolsets = savefiles.read_chat_log(stream)
        self.assertEqual(loaded_todos, todos)
        self.assertEqual(formats.active_events(loaded), selected)
        self.session.transcript_items.append(formats.message_item("user", "next ordinary prompt"))
        turn = await model_requests.async_chat_completion(
            self.session.transcript_items, None, context=loki._model_request_context(),
            thinking=loki.capture_turn_settings())
        self.assertEqual(formats.item_text(turn.to_event()), "ordinary answer")
        self.assertEqual(len(self.sent), 2)
        self.assertNotIn("uncommented-tool-fact", json.dumps(self.sent[-1]))
        self.assertIn("opaque-state", json.dumps(self.sent[-1]))
        self.assertIn("additional_tools", [item.get("type") for item in self.sent[0]["input"]])
        self.assertEqual(self.sent[0]["input"][-1], {"type": "compaction_trigger"})

    async def test_startup_noop_preserves_first_mode_prefix_without_any_publication(self):
        self.session.transcript_items = [formats.message_item("system", "saved initial instructions")]
        self.before_archive = self.session.transcript_items
        self.before = copy.deepcopy(self.before_archive)
        with mock.patch.object(loki, "save_chat_log") as save, \
                mock.patch.object(self.pipeline, "prepare_compaction", new=mock.AsyncMock()) as policy:
            self.assertEqual(await loki.compact_context_async(cancel_event=self.event), "Nothing to compact.")
            save.assert_not_called()
            policy.assert_not_called()
        self.assertEqual(self.sent, [])
        self.assertIsNone(self.session.last_instructed_agent_mode)
        self.assertEqual(self.session.transcript_items, self.before)
        loki.record_agent_mode_instruction()
        self.session.transcript_items.append(formats.message_item("user", "first normal prompt"))
        before_request = copy.deepcopy(self.session.transcript_items)
        turn = await model_requests.async_chat_completion(
            self.session.transcript_items, None, context=loki._model_request_context(), thinking=loki.capture_turn_settings())
        self.assertEqual(formats.item_text(turn.to_event()), "ordinary answer")
        self.assertEqual([item.get("type") for item in self.sent[-1]["input"]], ["message", "message"])
        self.assertEqual([item.get("role") for item in self.sent[-1]["input"]], ["developer", "user"])
        self.assertIn("Normal mode is active", self.sent[-1]["input"][0]["content"][0]["text"])
        self.assertEqual(self.session.transcript_items, before_request)
        count = len(self.session.transcript_items)
        loki.record_agent_mode_instruction()
        self.assertEqual(len(self.session.transcript_items), count)
        loaded, _todos = formats.load_log_blob(formats.new_log_blob(self.session.transcript_items, []))
        replayed = self.session.runtime_config.chat_provider.streaming_chat_payload(loaded, None, self.session.model)
        self.assertEqual(replayed["input"], self.sent[-1]["input"])
        # Actual work, unlike startup, is eligible for compaction. Its saved
        # instruction marker/prefix stays unchanged; no late-role repair exists.
        self.session.transcript_items.append(turn.to_event())
        self.before = copy.deepcopy(self.before_archive)
        await loki.compact_context_async(cancel_event=self.event)
        self.assertEqual(self.sent[-1]["input"][-1], {"type": "compaction_trigger"})
        self.assertEqual(self.session.last_instructed_agent_mode, "normal")

    async def test_late_local_roles_are_preserved_not_universally_rewritten(self):
        history = [formats.message_item("system", "prefix instructions"), formats.message_item("user", "earlier user"),
                   formats.model_response_event(formats.OPENAI_RESPONSES, [formats.message_item("assistant", "earlier answer")]),
                   formats.message_item("system", "late cwd or mode instruction"),
                   formats.message_item("developer", "late developer instruction"), formats.message_item("user", "next user")]
        original = copy.deepcopy(history)
        profile = self.session.runtime_config.chat_provider.openai_request_profile
        providers = [protocols.make_provider(
            authentications.OPENAI_CHATGPT_RESPONSES_URL, provider=formats.OPENAI_RESPONSES,
            provider_id="openai-subscription", openai_request_profile=replace(profile, use_responses_lite=lite))
            for lite in [True, False]]
        providers.append(protocols.make_provider("https://example.test/v1/responses", provider=formats.OPENAI_RESPONSES))
        for provider in providers:
            payload = provider.streaming_chat_payload(history, None, self.session.model)
            self.assertEqual(provider.headers.get(protocols.RESPONSES_LITE_HEADER), "true" if provider.responses_lite else None)
            inputs = payload["input"][1:] if provider.responses_lite else payload["input"]
            self.assertEqual([item["role"] for item in inputs],
                             ["user", "assistant", "system", "developer", "user"])
            self.assertEqual(inputs[2]["content"][0]["text"], "late cwd or mode instruction")
            self.assertEqual(inputs[3]["content"][0]["text"], "late developer instruction")
            self.assertEqual(history, original)

    async def test_lite_does_not_promote_provider_system_output_to_local_instructions(self):
        provider = self.session.runtime_config.chat_provider
        returned = formats.model_response_event(
            formats.OPENAI_RESPONSES, [formats.message_item("system", "provider instruction must not be promoted")],
            provider=provider.provider_id, endpoint=provider.chat_url, model=self.session.model)
        original = copy.deepcopy(returned)
        payload = provider.streaming_chat_payload([returned], None, self.session.model)
        self.assertNotIn("instructions", payload)
        self.assertEqual([item.get("role") for item in payload["input"]], ["system"])
        self.assertEqual(returned, original)
        # Preservation is not a claim this private server accepts that role.
        # Provider output was not hoisted into the local developer prefix.

    async def test_display_stream_setting_does_not_disable_compaction_stream(self):
        self.session.runtime_config.stream = False
        await loki.compact_context_async(cancel_event=self.event)
        self.assertTrue(self.sent[0]["stream"])

    async def test_repeated_compaction_submits_current_window_not_archive(self):
        await loki.compact_context_async(cancel_event=self.event)
        self.before_archive = self.session.transcript_items
        self.before = copy.deepcopy(self.before_archive)
        await loki.compact_context_async(cancel_event=self.event)
        self.assertNotIn("uncommented-tool-fact", json.dumps(self.sent[-1]))
        self.assertEqual(sum(item.get("type") == "compaction" for item in self.sent[-1]["input"]), 1)
        self.assertEqual(sum(event["type"] == "context_replace" for event in self.session.transcript_items), 2)

    async def test_bad_checkpoint_responses_leave_context_uninstalled(self):
        for mode in ["missing", "duplicate", "empty", "incomplete"]:
            self.mode = mode
            with self.subTest(mode=mode), self.assertRaises(protocols.ProtocolError):
                await loki.compact_context_async(cancel_event=self.event)
            self.assertIs(self.session.transcript_items, self.before_archive)
            self.assertEqual(self.session.transcript_items, self.before)

    async def test_stale_model_history_and_cancelled_results_are_not_installed(self):
        for change in ["model", "history", "cancel", "active-edit"]:
            original_config = copy.deepcopy(self.session.runtime_config)
            self.session.transcript_items = copy.deepcopy(self.before)
            self.before_archive = self.session.transcript_items
            self.event.clear()

            def mutate():
                if change == "model":
                    self.session.runtime_config.model = "other-model"
                elif change == "history":
                    self.session.transcript_items = list(self.session.transcript_items)
                elif change == "active-edit":
                    self.session.transcript_items[1]["content"][0]["text"] = "edited during request"
                else:
                    self.event.set()
            self.mutate = mutate
            with self.subTest(change=change), self.assertRaises(Exception) as failed:
                await loki.compact_context_async(cancel_event=self.event)
            self.assertIsInstance(failed.exception, compactions.CompactionError if change != "cancel" else recoveries.Cancelled)
            self.assertFalse(any(event["type"] == "context_replace" for event in self.session.transcript_items))
            self.session.runtime_config = original_config
        self.mutate = None

    async def test_unsaveable_or_unsettled_context_is_rejected_before_billing(self):
        with mock.patch.object(loki.files, "atomic_write_text", side_effect=OSError("cannot save")):
            with self.assertRaises(OSError):
                await loki.compact_context_async(cancel_event=self.event)
        self.assertEqual(self.sent, [])
        self.session.transcript_items.append(formats.model_response_event(
            formats.OPENAI_RESPONSES, [formats.tool_call_item("pending", "Read", {})]))
        with self.assertRaises(formats.TranscriptFormatError):
            await loki.compact_context_async(cancel_event=self.event)
        self.assertEqual(self.sent, [])

    async def test_failed_save_and_post_publication_cancel_boundary(self):
        writer = loki.files.atomic_write_text
        calls = []

        def save(path, content, *, on_published=None):
            document = json.loads(content)
            self.assertIs(self.session.transcript_items, self.before_archive)
            calls.append(document)
            if document["events"][-1].get("type") == "context_replace":
                raise OSError("candidate save failed")
            return writer(path, content, on_published=on_published)
        with mock.patch.object(loki.files, "atomic_write_text", side_effect=save):
            with self.assertRaisesRegex(OSError, "candidate save"):
                await loki.compact_context_async(cancel_event=self.event)
        self.assertIs(self.session.transcript_items, self.before_archive)
        self.assertEqual(self.session.transcript_items, self.before)
        calls.clear()

        def committed(path, content, *, on_published=None):
            save_result = writer(path, content, on_published=on_published)
            calls.append(content)
            if json.loads(content)["events"][-1].get("type") == "context_replace":
                self.event.set()
            return save_result
        with mock.patch.object(loki.files, "atomic_write_text", side_effect=committed):
            self.assertIn("Context compacted", await loki.compact_context_async(cancel_event=self.event))
        self.assertEqual(self.session.transcript_items[-1]["type"], "context_replace")
        self.assertTrue(self.event.is_set())

    async def test_precompact_policy_is_strict_once_and_uses_saved_log_metadata(self):
        definition = settings.HookSettings("pre_compact", "policy", ["policy"], self.session.shell_cwd,
                                           on_error="deny", workspace_side_effects=True)
        pipeline = tool_runtime.build_hook_pipeline([definition], create_exec=loki._tool_process_creator())
        results = [{}, {"action": "deny", "message": "not now"}, {"events": []}, {"action": "unknown"}, OSError("hook failed")]
        for output in results:
            self.session.transcript_items = copy.deepcopy(self.before)
            self.before_archive = self.session.transcript_items
            self.sent.clear()
            self.event.clear()
            self.session.file_observations["hook-stale"] = "old"

            async def policy(command, payload, cwd, timeout, **kwargs):
                self.assertEqual(payload["event"], "pre_compact")
                self.assertEqual(payload["chat_log_path"], self.session.chat_log_path)
                self.assertNotIn("access-sentinel", json.dumps(payload))
                with open(payload["chat_log_path"], encoding="utf-8") as stream:
                    self.assertEqual(json.load(stream)["events"], self.before)
                if isinstance(output, Exception):
                    raise output
                return output
            with self.subTest(output=output), mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline), \
                    mock.patch.object(tool_runtime, "_run_hook_command", side_effect=policy) as runner:
                if output == {}:
                    await loki.compact_context_async(cancel_event=self.event)
                    self.assertEqual(len(self.sent), 1)
                else:
                    with self.assertRaises(Exception):
                        await loki.compact_context_async(cancel_event=self.event)
                    self.assertEqual(self.sent, [])
                self.assertEqual(runner.await_count, 1)
                self.assertEqual(self.session.file_observations, {})

    async def test_precompact_runs_once_across_prepared_transport_retry(self):
        definition = settings.HookSettings("pre_compact", "once", ["policy"], self.session.shell_cwd, on_error="deny")
        pipeline = tool_runtime.build_hook_pipeline([definition], create_exec=loki._tool_process_creator())
        attempts = []

        @contextlib.asynccontextmanager
        async def retry(method, url, **kwargs):
            attempts.append([kwargs["body"], kwargs["headers_in"]["Idempotency-Key"]])
            if len(attempts) == 1:
                raise ConnectionResetError(errno.ECONNRESET, "synthetic pre-output reset")
            async with self.streaming(method, url, **kwargs) as response:
                yield response
        with mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline), \
                mock.patch.object(tool_runtime, "_run_hook_command", new=mock.AsyncMock(return_value={})) as policy, \
                mock.patch.object(http_client, "async_http_stream", new=retry), \
                mock.patch.object(http_client, "HTTP_RETRY_BASE_DELAY_S", 0), \
                mock.patch.object(http_client, "HTTP_RETRY_MAX_JITTER_S", 0):
            await loki.compact_context_async(cancel_event=self.event)
            self.assertEqual(policy.await_count, 1)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])

    def enable_compaction_recovery(self, *, retry_limit=5, delay=0):
        # Observe a real configured process through its existing diagnostic pipe,
        # not a writable file in the hook's deliberately read-only filesystem.
        observed = []
        program = '''import json,sys
payload = json.load(sys.stdin)
print(json.dumps(payload), file=sys.stderr)
result = {}
if payload['event'] == 'turn_error':
    result = ({'action': 'retry', 'delay_seconds': float(sys.argv[2])}
              if payload['retry_count'] < int(sys.argv[1]) else {'action': 'finish'})
print(json.dumps(result))
'''
        command = [sys.executable, "-c", program, str(retry_limit), str(delay)]
        definitions = [settings.HookSettings(phase, name, command, self.session.shell_cwd)
                       for phase, name in [["pre_compact", "once"], ["turn_error", "retry"]]]
        self.pipeline = tool_runtime.build_hook_pipeline(
            definitions, create_exec=loki._tool_process_creator(),
            stderr_reporter=lambda command, text: observed.extend(json.loads(line) for line in text.splitlines()))
        self.enterContext(mock.patch.object(loki, "TOOL_HOOK_PIPELINE", self.pipeline))
        return observed

    async def test_sampling_compaction_and_continuation_share_one_real_retry_hook_counter(self):
        observed = self.enable_compaction_recovery()
        self.idle_routing = False
        self.ordinary_outputs = [self.tool_batch()]
        requests = compactions.Requests()
        events, attempts = [], []
        counts = {"initial": 0, "compact": 0, "continuation": 0}

        def request():
            if len(self.sent) == 1:
                requests.request()
        self.mutate = request

        @contextlib.asynccontextmanager
        async def scripted(method, url, **kwargs):
            compact = json.loads(kwargs["body"])["input"][-1].get("type") == "compaction_trigger"
            stage = "compact" if compact else "initial" if not self.sent else "continuation"
            counts[stage] += 1
            attempts.append([stage, kwargs["body"], kwargs["headers_in"]["Idempotency-Key"]])
            if stage == "initial" and counts[stage] == 1:
                raise ConnectionResetError(errno.ECONNRESET, "initial node disappeared")
            if stage == "continuation" and counts[stage] == 1:
                raise model_requests.StreamingApiError(url, 503, "Unavailable", '{"error":{"message":"new node unavailable"}}')
            if compact:
                self.before = copy.deepcopy(self.session.transcript_items)
                self.mode = "missing" if counts[stage] == 1 else "valid"
            async with self.streaming(method, url, **kwargs) as response:
                yield response

        with mock.patch.object(http_client, "async_http_stream", new=scripted), contextlib.redirect_stderr(io.StringIO()):
            text = await loki.run_tool_loop_async(
                self.session.transcript_items, cancel_event=self.event, recovery=loki.make_turn_recovery(self.event),
                stream_chat=True, max_loops=2, compaction_requests=requests, on_event=events.append)
        self.assertEqual(text, "ordinary answer")
        self.assertEqual(counts, {"initial": 2, "compact": 2, "continuation": 2})
        policy = observed
        failures = [entry for entry in policy if entry["event"] == "turn_error"]
        self.assertEqual([entry["retry_count"] for entry in failures], [0, 1, 2])
        self.assertEqual([entry["error_type"] for entry in failures], ["network_error", "provider_error", "api_error"])
        self.assertTrue(all(entry["recovery_point"] == "request_replay" and not entry["output_started"] for entry in failures))
        self.assertEqual(sum(entry["event"] == "pre_compact" for entry in policy), 1)
        for stage in counts:
            pair = [entry[1:] for entry in attempts if entry[0] == stage]
            self.assertEqual(pair[0], pair[1])
        self.assertEqual(len({entry[2] for entry in attempts}), 3)
        self.assertEqual([event["call_id"] for event in events if event["type"] == "tool_result"], ["step-one", "step-two"])
        self.assertEqual([event["type"] for event in events if event["type"].startswith("compaction_")],
                         ["compaction_started", "compaction_completed"])
        self.assertEqual(sum(event["type"] == "context_replace" for event in self.session.transcript_items), 1)

    async def test_checkpoint_validation_retries_ignored_deltas_with_precompact_once(self):
        initial = copy.deepcopy(self.before)
        for mode in ["missing", "duplicate", "empty", "incomplete"]:
            with self.subTest(mode=mode):
                observed = self.enable_compaction_recovery()
                self.session.transcript_items = copy.deepcopy(initial)
                self.before_archive = self.session.transcript_items
                self.before = copy.deepcopy(initial)
                self.sent.clear()
                attempts = []

                @contextlib.asynccontextmanager
                async def repaired(method, url, **kwargs):
                    attempts.append([kwargs["body"], kwargs["headers_in"]["Idempotency-Key"]])
                    self.idle_routing = len(attempts) == 1
                    self.assertEqual(kwargs["headers_in"].get(model_requests.CODEX_TURN_STATE_HEADER),
                                     None if self.idle_routing else "operation-state")
                    self.mode = mode if len(attempts) == 1 else "valid"
                    async with self.streaming(method, url, **kwargs) as response:
                        async def chunks():
                            # No displayed assistant/thought content or history
                            # exists for these speculative provider deltas.
                            yield b'data: {"type":"response.output_text.delta","delta":"ignored text"}\n\n'
                            async for chunk in response.body:
                                yield chunk
                        yield http_client.HttpStreamResponse(url, response.status, response.reason, response.headers, chunks())

                with mock.patch.object(http_client, "async_http_stream", new=repaired), contextlib.redirect_stderr(io.StringIO()):
                    await loki.compact_context_async(cancel_event=self.event, recovery=loki.make_turn_recovery(self.event))
                self.assertEqual(attempts[0], attempts[1])
                policy = observed
                self.assertEqual([entry["event"] for entry in policy], ["pre_compact", "turn_error"])
                self.assertEqual(policy[1]["retry_count"], 0)
                self.assertFalse(policy[1]["output_started"])
                self.assertNotIn("ignored text", json.dumps(self.session.transcript_items))

    async def test_native_failed_checkpoint_diagnostics_reach_policy_for_sse_and_json(self):
        data = {"object": "response", "status": "failed", "output": [],
                "error": {"code": "server_error", "message": "Assigned cluster node failed"}}
        for streaming in [False, True]:
            with self.subTest(streaming=streaming):
                observed = self.enable_compaction_recovery(retry_limit=0)

                @contextlib.asynccontextmanager
                async def failed(method, url, **kwargs):
                    async def chunks():
                        if streaming:
                            yield b'data: {"type":"response.output_text.delta","delta":"discarded"}\n\n'
                            yield ("data: " + json.dumps({"type": "response.failed", "response": data}) + "\n\n").encode()
                        else:
                            yield json.dumps(data).encode()
                    yield http_client.HttpStreamResponse(url, 200, "OK", {
                        "content-type": "text/event-stream" if streaming else "application/json"}, chunks())

                with mock.patch.object(http_client, "async_http_stream", new=failed):
                    with self.assertRaises(protocols.ResponseApiError):
                        await loki.compact_context_async(cancel_event=self.event, recovery=loki.make_turn_recovery(self.event))
                self.assertEqual([entry["event"] for entry in observed], ["pre_compact", "turn_error"])
                self.assertEqual(observed[1]["provider_code"], "server_error")
                self.assertEqual(observed[1]["error_details"], data["error"])
                self.assertEqual(self.session.transcript_items, self.before)

    async def test_compaction_retry_decline_does_not_stack_fixed_transport_attempts(self):
        observed = self.enable_compaction_recovery(retry_limit=0)
        attempts = []

        @contextlib.asynccontextmanager
        async def failed(method, url, **kwargs):
            attempts.append(kwargs["body"])
            raise ConnectionResetError(errno.ECONNRESET, "declined by policy")
            yield

        with mock.patch.object(http_client, "async_http_stream", new=failed):
            with self.assertRaises(ConnectionResetError):
                await loki.compact_context_async(cancel_event=self.event, recovery=loki.make_turn_recovery(self.event))
        self.assertEqual(len(attempts), 1)
        self.assertEqual([entry["event"] for entry in observed], ["pre_compact", "turn_error"])
        self.assertEqual(self.session.transcript_items, self.before)

    async def test_compaction_cancel_during_retry_backoff_settles_controls_without_publication(self):
        observed = self.enable_compaction_recovery(delay=3600)
        requests = compactions.Requests()
        accepted = requests.request()
        ready = asyncio.Event()
        events = []
        notice = recoveries._retry_notice

        def waiting(*args):
            ready.set()
            return notice(*args)

        @contextlib.asynccontextmanager
        async def failed(method, url, **kwargs):
            raise ConnectionResetError(errno.ECONNRESET, "cancel during backoff")
            yield

        with mock.patch.object(http_client, "async_http_stream", new=failed), \
                mock.patch.object(recoveries, "_retry_notice", side_effect=waiting), contextlib.redirect_stderr(io.StringIO()):
            owner = asyncio.create_task(loki.run_tool_loop_async(
                self.session.transcript_items, cancel_event=self.event, recovery=loki.make_turn_recovery(self.event),
                stream_chat=True, compaction_requests=requests, on_event=events.append))
            try:
                await observe_fixture(ready.wait(), owner)
                self.event.set()
                await owner
            finally:
                self.event.set()
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        self.assertEqual(accepted.result()["type"], "compaction_cancelled")
        self.assertEqual(self.session.transcript_items, self.before)
        self.assertEqual([entry["event"] for entry in observed], ["pre_compact", "turn_error"])
        self.assertIsNone(requests.pending)

    async def test_local_compaction_failures_are_never_offered_to_retry_policy(self):
        initial = copy.deepcopy(self.before)
        config = copy.deepcopy(self.session.runtime_config)
        for stage in ["initial_save", "precompact", "stale", "candidate_save", "projection", "unsupported"]:
            with self.subTest(stage=stage), contextlib.ExitStack() as patches:
                self.session.transcript_items = copy.deepcopy(initial)
                self.before_archive = self.session.transcript_items
                self.before = copy.deepcopy(initial)
                self.session.runtime_config = copy.deepcopy(config)
                self.mutate = None
                self.sent.clear()
                policy = mock.AsyncMock(return_value=recoveries.RetryDecision(0))
                recovery = recoveries.TurnRecovery(policy, self.event)
                if stage == "initial_save":
                    patches.enter_context(mock.patch.object(loki, "save_chat_log", side_effect=OSError("initial save failed")))
                elif stage == "precompact":
                    patches.enter_context(mock.patch.object(self.pipeline, "prepare_compaction", new=mock.AsyncMock(
                        side_effect=tool_runtime.HookExecutionError("gate denied"))))
                elif stage == "stale":
                    self.mutate = lambda: setattr(self.session.runtime_config, "model", "changed")
                elif stage == "candidate_save":
                    save = loki.save_chat_log

                    def saving(*args, **kwargs):
                        if kwargs.get("on_published") is not None:
                            raise OSError("candidate save failed")
                        return save(*args, **kwargs)
                    patches.enter_context(mock.patch.object(loki, "save_chat_log", side_effect=saving))
                elif stage == "projection":
                    patches.enter_context(mock.patch.object(compactions, "build_v2_window", side_effect=formats.ProjectionError("projection failed")))
                else:
                    self.session.runtime_config.model = ""
                expected = {"initial_save": OSError, "precompact": tool_runtime.HookExecutionError,
                            "stale": compactions.CompactionError, "candidate_save": OSError,
                            "projection": formats.ProjectionError, "unsupported": compactions.CompactionError}
                with self.assertRaises(expected[stage]):
                    await loki.compact_context_async(cancel_event=self.event, recovery=recovery)
                policy.assert_not_awaited()
                self.assertFalse(any(event["type"] == "context_replace" for event in self.session.transcript_items))

    async def test_idle_compaction_does_not_borrow_a_configured_turn_retry_hook(self):
        observed = self.enable_compaction_recovery(retry_limit=0)
        attempts = []

        @contextlib.asynccontextmanager
        async def retry(method, url, **kwargs):
            attempts.append(kwargs["body"])
            if len(attempts) == 1:
                raise ConnectionResetError(errno.ECONNRESET, "idle transport replay")
            async with self.streaming(method, url, **kwargs) as response:
                yield response

        with mock.patch.object(http_client, "async_http_stream", new=retry), \
                mock.patch.object(http_client, "HTTP_RETRY_BASE_DELAY_S", 0), \
                mock.patch.object(http_client, "HTTP_RETRY_MAX_JITTER_S", 0):
            await loki.compact_context_async(cancel_event=self.event)
        self.assertEqual(len(attempts), 2)
        self.assertEqual([entry["event"] for entry in observed], ["pre_compact"])

    async def test_model_change_during_gate_stops_before_remote_billing(self):
        definition = settings.HookSettings("pre_compact", "stale", ["policy"], self.session.shell_cwd, on_error="deny")
        pipeline = tool_runtime.build_hook_pipeline([definition], create_exec=loki._tool_process_creator())

        async def change(*args, **kwargs):
            self.session.runtime_config.model = "changed-during-hook"
            return {}
        with mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline), \
                mock.patch.object(tool_runtime, "_run_hook_command", side_effect=change):
            with self.assertRaisesRegex(compactions.CompactionError, "changed"):
                await loki.compact_context_async(cancel_event=self.event)
        self.assertEqual(self.sent, [])
        self.assertIs(self.session.transcript_items, self.before_archive)

    async def test_request_failure_and_request_cancellation_keep_previous_context(self):
        ready = asyncio.Event()
        retired = []

        @contextlib.asynccontextmanager
        async def failed(method, url, **kwargs):
            raise protocols.StreamProtocolError("provider stream failed")
            yield
        with mock.patch.object(http_client, "async_http_stream", new=failed):
            with self.assertRaises(protocols.StreamProtocolError):
                await loki.compact_context_async(cancel_event=self.event)

        @contextlib.asynccontextmanager
        async def blocked(method, url, **kwargs):
            async def chunks():
                ready.set()
                await self.event.wait()
                raise http_client.HttpRequestCancelled()
                yield
            try:
                yield http_client.HttpStreamResponse(url, 200, "OK", {"content-type": "text/event-stream"}, chunks())
            finally:
                retired.append(True)
        with mock.patch.object(http_client, "async_http_stream", new=blocked):
            operation = asyncio.create_task(loki.compact_context_async(cancel_event=self.event))
            try:
                await observe_fixture(ready.wait(), operation)
                self.event.set()
                with self.assertRaises(recoveries.Cancelled):
                    await operation
            finally:
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
        self.assertEqual(retired, [True])
        self.assertIs(self.session.transcript_items, self.before_archive)
        self.assertEqual(self.session.transcript_items, self.before)

    async def test_cancelled_precompact_process_is_joined_without_request(self):
        creator = loki._tool_process_creator()
        created = asyncio.Event()
        children = []

        async def create(*args, **kwargs):
            process = await creator(*args, **kwargs)
            children.append(process)
            created.set()
            return process
        definition = settings.HookSettings("pre_compact", "blocked-policy", [sys.executable, "-c",
                                           "import asyncio; asyncio.run(asyncio.Event().wait())"],
                                           self.session.shell_cwd, on_error="deny")
        pipeline = tool_runtime.build_hook_pipeline([definition], create_exec=create)
        with mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline):
            operation = asyncio.create_task(loki.compact_context_async(cancel_event=self.event))
            try:
                await observe_fixture(created.wait(), operation)
                self.event.set()
                with self.assertRaises(recoveries.Cancelled):
                    await operation
            finally:
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
        self.assertEqual(self.sent, [])
        self.assertIsNotNone(children[0].returncode)
        self.assertIs(self.session.transcript_items, self.before_archive)

    def params(self, text):
        return {"sessionId": "compact-session", "prompt": [{"type": "text", "text": text}]}

    def tui_inventory(self):
        return loki.CredentialInventory({
            "LOKI_API_BASE": authentications.OPENAI_CHATGPT_RESPONSES_URL,
            "LOKI_PROVIDER": formats.OPENAI_RESPONSES, "LOKI_MODEL": self.session.model,
            "LOKI_CREDENTIAL_REF": authentications.CredentialRef.openai_subscription().encode(),
            "LOKI_OPENAI_REQUEST_PROFILE": json.dumps(self.session.runtime_config.chat_provider.openai_request_profile.to_dict()),
            "LOKI_STREAM": "1"}, self.broker.available())

    async def test_compact_is_advertised_and_arguments_cannot_become_model_input(self):
        with mock.patch.object(acp_commands.skills, "available", return_value=[]):
            self.assertIn("compact", [command["name"] for command in acp_commands.advertised_commands()])
        for command in ["/compact extra", "/compact\textra"]:
            self.assertEqual(acp_commands.parse(command), ("compact", "extra"))
            with mock.patch.object(loki, "compact_context_async") as compact:
                outcome = await acp_commands.run(command, self.session, cancel_event=self.event)
                self.assertEqual(outcome.text, "Usage: /compact")
                compact.assert_not_called()
        self.assertIsNone(acp_commands.parse("/compactness"))
        self.assertEqual(self.session.transcript_items, self.before)

    async def test_acp_admission_resets_cancel_saves_command_and_keeps_staged_input(self):
        messages = []
        worker = acp_worker.Worker(self.session, messages.append)
        self.addAsyncCleanup(worker.close)
        self.event = worker.cancel_event
        self.event.set()
        staged = object()
        worker._pending_images.append(staged)
        self.assertEqual(await worker.prompt(self.params("/compact")), {"stopReason": "end_turn"})
        self.assertFalse(self.event.is_set())
        self.assertEqual(worker._pending_images, [staged])
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.session.transcript_items[-1]["type"], "context_replace")
        self.assertNotIn('"text": "/compact"', json.dumps(self.session.transcript_items))
        with open(self.session.chat_log_path, encoding="utf-8") as stream:
            self.assertEqual(json.load(stream)["schema"], "day-agent.session.v5")
        self.assertIn("Context compacted", json.dumps(messages))
        self.assertFalse(any(message.get("params", {}).get("update", {}).get("sessionUpdate") == "usage_update" for message in messages))
        worker._pending_images.clear()
        self.assertEqual(await worker.prompt(self.params("ordinary after compact")), {"stopReason": "end_turn"})
        self.assertEqual(len(self.sent), 2)

    async def test_acp_busy_rejection_request_cancel_and_next_command_remain_usable(self):
        worker = acp_worker.Worker(self.session, lambda message: None)
        self.addAsyncCleanup(worker.close)
        self.event = worker.cancel_event
        ready, release = asyncio.Event(), asyncio.Event()

        @contextlib.asynccontextmanager
        async def held(method, url, **kwargs):
            async with self.streaming(method, url, **kwargs) as response:
                ready.set()
                await release.wait()
                yield response
        with mock.patch.object(http_client, "async_http_stream", new=held):
            operation = asyncio.create_task(worker.prompt(self.params("/compact")))
            try:
                await observe_fixture(ready.wait(), operation)
                with self.assertRaisesRegex(acps.TransportError, "already running"):
                    await worker.prompt(self.params("overlap"))
                self.event.set()
                release.set()
                self.assertEqual(await operation, {"stopReason": "cancelled"})
            finally:
                release.set()
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
        self.assertIs(self.session.transcript_items, self.before_archive)
        self.assertEqual(await worker.prompt(self.params("/compact")), {"stopReason": "end_turn"})

    async def test_acp_prestart_cancel_and_post_commit_cancel_have_different_outcomes(self):
        messages = []
        worker = acp_worker.Worker(self.session, messages.append)
        self.addAsyncCleanup(worker.close)
        self.event = worker.cancel_event
        await worker.handle({"jsonrpc": "2.0", "id": 1, "method": "session/prompt", "params": self.params("/compact")}, concurrent=True)
        self.event.set()
        await worker._prompt_task
        self.assertEqual([message["result"] for message in messages if message.get("id") == 1], [{"stopReason": "cancelled"}])
        self.assertEqual(self.sent, [])
        writer = loki.files.atomic_write_text

        def commit(path, content, *, on_published=None):
            writer(path, content, on_published=on_published)
            if json.loads(content)["events"][-1].get("type") == "context_replace":
                self.event.set()
        with mock.patch.object(loki.files, "atomic_write_text", side_effect=commit):
            self.assertEqual(await worker.prompt(self.params("/compact")), {"stopReason": "end_turn"})
        self.assertTrue(self.event.is_set())
        self.assertEqual(self.session.transcript_items[-1]["type"], "context_replace")

    async def test_tui_startup_control_is_noop_without_consuming_the_next_prompt(self):
        from test_loki_tool_loop import ScriptedInputSession
        outer = self

        class Inputs(ScriptedInputSession):
            async def get(inner):
                if len(inner.messages) == 3:
                    outer.before_archive = outer.session.transcript_items
                    outer.before = copy.deepcopy(outer.before_archive)
                    inner.reader.cancel_event.set()
                elif len(inner.messages) == 2:
                    outer.assertFalse(terminal_frontend._terminal_activity.turn_running)
                    outer.assertFalse(any(event["type"] == "context_replace" for event in outer.session.transcript_items))
                    outer.assertEqual(outer.sent, [])
                return await super().get()
        inputs = Inputs(["/compact", "after compact", "/quit"])
        inventory = self.tui_inventory()
        output = io.StringIO()
        with mock.patch.object(loki, "CREDENTIALS", inventory), \
                mock.patch.object(terminal_frontend, "input_session", return_value=inputs), \
                mock.patch.object(terminal_frontend.terminals, "open_terminal_stdin"), \
                mock.patch.object(terminal_frontend, "new_chat_log_path", return_value=self.session.chat_log_path), \
                mock.patch.object(terminal_frontend, "restore_output_area_after_input"), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            self.assertEqual(await terminal_frontend.async_main([], settings.Settings()), 0)
        self.assertFalse(terminal_frontend._terminal_activity.turn_running)
        self.assertEqual(len(self.sent), 1)
        self.assertIn("Nothing to compact", output.getvalue())
        self.assertNotIn('"text": "/compact"', json.dumps(self.session.transcript_items))
        self.assertIn("/compact", terminal_frontend.status_text())

    async def _tui_running_control(self, failure=False, *, compaction_retries=0):
        from test_loki_tool_loop import ScriptedInputSession
        inputs = ScriptedInputSession(["perform work", "/quit"])
        self.idle_routing = False
        self.ordinary_outputs = [self.tool_batch()]
        self.mode = "missing" if failure else "valid"
        self.pipeline.turn_end_hooks = [object()]
        self.pipeline.finish_turn = mock.AsyncMock()
        inventory = self.tui_inventory()
        output = io.StringIO()

        def receive():
            if len(self.sent) == 1:
                self.before_archive = self.session.transcript_items
                self.assertTrue(inputs.on_submit("/compact"))
                self.assertTrue(inputs.on_submit("/compact"))
                self.assertFalse(terminal_frontend._terminal_activity.compaction_running)
        self.mutate = receive

        @contextlib.asynccontextmanager
        async def inspect(method, url, **kwargs):
            compact = json.loads(kwargs["body"])["input"][-1].get("type") == "compaction_trigger"
            self.assertEqual(terminal_frontend._terminal_activity.compaction_running, compact)
            if compact:
                self.assertEqual(self.session.transcript_items[-1]["type"], "tool_result")
                self.before = copy.deepcopy(self.session.transcript_items)
            async with self.streaming(method, url, **kwargs) as response:
                yield response

        with mock.patch.object(loki, "CREDENTIALS", inventory), \
                mock.patch.object(terminal_frontend, "input_session", return_value=inputs), \
                mock.patch.object(terminal_frontend.terminals, "open_terminal_stdin"), \
                mock.patch.object(terminal_frontend, "new_chat_log_path", return_value=self.session.chat_log_path), \
                mock.patch.object(terminal_frontend, "restore_output_area_after_input"), \
                mock.patch.object(http_client, "async_http_stream", new=inspect), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            self.assertEqual(await terminal_frontend.async_main([], settings.Settings()), 0)
        self.assertEqual(len(self.sent), (2 if failure else 3) + compaction_retries)
        self.assertEqual(self.pipeline.finish_turn.await_count, 1)
        self.assertEqual(self.pipeline.finish_turn.call_args.args[0]["reason"], "error" if failure else "completed")
        self.assertIs(self.session.transcript_items, self.before_archive)
        self.assertIn("Compaction already requested", output.getvalue())
        self.assertNotIn('"text": "/compact"', json.dumps(self.session.transcript_items))
        self.assertFalse(terminal_frontend._terminal_activity.turn_running)
        self.assertFalse(terminal_frontend._terminal_activity.compaction_running)
        with open(self.session.chat_log_path, encoding="utf-8") as stream:
            saved, _todos, _state, _tools = savefiles.read_chat_log(stream)
        if failure:
            self.assertFalse(any(event["type"] == "context_replace" for event in saved))
            self.assertIn("Context compaction failed", output.getvalue())
        else:
            self.assertEqual(formats.item_text(formats.active_events(saved)[-1]), "ordinary answer")

    async def test_tui_compaction_recovery_keeps_status_active_and_finalizes_once(self):
        observed = self.enable_compaction_recovery()
        streaming = self.streaming
        attempts = 0

        @contextlib.asynccontextmanager
        async def repaired(method, url, **kwargs):
            nonlocal attempts
            if json.loads(kwargs["body"])["input"][-1].get("type") == "compaction_trigger":
                attempts += 1
                self.mode = "missing" if attempts == 1 else "valid"
            async with streaming(method, url, **kwargs) as response:
                yield response

        with mock.patch.object(self, "streaming", new=repaired):
            await self._tui_running_control(compaction_retries=1)
        self.assertEqual([entry["event"] for entry in observed], ["pre_compact", "turn_error"])

    async def test_tui_receipt_during_sampling_coalesces_and_saves_same_turn_continuation(self):
        await self._tui_running_control()

    async def test_tui_compaction_failure_notifies_once_stops_sampling_and_returns_to_input(self):
        await self._tui_running_control(failure=True)

    async def test_retained_only_active_window_noops_even_with_archived_tool_history(self):
        retained = copy.deepcopy(self.before[:2])
        self.session.transcript_items.append(formats.context_replace_event(retained))
        original = copy.deepcopy(self.session.transcript_items)
        with mock.patch.object(loki, "save_chat_log") as save, \
                mock.patch.object(self.pipeline, "prepare_compaction", new=mock.AsyncMock()) as policy:
            self.assertEqual(await loki.compact_context_async(cancel_event=self.event), "Nothing to compact.")
            save.assert_not_called()
            policy.assert_not_called()
        self.assertEqual(self.session.transcript_items, original)
        self.assertEqual(self.sent, [])

    async def test_post_publication_close_error_keeps_disk_and_live_selection_together(self):
        writer = loki.files.atomic_write_text
        close = loki.files.private_files.close
        replace_at = loki.files.private_files.replace_at
        state = {"candidate": False, "published": False, "raised": False}

        def writing(path, content, *, on_published=None):
            state["candidate"] = json.loads(content)["events"][-1].get("type") == "context_replace"
            state["published"] = False
            return writer(path, content, on_published=on_published)

        def replacing(*args, on_published=None):
            def published():
                state["published"] = True
                on_published()
            replace_at(*args, on_published=published)

        def closing(handle):
            close(handle)
            if state["candidate"] and state["published"] and not state["raised"]:
                state["raised"] = True
                raise OSError("close failed after replacement")

        with mock.patch.object(loki.files, "atomic_write_text", side_effect=writing), \
                mock.patch.object(loki.files.private_files, "replace_at", side_effect=replacing), \
                mock.patch.object(loki.files.private_files, "close", side_effect=closing):
            with self.assertRaisesRegex(OSError, "after replacement"):
                await loki.compact_context_async(cancel_event=self.event)
        self.assertTrue(state["raised"])
        self.assertIs(self.session.transcript_items, self.before_archive)
        self.assertEqual(self.session.transcript_items[-1]["type"], "context_replace")
        self.assertFalse(self.session.chat_log_dirty)
        self.assertIsNone(self.session.context_snapshot(live=True).usage)
        with open(self.session.chat_log_path, encoding="utf-8") as stream:
            saved, _todos, _state, _tools = savefiles.read_chat_log(stream)
        self.assertEqual(saved, self.session.transcript_items)

    async def test_publication_inside_replace_primitive_precedes_its_cleanup_failure(self):
        writer = loki.files.atomic_write_text
        replace_at = loki.files.private_files.replace_at
        candidate = False

        def writing(path, content, *, on_published=None):
            nonlocal candidate
            candidate = json.loads(content)["events"][-1].get("type") == "context_replace"
            return writer(path, content, on_published=on_published)

        def rename_then_fail(directory, temporary, name, *, on_published=None):
            replace_at(directory, temporary, name, on_published=on_published)
            if candidate:
                raise OSError("rename primitive cleanup failed")

        with mock.patch.object(loki.files, "atomic_write_text", side_effect=writing), \
                mock.patch.object(loki.files.private_files, "replace_at", side_effect=rename_then_fail):
            with self.assertRaisesRegex(OSError, "primitive cleanup"):
                await loki.compact_context_async(cancel_event=self.event)
        self.assertIs(self.session.transcript_items, self.before_archive)
        self.assertEqual(self.session.transcript_items[-1]["type"], "context_replace")
        with open(self.session.chat_log_path, encoding="utf-8") as stream:
            saved, _todos, _state, _tools = savefiles.read_chat_log(stream)
        self.assertEqual(saved, self.session.transcript_items)

    def tool_batch(self):
        return [{"type": "function_call", "call_id": call_id, "name": "TodoWrite",
                 "arguments": json.dumps({"todos": [{"content": content, "status": "pending", "priority": "high"}]})}
                for call_id, content in [["step-one", "first effect"], ["step-two", "second effect"]]]

    async def test_real_loop_coalesces_at_whole_batch_boundary_and_persists_continuation(self):
        self.idle_routing = False
        self.ordinary_outputs = [self.tool_batch()]
        requests = compactions.Requests()
        accepted = []
        events = []
        initial_thinking = loki.capture_turn_settings()

        def request_during_sampling():
            if len(self.sent) == 1:
                accepted.extend([requests.request(), requests.request()])
        self.mutate = request_during_sampling

        def on_event(event):
            events.append(event)
            if event["type"] == "tool_call":
                self.assertEqual(len(self.sent), 1, "compaction must not split the tool batch")
            if event["type"] == "tool_result":
                accepted.append(requests.request())
                self.session.agent_mode = "explore"
                self.session.reasoning_retention = "preserve"
                self.before = copy.deepcopy(self.session.transcript_items)

        text = await loki.run_tool_loop_async(
            self.session.transcript_items, cancel_event=self.event,
            stream_chat=True, max_loops=2, on_event=on_event,
            thinking=initial_thinking, compaction_requests=requests)
        self.assertEqual(text, "ordinary answer")
        self.assertTrue(all(future is accepted[0] for future in accepted))
        self.assertEqual(accepted[0].result()["type"], "compaction_completed")
        self.assertIsNone(requests.pending)
        self.assertEqual(len(self.sent), 3)
        self.assertEqual(self.sent[1]["input"][-1], {"type": "compaction_trigger"})
        self.assertEqual(self.headers[1][model_requests.CODEX_TURN_STATE_HEADER], "operation-state")
        self.assertEqual(self.headers[2][model_requests.CODEX_TURN_STATE_HEADER], "operation-state")
        self.assertEqual(self.sent[0]["input"][0]["tools"], self.sent[1]["input"][0]["tools"])
        self.assertEqual(self.sent[0]["reasoning"], self.sent[1]["reasoning"])
        self.assertNotIn("uncommented-tool-fact", json.dumps(self.sent[-1]))
        self.assertIn("opaque-state", json.dumps(self.sent[-1]))
        self.assertIs(self.session.transcript_items, self.before_archive)
        self.assertEqual([event["call_id"] for event in self.session.transcript_items
                          if event.get("type") == "tool_result" and event["call_id"].startswith("step-")],
                         ["step-one", "step-two"])
        self.assertEqual(self.session.session_todos[0]["content"], "second effect")
        self.assertFalse(any(event["type"] == "max_loops" for event in events))
        self.assertEqual(self.session.agent_mode, "explore")
        self.assertIsNone(self.session.last_instructed_agent_mode)
        self.assertTrue(self.session.chat_log_dirty, "continuation after the checkpoint must be saveable")
        loki.save_chat_log()
        with open(self.session.chat_log_path, encoding="utf-8") as stream:
            saved, _todos, state, _tools = savefiles.read_chat_log(stream)
        self.assertEqual(saved, self.session.transcript_items)
        self.assertEqual(state["reasoning_retention"], "preserve")
        self.assertEqual(formats.item_text(formats.active_events(saved)[-1]), "ordinary answer")

    async def _acp_control_journey(self, *, failure=False, cancel=False):
        self.idle_routing = False
        self.ordinary_outputs = [self.tool_batch()]
        self.mode = "missing" if failure else "valid"
        messages = []
        worker = acp_worker.Worker(self.session, messages.append)
        worker.can_report_compaction = True
        self.addAsyncCleanup(worker.close)
        self.event = worker.cancel_event
        self.pipeline.turn_end_hooks = [object()]
        self.pipeline.finish_turn = mock.AsyncMock()
        sampling_ready, sampling_release = asyncio.Event(), asyncio.Event()
        compact_ready, compact_release = asyncio.Event(), asyncio.Event()
        final_ready, final_release = asyncio.Event(), asyncio.Event()

        @contextlib.asynccontextmanager
        async def controlled(method, url, **kwargs):
            payload = json.loads(kwargs["body"])
            compact = payload["input"][-1].get("type") == "compaction_trigger"
            first = not self.sent
            if compact:
                self.assertEqual(self.session.transcript_items[-1]["type"], "tool_result")
                self.assertFalse(any(event["type"] == "context_replace" for event in self.session.transcript_items))
                self.before = copy.deepcopy(self.session.transcript_items)
            async with self.streaming(method, url, **kwargs) as response:
                ready, release = ([compact_ready, compact_release] if compact else
                                  [sampling_ready, sampling_release] if first else [final_ready, final_release])
                ready.set()
                await release.wait()
                yield response

        with mock.patch.object(http_client, "async_http_stream", new=controlled):
            await worker.handle({"id": 1, "method": "session/prompt", "params": self.params("do the work")}, concurrent=True)
            owner = worker._prompt_task
            try:
                await observe_fixture(sampling_ready.wait(), owner)
                staged = object()
                worker._pending_images.append(staged)
                for request_id in [2, 3]:
                    await worker.handle({"id": request_id, "method": "session/prompt", "params": self.params("/compact")}, concurrent=True)
                shared = worker.compaction_requests.pending
                self.assertIs(worker._prompt_task, owner)
                self.assertFalse(self.event.is_set())
                await worker.handle({"id": 4, "method": "session/prompt", "params": self.params("ordinary overlap")}, concurrent=True)
                self.assertIn("error", next(message for message in messages if message.get("id") == 4))
                sampling_release.set()
                await observe_fixture(compact_ready.wait(), owner)
                await worker.handle({"id": 5, "method": "session/prompt", "params": self.params("/compact")}, concurrent=True)
                self.assertIs(worker.compaction_requests.pending, shared)
                self.assertIs(worker._prompt_task, owner)
                self.assertFalse(any(message.get("id") in [1, 2, 3, 5] for message in messages))
                if cancel:
                    await worker.dispatch("session/cancel", {})
                compact_release.set()
                if not failure and not cancel:
                    await observe_fixture(final_ready.wait(), owner)
                    self.assertFalse(any(message.get("id") == 1 for message in messages))
                    self.assertEqual([message["result"] for message in messages if message.get("id") in [2, 3, 5]],
                                     [{"stopReason": "end_turn"}] * 3)
                    self.assertEqual(worker._pending_images, [staged])
                    self.assertEqual(self.pipeline.finish_turn.await_count, 0)
                    final_release.set()
                await owner
            finally:
                sampling_release.set()
                compact_release.set()
                final_release.set()
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        replies = [message for message in messages if message.get("id") in [1, 2, 3, 5]]
        self.assertEqual(sorted(message["id"] for message in replies), [1, 2, 3, 5])
        self.assertEqual(self.pipeline.finish_turn.await_count, 1)
        payload = self.pipeline.finish_turn.call_args.args[0]
        expected_reason = "error" if failure else "cancelled" if cancel else "completed"
        self.assertEqual(payload["reason"], expected_reason)
        if failure:
            self.assertTrue(all("error" in message for message in replies))
            self.assertEqual(len(self.sent), 2, "failed compaction must stop further sampling")
        elif cancel:
            self.assertEqual([message["result"] for message in replies], [{"stopReason": "cancelled"}] * 4)
            self.assertEqual(len(self.sent), 2)
        else:
            self.assertEqual(len(self.sent), 3)
        updates = [message["params"]["update"] for message in messages if message.get("method") == "session/update"
                   and message["params"]["update"]["sessionUpdate"] == "compaction_update"]
        self.assertEqual([update["status"] for update in updates],
                         ["in_progress", "failed" if failure else "cancelled" if cancel else "completed"])
        self.assertEqual(updates[0]["compactionId"], updates[1]["compactionId"])
        self.assertTrue(all("summary" not in update for update in updates))
        self.assertIsNone(worker.compaction_requests.pending)
        self.assertNotIn('"text": "/compact"', json.dumps(self.session.transcript_items))
        worker._pending_images.clear()

    async def _tool_less_request(self, follow_up):
        self.idle_routing = False
        self.end_turns = [not follow_up]
        requests = compactions.Requests()
        events = []

        def request():
            if len(self.sent) == 1:
                requests.request()
        self.mutate = request

        @contextlib.asynccontextmanager
        async def inspect(method, url, **kwargs):
            if json.loads(kwargs["body"])["input"][-1].get("type") == "compaction_trigger":
                self.assertEqual(self.session.transcript_items[-1]["type"], "model_response")
                self.before = copy.deepcopy(self.session.transcript_items)
            async with self.streaming(method, url, **kwargs) as response:
                yield response

        with mock.patch.object(http_client, "async_http_stream", new=inspect):
            self.assertEqual(await loki.run_tool_loop_async(
                self.session.transcript_items, cancel_event=self.event,
                stream_chat=True, max_loops=2, on_event=events.append,
                compaction_requests=requests), "ordinary answer")
        self.assertEqual(len(self.sent), 3 if follow_up else 2)
        self.assertEqual([event["type"] for event in events if event["type"].startswith("compaction_")],
                         ["compaction_started", "compaction_completed"])
        self.assertIsNone(requests.pending)
        self.assertFalse(any(event["type"] == "max_loops" for event in events))

    async def test_cancellation_before_final_service_is_not_a_completed_turn(self):
        self.idle_routing = False
        requests = compactions.Requests()
        events = []
        accepted = []

        def on_response(turn, event):
            accepted.append(requests.request())
            self.event.set()

        try:
            self.assertEqual(await loki.run_tool_loop_async(
                self.session.transcript_items, cancel_event=self.event,
                stream_chat=True, on_event=events.append, on_response=on_response,
                compaction_requests=requests), "")
        finally:
            requests.finish_pending(cancelled=self.event.is_set())
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(accepted[0].result()["type"], "compaction_cancelled")
        self.assertEqual([event["type"] for event in events].count("response_cancelled"), 1)
        self.pipeline.turn_end_hooks = [object()]
        self.pipeline.finish_turn = mock.AsyncMock()
        await loki.run_turn_end_hooks_async(events)
        self.assertEqual(self.pipeline.finish_turn.call_args.args[0]["reason"], "cancelled")

    async def test_tool_less_continuation_services_request_in_the_same_loop(self):
        await self._tool_less_request(True)

    async def test_final_response_services_request_without_an_invented_model_sample(self):
        await self._tool_less_request(False)

    async def test_acp_control_during_actual_ask_waits_for_answer_and_whole_batch(self):
        self.idle_routing = False
        self.ordinary_outputs = [[{"type": "function_call", "call_id": "question", "name": "Ask",
                                  "arguments": json.dumps({"question": "Continue?", "options": [{"label": "Yes"}, {"label": "No"}]})},
                                 self.tool_batch()[1]]]
        messages = []
        question_ready = asyncio.Event()

        def write(message):
            messages.append(message)
            if message.get("method") == "session/request_input":
                question_ready.set()

        worker = acp_worker.Worker(self.session, write)
        worker.can_ask_user = True
        self.addAsyncCleanup(worker.close)
        self.event = worker.cancel_event

        @contextlib.asynccontextmanager
        async def inspect(method, url, **kwargs):
            if json.loads(kwargs["body"])["input"][-1].get("type") == "compaction_trigger":
                self.assertEqual(self.session.transcript_items[-1]["call_id"], "step-two")
                self.before = copy.deepcopy(self.session.transcript_items)
            async with self.streaming(method, url, **kwargs) as response:
                yield response

        with mock.patch.object(http_client, "async_http_stream", new=inspect):
            await worker.handle({"id": 1, "method": "session/prompt", "params": self.params("ask then work")}, concurrent=True)
            owner = worker._prompt_task
            try:
                await observe_fixture(question_ready.wait(), owner)
                for request_id in [2, 3]:
                    await worker.handle({"id": request_id, "method": "session/prompt", "params": self.params("/compact")}, concurrent=True)
                self.assertEqual(len(self.sent), 1)
                self.assertIs(worker._prompt_task, owner)
                reverse = next(message for message in messages if message.get("method") == "session/request_input")
                worker.resolve_reverse(reverse["id"], {"result": {"action": "answered", "answer": "Yes"}})
                await owner
            finally:
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        self.assertEqual(len(self.sent), 3)
        self.assertEqual(sorted(message["id"] for message in messages if "result" in message), [1, 2, 3])
        self.assertEqual(len([event for event in self.session.transcript_items if event["type"] == "context_replace"]), 1)
        self.assertEqual(worker._reverse_requests, {})
        self.assertEqual(worker._compaction_replies, [])

    async def _finalization_control(self, failure):
        self.idle_routing = False
        self.mode = "missing" if failure else "valid"
        messages = []
        worker = acp_worker.Worker(self.session, messages.append)
        self.addAsyncCleanup(worker.close)
        self.event = worker.cancel_event
        ready, release = asyncio.Event(), asyncio.Event()
        self.pipeline.turn_end_hooks = [object()]

        async def hook(payload):
            self.assertEqual(payload["reason"], "completed")
            ready.set()
            await release.wait()
        self.pipeline.finish_turn = mock.AsyncMock(side_effect=hook)

        @contextlib.asynccontextmanager
        async def inspect(method, url, **kwargs):
            if json.loads(kwargs["body"])["input"][-1].get("type") == "compaction_trigger":
                self.assertIn(1, [message.get("id") for message in messages if "result" in message])
                self.before = copy.deepcopy(self.session.transcript_items)
            async with self.streaming(method, url, **kwargs) as response:
                yield response

        with mock.patch.object(http_client, "async_http_stream", new=inspect):
            await worker.handle({"id": 1, "method": "session/prompt", "params": self.params("finish normally")}, concurrent=True)
            owner = worker._prompt_task
            try:
                await observe_fixture(ready.wait(), owner)
                for request_id in [2, 3]:
                    # Inline receipt must also return, not block this input
                    # actor while the original finalization is held.
                    await worker.handle({"id": request_id, "method": "session/prompt", "params": self.params("/compact")})
                self.assertEqual(len(self.sent), 1)
                release.set()
                await owner
            finally:
                release.set()
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        self.assertEqual(self.pipeline.finish_turn.await_count, 1)
        self.assertEqual(len(self.sent), 2)
        self.assertEqual([message["result"] for message in messages if message.get("id") == 1], [{"stopReason": "end_turn"}])
        controls = [message for message in messages if message.get("id") in [2, 3]]
        self.assertEqual(len(controls), 2)
        self.assertTrue(all(("error" in message) == failure for message in controls))
        self.assertIsNone(worker.compaction_requests.pending)
        self.assertEqual(worker._compaction_replies, [])

    async def test_control_during_turn_end_is_a_single_idle_tail_after_original_reply(self):
        await self._finalization_control(False)

    async def test_failed_idle_tail_does_not_retroactively_fail_the_completed_turn(self):
        await self._finalization_control(True)

    async def test_v1_worker_channel_correlates_controls_while_original_request_is_pending(self):
        from loki_agent import acp
        from test_acps import _LocalWorkerProcess
        self.idle_routing = False
        self.ordinary_outputs = [self.tool_batch()]
        notifications = []
        process = _LocalWorkerProcess(self.session, "compact-session", lambda message: None)
        process.worker.can_report_compaction = True
        self.event = process.worker.cancel_event
        channel = acp.WorkerChannel("compact-session", process, notifications.append)
        self.addAsyncCleanup(channel.close)
        sampling_ready, sampling_release = asyncio.Event(), asyncio.Event()
        compact_ready, compact_release = asyncio.Event(), asyncio.Event()
        final_ready, final_release = asyncio.Event(), asyncio.Event()

        @contextlib.asynccontextmanager
        async def controlled(method, url, **kwargs):
            compact = json.loads(kwargs["body"])["input"][-1].get("type") == "compaction_trigger"
            first = not self.sent
            if compact:
                self.assertEqual(self.session.transcript_items[-1]["type"], "tool_result")
                self.before = copy.deepcopy(self.session.transcript_items)
            async with self.streaming(method, url, **kwargs) as response:
                ready, release = ([compact_ready, compact_release] if compact else
                                  [sampling_ready, sampling_release] if first else [final_ready, final_release])
                ready.set()
                await release.wait()
                yield response

        controls = []
        with mock.patch.object(http_client, "async_http_stream", new=controlled):
            original = asyncio.create_task(channel.request("session/prompt", self.params("work")))
            try:
                await observe_fixture(sampling_ready.wait(), original)
                for _ in range(2):
                    forwarded = asyncio.Event()
                    control = asyncio.create_task(channel.request("session/prompt", self.params("/compact"), forwarded=forwarded))
                    controls.append(control)
                    await observe_fixture(forwarded.wait(), original)
                self.assertTrue(all(not task.done() for task in controls))
                sampling_release.set()
                await observe_fixture(compact_ready.wait(), original)
                self.assertEqual(len(channel._pending), 3)
                compact_release.set()
                await observe_fixture(final_ready.wait(), original)
                self.assertEqual(await asyncio.gather(*controls), [{"stopReason": "end_turn"}] * 2)
                self.assertFalse(original.done())
                final_release.set()
                self.assertEqual(await original, {"stopReason": "end_turn"})
            finally:
                sampling_release.set()
                compact_release.set()
                final_release.set()
                for task in [original, *controls]:
                    task.cancel()
                await asyncio.gather(original, *controls, return_exceptions=True)
        self.assertEqual(channel._pending, {})
        self.assertEqual(process.worker._compaction_replies, [])
        self.assertEqual(len(self.sent), 3)
        self.assertTrue(any(packet["params"]["update"]["sessionUpdate"] == "compaction_update"
                            for packet in notifications if packet.get("method") == "session/update"))
        await channel.close()
        self.assertTrue(channel._reader_task.done())
        self.assertEqual(process.returncode, 0)

    async def test_idle_setting_capture_failure_settles_joined_v1_controls(self):
        messages = []
        worker = acp_worker.Worker(self.session, messages.append)
        self.addAsyncCleanup(worker.close)
        accepted = worker.compaction_requests.request()
        worker._compaction_replies.extend([2, 3])
        with mock.patch.object(loki, "capture_turn_settings", side_effect=protocols.ProtocolError("invalid admitted settings")):
            await worker._perform_idle_compaction()
        self.assertEqual(accepted.result()["type"], "compaction_failed")
        self.assertEqual(sorted(message["id"] for message in messages if "error" in message), [2, 3])
        self.assertEqual(self.sent, [])
        self.assertIsNone(worker.compaction_requests.pending)
        self.assertEqual(worker._compaction_replies, [])

    async def test_acp_v1_controls_join_running_turn_without_early_original_reply(self):
        await self._acp_control_journey()

    async def test_acp_v1_compaction_failure_fails_original_and_controls_and_notifies_once(self):
        await self._acp_control_journey(failure=True)

    async def test_acp_v1_cancel_settles_all_controls_without_another_operation(self):
        await self._acp_control_journey(cancel=True)

    async def _v2_control_journey(self, *, failure=False, cancel=False, compaction_retries=0):
        self.idle_routing = False
        self.ordinary_outputs = [self.tool_batch()]
        self.mode = "missing" if failure else "valid"
        messages = []
        worker = acp_worker.Worker(self.session, messages.append)
        worker.protocol_version = 2
        self.addAsyncCleanup(worker.close)
        self.event = worker.cancel_event
        self.pipeline.turn_end_hooks = [object()]
        self.pipeline.finish_turn = mock.AsyncMock()
        sampling_ready, sampling_release = asyncio.Event(), asyncio.Event()
        compact_ready, compact_release = asyncio.Event(), asyncio.Event()

        @contextlib.asynccontextmanager
        async def controlled(method, url, **kwargs):
            compact = json.loads(kwargs["body"])["input"][-1].get("type") == "compaction_trigger"
            first = not self.sent
            if compact:
                self.assertEqual(self.session.transcript_items[-1]["type"], "tool_result")
                self.before = copy.deepcopy(self.session.transcript_items)
            async with self.streaming(method, url, **kwargs) as response:
                if compact or first:
                    ready, release = [compact_ready, compact_release] if compact else [sampling_ready, sampling_release]
                    ready.set()
                    await release.wait()
                yield response

        with mock.patch.object(http_client, "async_http_stream", new=controlled):
            await worker.handle(acps.request(1, "session/prompt", self.params("work")))
            owner = worker._prompt_task
            try:
                await observe_fixture(sampling_ready.wait(), owner)
                for identifier in [2, 3]:
                    await worker.handle(acps.request(identifier, "session/prompt", self.params("/compact")))
                shared = worker.compaction_requests.pending
                self.assertIs(worker._prompt_task, owner)
                self.assertEqual([message["id"] for message in messages if "result" in message], [1, 2, 3])
                sampling_release.set()
                await observe_fixture(compact_ready.wait(), owner)
                await worker.handle(acps.request(4, "session/prompt", self.params("/compact")))
                self.assertIs(worker.compaction_requests.pending, shared)
                if cancel:
                    await worker.dispatch("session/cancel", {})
                    self.assertFalse(shared.done(), "the executing operation owns its final outcome")
                compact_release.set()
                await owner
            finally:
                sampling_release.set()
                compact_release.set()
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        replies = [message for message in messages if message.get("id") in [1, 2, 3, 4]]
        self.assertEqual(len(replies), 4)
        self.assertTrue(all(list(message["result"]) == ["messageId"] for message in replies))
        self.assertEqual(len({message["result"]["messageId"] for message in replies}), 4)
        updates = [message["params"]["update"] for message in messages if message.get("method") == "session/update"]
        lifecycle = [update for update in updates if update["sessionUpdate"] == "compaction_update"]
        self.assertEqual([update["status"] for update in lifecycle],
                         ["in_progress", "failed" if failure else "cancelled" if cancel else "completed"])
        self.assertEqual(lifecycle[0]["compactionId"], lifecycle[1]["compactionId"])
        self.assertEqual(updates[-1]["state"], "idle")
        self.assertEqual(updates[-1]["stopReason"], "error" if failure else "cancelled" if cancel else "end_turn")
        self.assertEqual(len(self.sent), (2 if failure or cancel else 3) + compaction_retries)
        self.assertEqual(self.pipeline.finish_turn.await_count, 1)
        self.assertIsNone(worker.compaction_requests.pending)
        self.assertNotIn('"text": "/compact"', json.dumps(self.session.transcript_items))
        users = [update for update in updates if update["sessionUpdate"] == "user_message_chunk"]
        self.assertEqual({update["messageId"] for update in users}, {message["result"]["messageId"] for message in replies})
        self.assertTrue(all("messageId" in update for update in updates
                            if update["sessionUpdate"] in ["agent_message_chunk", "agent_thought_chunk"]))

    async def test_v2_compaction_recovery_preserves_single_insertion_replies_and_lifecycle(self):
        observed = self.enable_compaction_recovery()
        streaming = self.streaming
        attempts = 0

        @contextlib.asynccontextmanager
        async def repaired(method, url, **kwargs):
            nonlocal attempts
            if json.loads(kwargs["body"])["input"][-1].get("type") == "compaction_trigger":
                attempts += 1
                self.mode = "missing" if attempts == 1 else "valid"
            async with streaming(method, url, **kwargs) as response:
                yield response

        with mock.patch.object(self, "streaming", new=repaired), contextlib.redirect_stderr(io.StringIO()):
            await self._v2_control_journey(compaction_retries=1)
        self.assertEqual([entry["event"] for entry in observed], ["pre_compact", "turn_error"])

    async def test_v2_midturn_compaction_acks_immediately_and_coalesces_one_operation(self):
        await self._v2_control_journey()

    async def test_v2_compaction_failure_ends_foreground_without_replacing_any_insertion_ack(self):
        await self._v2_control_journey(failure=True)

    async def test_v2_cancel_during_compaction_waits_for_the_executor_outcome(self):
        await self._v2_control_journey(cancel=True)

    async def test_v2_busy_ordinary_prompts_insert_then_execute_fifo_and_cancel_has_a_cutoff(self):
        messages = []
        worker = acp_worker.Worker(self.session, messages.append)
        worker.protocol_version = 2
        self.addAsyncCleanup(worker.close)
        ready, release = asyncio.Event(), asyncio.Event()
        consumed = []

        async def prompt(params, thinking, acp_message=None):
            text = worker._prompt_text(params)
            consumed.append(text)
            if text == "first":
                ready.set()
                await release.wait()
            return {"stopReason": "cancelled" if worker.cancel_event.is_set() else "end_turn"}

        with mock.patch.object(worker, "_prompt", new=prompt):
            await worker.handle(acps.request(1, "session/prompt", self.params("first")))
            owner = worker._prompt_task
            try:
                await observe_fixture(ready.wait(), owner)
                for identifier, text in [[2, "discard"], [3, "also discard"]]:
                    await worker.handle(acps.request(identifier, "session/prompt", self.params(text)))
                self.assertEqual(len([message for message in messages if "result" in message]), 3)
                await worker.dispatch("session/cancel", {})
                for identifier, text in [[4, "later"], [5, "last"]]:
                    await worker.handle(acps.request(identifier, "session/prompt", self.params(text)))
                release.set()
                await owner
            finally:
                release.set()
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        self.assertEqual(consumed, ["first", "later", "last"])
        self.assertFalse(worker._v2_pending)
        self.assertEqual(len([message for message in messages if "result" in message]), 5)

    async def test_real_precompact_process_uses_sanitized_existing_runner(self):
        program = "import json,os,sys; p=json.load(sys.stdin); assert p['event']=='pre_compact'; assert 'COMPACTION_SECRET_TOKEN' not in os.environ; print('{}')"
        definition = settings.HookSettings("pre_compact", "real-policy", [sys.executable, "-c", program],
                                           self.session.shell_cwd, on_error="deny")
        pipeline = tool_runtime.build_hook_pipeline([definition], create_exec=loki._tool_process_creator())
        with mock.patch.dict(os.environ, {"COMPACTION_SECRET_TOKEN": "never pass"}), \
                mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline):
            self.assertIn("Context compacted", await loki.compact_context_async(cancel_event=self.event))
        self.assertEqual(len(self.sent), 1)


class CompactionRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_receipt_and_running_duplicates_share_outcome_without_observer_cancellation(self):
        requests = compactions.Requests()
        future = requests.request()
        self.assertIs(requests.request(), future)
        ready, release = asyncio.Event(), asyncio.Event()
        events = []

        async def operation(started):
            started()
            ready.set()
            await release.wait()
            return "done"

        async def observe():
            return await asyncio.shield(future)

        owner = asyncio.create_task(requests.run(operation, events.append))
        observer = asyncio.create_task(observe())
        try:
            await observe_fixture(ready.wait(), owner)
            self.assertIs(requests.request(), future)
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            self.assertFalse(future.cancelled())
            release.set()
            self.assertEqual((await owner)["type"], "compaction_completed")
        finally:
            release.set()
            owner.cancel()
            observer.cancel()
            await asyncio.gather(owner, observer, return_exceptions=True)
        self.assertIsNone(requests.pending)
        self.assertEqual([event["type"] for event in events], ["compaction_started", "compaction_completed"])
        self.assertEqual(events[0]["compaction_id"], events[1]["compaction_id"])
        self.assertEqual(future.result()["text"], "done")
        next_request = requests.request()
        self.assertIsNot(next_request, future)
        requests.finish_pending(cancelled=True)
        self.assertEqual(next_request.result()["type"], "compaction_cancelled")

    async def test_failure_and_task_cancellation_settle_outcomes_without_exception_futures(self):
        requests = compactions.Requests()
        failed = requests.request()

        async def failure(started):
            raise compactions.CompactionError("cannot compact")

        outcome = await requests.run(failure)
        self.assertEqual(outcome["type"], "compaction_failed")
        self.assertIs(outcome, failed.result())
        ready = asyncio.Event()
        cancelled = requests.request()

        async def blocked(started):
            started()
            ready.set()
            await asyncio.Event().wait()

        owner = asyncio.create_task(requests.run(blocked))
        try:
            await observe_fixture(ready.wait(), owner)
            owner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await owner
        finally:
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
        self.assertEqual(cancelled.result()["type"], "compaction_cancelled")
        self.assertIsNone(requests.pending)

    async def test_credential_denial_is_an_operation_failure_not_an_unhandled_frontend_error(self):
        requests = compactions.Requests()
        accepted = requests.request()
        error = authentications.CredentialUnavailable("credential lease unavailable")

        async def denied(started):
            raise error

        result = await requests.run(denied)
        self.assertEqual(result["type"], "compaction_failed")
        self.assertIs(result["error"], error)
        self.assertIs(accepted.result(), result)
        self.assertIsNone(requests.pending)

    def test_private_wakeup_has_no_submission_id_count_or_editable_queue_entry(self):
        queue = terminals.UserMessageQueue()
        queue.put_nowait("first")
        queue.put_nowait(terminals.CONTROL_WAKE)
        queue.put_nowait("second")
        queue.put_nowait(None)
        first, second = queue.pending_entries()
        self.assertEqual(queue.message_count, 2)
        self.assertEqual([first.id, second.id], [1, 2])
        queue.edit_text(second.id, "edited second")
        queue.move_text(first.id)
        self.assertIs(queue.get_nowait(), terminals.CONTROL_WAKE)
        self.assertEqual(queue.message_count, 2)
        self.assertEqual(queue.get_nowait(), "edited second")
        self.assertEqual(queue.get_nowait(), "first")
        self.assertIsNone(queue.get_nowait())
        self.assertEqual(queue.message_count, 0)

    def test_v1_compaction_lifecycle_is_capability_gated_and_summary_free(self):
        for kind, status in [["compaction_started", "in_progress"], ["compaction_completed", "completed"],
                             ["compaction_failed", "failed"], ["compaction_cancelled", "cancelled"]]:
            event = {"type": kind, "compaction_id": "local-id", "text": "readable status"}
            self.assertEqual(acp_events.map_event("session", event, {})[0]["update"]["sessionUpdate"], "agent_message_chunk")
            typed = acp_events.map_event("session", event, {}, compaction=True)[0]["update"]
            self.assertEqual(typed["compactionId"], "local-id")
            self.assertEqual(typed["status"], status)
            self.assertNotIn("summary", typed)
            if kind == "compaction_failed":
                self.assertEqual(typed["error"], "readable status")


class PreCompactSettingsTests(unittest.TestCase):
    def test_gate_defaults_and_forbidden_tool_or_fail_open_options(self):
        base = "[hook.pre_compact.policy]\ncommand.0=policy\ncwd=.\n"
        parser = settings._parser()
        parser.read_file(io.StringIO(base))
        hook = settings.from_parser(parser).hooks[0]
        self.assertIsNone(hook.timeout_ms)
        self.assertEqual(hook.on_error, "deny")
        for extra in ["tools.0=*\n", "on_error=continue\n"]:
            parser = settings._parser()
            parser.read_file(io.StringIO(base + extra))
            with self.subTest(extra=extra), self.assertRaises(settings.HookConfigurationError):
                settings.from_parser(parser)


if __name__ == "__main__":
    unittest.main()
