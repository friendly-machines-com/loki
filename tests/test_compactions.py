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
from loki_agent import replays, response_headers, savefiles, settings, sse, terminal_frontend, tool_runtime, usages
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
        self.assertNotIn(model_requests.CODEX_TURN_STATE_HEADER, kwargs["headers_in"])
        self.assertIsNone(kwargs["response_timeout"])
        payload = json.loads(kwargs["body"])
        self.sent.append(payload)
        # Match the actual admitted Lite contract, not just SSE syntax. The
        # previous stub accepted an illegal late system message after a checkpoint.
        self.assertNotIn("system", [item.get("role") for item in payload["input"]])
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
            self.assertNotIn("uncommented-tool-fact", json.dumps(payload))
            self.assertIn("opaque-state", json.dumps(payload))
            output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ordinary answer"}]}]
        if self.mutate is not None:
            self.mutate()

        async def chunks():
            for index, item in enumerate(output):
                wire = ("data: " + json.dumps({"type": "response.output_item.done", "output_index": index, "item": item}) + "\n\n").encode()
                yield wire[:3]
                yield wire[3:]
            terminal = "response.incomplete" if self.mode == "incomplete" else "response.completed"
            yield ("data: " + json.dumps({"type": terminal, "response": {
                "object": "response", "status": "incomplete" if self.mode == "incomplete" else "completed", "output": [],
                "usage": {"input_tokens": 800, "output_tokens": 50}}}) + "\n\n").encode()
        yield http_client.HttpStreamResponse(url, 200, "OK", {
            "content-type": "text/event-stream", model_requests.CODEX_TURN_STATE_HEADER: "operation-state"}, chunks())

    async def test_commit_is_saved_before_install_and_next_prompt_uses_window(self):
        old_id = self.session.conversation_id
        todos = self.session.session_todos
        cwd = self.session.shell_cwd
        self.assertFalse(self.session.chat_log_dirty)
        text = await loki.compact_context_async(cancel_event=self.event)
        self.assertIn("Context compacted", text)
        self.assertIsNot(self.session.transcript_items, self.before_archive)
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
        self.assertIn("additional_tools", [item.get("type") for item in self.sent[0]["input"]])
        self.assertEqual(self.sent[0]["input"][-1], {"type": "compaction_trigger"})

    async def test_compact_before_first_turn_preserves_mode_guard_and_legal_lite_roles(self):
        # Reproduce the operator's failed chat: initial saved instructions, then
        # /compact BEFORE the first ordinary turn records its mode instruction.
        self.session.transcript_items = [formats.message_item("system", "saved initial instructions")]
        self.before_archive = self.session.transcript_items
        self.before = copy.deepcopy(self.before_archive)
        self.assertIsNone(self.session.last_instructed_agent_mode)
        await loki.compact_context_async(cancel_event=self.event)
        self.assertIsNone(self.session.last_instructed_agent_mode)
        loki.record_agent_mode_instruction()
        self.session.transcript_items.append(formats.message_item("user", "first normal prompt"))
        before_request = copy.deepcopy(self.session.transcript_items)
        turn = await model_requests.async_chat_completion(
            self.session.transcript_items, None, context=loki._model_request_context(), thinking=loki.capture_turn_settings())
        self.assertEqual(formats.item_text(turn.to_event()), "ordinary answer")
        self.assertEqual([item.get("type") for item in self.sent[-1]["input"]], ["message", "compaction", "message", "message"])
        self.assertEqual([item.get("role") for item in self.sent[-1]["input"]], ["developer", None, "developer", "user"])
        self.assertIn("Normal mode is active", self.sent[-1]["input"][2]["content"][0]["text"])
        self.assertEqual(self.session.transcript_items, before_request)
        self.assertEqual(self.session.transcript_items[-2]["role"], "system")
        self.assertEqual(self.session.last_instructed_agent_mode, "normal")
        count = len(self.session.transcript_items)
        loki.record_agent_mode_instruction()
        self.assertEqual(len(self.session.transcript_items), count)
        loaded, _todos = formats.load_log_blob(formats.new_log_blob(self.session.transcript_items, []))
        replayed = self.session.runtime_config.chat_provider.streaming_chat_payload(loaded, None, self.session.model)
        self.assertEqual(replayed["input"], self.sent[-1]["input"])

    async def test_late_local_instructions_keep_order_and_authority_in_lite_and_full(self):
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
                             ["user", "assistant", "developer" if provider.provider_id == "openai-subscription" else "system", "developer", "user"])
            self.assertEqual(inputs[2]["content"][0]["text"], "late cwd or mode instruction")
            self.assertEqual(inputs[3]["content"][0]["text"], "late developer instruction")
            self.assertEqual(history, original)

    async def test_lite_does_not_promote_provider_system_output_to_local_instructions(self):
        provider = self.session.runtime_config.chat_provider
        returned = formats.model_response_event(
            formats.OPENAI_RESPONSES, [formats.message_item("system", "provider instruction must not be promoted")],
            provider=provider.provider_id, endpoint=provider.chat_url, model=self.session.model)
        with self.assertRaises(formats.ProjectionError):
            provider.streaming_chat_payload([returned], None, self.session.model)

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
            with self.subTest(mode=mode), self.assertRaises(compactions.CompactionError):
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

        def save(path, content):
            document = json.loads(content)
            self.assertIs(self.session.transcript_items, self.before_archive)
            calls.append(document)
            if len(calls) == 2:
                raise OSError("candidate save failed")
            return writer(path, content)
        with mock.patch.object(loki.files, "atomic_write_text", side_effect=save):
            with self.assertRaisesRegex(OSError, "candidate save"):
                await loki.compact_context_async(cancel_event=self.event)
        self.assertIs(self.session.transcript_items, self.before_archive)
        self.assertEqual(self.session.transcript_items, self.before)
        calls.clear()

        def committed(path, content):
            save_result = writer(path, content)
            calls.append(content)
            if len(calls) == 2:
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

        def commit(path, content):
            writer(path, content)
            if json.loads(content)["events"][-1].get("type") == "context_replace":
                self.event.set()
        with mock.patch.object(loki.files, "atomic_write_text", side_effect=commit):
            self.assertEqual(await worker.prompt(self.params("/compact")), {"stopReason": "end_turn"})
        self.assertTrue(self.event.is_set())
        self.assertEqual(self.session.transcript_items[-1]["type"], "context_replace")

    async def test_tui_queued_command_runs_before_user_insertion_and_restores_idle(self):
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
                    outer.assertEqual(outer.session.transcript_items[-1]["type"], "context_replace")
                return await super().get()
        inputs = Inputs(["/compact", "after compact", "/quit"])
        inventory = loki.CredentialInventory({
            "LOKI_API_BASE": authentications.OPENAI_CHATGPT_RESPONSES_URL,
            "LOKI_PROVIDER": formats.OPENAI_RESPONSES, "LOKI_MODEL": self.session.model,
            "LOKI_CREDENTIAL_REF": authentications.CredentialRef.openai_subscription().encode(),
            "LOKI_OPENAI_REQUEST_PROFILE": json.dumps(self.session.runtime_config.chat_provider.openai_request_profile.to_dict()),
            "LOKI_STREAM": "1"}, self.broker.available())
        output = io.StringIO()
        with mock.patch.object(loki, "CREDENTIALS", inventory), \
                mock.patch.object(terminal_frontend, "input_session", return_value=inputs), \
                mock.patch.object(terminal_frontend.terminals, "open_terminal_stdin"), \
                mock.patch.object(terminal_frontend, "new_chat_log_path", return_value=self.session.chat_log_path), \
                mock.patch.object(terminal_frontend, "restore_output_area_after_input"), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            self.assertEqual(await terminal_frontend.async_main([], settings.Settings()), 0)
        self.assertFalse(terminal_frontend._terminal_activity.turn_running)
        self.assertEqual(len(self.sent), 2)
        self.assertIn("Context compacted", output.getvalue())
        self.assertNotIn('"text": "/compact"', json.dumps(self.session.transcript_items))
        self.assertIn("/compact", terminal_frontend.status_text())

    async def test_real_precompact_process_uses_sanitized_existing_runner(self):
        program = "import json,os,sys; p=json.load(sys.stdin); assert p['event']=='pre_compact'; assert 'COMPACTION_SECRET_TOKEN' not in os.environ; print('{}')"
        definition = settings.HookSettings("pre_compact", "real-policy", [sys.executable, "-c", program],
                                           self.session.shell_cwd, on_error="deny")
        pipeline = tool_runtime.build_hook_pipeline([definition], create_exec=loki._tool_process_creator())
        with mock.patch.dict(os.environ, {"COMPACTION_SECRET_TOKEN": "never pass"}), \
                mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline):
            self.assertIn("Context compacted", await loki.compact_context_async(cancel_event=self.event))
        self.assertEqual(len(self.sent), 1)


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
