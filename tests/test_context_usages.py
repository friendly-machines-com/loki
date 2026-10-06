"""Context displays describe one reported exchange, never a token budget."""

import asyncio
import contextlib
import io
import json
import unittest
from dataclasses import replace
from unittest import mock

from loki_agent import acp_events, formats, loki, models, protocols, sse
from loki_agent import terminal_frontend, usages
from loki_agent.authentications import CredentialRef
from loki_agent.connections import ConnectionDescriptor, ConnectionDescriptorError
from loki_agent.credentials import CredentialInventory
from loki_agent.sessions import Session
from loki_endpoints import assume_endpoints_approved


CAPACITY = usages.ContextCapacity(1000, "configured")


def config(**overrides):
    values = dict(model="alias", context_capacity=CAPACITY)
    values.update(overrides)
    return loki.make_runtime_config(
        "https://example.test/v1", protocols.OPENAI_CHAT, **values)


def response(session, input_tokens=300, output_tokens=70, **overrides):
    provider = session.runtime_config.chat_provider
    values = dict(
        provider=provider.provider_id or provider.provider_name,
        endpoint=provider.chat_url,
        model="resolved-model", requested_model=session.model,
        usage={"prompt_tokens": input_tokens,
               "completion_tokens": output_tokens},
    )
    values.update(overrides)
    return formats.model_response_event(provider.kind, [], **values)


class NormalizationTests(unittest.TestCase):
    def test_protocol_formats_and_nonadditive_details(self):
        examples = [
            (protocols.OPENAI_CHAT, {
                "prompt_tokens": 300, "completion_tokens": 70,
                "total_tokens": 370,
                "prompt_tokens_details": {"cached_tokens": 200},
                "completion_tokens_details": {"reasoning_tokens": 50},
            }),
            (protocols.OPENAI_RESPONSES, {
                "input_tokens": 300, "output_tokens": 70,
                "input_tokens_details": {"cached_tokens": 200},
                "output_tokens_details": {"reasoning_tokens": 50},
            }),
            (protocols.ANTHROPIC_MESSAGES, {
                "input_tokens": 50, "output_tokens": 70,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 50,
                "cache_creation": {"ephemeral_5m_input_tokens": 50},
            }),
        ]
        for protocol, raw in examples:
            with self.subTest(protocol=protocol):
                usage = usages.normalize_usage(protocol, raw)
                self.assertEqual(usage, usages.ResponseUsage(300, 70))
                self.assertEqual(usage.used, 370)

    def test_missing_malformed_and_zero_are_distinct(self):
        for raw in [None, [], {}, {"total_tokens": 123}]:
            self.assertIsNone(usages.normalize_usage(protocols.OPENAI_CHAT, raw))
        for invalid in [-1, True, 1.5, "100", None]:
            for field in ["prompt_tokens", "completion_tokens"]:
                raw = {"prompt_tokens": 100, "completion_tokens": 10}
                raw[field] = invalid
                self.assertIsNone(
                    usages.normalize_usage(protocols.OPENAI_CHAT, raw))
        self.assertEqual(usages.normalize_usage(protocols.OPENAI_CHAT, {
            "prompt_tokens": 0, "completion_tokens": 0,
        }).used, 0)
        self.assertIsNone(usages.normalize_usage(protocols.ANTHROPIC_MESSAGES, {
            "input_tokens": 50, "output_tokens": 10,
            "cache_read_input_tokens": -1,
        }))
        self.assertIsNone(usages.normalize_usage("unknown", {}))

    def test_integer_rounding_unknown_and_over_capacity(self):
        for used, expected in ((0, "0%"), (5, "1%"), (370, "37%"),
                               (1200, "120%")):
            snapshot = usages.ContextSnapshot(
                usages.ResponseUsage(used, 0), CAPACITY)
            self.assertEqual(snapshot.text, expected)
        self.assertEqual(usages.ContextSnapshot(None, CAPACITY).text, "unknown")
        self.assertEqual(usages.ContextSnapshot(
            usages.ResponseUsage(1, 1), None).text, "unknown")

    def test_chat_stream_requests_and_retains_usage_only_final_chunk(self):
        provider = config().chat_provider
        payload = provider.streaming_chat_payload([], [], "alias")
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertNotIn("stream_options", provider.chat_payload([], [], "alias"))
        accumulator = provider.stream_accumulator()
        for chunk in [
            {"choices": [{"index": 0, "delta": {
                "role": "assistant", "content": "Hi"},
                "finish_reason": "stop"}], "usage": None},
            {"choices": [], "usage": {
                "prompt_tokens": 300, "completion_tokens": 70}},
        ]:
            accumulator.feed(sse.SseEvent("message", json.dumps(chunk)))
        accumulator.feed(sse.SseEvent("message", "[DONE]"))
        turn = provider.parse_chat_response(accumulator.finish())
        self.assertEqual(usages.normalize_usage(
            provider.kind, turn.metadata["usage"]).used, 370)

    def test_anthropic_stream_replaces_cumulative_counts(self):
        provider = protocols.make_provider(
            "https://example.test/v1/messages",
            provider=protocols.ANTHROPIC_MESSAGES)
        self.assertNotIn("stream_options", provider.streaming_chat_payload(
            [], [], "claude"))
        accumulator = provider.stream_accumulator()
        events = [
            {"type": "message_start", "message": {
                "id": "msg", "type": "message", "role": "assistant",
                "content": [], "model": "claude", "usage": {
                    "input_tokens": 10, "output_tokens": 1,
                    "cache_read_input_tokens": 200,
                    "cache_creation_input_tokens": 50}}},
            {"type": "message_delta", "delta": {},
             "usage": {"output_tokens": 20}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"input_tokens": 50, "output_tokens": 70}},
            {"type": "message_stop"},
        ]
        for event in events:
            accumulator.feed(sse.SseEvent(event["type"], json.dumps(event)))
        turn = provider.parse_chat_response(accumulator.finish())
        self.assertEqual(usages.normalize_usage(
            provider.kind, turn.metadata["usage"]).used, 370)


class SessionSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.session = Session(runtime_config=config())

    def append_response(self, **kwargs):
        self.session.transcript_items.append(response(self.session, **kwargs))
        return self.session.context_snapshot(live=True)

    def test_latest_exchange_not_sum_and_tool_results_are_unmeasured(self):
        self.assertEqual(self.append_response().text, "37%")
        self.session.transcript_items.append(formats.message_item(
            "user", [formats.text_block("another input")]))
        self.assertEqual(self.session.context_snapshot().text, "37%*")
        self.assertEqual(self.append_response(input_tokens=400).text, "47%")
        self.session.transcript_items.append(formats.tool_result_item(
            "call", "unmeasured tool result"))
        self.assertEqual(self.session.context_snapshot().text, "47%*")

    def test_missing_usage_and_incomplete_response(self):
        self.append_response()
        self.assertEqual(self.append_response(usage=None).text, "unknown")
        self.assertEqual(self.append_response(status="incomplete").text, "37%*")

    def test_model_endpoint_and_transcript_replacement(self):
        self.append_response()
        self.session.runtime_config = config(model="different")
        self.assertEqual(self.session.context_snapshot().text, "unknown")
        self.session.runtime_config = config()
        self.assertEqual(self.session.context_snapshot().text, "37%*")
        self.session.runtime_config = loki.make_runtime_config(
            "https://elsewhere.test/v1", protocols.OPENAI_CHAT,
            model="alias", context_capacity=CAPACITY)
        self.assertEqual(self.session.context_snapshot().text, "unknown")
        self.session.runtime_config = config()
        self.session.replace_transcript([], [], [], {}, "new-chat.json")
        self.assertEqual(self.session.context_snapshot().text, "unknown")

    def test_resume_raw_events_and_selected_model_alias(self):
        self.append_response()
        saved = json.loads(json.dumps(self.session.transcript_items))
        resumed = Session(runtime_config=config())
        resumed.replace_transcript(saved, [], [], {}, "saved-chat.json")
        self.assertEqual(resumed.context_snapshot().text, "37%*")
        # Old events can still be matched when the effective model is exact.
        saved[0].pop("requested_model")
        saved[0]["model"] = "alias"
        resumed = Session(runtime_config=config(), transcript_items=saved)
        self.assertEqual(resumed.context_snapshot().text, "37%*")

    def test_child_or_helper_usage_does_not_modify_parent(self):
        self.append_response()
        child = Session(runtime_config=config())
        child.transcript_items.append(response(child, input_tokens=900))
        self.assertEqual(child.context_snapshot(live=True).text, "97%")
        self.assertEqual(self.session.context_snapshot().text, "37%")

    def test_unknown_capacity_preserves_observation(self):
        self.append_response()
        self.session.runtime_config = config(context_capacity=None)
        snapshot = self.session.context_snapshot()
        self.assertEqual(snapshot.usage.used, 370)
        self.assertEqual(snapshot.text, "unknown")
        self.assertIsNone(acp_events.context_usage("session", snapshot))


class CapacityConfigurationTests(unittest.TestCase):
    def test_catalog_validation_and_distinct_limits(self):
        for invalid in [None, 0, -1, True, "1000", 1000.5, {}]:
            self.assertIsNone(models.context_capacity({}, {
                "limit": {"context": invalid, "input": 100, "output": 50}}))
        capacity = models.context_capacity({}, {
            "limit": {"context": 1000, "input": 100, "output": 50}})
        self.assertEqual(capacity, usages.ContextCapacity(1000, "models.dev"))

    def test_selected_catalog_leaf_and_environment_override(self):
        assume_endpoints_approved(self)
        provider = {
            "id": "provider", "name": "Provider", "api": "https://example.test/v1",
            "npm": "@ai-sdk/openai-compatible", "env": ["EXAMPLE_KEY"],
        }
        model = {"id": "alias", "limit": {"context": 3000, "output": 200}}
        credential = CredentialRef.environment("EXAMPLE_KEY")
        runtime = loki.config_from_modelsdev_selection(
            "provider", provider, model, CredentialInventory({}, [credential]))
        self.assertEqual(runtime.context_capacity,
                         usages.ContextCapacity(3000, "models.dev"))
        self.assertEqual(runtime.chat_provider.max_tokens, 4096)
        overridden = loki.config_from_modelsdev_selection(
            "provider", provider, model,
            CredentialInventory({"LOKI_CONTEXT_WINDOW": "1000"}, [credential]))
        self.assertEqual(overridden.context_capacity, CAPACITY)

    def test_configured_override_and_descriptor_round_trip(self):
        descriptor = loki.connection_descriptor_from_config(config())
        restored = ConnectionDescriptor.from_dict(descriptor.to_dict())
        self.assertEqual(restored.context_capacity, CAPACITY)
        inventory = CredentialInventory({"LOKI_CONTEXT_WINDOW": "2000"})
        rebuilt = loki.config_from_connection_descriptor(restored, inventory)
        self.assertEqual(rebuilt.context_capacity.tokens, 2000)
        changed = loki.config_from_connection_descriptor(
            restored, CredentialInventory({"LOKI_MODEL": "other"}))
        self.assertIsNone(changed.context_capacity)
        for invalid in [0, True, -1, "1000"]:
            value = descriptor.to_dict()
            value["context_capacity"]["tokens"] = invalid
            with self.assertRaises(ConnectionDescriptorError):
                ConnectionDescriptor.from_dict(value)

    def test_environment_and_child_config_clear_stale_capacity(self):
        values = {"LOKI_API_BASE": "https://example.test/v1",
                  "LOKI_PROVIDER": protocols.OPENAI_CHAT,
                  "LOKI_MODEL": "alias", "LOKI_CONTEXT_WINDOW": "1000"}
        runtime = loki.build_config_from_env(credentials=CredentialInventory(values))
        self.assertEqual(runtime.context_capacity, CAPACITY)
        for invalid in ["0", "-1", "broken", "1.5"]:
            with self.assertRaisesRegex(ValueError, "LOKI_CONTEXT_WINDOW"):
                loki.build_config_from_env(credentials=CredentialInventory(
                    dict(values, LOKI_CONTEXT_WINDOW=invalid)))
        session = Session(runtime_config=runtime)
        with mock.patch.object(loki, "_DEFAULT_SESSION", session):
            env = loki._subagent_env(environ={})
            self.assertEqual(env["LOKI_CONTEXT_WINDOW"], "1000")
            loki.reinstall_provider(model="another")
            self.assertIsNone(session.runtime_config.context_capacity)
            self.assertNotIn("LOKI_CONTEXT_WINDOW", loki._subagent_env(
                environ={"LOKI_CONTEXT_WINDOW": "9999"}))

    def test_subscription_capacity_refresh_and_explicit_precedence(self):
        entry = {"slug": "alias", "visibility": "list",
                 "supports_parallel_tool_calls": True,
                 "supports_reasoning_summaries": False,
                 "support_verbosity": False,
                 "context_window": 1000, "max_context_window": 2000,
                 "auto_compact_token_limit": 800,
                 "effective_context_window_percent": 95}
        catalog = models.add_openai_subscription_catalog({}, {"models": [entry]})
        provider = catalog[models.OPENAI_SUBSCRIPTION_PROVIDER_ID]
        model = provider["models"]["alias"]
        capacity = models.context_capacity(provider, model)
        self.assertEqual(capacity, usages.ContextCapacity(1000, "openai-subscription"))
        descriptor = ConnectionDescriptor(
            provider_id=models.OPENAI_SUBSCRIPTION_PROVIDER_ID,
            provider_name="ChatGPT", model="alias",
            chat_url=provider["api"], models_url=None,
            protocol=protocols.OPENAI_RESPONSES,
            credential_ref=CredentialRef.openai_subscription(),
            openai_request_profile=models.openai_request_profile(provider, model),
            context_capacity=usages.ContextCapacity(500, "openai-subscription"))
        self.assertEqual(loki.reconcile_connection_descriptor(
            descriptor, catalog).context_capacity, capacity)
        self.assertEqual(loki.reconcile_connection_descriptor(
            descriptor, {}).context_capacity.tokens, 500)
        explicit = replace(descriptor, context_capacity=CAPACITY)
        self.assertEqual(loki.reconcile_connection_descriptor(
            explicit, catalog).context_capacity, CAPACITY)
        entry["context_window"] = "invalid"
        catalog = models.add_openai_subscription_catalog({}, {"models": [entry]})
        self.assertIsNone(loki.reconcile_connection_descriptor(
            descriptor, catalog).context_capacity)


class PresentationTests(unittest.TestCase):
    def test_acp_updates_each_response_and_records_requested_model(self):
        from loki_agent.acp_worker import Worker

        session = Session(runtime_config=config())
        messages = []
        worker = Worker(session, messages.append, "s")
        responses = [
            protocols.ProviderResponse({
                "choices": [{
                    "message": {
                        "role": "assistant", "content": None,
                        "tool_calls": [{"id": "c", "type": "function", "function": {
                            "name": "TodoRead", "arguments": "{}"}}],
                    },
                    "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 70},
            }, effective_model="resolved-model"),
            protocols.ProviderResponse({
                "choices": [{"message": {"role": "assistant", "content": "Done"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 80},
            }, effective_model="resolved-model"),
        ]
        with mock.patch.object(loki, "_DEFAULT_SESSION", session), \
                mock.patch.object(loki, "save_chat_log"), \
                mock.patch.object(loki, "async_provider_request",
                                  new=mock.AsyncMock(side_effect=responses)):
            asyncio.run(worker._run_turn(lambda event: None, None))
        updates = [m["params"]["update"] for m in messages]
        self.assertEqual([u["used"] for u in updates], [370, 480])
        self.assertTrue(all(u["sessionUpdate"] == "usage_update" for u in updates))
        self.assertEqual(session.context_snapshot().text, "48%")
        events = [e for e in session.transcript_items if e["type"] == "model_response"]
        self.assertEqual(events[0]["model"], "resolved-model")
        self.assertEqual(events[0]["requested_model"], "alias")

    def test_acp_load_restores_usage_but_resume_does_not_replay(self):
        from loki_agent.acp_worker import Worker, PendingSessionOpen, _UNCHANGED

        for method in ["session/load", "session/resume"]:
            with self.subTest(method=method):
                session = Session(runtime_config=config())
                session.transcript_items.append(response(session))
                runtime = session.runtime_config
                session.runtime_config = None
                messages = []
                worker = Worker(session, messages.append, "s")
                worker._pending_open = PendingSessionOpen(runtime, _UNCHANGED, method)
                with mock.patch.object(loki, "_DEFAULT_SESSION", session):
                    worker.commit_open()
                self.assertEqual(session.context_snapshot().text, "37%*")
                self.assertEqual(len(messages), 1 if method == "session/load" else 0)
                if messages:
                    self.assertEqual(messages[0]["params"]["update"], {
                        "sessionUpdate": "usage_update", "used": 370, "size": 1000})

    def test_cancelled_or_failed_exchange_cannot_publish_new_usage(self):
        from loki_agent.acp_worker import Worker

        for error in (loki.StreamCancelled(), OSError("offline")):
            with self.subTest(error=type(error).__name__):
                session = Session(runtime_config=config())
                session.transcript_items.append(response(session))
                session.context_snapshot(live=True)
                session.transcript_items.append(formats.message_item("user", "Next"))
                messages = []
                worker = Worker(session, messages.append)
                with mock.patch.object(loki, "_DEFAULT_SESSION", session), \
                        mock.patch.object(loki, "save_chat_log"), \
                        mock.patch.object(loki, "async_chat_completion",
                                          new=mock.AsyncMock(side_effect=error)):
                    asyncio.run(worker._run_turn(lambda event: None, None))
                self.assertEqual(messages, [])
                self.assertEqual(session.context_snapshot().text, "37%*")

    def test_terminal_renderers_and_acp_use_the_same_snapshot(self):
        session = Session(runtime_config=config())
        session.transcript_items.append(response(session))
        snapshot = session.context_snapshot(live=True)
        with mock.patch.object(loki, "_DEFAULT_SESSION", session):
            self.assertIn("Model: alias, Context: 37%;", terminal_frontend.status_text())
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(
                    terminal_frontend.terminal, "write_text",
                    side_effect=lambda text: print(text, end="")):
                terminal_frontend._write_status_text()
            self.assertIn("Model: alias, Context: 37%;", out.getvalue())
        self.assertEqual(acp_events.context_usage("s", snapshot), {
            "sessionId": "s", "update": {
                "sessionUpdate": "usage_update", "used": 370, "size": 1000}})

    def test_terminal_redraws_stale_usage_after_tool_results(self):
        session = Session(runtime_config=config())
        metadata = {key: value for key, value in response(session).items()
                    if key != "items"}
        first = formats.DecodedTurn(
            [formats.tool_call_item("c", "TodoRead", {})], metadata)
        last_metadata = dict(metadata, usage={
            "prompt_tokens": 400, "completion_tokens": 80})
        last = formats.DecodedTurn(
            [formats.message_item("assistant", "Done")], last_metadata)
        displays = []
        with mock.patch.object(loki, "_DEFAULT_SESSION", session), \
                mock.patch.object(terminal_frontend, "async_chat_completion",
                                  new=mock.AsyncMock(side_effect=[first, last])), \
                mock.patch.object(terminal_frontend, "_terminal_agent_event"), \
                mock.patch.object(terminal_frontend.terminals, "redraw_status_bar",
                                  side_effect=lambda: displays.append(
                                      session.context_snapshot().text)):
            asyncio.run(terminal_frontend.run_terminal_turn_async(
                session.transcript_items))
        self.assertEqual(displays, ["unknown", "37%", "37%*", "48%"])

    def test_terminal_response_callback_updates_snapshot_and_redraws(self):
        session = Session(runtime_config=config())
        event = response(session)
        turn = formats.DecodedTurn(
            [formats.message_item("assistant", [formats.text_block("Hi")])],
            {key: value for key, value in event.items() if key != "items"})
        # DecodedTurn uses provider_id/provider_name before to_event().
        turn.metadata["provider_id"] = event.get("provider")
        with mock.patch.object(loki, "_DEFAULT_SESSION", session), \
                mock.patch.object(terminal_frontend, "async_chat_completion",
                                  new=mock.AsyncMock(return_value=turn)), \
                mock.patch.object(terminal_frontend, "_terminal_agent_event"), \
                mock.patch.object(terminal_frontend.terminals,
                                  "redraw_status_bar") as redraw:
            asyncio.run(terminal_frontend.run_terminal_turn_async(
                session.transcript_items))
        self.assertEqual(session.context_snapshot().text, "37%")
        redraw.assert_called()
