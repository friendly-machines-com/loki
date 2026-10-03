"""Readable projection, native retention, and trace lifecycle invariants."""

import asyncio
import contextlib
import json
import unittest
from unittest import mock

from loki_agent import formats, loki, protocols, replays, savefiles, sse
from loki_agent.sessions import Session


def feed(accumulator, data):
    return accumulator.feed(sse.SseEvent(event="message", data=json.dumps(data)))


def chat_turn(reasoning="thought", content="answer"):
    return formats.openai_chat_response_to_items({
        "choices": [{"message": {"role": "assistant", "content": content,
                                 "reasoning_content": reasoning}, "finish_reason": "stop"}]})


def response_turn(summary="summary", identity="r"):
    return formats.openai_responses_response_to_items({
        "object": "response", "status": "completed", "output": [
            {"type": "reasoning", "id": identity, "summary": [
                {"type": "summary_text", "text": summary}], "encrypted_content": "SECRET"},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "answer"}]}]})


class ReasoningProjectionTests(unittest.TestCase):
    def test_aliases_and_opaque_details(self):
        fields = {"reasoning_content": "primary", "reasoning": "alias", "reasoning_details": [
            {"type": "reasoning.text", "text": "detail"},
            {"type": "reasoning.encrypted", "data": "SECRET"},
            {"type": "unknown", "text": "SECRET"}]}
        self.assertEqual([segment[2] for segment in formats.chat_reasoning_segments(fields)], ["primary"])
        self.assertEqual([segment[2] for segment in formats.chat_reasoning_segments(fields, source="reasoning")], ["alias"])
        self.assertEqual([segment[2] for segment in formats.chat_reasoning_segments(fields, source="reasoning_details")], ["detail"])

    def test_native_formats_do_not_put_reasoning_in_answers(self):
        turns = [chat_turn(), response_turn(), formats.anthropic_response_to_items({
            "type": "message", "role": "assistant", "content": [
                {"type": "thinking", "thinking": "thought", "signature": "SECRET"},
                {"type": "redacted_thinking", "data": "SECRET"},
                {"type": "text", "text": "answer"}], "stop_reason": "end_turn"})]
        for turn in turns:
            with self.subTest(protocol=turn.metadata.get("protocol")):
                self.assertEqual(formats.item_text(turn.to_event()), "answer")
                segments = formats.reasoning_segments(turn.items)
                self.assertEqual(len(segments), 1)
                self.assertNotIn("SECRET", str(segments))

    def test_replay_visibility_and_retention_are_independent(self):
        events = [response_turn("plain summary").to_event(), chat_turn().to_event()]
        hidden = savefiles.ResumeTranscriptRenderer().presentation(events)
        visible = savefiles.ResumeTranscriptRenderer(show_reasoning=True).presentation(events)
        self.assertFalse(any(kind == "reasoning" for kind, block in hidden))
        text = [value for kind, block in visible if kind == "reasoning"
                for name, value in block if name == "text"]
        replay = replays.classify_transcript(events, show_reasoning=True)
        self.assertEqual(text, [value for kind, value, key in replay if kind == "thought"])
        self.assertEqual(text, ["plain summary", "thought"])
        self.assertNotIn("SECRET", str(visible))
        self.assertIn("SECRET", str(events))
        self.assertFalse(any(kind == "thought" for kind, value, key in replays.classify_transcript(events)))


class ReasoningAccumulatorTests(unittest.TestCase):
    def test_chat_aliases_and_indexed_fragments(self):
        for field in ["reasoning_content", "reasoning", "reasoning_details"]:
            with self.subTest(field=field):
                thoughts, answers = [], []
                acc = protocols.OpenAIChatStreamAccumulator(answers.append, lambda *args: thoughts.append(args))
                for text in ["one", "two"]:
                    value = (text if field != "reasoning_details" else [
                        {"type": "reasoning.text", "index": 0, "id": "r", "text": text, "signature": "sig"}])
                    feed(acc, {"choices": [{"delta": {field: value}}]})
                feed(acc, {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]})
                acc.feed(sse.SseEvent(event="message", data="[DONE]"))
                self.assertEqual(answers, ["answer"])
                self.assertEqual([value[2] for value in thoughts], ["one", "two"])
                turn = formats.openai_chat_response_to_items(acc.finish())
                self.assertEqual(formats.reasoning_segments(turn.items)[0][2], "onetwo")
                if field == "reasoning_details":
                    detail = acc.choice["message"][field][0]
                    self.assertEqual(detail["id"], "r")
                    self.assertEqual(detail["signature"], "sigsig")

    def test_chat_combined_chunk_and_late_aliases(self):
        events = []
        acc = protocols.OpenAIChatStreamAccumulator(
            lambda text: events.append(["answer", text]),
            lambda key, kind, text: events.append(["thought", text]))
        feed(acc, {"choices": [{"delta": {"reasoning": "visible", "content": "answer"}}]})
        feed(acc, {"choices": [{"delta": {"reasoning_content": "alias", "reasoning_details": [
            {"type": "reasoning.text", "text": "detail"}]}}]})
        self.assertEqual(events, [["thought", "visible"], ["answer", "answer"]])
        self.assertEqual(acc.reasoning_field, "reasoning")

    def test_indexed_fragments_need_not_repeat_type(self):
        thoughts = []
        acc = protocols.OpenAIChatStreamAccumulator(lambda text: None, lambda key, kind, text: thoughts.append(text))
        feed(acc, {"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "first"}]}}]})
        feed(acc, {"choices": [{"delta": {"reasoning_details": [{"index": 0, "text": " second"}]}}]})
        self.assertEqual(thoughts, ["first", " second"])

    def test_anthropic_signatures_never_display(self):
        thoughts = []
        acc = protocols.AnthropicMessagesStreamAccumulator(lambda text: None, lambda *args: thoughts.append(args))
        feed(acc, {"type": "message_start", "message": {"role": "assistant"}})
        feed(acc, {"type": "content_block_start", "index": 0,
                   "content_block": {"type": "thinking", "thinking": "initial"}})
        feed(acc, {"type": "content_block_delta", "index": 0,
                   "delta": {"type": "thinking_delta", "thinking": " more"}})
        feed(acc, {"type": "content_block_delta", "index": 0,
                   "delta": {"type": "signature_delta", "signature": "SECRET"}})
        feed(acc, {"type": "message_stop"})
        turn = formats.anthropic_response_to_items(acc.finish())
        self.assertEqual([value[2] for value in thoughts], ["initial", " more"])
        self.assertEqual(formats.reasoning_segments(turn.items)[0][0], thoughts[0][0])
        self.assertEqual(turn.items[0]["signature"], "SECRET")

    def test_sparse_reasoning_parts_are_rejected(self):
        for index in [-1, True, 1000000000, "wrong"]:
            acc = protocols.OpenAIResponsesStreamAccumulator(lambda text: None)
            with self.subTest(index=index), self.assertRaises(protocols.StreamProtocolError):
                feed(acc, {"type": "response.reasoning_summary_text.delta", "output_index": 0,
                           "summary_index": index, "delta": "text"})

    def test_responses_fragments_and_completed_parts_are_delivered_once(self):
        thoughts = []
        acc = protocols.OpenAIResponsesStreamAccumulator(lambda text: None, lambda *args: thoughts.append(args))
        feed(acc, {"type": "response.reasoning_summary_text.delta", "item_id": "r", "output_index": 0,
                   "summary_index": 0, "delta": "first"})
        feed(acc, {"type": "response.reasoning_summary_text.done", "item_id": "r", "output_index": 0,
                   "summary_index": 0, "text": "first"})
        feed(acc, {"type": "response.reasoning_summary_part.done", "item_id": "r", "output_index": 0,
                   "summary_index": 1, "part": {"type": "summary_text", "text": "second"}})
        feed(acc, {"type": "response.completed", "response": {"output": []}})
        self.assertEqual([value[2] for value in thoughts], ["first", "second"])
        turn = formats.openai_responses_response_to_items(acc.finish())
        self.assertEqual(formats.reasoning_segments(turn.items), thoughts)

    def test_terminal_envelope_native_reasoning_is_not_discarded(self):
        acc = protocols.OpenAIResponsesStreamAccumulator(lambda text: None)
        message = {"type": "message", "id": "m", "role": "assistant",
                   "content": [{"type": "output_text", "text": "answer"}]}
        reasoning = {"type": "reasoning", "id": "r", "summary": [
            {"type": "summary_text", "text": "summary"}], "encrypted_content": "SECRET"}
        feed(acc, {"type": "response.output_item.done", "output_index": 1, "item": message})
        feed(acc, {"type": "response.completed", "response": {"output": [reasoning, message]}})
        self.assertEqual(acc.finish()["output"], [reasoning, message])


class ReasoningLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = Session()
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)

    async def test_final_suffix_deduplicates_and_never_contaminates_answer(self):
        events, transcript = [], []

        async def chat(items, on_text_delta, *, codex_turn_state, on_reasoning_delta):
            on_reasoning_delta(("chat", "reasoning_content", 0), "thinking", "tho")
            on_text_delta("answer")
            return chat_turn("thought")

        result = await loki.run_tool_loop_async(
            transcript, chat_fn=chat, stream_chat=True, on_event=events.append,
            thinking=loki.TurnThinkingSettings(traces="on"))
        self.assertEqual(result, "answer")
        self.assertEqual("".join(event["text"] for event in events if event["type"] == "reasoning_delta"), "thought")
        self.assertEqual([event["content"] for event in events if event["type"] == "assistant_delta"], ["answer"])
        self.assertFalse(any(event["type"] == "assistant_message" for event in events))
        self.assertEqual(len(transcript), 1)

    async def test_answer_and_thought_spans_can_interleave(self):
        events = []

        async def chat(items, on_text_delta, *, codex_turn_state, on_reasoning_delta):
            on_text_delta("a")
            on_reasoning_delta(("chat", "reasoning_content", 0), "thinking", "thought")
            on_text_delta("b")
            return chat_turn("thought", "ab")

        await loki.run_tool_loop_async([], chat_fn=chat, stream_chat=True, on_event=events.append,
                                       thinking=loki.TurnThinkingSettings(traces="on"))
        self.assertEqual([event["type"] for event in events], [
            "assistant_start", "assistant_delta", "assistant_end", "reasoning_start", "reasoning_delta",
            "reasoning_end", "assistant_start", "assistant_delta", "assistant_end"])

    async def test_buffered_visibility_and_item_order(self):
        turn = formats.DecodedTurn([
            formats.message_item("assistant", "before"),
            {"type": "anthropic_thinking", "thinking": "middle", "signature": "opaque"},
            formats.message_item("assistant", "after")])

        async def chat(items, *, codex_turn_state):
            return turn

        for visibility in ["off", "on"]:
            events = []
            result = await loki.run_tool_loop_async(
                [], chat_fn=chat, on_event=events.append,
                thinking=loki.TurnThinkingSettings(traces=visibility))
            self.assertEqual(result, "before\nafter")
            visible = [event.get("content", event.get("text")) for event in events
                       if event["type"] in ["assistant_message", "reasoning_delta"]]
            self.assertEqual(visible, ["before", "middle", "after"] if visibility == "on" else ["before\nafter"])

    async def test_partial_thinking_closes_without_persisting_transport_fragments(self):
        for exception in [loki.StreamCancelled(), protocols.StreamProtocolError("broken"), asyncio.CancelledError()]:
            events, transcript = [], []

            async def chat(items, on_text_delta, *, codex_turn_state, on_reasoning_delta):
                on_reasoning_delta(("chat", "reasoning_content", 0), "thinking", "partial")
                raise exception

            operation = loki.run_tool_loop_async(
                transcript, chat_fn=chat, stream_chat=True, on_event=events.append,
                thinking=loki.TurnThinkingSettings(traces="on"))
            if isinstance(exception, asyncio.CancelledError):
                with self.assertRaises(asyncio.CancelledError):
                    await operation
            else:
                await operation
            self.assertEqual(transcript, [])
            self.assertFalse(next(event for event in events if event["type"] == "reasoning_end")["complete"])

    async def test_late_stream_alias_has_same_saved_replay(self):
        self.session.runtime_config = loki.make_runtime_config(
            "https://gateway.example/v1", protocols.OPENAI_CHAT, model="exact", stream=True)
        self.session.reasoning_traces = "on"

        async def body():
            for delta in [{"reasoning": "visible"}, {"reasoning_content": "alias"}, {"content": "answer"}]:
                yield ("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n").encode()
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b'data: [DONE]\n\n'

        @contextlib.asynccontextmanager
        async def stream(method, url, **kwargs):
            yield loki.http_client.HttpStreamResponse(url, 200, "OK", {"content-type": "text/event-stream"}, body())

        events = []
        with mock.patch.object(loki.http_client, "async_http_stream", stream):
            await loki.run_tool_loop_async(self.session.transcript_items, stream_chat=True, on_event=events.append)
        self.assertEqual("".join(event["text"] for event in events if event["type"] == "reasoning_delta"), "visible")
        self.assertEqual(formats.reasoning_field(self.session.transcript_items[0]), "reasoning")
        replay = replays.classify_transcript(self.session.transcript_items, show_reasoning=True)
        self.assertEqual([text for kind, text, key in replay if kind == "thought"], ["visible"])
        native = formats.items_to_openai_chat_messages(self.session.transcript_items)[0]
        self.assertEqual(native["reasoning"], "visible")
        self.assertEqual(native["reasoning_content"], "alias")

    async def test_no_retry_after_thought_output(self):
        self.session.runtime_config = loki.make_runtime_config(
            "https://api.openai.com/v1", protocols.OPENAI_RESPONSES, model="gpt-5", stream=True)

        async def failed_request(*args, **kwargs):
            kwargs["on_reasoning_delta"](("responses", "r", "summary", 0), "summary", "partial")
            raise protocols.ResponseApiError("retryable", retryable=True)

        once = mock.AsyncMock(side_effect=failed_request)
        with mock.patch.object(loki, "_async_chat_stream_request_once", once):
            with self.assertRaises(protocols.ResponseApiError):
                await loki.async_chat_stream_request(
                    self.session.runtime_config.chat_provider.chat_url, {}, on_reasoning_delta=lambda *args: None)
        self.assertEqual(once.await_count, 1)
