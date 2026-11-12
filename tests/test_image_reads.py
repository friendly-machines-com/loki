"""Visual Read must reach the model, not merely acknowledge a file name."""

import base64
import contextlib
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from config_fixtures import setUpModule as set_up_config
from image_fixtures import PNG_BYTES
from response_header_fixtures import setUpModule as set_up_response_headers

from loki_agent import acp_events, acps, attachments, connections, files, formats, loki
from loki_agent import model_requests, openai_models, process_outputs, protocols, replays, savefiles, settings
from loki_agent import subagents, terminal_frontend, tool_runtime


def setUpModule():
    set_up_config()
    set_up_response_headers()


def image_block():
    return attachments.image_attachment_from_bytes("image", PNG_BYTES).content_block()


class ImageReadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.session = loki.Session(
            shell_cwd=self.directory,
            runtime_config=connections.make_runtime_config(
                "http://example.invalid/v1", protocols.DUMMY, model="test-model"))
        self.addCleanup(self.session.response_headers.close)
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)

    def make_file(self, payload=PNG_BYTES, suffix=".png"):
        # tempfile supplies the actual lookup key. No test-side path rewriting
        # or process-cwd changes may select a different file than Read opens.
        descriptor, path = tempfile.mkstemp(dir=self.directory, suffix=suffix)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        return path

    async def execute(self, path, *, pipeline=None, **arguments):
        call = formats.tool_call_item("read-image", "Read", {"file_path": path, **arguments})
        result, metadata = await loki.execute_tool_call_async(
            call, hook_pipeline=pipeline or tool_runtime.ToolHookPipeline())
        return call, result, metadata

    async def test_read_returns_image_bytes_and_observes_the_same_snapshot(self):
        for suffix in [".png", ".jpeg", ".dat", ""]:
            with self.subTest(suffix=suffix):
                path = self.make_file(suffix=suffix)
                _, result, metadata = await self.execute(path, limit=1)
                self.assertTrue(result["ok"], result)
                self.assertIsInstance(result["content"], list)
                self.assertEqual(result["content"][-1], image_block())
                self.assertNotIn("defaults", metadata or {})
                self.assertNotIn("offset", result["content"][0]["text"])
                observed = self.session.file_observations[path]
                sent = base64.b64decode(result["content"][-1]["source"]["data"])
                self.assertEqual(sent, PNG_BYTES)
                self.assertEqual(observed.digest, hashlib.sha256(sent).hexdigest())
                self.assertEqual(observed.size, len(sent))
                with open(path, "wb") as stream:
                    stream.write(b"changed after reading")
                self.assertEqual(base64.b64decode(result["content"][-1]["source"]["data"]), PNG_BYTES)

    async def test_other_supported_signatures_are_returned_as_images(self):
        # Local validation deliberately checks signatures, not pixel decoding.
        for payload, mime in [[b"\xff\xd8\xffjpeg", "image/jpeg"],
                              [b"GIF89agif", "image/gif"],
                              [b"RIFF\x04\x00\x00\x00WEBP", "image/webp"]]:
            with self.subTest(mime=mime):
                path = self.make_file(payload, suffix="")
                _, result, _ = await self.execute(path, offset=9999, limit=1)
                self.assertTrue(result["ok"], result)
                source = result["content"][-1]["source"]
                self.assertEqual(source["media_type"], mime)
                self.assertEqual(base64.b64decode(source["data"]), payload)

    async def test_invalid_missing_non_file_and_oversize_reads_fail_honestly(self):
        invalid = self.make_file(b"not an image")
        missing = self.make_file()
        os.unlink(missing)
        for path, diagnostic in [[invalid, "unsupported image data"],
                                 [missing, "File not found"],
                                 [self.directory, "directory"]]:
            with self.subTest(diagnostic=diagnostic):
                _, result, _ = await self.execute(path)
                self.assertFalse(result["ok"], result)
                self.assertIn(diagnostic, result["content"])
                self.assertNotIn(path, self.session.file_observations)
        path = self.make_file()
        with mock.patch.object(attachments, "IMAGE_ATTACHMENT_MAX_BYTES", len(PNG_BYTES) - 1):
            _, result, _ = await self.execute(path)
        self.assertFalse(result["ok"], result)
        self.assertIn("maximum", result["content"])
        self.assertNotIn(path, self.session.file_observations)
        with mock.patch.object(attachments, "IMAGE_ATTACHMENT_MAX_BYTES", len(PNG_BYTES)):
            _, result, _ = await self.execute(path)
        self.assertTrue(result["ok"], result)

    async def test_change_during_image_read_is_not_recorded_and_descriptor_closes(self):
        path = self.make_file()
        encode = attachments.image_attachment_from_bytes

        def change_after_snapshot(*args, **kwargs):
            image = encode(*args, **kwargs)
            with open(path, "ab") as stream:
                stream.write(b"changed")
            return image

        with mock.patch.object(attachments, "image_attachment_from_bytes", side_effect=change_after_snapshot), \
                mock.patch.object(files.os, "close", wraps=os.close) as closing:
            _, result, _ = await self.execute(path)
        self.assertFalse(result["ok"], result)
        self.assertIn("changed while it was being read", result["content"])
        self.assertNotIn(path, self.session.file_observations)
        closing.assert_called_once()

    async def test_image_read_satisfies_overwrite_guard_but_not_after_disk_change(self):
        for changed in [False, True]:
            with self.subTest(changed=changed):
                path = self.make_file()
                _, read, _ = await self.execute(path)
                self.assertTrue(read["ok"], read)
                if changed:
                    with open(path, "ab") as stream:
                        stream.write(b"changed")
                result = await loki.dispatch_tool_async("Write", {"file_path": path, "content": "replacement"})
                self.assertEqual(result["ok"], not changed, result)
                if changed:
                    self.assertIn("changed on disk", result["content"])

    async def test_real_read_continuation_sends_image_bytes_across_provider_request_paths(self):
        cases = [[protocols.OPENAI_CHAT, None], [protocols.ANTHROPIC_MESSAGES, None],
                 [protocols.OPENAI_RESPONSES, None], [protocols.OPENAI_RESPONSES, False],
                 [protocols.OPENAI_RESPONSES, True]]
        for protocol, lite in cases:
            with self.subTest(protocol=protocol, subscription_lite=lite):
                path = self.make_file()
                text_path = self.make_file(b"plain text result", suffix=".txt")
                calls = [formats.tool_call_item("read-image", "Read", {"file_path": path}),
                         formats.tool_call_item("read-text", "Read", {"file_path": text_path})]
                profile = None if lite is None else openai_models.CodexModelRequestProfile(
                    use_responses_lite=lite, supports_parallel_tool_calls=True,
                    supports_reasoning_summaries=False, default_reasoning_level=None,
                    default_reasoning_summary=None, supports_verbosity=False, default_verbosity=None)
                config = connections.make_runtime_config(
                    "http://example.invalid/v1", protocol, model="vision-model",
                    provider_id=None if lite is None else "openai-subscription",
                    openai_request_profile=profile)
                loki.apply_runtime_config(config)
                captured = []
                emitted = []

                def reply(with_calls):
                    if protocol == protocols.OPENAI_CHAT:
                        message = {"role": "assistant", "content": None if with_calls else "Image inspected"}
                        if with_calls:
                            message["tool_calls"] = [{"id": call["call_id"], "type": "function", "function": {
                                "name": "Read", "arguments": call["arguments"]}} for call in calls]
                        return {"object": "chat.completion", "model": "vision-model", "choices": [{
                            "message": message, "finish_reason": "tool_calls" if with_calls else "stop"}]}
                    if protocol == protocols.ANTHROPIC_MESSAGES:
                        content = [{"type": "tool_use", "id": call["call_id"], "name": "Read",
                                    "input": formats.tool_call_input(call)} for call in calls] if with_calls else [
                                        {"type": "text", "text": "Image inspected"}]
                        return {"type": "message", "role": "assistant", "model": "vision-model", "content": content,
                                "stop_reason": "tool_use" if with_calls else "end_turn"}
                    output = copy.deepcopy(calls) if with_calls else [
                        {"type": "message", "role": "assistant", "content": [
                            {"type": "output_text", "text": "Image inspected"}]}]
                    return {"object": "response", "model": "vision-model", "status": "completed", "output": output}

                async def request(method, url, payload=None, *, context, **kwargs):
                    self.assertEqual(method, "POST")
                    self.assertIs(context.config, config)
                    self.assertEqual(url, config.chat_provider.chat_url)
                    captured.append(json.loads(json.dumps(payload)))
                    self.assertLessEqual(len(captured), 2)
                    return protocols.ProviderResponse(reply(len(captured) == 1))

                transcript = [formats.message_item("user", "Inspect the image and text")]
                with mock.patch.object(model_requests, "async_provider_request", request):
                    answer = await loki.run_tool_loop_async(
                        transcript, allowed={"Read"}, max_loops=3, on_event=emitted.append,
                        hook_pipeline=tool_runtime.ToolHookPipeline())
                self.assertEqual(answer, "Image inspected")
                self.assertEqual(len(captured), 2)
                self.assertNotIn(image_block()["source"]["data"], json.dumps(captured[0]))
                results = [event for event in transcript if event.get("type") == "tool_result"]
                self.assertEqual([result["call_id"] for result in results], ["read-image", "read-text"])
                self.assertFalse(results[0]["is_error"])
                self.assertEqual(results[0]["content"][-1], image_block())
                live_result = next(event for event in emitted if event.get("type") == "tool_result")
                self.assertEqual(live_result["content"][-1], image_block())
                payload = captured[1]
                if protocol == protocols.OPENAI_CHAT:
                    messages = payload["messages"]
                    self.assertEqual([message["role"] for message in messages[-3:]], ["tool", "tool", "user"])
                    self.assertEqual([message["tool_call_id"] for message in messages[-3:-1]], ["read-image", "read-text"])
                    self.assertIn("plain text result", messages[-2]["content"])
                    data_url = messages[-1]["content"][-1]["image_url"]["url"]
                elif protocol == protocols.ANTHROPIC_MESSAGES:
                    tool_results = payload["messages"][-1]["content"]
                    self.assertEqual([result["tool_use_id"] for result in tool_results], ["read-image", "read-text"])
                    self.assertIn("plain text result", tool_results[1]["content"])
                    self.assertEqual(tool_results[0]["content"][-1], image_block())
                    data_url = "data:image/png;base64," + tool_results[0]["content"][-1]["source"]["data"]
                else:
                    tool_results = [item for item in payload["input"] if item.get("type") == "function_call_output"]
                    self.assertEqual([result["call_id"] for result in tool_results], ["read-image", "read-text"])
                    self.assertIn("plain text result", tool_results[1]["output"])
                    data_url = tool_results[0]["output"][-1]["image_url"]
                self.assertTrue(data_url.startswith("data:image/png;base64,"))
                self.assertEqual(base64.b64decode(data_url.partition(",")[2]), PNG_BYTES)

    async def test_explore_subagent_continuation_can_see_image_read(self):
        path = self.make_file()
        requests = []

        async def completion(items, tools, *, codex_turn_state, **kwargs):
            self.assertNotIn("Write", {tool["function"]["name"] for tool in tools})
            requests.append(copy.deepcopy(items))
            self.assertLessEqual(len(requests), 2)
            if len(requests) == 1:
                return formats.DecodedTurn([formats.tool_call_item("child-read", "Read", {"file_path": path})])
            result = next(item for item in items if item.get("type") == "tool_result")
            self.assertFalse(result["is_error"])
            self.assertEqual(result["content"][-1], image_block())
            return formats.DecodedTurn([formats.message_item("assistant", "Image inspected by Explore")])

        with mock.patch.object(loki, "async_chat_completion", completion):
            answer = await subagents.run_prompt_async("Explore", "Inspect the local image")
        self.assertEqual(answer, "Image inspected by Explore")
        self.assertEqual(len(requests), 2)

    async def test_notes_presentations_save_and_replay_preserve_image_without_dumping_it(self):
        path = self.make_file()
        pipeline = tool_runtime.ToolHookPipeline()
        hook_outcomes = []

        def post(invocation, outcome):
            hook_outcomes.append(copy.deepcopy(outcome.to_hook_dict()))
            return tool_runtime.PostHookDecision(note="Image hook note")

        pipeline.add_post("image.note", post)
        call, result, _ = await self.execute(path, pipeline=pipeline)
        self.assertTrue(result["ok"], result)
        self.assertEqual(hook_outcomes[0]["content"][-1], image_block())
        self.assertEqual(json.loads(json.dumps(hook_outcomes[0])), hook_outcomes[0])
        self.assertEqual(result["content"][0], formats.text_block("Image hook note"))
        self.assertEqual(result["content"][-1], image_block())
        event = formats.tool_result_for_call(call, result["content"], process_output=result["process_output"])
        before = copy.deepcopy(event)
        encoded = image_block()["source"]["data"]
        views = []
        for show in [False, True]:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                terminal_frontend._terminal_agent_event(
                    {"type": "tool_result", "name": "Read", "call_id": "read-image", **result},
                    ui_settings=settings.Settings(terminal=settings.TerminalSettings(show_read_stdout=show)))
            views.append(output.getvalue())
            blocks = savefiles.ResumeTranscriptRenderer(show_read_stdout=show).presentation([event])
            views.append("".join(text for _, block in blocks for _, text in block))
        views.append(json.dumps(acp_events.map_event("session", event, {})))
        views.append(str(replays.classify_event(event)))
        for view in views:
            self.assertIn("Image hook note", view)
            self.assertIn("[Image content]", view)
            self.assertNotIn(encoded, view)
        self.assertEqual(event, before)
        events = [formats.message_item("user", "Inspect image"),
                  formats.model_response_event(formats.OPENAI_RESPONSES, [call]), event]
        loaded, *_ = savefiles.read_chat_log(io.StringIO(savefiles.serialize_chat_log(events, [], {})))
        self.assertEqual(loaded, events)
        os.unlink(path)
        # Saved results contain a snapshot, not an instruction to reopen path.
        for protocol in [protocols.OPENAI_CHAT, protocols.ANTHROPIC_MESSAGES, protocols.OPENAI_RESPONSES]:
            with self.subTest(protocol=protocol):
                provider = protocols.make_provider("http://example.invalid/v1", provider=protocol)
                payload = provider.chat_payload(loaded, None, "vision-model")
                self.assertIn(encoded, json.dumps(payload))


class ImageResultProjectionTests(unittest.TestCase):
    def test_base64_result_projects_to_all_protocols_without_mutating_history(self):
        call = formats.tool_call_item("image-call", "Read", {"file_path": "image"})
        blocks = [formats.text_block("Image result"), image_block()]
        events = [formats.model_response_event(formats.OPENAI_RESPONSES, [call]),
                  formats.tool_result_for_call(call, blocks)]
        before = copy.deepcopy(events)
        _, anthropic = formats.items_to_anthropic_parts(events)
        self.assertEqual(anthropic[1]["content"][0]["content"], blocks)
        _, responses = formats.items_to_openai_responses_parts(events)
        data_url = "data:image/png;base64," + image_block()["source"]["data"]
        self.assertEqual(responses[1]["output"], [
            {"type": "input_text", "text": "Image result"},
            {"type": "input_image", "image_url": data_url}])
        chat = formats.items_to_openai_chat_messages(events)
        self.assertEqual([item["role"] for item in chat], ["assistant", "tool", "user"])
        self.assertEqual(chat[1]["content"], "Image result")
        self.assertEqual(chat[2]["content"][-1], {"type": "image_url", "image_url": {"url": data_url}})
        self.assertEqual(events, before)

    def test_chat_images_follow_entire_parallel_reply_batch_with_call_association(self):
        calls = [formats.tool_call_item(name, "Read", {"file_path": name})
                 for name in ["first", "text-only", "second"]]
        events = [formats.model_response_event(formats.OPENAI_CHAT, calls),
                  formats.tool_result_for_call(calls[0], [image_block()]),
                  formats.tool_result_for_call(calls[1], "read failed", is_error=True),
                  formats.tool_result_for_call(calls[2], [formats.text_block("second image"), image_block()]),
                  formats.message_item("assistant", "Done")]
        before = copy.deepcopy(events)
        projected = formats.items_to_openai_chat_messages(events)
        self.assertEqual([item["role"] for item in projected],
                         ["assistant", "tool", "tool", "tool", "user", "assistant"])
        self.assertEqual([item["tool_call_id"] for item in projected[1:4]], ["first", "text-only", "second"])
        self.assertEqual(projected[2]["content"], "read failed")
        content = projected[4]["content"]
        self.assertEqual([block["type"] for block in content], ["text", "image_url", "text", "image_url"])
        self.assertIn("first", content[0]["text"])
        self.assertIn("second", content[2]["text"])
        self.assertEqual(projected, formats.items_to_openai_chat_messages(events))
        self.assertEqual(events, before)

    def test_legacy_function_reply_keeps_its_role_and_supplies_image(self):
        turn = formats.openai_chat_response_to_items({"choices": [{"message": {
            "role": "assistant", "content": None,
            "function_call": {"name": "Read", "arguments": '{"file_path":"image"}'}}}]})
        call = formats.response_tool_calls(turn)[0]
        events = [turn.to_event(), formats.tool_result_for_call(call, [image_block()])]
        projected = formats.items_to_openai_chat_messages(events)
        self.assertEqual([item["role"] for item in projected], ["assistant", "function", "user"])
        self.assertEqual(projected[1]["name"], "Read")
        self.assertEqual(projected[2]["content"][-1]["type"], "image_url")

    def test_url_images_and_invalid_sources_are_not_silently_dropped(self):
        call = formats.tool_call_item("url-image", "Read", {})
        for source in [{"type": "url", "url": "https://example.invalid/image.png"}, {}]:
            events = [formats.model_response_event(formats.OPENAI_CHAT, [call]),
                      formats.tool_result_for_call(call, [{"type": "image", "source": source}])]
            if source:
                projected = formats.items_to_openai_chat_messages(events)
                self.assertEqual(projected[-1]["content"][-1]["image_url"]["url"], source["url"])
            else:
                with self.assertRaises(formats.ProjectionError):
                    formats.items_to_openai_chat_messages(events)

    def test_acp_image_result_notice_fits_even_when_image_exceeds_frame_budget(self):
        image = image_block()
        image["source"]["data"] = "A" * 1024
        event = formats.tool_result_item("image", [image], name="Read")
        with mock.patch.object(acps, "MAX_MESSAGE_BYTES", 512):
            update = acp_events.map_event("session", event, {})[0]
            frame = acps.encode_message(acps.notification("session/update", update))
        self.assertLess(len(frame), 512)
        self.assertIn(b"[Image content]", frame)
        self.assertEqual(event["content"][0]["source"]["data"], "A" * 1024)

    def test_image_only_result_has_a_placeholder_in_live_and_legacy_replay(self):
        blocks = [image_block()]
        self.assertEqual(process_outputs.presentation_text(blocks), "[Image content]")
        self.assertEqual(process_outputs.presentation_text(blocks, legacy_read=True), "[Image content]")
        item = formats.tool_result_item("image", blocks, name="Read")
        self.assertIn("[Image content]", replays.classify_tool_result(item)[0][1])
        for show in [None, False, True]:
            rendered = savefiles.ResumeTranscriptRenderer(show_read_stdout=show).presentation([item])
            self.assertIn("[Image content]", "".join(text for _, block in rendered for _, text in block))
