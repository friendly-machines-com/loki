import asyncio
import os
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("LOKI_API_KEY", "test-key")
os.environ.setdefault("LOKI_API_BASE", "https://api.openai.com/v1/responses")
os.environ.setdefault("LOKI_PROVIDER", "openai_responses")

from day_agent import formats
from day_agent import loki


class ResponsesToolLoopTests(unittest.TestCase):
    def test_load_models_initializes_lazy_provider_before_using_it(self):
        old_chat_provider = loki.chat_provider
        old_headers = loki.headers
        old_api_key = loki.api_key
        old_models = loki.models
        old_model = loki.model
        old_async_chat_request = loki.async_chat_request
        os.environ["LOKI_API_KEY"] = "test-key"

        async def fake_async_chat_request(request_url, payload, request_headers=None,
                                          report_errors=False, show_timing=False):
            self.assertIsNotNone(loki.chat_provider)
            self.assertIn("/models", request_url)
            self.assertIsNone(payload)
            self.assertEqual(request_headers["Authorization"], "Bearer test-key")
            return {"data": [{"id": "model-a"}]}

        try:
            loki.chat_provider = None
            loki.headers = {}
            loki.api_key = ""
            loki.models = []
            loki.model = ""
            loki.async_chat_request = fake_async_chat_request

            asyncio.run(loki.load_models_async())

            self.assertIsNotNone(loki.chat_provider)
            self.assertEqual(loki.models, ["model-a"])
            self.assertEqual(loki.model, "model-a")
        finally:
            loki.chat_provider = old_chat_provider
            loki.headers = old_headers
            loki.api_key = old_api_key
            loki.models = old_models
            loki.model = old_model
            loki.async_chat_request = old_async_chat_request

    def test_function_call_only_response_executes_tool_and_continues(self):
        transcript = [formats.message_item("user", "read README")]
        seen_inputs = []
        events = []

        async def chat_fn(items):
            seen_inputs.append([item.get("type") for item in items])
            if len(seen_inputs) == 1:
                return [
                    formats.response_metadata_item(
                        "openai",
                        "openai_responses",
                        {"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-test"},
                    ),
                    formats.tool_call_item("call_1", "Read", {"file_path": "README.md"}),
                ]
            return [formats.message_item("assistant", "done")]

        async def fake_dispatch(fn_name, args, allowed=None):
            self.assertEqual(fn_name, "Read")
            self.assertEqual(args, {"file_path": "README.md"})
            return {"ok": True, "content": "file contents"}

        old_dispatch = loki.dispatch_tool_async
        try:
            loki.dispatch_tool_async = fake_dispatch
            result = asyncio.run(loki.run_tool_loop_async(
                transcript,
                chat_fn=chat_fn,
                on_event=events.append,
                max_loops=3,
            ))
        finally:
            loki.dispatch_tool_async = old_dispatch

        self.assertEqual(result, "done")
        self.assertEqual(
            [item.get("type") for item in transcript],
            ["message", "response_metadata", "tool_call", "tool_result", "message"],
        )
        self.assertEqual(transcript[3]["tool_call_id"], "call_1")
        self.assertEqual(formats.item_text(transcript[3]), "file contents")
        self.assertEqual(seen_inputs[1], ["message", "response_metadata", "tool_call", "tool_result"])
        self.assertEqual([event.get("type") for event in events], ["tool_call", "assistant_message"])


if __name__ == "__main__":
    unittest.main()
