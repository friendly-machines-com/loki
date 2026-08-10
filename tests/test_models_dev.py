import asyncio
import os
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from loki_agent import models

# Minimal synthetic models.dev dataset (provider-keyed, like the real API).
DATA = {
    "zhipuai": {
        "id": "zhipuai", "name": "Zhipu AI", "npm": "@ai-sdk/openai-compatible",
        "env": ["ZHIPU_API_KEY"], "api": "https://open.bigmodel.cn/api/paas/v4",
        "models": {
            "glm-5.2": {
                "id": "glm-5.2", "name": "GLM-5.2", "family": "glm",
                "reasoning": True, "tool_call": True, "temperature": True,
                "attachment": False, "structured_output": True, "open_weights": False,
                "cost": {"input": 1.4, "output": 4.4},
            },
        },
    },
    "openrouter": {
        "id": "openrouter", "name": "OpenRouter", "npm": "@openrouter/ai-sdk-provider",
        "env": ["OPENROUTER_API_KEY"], "api": "https://openrouter.ai/api/v1",
        "models": {
            "z-ai/glm-5.2": {
                "id": "z-ai/glm-5.2", "name": "GLM-5.2", "family": "glm",
                "reasoning": True, "tool_call": True, "temperature": True,
                "attachment": True, "structured_output": True, "open_weights": False,
                "cost": {"input": 2, "output": 8},
            },
        },
    },
    "anthropic": {
        "id": "anthropic", "name": "Anthropic", "npm": "@ai-sdk/anthropic",
        "env": ["ANTHROPIC_API_KEY"], "api": "https://api.anthropic.com/v1/messages",
        "models": {
            "claude-sonnet-4-6": {
                "id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "family": "claude-sonnet",
                "reasoning": True, "tool_call": True, "temperature": True,
                "attachment": True, "structured_output": True, "open_weights": False,
                "cost": {"input": 3, "output": 15},
            },
        },
    },
}


def _groups():
    return models.build_groups(DATA)


def _input_script(inputs):
    """Async input_fn returning scripted strings; EOFError when exhausted."""
    iterator = iter(inputs)

    async def fake(prompt=None, history=None):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    return fake


class GroupingTests(unittest.TestCase):
    def test_build_groups_conflates_provider_specific_ids_by_name(self):
        groups = _groups()
        self.assertEqual(len(groups), 2)
        members = groups["GLM-5.2"]
        self.assertEqual(len(members), 2)
        # Provider-specific ids are kept per member, grouped under one name.
        ids = {m["id"] for _, _, m in members}
        self.assertEqual(ids, {"glm-5.2", "z-ai/glm-5.2"})

    def test_minimal_features_intersect_across_providers(self):
        members = _groups()["GLM-5.2"]
        bits = models.minimal_feature_bits(members)
        # reasoning/tool_call/struct/temp on both; attachment on only openrouter.
        self.assertEqual(bits, (True, True, True, True, False, False))
        self.assertEqual(models.feature_names(bits), "reasoning, tools, struct, temp")

    def test_union_features_or_across_providers(self):
        members = _groups()["GLM-5.2"]
        bits = models.union_feature_bits(members)
        # attachment is on openrouter but not zhipuai, so the union includes it.
        self.assertEqual(bits, (True, True, True, True, True, False))

    def test_feature_names_empty_bits(self):
        self.assertEqual(models.feature_names((False, False, False, False, False, False)), "")


class ProtocolAndKeyTests(unittest.TestCase):
    def test_provider_supported_accepts_endpoint_and_v1_base(self):
        self.assertTrue(models.provider_supported({"api": "https://x.test/v1/chat/completions"}))
        self.assertTrue(models.provider_supported({"api": "https://x.test/v1/messages"}))
        self.assertTrue(models.provider_supported({"api": "https://x.test/v1/responses"}))
        self.assertTrue(models.provider_supported({"api": "https://x.test/v1"}))
        self.assertTrue(models.provider_supported({"api": "https://x.test/compatible-mode/v1"}))

    def test_provider_supported_rejects_non_v1_and_missing_api(self):
        self.assertFalse(models.provider_supported({"api": "https://x.test/api/paas/v4"}))
        self.assertFalse(models.provider_supported({"api": "https://x.test"}))
        self.assertFalse(models.provider_supported({}))
        self.assertFalse(models.provider_supported({"api": None}))

    def test_filter_supported_groups_drops_unsupported_providers_and_models(self):
        # Synthetic DATA: zhipuai api is /api/paas/v4 (unsupported),
        # openrouter /api/v1 and anthropic /v1/messages are supported.
        filtered = models.filter_supported_groups(_groups())
        self.assertEqual(set(filtered), {"GLM-5.2", "Claude Sonnet 4.6"})
        # GLM-5.2 keeps only the supported provider (openrouter).
        self.assertEqual([pid for pid, _, _ in filtered["GLM-5.2"]], ["openrouter"])
        self.assertEqual([pid for pid, _, _ in filtered["Claude Sonnet 4.6"]], ["anthropic"])

    def test_filter_supported_groups_drops_fully_deprecated_models(self):
        data = {
            "p1": {"api": "https://x.test/v1", "models": {
                "m1": {"id": "m1", "name": "Old Model", "status": "deprecated"},
                "m2": {"id": "m2", "name": "Alive Model"},
            }},
            "p2": {"api": "https://x.test/v1", "models": {
                "m1": {"id": "m1", "name": "Old Model", "status": "deprecated"},
                "m3": {"id": "m3", "name": "Mixed Model", "status": "deprecated"},
            }},
            "p3": {"api": "https://x.test/v1", "models": {
                "m3": {"id": "m3", "name": "Mixed Model"},
            }},
        }
        groups = models.filter_supported_groups(models.build_groups(data))
        # Old Model: deprecated on every provider -> dropped entirely.
        # Alive Model: not deprecated -> kept. Mixed Model: deprecated only on
        # p2, so it stays but only with the live provider p3.
        self.assertEqual(set(groups), {"Alive Model", "Mixed Model"})
        self.assertEqual([pid for pid, _, _ in groups["Mixed Model"]], ["p3"])

    def test_protocol_label_detects_from_api_url(self):
        openai_chat = {"api": "https://x.test/v1/chat/completions", "npm": "@ai-sdk/openai"}
        anthropic = {"api": "https://x.test/v1/messages", "npm": "@ai-sdk/anthropic"}
        responses = {"api": "https://x.test/v1/responses", "npm": "@ai-sdk/openai"}
        self.assertEqual(models.protocol_label(openai_chat), "openai-chat")
        self.assertEqual(models.protocol_label(anthropic), "anthropic-messages")
        self.assertEqual(models.protocol_label(responses), "openai-responses")

    def test_protocol_label_falls_back_to_npm_then_no_api(self):
        bespoke = {"api": "https://x.test/api/paas/v4", "npm": "@ai-sdk/perplexity"}
        noapi = {"npm": "@ai-sdk/mistral"}
        neither = {}
        self.assertEqual(models.protocol_label(bespoke), "@ai-sdk/perplexity")
        self.assertEqual(models.protocol_label(noapi), "@ai-sdk/mistral")
        self.assertEqual(models.protocol_label(neither), "no-api")

    def test_api_key_for_prefers_provider_env_var_then_fallback(self):
        entry = DATA["zhipuai"]
        with mock.patch.dict(os.environ, {"ZHIPU_API_KEY": "zk", "OTHER": "no"}):
            self.assertEqual(models.api_key_for(entry, "fallback"), "zk")
        self.assertEqual(models.api_key_for(entry, "fallback"), "fallback")


class MenuTests(unittest.TestCase):
    def test_menu_selects_by_number(self):
        rows = [("a", "Alpha"), ("b", "Beta")]
        result = asyncio.run(models._numbered_menu_async(rows, "Choice: ", _input_script(["2"])))
        self.assertEqual(result, "b")

    def test_menu_filter_narrows_then_selects(self):
        rows = [("a", "Alpha beta"), ("b", "Beta gamma")]
        result = asyncio.run(models._numbered_menu_async(
            rows, "Choice: ", _input_script(["filter alpha", "1"])))
        self.assertEqual(result, "a")

    def test_menu_empty_cancels(self):
        rows = [("a", "Alpha")]
        result = asyncio.run(models._numbered_menu_async(rows, "Choice: ", _input_script([""])))
        self.assertIsNone(result)


class PickerTests(unittest.TestCase):
    def test_run_model_picker_two_level_flow(self):
        saved = models._index_cache
        models._index_cache = (DATA, models.build_groups(DATA))
        try:
            # Model menu (sorted): 1. Claude Sonnet 4.6, 2. GLM-5.2.
            # Provider menu (sorted): 1. OpenRouter, 2. Zhipu AI.
            result = asyncio.run(models.run_model_picker_async(
                _input_script(["2", "1"])))
        finally:
            models._index_cache = saved

        provider_id, provider_entry, model_entry = result
        self.assertEqual(provider_id, "openrouter")
        self.assertEqual(model_entry["id"], "z-ai/glm-5.2")

    def test_run_model_picker_cancel_returns_none(self):
        saved = models._index_cache
        models._index_cache = (DATA, models.build_groups(DATA))
        try:
            result = asyncio.run(models.run_model_picker_async(_input_script([""])))
        finally:
            models._index_cache = saved
        self.assertIsNone(result)

    def test_run_model_picker_provider_cancel_returns_none(self):
        # Pick a model, then cancel the provider menu -> None, not a fallback.
        saved = models._index_cache
        models._index_cache = (DATA, models.build_groups(DATA))
        try:
            result = asyncio.run(models.run_model_picker_async(_input_script(["1", ""])))
        finally:
            models._index_cache = saved
        self.assertIsNone(result)

    def test_run_flat_model_picker_selects_by_number(self):
        result = asyncio.run(models.run_flat_model_picker_async(
            _input_script(["2"]), ["alpha", "beta"]))
        self.assertEqual(result, "beta")

    def test_run_flat_model_picker_empty_list_returns_none(self):
        result = asyncio.run(models.run_flat_model_picker_async(
            _input_script([]), []))
        self.assertIsNone(result)

    def test_run_flat_model_picker_cancel_returns_none(self):
        result = asyncio.run(models.run_flat_model_picker_async(
            _input_script([""]), ["alpha"]))
        self.assertIsNone(result)

    def test_run_model_picker_fetch_failure_propagates(self):
        # When models.dev is unreachable, the fetch exception propagates out
        # of the picker (nothing is swallowed); /model catches it and falls
        # back to the provider's own model list.
        with mock.patch("loki_agent.models.ensure_index",
                        side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                asyncio.run(models.run_model_picker_async(_input_script([])))

    def test_picker_prompts_advertise_filter_gesture(self):
        saved = models._index_cache
        models._index_cache = (DATA, models.build_groups(DATA))
        prompts = []

        async def capture(prompt=None, history=None):
            prompts.append(prompt or "")
            raise EOFError

        try:
            asyncio.run(models.run_model_picker_async(capture))
        except EOFError:
            pass
        finally:
            models._index_cache = saved

        self.assertTrue(prompts)
        self.assertIn("Model choice", prompts[0])
        self.assertIn("filter WORDS", prompts[0])
        self.assertIn("empty cancels", prompts[0])

    def test_model_rows_show_minimal_features_in_parentheses(self):
        rows = models._model_rows(_groups())
        labels = [text for _, text in rows]
        glm = next(t for t in labels if t.startswith("GLM-5.2"))
        # Intersection shown; union (adds attachment) differs -> "[and more]".
        self.assertIn("(reasoning, tools, struct, temp)", glm)
        self.assertIn("[and more]", glm)
        self.assertIn("[2 providers]", glm)
        # Cost: min/max input and output across the two providers.
        self.assertIn("cost: in $1.4-$2 per 1M tokens, out $4.4-$8 per 1M tokens", glm)
        # Single-provider group: intersection == union, no "[and more]".
        claude = next(t for t in labels if t.startswith("Claude Sonnet 4.6"))
        self.assertNotIn("[and more]", claude)

    def test_provider_rows_show_per_provider_cost(self):
        members = _groups()["GLM-5.2"]
        rows = models._provider_rows(members)
        text = {pid: t for (pid, _, _), t in rows}
        self.assertIn("cost: in $1.4 per 1M tokens, out $4.4 per 1M tokens", text["zhipuai"])
        self.assertIn("cost: in $2 per 1M tokens, out $8 per 1M tokens", text["openrouter"])

    def test_cost_helpers(self):
        with_cost = {"cost": {"input": 1.5, "output": 3.25}}
        no_cost = {"name": "free"}
        self.assertEqual(models.cost_pair(with_cost), (1.5, 3.25))
        self.assertIsNone(models.cost_pair(no_cost))
        self.assertEqual(models.cost_text(with_cost),
                         " cost: in $1.5 per 1M tokens, out $3.25 per 1M tokens")
        self.assertEqual(models.cost_text(no_cost), "")
        self.assertEqual(models.cost_range_text([("p", {}, no_cost)]), "")


if __name__ == "__main__":
    unittest.main()
