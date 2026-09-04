import unittest

from loki_agent import reasonings


class ReasoningEffortProfileTests(unittest.TestCase):
    def test_modelsdev_effort_is_exact_and_ignores_other_controls(self):
        profile = reasonings.from_modelsdev_model({
            "reasoning_options": [
                {"type": "toggle"},
                {
                    "type": "effort",
                    "values": ["low", "high", "max"],
                },
                {"type": "budget_tokens", "min": 1024},
            ],
        })

        self.assertEqual(profile.values, ("low", "high", "max"))
        self.assertIsNone(profile.default_value)

    def test_modelsdev_without_effort_has_no_profile(self):
        self.assertIsNone(reasonings.from_modelsdev_model({
            "reasoning_options": [
                {"type": "toggle"},
                {"type": "budget_tokens", "min": 1024},
            ],
        }))

    def test_malformed_and_duplicate_values_are_rejected(self):
        for values in (
                [],
                ["low", "low"],
                ["low", None],
                ["low", "future-value"]):
            with self.subTest(values=values), self.assertRaises(
                    reasonings.ReasoningEffortError):
                reasonings.from_modelsdev_model({
                    "reasoning_options": [{
                        "type": "effort",
                        "values": values,
                    }],
                })

    def test_malformed_non_effort_option_is_rejected(self):
        with self.assertRaises(reasonings.ReasoningEffortError):
            reasonings.from_modelsdev_model({
                "reasoning_options": [
                    {"type": "effort", "values": ["low"]},
                    None,
                ],
            })

    def test_codex_profile_normalizes_ultra_and_preserves_descriptions(self):
        profile = reasonings.from_codex_catalog_model({
            "supports_reasoning_summaries": True,
            "supported_reasoning_levels": [
                {"effort": "high", "description": "Deep reasoning"},
                {"effort": "ultra", "description": "Maximum reasoning"},
            ],
            "default_reasoning_level": "ultra",
        })

        self.assertEqual(profile.values, ("high", "max"))
        self.assertEqual(profile.default_value, "max")
        self.assertEqual(
            profile.option("high").description, "Deep reasoning")

    def test_codex_request_gate_disables_effort_profile(self):
        self.assertIsNone(reasonings.from_codex_catalog_model({
            "supports_reasoning_summaries": False,
            "supported_reasoning_levels": [{"effort": "high"}],
            "default_reasoning_level": "high",
        }))

    def test_profile_round_trip_is_strict(self):
        profile = reasonings.ReasoningEffortProfile(
            options=(
                reasonings.ReasoningEffortOption("low"),
                reasonings.ReasoningEffortOption(
                    "high", "More reasoning"),
            ),
            default_value="low",
        )

        self.assertEqual(
            reasonings.ReasoningEffortProfile.from_dict(profile.to_dict()),
            profile,
        )
        encoded = profile.to_dict()
        encoded["unexpected"] = True
        with self.assertRaises(reasonings.ReasoningEffortError):
            reasonings.ReasoningEffortProfile.from_dict(encoded)

    def test_unsupported_preference_is_dormant_not_coerced(self):
        profile = reasonings.ReasoningEffortProfile(
            options=(
                reasonings.ReasoningEffortOption("low"),
                reasonings.ReasoningEffortOption("medium"),
            ),
            default_value="medium",
        )

        self.assertIsNone(reasonings.effective_effort(profile, "high"))
        self.assertIn(
            "preferred High is unavailable",
            reasonings.default_option_name(profile, "high"),
        )
        self.assertEqual(
            reasonings.effective_effort(profile, "low"), "low")


if __name__ == "__main__":
    unittest.main()
