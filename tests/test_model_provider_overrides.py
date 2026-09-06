import copy
import unittest

from loki_agent import authentications, loki, models, protocols
from loki_agent.credentials import CredentialStore


class ModelProviderOverrideTests(unittest.TestCase):
    def setUp(self):
        self.provider = {
            "id": "opencode-go",
            "name": "OpenCode Go",
            "env": ["OPENCODE_API_KEY"],
            "api": "https://opencode.ai/zen/go/v1",
            "npm": "@ai-sdk/openai-compatible",
            "models": {
                "grok-4.6": {
                    "id": "grok-4.6", "name": "Grok 4.6",
                    "provider": {"npm": "@ai-sdk/openai"},
                },
                "chat": {"id": "chat", "name": "Chat"},
                "claude": {
                    "id": "claude", "name": "Claude",
                    "provider": {"npm": "@ai-sdk/anthropic"},
                },
            },
        }
        self.credentials = CredentialStore({"OPENCODE_API_KEY": "test-key"})

    def test_group_filter_display_and_connection_agree(self):
        original = copy.deepcopy(self.provider)
        groups = models.filter_supported_groups(
            models.build_groups({"opencode-go": self.provider}),
            self.credentials)
        for name, protocol, suffix in (
                ("Grok 4.6", protocols.OPENAI_RESPONSES, "/responses"),
                ("Chat", protocols.OPENAI_CHAT, "/chat/completions"),
                ("Claude", protocols.ANTHROPIC_MESSAGES, "/messages")):
            with self.subTest(name=name):
                leaf = groups[name][0]
                self.assertEqual(models.provider_protocol(leaf[1]), protocol)
                rows = models._provider_rows(groups[name])
                self.assertIn(f"[{protocol.replace('_', '-')}]", rows[0][1])
                config = loki.config_from_modelsdev_selection(
                    *leaf, self.credentials)
                self.assertEqual(config.chat_provider.kind, protocol)
                self.assertEqual(
                    config.chat_provider.chat_url,
                    self.provider["api"] + suffix)
                self.assertEqual(
                    config.auth_spec.credential,
                    authentications.CredentialRef.environment(
                        "OPENCODE_API_KEY"))
        self.assertEqual(self.provider, original)

    def test_direct_selection_and_saved_connection_use_override(self):
        config = loki.config_from_modelsdev_selection(
            "opencode-go", self.provider,
            self.provider["models"]["grok-4.6"], self.credentials)
        restored = loki.config_from_connection_descriptor(
            loki.connection_descriptor_from_config(config), self.credentials)
        self.assertEqual(restored.chat_provider.kind, protocols.OPENAI_RESPONSES)
        self.assertEqual(
            restored.chat_provider.chat_url,
            "https://opencode.ai/zen/go/v1/responses")

    def test_api_and_explicit_shape_override(self):
        for shape, protocol, suffix in (
                ("responses", protocols.OPENAI_RESPONSES, "/responses"),
                ("completions", protocols.OPENAI_CHAT, "/chat/completions")):
            with self.subTest(shape=shape):
                model = {
                    "id": "test",
                    "provider": {
                        "api": "https://example.test/v1",
                        "npm": "@ai-sdk/openai",
                        "shape": shape,
                    },
                }
                config = loki.config_from_modelsdev_selection(
                    "opencode-go", self.provider, model, self.credentials)
                self.assertEqual(config.chat_provider.kind, protocol)
                self.assertEqual(
                    config.chat_provider.chat_url,
                    "https://example.test/v1" + suffix)

    def test_model_can_supply_missing_provider_transport(self):
        provider = dict(self.provider)
        provider.pop("api")
        provider.pop("npm")
        provider["models"] = {
            "test": {"id": "test", "provider": {
                "api": "https://example.test/v1",
                "npm": "@ai-sdk/openai",
            }},
        }
        groups = models.filter_supported_groups(
            models.build_groups({"opencode-go": provider}), self.credentials)
        self.assertIn("test", groups)

    def test_invalid_overrides_are_filtered_and_cannot_connect(self):
        for override in ([], {"npm": None}, {"api": ""},
                         {"shape": "unknown"}):
            with self.subTest(override=override):
                provider = dict(self.provider)
                model = {"id": "bad", "provider": override}
                provider["models"] = {"bad": model}
                groups = models.build_groups({"opencode-go": provider})
                self.assertEqual(
                    models.filter_supported_groups(groups, self.credentials),
                    {})
                with self.assertRaises(ValueError):
                    loki.config_from_modelsdev_selection(
                        "opencode-go", provider, model, self.credentials)

    def test_overrides_do_not_supply_credentials_or_request_headers(self):
        model = {"id": "test", "provider": {
            "npm": "@ai-sdk/openai",
            "env": ["OTHER_API_KEY"],
            "headers": {"Authorization": "untrusted"},
            "body": {"model": "other"},
        }}
        config = loki.config_from_modelsdev_selection(
            "opencode-go", self.provider, model, self.credentials)
        self.assertEqual(config.model, "test")
        self.assertNotIn("Authorization", config.chat_provider.headers)
        self.assertEqual(
            config.auth_spec.credential,
            authentications.CredentialRef.environment("OPENCODE_API_KEY"))

    def test_canonical_openai_endpoint_protection_survives_override(self):
        provider = models.normalize_catalog({"openai": {
            "id": "openai", "npm": "@ai-sdk/openai",
            "env": ["OPENAI_API_KEY"],
        }})["openai"]
        model = {"id": "test", "provider": {
            "api": "https://unrelated.example/v1",
        }}
        with self.assertRaises(ValueError):
            loki.config_from_modelsdev_selection(
                "openai", provider, model,
                CredentialStore({"OPENAI_API_KEY": "test-key"}))
