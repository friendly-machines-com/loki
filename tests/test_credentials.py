import unittest

from loki_agent.connections import (
    ConnectionDescriptor,
    ConnectionDescriptorError,
)
from loki_agent.credentials import CredentialStore, is_credential_name


class CredentialStoreTests(unittest.TestCase):
    def test_capture_retains_snapshot_and_scrubs_narrow_suffixes(self):
        env = {
            "OPENAI_API_KEY": "key",
            "GITHUB_TOKEN": "token",
            "CLARIFAI_PAT": "pat",
            "GOOGLE_APPLICATION_CREDENTIALS": "/secret/file",
            "CLOUDFLARE_ACCOUNT_ID": "account",
            "EMPTY_KEY": "",
        }

        store = CredentialStore.capture(env)

        self.assertEqual(
            env,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": "/secret/file",
                "CLOUDFLARE_ACCOUNT_ID": "account",
            },
        )
        self.assertEqual(store.get("OPENAI_API_KEY"), "key")
        self.assertEqual(store.get("CLOUDFLARE_ACCOUNT_ID"), "account")
        self.assertFalse(store.has("EMPTY_KEY"))
        self.assertNotIn("key", repr(store))
        self.assertNotIn("token", repr(store))
        self.assertNotIn("pat", repr(store))

    def test_name_policy_is_deliberately_narrow(self):
        for name in [
            "OPENAI_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "GITHUB_TOKEN",
            "CLARIFAI_PAT",
        ]:
            self.assertTrue(is_credential_name(name))
        for name in [
            "WATSONX_AI_APIKEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "CLOUDFLARE_ACCOUNT_ID",
        ]:
            self.assertFalse(is_credential_name(name))

    def test_first_available_preserves_declaration_order(self):
        store = CredentialStore({"FIRST_KEY": "one", "SECOND_TOKEN": "two"})
        self.assertEqual(
            store.first_available(["SECOND_TOKEN", "FIRST_KEY"]),
            ("SECOND_TOKEN", "two"),
        )


class ConnectionDescriptorTests(unittest.TestCase):
    def test_round_trip_contains_names_but_no_values(self):
        descriptor = ConnectionDescriptor(
            provider_id="openrouter",
            provider_name="OpenRouter",
            model="z-ai/glm",
            chat_url="https://openrouter.ai/api/v1/chat/completions",
            models_url="https://openrouter.ai/api/v1/models",
            protocol="openai_chat",
            credential_env="OPENROUTER_API_KEY",
            model_status="deprecated",
            prompt_cache=True,
        )

        encoded = descriptor.to_dict()

        self.assertEqual(ConnectionDescriptor.from_dict(encoded), descriptor)
        self.assertEqual(encoded["model_status"], "deprecated")
        self.assertIs(encoded["prompt_cache"], True)
        self.assertNotIn("api_url", encoded)
        self.assertNotIn("secret", repr(encoded))

    def test_old_api_url_field_is_accepted_and_not_reserialized(self):
        legacy = {
            "provider_id": "openrouter",
            "provider_name": "OpenRouter",
            "model": "z-ai/glm",
            "api_url": "https://openrouter.ai/api/v1",
            "chat_url":
                "https://openrouter.ai/api/v1/chat/completions",
            "models_url": "https://openrouter.ai/api/v1/models",
            "protocol": "openai_chat",
            "credential_env": "OPENROUTER_API_KEY",
        }

        descriptor = ConnectionDescriptor.from_dict(legacy)

        self.assertEqual(
            descriptor.chat_url,
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertNotIn("api_url", descriptor.to_dict())
        self.assertIsNone(descriptor.model_status)

    def test_credentialless_connection_round_trips_explicit_null(self):
        descriptor = ConnectionDescriptor(
            provider_id=None,
            provider_name="Explicit LOKI_* connection",
            model="local-model",
            chat_url="http://localhost:8000/v1/chat/completions",
            models_url="http://localhost:8000/v1/models",
            protocol="openai_chat",
            credential_env=None,
            stream=True,
        )

        encoded = descriptor.to_dict()

        self.assertIn("credential_env", encoded)
        self.assertIsNone(encoded["credential_env"])
        self.assertIs(encoded["stream"], True)
        self.assertEqual(ConnectionDescriptor.from_dict(encoded), descriptor)

    def test_rejects_invalid_persisted_shapes(self):
        with self.assertRaises(ConnectionDescriptorError):
            ConnectionDescriptor.from_dict({"model": "x"})
        with self.assertRaises(ConnectionDescriptorError):
            ConnectionDescriptor.from_dict({
                "model": "x",
                "chat_url": "https://example.test/v1/chat/completions",
                "protocol": "openai_chat",
                "credential_env": "EXAMPLE_API_KEY",
                "max_tokens": 0,
            })
        with self.assertRaises(ConnectionDescriptorError):
            ConnectionDescriptor.from_dict({
                "model": "x",
                "chat_url": "https://example.test/v1/chat/completions",
                "protocol": "openai_chat",
                "credential_env": "EXAMPLE_API_KEY",
                "model_status": False,
            })
        with self.assertRaises(ConnectionDescriptorError):
            ConnectionDescriptor.from_dict({
                "model": "x",
                "chat_url": "https://example.test/v1/chat/completions",
                "protocol": "openai_chat",
            })
        with self.assertRaises(ConnectionDescriptorError):
            ConnectionDescriptor.from_dict({
                "model": "x",
                "chat_url": "https://example.test/v1/chat/completions",
                "protocol": "openai_chat",
                "credential_env": None,
                "stream": "yes",
            })
        with self.assertRaises(ConnectionDescriptorError):
            ConnectionDescriptor.from_dict({
                "model": "x",
                "chat_url": "https://example.test/v1/messages",
                "protocol": "anthropic_messages",
                "credential_env": None,
                "prompt_cache": "yes",
            })


if __name__ == "__main__":
    unittest.main()
