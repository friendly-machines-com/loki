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
            api_url="https://openrouter.ai/api/v1",
            chat_url="https://openrouter.ai/api/v1/chat/completions",
            models_url="https://openrouter.ai/api/v1/models",
            protocol="openai_chat",
            credential_env="OPENROUTER_API_KEY",
        )

        encoded = descriptor.to_dict()

        self.assertEqual(ConnectionDescriptor.from_dict(encoded), descriptor)
        self.assertNotIn("secret", repr(encoded))

    def test_rejects_invalid_persisted_shapes(self):
        with self.assertRaises(ConnectionDescriptorError):
            ConnectionDescriptor.from_dict({"model": "x"})
        with self.assertRaises(ConnectionDescriptorError):
            ConnectionDescriptor.from_dict({
                "model": "x",
                "api_url": "https://example.test/v1",
                "chat_url": "https://example.test/v1/chat/completions",
                "protocol": "openai_chat",
                "credential_env": "EXAMPLE_API_KEY",
                "max_tokens": 0,
            })


if __name__ == "__main__":
    unittest.main()
