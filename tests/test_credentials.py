import json
import os
import subprocess
import sys
import unittest

from loki_agent import models, openai_models
from loki_agent.authentications import (
    CredentialRef,
    OPENAI_CHATGPT_MODELS_REQUEST_URL,
)
from loki_agent.connections import (
    ConnectionDescriptor,
    ConnectionDescriptorError,
)
from loki_agent.credentials import (
    CredentialScrubError,
    CredentialStore,
    _validate_entries_in_range,
    is_credential_name,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _codex_model(slug="gpt-5-codex", **overrides):
    value = {
        "slug": slug,
        "display_name": slug,
        "visibility": "list",
        "supported_reasoning_levels": [],
        "supports_reasoning_summaries": False,
        "default_reasoning_summary": "auto",
        "support_verbosity": False,
        "supports_parallel_tool_calls": True,
        "shell_type": "shell_command",
    }
    value.update(overrides)
    return openai_models.CodexModelRequestProfile.from_catalog_model(value)


class CredentialStoreTests(unittest.TestCase):
    def test_startup_scrubber_does_not_import_authentication_runtime(self):
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "from loki_agent import credentials\n"
                    "print('loki_agent.authentications' in sys.modules)\n"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "False")

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
            store.first_available_name(["SECOND_TOKEN", "FIRST_KEY"]),
            "SECOND_TOKEN",
        )

    def test_native_entry_range_validation_includes_terminating_nul(self):
        matches = {"EXAMPLE_TOKEN": [(100, 20, 14)]}

        _validate_entries_in_range(matches, 100, 121)

        for low, high in [(101, 121), (100, 120)]:
            with self.subTest(low=low, high=high):
                with self.assertRaises(CredentialScrubError):
                    _validate_entries_in_range(matches, low, high)

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "Linux /proc environment semantics")
    def test_process_capture_scrubs_record_without_hiding_later_entries(self):
        code = r'''
import ctypes
import json
import os
import subprocess
import sys

from loki_agent.credentials import capture_process_credentials

store = capture_process_credentials()
libc = ctypes.CDLL(None)
libc.getenv.argtypes = (ctypes.c_char_p,)
libc.getenv.restype = ctypes.c_char_p
raw = open("/proc/self/environ", "rb").read()
filler = b"x" * len(b"LOKI_TEST_TOKEN=top-secret")
child_code = (
    "import os; "
    "print(os.environ.get('LOKI_TEST_TOKEN', 'missing'))"
)
spawned = subprocess.check_output(
    [sys.executable, "-c", child_code],
    text=True,
).strip()
forked = subprocess.check_output(
    [sys.executable, "-c", child_code],
    text=True,
    preexec_fn=lambda: None,
).strip()
print(json.dumps({
    "stored": store.get("LOKI_TEST_TOKEN"),
    "python_has_secret": "LOKI_TEST_TOKEN" in os.environ,
    "native_secret": libc.getenv(b"LOKI_TEST_TOKEN") is not None,
    "native_after": libc.getenv(b"LOKI_TEST_AFTER") == b"after",
    "raw_has_name": b"LOKI_TEST_TOKEN=" in raw,
    "raw_has_value": b"top-secret" in raw,
    "filler_offset": raw.find(filler + b"\0"),
    "after_offset": raw.find(b"LOKI_TEST_AFTER=after\0"),
    "spawned": spawned,
    "forked": forked,
}))
'''
        env = {
            "LOKI_TEST_BEFORE": "before",
            "LOKI_TEST_TOKEN": "top-secret",
            "LOKI_TEST_AFTER": "after",
        }

        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["stored"], "top-secret")
        self.assertFalse(result["python_has_secret"])
        self.assertFalse(result["native_secret"])
        self.assertTrue(result["native_after"])
        self.assertFalse(result["raw_has_name"])
        self.assertFalse(result["raw_has_value"])
        self.assertGreaterEqual(result["filler_offset"], 0)
        self.assertGreater(
            result["after_offset"],
            result["filler_offset"],
        )
        self.assertEqual(result["spawned"], "missing")
        self.assertEqual(result["forked"], "missing")

    @unittest.skipUnless(
        sys.platform == "darwin", "Darwin KERN_PROCARGS2 semantics")
    def test_process_capture_scrubs_darwin_procargs_environment(self):
        code = r'''
import ctypes
import json
import os
import subprocess
import sys

from loki_agent.credentials import capture_process_credentials


def process_arguments():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sysctl.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    libc.sysctl.restype = ctypes.c_int
    mib = (ctypes.c_int * 3)(1, 49, os.getpid())
    size = ctypes.c_size_t(os.sysconf("SC_ARG_MAX"))
    buffer = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return buffer.raw[:size.value]


store = capture_process_credentials()
libc = ctypes.CDLL(None)
libc.getenv.argtypes = (ctypes.c_char_p,)
libc.getenv.restype = ctypes.c_char_p
raw = process_arguments()
filler = b"x" * len(b"LOKI_TEST_TOKEN=top-secret")
child_code = (
    "import os; "
    "print(os.environ.get('LOKI_TEST_TOKEN', 'missing'))"
)
spawned = subprocess.check_output(
    [sys.executable, "-c", child_code],
    text=True,
).strip()
print(json.dumps({
    "stored": store.get("LOKI_TEST_TOKEN"),
    "python_has_secret": "LOKI_TEST_TOKEN" in os.environ,
    "native_secret": libc.getenv(b"LOKI_TEST_TOKEN") is not None,
    "native_after": libc.getenv(b"LOKI_TEST_AFTER") == b"after",
    "raw_has_name": b"LOKI_TEST_TOKEN=" in raw,
    "raw_has_value": b"top-secret" in raw,
    "filler_offset": raw.find(filler + b"\0"),
    "after_offset": raw.find(b"LOKI_TEST_AFTER=after\0"),
    "spawned": spawned,
}))
'''
        env = {
            "LOKI_TEST_BEFORE": "before",
            "LOKI_TEST_TOKEN": "top-secret",
            "LOKI_TEST_AFTER": "after",
        }

        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["stored"], "top-secret")
        self.assertFalse(result["python_has_secret"])
        self.assertFalse(result["native_secret"])
        self.assertTrue(result["native_after"])
        self.assertFalse(result["raw_has_name"])
        self.assertFalse(result["raw_has_value"])
        self.assertGreaterEqual(result["filler_offset"], 0)
        self.assertGreater(
            result["after_offset"],
            result["filler_offset"],
        )
        self.assertEqual(result["spawned"], "missing")

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "Linux /proc environment semantics")
    def test_process_capture_scrubs_duplicate_execve_entries(self):
        second_stage = r'''
import json
import os

from loki_agent.credentials import capture_process_credentials

store = capture_process_credentials()
raw = open("/proc/self/environ", "rb").read()
records = raw.split(b"\0")
print(json.dumps({
    "stored": store.get("DUPLICATE_TOKEN"),
    "python_has_secret": "DUPLICATE_TOKEN" in os.environ,
    "raw_has_name": b"DUPLICATE_TOKEN=" in raw,
    "raw_has_one": b"one" in raw,
    "raw_has_two": b"two" in raw,
    "filler_count": records.count(
        b"x" * len(b"DUPLICATE_TOKEN=one")),
    "after": b"AFTER=visible" in records,
}))
'''
        launcher = f'''
import ctypes
import os
import sys

libc = ctypes.CDLL(None, use_errno=True)
libc.execve.argtypes = (
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
)
libc.execve.restype = ctypes.c_int
executable = os.fsencode(sys.executable)
code = {second_stage.encode("utf-8")!r}
argv = (ctypes.c_char_p * 4)(executable, b"-c", code, None)
entries = [
    b"BEFORE=visible",
    b"DUPLICATE_TOKEN=one",
    b"DUPLICATE_TOKEN=two",
    b"AFTER=visible",
]
envp = (ctypes.c_char_p * (len(entries) + 1))(*entries, None)
libc.execve(executable, argv, envp)
raise OSError(ctypes.get_errno(), "execve failed")
'''

        process = subprocess.run(
            [sys.executable, "-c", launcher],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertIn(result["stored"], ("one", "two"))
        self.assertFalse(result["python_has_secret"])
        self.assertFalse(result["raw_has_name"])
        self.assertFalse(result["raw_has_one"])
        self.assertFalse(result["raw_has_two"])
        self.assertEqual(result["filler_count"], 2)
        self.assertTrue(result["after"])


class ConnectionDescriptorTests(unittest.TestCase):
    def test_round_trip_contains_names_but_no_values(self):
        effort_profile = models.ReasoningEffortProfile(
            ("low", "high"))
        descriptor = ConnectionDescriptor(
            provider_id="openrouter",
            provider_name="OpenRouter",
            model="z-ai/glm",
            chat_url="https://openrouter.ai/api/v1/chat/completions",
            models_url="https://openrouter.ai/api/v1/models",
            protocol="openai_chat",
            credential_ref=CredentialRef.environment(
                "OPENROUTER_API_KEY"),
            model_status="deprecated",
            prompt_cache=True,
            reasoning_effort_profile=effort_profile,
        )

        encoded = descriptor.to_dict()
        encoded["reasoning_effort_profile"]["default_value"] = {
            "ignored": True}
        encoded["reasoning_effort_profile"]["options"][0][
            "description"] = {"ignored": True}
        restored = ConnectionDescriptor.from_dict(encoded)

        self.assertEqual(restored, descriptor)
        self.assertEqual(encoded["model_status"], "deprecated")
        self.assertIs(encoded["prompt_cache"], True)
        self.assertEqual(
            restored.to_dict()["reasoning_effort_profile"],
            effort_profile.to_dict(),
        )
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
        self.assertEqual(
            descriptor.credential_ref,
            CredentialRef.environment("OPENROUTER_API_KEY"),
        )
        self.assertNotIn("api_url", descriptor.to_dict())
        self.assertNotIn("credential_env", descriptor.to_dict())
        self.assertIsNone(descriptor.model_status)

    def test_credentialless_connection_round_trips_explicit_null(self):
        descriptor = ConnectionDescriptor(
            provider_id=None,
            provider_name="Explicit LOKI_* connection",
            model="local-model",
            chat_url="http://localhost:8000/v1/chat/completions",
            models_url="http://localhost:8000/v1/models",
            protocol="openai_chat",
            stream=True,
        )

        encoded = descriptor.to_dict()

        self.assertNotIn("credential_env", encoded)
        self.assertIsNone(encoded["credential"])
        self.assertIs(encoded["stream"], True)
        self.assertEqual(ConnectionDescriptor.from_dict(encoded), descriptor)

    def test_legacy_and_current_credential_identities_must_agree(self):
        with self.assertRaisesRegex(
                ConnectionDescriptorError, "credential.*disagree"):
            ConnectionDescriptor.from_dict({
                "model": "model",
                "chat_url": "https://example.test/v1/chat/completions",
                "models_url": "https://example.test/v1/models",
                "protocol": "openai_chat",
                "credential_env": "OLD_API_KEY",
                "credential": {
                    "kind": "env",
                    "name": "NEW_API_KEY",
                },
            })

    def test_subscription_connection_preserves_authentication_policy(self):
        descriptor = ConnectionDescriptor(
            provider_id="openai-subscription",
            provider_name="OpenAI ChatGPT subscription",
            model="gpt-5-codex",
            chat_url="https://chatgpt.com/backend-api/codex/responses",
            models_url=OPENAI_CHATGPT_MODELS_REQUEST_URL,
            protocol="openai_responses",
            credential_ref=CredentialRef.openai_subscription(),
            auth_scheme="openai-subscription",
            stream=True,
            openai_request_profile=_codex_model(
                use_responses_lite=True,
                context_window=200000,
            ),
        )

        encoded = descriptor.to_dict()
        restored = ConnectionDescriptor.from_dict(encoded)

        self.assertEqual(restored, descriptor)
        self.assertEqual(
            encoded["credential"],
            {"kind": "openai-subscription", "name": "openai"},
        )
        self.assertEqual(encoded["auth_scheme"], "openai-subscription")
        self.assertIs(
            encoded["openai_request_profile"]["use_responses_lite"], True)
        self.assertNotIn(
            "context_window", encoded["openai_request_profile"])

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
        with self.assertRaises(ConnectionDescriptorError):
            ConnectionDescriptor.from_dict({
                "provider_id": "openai-subscription",
                "model": "x",
                "chat_url":
                    "https://chatgpt.com/backend-api/codex/responses",
                "protocol": "openai_responses",
                "credential_env": None,
                "credential": {
                    "kind": "openai-subscription",
                    "name": "openai",
                },
                "openai_request_profile": {
                    "supports_parallel_tool_calls": "yes",
                },
            })


if __name__ == "__main__":
    unittest.main()
