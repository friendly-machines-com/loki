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

    def test_process_capture_scrubs_native_environment(self):
        # One property on every platform: after capture, the environment a child
        # inherits no longer contains the credential, while a later variable
        # survives in the same view.
        #
        # Why the Windows check is safe.  ``GetEnvironmentStringsW`` returns the
        # process environment block, which is exactly what ``CreateProcess``
        # copies to a child when ``lpEnvironment`` is NULL (documented), and
        # removing a variable through ``os.environ`` calls
        # ``SetEnvironmentVariableW``, which modifies that block.  The spawned
        # child below is the positive control that this is the environment
        # children receive, and ``native_after`` shows the check is not vacuous
        # (a later variable is still present).  So this asserts on Windows what
        # /proc/<pid>/environ and KERN_PROCARGS2 assert on Linux and Darwin: the
        # environment a child inherits no longer contains the credential.  Not
        # established: bytes left in freed memory or the original allocation
        # after removal -- the reference documents no contract for that.
        code = r'''
import json
import os
import subprocess
import sys

from loki_agent.credentials import capture_process_credentials


def native_environment():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        get = kernel.GetEnvironmentStringsW
        get.restype = ctypes.c_void_p
        free = kernel.FreeEnvironmentStringsW
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_int
        pointer = get()
        if not pointer:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entries, address = [], pointer
            while True:
                text = ctypes.wstring_at(address)
                if not text:
                    break
                entries.append(text)
                address += (len(text) + 1) * ctypes.sizeof(ctypes.c_wchar)
            return "\0".join(entries).encode("utf-8")
        finally:
            free(pointer)
    if sys.platform == "darwin":
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.sysctl.argtypes = (
            ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t)
        libc.sysctl.restype = ctypes.c_int
        mib = (ctypes.c_int * 3)(1, 49, os.getpid())
        size = ctypes.c_size_t(os.sysconf("SC_ARG_MAX"))
        buffer = ctypes.create_string_buffer(size.value)
        if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), "sysctl KERN_PROCARGS2 failed")
        return buffer.raw[:size.value]
    return open("/proc/self/environ", "rb").read()


store = capture_process_credentials()
raw = native_environment()
text = raw.decode("utf-8", "replace")
child_code = "import os; print(os.environ.get('LOKI_TEST_TOKEN', 'missing'))"
spawned = subprocess.check_output(
    [sys.executable, "-c", child_code], text=True).strip()
forked = None
if os.name == "posix":
    forked = subprocess.check_output(
        [sys.executable, "-c", child_code], text=True,
        preexec_fn=lambda: None).strip()
result = {
    "stored": store.get("LOKI_TEST_TOKEN"),
    "python_has_secret": "LOKI_TEST_TOKEN" in os.environ,
    "native_has_name": "LOKI_TEST_TOKEN=" in text,
    "native_has_value": "top-secret" in text,
    "native_after": "LOKI_TEST_AFTER=after" in text,
    "spawned": spawned,
    "forked": forked,
}
if sys.platform.startswith("linux"):
    # Linux overwrites the original records in place (unsetenv alone leaves the
    # initial exec region readable), so the lane also checks the overwrite.
    filler = b"x" * len(b"LOKI_TEST_TOKEN=top-secret")
    result["filler_offset"] = raw.find(filler + b"\0")
    result["after_offset"] = raw.find(b"LOKI_TEST_AFTER=after\0")
print(json.dumps(result))
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
        self.assertFalse(result["native_has_name"])
        self.assertFalse(result["native_has_value"])
        self.assertTrue(result["native_after"])
        self.assertEqual(result["spawned"], "missing")
        if result["forked"] is not None:
            self.assertEqual(result["forked"], "missing")
        if "filler_offset" in result:
            self.assertGreaterEqual(result["filler_offset"], 0)
            self.assertGreater(result["after_offset"], result["filler_offset"])

    def test_process_capture_scrubs_duplicate_entries(self):
        # The scrub handles several records for the same name (the validator
        # collects a range per match).  Linux reaches that with execve and a
        # duplicate-laden environment; Windows reaches it by handing
        # CreateProcessW an environment block that contains the duplicates, so
        # the same property is exercised on both.
        second_stage = r'''
import json
import os
import sys

from loki_agent.credentials import capture_process_credentials


def native_environment():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        get = kernel.GetEnvironmentStringsW
        get.restype = ctypes.c_void_p
        free = kernel.FreeEnvironmentStringsW
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_int
        pointer = get()
        try:
            entries, address = [], pointer
            while True:
                text = ctypes.wstring_at(address)
                if not text:
                    break
                entries.append(text)
                address += (len(text) + 1) * ctypes.sizeof(ctypes.c_wchar)
            return list(entries), "\0".join(entries).encode("utf-8")
        finally:
            free(pointer)
    if sys.platform == "darwin":
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.sysctl.argtypes = (
            ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t)
        libc.sysctl.restype = ctypes.c_int
        mib = (ctypes.c_int * 3)(1, 49, os.getpid())
        size = ctypes.c_size_t(os.sysconf("SC_ARG_MAX"))
        buffer = ctypes.create_string_buffer(size.value)
        if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), "sysctl failed")
        raw = buffer.raw[:size.value]
    else:
        raw = open("/proc/self/environ", "rb").read()
    return raw.split(b"\0"), raw


store = capture_process_credentials()
entries, raw = native_environment()
text = raw.decode("utf-8", "replace")
result = {
    "stored": store.get("DUPLICATE_TOKEN"),
    "python_has_secret": "DUPLICATE_TOKEN" in os.environ,
    "native_has_name": "DUPLICATE_TOKEN=" in text,
    "native_has_one": "one" in text,
    "native_has_two": "two" in text,
    "after": "AFTER=visible" in text,
}
if sys.platform.startswith("linux"):
    filler = b"x" * len(b"DUPLICATE_TOKEN=one")
    result["filler_count"] = entries.count(filler)
print(json.dumps(result))
'''
        launcher = r'''
import ctypes
import os
import subprocess
import sys

second_stage = %r

if os.name == "nt":
    from ctypes import wintypes

    class Startup(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("reserved", wintypes.LPWSTR),
            ("desktop", wintypes.LPWSTR), ("title", wintypes.LPWSTR),
            ("x", wintypes.DWORD), ("y", wintypes.DWORD),
            ("xsize", wintypes.DWORD), ("ysize", wintypes.DWORD),
            ("xchars", wintypes.DWORD), ("ychars", wintypes.DWORD),
            ("fill", wintypes.DWORD), ("flags", wintypes.DWORD),
            ("show", wintypes.WORD), ("reserved_size", wintypes.WORD),
            ("reserved_bytes", ctypes.c_void_p),
            ("stdin", ctypes.c_void_p), ("stdout", ctypes.c_void_p),
            ("stderr", ctypes.c_void_p)]

    class Information(ctypes.Structure):
        _fields_ = [("process", ctypes.c_void_p), ("thread", ctypes.c_void_p),
                    ("pid", wintypes.DWORD), ("tid", wintypes.DWORD)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateProcessW
    create.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p,
                       ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.LPCWSTR,
                       ctypes.POINTER(Startup),
                       ctypes.POINTER(Information)]
    create.restype = wintypes.BOOL
    entries = ["BEFORE=visible", "DUPLICATE_TOKEN=one",
               "DUPLICATE_TOKEN=two", "AFTER=visible"]
    block = ctypes.create_unicode_buffer("\0".join(entries) + "\0\0")
    startup = Startup()
    startup.cb = ctypes.sizeof(Startup)
    info = Information()
    command = ctypes.create_unicode_buffer(subprocess.list2cmdline(
        [sys.executable, "-c", second_stage]))
    # CREATE_UNICODE_ENVIRONMENT: the block handed to CreateProcessW is
    # Unicode, and without the flag the API reads it as ANSI (WinError 87).
    if not create(sys.executable, command, None, None, False, 0x400, block,
                  None, ctypes.byref(startup), ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.WaitForSingleObject(info.process, 10000)
    kernel.CloseHandle(info.thread)
    kernel.CloseHandle(info.process)
else:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.execve.argtypes = (
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p))
    libc.execve.restype = ctypes.c_int
    executable = os.fsencode(sys.executable)
    code = second_stage.encode("utf-8")
    argv = (ctypes.c_char_p * 4)(executable, b"-c", code, None)
    entries = [b"BEFORE=visible", b"DUPLICATE_TOKEN=one",
               b"DUPLICATE_TOKEN=two", b"AFTER=visible"]
    envp = (ctypes.c_char_p * (len(entries) + 1))(*entries, None)
    libc.execve(executable, argv, envp)
    raise OSError(ctypes.get_errno(), "execve failed")
''' % (second_stage,)

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
        self.assertFalse(result["native_has_name"])
        self.assertFalse(result["native_has_one"])
        self.assertFalse(result["native_has_two"])
        self.assertTrue(result["after"])
        if "filler_count" in result:
            self.assertEqual(result["filler_count"], 2)


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
