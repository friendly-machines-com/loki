import contextlib
import io
import os
import stat
import tempfile
import unittest
from unittest import mock
from unittest.mock import AsyncMock

from loki_agent import endpoint_pins
from loki_agent import loki
from loki_agent import models
from loki_agent.credentials import CredentialStore


API = "https://acme.invalid/v1"
OTHER_API = "https://elsewhere.invalid/v1"
CREDENTIAL = "env:ACME_API_KEY"


def catalog(api=API, model_extra=None):
    model = {"id": "m", "name": "M"}
    if model_extra is not None:
        model.update(model_extra)
    return {"acme": {
        "id": "acme",
        "name": "Acme",
        "npm": "@ai-sdk/openai-compatible",
        "api": api,
        "env": ["ACME_API_KEY"],
        "models": {"m": model},
    }}


class _StateDir:
    """Isolate the pin store in a temporary state directory."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(
            os.environ, {"XDG_STATE_HOME": self._tmp.name})
        self._env.start()
        return self._tmp.name

    def __exit__(self, *exc):
        self._env.stop()
        self._tmp.cleanup()
        return False


class PinStoreTests(unittest.TestCase):
    def test_different_endpoint_is_changed_with_approved_value(self):
        with _StateDir():
            endpoint_pins.record("acme", API, CREDENTIAL)

            self.assertEqual(
                endpoint_pins.status("acme", OTHER_API, CREDENTIAL),
                (endpoint_pins.CHANGED,
                 {"api": API, "credential": CREDENTIAL}))

    def test_swapped_credential_is_changed(self):
        """An approved endpoint with a different secret is not approved."""
        with _StateDir():
            endpoint_pins.record("acme", API, CREDENTIAL)

            self.assertEqual(
                endpoint_pins.status("acme", API, "env:OTHER_API_KEY"),
                (endpoint_pins.CHANGED,
                 {"api": API, "credential": CREDENTIAL}))

    def test_corrupt_store_reads_as_empty(self):
        with _StateDir() as state:
            path = os.path.join(state, "loki", "provider-endpoints.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as stream:
                stream.write("{not json")

            self.assertEqual(endpoint_pins.load(), {})
            self.assertEqual(
                endpoint_pins.status("acme", API, CREDENTIAL),
                (endpoint_pins.NEW, None))

    def test_store_is_owner_only(self):
        if os.name != "posix":
            self.skipTest("POSIX permissions")
        with _StateDir() as state:
            endpoint_pins.record("acme", API, CREDENTIAL)

            path = os.path.join(state, "loki", "provider-endpoints.json")
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertEqual(
                stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode), 0o700)

    def test_rejects_invalid_arguments(self):
        with _StateDir():
            for provider_id, api_url, credential in (
                    ("", API, CREDENTIAL),
                    ("acme", "", CREDENTIAL),
                    ("acme", API, "")):
                with self.assertRaises(ValueError):
                    endpoint_pins.record(provider_id, api_url, credential)


class ConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def _confirm(self, answers, provider_entry=None, credentials=None):
        entry = provider_entry or catalog()["acme"]
        model = entry["models"]["m"]
        printed = []
        with contextlib.redirect_stdout(io.StringIO()):
            picked = await models._confirm_catalog_endpoint(
                AsyncMock(side_effect=answers),
                printed.append,
                credentials or CredentialStore({"ACME_API_KEY": "secret"}),
                "acme",
                entry,
                model,
            )
        return picked, "\n".join(printed)

    async def test_declining_leaves_endpoint_unapproved(self):
        with _StateDir():
            picked, shown = await self._confirm([""])

            self.assertFalse(picked)
            self.assertIn(API, shown)
            self.assertEqual(
                endpoint_pins.status("acme", API, CREDENTIAL)[0],
                endpoint_pins.NEW)

    async def test_changed_pair_shows_both_values(self):
        from loki_agent import http_client

        for api, credential_name in ((OTHER_API, "ACME_API_KEY"), (API, "OTHER_API_KEY")):
            with self.subTest(api=api, credential=credential_name), _StateDir() as state:
                endpoint_pins.record("acme", API, CREDENTIAL)
                path = os.path.join(state, "loki", "provider-endpoints.json")
                with open(path, "rb") as stream:
                    before = stream.read()
                entry = catalog(api=api)["acme"]
                entry["env"] = [credential_name]
                credentials = CredentialStore({credential_name: "must-not-be-sent"})
                original_entry = catalog()["acme"]
                original_config = loki.config_from_modelsdev_selection(
                    "acme", original_entry, original_entry["models"]["m"],
                    CredentialStore({"ACME_API_KEY": "approved-secret"}))
                session = loki.Session(runtime_config=original_config)
                with mock.patch.object(loki, "_DEFAULT_SESSION", session), mock.patch.object(
                        http_client, "async_http_request", new=AsyncMock()) as transport:
                    for _ in range(2):
                        picked, shown = await self._confirm(
                            ["n"], provider_entry=entry, credentials=credentials)
                        self.assertFalse(picked)
                        self.assertIn(API, shown)
                        self.assertIn(api, shown)
                        self.assertIn(CREDENTIAL, shown)
                        self.assertIn("env:" + credential_name, shown)
                        with self.assertRaisesRegex(ValueError, "differs from the approved endpoint"):
                            loki.config_from_modelsdev_selection(
                                "acme", entry, entry["models"]["m"], credentials)
                        self.assertIs(session.runtime_config, original_config)
                        with open(path, "rb") as stream:
                            self.assertEqual(stream.read(), before)
                    transport.assert_not_awaited()
                self.assertEqual(endpoint_pins.status("acme", API, CREDENTIAL)[0], endpoint_pins.PINNED)


class PickerTests(unittest.IsolatedAsyncioTestCase):
    async def test_declining_returns_none(self):
        from loki_agent import http_client

        entry = catalog()
        credentials = CredentialStore({"ACME_API_KEY": "must-not-be-sent"})
        session = loki.Session()
        with _StateDir(), mock.patch.object(loki, "_DEFAULT_SESSION", session), \
                mock.patch.object(models, "_index_cache", (entry, models.build_groups(entry))), \
                mock.patch.object(http_client, "async_http_request", new=AsyncMock()) as transport, \
                contextlib.redirect_stdout(io.StringIO()):
            inputs = AsyncMock(side_effect=["1", "1", ""])
            self.assertIsNone(await models.run_model_picker_async(
                inputs, credentials, text_writer=lambda text: None))
            self.assertEqual(inputs.await_count, 3)
            self.assertEqual(endpoint_pins.load(), {})
            self.assertEqual(endpoint_pins.status("acme", API, CREDENTIAL), (endpoint_pins.NEW, None))
            with self.assertRaisesRegex(ValueError, "has not been approved"):
                loki.config_from_modelsdev_selection("acme", entry["acme"], entry["acme"]["models"]["m"], credentials)
            self.assertIsNone(session.runtime_config)
            transport.assert_not_awaited()


class SelectionRefusalTests(unittest.IsolatedAsyncioTestCase):
    def _selection(self, api=API):
        entry = catalog(api=api)
        return (entry["acme"], entry["acme"]["models"]["m"],
                CredentialStore({"ACME_API_KEY": "secret"}))

    async def test_unapproved_pair_is_refused(self):
        with _StateDir():
            provider_entry, model_entry, credentials = self._selection()

            with self.assertRaises(ValueError) as raised:
                loki.config_from_modelsdev_selection(
                    "acme", provider_entry, model_entry, credentials)

            self.assertIn(API, str(raised.exception))
            self.assertIn("approve it once", str(raised.exception))

    async def test_changed_endpoint_names_the_approved_one(self):
        with _StateDir():
            endpoint_pins.record("acme", API, CREDENTIAL)
            provider_entry, model_entry, credentials = self._selection(
                api=OTHER_API)

            with self.assertRaises(ValueError) as raised:
                loki.config_from_modelsdev_selection(
                    "acme", provider_entry, model_entry, credentials)

            self.assertIn(OTHER_API, str(raised.exception))
            self.assertIn(API, str(raised.exception))


class WorkerSelectionTests(unittest.TestCase):
    """The read-only answer the ACP front turns into an approval request."""

    def _worker_for(self, leaf):
        from loki_agent import acp_worker
        from loki_agent.sessions import Session

        worker = acp_worker.Worker(Session(shell_cwd="/tmp"),
                                   lambda message: None, "s")
        worker._option_leaves = {"v": leaf}
        return worker

    def _with_credentials(self, values):
        old = loki.CREDENTIALS
        loki.CREDENTIALS = CredentialStore(values)
        self.addCleanup(setattr, loki, "CREDENTIALS", old)

    def test_reports_a_change_with_the_approved_pair(self):
        with _StateDir():
            endpoint_pins.record("acme", API, CREDENTIAL)
            entry = catalog(api=OTHER_API)
            self._with_credentials({"ACME_API_KEY": "k"})
            worker = self._worker_for(
                ("acme", entry["acme"], entry["acme"]["models"]["m"]))

            selection = worker.describe_config_selection({"value": "v"})

            self.assertTrue(selection["changed"])
            self.assertEqual(selection["approvedEndpoint"], API)
            self.assertEqual(selection["approvedCredential"], CREDENTIAL)

    def test_synthetic_provider_needs_no_approval(self):
        response = {"models": [{
            "slug": "gpt-x", "visibility": "list",
            "input_modalities": ["text"], "display_name": "GPT X"}]}
        entry = models.add_openai_subscription_catalog({}, response)
        provider = entry["openai-subscription"]
        self.assertTrue(models.provider_is_synthetic(provider))
        self._with_credentials({})
        worker = self._worker_for(
            ("openai-subscription", provider,
             provider["models"]["gpt-x"]))

        self.assertEqual(worker.describe_config_selection({"value": "v"}), {})

    def test_downloaded_entry_cannot_be_synthetic(self):
        entry = catalog()
        entry["acme"]["_loki_synthetic"] = "sentinel"
        entry["acme"]["_loki_credential_ref"] = "openai-subscription:openai"

        self.assertFalse(models.provider_is_synthetic(entry["acme"]))

    def test_non_catalog_leaves_need_no_approval(self):
        worker = self._worker_for(models.ExplicitConnectionOption(
            model="m", api_url=API, protocol="openai_chat"))

        self.assertEqual(worker.describe_config_selection({"value": "v"}), {})


if __name__ == "__main__":
    unittest.main()
