import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock
from loki_entrypoints import configure_container, entrypoint
from response_header_fixtures import isolated_response_headers
from process_lifecycle_fixtures import close_fixture_connection
from response_header_fixtures import setUpModule  # noqa: F401 - unittest hook

from loki_agent import file_locks, private_files
from loki_agent import http_client, loki, protocols, response_headers
from loki_agent.sessions import Session
from loki_agent import connections
from loki_agent import model_requests


class ResponseHeadersTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.path = os.path.join(self.directory, "headers.json")
        self.store = self.enterContext(contextlib.closing(response_headers.Store(self.path)))

    def test_explicit_invalid_filename_does_not_select_default_or_create_storage(self):
        with mock.patch.object(response_headers, '_snapshot_location', side_effect=AssertionError('default selected')), \
                mock.patch.object(private_files, 'open_directory_tree') as acquire:
            for operand in ['', '.', '..', self.path + '/']:
                with self.subTest(operand=operand), self.assertRaisesRegex(ValueError, 'must name a file'):
                    response_headers.Store(operand)
            acquire.assert_not_called()

    def observe(self, store=None, headers=None, status=200, model="one",
                credential="env:KEY", provider=None):
        (store or self.store).observer(
            "https://example.com/chat", credential, model, provider)(
                status, headers or {"x-remaining": "10"})

    async def test_runtime_cleanup_uses_fixture_storage_not_preexisting_user_state(self):
        protected = Path(self.path)
        protected.write_bytes(b"user-owned status must not be read or changed")
        original = protected.read_bytes()
        existing_store = self.enterContext(contextlib.closing(response_headers.Store(self.path)))
        existing_session = Session(response_headers=existing_store)
        with mock.patch.object(response_headers, '_snapshot_location',
                               return_value=(self.directory, 'headers.json')), \
                mock.patch.object(loki, "_DEFAULT_SESSION", existing_session):
            with isolated_response_headers() as isolated_path:
                # Cover a pre-existing default and a newly constructed session,
                # releasing both test-owned stores before their directory.
                new_session = Session()
                with contextlib.closing(new_session.response_headers):
                    for index, session in enumerate([loki.current_session(), new_session]):
                        session.response_headers.observer(
                            f"https://test.example/{index}", None, "test-model")(
                                200, {"content-type": "application/json"})
                        await session.response_headers.save_on_exit()
                    with contextlib.closing(response_headers.Store(isolated_path)) as inspected:
                        saved = inspected.snapshot()['endpoints']
                    self.assertEqual({entry["endpoint"] for entry in saved}, {
                        "https://test.example/0", "https://test.example/1"})
                    self.assertEqual(protected.read_bytes(), original)
        self.assertFalse(Path(isolated_path).exists())
        self.assertEqual(protected.read_bytes(), original)

    async def test_updates_keys_retains_absent_and_does_not_write(self):
        with mock.patch.object(response_headers.time, "time_ns", return_value=1):
            self.observe(headers={"X-Remaining": "10", "x-reset": "soon"})
        with mock.patch.object(response_headers.time, "time_ns", return_value=2):
            self.observe(headers={"x-remaining": "9"}, status=429, model="two",
                         provider="optional-catalog-label")
        self.assertFalse(os.path.exists(self.path))
        entries = self.store.snapshot()["endpoints"]
        self.assertEqual(len(entries), 1)
        headers = entries[0]["headers"]
        self.assertEqual(headers["x-remaining"]["value"], "9")
        self.assertEqual(headers["x-reset"]["observed_at_ns"], 1)
        self.assertEqual(headers["x-remaining"]["status"], 429)
        self.assertEqual(entries[0]["latest"]["header_names"], ["x-remaining"])
        self.assertIn("retained", response_headers.render(self.store.snapshot()))

    async def test_retry_after_seconds_preserves_raw_and_counts_down(self):
        with mock.patch.object(response_headers.time, "time_ns", return_value=1_000_000_000):
            self.observe(headers={"Retry-After": "34323"}, status=429)
        document = self.store.snapshot()
        original = json.dumps(document, sort_keys=True)
        with mock.patch.object(response_headers.time, "time", return_value=1):
            text = response_headers.render(document)
        self.assertIn("'retry-after': '34323'", text)
        self.assertIn("until 1970-01-01 09:32:04 UTC (in 9 hours 32 minutes)", text)
        with mock.patch.object(response_headers.time, "time", return_value=34323):
            self.assertIn("(in 1 second)", response_headers.render(document))
        self.assertEqual(json.dumps(document, sort_keys=True), original)

    async def test_retry_after_http_dates(self):
        for value in ["Thu, 01 Jan 1970 00:02:00 GMT",
                      "Thursday, 01-Jan-70 00:02:00 GMT",
                      "Thu Jan  1 00:02:00 1970"]:
            with self.subTest(value=value):
                self.observe(headers={"retry-after": value}, status=429)
                with mock.patch.object(response_headers.time, "time", return_value=60):
                    text = response_headers.render(self.store.snapshot())
                self.assertIn("until 1970-01-01 00:02:00 UTC (in 1 minute)", text)

    async def test_retry_after_retained_saved_and_expired(self):
        with mock.patch.object(response_headers.time, "time_ns", return_value=1_000_000_000):
            self.observe(headers={"retry-after": "120"}, status=429)
        with mock.patch.object(response_headers.time, "time_ns", return_value=30_000_000_000):
            self.observe(headers={"x-other": "yes"})
        await self.store.save()
        with contextlib.closing(response_headers.Store(self.path)) as reopened:
            document = reopened.snapshot()
        with mock.patch.object(response_headers.time, "time", return_value=120.5):
            text = response_headers.render(document)
        self.assertIn("00:02:01 UTC (in 1 second) [retained observation]", text)
        with mock.patch.object(response_headers.time, "time", return_value=121):
            text = response_headers.render(document)
        self.assertIn("00:02:01 UTC has passed", text)
        self.assertIn("availability has not been rechecked", text)

    async def test_retry_after_zero_invalid_and_overflow(self):
        with mock.patch.object(response_headers.time, "time_ns", return_value=1_000_000_000):
            self.observe(headers={"retry-after": "0"}, status=429)
        with mock.patch.object(response_headers.time, "time", return_value=1):
            self.assertIn("has passed", response_headers.render(self.store.snapshot()))
        for value in ["", "-1", "+1", "1.5", "NaN", "soon", "9" * 400,
                      "Thu, 99 Jan 1970 00:00:00 GMT", "120\n", "\x1b[31m"]:
            with self.subTest(value=value):
                self.observe(headers={"retry-after": value}, status=429)
                text = response_headers.render(self.store.snapshot())
                self.assertIn("Retry-After: unrecognized value", text)
                self.assertIn(ascii(value), text)

    def codex_observer(self):
        return self.store.observer(
            "https://chatgpt.com/backend-api/codex/responses",
            "openai-subscription:openai", "test-model")

    async def test_codex_summary_added_without_replacing_raw_headers(self):
        self.codex_observer()(200, {
            "x-codex-primary-used-percent": "56",
            "x-codex-primary-window-minutes": "10080",
            "x-codex-secondary-used-percent": "0",
            "x-codex-secondary-window-minutes": "0",
            "x-codex-bengalfox-limit-name": "GPT-5.3-Codex-Spark",
            "x-codex-bengalfox-primary-used-percent": "0",
            "x-codex-bengalfox-primary-window-minutes": "300",
            "x-codex-bengalfox-secondary-used-percent": "0",
            "x-codex-bengalfox-secondary-window-minutes": "10080",
        })
        document = self.store.snapshot()
        before = json.dumps(document)
        text = response_headers.render(document)
        self.assertIn("Main subscription bucket: 56% used, 44% remaining; window: 7 days", text)
        self.assertIn("GPT-5.3-Codex-Spark: 0% used, 100% remaining; window: 5 hours", text)
        self.assertIn("GPT-5.3-Codex-Spark: 0% used, 100% remaining; window: 7 days", text)
        self.assertEqual(text.count("% remaining"), 3)
        self.assertEqual(text.count("reset time unknown"), 3)
        self.assertIn("'x-codex-primary-used-percent': '56'", text)
        self.assertEqual(json.dumps(document), before)
        document["endpoints"][0]["endpoint"] = "https://other.example/chat"
        self.assertNotIn("Subscription quota", response_headers.render(document))

    async def test_codex_summary_rejects_bad_values_and_escapes_labels(self):
        for used, minutes in [("NaN", "300"), ("inf", "300"),
                              ("101", "300"), ("-1", "300"),
                              ("56", "-1"), ("56", ""), ("56", "1.5")]:
            with self.subTest(used=used, minutes=minutes):
                self.codex_observer()(200, {
                    "x-codex-primary-used-percent": used,
                    "x-codex-primary-window-minutes": minutes,
                })
                self.assertNotIn("Subscription quota", response_headers.render(
                    self.store.snapshot()))
        self.codex_observer()(200, {
            "x-codex-new-limit-name": "Model\x1b[2J\nspoof",
            "x-codex-new-primary-used-percent": "12.5",
            "x-codex-new-primary-window-minutes": "60",
        })
        text = response_headers.render(self.store.snapshot())
        self.assertIn("12.5% used, 87.5% remaining; window: 1 hour", text)
        self.assertNotIn("\x1b", text)
        self.assertIn("Model\\x1b[2J\\nspoof", text)

    async def test_codex_summary_does_not_mix_observations_in_one_tick(self):
        # Windows can return the same time_ns() for two responses. The summary
        # must still not pair a window from one with a used-percent from the
        # other; only the observation id says which values belong together.
        with mock.patch.object(response_headers.time, "time_ns",
                               return_value=1):
            observer = self.codex_observer()
            observer(200, {"x-codex-primary-window-minutes": "300"})
            observer(200, {"x-codex-primary-used-percent": "56"})
        self.assertNotIn(
            "Subscription quota",
            response_headers.render(self.store.snapshot()))

    async def test_codex_summary_marks_retained_and_omits_mixed_observations(self):
        with mock.patch.object(response_headers.time, "time_ns", return_value=1):
            self.codex_observer()(200, {
                "x-codex-primary-used-percent": "56",
                "x-codex-primary-window-minutes": "10080",
            })
        with mock.patch.object(response_headers.time, "time_ns", return_value=2):
            self.codex_observer()(200, {"date": "later"})
        self.assertIn(
            "window: 7 days; reset time unknown [retained observation]",
            response_headers.render(self.store.snapshot()))
        with mock.patch.object(response_headers.time, "time_ns", return_value=3):
            self.codex_observer()(200, {"x-codex-primary-used-percent": "57"})
        text = response_headers.render(self.store.snapshot())
        self.assertNotIn("Subscription quota", text)
        self.assertIn("'x-codex-primary-used-percent': '57'", text)

    async def test_codex_reset_dates_and_countdowns_are_window_specific(self):
        start = 1704067200  # 2024-01-01 00:00:00 UTC
        with mock.patch.object(response_headers.time, "time_ns",
                               return_value=start * 10**9):
            self.codex_observer()(200, {
                "x-codex-primary-used-percent": "54",
                "x-codex-primary-window-minutes": "10080",
                "x-codex-primary-reset-at": str(start + 3 * 86400 + 4 * 3600),
                "x-codex-bengalfox-limit-name": "GPT-5.3-Codex-Spark",
                "x-codex-bengalfox-primary-used-percent": "0",
                "x-codex-bengalfox-primary-window-minutes": "300",
                "x-codex-bengalfox-primary-reset-at": str(start + 7200),
                "x-codex-bengalfox-secondary-used-percent": "0",
                "x-codex-bengalfox-secondary-window-minutes": "10080",
                "x-codex-bengalfox-secondary-reset-at": str(start + 86400),
            })
        document = self.store.snapshot()
        before = json.dumps(document)
        with mock.patch.object(response_headers.time, "time", return_value=start + 3600):
            text = response_headers.render(document)
        self.assertIn(
            "Main subscription bucket: 54% used, 46% remaining; window: 7 days; "
            "resets at 2024-01-04 04:00:00 UTC (in 3 days 3 hours)", text)
        self.assertIn(
            "GPT-5.3-Codex-Spark: 0% used, 100% remaining; window: 5 hours; "
            "resets at 2024-01-01 02:00:00 UTC (in 1 hour)", text)
        self.assertIn(
            "GPT-5.3-Codex-Spark: 0% used, 100% remaining; window: 7 days; "
            "resets at 2024-01-02 00:00:00 UTC (in 23 hours)", text)
        self.assertEqual(json.dumps(document), before)

    async def test_saved_retained_reset_ages_without_resetting_reported_usage(self):
        start = 1704067200
        observer = self.codex_observer()
        with mock.patch.object(response_headers.time, "time_ns",
                               return_value=start * 10**9):
            observer(200, {
                "x-codex-primary-used-percent": "54",
                "x-codex-primary-window-minutes": "10080",
                "x-codex-primary-reset-at": str(start + 7200),
            })
        with mock.patch.object(response_headers.time, "time_ns",
                               return_value=(start + 60) * 10**9):
            observer(200, {"date": "a later response without quota headers"})
        await self.store.save()
        with contextlib.closing(response_headers.Store(self.path)) as reopened:
            document = reopened.snapshot()
        for elapsed, expected in (
                (3600, "resets at 2024-01-01 02:00:00 UTC (in 1 hour)"),
                (7199.5, "resets at 2024-01-01 02:00:00 UTC (in 1 second)"),
                (7200, "reported reset at 2024-01-01 02:00:00 UTC has passed; "
                 "use /account to refresh"),
                (86400, "reported reset at 2024-01-01 02:00:00 UTC has passed; "
                 "use /account to refresh")):
            with self.subTest(elapsed=elapsed), mock.patch.object(
                    response_headers.time, "time", return_value=start + elapsed):
                text = response_headers.render(document)
                self.assertIn(expected + " [retained observation]", text)
                self.assertIn("54% used, 46% remaining; window: 7 days", text)
                self.assertNotIn("resets in 0", text)

    async def test_reset_must_match_usage_observation_even_in_same_clock_tick(self):
        for newer in [
                {"x-codex-primary-used-percent": "54",
                 "x-codex-primary-window-minutes": "10080"},
                {"x-codex-primary-reset-at": "1704074401"}]:
            with self.subTest(newer=newer), mock.patch.object(
                    response_headers.time, "time_ns", return_value=1704067200 * 10**9):
                observer = self.codex_observer()
                observer(200, {
                    "x-codex-primary-used-percent": "54",
                    "x-codex-primary-window-minutes": "10080",
                    "x-codex-primary-reset-at": "1704074400",
                })
                observer(200, newer)
                with mock.patch.object(response_headers.time, "time", return_value=1704067200):
                    text = response_headers.render(self.store.snapshot())
                self.assertIn("window: 7 days; reset time unknown", text)
                self.assertNotIn("resets at", text)

    async def test_bad_reset_headers_leave_quota_readable(self):
        for reset in ["", "NaN", "inf", "tomorrow", "1.5", "-1", "0",
                      "9" * 100, "1704074400000", "1704074400\x1b[2J"]:
            with self.subTest(reset=reset):
                self.codex_observer()(200, {
                    "x-codex-primary-used-percent": "54",
                    "x-codex-primary-window-minutes": "10080",
                    "x-codex-primary-reset-at": reset,
                })
                text = response_headers.render(self.store.snapshot())
                self.assertIn(
                    "54% used, 46% remaining; window: 7 days; reset time unknown", text)
                self.assertNotIn("resets at", text)
                self.assertNotIn("\x1b", text)

    async def test_secrets_redacted_in_memory_disk_and_loaded_snapshots(self):
        # Independent protocol examples: removing a name from the production
        # filter must not also remove that name from this test's inputs.
        headers = dict.fromkeys([
            "Set-Cookie", "COOKIE", "Authorization", "Proxy-Authorization",
            "X-API-Key", "X-Auth-Token", "X-Access-Token", "X-Refresh-Token",
            "X-Session-Token", "X-Codex-Turn-State",
        ], "do-not-store")
        headers["x-ratelimit-remaining-tokens"] = "42"
        self.observe(headers=headers)
        document = self.store.snapshot()
        self.assertNotIn("do-not-store", json.dumps(document))
        values = document["endpoints"][0]["headers"]
        self.assertEqual(values["set-cookie"]["value"], "[redacted]")
        self.assertEqual(values["x-ratelimit-remaining-tokens"]["value"], "42")
        await self.store.save()
        self.assertNotIn("do-not-store", Path(self.path).read_text())
        # Saved data is untrusted too: inspection must apply the same filter.
        values["set-cookie"]["value"] = "secret-from-file"
        Path(self.path).write_text(json.dumps(document))
        with contextlib.closing(response_headers.Store(self.path)) as reopened:
            loaded = reopened.snapshot()
        self.assertNotIn("secret-from-file", json.dumps(loaded))

    async def test_snapshot_filters_endpoint_and_credential_including_anonymous(self):
        endpoint = "https://example.com/chat"
        self.observe(credential="env:KEY")
        self.observe(credential="env:OTHER")
        self.observe(credential=None)
        await self.store.save()
        live = response_headers.Store(self.path)
        self.addCleanup(live.close)
        live.observer("https://other.example/chat", "env:KEY", "model")(
            200, {"remaining": "1"})
        live.observer(endpoint, "env:KEY", "model")(
            200, {"new-header": "live"})
        for credential in ["env:KEY", "env:OTHER", None]:
            with self.subTest(credential=credential):
                entries = live.snapshot(
                    "https://EXAMPLE.com:443/chat", credential=credential)["endpoints"]
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0]["credential"], credential)
                self.assertEqual(entries[0]["endpoint"], endpoint)
                if credential == "env:KEY":
                    self.assertIn("new-header", entries[0]["headers"])
        self.assertEqual(live.snapshot(endpoint, credential="env:MISSING")["endpoints"], [])
        self.assertEqual(len(live.snapshot()["endpoints"]), 4)
        self.assertEqual(len(live.snapshot(endpoint)["endpoints"]), 3)

    async def test_identity_url_and_reference_not_provider_or_model(self):
        self.observe()
        self.observe(provider="new-label", model="two")
        self.observe(credential="env:OTHER")
        self.store.observer("https://example.com/messages", "env:KEY", "one")(
            200, {"x-remaining": "20"})
        self.assertEqual(len(self.store.snapshot()["endpoints"]), 3)
        self.assertEqual(response_headers.sanitized_endpoint(
            "HTTPS://user:password@EXAMPLE.com:443/chat?key=secret#fragment"),
            "https://example.com/chat")
        self.assertNotEqual(response_headers.sanitized_endpoint(
            "http://example.com/chat"), "https://example.com/chat")

    async def test_older_runtime_exiting_last_cannot_overwrite_newer(self):
        other = response_headers.Store(self.path)
        self.addCleanup(other.close)
        with mock.patch.object(response_headers.time, "time_ns", return_value=1):
            self.observe(headers={"x-remaining": "10", "x-reset": "soon"})
        with mock.patch.object(response_headers.time, "time_ns", return_value=2):
            self.observe(other, {"x-remaining": "9"})
        await other.save()
        await self.store.save()
        with contextlib.closing(response_headers.Store(self.path)) as reopened:
            entry = reopened.snapshot()["endpoints"][0]
        self.assertEqual(entry["headers"]["x-remaining"]["value"], "9")
        self.assertEqual(entry["headers"]["x-reset"]["value"], "soon")
        if os.name == "posix":
            # Windows has no POSIX permission bits; privacy there is the
            # directory DACL, and the synthesised mode is 0o666.
            self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        before = os.stat(self.path)
        await self.store.save()
        after = os.stat(self.path)
        # A clean save may read, but must not rewrite or replace the file.
        self.assertEqual((after.st_ino, after.st_mtime_ns),
                         (before.st_ino, before.st_mtime_ns))

    def test_try_save_merges_without_a_running_event_loop(self):
        with self.assertRaises(RuntimeError):
            asyncio.get_running_loop()
        with mock.patch.object(response_headers.time, "time_ns", return_value=1):
            self.observe(headers={"x-remaining": "10", "x-reset": "soon"})
        with contextlib.closing(response_headers.Store(self.path)) as other:
            with mock.patch.object(response_headers.time, "time_ns", return_value=2):
                self.observe(other, {"x-remaining": "9"})
            self.assertTrue(other.try_save())
        self.assertTrue(self.store.try_save())
        self.assertFalse(self.store.dirty)
        with contextlib.closing(response_headers.Store(self.path)) as saved:
            headers = saved.snapshot()["endpoints"][0]["headers"]
            self.assertEqual(headers["x-remaining"]["value"], "9")
            self.assertEqual(headers["x-reset"]["value"], "soon")
        before = Path(self.path).read_bytes()
        with mock.patch.object(response_headers.private_files, "open_lock_file_at") as lock:
            self.assertTrue(self.store.try_save())
            lock.assert_not_called()
        self.assertEqual(Path(self.path).read_bytes(), before)

    def test_try_save_busy_does_not_read_or_change_the_snapshot(self):
        self.observe()
        self.assertTrue(self.store.try_save())
        original = Path(self.path).read_bytes()
        self.observe(headers={"x-remaining": "5"})
        # A separately opened handle represents an independent runtime. The
        # reader/merge must not run until this writer's real lock is released.
        lock_fd = response_headers.private_files.open_lock_file_at(
            self.store._directory(), "headers.json.lock", 0o600)
        try:
            file_locks.try_lock_exclusive(lock_fd)
            with mock.patch.object(response_headers, "_read",
                                   side_effect=AssertionError("read without lock")):
                self.assertFalse(self.store.try_save())
            self.assertTrue(self.store.dirty)
            self.assertEqual(Path(self.path).read_bytes(), original)
        finally:
            response_headers.private_files.close(lock_fd)
        self.assertTrue(self.store.try_save())
        self.assertFalse(self.store.dirty)
        with contextlib.closing(response_headers.Store(self.path)) as saved:
            self.assertEqual(saved.snapshot()["endpoints"][0]["headers"]
                             ["x-remaining"]["value"], "5")

    async def test_async_save_retries_contention_and_includes_new_observations(self):
        with mock.patch.object(response_headers.time, "time_ns", return_value=1):
            self.observe(headers={"x-remaining": "10", "x-reset": "soon"})

        async def observe_while_waiting(delay):
            self.assertEqual(delay, 0.01)
            with mock.patch.object(response_headers.time, "time_ns", return_value=2):
                self.observe(headers={"x-remaining": "5"})

        with mock.patch.object(file_locks, "try_lock_exclusive",
                               side_effect=[BlockingIOError, None]) as lock, \
                mock.patch.object(asyncio, "sleep", side_effect=observe_while_waiting) as sleep:
            await self.store.save()
            self.assertEqual(lock.call_count, 2)
            sleep.assert_awaited_once_with(0.01)
        self.assertFalse(self.store.dirty)
        with contextlib.closing(response_headers.Store(self.path)) as saved:
            headers = saved.snapshot()["endpoints"][0]["headers"]
            self.assertEqual(headers["x-remaining"]["value"], "5")
            self.assertEqual(headers["x-reset"]["value"], "soon")

    async def test_async_save_does_not_retry_publication_errors(self):
        self.observe()
        with mock.patch.object(response_headers.private_files, "write",
                               side_effect=BlockingIOError("write unavailable")), \
                mock.patch.object(asyncio, "sleep") as sleep:
            with self.assertRaisesRegex(BlockingIOError, "write unavailable"):
                await self.store.save()
            sleep.assert_not_called()
        self.assertTrue(self.store.dirty)
        self.assertFalse(os.path.exists(self.path))
        self.assertTrue(self.store.try_save())

    async def test_busy_snapshot_survives_failed_save_and_retry_keeps_observations(self):
        self.observe()
        await self.store.save()
        original = Path(self.path).read_bytes()
        self.observe(headers={"x-remaining": "5"})
        with mock.patch.object(file_locks, 'try_lock_exclusive',
                               side_effect=BlockingIOError):
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                await self.store.save_on_exit()
            self.assertIn("Could not save", errors.getvalue())
            self.assertEqual(Path(self.path).read_bytes(), original)
        await self.store.save()
        with contextlib.closing(response_headers.Store(self.path)) as reopened:
            saved = reopened.snapshot()["endpoints"][0]
        self.assertEqual(saved["headers"]["x-remaining"]["value"], "5")

    async def test_bad_snapshot_is_preserved_until_removed_then_observations_can_retry(self):
        self.observe()
        with open(self.path, 'w', encoding='utf-8') as stream:
            stream.write('not JSON')
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            await self.store.save_on_exit()
        self.assertIn('Could not save', errors.getvalue())
        with open(self.path, encoding='utf-8') as stream:
            self.assertEqual(stream.read(), 'not JSON')
        self.assertTrue(self.store.dirty)
        os.unlink(self.path)
        await self.store.save()
        self.assertFalse(self.store.dirty)
        with contextlib.closing(response_headers.Store(self.path)) as reopened:
            saved = reopened.snapshot()['endpoints'][0]
        self.assertEqual(saved['headers']['x-remaining']['value'], '10')

    def test_failed_replace_preserves_existing_file_and_dirty_observations_for_retry(self):
        self.observe()
        self.assertTrue(self.store.try_save())
        with open(self.path, 'rb') as stream:
            original = stream.read()
        self.observe(headers={'x-remaining': '5'})
        with mock.patch.object(private_files, 'replace_at', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                self.store.try_save()
        with open(self.path, 'rb') as stream:
            self.assertEqual(stream.read(), original)
        self.assertTrue(self.store.dirty)
        self.assertFalse(any(name.startswith('.headers.json.') for name in os.listdir(self.directory)))
        # Immediate retry proves the failed publication released its real lock.
        self.assertTrue(self.store.try_save())
        self.assertFalse(self.store.dirty)
        with contextlib.closing(response_headers.Store(self.path)) as reopened:
            saved = reopened.snapshot()['endpoints'][0]
        self.assertEqual(saved['headers']['x-remaining']['value'], '5')

    def test_directory_sync_failure_leaves_published_observations_dirty_for_retry(self):
        self.observe()
        self.assertTrue(self.store.try_save())
        self.observe(headers={'x-remaining': '5'})
        sync = private_files.fsync

        def failed_directory_sync(handle):
            if handle == self.store.directory_handle:
                raise OSError('directory sync failed')
            sync(handle)

        with mock.patch.object(private_files, 'fsync', side_effect=failed_directory_sync):
            with self.assertRaisesRegex(OSError, 'directory sync failed'):
                self.store.try_save()
        self.assertTrue(self.store.dirty)
        with open(self.path, encoding='utf-8') as stream:
            self.assertEqual(json.load(stream)['endpoints'][0]['headers']['x-remaining']['value'], '5')
        self.assertFalse(any(name.startswith('.headers.json.') for name in os.listdir(self.directory)))
        self.assertTrue(self.store.try_save())
        self.assertFalse(self.store.dirty)
        with contextlib.closing(response_headers.Store(self.path)) as reopened:
            saved = reopened.snapshot()['endpoints'][0]
        self.assertEqual(saved['headers']['x-remaining']['value'], '5')

    async def test_inspection_escapes_and_never_flushes(self):
        self.observe(headers={"x-odd": "\x1b[2J\nsecret"})
        rendered = response_headers.render(self.store.snapshot())
        self.assertNotIn("\x1b", rendered)
        self.assertIn("\\x1b", rendered)
        self.assertFalse(os.path.exists(self.path))
        await self.store.save()
        with mock.patch.object(response_headers, '_snapshot_location',
                               return_value=(self.directory, 'headers.json')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(response_headers.main(["--json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["endpoints"][0]
                         ["headers"]["x-odd"]["value"], "\x1b[2J\nsecret")


class ResponseCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.workspace = os.path.join(self.directory, "workspace")
        os.makedirs(self.workspace)
        self.enterContext(isolated_response_headers())
        self.session = loki.current_session()

    async def start_child(self, *arguments, **options):
        child = await asyncio.create_subprocess_exec(*arguments, **options)
        # Registered after directory cleanup, hence run before it. Cancelling
        # readers or failing an assertion does not retire the child.
        self.addAsyncCleanup(self.stop_child, child)
        return child

    async def stop_child(self, child):
        if child.returncode is None:
            if child.stdin is not None:
                child.stdin.close()  # The real ACP front joins its worker on EOF.
            else:
                with contextlib.suppress(ProcessLookupError):
                    child.kill()  # Failure backstop, not a timed execution policy.
        await child.communicate()

    async def test_child_cleanup_reaps_after_timeout_cancellation_or_assertion(self):
        for outcome in ['timeout', 'cancel', 'assertion']:
            with self.subTest(outcome=outcome):
                expected_error = (
                    self.assertRaisesRegex(AssertionError, 'injected readiness assertion')
                    if outcome == 'assertion' else contextlib.nullcontext())
                with expected_error:
                    # Write the readiness byte by byte through the buffer, so
                    # the child interpreter's stdout text mode cannot turn it
                    # into a different line ending than the test expects.
                    child = await self.start_child(
                        sys.executable, '-c',
                        'import sys, threading; '
                        "sys.stdout.buffer.write(b'ready\\n'); "
                        'sys.stdout.buffer.flush(); threading.Event().wait()',
                        cwd=self.workspace, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE)
                    try:
                        line = await child.stdout.readline()
                        self.assertEqual(line, b'ready\n')
                        self.assertIsNone(child.returncode)
                        if outcome == 'assertion':
                            self.fail('injected readiness assertion')
                        communication = asyncio.create_task(child.communicate())
                        if outcome == 'cancel':
                            await asyncio.sleep(0)
                            communication.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await communication
                        else:
                            with self.assertRaises(asyncio.TimeoutError):
                                await asyncio.wait_for(communication, 0.01)
                    finally:
                        # Registered test cleanup runs too late for this
                        # method's rmdir, especially after a failed subtest.
                        await self.stop_child(child)
                self.assertIsNotNone(child.returncode)
        # On Windows a child still using this cwd would prevent removal.
        os.rmdir(self.workspace)

    async def test_child_cleanup_signals_its_owner_and_joins_without_a_watchdog(self):
        for front in [False, True]:
            with self.subTest(front=front):
                child = mock.Mock(returncode=None, stdin=mock.Mock() if front else None)

                async def communicate():
                    if front:
                        child.stdin.close.assert_called_once_with()
                        child.kill.assert_not_called()
                    else:
                        child.kill.assert_called_once_with()
                    child.returncode = 1
                    return b'', b''

                child.communicate = mock.AsyncMock(side_effect=communicate)
                await self.stop_child(child)
                child.terminate.assert_not_called()
                self.assertEqual(child.communicate.await_count, 1)
                self.assertEqual(child.returncode, 1)

    async def server(self, responses):
        cleanup = contextlib.AsyncExitStack()

        async def serve(reader, writer):
            try:
                request = await reader.readuntil(b"\r\n\r\n")
                for line in request.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        await reader.readexactly(int(line.split(b":")[1]))
                writer.write(responses.pop(0))
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        def accept(reader, writer):
            task = asyncio.create_task(serve(reader, writer))
            cleanup.push_async_callback(close_fixture_connection, task, writer)
            return task

        server = await asyncio.start_server(accept, "127.0.0.1", 0, start_serving=False)
        self.addAsyncCleanup(self.close_server, server, cleanup)
        await server.start_serving()
        port = server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        loki.apply_runtime_config(connections.make_runtime_config(
            url, protocols.OPENAI_CHAT, model="test-model"))
        return url

    async def close_server(self, server, cleanup):
        server.close()
        try:
            await cleanup.aclose()
        finally:
            await server.wait_closed()

    async def test_server_close_reports_completed_handler_failure(self):
        server = mock.Mock(start_serving=mock.AsyncMock(), wait_closed=mock.AsyncMock())
        server.sockets = [mock.Mock()]
        server.sockets[0].getsockname.return_value = ('127.0.0.1', 4321)
        reader = mock.Mock(readuntil=mock.AsyncMock(return_value=b'GET / HTTP/1.1\r\n\r\n'))
        failure = RuntimeError('header server close failed')
        writer = mock.Mock(drain=mock.AsyncMock(), wait_closed=mock.AsyncMock(side_effect=failure))
        cleanups = []
        with mock.patch.object(asyncio, 'start_server', new=mock.AsyncMock(return_value=server)) as start, \
                mock.patch.object(self, 'addAsyncCleanup', side_effect=lambda *args: cleanups.append(args)):
            try:
                await self.server([b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n'])
                operation = start.await_args.args[0](reader, writer)
                task = asyncio.create_task(operation) if asyncio.iscoroutine(operation) else operation
                await asyncio.gather(task, return_exceptions=True)
                close, *arguments = cleanups.pop()
                with self.assertRaises(RuntimeError) as caught:
                    await close(*arguments)
                self.assertIs(caught.exception, failure)
            finally:
                for close, *arguments in reversed(cleanups):
                    await close(*arguments)
        writer.close.assert_called()
        server.close.assert_called_once()
        server.wait_closed.assert_awaited_once()

    async def test_buffered_error_and_stream_body_failure_capture_headers(self):
        for stream in [False, True]:
            with self.subTest(stream=stream):
                response = (b"HTTP/1.1 429 Too Many Requests\r\n"
                            b"X-Remaining: 0\r\nContent-Length: 2\r\n\r\n{}")
                if stream:
                    response = (b"HTTP/1.1 200 OK\r\nX-Remaining: 1\r\n"
                                b"Content-Length: 20\r\n\r\n")
                url = await self.server([response])
                with mock.patch.object(model_requests, 'HTTP_RETRY_MAX_ATTEMPTS_LLM', 1):
                    with self.assertRaises(OSError if stream else model_requests.ApiError):
                        if stream:
                            await loki.async_chat_stream_request(url, {})
                        else:
                            await loki.async_provider_request(
                                "POST", url, {}, request_headers={
                                    "X-Request-Only": "not-a-response-header"})
                entry = self.session.response_headers.snapshot(url)["endpoints"][0]
                self.assertEqual(entry["headers"]["x-remaining"]["value"],
                                 "1" if stream else "0")
                self.assertEqual(entry["latest"]["status"], 200 if stream else 429)
                self.assertNotIn("x-request-only", entry["headers"])
                self.assertFalse(os.path.exists(self.session.response_headers.path))

    async def test_real_headless_exit_flushes_and_status_reads_without_request(self):
        body = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()
        url = await self.server([
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"X-Remaining: 12\r\nSet-Cookie: secret-cookie\r\nContent-Length: "
            + str(len(body)).encode() + b"\r\n\r\n" + body])
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("LOKI_")
                       and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
        environment.update({
            "XDG_STATE_HOME": self.directory,
            "XDG_CONFIG_HOME": os.path.join(self.directory, "config"),
            "LOKI_API_BASE": url, "LOKI_MODEL": "test-model",
            "LOKI_API_KEY": "test-key-not-for-recording",
        })
        configure_container(environment, self.workspace)
        child = await self.start_child(
            entrypoint("loki"), "--headless", "--prompt", "Say ok",
            cwd=self.workspace, env=environment,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output, errors = await child.communicate()
        self.assertEqual(child.returncode, 0, errors.decode())
        snapshot = Path(self.directory) / "loki" / "response-headers.json"
        saved = json.loads(snapshot.read_text())
        self.assertEqual(saved["endpoints"][0]["headers"]["x-remaining"]["value"], "12")
        self.assertNotIn("test-key-not-for-recording", snapshot.read_text())
        self.assertNotIn("secret-cookie", snapshot.read_text())
        child = await self.start_child(
            entrypoint("loki"), "status", "--json",
            cwd=self.workspace, env=environment,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output, errors = await child.communicate()
        self.assertEqual(child.returncode, 0, errors.decode())
        self.assertEqual(json.loads(output), saved)

    async def test_real_acp_worker_exit_flushes(self):
        body = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()
        url = await self.server([
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"X-Remaining: 11\r\nContent-Length: "
            + str(len(body)).encode() + b"\r\n\r\n" + body])
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("LOKI_")
                       and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
        environment.update({
            "XDG_STATE_HOME": self.directory,
            "XDG_CONFIG_HOME": os.path.join(self.directory, "config"),
            "LOKI_API_BASE": url, "LOKI_MODEL": "test-model",
            "LOKI_API_KEY": "test-key",
        })
        configure_container(environment, self.workspace)
        child = await self.start_child(
            entrypoint("loki-acp"), cwd=self.workspace, env=environment,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

        async def request(number, method, params):
            child.stdin.write((json.dumps({"jsonrpc": "2.0", "id": number,
                                          "method": method, "params": params})
                               + "\n").encode())
            await child.stdin.drain()
            while True:
                line = await child.stdout.readline()
                self.assertTrue(line, "ACP front exited before replying")
                reply = json.loads(line)
                if reply.get("id") == number:
                    self.assertNotIn("error", reply, reply)
                    return reply["result"]

        try:
            await request(1, "initialize", {"protocolVersion": 1})
            opened = await request(2, "session/new", {
                "cwd": self.workspace, "mcpServers": []})
            await request(3, "session/prompt", {
                "sessionId": opened["sessionId"],
                "prompt": [{"type": "text", "text": "Say ok"}]})
            child.stdin.close()
            output, errors = await child.communicate()
            self.assertEqual(child.returncode, 0, errors.decode())
        finally:
            await self.stop_child(child)
        snapshot = Path(self.directory) / "loki" / "response-headers.json"
        saved = json.loads(snapshot.read_text())
        self.assertEqual(saved["endpoints"][0]["headers"]["x-remaining"]["value"], "11")

    async def test_non_chat_get_not_captured_and_identity_is_snapshotted(self):
        url = await self.server([])

        async def request(method, request_url, **options):
            if method == "POST":
                loki.apply_runtime_config(connections.make_runtime_config(
                    "https://other.example/chat", protocols.OPENAI_CHAT,
                    model="other-model"))
            if options.get("on_response_headers") is not None:
                options["on_response_headers"](200, {"x-remaining": "7"})
            return http_client.HttpResponse(
                request_url, 200, "OK", {}, b"{}")

        with mock.patch.object(http_client, "async_http_request", new=request):
            await loki.async_provider_request("GET", url + "/models")
            self.assertEqual(self.session.response_headers.snapshot()["endpoints"], [])
            await loki.async_provider_request("POST", url, {})
        entry = self.session.response_headers.snapshot()["endpoints"][0]
        self.assertEqual(entry["endpoint"], url)
        self.assertEqual(entry["latest"]["model"], "test-model")
        before = self.session.response_headers.snapshot()
        self.session.replace_transcript([], [], [], {},
                                        os.path.join(self.directory, "chat.json"))
        self.assertEqual(self.session.response_headers.snapshot(), before)
