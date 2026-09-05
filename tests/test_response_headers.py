import asyncio
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from loki_agent import http_client, loki, protocols, response_headers
from loki_agent.sessions import Session


class ResponseHeadersTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "headers.json")
        self.store = response_headers.Store(self.path)

    def observe(self, store=None, headers=None, status=200, model="one",
                credential="env:KEY", provider=None):
        (store or self.store).observer(
            "https://example.com/chat", credential, model, provider)(
                status, headers or {"x-remaining": "10"})

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
        self.assertIn("Main subscription bucket: 56% used, 44% remaining - 7 days", text)
        self.assertIn("GPT-5.3-Codex-Spark: 0% used, 100% remaining - 5 hours", text)
        self.assertIn("GPT-5.3-Codex-Spark: 0% used, 100% remaining - 7 days", text)
        self.assertEqual(text.count("% remaining"), 3)
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
        self.assertIn("12.5% used, 87.5% remaining - 1 hour", text)
        self.assertNotIn("\x1b", text)
        self.assertIn("Model\\x1b[2J\\nspoof", text)

    async def test_codex_summary_marks_retained_and_omits_mixed_observations(self):
        with mock.patch.object(response_headers.time, "time_ns", return_value=1):
            self.codex_observer()(200, {
                "x-codex-primary-used-percent": "56",
                "x-codex-primary-window-minutes": "10080",
            })
        with mock.patch.object(response_headers.time, "time_ns", return_value=2):
            self.codex_observer()(200, {"date": "later"})
        self.assertIn("7 days [retained observation]", response_headers.render(
            self.store.snapshot()))
        with mock.patch.object(response_headers.time, "time_ns", return_value=3):
            self.codex_observer()(200, {"x-codex-primary-used-percent": "57"})
        text = response_headers.render(self.store.snapshot())
        self.assertNotIn("Subscription quota", text)
        self.assertIn("'x-codex-primary-used-percent': '57'", text)

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
        loaded = response_headers.Store(self.path).snapshot()
        self.assertNotIn("secret-from-file", json.dumps(loaded))

    async def test_snapshot_filters_endpoint_and_credential_including_anonymous(self):
        endpoint = "https://example.com/chat"
        self.observe(credential="env:KEY")
        self.observe(credential="env:OTHER")
        self.observe(credential=None)
        await self.store.save()
        live = response_headers.Store(self.path)
        live.observer("https://other.example/chat", "env:KEY", "model")(
            200, {"remaining": "1"})
        live.observer(endpoint, "env:KEY", "model")(
            200, {"new-header": "live"})
        for credential in ("env:KEY", "env:OTHER", None):
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
        with mock.patch.object(response_headers.time, "time_ns", return_value=1):
            self.observe(headers={"x-remaining": "10", "x-reset": "soon"})
        with mock.patch.object(response_headers.time, "time_ns", return_value=2):
            self.observe(other, {"x-remaining": "9"})
        await other.save()
        await self.store.save()
        entry = response_headers.Store(self.path).snapshot()["endpoints"][0]
        self.assertEqual(entry["headers"]["x-remaining"]["value"], "9")
        self.assertEqual(entry["headers"]["x-reset"]["value"], "soon")
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        before = os.stat(self.path)
        await self.store.save()
        after = os.stat(self.path)
        # A clean save may read, but must not rewrite or replace the file.
        self.assertEqual((after.st_ino, after.st_mtime_ns),
                         (before.st_ino, before.st_mtime_ns))

    async def test_busy_snapshot_survives_failed_save_and_retry_keeps_observations(self):
        self.observe()
        await self.store.save()
        original = Path(self.path).read_bytes()
        self.observe(headers={"x-remaining": "5"})
        with open(self.path + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                await self.store.save_on_exit()
            self.assertIn("Could not save", errors.getvalue())
            self.assertEqual(Path(self.path).read_bytes(), original)
        await self.store.save()
        saved = response_headers.Store(self.path).snapshot()["endpoints"][0]
        self.assertEqual(saved["headers"]["x-remaining"]["value"], "5")

    async def test_bad_snapshot_and_failed_write_preserve_existing_file(self):
        self.observe()
        Path(self.path).write_text("not JSON")
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            await self.store.save_on_exit()
        self.assertIn("Could not save", errors.getvalue())
        self.assertEqual(Path(self.path).read_text(), "not JSON")
        os.unlink(self.path)
        await self.store.save()
        original = Path(self.path).read_bytes()
        self.observe(headers={"x-remaining": "5"})
        with mock.patch.object(response_headers.os, "replace",
                               side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                await self.store.save()
        self.assertEqual(Path(self.path).read_bytes(), original)
        await self.store.save()
        saved = response_headers.Store(self.path).snapshot()["endpoints"][0]
        self.assertEqual(saved["headers"]["x-remaining"]["value"], "5")

    async def test_inspection_escapes_and_never_flushes(self):
        self.observe(headers={"x-odd": "\x1b[2J\nsecret"})
        rendered = response_headers.render(self.store.snapshot())
        self.assertNotIn("\x1b", rendered)
        self.assertIn("\\x1b", rendered)
        self.assertFalse(os.path.exists(self.path))
        await self.store.save()
        with mock.patch.object(response_headers, "snapshot_path",
                               return_value=self.path):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(response_headers.main(["--json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["endpoints"][0]
                         ["headers"]["x-odd"]["value"], "\x1b[2J\nsecret")


class ResponseCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.previous = loki._DEFAULT_SESSION
        self.session = Session(response_headers=response_headers.Store(
            os.path.join(self.directory.name, "headers.json")))
        loki._DEFAULT_SESSION = self.session
        self.addCleanup(setattr, loki, "_DEFAULT_SESSION", self.previous)

    async def server(self, responses):
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
        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        self.addAsyncCleanup(self.close_server, server)
        port = server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        loki.apply_runtime_config(loki.make_runtime_config(
            url, protocols.OPENAI_CHAT, model="test-model"))
        return url

    async def close_server(self, server):
        server.close()
        await server.wait_closed()

    async def test_buffered_error_and_stream_body_failure_capture_headers(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = (b"HTTP/1.1 429 Too Many Requests\r\n"
                            b"X-Remaining: 0\r\nContent-Length: 2\r\n\r\n{}")
                if stream:
                    response = (b"HTTP/1.1 200 OK\r\nX-Remaining: 1\r\n"
                                b"Content-Length: 20\r\n\r\n")
                url = await self.server([response])
                with mock.patch.object(loki, "HTTP_RETRY_MAX_ATTEMPTS_LLM", 1):
                    with self.assertRaises(OSError if stream else loki.ApiError):
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
        root = Path(__file__).resolve().parents[1]
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("LOKI_")
                       and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
        environment.update({
            "XDG_STATE_HOME": self.directory.name,
            "XDG_CONFIG_HOME": os.path.join(self.directory.name, "config"),
            "LOKI_API_BASE": url, "LOKI_MODEL": "test-model",
            "LOKI_API_KEY": "test-key-not-for-recording",
        })
        child = await asyncio.create_subprocess_exec(
            str(root / "loki.py"), "--headless", "--prompt", "Say ok",
            cwd=self.directory.name, env=environment,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output, errors = await asyncio.wait_for(child.communicate(), 20)
        self.assertEqual(child.returncode, 0, errors.decode())
        snapshot = Path(self.directory.name) / "loki" / "response-headers.json"
        saved = json.loads(snapshot.read_text())
        self.assertEqual(saved["endpoints"][0]["headers"]["x-remaining"]["value"], "12")
        self.assertNotIn("test-key-not-for-recording", snapshot.read_text())
        self.assertNotIn("secret-cookie", snapshot.read_text())
        child = await asyncio.create_subprocess_exec(
            str(root / "loki.py"), "status", "--json",
            cwd=self.directory.name, env=environment,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output, errors = await asyncio.wait_for(child.communicate(), 10)
        self.assertEqual(child.returncode, 0, errors.decode())
        self.assertEqual(json.loads(output), saved)

    async def test_real_acp_worker_exit_flushes(self):
        body = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()
        url = await self.server([
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"X-Remaining: 11\r\nContent-Length: "
            + str(len(body)).encode() + b"\r\n\r\n" + body])
        root = Path(__file__).resolve().parents[1]
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("LOKI_")
                       and not key.endswith(("_KEY", "_TOKEN", "_PAT"))}
        environment.update({
            "XDG_STATE_HOME": self.directory.name,
            "XDG_CONFIG_HOME": os.path.join(self.directory.name, "config"),
            "LOKI_API_BASE": url, "LOKI_MODEL": "test-model",
            "LOKI_API_KEY": "test-key",
        })
        child = await asyncio.create_subprocess_exec(
            str(root / "loki-acp"), cwd=self.directory.name, env=environment,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

        async def request(number, method, params):
            child.stdin.write((json.dumps({"jsonrpc": "2.0", "id": number,
                                          "method": method, "params": params})
                               + "\n").encode())
            await child.stdin.drain()
            while True:
                line = await asyncio.wait_for(child.stdout.readline(), 15)
                self.assertTrue(line, "ACP front exited before replying")
                reply = json.loads(line)
                if reply.get("id") == number:
                    self.assertNotIn("error", reply, reply)
                    return reply["result"]

        try:
            await request(1, "initialize", {"protocolVersion": 1})
            opened = await request(2, "session/new", {
                "cwd": self.directory.name, "mcpServers": []})
            await request(3, "session/prompt", {
                "sessionId": opened["sessionId"],
                "prompt": [{"type": "text", "text": "Say ok"}]})
            child.stdin.close()
            output, errors = await asyncio.wait_for(child.communicate(), 10)
            self.assertEqual(child.returncode, 0, errors.decode())
        finally:
            if child.returncode is None:
                child.kill()
                await child.communicate()
        snapshot = Path(self.directory.name) / "loki" / "response-headers.json"
        saved = json.loads(snapshot.read_text())
        self.assertEqual(saved["endpoints"][0]["headers"]["x-remaining"]["value"], "11")

    async def test_non_chat_get_not_captured_and_identity_is_snapshotted(self):
        url = await self.server([])

        async def request(method, request_url, **options):
            if method == "POST":
                loki.apply_runtime_config(loki.make_runtime_config(
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
                                        os.path.join(self.directory.name, "chat.json"))
        self.assertEqual(self.session.response_headers.snapshot(), before)
