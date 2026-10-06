import asyncio
import contextlib
import sys
import time
import unittest
from unittest import mock

from loki_agent import http_client


@contextlib.asynccontextmanager
async def _running_request(awaitable):
    task = asyncio.create_task(awaitable)
    try:
        yield task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), 1)


class FakeWriter:
    def __init__(self):
        self.data = bytearray()
        self.closed = False
        self.wait_closed_called = False

    def write(self, data):
        self.data.extend(data)

    async def drain(self):
        pass

    def close(self):
        self.closed = True

    async def wait_closed(self):
        self.wait_closed_called = True


class FailingWriter(FakeWriter):
    def __init__(self, *, fail_write=False, fail_drain=False):
        super().__init__()
        self.fail_write = fail_write
        self.fail_drain = fail_drain

    def write(self, data):
        if self.fail_write:
            raise BrokenPipeError("write failed")
        super().write(data)

    async def drain(self):
        if self.fail_drain:
            raise BrokenPipeError("drain failed")


class FakeConnector:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.writers = []

    async def open_connection(self, host, port, ssl=None, server_hostname=None):
        self.calls.append({
            "host": host,
            "port": port,
            "ssl": ssl,
            "server_hostname": server_hostname,
        })
        reader = asyncio.StreamReader()
        reader.feed_data(self.responses.pop(0))
        reader.feed_eof()
        writer = FakeWriter()
        self.writers.append(writer)
        return reader, writer


class PatchedOpenConnection:
    def __init__(self, connector, tls_context=None):
        self.connector = connector
        self.tls_context = tls_context if tls_context is not None else object()
        self.old_open_connection = None
        self.old_create_default_context = None

    def __enter__(self):
        self.old_open_connection = http_client.asyncio.open_connection
        self.old_create_default_context = http_client.ssl.create_default_context
        http_client.asyncio.open_connection = self.connector.open_connection
        http_client.ssl.create_default_context = lambda: self.tls_context
        return self

    def __exit__(self, exc_type, exc, tb):
        http_client.asyncio.open_connection = self.old_open_connection
        http_client.ssl.create_default_context = self.old_create_default_context
        return False


class HttpClientRequestTests(unittest.TestCase):
    def test_connect_failure_is_annotated_as_not_sent(self):
        connector = FakeConnector([])

        async def failed_open(*args, **kwargs):
            raise ConnectionRefusedError("offline")

        connector.open_connection = failed_open
        with PatchedOpenConnection(connector):
            with self.assertRaises(
                    http_client.HttpRequestDeliveryError) as raised:
                asyncio.run(http_client.async_http_request(
                    "POST",
                    "https://example.test/token",
                    body=b"{}",
                ))

        self.assertFalse(raised.exception.request_may_have_been_sent)

    def test_drain_failure_is_annotated_as_possibly_sent(self):
        connector = FakeConnector([
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n",
        ])

        async def open_with_failing_drain(*args, **kwargs):
            reader = asyncio.StreamReader()
            reader.feed_data(connector.responses.pop(0))
            reader.feed_eof()
            writer = FailingWriter(fail_drain=True)
            connector.writers.append(writer)
            return reader, writer

        connector.open_connection = open_with_failing_drain
        with PatchedOpenConnection(connector):
            with self.assertRaises(
                    http_client.HttpRequestDeliveryError) as raised:
                asyncio.run(http_client.async_http_request(
                    "POST",
                    "https://example.test/token",
                    body=b"{}",
                ))

        self.assertTrue(raised.exception.request_may_have_been_sent)

    def test_write_failure_is_conservatively_possibly_sent(self):
        connector = FakeConnector([
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n",
        ])

        async def open_with_failing_write(*args, **kwargs):
            reader = asyncio.StreamReader()
            writer = FailingWriter(fail_write=True)
            connector.writers.append(writer)
            return reader, writer

        connector.open_connection = open_with_failing_write
        with PatchedOpenConnection(connector):
            with self.assertRaises(
                    http_client.HttpRequestDeliveryError) as raised:
                asyncio.run(http_client.async_http_request(
                    "POST",
                    "https://example.test/token",
                    body=b"{}",
                ))

        self.assertTrue(raised.exception.request_may_have_been_sent)

    def test_task_cancellation_preserves_delivery_state(self):
        async def scenario(phase):
            connector = FakeConnector([])
            entered, interrupted = asyncio.Event(), asyncio.Event()
            delivery_states = []
            operations = []

            async def block():
                operations.append(asyncio.current_task())
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    interrupted.set()

            async def open_connection(host, port, ssl=None, server_hostname=None):
                connector.calls.append((host, port, ssl, server_hostname))
                if phase == "connect":
                    await block()
                reader, writer = asyncio.StreamReader(), FakeWriter()
                connector.writers.append(writer)
                writer.drain = block
                return reader, writer

            async def request():
                try:
                    return await http_client.async_http_request(
                        "POST", "https://example.test/token", body=b"{}",
                        timeout=20, retry_max_attempts=3,
                        retry_base_delay_s=0, retry_max_jitter_s=0)
                except asyncio.CancelledError as error:
                    # Observe the production annotation before the outer Task
                    # boundary can discard it on historical CPython 3.10.
                    delivery_states.append(error.request_may_have_been_sent)
                    raise

            connector.open_connection = open_connection
            tls_context = object()
            with PatchedOpenConnection(connector, tls_context):
                async with _running_request(request()) as task:
                    await asyncio.wait_for(entered.wait(), 1)
                    self.assertFalse(task.done())
                    self.assertFalse(interrupted.is_set())
                    self.assertEqual(len(connector.writers), phase == "drain")
                    if connector.writers:
                        self.assertEqual(bytes(connector.writers[0].data), (
                            "POST /token HTTP/1.1\r\nHost: example.test\r\nConnection: close\r\n"
                            f"User-Agent: {http_client.APPLICATION_USER_AGENT}\r\n"
                            "Content-Length: 2\r\n\r\n{}").encode())
                        self.assertFalse(connector.writers[0].closed)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError) as raised:
                        await asyncio.wait_for(task, 1)
                    expected = phase == "drain"
                    self.assertEqual(delivery_states, [expected])
                    legacy_loss = (
                        sys.implementation.name == "cpython"
                        and sys.version_info[:2] == (3, 10)
                        and not hasattr(raised.exception, "request_may_have_been_sent"))
                    if not legacy_loss:
                        self.assertIs(raised.exception.request_may_have_been_sent, expected)
                    # The legacy missing fact is not classified as a successful
                    # Task-boundary annotation proof or defaulted to True here.
                    self.assertTrue(task.done())
                    self.assertTrue(task.cancelled())
                    self.assertTrue(interrupted.is_set())
                    self.assertEqual(len(operations), 1)
                    self.assertTrue(operations[0].done())
                    self.assertTrue(operations[0].cancelled())
                    self.assertEqual(connector.calls, [("example.test", 443, tls_context, "example.test")])
                    for writer in connector.writers:
                        self.assertTrue(writer.closed)
                        self.assertTrue(writer.wait_closed_called)

        for phase in ["connect", "drain"]:
            with self.subTest(phase=phase):
                asyncio.run(scenario(phase))

    def test_buffered_request_can_be_cancelled_during_connect(self):
        connector = FakeConnector([])
        entered = asyncio.Event()

        async def blocked_open(*args, **kwargs):
            entered.set()
            await asyncio.Future()

        connector.open_connection = blocked_open

        async def scenario():
            cancelled = False

            def cancel_check():
                return cancelled

            async def request():
                return await http_client.async_http_request(
                    "POST", "https://example.test/v1/chat/completions",
                    body=b"{}", timeout=30, cancel_check=cancel_check)

            task = asyncio.create_task(request())
            await entered.wait()
            cancelled = True
            with self.assertRaises(http_client.HttpRequestCancelled):
                await asyncio.wait_for(task, timeout=1)

        with PatchedOpenConnection(connector):
            asyncio.run(scenario())

    def test_buffered_request_cancel_interrupts_retry_backoff(self):
        async def scenario():
            connector = FakeConnector([])
            sleeping, interrupted = asyncio.Event(), asyncio.Event()
            sleep_tasks = []
            cancelled = False
            real_sleep = asyncio.sleep

            async def failed_open(host, port, ssl=None, server_hostname=None):
                connector.calls.append((host, port, ssl, server_hostname))
                raise ConnectionResetError("offline")

            async def observed_sleep(delay):
                self.assertEqual(delay, 10)
                sleep_tasks.append(asyncio.current_task())
                sleeping.set()
                try:
                    await real_sleep(delay)
                finally:
                    interrupted.set()

            connector.open_connection = failed_open
            tls_context = object()
            with PatchedOpenConnection(connector, tls_context), mock.patch.object(
                    asyncio, "sleep", new=observed_sleep):
                async with _running_request(http_client.async_http_request(
                        "POST", "https://example.test/v1/chat/completions",
                        body=b"{}", timeout=20, retry_max_attempts=3,
                        retry_base_delay_s=10, retry_max_jitter_s=0,
                        cancel_check=lambda: cancelled)) as task:
                    await asyncio.wait_for(sleeping.wait(), 1)
                    self.assertFalse(cancelled)
                    self.assertFalse(task.done())
                    self.assertFalse(interrupted.is_set())
                    self.assertEqual(len(connector.calls), 1)
                    cancelled = True
                    with self.assertRaises(http_client.HttpRequestCancelled):
                        await asyncio.wait_for(task, 1)
                    self.assertTrue(task.done())
                    self.assertTrue(interrupted.is_set())
                    self.assertEqual(len(sleep_tasks), 1)
                    self.assertTrue(sleep_tasks[0].done())
                    self.assertTrue(sleep_tasks[0].cancelled())
                    self.assertEqual(connector.calls, [("example.test", 443, tls_context, "example.test")])
                    self.assertEqual(connector.writers, [])

        asyncio.run(scenario())

    def test_https_request_serializes_headers_body_and_tls_connection(self):
        connector = FakeConnector([
            b"HTTP/1.1 201 Created\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 5\r\n"
            b"\r\n"
            b"hello"
        ])
        tls_context = object()

        with PatchedOpenConnection(connector, tls_context):
            response = asyncio.run(http_client.async_http_request(
                "post",
                "https://api.example.test:8443/v1/messages?trace=1",
                headers_in={"X-Test": "ok", "Authorization": "Bearer token"},
                body=b"{}",
                timeout=5,
                max_bytes=100,
            ))

        self.assertEqual(response.status, 201)
        self.assertEqual(response.reason, "Created")
        self.assertEqual(response.header("Content-Type"), "text/plain")
        self.assertEqual(response.body, b"hello")
        self.assertFalse(response.truncated)
        self.assertEqual(
            connector.calls,
            [{
                "host": "api.example.test",
                "port": 8443,
                "ssl": tls_context,
                "server_hostname": "api.example.test",
            }],
        )
        self.assertTrue(connector.writers[0].closed)
        self.assertTrue(connector.writers[0].wait_closed_called)

        raw_headers, sent_body = bytes(connector.writers[0].data).split(b"\r\n\r\n", 1)
        self.assertEqual(sent_body, b"{}")
        self.assertEqual(raw_headers.split(b"\r\n")[0], b"POST /v1/messages?trace=1 HTTP/1.1")
        self.assertIn(b"Host: api.example.test:8443", raw_headers.split(b"\r\n"))
        self.assertIn(b"Connection: close", raw_headers.split(b"\r\n"))
        self.assertIn(
            f"User-Agent: {http_client.APPLICATION_USER_AGENT}".encode(
                "ascii"),
            raw_headers.split(b"\r\n"),
        )
        self.assertIn(b"X-Test: ok", raw_headers.split(b"\r\n"))
        self.assertIn(b"Authorization: Bearer token", raw_headers.split(b"\r\n"))
        self.assertIn(b"Content-Length: 2", raw_headers.split(b"\r\n"))

    def test_http_request_uses_plain_connection_and_default_port(self):
        connector = FakeConnector([
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
        ])

        with PatchedOpenConnection(connector):
            response = asyncio.run(http_client.async_http_request(
                "GET",
                "http://example.test/path;param?q=1",
                timeout=5,
                max_bytes=100,
            ))

        self.assertEqual(response.status, 200)
        self.assertEqual(connector.calls[0]["host"], "example.test")
        self.assertEqual(connector.calls[0]["port"], 80)
        self.assertIsNone(connector.calls[0]["ssl"])
        self.assertIsNone(connector.calls[0]["server_hostname"])
        raw_headers = bytes(connector.writers[0].data).split(b"\r\n\r\n", 1)[0]
        self.assertEqual(raw_headers.split(b"\r\n")[0], b"GET /path;param?q=1 HTTP/1.1")
        self.assertIn(b"Host: example.test", raw_headers.split(b"\r\n"))
        self.assertNotIn(b"Content-Length: 0", raw_headers.split(b"\r\n"))

    def test_rejects_header_injection_before_connecting(self):
        connector = FakeConnector([])

        with PatchedOpenConnection(connector):
            with self.assertRaises(ValueError):
                asyncio.run(http_client.async_http_request(
                    "GET",
                    "https://example.test/",
                    headers_in={"X-Bad": "ok\r\nInjected: yes"},
                    timeout=5,
                ))

        self.assertEqual(connector.calls, [])

    def test_rejects_endpoint_owned_user_agent_before_connecting(self):
        connector = FakeConnector([])

        with PatchedOpenConnection(connector):
            with self.assertRaisesRegex(
                    ValueError, "User-Agent is owned by Loki"):
                asyncio.run(http_client.async_http_request(
                    "GET",
                    "https://example.test/",
                    headers_in={"user-agent": "feature-specific/1.0"},
                    timeout=5,
                ))

        self.assertEqual(connector.calls, [])

    def test_content_length_body_is_truncated_at_limit(self):
        connector = FakeConnector([
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"
        ])

        with PatchedOpenConnection(connector):
            response = asyncio.run(http_client.async_http_request(
                "GET",
                "https://example.test/",
                timeout=5,
                max_bytes=3,
            ))

        self.assertEqual(response.body, b"hel")
        self.assertTrue(response.truncated)

    def test_chunked_body_and_duplicate_response_headers(self):
        connector = FakeConnector([
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"X-Test: one\r\n"
            b"X-Test: two\r\n"
            b"\r\n"
            b"3\r\nabc\r\n"
            b"4;ext=value\r\ndefg\r\n"
            b"0\r\n\r\n"
        ])

        with PatchedOpenConnection(connector):
            response = asyncio.run(http_client.async_http_request(
                "GET",
                "https://example.test/",
                timeout=5,
                max_bytes=100,
            ))

        self.assertEqual(response.body, b"abcdefg")
        self.assertFalse(response.truncated)
        self.assertEqual(response.headers["x-test"], "one, two")

    def test_invalid_status_line_raises(self):
        connector = FakeConnector([b"not-http\r\n\r\n"])

        with PatchedOpenConnection(connector):
            with self.assertRaises(OSError):
                asyncio.run(http_client.async_http_request(
                    "GET",
                    "https://example.test/",
                    timeout=5,
                    max_bytes=100,
                ))


class HttpClientRedirectTests(unittest.TestCase):
    def test_same_host_redirect_is_followed(self):
        connector = FakeConnector([
            b"HTTP/1.1 302 Found\r\nLocation: /next\r\nContent-Length: 7\r\n\r\ndiscard",
            b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\ndone",
        ])
        tls_context = object()
        headers = {"X-Test": "ok", "Authorization": "Bearer redirect-secret"}
        real_open = connector.open_connection

        async def open_connection(*args, **kwargs):
            if connector.writers:
                self.assertTrue(connector.writers[0].closed)
                self.assertTrue(connector.writers[0].wait_closed_called)
            return await real_open(*args, **kwargs)

        connector.open_connection = open_connection

        async def scenario():
            return await asyncio.wait_for(http_client.async_http_request_follow_same_host(
                "GET", "https://example.test/start", headers_in=headers,
                timeout=5, max_bytes=10), 3)

        with PatchedOpenConnection(connector, tls_context):
            response = asyncio.run(scenario())
            self.assertEqual(response.status, 200)
            self.assertEqual(response.reason, "OK")
            self.assertEqual(response.url, "https://example.test/next")
            self.assertEqual(response.body, b"done")
            self.assertEqual(response.headers, {"content-length": "4"})
            self.assertFalse(response.truncated)
            self.assertIsNone(response.redirect_url)
            self.assertEqual(connector.responses, [])
            self.assertEqual(connector.calls, [{
                "host": "example.test", "port": 443,
                "ssl": tls_context, "server_hostname": "example.test",
            }] * 2)
            self.assertEqual(len(connector.writers), 2)
            for path, writer in zip(("/start", "/next"), connector.writers):
                self.assertEqual(bytes(writer.data), (
                    f"GET {path} HTTP/1.1\r\nHost: example.test\r\nConnection: close\r\n"
                    f"User-Agent: {http_client.APPLICATION_USER_AGENT}\r\n"
                    "X-Test: ok\r\nAuthorization: Bearer redirect-secret\r\n\r\n").encode())
                self.assertTrue(writer.closed)
                self.assertTrue(writer.wait_closed_called)
            self.assertEqual(headers, {"X-Test": "ok", "Authorization": "Bearer redirect-secret"})

    def test_cross_host_redirect_is_reported_not_followed(self):
        old_request = http_client.async_http_request
        calls = []

        async def fake_request(method, request_url, **kwargs):
            calls.append(request_url)
            return http_client.HttpResponse(
                request_url,
                302,
                "Found",
                {"location": "https://other.test/path"},
                b"",
            )

        try:
            http_client.async_http_request = fake_request
            response = asyncio.run(http_client.async_http_request_follow_same_host(
                "GET",
                "https://example.test/start",
                timeout=5,
                max_bytes=10,
            ))
        finally:
            http_client.async_http_request = old_request

        self.assertEqual(calls, ["https://example.test/start"])
        self.assertEqual(response.status, 302)
        self.assertEqual(response.redirect_url, "https://other.test/path")

    def test_https_redirect_cannot_downgrade_on_the_same_host(self):
        calls = []

        async def fake_request(method, request_url, **kwargs):
            calls.append(request_url)
            return http_client.HttpResponse(
                request_url,
                302,
                "Found",
                {"location": "http://example.test/plaintext"},
                b"",
            )

        with mock.patch.object(
                http_client, "async_http_request", new=fake_request):
            response = asyncio.run(
                http_client.async_http_request_follow_same_host(
                    "GET",
                    "https://example.test/start",
                    headers_in={"Authorization": "Bearer secret"},
                ))

        self.assertEqual(calls, ["https://example.test/start"])
        self.assertEqual(
            response.redirect_url,
            "http://example.test/plaintext",
        )

    def test_http_may_upgrade_but_cannot_downgrade_afterward(self):
        calls = []

        async def fake_request(method, request_url, **kwargs):
            calls.append(request_url)
            location = (
                "https://example.test/secure"
                if request_url.startswith("http:")
                else "http://example.test/plaintext")
            return http_client.HttpResponse(
                request_url, 302, "Found", {"location": location}, b"")

        with mock.patch.object(
                http_client, "async_http_request", new=fake_request):
            response = asyncio.run(
                http_client.async_http_request_follow_same_host(
                    "GET", "http://example.test/start"))

        self.assertEqual(calls, [
            "http://example.test/start",
            "https://example.test/secure",
        ])
        self.assertEqual(
            response.redirect_url,
            "http://example.test/plaintext",
        )


class HttpClientRetryTests(unittest.TestCase):
    def test_retry_preparation_sees_state_from_prior_response_headers(self):
        async def scenario():
            state = {"value": None}
            prepared, observed, body_reads = [], [], []
            headers = {"Authorization": "Bearer retry-secret"}
            body = b'{"input":"request"}'
            connector = FakeConnector([
                b"HTTP/1.1 200 OK\r\nX-Routing-State: final-state\r\nContent-Length: 4\r\n\r\ndone",
            ])
            real_open = connector.open_connection

            class ResettingReader(asyncio.StreamReader):
                async def readexactly(inner_self, size):
                    self.assertEqual(state["value"], "server-state")
                    data = await super().readexactly(size)
                    body_reads.append((size, data))
                    if data == b"\r\n":
                        # Reset only after real headers and a complete body
                        # chunk were consumed, not before parsing can start.
                        inner_self.set_exception(ConnectionResetError("body read reset"))
                    return data

            async def open_connection(host, port, ssl=None, server_hostname=None):
                if connector.calls:
                    self.assertTrue(connector.writers[0].closed)
                    self.assertTrue(connector.writers[0].wait_closed_called)
                    return await real_open(host, port, ssl=ssl, server_hostname=server_hostname)
                connector.calls.append({
                    "host": host, "port": port, "ssl": ssl, "server_hostname": server_hostname})
                reader, writer = ResettingReader(), FakeWriter()
                reader.feed_data(
                    b"HTTP/1.1 200 OK\r\nX-Routing-State: server-state\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n5\r\nstale\r\n")
                connector.writers.append(writer)
                return reader, writer

            def prepare(attempt_headers):
                prepared.append((dict(attempt_headers), state["value"]))
                if state["value"] is not None:
                    attempt_headers["X-Routing-State"] = state["value"]

            def capture(status, response_headers):
                observed.append((status, dict(response_headers)))
                state["value"] = response_headers.get("x-routing-state")

            connector.open_connection = open_connection
            tls_context = object()
            with PatchedOpenConnection(connector, tls_context):
                response = await asyncio.wait_for(http_client.async_http_request(
                    "POST", "https://example.test/responses", headers_in=headers, body=body,
                    timeout=5, retry_max_attempts=2, retry_base_delay_s=0,
                    retry_max_jitter_s=0, prepare_attempt_headers=prepare,
                    on_response_headers=capture), 3)
                self.assertEqual(response.url, "https://example.test/responses")
                self.assertEqual(response.status, 200)
                self.assertEqual(response.reason, "OK")
                self.assertEqual(response.body, b"done")
                self.assertFalse(response.truncated)
                self.assertEqual(response.headers, {"x-routing-state": "final-state", "content-length": "4"})
                self.assertEqual(body_reads, [(5, b"stale"), (2, b"\r\n")])
                self.assertEqual(prepared, [(headers, None), (headers, "server-state")])
                self.assertEqual(observed, [
                    (200, {"x-routing-state": "server-state", "transfer-encoding": "chunked"}),
                    (200, {"x-routing-state": "final-state", "content-length": "4"}),
                ])
                self.assertEqual(state["value"], "final-state")
                self.assertEqual(connector.responses, [])
                self.assertEqual(connector.calls, [{
                    "host": "example.test", "port": 443,
                    "ssl": tls_context, "server_hostname": "example.test",
                }] * 2)
                self.assertEqual(len(connector.writers), 2)
                for routing, writer in zip(("", "X-Routing-State: server-state\r\n"), connector.writers):
                    self.assertEqual(bytes(writer.data), (
                        "POST /responses HTTP/1.1\r\nHost: example.test\r\nConnection: close\r\n"
                        f"User-Agent: {http_client.APPLICATION_USER_AGENT}\r\n"
                        f"Authorization: Bearer retry-secret\r\n{routing}Content-Length: {len(body)}\r\n\r\n"
                    ).encode() + body)
                    self.assertTrue(writer.closed)
                    self.assertTrue(writer.wait_closed_called)
                self.assertEqual(headers, {"Authorization": "Bearer retry-secret"})

        asyncio.run(scenario())

    def test_response_headers_are_reported_before_body_read_failure(self):
        connector = FakeConnector([
            b"HTTP/1.1 200 OK\r\n"
            b"X-Routing-State: server-state\r\n"
            b"Content-Length: 5\r\n"
            b"\r\n"
            b"no"
        ])
        observed = []

        with PatchedOpenConnection(connector):
            with self.assertRaises(
                    http_client.HttpRequestDeliveryError):
                asyncio.run(http_client.async_http_request(
                    "POST",
                    "https://example.test/responses",
                    retry_max_attempts=1,
                    on_response_headers=(
                        lambda status, headers: observed.append(
                            (status, dict(headers)))),
                ))

        self.assertEqual(observed, [(
            200,
            {
                "x-routing-state": "server-state",
                "content-length": "5",
            },
        )])

    def test_retries_once_on_connection_reset_then_succeeds(self):
        # First open_connection raises a transient transport error; the second
        # returns a valid response. The wrapper should retry and succeed.
        connector = FakeConnector([
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"
        ])

        attempts = {"n": 0}
        connector_open = connector.open_connection

        async def flaky_open(host, port, ssl=None, server_hostname=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionResetError("simulated reset")
            return await connector_open(host, port, ssl=ssl, server_hostname=server_hostname)

        connector.open_connection = flaky_open

        with PatchedOpenConnection(connector):
            # perf_counter, not monotonic: Windows monotonic() is tick-
            # resolution (GetTickCount64, ~15.6 ms) and cannot measure a
            # 50 ms bound.
            start = time.perf_counter()
            response = asyncio.run(http_client.async_http_request(
                "GET",
                "https://example.test/",
                timeout=5,
                max_bytes=100,
                retry_max_attempts=3,
                retry_base_delay_s=0.05,
                retry_max_jitter_s=0.0,
                retry_backoff_factor=2.0,
            ))
            elapsed = time.perf_counter() - start

        self.assertEqual(response.status, 200)
        self.assertEqual(response.body, b"hello")
        self.assertEqual(attempts["n"], 2)
        # Backoff: at least one base_delay_s sleep before the second attempt.
        self.assertGreaterEqual(elapsed, 0.05)

    def test_does_not_retry_on_malformed_response(self):
        # An OSError raised during response parsing (no errno) must not retry.
        connector = FakeConnector([b"not-http\r\n\r\n"])

        with PatchedOpenConnection(connector):
            with self.assertRaises(OSError):
                asyncio.run(http_client.async_http_request(
                    "GET",
                    "https://example.test/",
                    timeout=5,
                    max_bytes=100,
                    retry_max_attempts=3,
                    retry_base_delay_s=0.0,
                    retry_max_jitter_s=0.0,
                ))

        self.assertEqual(len(connector.calls), 1)


class HttpClientStreamingTests(unittest.TestCase):
    def test_stream_can_be_cancelled_during_connection_setup(self):
        async def scenario(active):
            connector = FakeConnector([])
            entered, interrupted = asyncio.Event(), asyncio.Event()
            operations = []
            cancelled = not active
            contexts = []

            async def open_connection(host, port, ssl=None, server_hostname=None):
                connector.calls.append((host, port, ssl, server_hostname))
                operations.append(asyncio.current_task())
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    interrupted.set()

            async def request():
                async with http_client.async_http_stream(
                        "POST", "http://example.test/stream", body=b"{}",
                        timeout=20, max_bytes=100, cancel_check=lambda: cancelled):
                    contexts.append("entered")

            connector.open_connection = open_connection
            with PatchedOpenConnection(connector):
                async with _running_request(request()) as task:
                    if active:
                        await asyncio.wait_for(entered.wait(), 1)
                        self.assertFalse(cancelled)
                        self.assertFalse(task.done())
                        self.assertFalse(interrupted.is_set())
                        cancelled = True
                    with self.assertRaises(http_client.HttpRequestCancelled):
                        await asyncio.wait_for(task, 1)
                    self.assertTrue(task.done())
                    self.assertEqual(contexts, [])
                    self.assertEqual(connector.writers, [])
                    self.assertEqual(connector.calls, [("example.test", 80, None, None)] if active else [])
                    self.assertEqual(entered.is_set(), active)
                    self.assertEqual(interrupted.is_set(), active)
                    self.assertEqual(len(operations), int(active))
                    for operation in operations:
                        self.assertTrue(operation.done())
                        self.assertTrue(operation.cancelled())

        for active in [False, True]:
            with self.subTest(active=active):
                asyncio.run(scenario(active))

    def test_chunked_response_is_exposed_incrementally_and_closed(self):
        async def scenario():
            connector = FakeConnector([])
            first_observed, second_read = asyncio.Event(), asyncio.Event()
            released = False
            chunks, responses = [], []

            class GatedReader(asyncio.StreamReader):
                async def readline(inner_self):
                    if first_observed.is_set() and not released:
                        second_read.set()
                    return await super().readline()

            reader = GatedReader()
            reader.feed_data(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n")
            writer = FakeWriter()

            async def open_connection(host, port, ssl=None, server_hostname=None):
                connector.calls.append((host, port, ssl, server_hostname))
                connector.writers.append(writer)
                return reader, writer

            async def consume():
                async with http_client.async_http_stream(
                        "POST", "http://example.test/stream", body=b"{}",
                        timeout=20, max_bytes=100) as response:
                    responses.append(response)
                    async for chunk in response.body:
                        chunks.append(chunk)
                        if len(chunks) == 1:
                            first_observed.set()

            connector.open_connection = open_connection
            with PatchedOpenConnection(connector):
                async with _running_request(consume()) as task:
                    await asyncio.wait_for(first_observed.wait(), 1)
                    await asyncio.wait_for(second_read.wait(), 1)
                    self.assertEqual(chunks, [b"abc"])
                    self.assertFalse(task.done())
                    self.assertFalse(released)
                    self.assertFalse(reader.at_eof())
                    self.assertFalse(writer.closed)
                    self.assertFalse(writer.wait_closed_called)
                    self.assertEqual(len(responses), 1)
                    self.assertEqual(responses[0].status, 200)
                    self.assertEqual(responses[0].header("content-type"), "text/event-stream")
                    released = True
                    reader.feed_data(b"4\r\ndefg\r\n0\r\nX-Trailer: ignored\r\n\r\n")
                    reader.feed_eof()
                    await asyncio.wait_for(task, 1)
                    self.assertTrue(task.done())
                    self.assertFalse(task.cancelled())
                    self.assertEqual(chunks, [b"abc", b"defg"])
                    self.assertEqual(responses[0].headers, {
                        "content-type": "text/event-stream", "transfer-encoding": "chunked"})
                    self.assertEqual(connector.calls, [("example.test", 80, None, None)])
                    self.assertEqual(connector.writers, [writer])
                    self.assertEqual(bytes(writer.data), (
                        "POST /stream HTTP/1.1\r\nHost: example.test\r\nConnection: close\r\n"
                        f"User-Agent: {http_client.APPLICATION_USER_AGENT}\r\n"
                        "Content-Length: 2\r\n\r\n{}").encode())
                    self.assertTrue(writer.closed)
                    self.assertTrue(writer.wait_closed_called)

        asyncio.run(scenario())

    def test_stream_content_length_is_read_without_buffering_api(self):
        connector = FakeConnector([
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 7\r\n"
            b"\r\n"
            b"abcdefg"
        ])

        async def request():
            async with http_client.async_http_stream(
                    "GET", "http://example.test/data",
                    timeout=5, max_bytes=100) as response:
                return b"".join(
                    [chunk async for chunk in response.body])

        with PatchedOpenConnection(connector):
            body = asyncio.run(request())

        self.assertEqual(body, b"abcdefg")
        self.assertTrue(connector.writers[0].closed)

    def test_stream_byte_limit_closes_connection(self):
        connector = FakeConnector([
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"4\r\nabcd\r\n"
            b"0\r\n\r\n"
        ])

        async def request():
            async with http_client.async_http_stream(
                    "GET", "http://example.test/data",
                    timeout=5, max_bytes=3) as response:
                return [chunk async for chunk in response.body]

        with PatchedOpenConnection(connector):
            with self.assertRaisesRegex(OSError, "byte limit"):
                asyncio.run(request())

        self.assertTrue(connector.writers[0].closed)
        self.assertTrue(connector.writers[0].wait_closed_called)


if __name__ == "__main__":
    unittest.main()
