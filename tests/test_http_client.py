import asyncio
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import http_client


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
        old_request = http_client.async_http_request
        calls = []

        async def fake_request(method, request_url, **kwargs):
            calls.append((method, request_url, kwargs))
            if request_url.endswith("/start"):
                return http_client.HttpResponse(
                    request_url,
                    302,
                    "Found",
                    {"location": "/next"},
                    b"",
                )
            return http_client.HttpResponse(request_url, 200, "OK", {}, b"done")

        try:
            http_client.async_http_request = fake_request
            response = asyncio.run(http_client.async_http_request_follow_same_host(
                "GET",
                "https://example.test/start",
                headers_in={"X-Test": "ok"},
                timeout=5,
                max_bytes=10,
            ))
        finally:
            http_client.async_http_request = old_request

        self.assertEqual(response.status, 200)
        self.assertEqual(response.url, "https://example.test/next")
        self.assertEqual(response.body, b"done")
        self.assertEqual([call[1] for call in calls], [
            "https://example.test/start",
            "https://example.test/next",
        ])
        self.assertEqual(calls[0][2]["headers_in"], {"X-Test": "ok"})

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


if __name__ == "__main__":
    unittest.main()
