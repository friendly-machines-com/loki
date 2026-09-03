import asyncio
import contextlib
import random
import ssl
import urllib.parse
from dataclasses import dataclass

from . import __version__


# This is Loki's small dependency-free HTTP/1.1 transport, not a general browser
# stack. Every production Internet request uses it. Keep transport policy here
# explicit because callers depend on exact redirect, truncation, identity, and
# header-validation behavior.
HTTP_HEADER_MAX_BYTES = 64 * 1024
HTTP_MAX_RESPONSE_BYTES = 50 * 1024 * 1024
# Application identity is a property of Loki, not of an endpoint or feature.
# Every production HTTP path reaches _build_raw_request(), which installs this
# value and rejects caller overrides. Protocol, authentication, and per-request
# headers remain caller-owned and therefore cannot silently fork Loki's name.
APPLICATION_USER_AGENT = f"loki/{__version__}"


# Errno values for which retrying is safe: the failure is transport-level and
# the server is unlikely to have processed the request. HTTP-parsing OSErrors
# raised inside this module (empty response, header overflow, bad chunk size,
# invalid status line) are not in this set, so a malformed response propagates
# immediately instead of being retried.
_RETRYABLE_ERRNOS = frozenset({
    104,   # ECONNRESET
    110,   # ETIMEDOUT
    111,   # ECONNREFUSED
    113,   # EHOSTUNREACH
    100,   # ENETUNREACH
    32,    # EPIPE
    58,    # ESHUTDOWN
})


class HttpRequestCancelled(Exception):
    pass


class HttpRequestDeliveryError(OSError):
    """Transport failure annotated with whether request bytes were queued."""

    def __init__(self, cause: BaseException, request_may_have_been_sent: bool):
        self.cause = cause
        self.request_may_have_been_sent = request_may_have_been_sent
        super().__init__(str(cause))


async def _wait_with_cancel(awaitable, timeout, cancel_check):
    task = asyncio.create_task(awaitable)
    deadline = asyncio.get_running_loop().time() + timeout
    try:
        while True:
            if cancel_check():
                if task.done():
                    try:
                        task.result()
                    except BaseException:
                        pass
                raise HttpRequestCancelled()
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            done, _ = await asyncio.wait(
                {task}, timeout=min(0.05, remaining))
            if done:
                # Cancellation is an instruction, not a race for whichever
                # future happens to be inspected first.  If both completed in
                # the same event-loop turn, cancellation wins.
                if cancel_check():
                    try:
                        task.result()
                    except BaseException:
                        pass
                    raise HttpRequestCancelled()
                return task.result()
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, HttpRequestDeliveryError):
        return _is_transient(exc.cause)
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError,
                        ConnectionRefusedError, BrokenPipeError)):
        return True
    if isinstance(exc, OSError):
        return getattr(exc, "errno", None) in _RETRYABLE_ERRNOS
    return False


def is_transient_error(exc: BaseException) -> bool:
    return _is_transient(exc)


@dataclass
class HttpResponse:
    url: str
    status: int
    reason: str
    headers: dict
    body: bytes
    truncated: bool = False
    redirect_url: str | None = None

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name.lower(), default)


@dataclass
class HttpStreamResponse:
    url: str
    status: int
    reason: str
    headers: dict
    body: object

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name.lower(), default)


class HttpBodyStream:
    """Incrementally expose one HTTP response body without buffering it."""

    def __init__(self, reader, headers, *, timeout, max_bytes,
                 cancel_check=None):
        self.reader = reader
        self.headers = headers
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.bytes_received = 0
        self._iterated = False
        self.cancel_check = cancel_check or (lambda: False)

    def __aiter__(self):
        if self._iterated:
            raise RuntimeError("HTTP response body can only be consumed once")
        self._iterated = True
        return self._iterate()

    async def _read(self, awaitable):
        return await _wait_with_cancel(
            awaitable, self.timeout, self.cancel_check)

    async def _read_exact_idle(self, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = await self._read(self.reader.read(remaining))
            if not chunk:
                raise OSError("unexpected EOF in HTTP response body")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _count(self, data):
        self.bytes_received += len(data)
        if self.bytes_received > self.max_bytes:
            raise OSError(
                f"HTTP response body exceeds {self.max_bytes} byte limit")
        return data

    async def _iterate(self):
        transfer_encoding = self.headers.get("transfer-encoding", "").lower()
        content_length = self.headers.get("content-length", "")
        if "chunked" in transfer_encoding:
            async for chunk in self._iterate_chunked():
                yield chunk
            return
        if content_length.isdigit():
            remaining = int(content_length)
            if remaining > self.max_bytes:
                raise OSError(
                    f"HTTP response body exceeds {self.max_bytes} byte limit")
            while remaining:
                size = min(65536, remaining)
                chunk = await self._read(self.reader.read(size))
                if not chunk:
                    raise OSError("unexpected EOF in HTTP response body")
                remaining -= len(chunk)
                if chunk:
                    yield self._count(chunk)
            return
        while True:
            remaining_capacity = self.max_bytes - self.bytes_received
            chunk = await self._read(
                self.reader.read(min(65536, remaining_capacity + 1)))
            if not chunk:
                return
            yield self._count(chunk)

    async def _iterate_chunked(self):
        while True:
            line = await self._read(self.reader.readline())
            if not line:
                raise OSError("unexpected EOF in chunked HTTP response")
            size_text = line.split(b";", 1)[0].strip()
            try:
                size = int(size_text, 16)
            except ValueError:
                raise OSError(f"invalid chunk size {size_text!r}")
            if size < 0:
                raise OSError(f"invalid chunk size {size_text!r}")
            if size == 0:
                while True:
                    trailer = await self._read(self.reader.readline())
                    if trailer in (b"\r\n", b"\n"):
                        return
                    if not trailer:
                        return
            if self.bytes_received + size > self.max_bytes:
                raise OSError(
                    f"HTTP response body exceeds {self.max_bytes} byte limit")
            data = await self._read_exact_idle(size)
            terminator = await self._read_exact_idle(2)
            if terminator != b"\r\n":
                raise OSError("invalid chunk terminator")
            if data:
                yield self._count(data)


def _host_header(parsed) -> str:
    default_port = 443 if parsed.scheme == 'https' else 80
    if parsed.port and parsed.port != default_port:
        return f"{parsed.hostname}:{parsed.port}"
    return parsed.hostname or ''


def _request_target(parsed) -> str:
    path = parsed.path or '/'
    if parsed.params:
        path += ';' + parsed.params
    if parsed.query:
        path += '?' + parsed.query
    return path


def _validate_header_line(name, value):
    # This module writes HTTP/1.1 requests directly to a socket. Since no
    # library validates header fields for us, reject syntax that could break out
    # of the current header line before formatting "Name: value".
    name = str(name)
    value = str(value)
    if not name or any(ord(ch) <= 32 or ord(ch) >= 127 or ch == ':' for ch in name):
        raise ValueError(f"invalid HTTP header name: {name!r}")
    if '\r' in value or '\n' in value:
        raise ValueError(f"invalid HTTP header value for {name!r}")


async def _read_headers(reader: asyncio.StreamReader) -> tuple[str, dict]:
    total = 0
    status_line = await reader.readline()
    if not status_line:
        raise OSError("empty HTTP response")
    total += len(status_line)
    header_lines = []
    while True:
        line = await reader.readline()
        if not line:
            break
        total += len(line)
        if total > HTTP_HEADER_MAX_BYTES:
            raise OSError("HTTP headers too large")
        if line in [b'\r\n', b'\n']:
            break
        header_lines.append(line.decode('iso-8859-1').rstrip('\r\n'))

    headers_out = {}
    for line in header_lines:
        if ':' not in line:
            continue
        name, value = line.split(':', 1)
        key = name.strip().lower()
        value = value.strip()
        if key in headers_out:
            headers_out[key] += ', ' + value
        else:
            headers_out[key] = value
    return status_line.decode('iso-8859-1').rstrip('\r\n'), headers_out


async def _read_until_eof(reader: asyncio.StreamReader, max_bytes: int) -> tuple[bytes, bool]:
    chunks = []
    total = 0
    truncated = False
    while True:
        # Read one byte past the limit so callers can distinguish exactly
        # max_bytes bytes from a response that was cut short.
        chunk = await reader.read(min(65536, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            truncated = True
            break
    body = b''.join(chunks)
    if len(body) > max_bytes:
        return body[:max_bytes], True
    return body, truncated


async def _read_content_length(reader: asyncio.StreamReader, length: int,
                               max_bytes: int) -> tuple[bytes, bool]:
    # When the declared length exceeds the cap, read one byte past the cap so
    # the returned body follows the same truncation path as the other readers.
    to_read = min(length, max_bytes + 1)
    body = await reader.readexactly(to_read) if to_read else b''
    truncated = length > max_bytes or len(body) > max_bytes
    if len(body) > max_bytes:
        body = body[:max_bytes]
    return body, truncated


async def _read_chunked_body(reader: asyncio.StreamReader, max_bytes: int) -> tuple[bytes, bool]:
    chunks = []
    total = 0
    truncated = False
    while True:
        line = await reader.readline()
        if not line:
            break
        size_text = line.split(b';', 1)[0].strip()
        try:
            size = int(size_text, 16)
        except ValueError:
            raise OSError(f"invalid chunk size {size_text!r}")
        if size == 0:
            await reader.readline()
            break
        data = await reader.readexactly(size)
        await reader.readexactly(2)  # CRLF
        keep = 0
        if total < max_bytes:
            keep = min(size, max_bytes - total)
            chunks.append(data[:keep])
            total += keep
        if size > keep:
            # The connection is closed by async_http_request(); draining the
            # rest of an over-limit response would defeat the byte cap.
            truncated = True
            break
    return b''.join(chunks), truncated


def _build_raw_request(method: str, request_url: str, headers_in: dict = None,
                       body: bytes = b''):
    parsed = urllib.parse.urlparse(request_url)
    if parsed.scheme not in ['http', 'https'] or not parsed.hostname:
        raise ValueError(f"unsupported URL: {request_url}")
    if any(
            str(name).lower() == "user-agent"
            for name in (headers_in or {})):
        raise ValueError(
            "User-Agent is owned by Loki's HTTP transport")
    request_headers = {
        'Host': _host_header(parsed),
        'Connection': 'close',
        'User-Agent': APPLICATION_USER_AGENT,
    }
    if headers_in:
        request_headers.update(headers_in)
    if body:
        request_headers['Content-Length'] = str(len(body))

    lines = [f"{method.upper()} {_request_target(parsed)} HTTP/1.1"]
    for name, value in request_headers.items():
        _validate_header_line(name, value)
    lines.extend(f"{name}: {value}" for name, value in request_headers.items())
    raw_request = ("\r\n".join(lines) + "\r\n\r\n").encode('iso-8859-1') + body
    return parsed, raw_request


async def _async_http_request_once(method: str, request_url: str, *, headers_in: dict = None,
                                   body: bytes = b'', timeout: int = 30,
                                   max_bytes: int = HTTP_MAX_RESPONSE_BYTES,
                                   cancel_check=None,
                                   on_response_headers=None) -> HttpResponse:
    # Validate and format caller-controlled URL/header data before entering
    # transport delivery tracking. A local ValueError is not a failed network
    # delivery and must remain distinguishable to callers.
    parsed, raw_request = _build_raw_request(
        method, request_url, headers_in, body)
    delivery = {"request_may_have_been_sent": False}

    async def request_once() -> HttpResponse:
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        ssl_context = ssl.create_default_context() if parsed.scheme == 'https' else None
        reader, writer = await asyncio.open_connection(
            parsed.hostname,
            port,
            ssl=ssl_context,
            server_hostname=parsed.hostname if ssl_context else None,
        )
        try:
            # Mark first, because an arbitrary transport's write() contract
            # does not prove that an exception means zero bytes were queued.
            # A false positive costs a login; a false negative can replay a
            # rotating refresh token and destroy the newer credential state.
            delivery["request_may_have_been_sent"] = True
            writer.write(raw_request)
            await writer.drain()

            status_line, response_headers = await _read_headers(reader)
            parts = status_line.split(' ', 2)
            if len(parts) < 2 or not parts[1].isdigit():
                raise OSError(f"invalid HTTP status line: {status_line!r}")
            status = int(parts[1])
            reason = parts[2] if len(parts) > 2 else ''
            if on_response_headers is not None:
                # Some protocols mint routing state in response headers that
                # must be used if reading this response fails and the request
                # is retried. Report headers before consuming the body so the
                # caller can update its per-attempt state in time.
                on_response_headers(status, response_headers)
            transfer_encoding = response_headers.get('transfer-encoding', '').lower()
            if 'chunked' in transfer_encoding:
                response_body, truncated = await _read_chunked_body(reader, max_bytes)
            elif response_headers.get('content-length', '').isdigit():
                response_body, truncated = await _read_content_length(
                    reader, int(response_headers['content-length']), max_bytes)
            else:
                response_body, truncated = await _read_until_eof(reader, max_bytes)
            return HttpResponse(request_url, status, reason, response_headers, response_body, truncated)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                # Response handling already finished or failed; close errors
                # should not replace the real request result.
                pass

    try:
        return await _wait_with_cancel(
            request_once(), timeout, cancel_check or (lambda: False))
    except asyncio.CancelledError as error:
        # Task cancellation is control flow, so preserve its type. Rotating
        # credential callers still need the delivery fact to decide whether
        # replaying a refresh token is safe. CPython 3.10 reconstructs a plain
        # CancelledError when an already-cancelled Task is awaited, losing
        # custom attributes; callers must therefore default a missing fact to
        # "possibly sent", as the refresh broker does.
        error.request_may_have_been_sent = (
            delivery["request_may_have_been_sent"])
        raise
    except HttpRequestCancelled:
        raise
    except HttpRequestDeliveryError:
        raise
    except Exception as error:
        raise HttpRequestDeliveryError(
            error, delivery["request_may_have_been_sent"]) from error


@contextlib.asynccontextmanager
async def async_http_stream(method: str, request_url: str, *,
                            headers_in: dict = None, body: bytes = b"",
                            timeout: int = 30,
                            max_bytes: int = HTTP_MAX_RESPONSE_BYTES,
                            cancel_check=None):
    """Open an HTTP request and expose its response body as an async iterator.

    ``timeout`` is applied to connection setup, response headers, and each
    individual body read. It is therefore an idle timeout, not a cap on the
    total duration of an active stream.
    """
    parsed, raw_request = _build_raw_request(
        method, request_url, headers_in, body)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ssl_context = (
        ssl.create_default_context() if parsed.scheme == "https" else None)
    cancel = cancel_check or (lambda: False)
    writer = None
    try:
        reader, writer = await _wait_with_cancel(
            asyncio.open_connection(
                parsed.hostname,
                port,
                ssl=ssl_context,
                server_hostname=parsed.hostname if ssl_context else None,
            ),
            timeout,
            cancel,
        )
        writer.write(raw_request)
        await _wait_with_cancel(writer.drain(), timeout, cancel)
        status_line, response_headers = await _wait_with_cancel(
            _read_headers(reader), timeout, cancel)
        parts = status_line.split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            raise OSError(f"invalid HTTP status line: {status_line!r}")
        status = int(parts[1])
        reason = parts[2] if len(parts) > 2 else ""
        response_body = HttpBodyStream(
            reader, response_headers, timeout=timeout, max_bytes=max_bytes,
            cancel_check=cancel)
        yield HttpStreamResponse(
            request_url, status, reason, response_headers, response_body)
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


async def async_http_request(method: str, request_url: str, *, headers_in: dict = None,
                             body: bytes = b'', timeout: int = 30,
                             max_bytes: int = HTTP_MAX_RESPONSE_BYTES,
                             retry_max_attempts: int = 1,
                             retry_base_delay_s: float = 0.5,
                             retry_max_jitter_s: float = 0.5,
                             retry_backoff_factor: float = 2.0,
                             cancel_check=None,
                             prepare_attempt_headers=None,
                             on_response_headers=None) -> HttpResponse:
    cancel = cancel_check or (lambda: False)
    attempt = 0
    while True:
        if cancel():
            raise HttpRequestCancelled()
        attempt += 1
        try:
            attempt_headers = dict(headers_in or {})
            if prepare_attempt_headers is not None:
                # Rebuild headers for every transport attempt. A response-
                # header callback from the previous attempt may have acquired
                # protocol state that must be replayed on this retry.
                prepare_attempt_headers(attempt_headers)
            return await _async_http_request_once(
                method, request_url,
                headers_in=attempt_headers, body=body,
                timeout=timeout, max_bytes=max_bytes,
                cancel_check=cancel,
                on_response_headers=on_response_headers,
            )
        except Exception as exc:
            if attempt >= retry_max_attempts or not _is_transient(exc):
                raise
            delay = retry_base_delay_s * (retry_backoff_factor ** (attempt - 1))
            delay += random.uniform(0, retry_max_jitter_s)
            await _wait_with_cancel(asyncio.sleep(delay), delay + 0.1, cancel)


def _redirect_location(response: HttpResponse) -> str | None:
    location = response.header('location')
    if response.status in range(300, 400) and location:
        return urllib.parse.urljoin(response.url, location)
    return None


async def async_http_request_follow_same_host(method: str, request_url: str, *,
                                              headers_in: dict = None, body: bytes = b'',
                                              timeout: int = 30,
                                              max_bytes: int = HTTP_MAX_RESPONSE_BYTES,
                                              max_redirects: int = 5,
                                              retry_max_attempts: int = 1,
                                              retry_base_delay_s: float = 0.5,
                                              retry_max_jitter_s: float = 0.5,
                                              retry_backoff_factor: float = 2.0,
                                              cancel_check=None) -> HttpResponse:
    current_url = request_url
    original_host = urllib.parse.urlparse(request_url).netloc
    for _ in range(max_redirects + 1):
        response = await async_http_request(method, current_url, headers_in=headers_in,
                                            body=body, timeout=timeout, max_bytes=max_bytes,
                                            retry_max_attempts=retry_max_attempts,
                                            retry_base_delay_s=retry_base_delay_s,
                                            retry_max_jitter_s=retry_max_jitter_s,
                                            retry_backoff_factor=retry_backoff_factor,
                                            cancel_check=cancel_check)
        next_url = _redirect_location(response)
        if not next_url:
            return response
        next_host = urllib.parse.urlparse(next_url).netloc
        if next_host != original_host:
            # WebFetch surfaces cross-origin redirects to the model/user rather
            # than silently fetching a different authority.
            response.redirect_url = next_url
            return response
        current_url = next_url
    return response
