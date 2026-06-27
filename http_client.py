import asyncio
import ssl
import urllib.parse
from dataclasses import dataclass


HTTP_HEADER_MAX_BYTES = 64 * 1024
HTTP_MAX_RESPONSE_BYTES = 50 * 1024 * 1024


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
            truncated = True
            break
    return b''.join(chunks), truncated


def _build_raw_request(method: str, request_url: str, headers_in: dict = None,
                       body: bytes = b''):
    parsed = urllib.parse.urlparse(request_url)
    if parsed.scheme not in ['http', 'https'] or not parsed.hostname:
        raise ValueError(f"unsupported URL: {request_url}")
    request_headers = {
        'Host': _host_header(parsed),
        'Connection': 'close',
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


async def async_http_request(method: str, request_url: str, *, headers_in: dict = None,
                             body: bytes = b'', timeout: int = 30,
                             max_bytes: int = HTTP_MAX_RESPONSE_BYTES) -> HttpResponse:
    async def request_once() -> HttpResponse:
        parsed, raw_request = _build_raw_request(method, request_url, headers_in, body)
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        ssl_context = ssl.create_default_context() if parsed.scheme == 'https' else None
        reader, writer = await asyncio.open_connection(
            parsed.hostname,
            port,
            ssl=ssl_context,
            server_hostname=parsed.hostname if ssl_context else None,
        )
        try:
            writer.write(raw_request)
            await writer.drain()

            status_line, response_headers = await _read_headers(reader)
            parts = status_line.split(' ', 2)
            if len(parts) < 2 or not parts[1].isdigit():
                raise OSError(f"invalid HTTP status line: {status_line!r}")
            status = int(parts[1])
            reason = parts[2] if len(parts) > 2 else ''
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
                pass

    return await asyncio.wait_for(request_once(), timeout=timeout)


def _redirect_location(response: HttpResponse) -> str | None:
    location = response.header('location')
    if response.status in range(300, 400) and location:
        return urllib.parse.urljoin(response.url, location)
    return None


async def async_http_request_follow_same_host(method: str, request_url: str, *,
                                              headers_in: dict = None, body: bytes = b'',
                                              timeout: int = 30,
                                              max_bytes: int = HTTP_MAX_RESPONSE_BYTES,
                                              max_redirects: int = 5) -> HttpResponse:
    current_url = request_url
    original_host = urllib.parse.urlparse(request_url).netloc
    for _ in range(max_redirects + 1):
        response = await async_http_request(method, current_url, headers_in=headers_in,
                                            body=body, timeout=timeout, max_bytes=max_bytes)
        next_url = _redirect_location(response)
        if not next_url:
            return response
        next_host = urllib.parse.urlparse(next_url).netloc
        if next_host != original_host:
            response.redirect_url = next_url
            return response
        current_url = next_url
    return response
