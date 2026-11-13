"""ACP transport: JSON-RPC over stdio, one message per line.

The front process speaks this on its real stdin/stdout; worker processes
use subprocess pipes.  fd 1 carries protocol messages only, so the
front process quarantines it (see quarantine_stdout) and every other
writer in the process inherits a devnull instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from . import handle_reader
from .lifetimes import finish_cleanup
from .streams import BufferedReader


class TransportError(Exception):
    def __init__(self, message: str, *, code: int = -32603, data=None):
        super().__init__(message)
        self.code = code
        self.data = data


# Enable a major only together with its wire/lifecycle implementation. Selection
# understands newer requests without pretending a version-constant change is v2.
SUPPORTED_PROTOCOL_VERSIONS = [1, 2]


def negotiate_protocol_version(params):
    """Select before parsing version-specific fields; never upgrade the peer.

    Initialization is the connection's contract, not catalog/model metadata.
    ACP's uint16 major and highest-supported <= request rule make downgrade
    explicit; malformed majors and an empty intersection are different errors.
    """
    requested = params.get("protocolVersion") if isinstance(params, dict) else None
    if isinstance(requested, bool) or not isinstance(requested, int) or not 0 <= requested <= 65535:
        raise TransportError("protocolVersion must be an integer from 0 to 65535", code=-32602)
    available = [version for version in SUPPORTED_PROTOCOL_VERSIONS if version <= requested]
    if not available:
        raise TransportError("Unsupported protocol version", code=-32022,
                             data={"requested": requested, "supported": list(SUPPORTED_PROTOCOL_VERSIONS)})
    return max(available)


# ACP carries prompts/catalogs, unlike small credential RPC. The public budget
# counts UTF-8 JSON bytes before LF; both subprocess backends use this same bound.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class MessageTooLarge(TransportError):
    def __init__(self):
        super().__init__(f'ACP message exceeds the {MAX_MESSAGE_BYTES}-byte limit')


def encode_message(message: dict) -> bytes:
    """Encode one bounded frame before writing any part of it."""
    data = bytearray()
    try:
        for piece in json.JSONEncoder(ensure_ascii=False).iterencode(message):
            remaining = MAX_MESSAGE_BYTES - len(data)
            if len(piece) > remaining:
                raise MessageTooLarge()
            encoded = piece.encode('utf-8')
            if len(encoded) > remaining:
                raise MessageTooLarge()
            data.extend(encoded)
    except (ValueError, TypeError, RecursionError) as error:
        raise TransportError('ACP message cannot be encoded as UTF-8 JSON') from error
    data.append(10)
    return bytes(data)


def error_response(request_id, error: dict) -> dict:
    """An error must fit too, even when its original diagnostic/id cannot."""
    message = response(request_id, error=error)
    try:
        encode_message(message)
        return message
    except TransportError as failure:
        bounded = {'code': INTERNAL_ERROR, 'message': str(failure)}
    message = response(request_id, error=bounded)
    try:
        encode_message(message)
    except TransportError:
        message = response(None, error={
            'code': INVALID_REQUEST,
            'message': 'ACP request identifier cannot fit a bounded response'})
    return message


def decode_message(raw: bytes) -> dict:
    """Decode a frame and reject unanswerable requests before admission.

    An id must leave room for an error reply, not only an unusually compact
    input. Otherwise a valid-sized request could execute yet be unreportable.
    """
    if len(raw) - int(raw.endswith(b'\n')) > MAX_MESSAGE_BYTES:
        raise MessageTooLarge()
    try:
        message = json.loads(raw.decode('utf-8'))
    except (ValueError, RecursionError) as error:
        raise TransportError(str(error), code=PARSE_ERROR) from error
    if not isinstance(message, dict):
        raise TransportError('not a JSON-RPC object', code=INVALID_REQUEST)
    request_id = message.get('id')
    if message.get('method') is not None and request_id is not None:
        probe = error_response(request_id, {
            'code': INTERNAL_ERROR, 'message': str(MessageTooLarge())})
        if probe['id'] is None:
            raise TransportError(
                'ACP request identifier cannot fit a bounded response', code=INVALID_REQUEST)
    return message


def encoded_messages(message: dict):
    """Split only append-defined text updates; never split structured snapshots.

    Replay, live events and local-command output share this wire policy. Six
    bytes per character bounds JSON escapes as well as four-byte UTF-8, leaving
    the measured envelope intact. Callers' dictionaries/text remain unchanged.
    """
    params = message.get('params')
    update = params.get('update') if isinstance(params, dict) else None
    content = update.get('content') if isinstance(update, dict) else None
    text = content.get('text') if isinstance(content, dict) else None
    if (message.get('method') != 'session/update' or 'id' in message
            or not isinstance(params, dict) or not isinstance(update, dict)
            or not isinstance(content, dict) or not isinstance(text, str) or content.get('type') != 'text'
            or update.get('sessionUpdate') not in [
                'user_message_chunk', 'agent_message_chunk', 'agent_thought_chunk']):
        yield encode_message(message)
        return
    packet, params, update, content = dict(message), dict(params), dict(update), dict(content)
    packet['params'], params['update'], update['content'] = params, update, content
    content['text'] = ''
    envelope = len(encode_message(packet)) - 1
    room = MAX_MESSAGE_BYTES - envelope
    if text and room == 0:
        raise MessageTooLarge()
    count = max(1, room // 6)
    if not text:
        yield encode_message(packet)
    for start in range(0, len(text), count):
        content['text'] = text[start:start + count]
        yield encode_message(packet)


class AsyncFdLineReader:
    """Read lines from a Unix fd through event-loop readiness notifications.

    ``connect_read_pipe`` changes the underlying open file description to
    nonblocking mode. That is undesirable for stdin because a terminal may
    share its open file description with stdout. ``add_reader`` lets the loop
    wait for readability without changing fd flags; the callback performs one
    bounded read only after the kernel reports that it cannot block.

    Only one ``readline`` call may be active. Leaving the fd unregistered
    between calls preserves kernel backpressure while the consumer processes
    the previous ACP message.
    """

    def __init__(self, fd: int, chunk_size: int = 64 * 1024):
        self.fd = fd
        self.chunk_size = chunk_size
        self._reading = False
        self._threaded = None
        # Accepted raw bytes outlive a cancelled wakeup or reader-thread join.
        # Framing, including partial prefixes and EOF, has one shared owner.
        self._chunks = asyncio.Queue(maxsize=1)
        self._reader = BufferedReader(self._receive, limit=MAX_MESSAGE_BYTES)

    async def readline(self) -> bytes:
        if self._reading:
            raise RuntimeError("concurrent reads from one ACP fd")
        self._reading = True
        try:
            return await self._reader.readline()
        except ValueError as error:
            # A limit failure is terminal: retrying the same retained buffer
            # cannot make progress, and a native producer may be backpressured.
            await finish_cleanup(self.aclose(), preserve_failure=True)
            raise MessageTooLarge() from error
        finally:
            self._reading = False

    async def _receive(self):
        if sys.platform == "win32":
            return await self._receive_threaded()
        if self._chunks.empty():
            loop = asyncio.get_running_loop()
            ready = loop.create_future()

            def on_readable():
                if ready.done():
                    loop.remove_reader(self.fd)
                    return
                try:
                    chunk = os.read(self.fd, self.chunk_size)
                except BlockingIOError:
                    return
                except OSError as error:
                    chunk = error
                # Queue before waking: cancellation can discard a Future result
                # after this read has already consumed bytes from the borrowed fd.
                self._chunks.put_nowait(chunk)
                loop.remove_reader(self.fd)
                ready.set_result(None)

            loop.add_reader(self.fd, on_readable)
            try:
                await ready
            finally:
                loop.remove_reader(self.fd)
        chunk = self._chunks.get_nowait()
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    async def _receive_threaded(self) -> bytes:
        """The proactor cannot register synchronous pipe/console handles.

        Keep the existing HandleReader acquisition and physical join; only byte
        framing is shared with native IPC and subprocess protocol readers.
        """
        loop = asyncio.get_running_loop()
        if self._threaded is None and self._chunks.empty():
            self._threaded = handle_reader.HandleReader(
                self.fd, loop, self._chunks, eof_sentinel=True)
            try:
                self._threaded.start()
            except BaseException:
                await finish_cleanup(self.aclose(), preserve_failure=True)
                raise
        try:
            chunk = await self._chunks.get()
            if isinstance(chunk, Exception):
                raise chunk
            return chunk
        except BaseException:
            # Cancelled or failed. The thread and stop-event handle remain ours
            # until their actual join, not merely until their queue is readable.
            await finish_cleanup(self.aclose(), preserve_failure=True)
            raise

    async def aclose(self) -> None:
        """Join a started reader without discarding bytes it already delivered."""
        reader = self._threaded
        if reader is not None:
            try:
                await reader.aclose()
            finally:
                if reader.closed and self._threaded is reader:
                    self._threaded = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        line = await self.readline()
        if not line:
            await self.aclose()
            raise StopAsyncIteration
        return line


def make_writer(fd: int):
    """Write checked JSON frames; only append-defined text can span messages.

    A frame is fully size-checked before its first byte, so callers can report
    an oversized structured result without leaving a partial JSON line behind.
    """
    stream = os.fdopen(os.dup(fd), "w", encoding="utf-8", buffering=1)

    def write(message: dict) -> None:
        for encoded in encoded_messages(message):
            stream.write(encoded.decode('utf-8'))
            stream.flush()

    return write


def response(request_id, result=None, error=None) -> dict:
    message = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    return message


def notification(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "method": method, "params": params}


def request(request_id, method: str, params: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }


PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def quarantine_stdout(null_fd: int | None = None) -> None:
    """Reserve fd 1 for the protocol; everything else writes devnull.

    The original fd 1 is dup'd before being replaced, so the protocol
    writer keeps working; code that prints to stdout (ours or a
    library's) silently discards instead of corrupting the message
    stream.  Stderr stays untouched: it is ACP's log channel.
    """
    # A contained Windows worker borrows its broker-opened descriptor. The
    # caller retains ownership; only fd 1's duplicate survives redirection.
    if null_fd is None:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
        finally:
            os.close(devnull)
    else:
        os.dup2(null_fd, 1, inheritable=False)
    # Replace the Python-level objects too, or print() would keep its old
    # buffer into the (now devnull) fd 1 while flush ordering gets strange.
    sys.stdout = os.fdopen(1, "w", encoding="utf-8", buffering=1)
    sys.__stdout__ = sys.stdout
