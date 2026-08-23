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


class TransportError(Exception):
    def __init__(self, message: str, *, code: int = -32603):
        super().__init__(message)
        self.code = code


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
        self._buffer = bytearray()
        self._eof = False
        self._reading = False

    def _take_line(self) -> bytes | None:
        newline = self._buffer.find(b"\n")
        if newline >= 0:
            end = newline + 1
            line = bytes(self._buffer[:end])
            del self._buffer[:end]
            return line
        if self._eof:
            line = bytes(self._buffer)
            self._buffer.clear()
            return line
        return None

    async def readline(self) -> bytes:
        if self._reading:
            raise RuntimeError("concurrent reads from one ACP fd")
        self._reading = True
        try:
            line = self._take_line()
            if line is not None:
                return line

            loop = asyncio.get_running_loop()
            ready = loop.create_future()

            def finish(error=None):
                loop.remove_reader(self.fd)
                if ready.done():
                    return
                if error is None:
                    ready.set_result(None)
                else:
                    ready.set_exception(error)

            def on_readable():
                if ready.done():
                    loop.remove_reader(self.fd)
                    return
                try:
                    chunk = os.read(self.fd, self.chunk_size)
                except BlockingIOError:
                    return
                except OSError as error:
                    finish(error)
                    return
                if not chunk:
                    self._eof = True
                    finish()
                    return
                self._buffer.extend(chunk)
                if b"\n" in chunk:
                    finish()

            loop.add_reader(self.fd, on_readable)
            try:
                await ready
            finally:
                loop.remove_reader(self.fd)
            line = self._take_line()
            if line is None:
                raise RuntimeError("ACP fd became unreadable without a line")
            return line
        finally:
            self._reading = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        line = await self.readline()
        if not line:
            raise StopAsyncIteration
        return line


def make_writer(fd: int):
    """Line-buffered writer for one JSON-RPC message per line.

    Payload and delimiter are separate writes: no concatenation means no
    second copy of a potentially large message, and the newline doubles
    as the flush marker.
    """
    stream = os.fdopen(os.dup(fd), "w", encoding="utf-8", buffering=1)

    def write(message: dict) -> None:
        stream.write(json.dumps(message, ensure_ascii=False))
        stream.write("\n")
        stream.flush()

    return write


def read_messages(fin):
    """Yield parsed JSON-RPC messages, one per line; stop at EOF."""
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            raise TransportError(f"line is not JSON: {error}") from error
        if not isinstance(message, dict):
            raise TransportError("line is not a JSON object")
        yield message


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
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def quarantine_stdout() -> None:
    """Reserve fd 1 for the protocol; everything else writes devnull.

    The original fd 1 is dup'd before being replaced, so the protocol
    writer keeps working; code that prints to stdout (ours or a
    library's) silently discards instead of corrupting the message
    stream.  Stderr stays untouched: it is ACP's log channel.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    # Replace the Python-level objects too, or print() would keep its old
    # buffer into the (now devnull) fd 1 while flush ordering gets strange.
    sys.stdout = os.fdopen(1, "w", encoding="utf-8", buffering=1)
    sys.__stdout__ = sys.stdout
