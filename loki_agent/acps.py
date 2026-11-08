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
        self._reading = False
        self._threaded = None
        # Accepted raw bytes outlive a cancelled wakeup or reader-thread join.
        # Framing, including partial prefixes and EOF, has one shared owner.
        self._chunks = asyncio.Queue(maxsize=1)
        self._reader = BufferedReader(self._receive, limit=None)

    async def readline(self) -> bytes:
        if self._reading:
            raise RuntimeError("concurrent reads from one ACP fd")
        self._reading = True
        try:
            return await self._reader.readline()
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
