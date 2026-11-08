"""Owned subprocess byte streams using documented asyncio protocol callbacks.

Public subprocess transports expose pipe callbacks, not a supported constructor
for asyncio's Process/StreamReaderProtocol internals. Buffer framing is shared
with inherited native IPC instead of maintaining another decoder/stream owner.
Network connections still use asyncio's public stream factories unchanged.
"""

import asyncio
from contextlib import ExitStack

from .lifetimes import finish_cleanup


class BufferedReader:
    """Frame bytes from one asynchronous receiver; EOF and partial data stay here."""

    def __init__(self, receive, limit=65536):
        self._receive = receive
        self._limit = limit
        self._buffer = bytearray()
        self._eof = False

    async def _fill(self):
        data = await self._receive()
        if data:
            self._buffer.extend(data)
        else:
            self._eof = True

    def at_eof(self):
        return self._eof and not self._buffer

    async def read(self, size=-1):
        if not size:
            return b''
        if size < 0:
            while not self._eof:
                await self._fill()
            size = len(self._buffer)
        elif not self._buffer and not self._eof:
            await self._fill()
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result

    async def readline(self):
        while True:
            end = self._buffer.find(b'\n')
            if end >= 0:
                if self._limit is not None and end > self._limit:
                    raise ValueError('pipe line exceeds size limit')
                return await self.read(end + 1)
            if self._limit is not None and len(self._buffer) > self._limit:
                raise ValueError('pipe line exceeds size limit')
            if self._eof:
                return await self.read(len(self._buffer))
            await self._fill()


def _closed(future, error):
    if future.done():
        return
    if error is None:
        future.set_result(None)
    else:
        future.set_exception(error)
        # A caller need not await every individual pipe. Retrieval prevents an
        # orphan diagnostic without changing what an actual waiter receives.
        future.exception()


class _PipeWriter:
    """The one stdin transport and its public protocol flow/close signals."""

    def __init__(self, transport, protocol):
        self.transport = transport
        self._protocol = protocol

    def write(self, data):
        self.transport.write(data)

    async def drain(self):
        if self.transport.is_closing():
            await asyncio.sleep(0)
        await self._protocol._writable.wait()
        if self._protocol._stdin_closed.done():
            self._protocol._stdin_closed.result()
            raise ConnectionResetError('pipe is closed')

    def is_closing(self):
        return self.transport.is_closing()

    def close(self):
        self.transport.close()

    async def wait_closed(self):
        await asyncio.shield(self._protocol._stdin_closed)


class ProcessProtocol(asyncio.SubprocessProtocol):
    """One leader signal plus independently owned stdio completion.

    A low-level public subprocess transport is required so cancellation can
    abort pipe ownership without reading private Process._transport fields.
    Pausing at each delivered chunk bounds prefetch without another byte-count
    cache; the buffered receiver resumes its actual pipe when consuming it.
    """

    def __init__(self, limit=65536):
        loop = asyncio.get_running_loop()
        self._limit = limit
        self._transport = None
        self._read_pipes = {}
        self.stdin = self.stdout = self.stderr = None
        self.exited = asyncio.Event()
        self.closed = loop.create_future()
        self._stdin_closed = loop.create_future()
        self._writable = asyncio.Event()
        self._writable.set()

    def connection_made(self, transport):
        self._transport = transport
        for fd, name in [[1, 'stdout'], [2, 'stderr']]:
            pipe = transport.get_pipe_transport(fd)
            if pipe is None:
                continue
            chunks = asyncio.Queue()
            self._read_pipes[fd] = chunks

            async def receive(pipe=pipe, chunks=chunks):
                data = await chunks.get()
                if isinstance(data, BaseException):
                    raise data
                if data and not pipe.is_closing():
                    pipe.resume_reading()
                return data

            setattr(self, name, BufferedReader(receive, self._limit))
        pipe = transport.get_pipe_transport(0)
        if pipe is not None:
            self.stdin = _PipeWriter(pipe, self)

    def pipe_data_received(self, fd, data):
        chunks = self._read_pipes.get(fd)
        if chunks is not None:
            self._transport.get_pipe_transport(fd).pause_reading()
            chunks.put_nowait(data)

    def pipe_connection_lost(self, fd, error):
        if fd == 0:
            _closed(self._stdin_closed, error)
            self._writable.set()
        else:
            chunks = self._read_pipes.pop(fd, None)
            if chunks is not None:
                chunks.put_nowait(error if error is not None else b'')
        self._finish()

    def process_exited(self):
        self.exited.set()
        self._finish()

    def _finish(self):
        if self.exited.is_set() and not self._read_pipes and self._transport is not None:
            self._transport.close()

    def pause_writing(self):
        self._writable.clear()

    def resume_writing(self):
        self._writable.set()

    def connection_lost(self, error):
        _closed(self.closed, error)
        _closed(self._stdin_closed, error)
        self._writable.set()


class Process:
    """A public transport/protocol pair, with leader wait separate from pipe EOF."""

    def __init__(self, transport, protocol):
        self._transport = transport
        self._protocol = protocol
        self.stdin = protocol.stdin
        self.stdout = protocol.stdout
        self.stderr = protocol.stderr
        self.pid = transport.get_pid()

    @property
    def returncode(self):
        return self._transport.get_returncode()

    async def wait(self):
        await self._protocol.exited.wait()
        return self.returncode

    def send_signal(self, signal):
        self._transport.send_signal(signal)

    def terminate(self):
        self._transport.terminate()

    def kill(self):
        self._transport.kill()

    def close(self):
        # Explicit retirement discards pending stdin and descendant-held output;
        # normal completion retains the transport's ordinary flush semantics.
        try:
            with ExitStack() as cleanup:
                for fd in [2, 1, 0]:
                    pipe = self._transport.get_pipe_transport(fd)
                    if pipe is not None and not pipe.is_closing():
                        cleanup.callback(pipe.abort if fd == 0 else pipe.close)
        finally:
            self._transport.close()

    async def aclose(self):
        try:
            self.close()
        finally:
            await finish_cleanup(self._protocol.closed)
