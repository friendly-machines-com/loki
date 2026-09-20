"""Contained ACP launches with asyncio-owned Windows pipe transports.

Only the unrestricted front calls this module. There is no generic process
factory or uncontained fallback. Worker stdio remains synchronous; only the
front's ends are overlapped. Credentials and owner channels are unchanged.

The pipe layout follows CPython 3.14.7 Lib/asyncio/windows_utils.py: duplex
stdin with an overlapped front writer (the Proactor also probes it for EOF),
and inbound stdout with an overlapped front reader. Unlike its general helper,
creation here specifies a private DACL and rejects remote clients. Public loop
pipe APIs own all pending I/O; Loki never owns OVERLAPPED buffers or IOCP state.
"""

import asyncio
import ctypes
import os
import sys
import uuid
from ctypes import wintypes

from . import host_ipc, windows_api as api, windows_runtime as runtime


class PipeHandle:
    """One owned native HANDLE, not a CRT descriptor, for asyncio pipe APIs.

    Matches the small file-like interface used by CPython's PipeHandle;
    ownership transfers to the transport on connection_made.
    """

    def __init__(self, handle):
        self._handle = handle

    def fileno(self):
        if self._handle is None:
            raise ValueError("pipe is closed")
        return self._handle

    def close(self):
        handle, self._handle = self._handle, None
        if handle is not None:
            api.close_handle(handle)


def _stdio_pair(*, stdin):
    """Return (front, child), connected before any process receives them."""
    import _winapi

    convert = api.bind(
        "advapi32", "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        wintypes.BOOL, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    free = api.bind("kernel32", "LocalFree", ctypes.c_void_p, ctypes.c_void_p)
    descriptor = ctypes.c_void_p()
    # No Everyone, anonymous, or package grant: the front connects both ends.
    # The child receives existing handles, not permission to open the name.
    sddl = f"D:P(A;;GA;;;SY)(A;;GA;;;{api.current_user_sid()})"
    if not convert(sddl, 1, ctypes.byref(descriptor), None):
        raise api.WindowsApiError("Cannot create worker pipe security descriptor")
    server = client = None
    try:
        attributes = api.SecurityAttributes(
            ctypes.sizeof(api.SecurityAttributes), descriptor.value, 0)
        name = r"\\.\pipe\loki-worker-" + uuid.uuid4().hex
        mode = _winapi.PIPE_ACCESS_DUPLEX if stdin else _winapi.PIPE_ACCESS_INBOUND
        mode |= _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE
        if not stdin:
            mode |= _winapi.FILE_FLAG_OVERLAPPED
        server = _winapi.CreateNamedPipe(
            name, mode, _winapi.PIPE_WAIT | 0x8,  # PIPE_REJECT_REMOTE_CLIENTS
            1, 8192 if stdin else 0, 8192, 0, ctypes.addressof(attributes))
        access = _winapi.GENERIC_WRITE | (_winapi.GENERIC_READ if stdin else 0)
        client = _winapi.CreateFile(
            name, access, 0, 0, _winapi.OPEN_EXISTING,
            _winapi.FILE_FLAG_OVERLAPPED if stdin else 0, 0)
        # The local client is already connected, as in windows_utils.pipe.
        # With one first-instance server a competing connection makes this
        # creation fail; there is no accept/reconnect loop or data exchange.
        connection = _winapi.ConnectNamedPipe(server, overlapped=True)
        connection.GetOverlappedResult(True)
        front, child = (client, server) if stdin else (server, client)
        api.set_handle_information(child, api.HANDLE_FLAG_INHERIT,
                                   api.HANDLE_FLAG_INHERIT)
        return front, child
    except BaseException:
        for handle in (server, client):
            if handle is not None:
                api.close_handle(handle)
        raise
    finally:
        free(descriptor)


def _stdio():
    writer, child_read = _stdio_pair(stdin=True)
    try:
        reader, child_write = _stdio_pair(stdin=False)
    except BaseException:
        api.close_handle(writer)
        api.close_handle(child_read)
        raise
    return (reader, writer), (child_read, child_write)


class _Reader(asyncio.Protocol):
    def __init__(self):
        self.stream = asyncio.StreamReader()
        self.transport = None
        self.closed = asyncio.get_running_loop().create_future()

    def connection_made(self, transport):
        self.transport = transport
        self.stream.set_transport(transport)

    def data_received(self, data):
        self.stream.feed_data(data)

    def eof_received(self):
        self.stream.feed_eof()
        return False

    def connection_lost(self, exc):
        if exc is None:
            self.stream.feed_eof()
        else:
            self.stream.set_exception(exc)
        if not self.closed.done():
            self.closed.set_result(None)


class _Writer(asyncio.Protocol):
    """Public protocol flow control, with the StreamWriter-shaped ACP surface.

    drain means backpressure relief, not a promise of a completed OS write.
    wait_closed observes connection_lost after graceful flush or abort.
    """

    def __init__(self):
        self.transport = None
        self._error = None
        self._ready = asyncio.Event()
        self._ready.set()
        self.closed = asyncio.get_running_loop().create_future()

    def connection_made(self, transport):
        self.transport = transport

    def pause_writing(self):
        self._ready.clear()

    def resume_writing(self):
        self._ready.set()

    def connection_lost(self, exc):
        self._error = exc
        self._ready.set()
        if not self.closed.done():
            self.closed.set_result(None)

    def write(self, data):
        if self.transport is None or self.transport.is_closing():
            raise BrokenPipeError("worker stdin is closed")
        self.transport.write(data)

    async def drain(self):
        await self._ready.wait()
        if self._error is not None:
            raise self._error
        if self.closed.done():
            raise BrokenPipeError("worker stdin is closed")

    def close(self):
        if self.transport is not None:
            self.transport.close()

    async def wait_closed(self):
        await asyncio.shield(self.closed)
        if self._error is not None:
            raise self._error


class _Streams:
    """Own front endpoints; public asyncio transports adopt them individually."""

    def __init__(self, reader, writer):
        self._source = _Reader()
        self.stdin = _Writer()
        self.stdout = self._source.stream
        self._pipes = (PipeHandle(reader), PipeHandle(writer))
        self._connecting = None
        self._close_task = None

    async def connect(self):
        loop = asyncio.get_running_loop()
        for connect, protocol, pipe in (
                (loop.connect_read_pipe, self._source, self._pipes[0]),
                (loop.connect_write_pipe, self.stdin, self._pipes[1])):
            # Retain attachment through caller cancellation. Cleanup first
            # settles it, then releases via its transport or its raw owner.
            self._connecting = asyncio.create_task(connect(lambda: protocol, pipe))
            await asyncio.shield(self._connecting)

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._dispose())
        await asyncio.shield(self._close_task)

    async def _dispose(self):
        if self._connecting is not None:
            await asyncio.gather(self._connecting, return_exceptions=True)
        waits, errors = [], []
        for protocol, pipe in zip((self._source, self.stdin), self._pipes):
            try:
                if protocol.transport is None:
                    pipe.close()
                else:
                    if protocol is self.stdin:
                        protocol.transport.abort()
                    else:
                        protocol.transport.close()
                    waits.append(protocol.closed)
            except Exception as error:
                errors.append(error)
        if waits:
            try:
                await asyncio.wait_for(asyncio.shield(asyncio.gather(*waits)), 2)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Worker pipe cleanup failed", errors)


class Worker:
    """A contained process and its asyncio stdio, with one cleanup owner."""

    def __init__(self, process, streams):
        self._process = process
        self._streams = streams
        self.stdin, self.stdout = streams.stdin, streams.stdout
        self._close_task = None
        self._exit_task = asyncio.create_task(self._observe_exit())

    @property
    def returncode(self):
        return self._process.returncode

    async def _observe_exit(self):
        try:
            code = await self._process.wait()
            # Descendants holding stdout must not hide root death. The job is
            # owned here, and its lifetime ends when its root runtime does.
            self._process.close_job()
            return code
        except Exception as error:
            self.stdout.set_exception(error)
            raise

    async def wait(self):
        return await asyncio.shield(self._exit_task)

    def terminate(self):
        if self.returncode is None:
            self._process.terminate()

    kill = terminate

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._dispose())
        await asyncio.shield(self._close_task)

    async def _dispose(self):
        try:
            self._process.close_job()
            await asyncio.wait_for(asyncio.shield(self._exit_task), 2)
        finally:
            if not self._exit_task.done():
                self._exit_task.cancel()
            await asyncio.gather(self._exit_task, return_exceptions=True)
            try:
                await self._streams.close()
            finally:
                self._process.close()


async def start_worker(cwd, environment, delegation):
    """The only launch path here: gated, token-verified, job-contained."""
    workspace = runtime.required_workspace(cwd)
    front, child = _stdio()
    streams = process = None
    try:
        streams = _Streams(*front)
        await streams.connect()
        inherited = [*host_ipc.handles(delegation.owner_child),
                     *host_ipc.handles(delegation.credential_child)]
        with runtime.worker_stdout_null() as null_handle:
            process = runtime.launch(
                sys.argv[0],
                ["--worker", *delegation.child_arguments(),
                 "--stdout-null-handle", str(null_handle)],
                environment, workspace, [*inherited, null_handle],
                stdio=child, current_directory=os.getcwd())
        return Worker(process, streams)
    except BaseException:
        async def unwind():
            try:
                if process is not None:
                    process.close_job()
                    await asyncio.wait_for(process.wait(), 2)
            finally:
                try:
                    if streams is not None:
                        await streams.close()
                    else:
                        for handle in front:
                            api.close_handle(handle)
                finally:
                    if process is not None:
                        process.close()
        await asyncio.shield(asyncio.create_task(unwind()))
        raise
    finally:
        # These are the front's copies, on both launch success and failure.
        # Keeping either copy alive defeats EOF/broken-pipe detection.
        for handle in child:
            api.close_handle(handle)
