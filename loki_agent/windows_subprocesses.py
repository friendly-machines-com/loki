"""Contained worker processes over overlapped named pipes.

The stream and orchestration layer below is copied from CPython 3.14.7 --
``Lib/asyncio/base_subprocess.py``, ``Lib/asyncio/subprocess.py``,
``Lib/asyncio/streams.py`` (``FlowControlMixin``) and
``Lib/asyncio/windows_utils.py`` -- and adapted, so that Loki depends only on
public asyncio plus its own Win32 bindings instead of private asyncio
internals.  CPython is distributed under the PSF licence; the copied code
keeps its structure and comments.  Intentional differences:

* ``pipe`` names the pipe with ``os.urandom`` and retries, passes an explicit
  security descriptor (SYSTEM and the current user only) and sets
  ``PIPE_REJECT_REMOTE_CLIENTS``.  The reference implementation passes NULL
  security attributes, whose default DACL grants Everyone read access.
* The client calls ``ConnectNamedPipe`` expecting ``ERROR_PIPE_CONNECTED``:
  both ends are created and connected here, in this process, so no wait for a
  connection is needed.
* ``_start`` runs Loki's contained launcher (AppContainer, token check, job)
  instead of ``windows_utils.Popen``; child ends cross as native handles.
* Exit is observed by waiting on the process through Loki's own non-blocking
  wait, not by registering the handle with the loop's completion port.
* ``_process_exited`` wakes ``_wait`` waiters immediately (the 3.13+ behavior),
  so a descendant holding the worker's stdout cannot hang ``Process.wait()``.
"""

from __future__ import annotations

import asyncio
import collections
import ctypes
import subprocess
import sys
import uuid
import warnings
from ctypes import wintypes

from . import windows_api as api
from . import windows_runtime

PIPE = subprocess.PIPE

PIPE_ACCESS_DUPLEX = 0x00000003
PIPE_ACCESS_INBOUND = 0x00000001
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
PIPE_WAIT = 0x00000000
PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
ERROR_PIPE_BUSY = 231
ERROR_PIPE_CONNECTED = 535
ERROR_ACCESS_DENIED = 5
INVALID_HANDLE_VALUE = -1
BUFSIZE = 8192
_MAX_NAME_ATTEMPTS = 20


def pipe(*, duplex=False, overlapped=(True, True), bufsize=BUFSIZE):
    """Like ``os.pipe()`` but with overlapped support and using handles.

    Returns ``(h1, h2)``: ``h1`` is the ``CreateNamedPipe`` end and ``h2`` the
    connecting client.  ``overlapped`` selects which is opened for overlapped
    I/O; the child's end is deliberately synchronous, as in CPython.
    """
    convert = api.bind(
        "advapi32", "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        wintypes.BOOL, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    free = api.bind("kernel32", "LocalFree", ctypes.c_void_p,
                    ctypes.c_void_p)
    create_pipe = api.bind(
        "kernel32", "CreateNamedPipeW", ctypes.c_void_p, wintypes.LPCWSTR,
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p)
    connect = api.bind("kernel32", "ConnectNamedPipe", wintypes.BOOL,
                       ctypes.c_void_p, ctypes.c_void_p)

    if duplex:
        openmode = PIPE_ACCESS_DUPLEX
        access = GENERIC_READ | GENERIC_WRITE
        obsize, ibsize = bufsize, bufsize
    else:
        openmode = PIPE_ACCESS_INBOUND
        access = GENERIC_WRITE
        obsize, ibsize = 0, bufsize
    openmode |= FILE_FLAG_FIRST_PIPE_INSTANCE
    if overlapped[0]:
        openmode |= FILE_FLAG_OVERLAPPED
    client_flags = FILE_FLAG_OVERLAPPED if overlapped[1] else 0

    # The pipe name is not a control: the ends are connected here and only the
    # child's inherited handle reaches the child.  The DACL nevertheless keeps
    # the default one -- which grants Everyone read -- from applying while the
    # name is briefly visible in the namespace.
    sddl = f"D:P(A;;GA;;;SY)(A;;GA;;;{api.current_user_sid()})"
    descriptor = ctypes.c_void_p()
    if not convert(sddl, 1, ctypes.byref(descriptor), None):
        raise api.WindowsApiError("cannot build the worker pipe descriptor")
    server = client = None
    try:
        attributes = api.SecurityAttributes(
            ctypes.sizeof(api.SecurityAttributes), descriptor.value, 0)
        for attempt in range(_MAX_NAME_ATTEMPTS):
            address = r"\\.\pipe\loki-worker-" + uuid.uuid4().hex
            handle = create_pipe(address, openmode, PIPE_WAIT
                                 | PIPE_REJECT_REMOTE_CLIENTS, 1, obsize,
                                 ibsize, 0, ctypes.byref(attributes))
            if handle and handle != INVALID_HANDLE_VALUE:
                server = handle
                break
            status = ctypes.get_last_error()
            # Another process holding this name cannot be distinguished from
            # a name collision here; try another random name either way.
            if (attempt == _MAX_NAME_ATTEMPTS - 1
                    or status not in (ERROR_PIPE_BUSY, ERROR_ACCESS_DENIED)):
                raise api.WindowsApiError("CreateNamedPipeW failed",
                                          status=status)
        try:
            client = api.open_with_access(address, access,
                                          creation=OPEN_EXISTING,
                                          flags=client_flags,
                                          share_mode=0)
        except api.WindowsApiError:
            api.close_handle(server)
            server = None
            raise
        # The client connected first, so this reports ERROR_PIPE_CONNECTED.
        if not connect(server, None):
            status = ctypes.get_last_error()
            if status != ERROR_PIPE_CONNECTED:
                raise api.WindowsApiError("ConnectNamedPipe failed",
                                          status=status)
        return server, client
    except BaseException:
        for handle in (server, client):
            if handle is not None:
                api.close_handle(handle)
        raise
    finally:
        free(descriptor)


class PipeHandle:
    """Wrapper for an overlapped pipe handle which is vaguely file-like."""

    def __init__(self, handle):
        self._handle = handle

    def __repr__(self):
        handle = (f'handle={self._handle!r}' if self._handle is not None
                  else 'closed')
        return f'<{self.__class__.__name__} {handle}>'

    @property
    def handle(self):
        return self._handle

    def fileno(self):
        """The raw Windows handle: what the proactor's pipe calls expect."""
        if self._handle is None:
            raise ValueError("I/O operation on closed pipe")
        return self._handle

    def close(self):
        handle, self._handle = self._handle, None
        if handle is not None:
            api.close_handle(handle)

    def __del__(self, _warn=warnings.warn):
        if self._handle is not None:
            _warn(f"unclosed {self!r}", ResourceWarning, source=self)
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


# -- copied from Lib/asyncio/streams.py -----------------------------------


class FlowControlMixin(asyncio.Protocol):
    """Reusable flow control logic for ``StreamWriter.drain()``.

    This implements the protocol methods ``pause_writing()``,
    ``resume_writing()`` and ``connection_lost()``.  If the subclass overrides
    these it must call the super methods.  ``StreamWriter.drain()`` must wait
    for ``_drain_helper()``.
    """

    def __init__(self, loop=None):
        self._loop = loop
        self._paused = False
        self._drain_waiters = collections.deque()
        self._connection_lost = False

    def pause_writing(self):
        assert not self._paused
        self._paused = True

    def resume_writing(self):
        assert self._paused
        self._paused = False
        for waiter in self._drain_waiters:
            if not waiter.done():
                waiter.set_result(None)

    def connection_lost(self, exc):
        self._connection_lost = True
        # Wake up the writer(s) if currently paused.
        if not self._paused:
            return
        for waiter in self._drain_waiters:
            if not waiter.done():
                if exc is None:
                    waiter.set_result(None)
                else:
                    waiter.set_exception(exc)

    async def _drain_helper(self):
        if self._connection_lost:
            raise ConnectionResetError('Connection lost')
        if not self._paused:
            return
        waiter = self._loop.create_future()
        self._drain_waiters.append(waiter)
        try:
            await waiter
        finally:
            self._drain_waiters.remove(waiter)

    def _get_close_waiter(self, stream):
        raise NotImplementedError


# -- copied from Lib/asyncio/base_subprocess.py ---------------------------


class WriteSubprocessPipeProto(asyncio.BaseProtocol):
    def __init__(self, proc, fd):
        self.proc = proc
        self.fd = fd
        self.pipe = None
        self.disconnected = False

    def connection_made(self, transport):
        self.pipe = transport

    def __repr__(self):
        return f'<{self.__class__.__name__} fd={self.fd} pipe={self.pipe!r}>'

    def connection_lost(self, exc):
        self.disconnected = True
        self.proc._pipe_connection_lost(self.fd, exc)
        self.proc = None

    def pause_writing(self):
        self.proc._protocol.pause_writing()

    def resume_writing(self):
        self.proc._protocol.resume_writing()


class ReadSubprocessPipeProto(WriteSubprocessPipeProto, asyncio.Protocol):
    def data_received(self, data):
        self.proc._pipe_data_received(self.fd, data)


class BaseSubprocessTransport(asyncio.SubprocessTransport):
    """The copied pipe-orchestration base; see the module docstring."""

    def __init__(self, loop, protocol, args, shell,
                 stdin, stdout, stderr, bufsize,
                 waiter=None, extra=None, **kwargs):
        super().__init__(extra)
        self._closed = False
        self._protocol = protocol
        self._loop = loop
        self._proc = None
        self._pid = None
        self._returncode = None
        self._exit_waiters = set()
        self._pending_calls = collections.deque()
        self._pipes = {}
        self._finished = False

        if stdin == PIPE:
            self._pipes[0] = None
        if stdout == PIPE:
            self._pipes[1] = None
        if stderr == PIPE:
            self._pipes[2] = None

        try:
            self._start(args=args, shell=shell, stdin=stdin, stdout=stdout,
                        stderr=stderr, bufsize=bufsize, **kwargs)
        except BaseException:
            self.close()
            raise

        self._pid = self._proc.pid
        self._extra['subprocess'] = self._proc
        self._connect_task = self._loop.create_task(self._connect_pipes(waiter))

    def _start(self, args, shell, stdin, stdout, stderr, bufsize, **kwargs):
        raise NotImplementedError

    def set_protocol(self, protocol):
        self._protocol = protocol

    def get_protocol(self):
        return self._protocol

    def is_closing(self):
        return self._closed

    def close(self):
        if self._closed:
            return
        self._closed = True

        for proto in self._pipes.values():
            if proto is None:
                continue
            if self._loop is not None and not self._loop.is_closed():
                proto.pipe.close()

        if (self._proc is not None
                and self._returncode is None
                and self._proc.poll() is None):
            try:
                self._proc.kill()
            except (ProcessLookupError, PermissionError):
                pass

    def __del__(self, _warn=warnings.warn):
        if not self._closed:
            _warn(f"unclosed transport {self!r}", ResourceWarning, source=self)
            self.close()

    def get_pid(self):
        return self._pid

    def get_returncode(self):
        return self._returncode

    def get_pipe_transport(self, fd):
        if fd in self._pipes:
            return self._pipes[fd].pipe
        return None

    def _check_proc(self):
        if self._proc is None:
            raise ProcessLookupError()

    def send_signal(self, sig):
        self._check_proc()
        self._proc.send_signal(sig)

    def terminate(self):
        self._check_proc()
        self._proc.terminate()

    def kill(self):
        self._check_proc()
        self._proc.kill()

    async def _connect_pipes(self, waiter):
        try:
            proc = self._proc
            loop = self._loop

            if proc.stdin is not None:
                _, pipe_transport = await loop.connect_write_pipe(
                    lambda: WriteSubprocessPipeProto(self, 0), proc.stdin)
                self._pipes[0] = pipe_transport

            if proc.stdout is not None:
                _, pipe_transport = await loop.connect_read_pipe(
                    lambda: ReadSubprocessPipeProto(self, 1), proc.stdout)
                self._pipes[1] = pipe_transport

            if proc.stderr is not None:
                _, pipe_transport = await loop.connect_read_pipe(
                    lambda: ReadSubprocessPipeProto(self, 2), proc.stderr)
                self._pipes[2] = pipe_transport

            assert self._pending_calls is not None
            loop.call_soon(self._protocol.connection_made, self)
            for callback, data in self._pending_calls:
                loop.call_soon(callback, *data)
            self._pending_calls = None
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            # Close any pipes that were already connected before the
            # error/cancellation, so a failed attachment leaks nothing.
            for proto in self._pipes.values():
                if proto is not None:
                    proto.pipe.close()
            for raw_pipe in (proc.stdin, proc.stdout, proc.stderr):
                if raw_pipe is not None:
                    raw_pipe.close()
            if waiter is not None and not waiter.cancelled():
                waiter.set_exception(exc)
        else:
            if waiter is not None and not waiter.cancelled():
                waiter.set_result(None)

    def _call(self, callback, *data):
        if self._pending_calls is not None:
            self._pending_calls.append((callback, data))
        else:
            self._loop.call_soon(callback, *data)

    def _pipe_connection_lost(self, fd, exc):
        self._call(self._protocol.pipe_connection_lost, fd, exc)
        self._try_finish()

    def _pipe_data_received(self, fd, data):
        self._call(self._protocol.pipe_data_received, fd, data)

    def _process_exited(self, returncode):
        assert returncode is not None, returncode
        assert self._returncode is None, self._returncode
        self._returncode = returncode
        if self._proc.returncode is None:
            self._proc.returncode = returncode
        self._call(self._protocol.process_exited)
        # Wake up futures waiting for wait() as soon as the process exits: a
        # descendant holding the worker's stdout must not hide root death.
        for waiter in self._exit_waiters:
            if not waiter.done():
                waiter.set_result(returncode)
        self._exit_waiters = None
        self._try_finish()

    async def _wait(self):
        """Wait until the process exit and return the process return code."""
        if self._returncode is not None:
            return self._returncode
        waiter = self._loop.create_future()
        self._exit_waiters.add(waiter)
        try:
            return await waiter
        finally:
            if self._exit_waiters is not None:
                self._exit_waiters.discard(waiter)

    def _try_finish(self):
        assert not self._finished
        if self._returncode is None:
            return
        if all(pipe_transport is not None and pipe_transport.disconnected
               for pipe_transport in self._pipes.values()):
            self._finished = True
            self._call(self._call_connection_lost, None)

    def _call_connection_lost(self, exc):
        try:
            self._protocol.connection_lost(exc)
        finally:
            self._loop = None
            self._proc = None
            self._protocol = None


# -- copied from Lib/asyncio/subprocess.py --------------------------------


class SubprocessStreamProtocol(FlowControlMixin, asyncio.SubprocessProtocol):
    """Like ``StreamReaderProtocol``, but for a subprocess."""

    def __init__(self, limit, loop):
        super().__init__(loop=loop)
        self._limit = limit
        self.stdin = self.stdout = self.stderr = None
        self._transport = None
        self._process_exited = False
        self._pipe_fds = []
        self._stdin_closed = self._loop.create_future()

    def connection_made(self, transport):
        self._transport = transport

        stdout_transport = transport.get_pipe_transport(1)
        if stdout_transport is not None:
            self.stdout = asyncio.StreamReader(limit=self._limit)
            self.stdout.set_transport(stdout_transport)
            self._pipe_fds.append(1)

        stderr_transport = transport.get_pipe_transport(2)
        if stderr_transport is not None:
            self.stderr = asyncio.StreamReader(limit=self._limit)
            self.stderr.set_transport(stderr_transport)
            self._pipe_fds.append(2)

        stdin_transport = transport.get_pipe_transport(0)
        if stdin_transport is not None:
            self.stdin = asyncio.StreamWriter(stdin_transport,
                                              protocol=self,
                                              reader=None,
                                              loop=self._loop)

    def pipe_data_received(self, fd, data):
        reader = self.stdout if fd == 1 else None
        if reader is not None:
            reader.feed_data(data)

    def pipe_connection_lost(self, fd, exc):
        if fd == 0:
            if self.stdin is not None:
                self.stdin.close()
            self.connection_lost(exc)
            if not self._stdin_closed.done():
                if exc is None:
                    self._stdin_closed.set_result(None)
                else:
                    self._stdin_closed.set_exception(exc)
                    # Calling wait_closed() is not mandatory, so do not log
                    # the traceback when it is not awaited.
                    self._stdin_closed._log_traceback = False
            return
        reader = self.stdout if fd == 1 else None
        if reader is not None:
            if exc is None:
                reader.feed_eof()
            else:
                reader.set_exception(exc)
        if fd in self._pipe_fds:
            self._pipe_fds.remove(fd)
        self._maybe_close_transport()

    def process_exited(self):
        self._process_exited = True
        self._maybe_close_transport()

    def _maybe_close_transport(self):
        if len(self._pipe_fds) == 0 and self._process_exited:
            self._transport.close()
            self._transport = None

    def _get_close_waiter(self, stream):
        if stream is self.stdin:
            return self._stdin_closed
        return None


class Process:
    """The ``asyncio.subprocess.Process`` surface the ACP front relies on."""

    def __init__(self, transport, protocol, loop):
        self._transport = transport
        self._protocol = protocol
        self._loop = loop
        self.stdin = protocol.stdin
        self.stdout = protocol.stdout
        self.stderr = protocol.stderr
        self.pid = transport.get_pid()

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.pid}>'

    @property
    def returncode(self):
        return self._transport.get_returncode()

    async def wait(self):
        return await self._transport._wait()

    def send_signal(self, sig):
        self._transport.send_signal(sig)

    def terminate(self):
        self._transport.terminate()

    def kill(self):
        self._transport.kill()


# -- Loki's contained worker ---------------------------------------------


class ContainedWorkerProcess:
    """The process-shaped object the copied transport drives.

    Owns the native process, its job object and its thread handle.  ``poll``
    is synchronous because the base transport calls it from ``close()``;
    ``wait`` is Loki's non-blocking native wait.
    """

    def __init__(self, contained):
        self._contained = contained
        self.pid = contained.pid
        self.returncode = None
        self.stdin = None
        self.stdout = None
        self.stderr = None

    def poll(self):
        if self.returncode is None and self._contained.returncode is not None:
            self.returncode = self._contained.returncode
        return self.returncode

    async def wait(self):
        code = await self._contained.wait()
        self.returncode = code
        return code

    def send_signal(self, sig):  # Windows has no signals: stop the job.
        self.terminate()

    def terminate(self):
        self._contained.terminate()

    kill = terminate

    def release(self):
        """Release the native handles; idempotent, never under a live wait."""
        if self._contained is not None:
            contained, self._contained = self._contained, None
            contained.close()


class ContainedWorkerTransport(BaseSubprocessTransport):
    """One contained worker: Loki's launcher, the copied pipe orchestration."""

    def __init__(self, loop, protocol, *, workspace, environment,
                 arguments, inherited_handles, current_directory,
                 waiter=None):
        self._exit_task = None
        super().__init__(loop, protocol, args=None, shell=False,
                         stdin=PIPE, stdout=PIPE, stderr=None, bufsize=0,
                         waiter=waiter, extra=None,
                         workspace=workspace, environment=environment,
                         arguments=arguments, inherited_handles=inherited_handles,
                         current_directory=current_directory)

    def _start(self, args, shell, stdin, stdout, stderr, bufsize, **kwargs):
        child_read, front_write = pipe(overlapped=(False, True), duplex=True)
        try:
            front_read, child_write = pipe(overlapped=(True, False))
        except BaseException:
            api.close_handle(child_read)
            api.close_handle(front_write)
            raise
        try:
            api.set_handle_information(
                child_read, api.HandleFlags.HANDLE_FLAG_INHERIT,
                api.HandleFlags.HANDLE_FLAG_INHERIT)
            api.set_handle_information(
                child_write, api.HandleFlags.HANDLE_FLAG_INHERIT,
                api.HandleFlags.HANDLE_FLAG_INHERIT)
            with windows_runtime.worker_stdout_null() as null_handle:
                contained = windows_runtime.launch(
                    sys.argv[0],
                    [*kwargs['arguments'], "--stdout-null-handle",
                     str(null_handle)],
                    kwargs['environment'], kwargs['workspace'],
                    [*kwargs['inherited_handles'], null_handle],
                    stdio=(child_read, child_write),
                    current_directory=kwargs['current_directory'])
        except BaseException:
            for handle in (front_read, front_write, child_read, child_write):
                api.close_handle(handle)
            raise
        # The front's copies of the child's ends must not outlive the launch,
        # or the worker's exit would never be visible as EOF here.
        api.close_handle(child_read)
        api.close_handle(child_write)
        self._proc = ContainedWorkerProcess(contained)
        self._proc.stdin = PipeHandle(front_write)
        self._proc.stdout = PipeHandle(front_read)
        self._exit_task = self._loop.create_task(self._watch_exit())

    async def _watch_exit(self):
        try:
            returncode = await self._proc.wait()
            self._process_exited(returncode)
        finally:
            if self._closed:
                self._release()

    def close(self):
        super().close()
        # close() requests termination, but the observer must still publish
        # the exit. Releasing handles or cancelling it here strands waiters.
        if self._exit_task is None or self._exit_task.done():
            self._release()

    def _call_connection_lost(self, exc):
        proc = self._proc
        try:
            super()._call_connection_lost(exc)
        finally:
            if proc is not None:
                proc.release()

    def _release(self):
        if self._proc is not None:
            self._proc.release()


async def create_worker_process(*, workspace, environment, arguments,
                                inherited_handles, current_directory):
    """Start one contained worker and return the asyncio ``Process``."""
    loop = asyncio.get_running_loop()
    protocol = SubprocessStreamProtocol(limit=64 * 1024, loop=loop)
    waiter = loop.create_future()
    transport = ContainedWorkerTransport(
        loop, protocol, workspace=workspace, environment=environment,
        arguments=arguments, inherited_handles=inherited_handles,
        current_directory=current_directory, waiter=waiter)
    try:
        await waiter
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:
        # Cancelling the caller cancels the waiter, not pipe attachment. Join
        # attachment cleanup before returning so it cannot publish late pipes.
        proc = transport._proc
        transport._connect_task.cancel()
        try:
            await transport._connect_task
        finally:
            transport.close()
            try:
                # Attachment may have been cancelled before its coroutine
                # started, in which case it never took ownership of these.
                if proc is not None:
                    for raw_pipe in (proc.stdin, proc.stdout):
                        raw_pipe.close()
            finally:
                await transport._exit_task
        raise
    return Process(transport, protocol, loop)
