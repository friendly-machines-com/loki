# pytype: skip-file
"""Windows worker and local tool processes with native creation and async stdio.

The native pipe-attachment orchestration is adapted from CPython 3.14.7
``Lib/asyncio/base_subprocess.py`` under its PSF licence. Byte framing and the
process surface now use Loki's shared documented-protocol bridge; they do not
construct stdlib subprocess/stream internals. Secured pipe allocation is shared
with inherited IPC in host_ipc. Intentional native lifetime differences follow.

* ``host_ipc.overlapped_pipe`` uses random names, explicit endpoint-specific security and
  ``PIPE_REJECT_REMOTE_CLIENTS``. Contained creators use the LOCAL namespace;
  their package receives only the access needed to connect this stdio pair.
  Confidential owner/credential channels remain separate one-way allocations.
* The client calls ``ConnectNamedPipe`` expecting ``ERROR_PIPE_CONNECTED``:
  both ends are created and connected here, in this process, so no wait for a
  connection is needed.
* ACP startup uses Loki's contained launcher; local jobs/hooks instead use
  the caller-selected image, cwd and role. Local children remain suspended
  through stream attachment and the owner's final pre-resume check.
* Exit is observed by waiting on the process through Loki's own non-blocking
  wait, not by registering the handle with the loop's completion port.
* Leader exit signals the shared process protocol independently of stream EOF,
  so a descendant holding stdout cannot hang ``Process.wait()``.
"""

from __future__ import annotations

import asyncio
import collections
from contextlib import ExitStack
import subprocess
import sys
import warnings

from . import windows_api as api
from . import windows_runtime, host_ipc, streams, acps
from .lifetimes import finish_cleanup

PIPE = subprocess.PIPE


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
                 waiter=None, **kwargs):
        super().__init__()
        self._closed = False
        self._protocol = protocol
        self._loop = loop
        self._proc = None
        self._pid = None
        self._pending_calls = collections.deque()
        self._pipes = {}
        self._finished = False

        if stdin == PIPE:
            self._pipes[0] = None
        if stdout == PIPE:
            self._pipes[1] = None
        if stderr == PIPE:
            self._pipes[2] = None

        self._connect_task = None
        try:
            self._start(args=args, shell=shell, stdin=stdin, stdout=stdout,
                        stderr=stderr, bufsize=bufsize, **kwargs)
            self._pid = self._proc.pid
            self._connect_task = self._loop.create_task(self._connect_pipes(waiter))
        except BaseException:
            with ExitStack() as cleanup:
                if self._proc is not None:
                    for raw in [self._proc.stdin, self._proc.stdout, self._proc.stderr]:
                        if raw is not None:
                            cleanup.callback(raw.close)
                self.close()
            raise

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

        try:
            with ExitStack() as cleanup:
                if self._loop is not None and not self._loop.is_closed():
                    for proto in reversed(list(self._pipes.values())):
                        if proto is not None:
                            cleanup.callback(proto.pipe.close)
        finally:
            if self._proc is not None and self._proc.returncode is None:
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass

    def __del__(self, _warn=warnings.warn):
        if not self._closed:
            _warn(f"unclosed transport {self!r}", ResourceWarning, source=self)
            self.close()

    def get_pid(self):
        return self._pid

    def get_returncode(self):
        return self._proc.returncode if self._proc is not None else None

    def get_extra_info(self, name, default=None):
        return self._proc if name == 'subprocess' else default

    def get_pipe_transport(self, fd):
        if fd in self._pipes:
            return self._pipes[fd].pipe
        return None

    def _check_proc(self):
        if self._proc is None or self._protocol is None:
            raise ProcessLookupError()

    def send_signal(self, sig):
        # Windows has no signals here: preserve the existing job-stop behavior.
        self.terminate()

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
            # Attempt every independent pipe release, and notify the factory
            # even if a checked native close itself reports failure.
            try:
                with ExitStack() as cleanup:
                    for raw_pipe in [proc.stderr, proc.stdout, proc.stdin]:
                        if raw_pipe is not None:
                            cleanup.callback(raw_pipe.close)
                    for proto in reversed(list(self._pipes.values())):
                        if proto is not None:
                            cleanup.callback(proto.pipe.close)
            finally:
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
        assert self._proc.returncode == returncode, self._proc.returncode
        self._call(self._protocol.process_exited)
        self._try_finish()

    def _try_finish(self):
        assert not self._finished
        if self.get_returncode() is None:
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
            # Retain the native owner's cached result, not another result copy.
            # A retired protocol forbids new signalling through this transport.
            self._protocol = None


class Process(streams.Process):
    """The subprocess surface used by ACP, file-backed jobs and streamed hooks."""

    def __init__(self, transport, protocol):
        self._family = None
        super().__init__(transport, protocol)

    async def wait(self):
        # The native observer can itself report a failed wait/release. Waiting
        # only on the protocol event would hide that error or strand this caller.
        await asyncio.shield(self._transport._exit_task)
        return await super().wait()

    async def aclose(self):
        # The shared process waits for protocol closure only. Native retirement
        # must also observe its existing exit task, including release failures.
        try:
            self.close()
        finally:
            async def retired():
                try:
                    await self.wait()
                finally:
                    if self.returncode is not None:
                        try:
                            await self._protocol.closed
                        finally:
                            native = self._transport._proc
                            with ExitStack() as errors:
                                for raw in [native.stderr, native.stdout, native.stdin]:
                                    if raw is not None:
                                        errors.callback(raw.raise_close_error)
            await finish_cleanup(retired())

    def close_family(self):
        family, self._family = self._family, None
        if family is not None:
            family.close_family()

    def __del__(self):
        self.close_family()

    def family_exists(self):
        return (self.returncode is None if self._family is None
                else windows_runtime.job_has_processes(self._family.job))

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.pid}>'

    def terminate(self):
        if self._family is not None:
            windows_runtime.terminate_job(self._family.job)
        else:
            self._transport.terminate()

    kill = terminate


# -- Loki's contained worker ---------------------------------------------


class ContainedWorkerTransport(BaseSubprocessTransport):
    """One contained worker: Loki's launcher, the copied pipe orchestration."""

    def __init__(self, loop, protocol, *, native, workspace, environment,
                 arguments, inherited_handles, current_directory,
                 waiter=None):
        self._exit_task = None
        super().__init__(loop, protocol, args=None, shell=False,
                         stdin=PIPE, stdout=PIPE, stderr=None, bufsize=0,
                         waiter=waiter,
                         native=native, workspace=workspace, environment=environment,
                         arguments=arguments, inherited_handles=inherited_handles,
                         current_directory=current_directory)

    def _start(self, args, shell, stdin, stdout, stderr, bufsize, **kwargs):
        with ExitStack() as parents, ExitStack() as children:
            child_read, front_write = host_ipc.overlapped_pipe(overlapped=(False, True), duplex=True)
            children.callback(api.close_handle, child_read)
            parents.callback(api.close_handle, front_write)
            front_read, child_write = host_ipc.overlapped_pipe(overlapped=(True, False))
            children.callback(api.close_handle, child_write)
            parents.callback(api.close_handle, front_read)
            api.set_handle_information(
                child_read, api.HandleFlags.HANDLE_FLAG_INHERIT,
                api.HandleFlags.HANDLE_FLAG_INHERIT)
            api.set_handle_information(
                child_write, api.HandleFlags.HANDLE_FLAG_INHERIT,
                api.HandleFlags.HANDLE_FLAG_INHERIT)
            with windows_runtime.worker_stdout_null() as null_handle:
                contained = kwargs['native']
                windows_runtime.prepare_contained(
                    contained, sys.argv[0],
                    [*kwargs['arguments'], "--stdout-null-handle",
                     str(null_handle)],
                    kwargs['environment'], kwargs['workspace'],
                    [*kwargs['inherited_handles'], null_handle],
                    stdio=(child_read, child_write),
                    current_directory=kwargs['current_directory'])
            # Child copies must retire before parent transport adoption. Their
            # checked-close failure still leaves both parents owned for rollback;
            # an ExitStack consumes each callback once even when it raises.
            children.close()
            self._proc = contained
            self._proc.stdin = host_ipc.PipeHandle(front_write)
            self._proc.stdout = host_ipc.PipeHandle(front_read)
            parents.pop_all()
        self._exit_task = self._loop.create_task(self._watch_exit())

    async def _watch_exit(self):
        try:
            returncode = await self._proc.wait()
            self._process_exited(returncode)
        finally:
            if self._closed and self._proc.returncode is not None:
                self._release()

    def _call_connection_lost(self, exc):
        # The exit task owns early-close release; final callbacks own normal
        # retirement. Publish their native failure through the existing closed
        # future only after all stdio has retired, not as a loop-only diagnostic.
        try:
            self._release()
        except Exception as error:
            exc = error
        super()._call_connection_lost(exc)

    def _release(self):
        if self._proc is not None:
            self._proc.close()


class LocalProcessTransport(ContainedWorkerTransport):
    """Reuse the pipe/exit machinery, but keep local image and cwd explicit."""

    def __init__(self, loop, protocol, *, native, command, environment, cwd,
                 stdin, stdout, stderr, inherited_handles, owning, creationflags, waiter):
        self._exit_task = None
        BaseSubprocessTransport.__init__(
            self, loop, protocol, args=command, shell=False, stdin=stdin,
            stdout=stdout, stderr=stderr, bufsize=0, waiter=waiter,
            native=native, environment=environment, cwd=cwd,
            inherited_handles=inherited_handles, owning=owning, creationflags=creationflags)

    def _release(self):
        if self._proc is not None:
            self._proc.release_leader()

    def _start(self, args, shell, stdin, stdout, stderr, bufsize, **kwargs):
        parent_pipes = {}
        try:
            with ExitStack() as cleanup:
                children = []
                for fd, value in enumerate([stdin, stdout, stderr]):
                    if value == PIPE:
                        first, second = host_ipc.overlapped_pipe(duplex=fd == 0, overlapped=(fd != 0, fd == 0))
                        child, parent = [first, second] if fd == 0 else [second, first]
                        parent_pipes[fd] = host_ipc.PipeHandle(parent)
                        cleanup.callback(api.close_handle, child)
                        api.set_handle_information(child, api.HandleFlags.HANDLE_FLAG_INHERIT,
                                                   api.HandleFlags.HANDLE_FLAG_INHERIT)
                    else:
                        child = windows_runtime.duplicate_standard_handle(value, fd)
                        cleanup.callback(api.close_handle, child)
                    children.append(child)
                native = kwargs['native']
                windows_runtime.prepare_local(
                    native, args, kwargs['environment'], kwargs['cwd'],
                    [*kwargs['inherited_handles'], *children], children,
                    owning=kwargs['owning'], creationflags=kwargs['creationflags'])
            self._proc = native
            self._proc.stdin = parent_pipes.get(0)
            self._proc.stdout = parent_pipes.get(1)
            self._proc.stderr = parent_pipes.get(2)
            self._exit_task = self._loop.create_task(self._watch_exit())
        except BaseException:
            with ExitStack() as cleanup:
                for raw in parent_pipes.values():
                    cleanup.callback(raw.close)
            raise


async def create_local_process(*command, stdin=None, stdout=None, stderr=None,
                               close_fds=True, env=None, cwd=None, creationflags=0,
                               startupinfo=None, owning=False, before_resume=None):
    """Keep the native child suspended through attachment and owner validation."""
    if not close_fds:
        raise ValueError('local child handles must be explicit')
    inherited = []
    if startupinfo is not None:
        attributes = startupinfo.lpAttributeList or {}
        if startupinfo.dwFlags or attributes.keys() - {'handle_list'}:
            raise ValueError('unsupported local startup attributes')
        inherited = list(attributes.get('handle_list', []))
    native = windows_runtime.ContainedProcess(None)
    transport = None
    try:
        loop = asyncio.get_running_loop()
        protocol = streams.ProcessProtocol()
        waiter = loop.create_future()
        transport = LocalProcessTransport(
            loop, protocol, native=native, command=list(command), environment=env,
            cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr,
            inherited_handles=inherited, owning=owning, creationflags=creationflags, waiter=waiter)

        def resume():
            if before_resume is not None:
                before_resume()
            api.resume_thread(native.information.hThread)

        return await _complete_creation(transport, protocol, waiter, resume, native)
    except BaseException as error:
        if transport is None:
            await windows_runtime.retire_local(native, preserve_failure=True)
        if isinstance(error, api.WindowsApiError):
            if error.status is not None and not isinstance(error, api.WindowsResourceError):
                raise OSError(None, str(error), None, error.status) from error
            raise OSError(str(error)) from error
        raise


async def _complete_creation(transport, protocol, waiter, ready=None, family=None):
    try:
        await waiter
        process = Process(transport, protocol)
        if ready is not None:
            ready()
        # The factory owns family control through every pre-publication failure;
        # a provisional Process destructor must not release a still-live family.
        process._family = family
        return process
    except BaseException:
        async def finish():
            proc = transport._proc
            transport._connect_task.cancel()
            try:
                await transport._connect_task
            finally:
                try:
                    transport.close()
                finally:
                    try:
                        if proc is not None:
                            with ExitStack() as cleanup:
                                for raw in [proc.stderr, proc.stdout, proc.stdin]:
                                    if raw is not None:
                                        cleanup.callback(raw.close)
                    finally:
                        # Failed termination is not observed exit. This owner
                        # remains alive, with family control, through the join.
                        try:
                            await transport._exit_task
                        finally:
                            if proc is not None:
                                with ExitStack() as errors:
                                    # Failed attachment may never publish protocol
                                    # callbacks. Retire the observed native owner
                                    # here too, and retain an earlier callback's
                                    # error without another native release attempt.
                                    if protocol.closed.done():
                                        errors.callback(protocol.closed.result)
                                    if proc.returncode is not None:
                                        errors.callback(transport._release)
                                    for raw in [proc.stderr, proc.stdout, proc.stdin]:
                                        if raw is not None:
                                            errors.callback(raw.raise_close_error)

        try:
            await finish_cleanup(finish(), preserve_failure=True)
        finally:
            if family is not None and family.returncode is not None:
                family.close_family()
        raise


async def create_worker_process(*, workspace, environment, arguments,
                                inherited_handles, current_directory):
    """Start one contained worker and return the asyncio ``Process``."""
    native = windows_runtime.ContainedProcess(None)
    transport = None
    try:
        loop = asyncio.get_running_loop()
        protocol = streams.ProcessProtocol(limit=acps.MAX_MESSAGE_BYTES)
        waiter = loop.create_future()
        transport = ContainedWorkerTransport(
            loop, protocol, native=native, workspace=workspace, environment=environment,
            arguments=arguments, inherited_handles=inherited_handles,
            current_directory=current_directory, waiter=waiter)
        return await _complete_creation(
            transport, protocol, waiter,
            ready=lambda: api.resume_thread(native.information.hThread))
    except BaseException:
        if transport is None:
            await windows_runtime.retire_local(native, preserve_failure=True)
        raise
