"""Handing a channel end to a child process, per host OS.

POSIX hands a descriptor through ``pass_fds``.  Windows has no such thing: a
handle must be marked inheritable and named in the child's handle list
(``subprocess`` supports that through ``STARTUPINFO.lpAttributeList``), and
because an inherited handle keeps its value, argv can carry the handle number.

Two channels need this: the session-lifetime channel and the credential
capability channel.  POSIX keeps a pipe for the first and an AF_UNIX socket for
the second, exactly as before.

Windows uses directly inherited secured pipes: one for owner lifetime, two for
the bidirectional credential channel. The tested Windows interpreters do not expose AF_UNIX;
CPython's TCP-emulated socketpair would instead introduce an address-reachable
listener and a connection-substitution check. Neither is needed for endpoints
handed directly to a child.

Each Windows pipe has an explicit descriptor, independent of the creator's
future-thread default. Authorization depends on the endpoint DACL, the exact
inherited-handle list and protection of processes holding the handles -- not a
promise that every Windows version implements anonymous pipes without a name.
No additional allocation or launch protocol is involved.

The secured overlapped allocator is shared with subprocess stdio. Confidential
owner/credential channels remain separate one-way ends and never acquire the
stdio writer's duplex data-read grant. Inheritance and explicit process-handle
lists, not names or a second handshake, select the child's ends.

Pending asynchronous I/O belongs to the adopted endpoint's stream owner. Kernel
completion, including joined cancellation, precedes endpoint closure; there are
no IPC reader/writer threads or fixed joins. EOF stdin remains synchronous.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import ctypes
from ctypes import wintypes
import os
import socket
import secrets
import sys
import uuid
import warnings

from .lifetimes import finish_cleanup
from .streams import BufferedReader

from . import windows_api

PIPE_ACCESS_DUPLEX = 0x00000003
PIPE_ACCESS_INBOUND = 0x00000001
PIPE_WAIT = 0x00000000
PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
ERROR_PIPE_BUSY = 231
ERROR_PIPE_CONNECTED = 535
ERROR_ACCESS_DENIED = 5
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
BUFSIZE = 8192
_MAX_NAME_ATTEMPTS = 20


if sys.platform == "win32":
    def overlapped_pipe(*, duplex=False, overlapped=(True, True), bufsize=BUFSIZE):
        """Like ``os.pipe()`` but with overlapped support and using handles.

        Returns ``(h1, h2)``: ``h1`` is the ``CreateNamedPipe`` end and ``h2`` the
        connecting client. ``overlapped`` selects each endpoint's native I/O mode:
        child stdio remains synchronous; both capability-channel ends are async.
        """
        create_pipe = windows_api.bind(
            "kernel32", "CreateNamedPipeW", ctypes.c_void_p, wintypes.LPCWSTR,
            wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
            wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p)
        connect = windows_api.bind(
            "kernel32", "ConnectNamedPipe", wintypes.BOOL, ctypes.c_void_p, ctypes.c_void_p)

        if duplex:
            openmode = PIPE_ACCESS_DUPLEX
            access = windows_api.AccessMask.GENERIC_READ | windows_api.AccessMask.GENERIC_WRITE
            obsize, ibsize = bufsize, bufsize
        else:
            openmode = PIPE_ACCESS_INBOUND
            access = windows_api.AccessMask.GENERIC_WRITE
            obsize, ibsize = 0, bufsize
        openmode |= windows_api.FileFlags.FILE_FLAG_FIRST_PIPE_INSTANCE
        if overlapped[0]:
            openmode |= windows_api.FileFlags.FILE_FLAG_OVERLAPPED
        client_flags = (windows_api.FileFlags.FILE_FLAG_OVERLAPPED if overlapped[1] else 0)

        # The pipe name is not a control: the ends are connected here and only the
        # child's inherited handle reaches the child.  The DACL nevertheless keeps
        # the default one -- which grants Everyone read -- from applying while the
        # name is briefly visible in the namespace.
        policy = pipe_policy(duplex=duplex)
        with contextlib.ExitStack() as rollback:
            with windows_api._converted_security_descriptor(policy['descriptor']) as descriptor:
                attributes = windows_api.SecurityAttributes(
                    ctypes.sizeof(windows_api.SecurityAttributes), descriptor, 0)
                for attempt in range(_MAX_NAME_ATTEMPTS):
                    # Contained creators need the LOCAL pipe namespace. The ACP front
                    # remains uncontained; its existing child-handle protocol is intact.
                    prefix = r"\\.\pipe\LOCAL" if policy['package'] is not None else r"\\.\pipe"
                    address = prefix + "\\" + "loki-worker-" + uuid.uuid4().hex
                    handle = create_pipe(address, openmode, PIPE_WAIT
                                         | PIPE_REJECT_REMOTE_CLIENTS, 1, obsize,
                                         ibsize, 0, ctypes.byref(attributes))
                    if handle and handle != INVALID_HANDLE_VALUE:
                        server = handle
                        rollback.callback(windows_api.close_handle, server)
                        break
                    status = ctypes.get_last_error()
                    # Another process holding this name cannot be distinguished from
                    # a name collision here; try another random name either way.
                    if (attempt == _MAX_NAME_ATTEMPTS - 1
                            or status not in [ERROR_PIPE_BUSY, ERROR_ACCESS_DENIED]):
                        raise windows_api.WindowsApiError("CreateNamedPipeW failed", status=status)
                client = windows_api.open_with_access(
                    address, access, creation=windows_api.FileCreateDisposition.OPEN_EXISTING,
                    flags=client_flags, share_mode=0)
                rollback.callback(windows_api.close_handle, client)
                # The client connected first, so this reports ERROR_PIPE_CONNECTED.
                if not connect(server, None):
                    status = ctypes.get_last_error()
                    if status != ERROR_PIPE_CONNECTED:
                        raise windows_api.WindowsApiError("ConnectNamedPipe failed", status=status)
            # Both ends are still unpublished while the shared descriptor retires.
            # Rollback attempts each acquired close even when another one raises.
            rollback.pop_all()
            return server, client


class PipeHandle:
    """The file-like raw owner lent to the public proactor pipe adapters.

    Those adapters notify their protocol before raw close; a close exception is
    otherwise only a loop diagnostic. Retain that one release failure so the
    owning process can report it after protocol retirement, without another
    closer task or another attempt on the consumed native handle.
    """

    def __init__(self, handle):
        self._handle = handle
        self._close_error: BaseException | None = None

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
            try:
                windows_api.close_handle(handle)
            except BaseException as error:
                self._close_error = error
                raise

    def raise_close_error(self):
        if self._close_error is not None:
            raise self._close_error

    def __del__(self, _warn=warnings.warn):
        if self._handle is not None:
            _warn(f"unclosed {self!r}", ResourceWarning, source=self)
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


_PAIR_CONFIRM_BYTES = 32
_PAIR_CONFIRM_TIMEOUT = 5.0


class PairConfirmationError(RuntimeError):
    """The two channel ends are not two halves of one connection."""


def _receive_exactly(end, count: int) -> bytes:
    received = bytearray()
    while len(received) < count:
        chunk = end.recv(count - len(received))
        if not chunk:
            raise PairConfirmationError(
                "a channel end closed before it confirmed the pair")
        received.extend(chunk)
    return bytes(received)


def confirm_pair(first, second) -> None:
    """Prove ``first`` and ``second`` are connected to each other.

    One direction suffices: a nonce sent on ``second`` must arrive on
    ``first``.  If it does not -- because a racing process was accepted in
    place of our own connect -- the two ends are not a pair, and this closes
    both and raises rather than letting a mismatched channel be handed to a
    child.  Both platforms run it: POSIX ``socketpair()`` cannot be
    substituted, but the contract is the same, and this host is the only place
    the check itself can be tested.
    """
    nonce = secrets.token_bytes(_PAIR_CONFIRM_BYTES)
    first_timeout = first.gettimeout()
    second_timeout = second.gettimeout()
    first.settimeout(_PAIR_CONFIRM_TIMEOUT)
    second.settimeout(_PAIR_CONFIRM_TIMEOUT)
    try:
        second.sendall(nonce)
        received = _receive_exactly(first, len(nonce))
        if received != nonce:
            raise PairConfirmationError(
                "the channel ends are not connected to each other")
    except OSError as error:
        first.close()
        second.close()
        raise PairConfirmationError(
            "the channel ends did not confirm the pair") from error
    except PairConfirmationError:
        first.close()
        second.close()
        raise
    finally:
        # The caller owns both ends afterwards; leave their modes as found.
        # A closed end accepts no timeout and raises OSError, suppressed here.
        for end, timeout in ((first, first_timeout), (second, second_timeout)):
            with contextlib.suppress(OSError):
                end.settimeout(timeout)


def pipe_policy(*, duplex=False):
    """Object-local policy and kernel identity for local pipe allocation.

    Duplex subprocess stdin is deliberately tool-facing: its asynchronous
    writer needs read access to observe peer closure. Confidential credential
    and owner channels use separate one-way pipes and never request that grant.
    """
    # Use kernel-reported process identity: a workspace path, environment
    # value or requested SID must not select who can access a credential pipe.
    owner = windows_api.current_user_sid()
    token = windows_api.open_process_token(
        windows_api.current_process_handle())
    try:
        package = (windows_api.token_app_container_sid(token)
                   if windows_api.token_is_app_container(token) else None)
    finally:
        windows_api.close_handle(token)
    descriptor = f"D:P(A;;GA;;;{owner})(A;;GA;;;SY)(A;;GA;;;BA)"
    if package is not None:
        # A user-only DACL denies the contained creator's client-end reopen.
        # FILE_GENERIC_WRITE includes READ_CONTROL/SYNCHRONIZE, but not the
        # READ_ATTRIBUTES that open also requests. One-way confidential pipes
        # must not grant package data reads; only tool-facing duplex stdio does.
        # Keep this ACE off the token default, which also secures new threads.
        # Tool write-open denial still requires its restricting-SID check;
        # inherited handles bypass a new open and must be selected separately.
        access = (windows_api.AccessMask.FILE_GENERIC_WRITE
                  | windows_api.AccessMask.FILE_READ_ATTRIBUTES)
        if duplex:
            access |= windows_api.AccessMask.FILE_GENERIC_READ
        descriptor += f"(A;;0x{int(access):x};;;{package})"
    return {'descriptor': descriptor, 'package': package}


if sys.platform == "win32":
    def private_pipe(inherit: bool = True, *, overlapped=False):
        """One-way private ends; only asynchronous channel callers request overlap.

        EOF stdin remains synchronous. Owner/credential endpoints use the same
        secured allocator as stdio without its duplex data-read permission.
        """
        if not overlapped:
            return windows_api.create_pipe(
                inherit=inherit, security_descriptor=pipe_policy()['descriptor'])
        read, write = overlapped_pipe()
        try:
            for handle in [read, write]:
                windows_api.set_handle_information(handle, windows_api.HandleFlags.HANDLE_FLAG_INHERIT,
                                                   windows_api.HandleFlags.HANDLE_FLAG_INHERIT if inherit else 0)
        except BaseException:
            with contextlib.ExitStack() as cleanup:
                cleanup.callback(windows_api.close_handle, write)
                cleanup.callback(windows_api.close_handle, read)
            raise
        return read, write


def eof_stdin():
    """A stdin end that is already at end of file, as a child-side object.

    ``subprocess.DEVNULL`` asks this process to open the null device and hand
    the result to the child.  That works on POSIX, but the AppContainer which
    confines Loki's Windows workers denies the device, so the contained launch
    cannot deliver stdin at all.  An anonymous pipe needs no device: the write
    end is closed before the caller launches anything, so the child reads end
    of file because no writer exists.

    Returns the child-side object and a ``close`` callable for the caller's
    copy.  The caller owns the returned end until it has been handed over, and
    must release both on every path, including launch failure.
    """
    if sys.platform == "win32":
        import msvcrt

        # CPython subprocess takes a CRT descriptor here, not a raw HANDLE.
        # Ownership transfers exactly once; subprocess duplicates the standard
        # handle for the child, so this parent copy need not be inheritable.
        read, write = private_pipe(inherit=False)
        try:
            windows_api.close_handle(write)
        except BaseException:
            windows_api.close_handle(read)
            raise
        try:
            fd = msvcrt.open_osfhandle(read, os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT)
        except BaseException:
            windows_api.close_handle(read)
            raise

        def close():
            os.close(fd)

        return fd, close
    read, write = os.pipe()
    os.close(write)

    def close():
        os.close(read)

    return read, close


if sys.platform == "win32":
    def _private_pipe_pair():
        """Two private one-way pipes forming a bidirectional byte channel.

        Returns ``(request, response)``, each a ``(parent, child)`` pair.  The
        request pipe carries child-to-parent bytes (``request`` is
        ``(parent_read, child_write)``); the response pipe carries parent-to-child
        bytes (``response`` is ``(child_read, parent_write)``).  A delegated
        process receives exactly the two ``child`` handles through the explicit
        handle list, and the two ``parent`` handles stay here with inheritance
        cleared.

        The creator connects both ends before handoff. The pipe DACL and
        protection of handle holders must also deny
        unauthorized access/recovery; anonymity is not that security proof.

        Only Windows channel factories allocate these native handles. POSIX
        credential channels use the kernel's connected socketpair instead.
        """
        request_read, request_write = private_pipe(overlapped=True)
        try:
            response_read, response_write = private_pipe(overlapped=True)
        except BaseException:
            with contextlib.ExitStack() as cleanup:
                cleanup.callback(windows_api.close_handle, request_write)
                cleanup.callback(windows_api.close_handle, request_read)
            raise
        try:
            # Only the child's ends stay inheritable; the parent's ends are cleared
            # so they can never be inherited by an unrestricted child.
            windows_api.clear_handle_inheritance(request_read)
            windows_api.clear_handle_inheritance(response_write)
        except BaseException:
            with contextlib.ExitStack() as cleanup:
                for handle in [response_write, response_read, request_write, request_read]:
                    cleanup.callback(windows_api.close_handle, handle)
            raise
        return (request_read, request_write), (response_read, response_write)


class PipeEndpoint:
    """One side of a directly inherited Windows pipe channel.

    ``read``/``write`` are Windows handles, or ``None`` when that direction is
    absent.  The owner channel is unidirectional (parent write-only, child
    read-only); the credential channel is bidirectional, so each side carries
    one handle per direction.  An endpoint owns its handles and nothing else:
    the two sides of a channel hold disjoint handles, and only a *child*
    endpoint is ever inherited or named in a handle list.
    """

    def __init__(self, read=None, write=None):
        self.read = read
        self.write = write

    def handles(self) -> list:
        """The handles on this side, in a stable order (read, then write)."""
        return list(handle for handle in [self.read, self.write]
                    if handle is not None)

    def reference(self) -> str:
        """An argv value the child parses back with :meth:`parse`."""
        parts = []
        if self.read is not None:
            parts.append(f"r={self.read}")
        if self.write is not None:
            parts.append(f"w={self.write}")
        return ",".join(parts)

    @classmethod
    def parse(cls, value: str) -> "PipeEndpoint":
        """Rebuild the endpoint an argv reference names."""
        read = write = None
        for part in str(value).split(","):
            name, separator, number = part.partition("=")
            if not separator:
                raise ValueError("malformed pipe endpoint reference")
            handle = int(number)
            if handle <= 0:
                raise ValueError("pipe endpoint handle was not inherited")
            if name == "r":
                if read is not None:
                    raise ValueError("duplicate read end")
                read = handle
            elif name == "w":
                if write is not None:
                    raise ValueError("duplicate write end")
                write = handle
            else:
                raise ValueError("unknown pipe endpoint direction")
        if read is None and write is None:
            raise ValueError("pipe endpoint reference names no end")
        return cls(read, write)

    def close(self) -> None:
        handles = self.handles()
        # Consume once even on failure; retrying a possibly invalid numeric
        # handle could close a replacement. Errors still propagate after both
        # independent ends have received their release attempt.
        self.read = self.write = None
        with contextlib.ExitStack() as cleanup:
            for handle in reversed(handles):
                cleanup.callback(windows_api.close_handle, handle)


def is_endpoint(value) -> bool:
    """Whether a value carries native pipe handles rather than a CRT/POSIX fd."""
    return isinstance(value, PipeEndpoint)


class _PipeStreams:
    """Own an adopted overlapped endpoint and its serial byte operations.

    Public asyncio's write-pipe adapter requires read access; private proactor
    hooks are not a supported substitute. The native operation primitive owns
    pending buffers, so this replaces two threads and a forwarding pump rather
    than adding another queue/reader lifecycle. No address or new authority is
    introduced, and closing admission never closes handles under active I/O.
    """

    def __init__(self, endpoint, *, limit=None):
        self._endpoint = endpoint
        self._reader = BufferedReader(self._receive, 65536 if limit is None else limit)
        self._reading = None
        self._writing = None
        self._pending = collections.deque()
        self._closing = False

    @property
    def transport(self):
        """The same operation owner implements the public writer transport seam."""
        return self

    def is_closing(self):
        return self._closing

    def abort(self):
        self.close()

    async def readline(self):
        return await self._reader.readline()

    async def _receive(self):
        if self._closing:
            raise ConnectionError('pipe is closed')
        if self._reading is not None:
            raise RuntimeError('pipe already has a reader')
        self._reading = asyncio.create_task(windows_api.read_pipe(self._endpoint.read))
        try:
            return await self._reading
        except windows_api.WindowsApiError as error:
            raise OSError(None, str(error), None, error.status) from error
        finally:
            self._reading = None

    def write(self, data):
        if self._closing:
            raise ConnectionError('pipe is closed')
        if self._writing is not None and self._writing.done():
            self._writing.result()
        if data:
            self._pending.append(bytes(data))
            if self._writing is None or self._writing.done():
                self._writing = asyncio.create_task(self._flush())

    async def _flush(self):
        try:
            while self._pending:
                await windows_api.write_pipe(self._endpoint.write, self._pending.popleft())
        except windows_api.WindowsApiError as error:
            kind = BrokenPipeError if error.status in windows_api.PIPE_CLOSED_ERRORS else OSError
            raise kind(None, str(error), None, error.status) from error

    async def drain(self):
        if self._writing is not None:
            await asyncio.shield(self._writing)

    def close(self):
        if self._closing:
            return
        self._closing = True
        self._pending.clear()
        for task in [self._reading, self._writing]:
            if task is not None and not task.done():
                task.cancel()

    async def wait_closed(self):
        self.close()

        async def retire():
            try:
                outcomes = await asyncio.gather(*[
                    task for task in [self._reading, self._writing] if task is not None],
                    return_exceptions=True)
            finally:
                self._endpoint.close()
            for outcome in outcomes:
                if isinstance(outcome, Exception):
                    raise outcome

        await finish_cleanup(retire())


def close_end(end) -> None:
    """Release the original channel object, not a reconstructed handle copy."""
    if is_endpoint(end) or isinstance(end, socket.socket):
        end.close()
    else:
        os.close(end)


def watch_closed(end):
    """Borrow an owner end; the caller owns its release after watcher retirement.

    SessionOwner and direct process fixtures use the same close-only protocol.
    Clear fd inheritance here, before a returned coroutine can first run; sockets
    keep their existing public-stream adaptation and native pipes their I/O join.
    """
    if isinstance(end, socket.socket):
        async def socket_closed():
            reader, writer = await asyncio.open_connection(sock=end)
            try:
                return await reader.read()
            finally:
                writer.close()
        return socket_closed()
    if is_endpoint(end):
        async def pipe_closed():
            while await windows_api.read_pipe(end.read):
                pass
        return pipe_closed()
    os.set_inheritable(end, False)
    from . import acps
    return acps.AsyncFdLineReader(end).readline()


if sys.platform != "win32":
    def duplicate_for_child(fd: int) -> int:
        """Own a non-inheritable copy outside the child's standard streams.

        Only explicit pass_fds makes this copy inheritable at exec. The source
        remains borrowed and unchanged, including when it occupies a closed
        standard-stream slot in the parent.
        """
        import fcntl

        return fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)

    def owner_channel():
        """(parent end, child end) for the session-lifetime channel.

        The kernel creates both unidirectional ends; no connection by address
        selects the child endpoint.
        """
        with contextlib.ExitStack() as cleanup:
            read_fd, write_fd = os.pipe()
            cleanup.callback(os.close, write_fd)
            try:
                child_fd = duplicate_for_child(read_fd)
                cleanup.callback(os.close, child_fd)
            finally:
                os.close(read_fd)
            cleanup.pop_all()
            return write_fd, child_fd

    def socket_pair():
        """The kernel creates both connected ends; there is no rendezvous."""
        pair = socket.socketpair()
        confirm_pair(*pair)
        return pair

    def prepare_child_socket(end):
        """Transfer an owned child copy without occupying a standard stream."""
        with contextlib.ExitStack() as cleanup:
            try:
                end.set_inheritable(False)
                fd = duplicate_for_child(end.fileno())
                cleanup.callback(os.close, fd)
            finally:
                end.close()
            cleanup.pop_all()
            return fd

    def reference(child_end) -> int:
        """The argv value that names ``child_end`` in the child."""
        return int(child_end)

    def handles(child_end) -> tuple:
        """The descriptor(s) that must cross to the child for ``child_end``."""
        if isinstance(child_end, socket.socket):
            return (child_end.fileno(),)
        return (int(child_end),)

    def spawn_kwargs(references) -> dict:
        return {"pass_fds": list(
            fd for end in references for fd in handles(end))}

    def child_endpoint(value):
        """The child's view of a reference: the descriptor it names."""
        fd = int(value)
        if fd < 3:
            raise ValueError("descriptor was not inherited")
        return fd

    async def open_streams(end, limit=None):
        """The credential channel's async streams: ``open_connection``."""
        # The parent end must never leak into a child, and the loop takes a
        # non-blocking socket.
        end.set_inheritable(False)
        end.setblocking(False)
        return await asyncio.open_connection(sock=end, limit=limit)

else:
    import subprocess

    def owner_channel():
        """A one-way inherited pipe: the parent's write end is the lifetime.

        The parent keeps the write end and closes it to signal the child; the
        child reads.  Only the child's read handle stays inheritable.
        """
        with contextlib.ExitStack() as rollback:
            read_handle, write_handle = private_pipe(overlapped=True)
            rollback.callback(windows_api.close_handle, read_handle)
            rollback.callback(windows_api.close_handle, write_handle)
            # Neither raw end has a caller until inheritance and both endpoint
            # owners are ready. A failed flag update must retire both ends here.
            windows_api.clear_handle_inheritance(write_handle)
            ends = (PipeEndpoint(write=write_handle), PipeEndpoint(read=read_handle))
            rollback.pop_all()
            return ends

    def socket_pair():
        """The credential channel: two explicitly secured one-way pipes."""
        request, response = _private_pipe_pair()
        # request is (parent_read, child_write); response is
        # (child_read, parent_write).  Each side gets one read and one write.
        return (PipeEndpoint(read=request[0], write=response[1]),
                PipeEndpoint(read=response[0], write=request[1]))

    def prepare_child_socket(end):
        """Keep ``end`` alive, inheritable, for the child to inherit."""
        return end

    def reference(child_end) -> str:
        """The argv value that names the child endpoint's handles."""
        return child_end.reference()

    def handles(child_end) -> list:
        """The child handles that must cross for ``child_end``."""
        return child_end.handles()

    def spawn_kwargs(references) -> dict:
        startup = subprocess.STARTUPINFO()  # pytype: disable=module-attr
        startup.lpAttributeList = {
            "handle_list": [handle for end in references
                            for handle in handles(end)]}
        return {"startupinfo": startup}

    def child_endpoint(value):
        """The child's view of a reference: the endpoint the handles name."""
        return (value if isinstance(value, PipeEndpoint)
                else PipeEndpoint.parse(value))

    async def open_streams(end, limit=None):
        """One operation owner replaces the former reader/writer threads."""
        stream = _PipeStreams(end, limit=limit)
        return stream, stream
