"""Handing a channel end to a child process, per host OS.

POSIX hands a descriptor through ``pass_fds``.  Windows has no such thing: a
handle must be marked inheritable and named in the child's handle list
(``subprocess`` supports that through ``STARTUPINFO.lpAttributeList``), and
because an inherited handle keeps its value, argv can carry the handle number.

Two channels need this: the session-lifetime channel and the credential
capability channel.  POSIX keeps a pipe for the first and an AF_UNIX socket for
the second, exactly as before.

**Why Windows does not use ``socket.socketpair()``.**  Winsock has no
``socketpair``, so CPython falls back to an AF_INET loopback emulation:
it binds and listens on 127.0.0.1, connects a second socket to it, and accepts.
That means a real *listening TCP port* exists for the window between ``listen``
and ``accept``.  Any local process, under any user, can connect to it; CPython
detects the substitution afterwards by comparing peer addresses and raises
``ConnectionError``, so a hijack fails rather than succeeds -- but a failure is
still a local denial, and, worse, loopback traffic is subject to no policy at
all, so there is no control to reason about.  Reachability by *address* is the
wrong property for a credential channel.

**What Windows uses instead: AF_UNIX.**  Windows 10 1803 (and Server 2019)
onwards has AF_UNIX: ``bind`` creates a socket file -- an NTFS reparse point --
and, per Microsoft's documentation, connecting to it requires write permission
on that file, with ``bind`` requiring write permission on its directory.  So
reachability is bounded by the filesystem ACL on a directory this module
creates private (``tempfile.mkdtemp``, i.e. mode 0o700) and removes as soon as
the pair is connected.  Winsock has no ``socketpair`` for AF_UNIX either, so
the pair is built by hand with bind/listen/connect/accept; the socket file must
be unlinked before the path can be reused.  There is no ``SCM_RIGHTS`` or
``SCM_CREDENTIALS``, so the peer's identity is never learned -- it does not
need to be, because the child end is handed over as an inherited handle and
possession of that handle is the authorization.

**AF_UNIX is unreachable through CPython, so this is being replaced by pipes.**
Measured 2026-09-13: ``socket.AF_UNIX`` is absent on every Windows interpreter
Loki runs -- native CPython 3.12 and 3.13 and the mingw-w64 UCRT64 build, host
and staged copy alike.  The cause is CPython's gate, not the OS:
``Modules/socketmodule.h`` does ``#undef AF_UNIX`` whenever ``HAVE_SYS_UN_H``
is absent, MSVC's ``pyconfig.h`` never defines it, and ``configure`` cannot
find ``sys/un.h`` under mingw-w64 either.  The AF_UNIX branch below therefore
cannot execute on any supported interpreter.  The intended transport is
:func:`_private_pipe_pair`: two anonymous pipes, which have no name to
enumerate, squat or confirm.  Possession of the inherited handle remains the
authorization, exactly as for the session-lifetime channel.  The swap of
``socket_pair`` and the transport-neutral endpoint is pending; the AF_UNIX
branch stays until then so its reasoning and tests are not lost.

Both channels are therefore the same shape on Windows: a private channel,
parent end non-inheritable, child end inheritable and named in the handle list.

**The Windows ends are :class:`PipeEndpoint`.**  Every channel is a
``(parent endpoint, child endpoint)`` pair at all times.  On Windows an
endpoint wraps one handle per direction (the owner channel is unidirectional,
so its endpoints have one handle each; the credential channel is
bidirectional, so the child endpoint carries a read handle and a write handle
and the parent endpoint carries the complementary two).  Only the child
endpoint's handles are ever inherited or named in the child's handle list, and
``reference``/``handles`` take the endpoint object -- never a bare handle -- so
a caller cannot hand over the wrong side.  The child ends are distinct and
unidirectional: the child cannot read its own requests or write its own
responses.

POSIX keeps the objects it always had -- an ``int`` descriptor for the owner
channel and a ``socket.socket`` for the credential channel -- so its behaviour
and tests are unchanged; the endpoints are a Windows type.  ``open_streams``
and ``watch_closed`` are the transport-neutral seam the runtime layer uses: the
Windows implementation builds an :class:`asyncio.StreamReader` fed by
``handle_reader.HandleReader`` and flushes writes from a dedicated thread, so a
blocking ``WriteFile`` on a full pipe never freezes the event loop.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import os
import secrets
import socket
import threading

from . import windows_api

# A confirmed pair is one whose two ends really are connected to each other.
# The nonce round-trip cannot fail for a POSIX ``socketpair()``; it exists
# because the Windows pair is built around a filesystem name, and ``accept``
# cannot say whose connection it accepted, so a process that wins the
# bind/connect race would otherwise be handed one end of a foreign connection.
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


def _private_pipe_pair():
    """Two anonymous pipes: a named-nothing bidirectional byte channel.

    Returns ``(request, response)``, each a ``(parent, child)`` pair.  The
    request pipe carries child-to-parent bytes (``request`` is
    ``(parent_read, child_write)``); the response pipe carries parent-to-child
    bytes (``response`` is ``(child_read, parent_write)``).  A delegated
    process receives exactly the two ``child`` handles through the explicit
    handle list, and the two ``parent`` handles stay here with inheritance
    cleared.

    There is no name, so -- unlike the AF_UNIX emulation -- nothing else can
    connect, squat, or win a bind/connect race, and ``confirm_pair`` is not
    needed: the only way to hold an end is to have been handed it.

    Defined outside the platform branch because it depends only on
    ``windows_api``'s import-safe declarations; calling it off Windows raises
    the same ``WindowsUnavailableError`` any other Windows call does.
    """
    request_read, request_write = windows_api.create_pipe()
    try:
        response_read, response_write = windows_api.create_pipe()
    except BaseException:
        windows_api.close_handle(request_read)
        windows_api.close_handle(request_write)
        raise
    try:
        # Only the child's ends stay inheritable; the parent's ends are cleared
        # so they can never be inherited by an unrestricted child.
        windows_api.clear_handle_inheritance(request_read)
        windows_api.clear_handle_inheritance(response_write)
    except BaseException:
        for handle in (request_read, request_write,
                       response_read, response_write):
            windows_api.close_handle(handle)
        raise
    return (request_read, request_write), (response_read, response_write)


class PipeEndpoint:
    """One side of a Windows anonymous-pipe channel.

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

    def handles(self) -> tuple:
        """The handles on this side, in a stable order (read, then write)."""
        return tuple(handle for handle in (self.read, self.write)
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
        for handle in self.handles():
            windows_api.close_handle(handle)
        self.read = self.write = None


def is_endpoint(value) -> bool:
    """Whether ``value`` is a Windows pipe endpoint (never true on POSIX)."""
    return isinstance(value, PipeEndpoint)


class _HandleWriter:
    """Flush queued bytes to a Windows handle from one thread.

    A blocking ``WriteFile`` on a full pipe must not run on the loop thread, or
    it freezes every task -- the same reason ``HandleReader`` exists for the
    read side (anonymous pipe handles support no overlapped I/O, so the
    proactor loop cannot take them).  Writes are queued and flushed in order by
    a single thread; the loop only enqueues and awaits the drain signal.
    Closing the peer's complementary end makes an in-flight ``WriteFile`` fail,
    which is what lets teardown stop a blocked thread.
    """

    def __init__(self, handle, loop):
        self.handle = handle
        self.loop = loop
        self._pending = collections.deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._drained = None
        self._error = None
        self.thread = None

    def start(self) -> None:
        self._drained = asyncio.Event()
        self.thread = threading.Thread(
            target=self._run, name="loki-handle-writer", daemon=True)
        self.thread.start()

    def write(self, data) -> None:
        with self._lock:
            self._pending.append(bytes(data))
        self._wake.set()

    def _run(self) -> None:
        while True:
            # Poll the stop event as well as the work event, so an idle writer
            # is not a thread that can never be joined.
            self._wake.wait(0.05)
            self._wake.clear()
            failure = None
            while True:
                with self._lock:
                    if not self._pending:
                        break
                    chunk = self._pending.popleft()
                try:
                    offset = 0
                    while offset < len(chunk):
                        offset += windows_api.write_file(
                            self.handle, chunk[offset:])
                except BaseException as error:  # noqa: BLE001 - re-raised on loop
                    failure = error
                    break
            if failure is not None:
                with self._lock:
                    self._pending.clear()
                self._error = failure
                self._signal()
                return
            self._signal()
            if self._stop.is_set():
                return

    def _signal(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self._set_drained)
        except RuntimeError:
            # The loop is closed; teardown is already running.
            pass

    def _set_drained(self) -> None:
        if self._drained is not None:
            self._drained.set()

    async def drain(self) -> None:
        while True:
            if self._error is not None:
                raise self._error
            with self._lock:
                if not self._pending:
                    return
            # Clear, then re-check under the lock: a flush that completed
            # between the check and the clear must not be lost, or this waits
            # for a signal that already fired.
            self._drained.clear()
            with self._lock:
                if not self._pending:
                    continue
            await self._drained.wait()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()

    def stop_and_join(self, timeout: float = 2.0) -> None:
        self.close()
        if self.thread is not None:
            self.thread.join(timeout)
            if self.thread.is_alive():
                # Closing a handle under a live writer is worse than failing.
                raise RuntimeError("handle writer thread did not stop")
            self.thread = None


async def _feed_reader(reader: "asyncio.StreamReader", queue) -> None:
    """Forward the reader thread's byte posts into an asyncio StreamReader."""
    while True:
        data = await queue.get()
        if not data:
            reader.feed_eof()
            return
        reader.feed_data(data)


class _PipeStreamWriter:
    """A ``StreamWriter``-shaped adapter over a handle writer and its reader.

    The credential client and server are written against
    ``StreamReader``/``StreamWriter``; this lets the Windows pipe transport
    satisfy that interface without a parallel protocol implementation.  Writes
    are queued to the writer thread, reads come from the handle reader, and
    closing the adapter stops both threads and closes the endpoint's handles.
    """

    def __init__(self, source, pump, sink, endpoint):
        self._source = source
        self._pump = pump
        self._sink = sink
        self._endpoint = endpoint

    def write(self, data) -> None:
        self._sink.write(data)

    async def drain(self) -> None:
        await self._sink.drain()

    def close(self) -> None:
        self._sink.close()
        self._source.stop()
        if not self._pump.done():
            self._pump.cancel()

    async def wait_closed(self) -> None:
        await asyncio.gather(self._pump, return_exceptions=True)
        self._sink.stop_and_join()
        self._endpoint.close()


if os.name == "posix":
    def owner_channel():
        """(parent end, child end) for the session-lifetime channel.

        A pipe carries no name, so there is no connection for another process
        to race; ``confirm_pair`` does not apply to its two unidirectional ends
        and is not needed.
        """
        read_fd, write_fd = os.pipe()
        return write_fd, read_fd

    def socket_pair():
        """A connected AF_UNIX pair, confirmed as our own; ``socketpair``."""
        pair = socket.socketpair()
        confirm_pair(*pair)
        return pair

    def prepare_child_socket(end):
        """Detach ``end`` so only the child's copy of the descriptor remains."""
        end.set_inheritable(False)
        return end.detach()

    def reference(child_end) -> int:
        """The argv value that names ``child_end`` in the child."""
        return int(child_end)

    def handles(child_end) -> tuple:
        """The descriptor(s) that must cross to the child for ``child_end``."""
        if isinstance(child_end, socket.socket):
            return (child_end.fileno(),)
        return (int(child_end),)

    def spawn_kwargs(references) -> dict:
        return {"pass_fds": tuple(
            fd for end in references for fd in handles(end))}

    def close_end(end) -> None:
        if isinstance(end, socket.socket):
            end.close()
        else:
            os.close(end)

    def child_endpoint(value):
        """The child's view of a reference: a validated plain descriptor."""
        fd = int(value)
        if fd < 3:
            raise ValueError("descriptor was not inherited")
        os.fstat(fd)
        return fd

    async def open_streams(end, limit=None):
        """The credential channel's async streams: ``open_connection``."""
        # The parent end must never leak into a child, and the loop takes a
        # non-blocking socket.
        end.set_inheritable(False)
        end.setblocking(False)
        return await asyncio.open_connection(sock=end, limit=limit)

    async def watch_closed(end):
        """Await EOF on an inherited owner-channel end."""
        if isinstance(end, socket.socket):
            reader, writer = await asyncio.open_connection(sock=end)
            try:
                await reader.read()
            finally:
                writer.close()
        else:
            os.set_inheritable(end, False)
            from . import acps
            await acps.AsyncFdLineReader(end).readline()

else:
    import subprocess

    def owner_channel():
        """An anonymous pipe pair: the parent's write end is the lifetime.

        The parent keeps the write end and closes it to signal the child; the
        child reads.  Only the child's read handle stays inheritable.
        """
        read_handle, write_handle = windows_api.create_pipe()
        windows_api.clear_handle_inheritance(write_handle)
        return (PipeEndpoint(write=write_handle),
                PipeEndpoint(read=read_handle))

    def socket_pair():
        """The credential channel: two anonymous pipes, no name to race."""
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

    def handles(child_end) -> tuple:
        """The child handles that must cross for ``child_end``."""
        return child_end.handles()

    def spawn_kwargs(references) -> dict:
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {
            "handle_list": [handle for end in references
                            for handle in handles(end)]}
        return {"startupinfo": startup}

    def close_end(end) -> None:
        end.close()

    def child_endpoint(value):
        """The child's view of a reference: the endpoint the handles name."""
        if isinstance(value, PipeEndpoint):
            return value
        return PipeEndpoint.parse(value)

    async def open_streams(end, limit=None):
        """Build ``StreamReader``/``StreamWriter``-shaped streams over pipes."""
        from . import handle_reader

        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=limit)
        queue = asyncio.Queue()
        source = handle_reader.HandleReader(
            None, loop, queue, eof_sentinel=True, handle=end.read)
        source.start()
        pump = loop.create_task(
            _feed_reader(reader, queue), name="loki-pipe-reader")
        sink = _HandleWriter(end.write, loop)
        sink.start()
        return reader, _PipeStreamWriter(source, pump, sink, end)

    async def watch_closed(end):
        """Await EOF on an inherited owner-channel read end."""
        from . import handle_reader

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        source = handle_reader.HandleReader(
            None, loop, queue, eof_sentinel=True, handle=end.read)
        source.start()
        try:
            while True:
                data = await queue.get()
                if not data:
                    return
        finally:
            source.stop()
