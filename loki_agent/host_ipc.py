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
"""

from __future__ import annotations

import contextlib
import os
import secrets
import socket

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

    def spawn_kwargs(references) -> dict:
        return {"pass_fds": tuple(int(value) for value in references)}

    def close_end(end) -> None:
        os.close(end)

    def child_endpoint(value):
        """The child's view of a reference: a validated plain descriptor."""
        fd = int(value)
        if fd < 3:
            raise ValueError("descriptor was not inherited")
        os.fstat(fd)
        return fd

else:
    import subprocess
    import tempfile

    def _private_af_unix_pair():
        """A connected AF_UNIX pair built by hand, in a private directory.

        Winsock has no ``socketpair`` for AF_UNIX, so this is the emulation
        Microsoft's own documentation implies: bind a path, connect to it,
        accept.  The difference from ``socket.socketpair()`` is what a third
        party can address -- a path in a directory only this user can write,
        not a port any local process can reach.  That path exists for the
        bind/connect/accept instant, so ``accept`` is confirmed against our own
        connect before either end is returned.
        """
        directory = tempfile.mkdtemp(prefix="loki-ipc-")
        path = os.path.join(directory, "socket")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client = None
        server = None
        try:
            listener.bind(path)
            listener.listen(1)
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(path)
            server, _ = listener.accept()
            # accept() cannot say whose connection it took, so prove the two
            # ends are each other's before the name is even removed.
            confirm_pair(server, client)
        except BaseException:
            listener.close()
            for end in (client, server):
                if end is not None:
                    end.close()
            with contextlib.suppress(OSError):
                os.unlink(path)
            with contextlib.suppress(OSError):
                os.rmdir(directory)
            raise
        # Only connecting needs the path; the connected pair outlives it, and
        # leaving it behind would block the next bind on the same name.
        listener.close()
        os.unlink(path)
        os.rmdir(directory)
        return server, client

    def owner_channel():
        """A private AF_UNIX pair; the parent closing its end signals the child."""
        parent_end, child_end = _private_af_unix_pair()
        parent_end.set_inheritable(False)
        child_end.set_inheritable(True)
        return parent_end, child_end

    def socket_pair():
        """The credential channel: a private AF_UNIX pair, not a loopback port."""
        return _private_af_unix_pair()

    def prepare_child_socket(end):
        """Keep ``end`` alive, inheritable, for the child to inherit."""
        end.set_inheritable(True)
        return end

    def reference(child_end) -> int:
        return int(child_end.fileno())

    def spawn_kwargs(references) -> dict:
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {
            "handle_list": [int(value) for value in references]}
        return {"startupinfo": startup}

    def close_end(end) -> None:
        end.close()

    def child_endpoint(value):
        """The child's view of a reference: the socket the handle names."""
        handle = int(value)
        if handle == 0:
            raise OSError("handle was not inherited")
        return socket.socket(fileno=handle)
