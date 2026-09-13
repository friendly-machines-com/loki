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

Both channels are therefore the same shape on Windows: an AF_UNIX pair, parent
end non-inheritable, child end inheritable and named in the handle list.
"""

from __future__ import annotations

import contextlib
import os
import socket

if os.name == "posix":
    def owner_channel():
        """(parent end, child end) for the session-lifetime channel."""
        read_fd, write_fd = os.pipe()
        return write_fd, read_fd

    def socket_pair():
        """A connected AF_UNIX pair; the platform has ``socketpair``."""
        return socket.socketpair()

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
        not a port any local process can reach.
        """
        directory = tempfile.mkdtemp(prefix="loki-ipc-")
        path = os.path.join(directory, "socket")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client = None
        try:
            listener.bind(path)
            listener.listen(1)
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(path)
            server, _ = listener.accept()
        except BaseException:
            listener.close()
            if client is not None:
                client.close()
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
