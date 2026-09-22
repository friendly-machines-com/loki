"""Pseudo-terminal harness for the TUI test suite.

Intended for test use only: it runs a child with a terminal it does not
share, so a test can observe the exact byte stream the child writes and drive
its stdin.  POSIX uses ``pty.fork``; Windows uses ConPTY (``CreatePseudoConsole``
plus a plain ``CreateProcessW`` carrying ``PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE``
and ``STARTF_USESTDHANDLES`` with null handles).

The two platforms do not agree on what the read side returns.  A POSIX pty is
transparent: every byte is the child's own.  ConPTY's output pipe is conhost's
*rendering* of the child's console, so it adds its own lifecycle sequences
(mode queries, cursor hide/show, a title OSC naming the child image) and may
re-serialize the child's SGR spellings.  ``read`` therefore returns raw bytes;
tests must parse, not strip.
"""

from __future__ import annotations

import os
import signal
import struct
import sys
import time


class PtyHandle:
    """One child on a fresh pseudo-terminal."""

    def read(self, max_bytes: int, timeout: float) -> bytes:
        raise NotImplementedError

    def write(self, data: bytes) -> int:
        raise NotImplementedError

    def set_size(self, cols: int, rows: int) -> None:
        raise NotImplementedError

    def poll(self):
        """Exit code, or ``None`` while the child is still running."""
        raise NotImplementedError

    def wait(self) -> int:
        raise NotImplementedError

    def terminate(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def spawn_pty(argv, *, cols=80, rows=24, env=None, cwd=None) -> PtyHandle:
    """Spawn ``argv`` on a fresh pseudo-terminal of ``cols`` x ``rows``.

    ``argv`` is the full command line, as ``os.execve`` / ``CreateProcessW``
    take it.  ``env`` is a full environment mapping or ``None`` to inherit.
    ``cwd`` is the child's working directory or ``None`` to inherit.
    """
    if os.name == "posix":
        return _posix_spawn(argv, cols, rows, env, cwd)
    return _windows_spawn(argv, cols, rows, env, cwd)


# -- POSIX -------------------------------------------------------------


class _PosixPty(PtyHandle):
    def __init__(self, pid, master):
        self.pid = pid
        self.master = master

    def read(self, max_bytes, timeout):
        import select

        r, _, _ = select.select([self.master], [], [], timeout)
        if not r:
            return b""
        try:
            return os.read(self.master, max_bytes)
        except OSError:
            return b""

    def write(self, data):
        return os.write(self.master, data)

    def set_size(self, cols, rows):
        import fcntl
        import termios

        fcntl.ioctl(self.master, termios.TIOCSWINSZ,
                    struct.pack("HHHH", rows, cols, 0, 0))

    def poll(self):
        try:
            done, status = os.waitpid(self.pid, os.WNOHANG)
        except OSError:
            return 0
        if done:
            return os.waitstatus_to_exitcode(status)
        return None

    def wait(self):
        try:
            _, status = os.waitpid(self.pid, 0)
        except OSError:
            return 0
        return os.waitstatus_to_exitcode(status)

    def terminate(self):
        for signum in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(self.pid, signum)
            except OSError:
                return
            time.sleep(0.2)

    def close(self):
        try:
            os.close(self.master)
        except OSError:
            pass


def _posix_spawn(argv, cols, rows, env, cwd):
    import pty

    pid, master = pty.fork()
    if pid == 0:  # child
        try:
            _child_set_size(0, cols, rows)
            if cwd is not None:
                os.chdir(cwd)
            os.execve(argv[0], list(argv), dict(env or os.environ))
        except Exception:
            pass
        os._exit(127)
    handle = _PosixPty(pid, master)
    handle.set_size(cols, rows)
    return handle


def _child_set_size(fd, cols, rows):
    import fcntl
    import termios

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


# -- Windows (ConPTY) --------------------------------------------------


class _WindowsPty(PtyHandle):
    def __init__(self, hpc, output_read, input_write, information):
        self.hpc = hpc
        self.output_read = output_read
        self.input_write = input_write
        self.information = information
        self._closed = False

    def read(self, max_bytes, timeout):
        import ctypes
        from ctypes import wintypes

        from . import windows_api

        # A raw ReadFile, not windows_api.read_file: a broken pipe is a normal
        # end-of-session here, not an error.  The signature matches the one
        # windows_api already records, so bind returns the same object.
        read = windows_api.bind(
            "kernel32", "ReadFile", wintypes.BOOL, ctypes.c_void_p,
            ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p)
        deadline = time.monotonic() + timeout
        while True:
            available = windows_api.peek_named_pipe(self.output_read)
            if available:
                size = min(available, max_bytes)
                buffer = ctypes.create_string_buffer(size)
                count = wintypes.DWORD()
                if read(self.output_read, ctypes.cast(
                        buffer, ctypes.c_void_p), size,
                        ctypes.byref(count), None):
                    return buffer.raw[:count.value]
                return b""
            if windows_api.wait_for_single_object(
                    self.information.hProcess, 0) == 0:
                return b""
            if time.monotonic() >= deadline:
                return b""
            time.sleep(0.01)

    def write(self, data):
        from . import windows_api

        return windows_api.write_file(self.input_write, data)

    def set_size(self, cols, rows):
        from . import windows_api

        windows_api.pseudoconsole_resize(self.hpc, cols, rows)

    def poll(self):
        from . import windows_api

        if windows_api.wait_for_single_object(
                self.information.hProcess, 0) != 0:
            return None
        return windows_api.get_exit_code_process(self.information.hProcess)

    def wait(self):
        from . import windows_api

        windows_api.wait_for_single_object(self.information.hProcess, 0xffffffff)
        return windows_api.get_exit_code_process(self.information.hProcess)

    def terminate(self):
        from . import windows_api

        try:
            windows_api.terminate_process(self.information.hProcess)
        except windows_api.WindowsApiError:
            pass
        windows_api.wait_for_single_object(self.information.hProcess, 0xffffffff)

    def close(self):
        if self._closed:
            return
        self._closed = True
        from . import windows_api

        for handle in (self.information.hProcess, self.information.hThread):
            windows_api.close_handle(handle)
        windows_api.pseudoconsole_close(self.hpc)
        windows_api.close_handle(self.output_read)
        windows_api.close_handle(self.input_write)


def _windows_spawn(argv, cols, rows, env, cwd):
    from . import windows_api

    # Host -> child input pipe, child -> host output pipe.
    input_read, input_write = windows_api.create_pipe(inherit=False)
    output_read, output_write = windows_api.create_pipe(inherit=False)
    hpc = windows_api.pseudoconsole_create(
        cols, rows, input_read, output_write)
    information = None
    try:
        information = windows_api.create_process_with_pseudoconsole(
            argv[0], argv[1:], hpc, environment=env, current_directory=cwd)
    finally:
        # The pseudoconsole keeps its own copies of the pipe ends; release
        # ours so a child exit shows up as a broken pipe.
        windows_api.close_handle(input_read)
        windows_api.close_handle(output_write)
        if information is None:
            windows_api.pseudoconsole_close(hpc)
            windows_api.close_handle(output_read)
            windows_api.close_handle(input_write)
            raise
    return _WindowsPty(hpc, output_read, input_write, information)


if sys.platform != "win32":
    __all__ = ["PtyHandle", "spawn_pty"]
