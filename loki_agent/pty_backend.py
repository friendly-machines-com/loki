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

import contextlib
import os
import signal
import struct
import sys


class PtyHandle:
    """One child on a fresh pseudo-terminal."""

    def read(self, max_bytes: int) -> bytes:
        """Block for output; empty bytes mean EOF, not temporary silence."""
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
        self.exit_code = None

    def read(self, max_bytes):
        import errno

        try:
            return os.read(self.master, max_bytes)
        except OSError as error:
            # Linux reports a closed PTY slave as EIO rather than EOF.
            if error.errno != errno.EIO:
                raise
            return b""

    def write(self, data):
        return os.write(self.master, data)

    def set_size(self, cols, rows):
        import fcntl
        import termios

        fcntl.ioctl(self.master, termios.TIOCSWINSZ,
                    struct.pack("HHHH", rows, cols, 0, 0))

    def poll(self):
        if self.exit_code is None:
            done, status = os.waitpid(self.pid, os.WNOHANG)
            if done:
                self.exit_code = os.waitstatus_to_exitcode(status)
        return self.exit_code

    def wait(self):
        if self.exit_code is None:
            _, status = os.waitpid(self.pid, 0)
            self.exit_code = os.waitstatus_to_exitcode(status)
        return self.exit_code

    def terminate(self):
        if self.poll() is None:
            # pty.fork gives the child its own session/process group. Signal
            # that group only while its unreaped leader still owns the ID;
            # the real Loki runtime shares this terminal with its supervisor.
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                # An immediate abort can precede the child's setsid/login_tty.
                # Its unreaped PID is still ours even if the group is not yet
                # established (or has already emptied).
                with contextlib.suppress(ProcessLookupError):
                    os.kill(self.pid, signal.SIGKILL)
            self.wait()

    def close(self):
        if self.master is not None:
            os.close(self.master)
            self.master = None


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
    # The child sets its size before exec; there is no parent-side setup
    # operation that could fail after spawning and orphan the child.
    return _PosixPty(pid, master)


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

    def read(self, max_bytes):
        from . import windows_api

        try:
            return windows_api.read_file(self.output_read, max_bytes)
        except windows_api.WindowsApiError as error:
            if error.status != windows_api.ERROR_BROKEN_PIPE:
                raise
            return b""

    def write(self, data):
        from . import windows_api

        return windows_api.write_file(self.input_write, data)

    def set_size(self, cols, rows):
        from . import windows_api

        windows_api.pseudoconsole_resize(self.hpc, cols, rows)

    def poll(self):
        from . import windows_api

        status = windows_api.wait_for_single_object(self.information.hProcess, 0)
        if status == windows_api.WAIT_TIMEOUT:
            return None
        if status != windows_api.WAIT_OBJECT_0:
            raise windows_api.WindowsApiError("PTY process wait failed", status=status)
        return windows_api.get_exit_code_process(self.information.hProcess)

    def wait(self):
        from . import windows_api

        status = windows_api.wait_for_single_object(
            self.information.hProcess, windows_api.INFINITE)
        if status != windows_api.WAIT_OBJECT_0:
            raise windows_api.WindowsApiError("PTY process wait failed", status=status)
        return windows_api.get_exit_code_process(self.information.hProcess)

    def terminate(self):
        from . import windows_api

        if self.information is not None and self.poll() is None:
            try:
                windows_api.terminate_process(self.information.hProcess)
            except windows_api.WindowsApiError:
                # Termination can race a normal exit, but a live child's
                # termination failure must not silently become success.
                if self.poll() is None:
                    raise
            self.wait()

    def close(self):
        from . import windows_api

        # Normal callers have drained EOF. Failed callers discard output by
        # closing its pipe first: ClosePseudoConsole must never wait for a
        # drain on this same thread. Attempt every release even if one fails.
        try:
            if self.output_read is not None:
                windows_api.close_handle(self.output_read)
                self.output_read = None
        finally:
            try:
                if self.input_write is not None:
                    windows_api.close_handle(self.input_write)
                    self.input_write = None
            finally:
                try:
                    if self.hpc is not None:
                        windows_api.pseudoconsole_close(self.hpc)
                        self.hpc = None
                finally:
                    if self.information is not None:
                        try:
                            if self.information.hThread:
                                windows_api.close_handle(self.information.hThread)
                                self.information.hThread = None
                        finally:
                            if self.information.hProcess:
                                windows_api.close_handle(self.information.hProcess)
                                self.information.hProcess = None
                        self.information = None


def _windows_spawn(argv, cols, rows, env, cwd):
    from . import windows_api

    handle = _WindowsPty(None, None, None, None)
    try:
        # The handle owns host ends as soon as they are acquired. This stack
        # owns only the temporary ends ConPTY duplicates during creation.
        with contextlib.ExitStack() as copies:
            input_read, handle.input_write = windows_api.create_pipe(inherit=False)
            copies.callback(windows_api.close_handle, input_read)
            handle.output_read, output_write = windows_api.create_pipe(inherit=False)
            copies.callback(windows_api.close_handle, output_write)
            handle.hpc = windows_api.pseudoconsole_create(
                cols, rows, input_read, output_write)
            handle.information = windows_api.create_process_with_pseudoconsole(
                argv[0], argv[1:], handle.hpc, environment=env, current_directory=cwd)
        # Root exit alone is not terminal EOF. Release our session reference
        # so ConPTY closes output after its last attached client disconnects.
        windows_api.pseudoconsole_release(handle.hpc)
        return handle
    except BaseException:
        try:
            handle.terminate()
        finally:
            handle.close()
        raise


if sys.platform != "win32":
    __all__ = ["PtyHandle", "spawn_pty"]
