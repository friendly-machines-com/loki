"""Windows console control for the terminal frontend.

Only what differs from a POSIX tty lives here; ``terminals`` keeps the reader
state machine and the rendering.

* **Output needs nothing.**  Loki writes through Python's console stream, which
  uses ``WriteConsoleW`` (PEP 528), so the console code page never enters the
  output path.  The one output-side change is
  ``ENABLE_VIRTUAL_TERMINAL_PROCESSING``, so the ANSI sequences this frontend
  emits reach the screen.
* **Raw input** is ``SetConsoleMode`` with ``ENABLE_LINE_INPUT``,
  ``ENABLE_ECHO_INPUT`` and ``ENABLE_PROCESSED_INPUT`` cleared and
  ``ENABLE_VIRTUAL_TERMINAL_INPUT`` set.  That makes reads return when
  characters are available, stops the echo, delivers Ctrl+C as a byte instead
  of a signal, and delivers special keys as VT sequences rather than as
  ``KEY_EVENT`` records -- so the reader still produces the same byte stream a
  POSIX tty does.  The input code page is set to UTF-8 so typed non-ASCII
  arrives as UTF-8 bytes on that path.
* **Input is read by a thread.**  A console handle cannot be registered with
  the proactor loop (``add_reader`` is selector-only) and does not support
  overlapped I/O, so there is nothing to hand the loop.  The thread waits on
  *{console input, stop event}* so shutdown is deterministic rather than
  depending on input that may never arrive, reads with ``ReadFile``, and posts
  to the loop with ``call_soon_threadsafe``.
"""

from __future__ import annotations

import ctypes
import msvcrt
from ctypes import wintypes


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


ENABLE_PROCESSED_OUTPUT = 0x0001
ENABLE_PROCESSED_INPUT = 0x0001
ENABLE_LINE_INPUT = 0x0002
ENABLE_ECHO_INPUT = 0x0004
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200

CP_UTF8 = 65001
STD_OUTPUT_HANDLE = -11

_GetConsoleMode = _kernel32.GetConsoleMode
_GetConsoleMode.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
_GetConsoleMode.restype = wintypes.BOOL

_SetConsoleMode = _kernel32.SetConsoleMode
_SetConsoleMode.argtypes = (wintypes.HANDLE, wintypes.DWORD)
_SetConsoleMode.restype = wintypes.BOOL

_GetConsoleCP = _kernel32.GetConsoleCP
_GetConsoleCP.restype = wintypes.UINT

_SetConsoleCP = _kernel32.SetConsoleCP
_SetConsoleCP.argtypes = (wintypes.UINT,)
_SetConsoleCP.restype = wintypes.BOOL

_GetStdHandle = _kernel32.GetStdHandle
_GetStdHandle.argtypes = (wintypes.DWORD,)
_GetStdHandle.restype = wintypes.HANDLE


def raw_input_mode(mode: int) -> int:
    """The console input mode that makes reads byte-oriented and unbuffered."""
    return ((mode & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT
                      | ENABLE_PROCESSED_INPUT))
            | ENABLE_VIRTUAL_TERMINAL_INPUT)


def _handle(fd: int):
    return wintypes.HANDLE(msvcrt.get_osfhandle(fd))


def _check(result, what: str):
    if not result:
        raise OSError(ctypes.get_last_error(), what)
    return result


def _console_mode(handle) -> int:
    mode = wintypes.DWORD()
    _check(_GetConsoleMode(handle, ctypes.byref(mode)), "GetConsoleMode")
    return mode.value


def _set_console_mode(handle, mode: int) -> None:
    _check(_SetConsoleMode(handle, mode), "SetConsoleMode")


def _enable_virtual_terminal_output() -> None:
    """Let ANSI sequences reach the screen; a no-op when output is redirected."""
    handle = _GetStdHandle(STD_OUTPUT_HANDLE)
    mode = wintypes.DWORD()
    if not _GetConsoleMode(handle, ctypes.byref(mode)):
        return
    _set_console_mode(handle, mode.value | ENABLE_PROCESSED_OUTPUT
                      | ENABLE_VIRTUAL_TERMINAL_PROCESSING)


class RawMode:
    """Raw console input for one interactive session, restored on exit."""

    def __init__(self, fd: int):
        self.fd = fd
        self.handle = None
        self.old_mode = None
        self.old_cp = None

    def __enter__(self):
        self.handle = _handle(self.fd)
        self.old_mode = _console_mode(self.handle)
        _set_console_mode(self.handle, raw_input_mode(self.old_mode))
        self.old_cp = _GetConsoleCP()
        if self.old_cp != CP_UTF8:
            _SetConsoleCP(CP_UTF8)
        _enable_virtual_terminal_output()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.restore()

    def restore(self) -> None:
        if self.old_cp is not None:
            _SetConsoleCP(self.old_cp)
            self.old_cp = None
        if self.old_mode is not None:
            _set_console_mode(self.handle, self.old_mode)
            self.old_mode = None


def control_bytes(fallbacks):
    """The console exposes no erase/word-erase/interrupt character.

    There is no equivalent of termios' ``VERASE``/``VWERASE``/``VINTR`` to
    query, so the caller's documented defaults stand.
    """
    return fallbacks


def processed_input_enabled(fd: int) -> bool:
    """Whether ``ENABLE_PROCESSED_INPUT`` is set on ``fd``'s console.

    This is the Windows analogue of POSIX ``ISIG``: with the flag set, Ctrl+C
    becomes a ``CTRL_C_EVENT`` before the byte reaches a reader; with it clear
    the byte arrives as ``0x03``.  ``RawMode`` clears it.
    """
    return bool(_console_mode(_handle(fd)) & ENABLE_PROCESSED_INPUT)
