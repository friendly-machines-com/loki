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
import threading
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

INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0

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

_ReadFile = _kernel32.ReadFile
_ReadFile.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                      ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p)
_ReadFile.restype = wintypes.BOOL

_CreateEventW = _kernel32.CreateEventW
_CreateEventW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL,
                          wintypes.LPCWSTR)
_CreateEventW.restype = wintypes.HANDLE

_SetEvent = _kernel32.SetEvent
_SetEvent.argtypes = (wintypes.HANDLE,)
_SetEvent.restype = wintypes.BOOL

_WaitForMultipleObjects = _kernel32.WaitForMultipleObjects
_WaitForMultipleObjects.argtypes = (wintypes.DWORD,
                                    ctypes.POINTER(wintypes.HANDLE),
                                    wintypes.BOOL, wintypes.DWORD)
_WaitForMultipleObjects.restype = wintypes.DWORD

_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = (wintypes.HANDLE,)
_CloseHandle.restype = wintypes.BOOL


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


# Concurrency rules for the reader thread.  There are no locks here; these are
# what replace them, and every one of them is load-bearing:
#
# 1. The loop thread owns all mutable state: the byte queue, the caller's
#    reader fields, and everything the bytes feed into.  The worker thread
#    reads the console and posts; it touches nothing else.
# 2. The only objects shared between threads are the stop event and the loop
#    reference.  The loop reference is written once, before the thread starts.
# 3. Bytes reach the loop only through ``loop.call_soon_threadsafe``, so
#    ``queue.put_nowait`` always runs on the loop thread.  Nothing calls into
#    the queue from anywhere else, and nothing reads it off the loop thread,
#    which is why a single-producer/single-consumer queue needs no lock.
# 4. Teardown order is fixed: set the stop event, join the thread, and only
#    then close the event.  No handle is closed, and no console mode restored,
#    while the thread could still be using it.
# 5. If the thread does not stop within the join timeout, raise.  A worker
#    holding a closed event handle is worse than a failed shutdown.
# 6. A post that races a closing loop raises ``RuntimeError`` inside
#    ``call_soon_threadsafe``; that one is ignored, because it means teardown
#    is already running.
# 7. The thread is a daemon only as a backstop for a path that never reaches
#    ``__aexit__``.  Normal shutdown stops and joins it explicitly; a
#    non-joined worker blocked in a wait is how a process hangs at exit.
class Reader:
    """One thread reading the console input handle for one byte reader."""

    def __init__(self, fd: int, loop, queue):
        self.fd = fd
        self.loop = loop
        self.queue = queue
        self.handle = None
        self.stop_event = None
        self.thread = None

    def start(self) -> None:
        self.handle = _handle(self.fd)
        self.stop_event = _CreateEventW(None, True, False, None)
        if not self.stop_event:
            raise OSError(ctypes.get_last_error(), "CreateEventW")
        self.thread = threading.Thread(
            target=self._run, name="loki-console-reader", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        waited = (wintypes.HANDLE * 2)(self.handle, self.stop_event)
        buffer = ctypes.create_string_buffer(4096)
        while True:
            index = _WaitForMultipleObjects(2, waited, False, INFINITE)
            if index != WAIT_OBJECT_0:
                # The stop event, or a wait failure nothing here can act on.
                return
            read = wintypes.DWORD()
            if not _ReadFile(self.handle, buffer, len(buffer),
                             ctypes.byref(read), None):
                return
            if read.value:
                self._post(buffer.raw[:read.value])

    def _post(self, data: bytes) -> None:
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, data)
        except RuntimeError:
            # The loop is closed; teardown is already in progress.
            pass

    def stop(self) -> None:
        if self.stop_event:
            _SetEvent(self.stop_event)
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                # Closing the event under a live thread is worse than failing.
                raise RuntimeError("console reader thread did not stop")
            self.thread = None
        if self.stop_event:
            _CloseHandle(self.stop_event)
            self.stop_event = None
