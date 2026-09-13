"""Read a Windows handle from a thread, posting bytes to an asyncio loop.

A handle cannot be registered with the proactor loop (``add_reader`` is
selector-only) and a console handle does not support overlapped I/O, so there
is nothing to hand the loop.  The thread waits on *{handle, stop event}* so
shutdown is deterministic rather than depending on input that may never
arrive, reads with ``ReadFile``, and posts with ``call_soon_threadsafe``.

Concurrency rules.  There are no locks here; these replace them, and every one
is load-bearing:

1. The loop thread owns all mutable state: the byte queue and everything the
   bytes feed into.  The worker thread reads and posts; it touches nothing else.
2. The only objects shared between threads are the stop event and the loop
   reference.  The loop reference is written once, before the thread starts.
3. Bytes reach the loop only through ``loop.call_soon_threadsafe``, so
   ``queue.put_nowait`` always runs on the loop thread -- a single-producer,
   single-consumer queue with no lock.
4. Teardown order is fixed: set the stop event, join the thread, and only then
   close the event.  No handle is closed while the thread could still use it.
5. If the thread does not stop within the join timeout, raise.  A worker
   holding a closed event handle is worse than a failed shutdown.
6. A post that races a closing loop raises ``RuntimeError`` inside
   ``call_soon_threadsafe``; that one is ignored, because it means teardown is
   already running.
7. The thread is a daemon only as a backstop for a path that never reaches
   teardown.  Normal shutdown stops and joins it explicitly.

Import-safe off Windows: nothing is bound until ``start``, which only a
Windows caller reaches.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes

from . import windows_api

INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0


class HandleReader:
    """One thread reading one handle for one byte consumer."""

    def __init__(self, fd: int, loop, queue, eof_sentinel: bool = False):
        self.fd = fd
        self.loop = loop
        self.queue = queue
        # A pipe ends; a console does not.  The consumer that can see EOF asks
        # for an empty post when the read loop ends, so it does not wait for
        # input that will never come.
        self.eof_sentinel = eof_sentinel
        self.handle = None
        self.stop_event = None
        self.thread = None

    def start(self) -> None:
        if sys.platform != "win32":
            raise windows_api.WindowsUnavailableError(
                "the handle reader is available on Windows only")
        self.handle = _handle(self.fd)
        self.stop_event = _CreateEventW(None, True, False, None)
        if not self.stop_event:
            raise OSError(ctypes.get_last_error(), "CreateEventW")
        self.thread = threading.Thread(
            target=self._run, name="loki-handle-reader", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        waited = (wintypes.HANDLE * 2)(self.handle, self.stop_event)
        buffer = ctypes.create_string_buffer(4096)
        while True:
            if _WaitForMultipleObjects(2, waited, False, INFINITE) != WAIT_OBJECT_0:
                # The stop event, or a wait failure nothing here can act on.
                break
            read = wintypes.DWORD()
            if not _ReadFile(self.handle, buffer, len(buffer),
                             ctypes.byref(read), None):
                break
            if read.value:
                self._post(buffer.raw[:read.value])
        if self.eof_sentinel:
            self._post(b"")

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
                raise RuntimeError("handle reader thread did not stop")
            self.thread = None
        if self.stop_event:
            _CloseHandle(self.stop_event)
            self.stop_event = None


# The calls are declared once through ``windows_api.bind`` (one binding per
# symbol) and resolved on first use, so importing this module off Windows binds
# nothing.  They stay module-level functions rather than instance attributes so
# a test can replace them exactly as it would any other call.
def _handle(fd: int):
    import msvcrt
    return wintypes.HANDLE(msvcrt.get_osfhandle(fd))


_calls = {}


def _call(library, symbol, restype, *argtypes):
    key = (library, symbol)
    bound = _calls.get(key)
    if bound is None:
        bound = _calls[key] = windows_api.bind(
            library, symbol, restype, *argtypes)
    return bound


def _ReadFile(handle, buffer, size, read, overlapped):
    return _call("kernel32", "ReadFile", wintypes.BOOL, ctypes.c_void_p,
                 ctypes.c_void_p, wintypes.DWORD,
                 ctypes.POINTER(wintypes.DWORD),
                 ctypes.c_void_p)(handle, buffer, size, read, overlapped)


def _WaitForMultipleObjects(count, handles, wait_all, timeout):
    return _call("kernel32", "WaitForMultipleObjects", wintypes.DWORD,
                 wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
                 wintypes.BOOL, wintypes.DWORD)(count, handles, wait_all,
                                                timeout)


def _CreateEventW(attributes, manual_reset, initial, name):
    return _call("kernel32", "CreateEventW", ctypes.c_void_p,
                 ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL,
                 wintypes.LPCWSTR)(attributes, manual_reset, initial, name)


def _SetEvent(handle):
    return _call("kernel32", "SetEvent", wintypes.BOOL,
                 ctypes.c_void_p)(handle)


def _CloseHandle(handle):
    return _call("kernel32", "CloseHandle", wintypes.BOOL,
                 ctypes.c_void_p)(handle)
