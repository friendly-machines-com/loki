# pytype: skip-file
"""Read a Windows handle from a thread, posting bytes to an asyncio loop.

These input handles are synchronous; consoles cannot supply overlapped reads
for the proactor, and ``add_reader`` is selector-only. The existing reader
thread waits on *{stop event, handle}*, reads with ``ReadFile``, and posts with
``call_soon_threadsafe``. Console readiness can mean non-byte events, so shutdown
also cancels synchronous I/O rather than depending on another key arriving.

Concurrency rules. A one-shot startup future transfers the native capability;
the byte path otherwise uses loop ownership instead of shared-state locks:

1. The loop thread owns the byte queue and everything the bytes feed into.
   A non-pipe worker publishes its cancellation handle once through a
   thread-safe startup future before it can enter a blocking read.
2. The input handle, pipe/console choice, stop event, loop reference and startup
   future are fixed before the thread starts. Once published, the real thread
   handle belongs to the loop thread; the reader never uses or closes it.
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
8. Delivery is gated on the loop thread once shutdown begins, including posts
   already queued before stop(). A failed join cannot keep delivering bytes.

Import-safe off Windows: nothing is bound until ``start``, which only a
Windows caller reaches.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from concurrent.futures import Future
from ctypes import wintypes

from . import windows_api
from .windows_api import FILE_TYPE_PIPE, INFINITE, WAIT_OBJECT_0


class HandleReader:
    """One thread reading one handle for one byte consumer."""

    def __init__(self, fd: int | None, loop, queue,
                 eof_sentinel: bool = False, handle=None):
        self.fd = fd
        self.loop = loop
        self.queue = queue
        # A pipe ends; a console does not.  The consumer that can see EOF asks
        # for an empty post when the read loop ends, so it does not wait for
        # input that will never come.
        self.eof_sentinel = eof_sentinel
        # A caller holding a raw Windows HANDLE (an anonymous pipe from
        # ``CreatePipe``) has no C descriptor to convert, so it passes the
        # handle directly and ``fd`` is None.
        self._explicit_handle = handle
        self.handle = None
        self.stop_event = None
        self.thread = None
        self._pipe = False
        self._cancel_thread = None
        self._accepting = False

    def start(self) -> None:
        if sys.platform != "win32":
            raise windows_api.WindowsUnavailableError(
                "the handle reader is available on Windows only")
        if self.stop_event is not None or self.thread is not None or self._cancel_thread is not None:
            raise RuntimeError("handle reader still owns resources")
        if self._explicit_handle is not None:
            self.handle = self._explicit_handle
        else:
            self.handle = _handle(self.fd)
        try:
            self._pipe = _GetFileType(self.handle) == FILE_TYPE_PIPE
            event = _CreateEventW(None, True, False, None)
            if not event:
                raise OSError(ctypes.get_last_error(), "CreateEventW")
            self.stop_event = event
            if not self._pipe:
                # Thread.start exposes an ID, not the creator's native handle.
                # A private thread DACL can deny reopening even inside this
                # process. This local startup handoff covers only native handle
                # acquisition, never input/network I/O. An asyncio callback
                # cannot serve it while this synchronous resource owner waits.
                # Keep the result in the future: interrupted startup must not
                # lose a handle published later. This future is never cancelled.
                self._cancel_thread = Future()
            self.thread = threading.Thread(
                target=self._run, name="loki-handle-reader", daemon=True)
            self._accepting = True
            self.thread.start()
            if self._cancel_thread is not None:
                self._cancel_thread.result()
        except BaseException as error:
            if (isinstance(error, RuntimeError) and self.thread is not None
                    and self.thread.ident is None):
                # Thread.start's refusal path has no thread to join. Other
                # interruptions may race native startup: retain that ownership
                # rather than close an event a starting thread could still use.
                self.thread = None
            self.stop()
            raise

    def _run(self) -> None:
        cancellation = self._cancel_thread
        try:
            buffer = ctypes.create_string_buffer(4096)
            if cancellation is not None:
                handle = _DuplicateCurrentThread()
        except BaseException as error:
            if cancellation is None:
                raise
            # No blocking read has begun. start() receives the actual failure
            # and can join this worker without a cancellation capability.
            cancellation.set_exception(error)
            return
        if cancellation is not None:
            cancellation.set_result(handle)
        if self._pipe:
            self._read_pipe(buffer)
        else:
            self._read_handle(buffer)
        if self.eof_sentinel:
            self._post(b"")

    def _read_handle(self, buffer) -> None:
        """A console can signal for non-byte events; ReadFile may still block.

        Stop wins simultaneous readiness. A read already entered (or racing
        entry) is cancelled by stop(), without changing VT byte translation.
        """
        waited = (wintypes.HANDLE * 2)(self.stop_event, self.handle)
        while True:
            if _WaitForMultipleObjects(2, waited, False, INFINITE) != WAIT_OBJECT_0 + 1:
                # The stop event, or a wait failure nothing here can act on.
                break
            read = wintypes.DWORD()
            if not _ReadFile(self.handle, buffer, len(buffer),
                             ctypes.byref(read), None):
                break
            if not read.value:
                break
            self._post(buffer.raw[:read.value])

    def _read_pipe(self, buffer) -> None:
        """A pipe: it must never block in ``ReadFile``.

        A pipe handle reports readiness whether or not a read would block, so
        reading first wedges the thread -- and a thread stuck in ``ReadFile``
        cannot be stopped.  Ask how many bytes a read can return, read only
        those, and wait on the stop event between polls.  A failed peek means
        the other end is gone, which is this stream's end of file.
        """
        available = wintypes.DWORD()
        total = wintypes.DWORD()
        while True:
            if _WaitForSingleObject(self.stop_event, 0) == WAIT_OBJECT_0:
                return
            if not _PeekNamedPipe(self.handle, None, 0, None,
                                  ctypes.byref(available), ctypes.byref(total)):
                return
            if not available.value:
                if _WaitForSingleObject(self.stop_event, 50) == WAIT_OBJECT_0:
                    return
                continue
            read = wintypes.DWORD()
            size = min(len(buffer), available.value)
            if not _ReadFile(self.handle, buffer, size, ctypes.byref(read), None):
                return
            if read.value:
                self._post(buffer.raw[:read.value])

    def _post(self, data: bytes) -> None:
        try:
            self.loop.call_soon_threadsafe(self._deliver, data)
        except RuntimeError:
            # The loop is closed; teardown is already in progress.
            pass

    def _deliver(self, data: bytes) -> None:
        # Runs on the owning loop. A post queued before stop() must not deliver
        # input after shutdown has begun, even when joining the thread fails.
        if self._accepting:
            self.queue.put_nowait(data)

    def _join_synchronous_reader(self):
        if not self.thread.is_alive():
            self.thread.join()
            return
        cancellation = self._cancel_thread
        if cancellation is None:
            raise RuntimeError("reader cancellation handoff is missing")
        deadline = time.monotonic() + 2  # Preserve the existing join budget.
        while self.thread.is_alive():
            # An interrupted start can reach stop() before publication. Never
            # discard that future or wait beyond the existing teardown budget.
            # Acquisition failure means the worker exits before reading.
            if cancellation.done() and cancellation.exception() is None:
                if not _CancelSynchronousIo(cancellation.result()):
                    error = ctypes.get_last_error()
                    if error != windows_api.ERROR_NOT_FOUND:
                        raise OSError(error, "CancelSynchronousIo")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            # A single cancellation can precede ReadFile and miss it. Retry
            # after yielding to the reader; never close its input/event under it.
            self.thread.join(timeout=min(0.01, remaining))

    def stop(self) -> None:
        self._accepting = False
        try:
            if self.stop_event and not _SetEvent(self.stop_event):
                raise OSError(ctypes.get_last_error(), "SetEvent")
        finally:
            try:
                if self.thread is not None:
                    try:
                        if self._pipe:
                            self.thread.join(timeout=2)
                        else:
                            self._join_synchronous_reader()
                    finally:
                        if self.thread.ident is not None and not self.thread.is_alive():
                            self.thread = None
                    if self.thread is not None:
                        raise RuntimeError("handle reader thread did not stop")
            finally:
                try:
                    # A failed join still needs this exact capability on the
                    # next stop(). Closing it now would force a forbidden TID
                    # reopen, and interrupted startup can publish it late.
                    if self.thread is None and self._cancel_thread is not None:
                        cancellation = self._cancel_thread
                        if cancellation.done() and cancellation.exception() is None:
                            if not _CloseHandle(cancellation.result()):
                                raise OSError(ctypes.get_last_error(), "CloseHandle(thread)")
                        self._cancel_thread = None
                finally:
                    # No event is closed under a live thread. A failed close
                    # retains ownership so a retry cannot close a reused number.
                    if self.thread is None and self.stop_event:
                        if not _CloseHandle(self.stop_event):
                            raise OSError(ctypes.get_last_error(), "CloseHandle")
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


def _DuplicateCurrentThread():
    # Called on the reader, not the loop thread: a pseudo-handle denotes its
    # caller. Duplicate it before ReadFile, requesting only cancellation rights
    # and no inheritance. This does not broaden the thread's private DACL.
    process = windows_api.current_process_handle()
    thread = _call("kernel32", "GetCurrentThread", ctypes.c_void_p)()
    result = ctypes.c_void_p()
    duplicate = _call(
        "kernel32", "DuplicateHandle", wintypes.BOOL,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD,
        wintypes.BOOL, wintypes.DWORD)
    if not duplicate(process, thread, process, ctypes.byref(result),
                     windows_api.ThreadAccess.THREAD_TERMINATE, False, 0):
        raise OSError(ctypes.get_last_error(), "DuplicateHandle(thread)")
    if not result.value:
        raise RuntimeError("reader cancellation handle is missing")
    return result.value


def _CancelSynchronousIo(thread):
    # https://learn.microsoft.com/en-us/windows/win32/fileio/cancelsynchronousio-func
    return _call("kernel32", "CancelSynchronousIo", wintypes.BOOL,
                 ctypes.c_void_p)(thread)


def _CloseHandle(handle):
    return _call("kernel32", "CloseHandle", wintypes.BOOL,
                 ctypes.c_void_p)(handle)


def _GetFileType(handle):
    return _call("kernel32", "GetFileType", wintypes.DWORD,
                 ctypes.c_void_p)(handle)


def _PeekNamedPipe(handle, buffer, size, read, available, left):
    return _call("kernel32", "PeekNamedPipe", wintypes.BOOL,
                 ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                 ctypes.POINTER(wintypes.DWORD),
                 ctypes.POINTER(wintypes.DWORD),
                 ctypes.POINTER(wintypes.DWORD))(handle, buffer, size, read,
                                                 available, left)


def _WaitForSingleObject(handle, timeout):
    return _call("kernel32", "WaitForSingleObject", wintypes.DWORD,
                 ctypes.c_void_p, wintypes.DWORD)(handle, timeout)
