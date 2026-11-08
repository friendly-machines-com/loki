# pytype: skip-file
"""Read a Windows handle from a thread, posting bytes to an asyncio loop.

These input handles are synchronous; consoles cannot supply overlapped reads
for the proactor, and ``add_reader`` is selector-only. The existing reader
thread waits on *{stop event, handle}*, reads with ``ReadFile``, and posts with
``call_soon_threadsafe``. Console readiness also includes non-key notifications:
consume those before ReadFile, leaving keyboard/VT translation with Windows.
Shutdown still cancels an entered synchronous read without needing another key.

Concurrency rules. A one-shot startup future transfers the native capability;
the byte path otherwise uses loop ownership instead of shared-state locks:

1. The loop thread owns the byte queue and everything the bytes feed into.
   Every worker publishes startup once through a thread-safe future before
   reading; a console worker also transfers its exact cancellation handle.
2. The input handle, pipe/console choice, stop event, loop reference and startup
   future are fixed before the thread starts. Once published, the real thread
   handle belongs to the loop thread; the reader never uses or closes it.
3. Bytes reach the loop only through ``loop.call_soon_threadsafe``, so
   ``queue.put_nowait`` always runs on the loop thread -- a single-producer,
   single-consumer queue with no lock.
4. Teardown requests stop, awaits actual thread exit without blocking the
   event loop, then closes owned handles. A live thread never loses its event.
5. No elapsed budget establishes completion. Cancellation failure retains a
   live reader's exact capabilities instead of reopening a thread by ID. After
   observed exit, each native close consumes its slot once and reports failure;
   a possibly invalid numeric handle must not be retried by outer cleanup.
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

import asyncio
import ctypes
import sys
import threading
from concurrent.futures import CancelledError as FutureCancelledError, Future
from ctypes import wintypes

from . import windows_api
from .windows_api import FILE_TYPE_PIPE, INFINITE, WAIT_OBJECT_0
from .lifetimes import finish_cleanup


class HandleReader:
    """One thread reading one handle for one byte consumer."""

    def __init__(self, fd: int, loop, queue,
                 eof_sentinel: bool = False):
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
        self._pipe = False
        self._cancel_thread = None
        self._accepting = False
        self._delivery_task = None

    def start(self) -> None:
        if sys.platform != "win32":
            raise windows_api.WindowsUnavailableError(
                "the handle reader is available on Windows only")
        if self.stop_event is not None or self.thread is not None or self._cancel_thread is not None:
            raise RuntimeError("handle reader still owns resources")
        self.handle = _handle(self.fd)
        try:
            self._pipe = _GetFileType(self.handle) == FILE_TYPE_PIPE
            event = _CreateEventW(None, True, False, None)
            if not event:
                raise OSError(ctypes.get_last_error(), "CreateEventW")
            self.stop_event = event
            # Publish startup even for pipes: an interrupted Thread.start is
            # not proof that its worker cannot still enter. Console readers
            # additionally transfer their exact native cancellation capability;
            # Thread.start supplies no handle and a private DACL can deny reopen.
            # This local handoff covers acquisition, never input/network I/O.
            self._cancel_thread = Future()
            self.thread = threading.Thread(
                target=self._run, name="loki-handle-reader", daemon=True)
            self._accepting = True
            self.thread.start()
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
            handle = None if self._pipe else _DuplicateCurrentThread()
        except BaseException as error:
            # No blocking read has begun. The owner still joins actual exit.
            cancellation.set_exception(error)
            return
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
            try:
                if windows_api.discard_console_notification(self.handle):
                    continue
            except windows_api.WindowsApiError as error:
                self._post(OSError(error.status, str(error)))
                break
            read = wintypes.DWORD()
            if not _ReadFile(self.handle, buffer, len(buffer),
                             ctypes.byref(read), None):
                self._post(OSError(ctypes.get_last_error(), 'ReadFile'))
                break
            if not read.value:
                break
            if not self._post(buffer.raw[:read.value]):
                return

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
            if read.value and not self._post(buffer.raw[:read.value]):
                return

    def _post(self, data: bytes | Exception) -> bool:
        # ACP's bounded queue must backpressure this existing native thread,
        # not accumulate call_soon callbacks or drop already accepted bytes.
        # Unbounded terminal queues retain their direct callback delivery.
        if self.queue.maxsize > 0:
            delivery = self._deliver_bounded(data)
            try:
                acknowledged = asyncio.run_coroutine_threadsafe(delivery, self.loop)
            except RuntimeError:
                delivery.close()
                return False
            try:
                return acknowledged.result()
            except FutureCancelledError:
                return False
        try:
            self.loop.call_soon_threadsafe(self._deliver, data)
            return True
        except RuntimeError:
            # The loop is closed; teardown is already in progress.
            return False

    async def _deliver_bounded(self, data):
        # This task/acceptance decision belongs only to the loop. If stop wins
        # before its first step, it must never enqueue or strand the producer.
        if not self._accepting:
            return False
        self._delivery_task = asyncio.current_task()
        try:
            await self.queue.put(data)
            return True
        finally:
            self._delivery_task = None

    def _deliver(self, data: bytes) -> None:
        # Runs on the owning loop. A post queued before stop() must not deliver
        # input after shutdown has begun, even when joining the thread fails.
        if self._accepting:
            self.queue.put_nowait(data)

    def stop(self) -> None:
        """Revoke delivery and request stop; completion is asynchronous."""
        self._accepting = False
        if self._delivery_task is not None:
            self._delivery_task.cancel()
        if self.stop_event and not _SetEvent(self.stop_event):
            raise OSError(ctypes.get_last_error(), "SetEvent")
        cancellation = self._cancel_thread
        if (self.thread is not None and self.thread.is_alive() and cancellation is not None
                and cancellation.done() and cancellation.exception() is None):
            handle = cancellation.result()
            if handle is not None and not _CancelSynchronousIo(handle):
                error = ctypes.get_last_error()
                if error != windows_api.ERROR_NOT_FOUND:
                    raise OSError(error, "CancelSynchronousIo")

    def wait_closed(self):
        """Capture and join this thread generation without blocking the loop.

        A pending startup handoff is not proof of an unstarted/exited worker.
        Capture before yielding so concurrent closers cannot consume a later
        start's resources. Cancel/read-entry races retain the same live-thread
        capability. Only after actual exit do native release attempts consume
        their records, including on reported close failure.
        """
        thread = self.thread
        startup = self._cancel_thread

        async def joined():
            while thread is not None:
                if startup is None:
                    raise RuntimeError('reader startup ownership is missing')
                if startup.done():
                    if not thread.is_alive():
                        break
                    if not self._accepting:
                        self.stop()
                try:
                    await asyncio.sleep(.01)
                except asyncio.CancelledError:
                    if self._accepting:
                        raise
            if self._cancel_thread is not startup or (self.thread is not None and self.thread is not thread):
                return
            if thread is not None:
                thread.join()  # Actual exit is established; this cannot wait for I/O.
                self.thread = None
            self._cancel_thread = None
            try:
                if startup is not None and startup.done() and startup.exception() is None:
                    handle = startup.result()
                    if handle is not None and not _CloseHandle(handle):
                        raise OSError(ctypes.get_last_error(), "CloseHandle(thread)")
            finally:
                event, self.stop_event = self.stop_event, None
                if event and not _CloseHandle(event):
                    raise OSError(ctypes.get_last_error(), "CloseHandle")

        return joined()

    @property
    def closed(self):
        """No live reader or owned release attempt remains.

        Failed native closes still raise; consumed records are not proof of
        physical release and must not trigger a retry on a possibly reused value.
        """
        return self.thread is None and self.stop_event is None and self._cancel_thread is None

    async def aclose(self):
        try:
            self.stop()
        except BaseException:
            await finish_cleanup(self.wait_closed(), preserve_failure=True)
            raise
        await finish_cleanup(self.wait_closed())


# The calls are declared once through ``windows_api.bind`` (one binding per
# symbol) and resolved on first use, so importing this module off Windows binds
# nothing.  They stay module-level functions rather than instance attributes so
# a test can replace them exactly as it would any other call.
def _handle(fd: int):
    import msvcrt
    return wintypes.HANDLE(msvcrt.get_osfhandle(fd))


def _ReadFile(handle, buffer, size, read, overlapped):
    return windows_api.bind(
        "kernel32", "ReadFile", wintypes.BOOL, ctypes.c_void_p,
        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p)(handle, buffer, size, read, overlapped)


def _WaitForMultipleObjects(count, handles, wait_all, timeout):
    return windows_api.bind(
        "kernel32", "WaitForMultipleObjects", wintypes.DWORD,
        wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
        wintypes.BOOL, wintypes.DWORD)(count, handles, wait_all, timeout)


def _CreateEventW(attributes, manual_reset, initial, name):
    return windows_api.bind(
        "kernel32", "CreateEventW", ctypes.c_void_p,
        ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL,
        wintypes.LPCWSTR)(attributes, manual_reset, initial, name)


def _SetEvent(handle):
    return windows_api.bind("kernel32", "SetEvent", wintypes.BOOL,
                            ctypes.c_void_p)(handle)


def _DuplicateCurrentThread():
    # Called on the reader, not the loop thread: a pseudo-handle denotes its
    # caller. Duplicate it before ReadFile, requesting only cancellation rights
    # and no inheritance. This does not broaden the thread's private DACL.
    process = windows_api.current_process_handle()
    thread = windows_api.bind("kernel32", "GetCurrentThread", ctypes.c_void_p)()
    result = ctypes.c_void_p()
    duplicate = windows_api.bind(
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
    return windows_api.bind("kernel32", "CancelSynchronousIo", wintypes.BOOL,
                            ctypes.c_void_p)(thread)


def _CloseHandle(handle):
    return windows_api.bind("kernel32", "CloseHandle", wintypes.BOOL,
                            ctypes.c_void_p)(handle)


def _GetFileType(handle):
    return windows_api.bind("kernel32", "GetFileType", wintypes.DWORD,
                            ctypes.c_void_p)(handle)


def _PeekNamedPipe(handle, buffer, size, read, available, left):
    return windows_api.bind(
        "kernel32", "PeekNamedPipe", wintypes.BOOL,
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD))(handle, buffer, size, read, available, left)


def _WaitForSingleObject(handle, timeout):
    return windows_api.bind("kernel32", "WaitForSingleObject", wintypes.DWORD,
                            ctypes.c_void_p, wintypes.DWORD)(handle, timeout)
