"""Portable overlapped-call tests; these are not native Windows qualification."""

import asyncio
from contextlib import contextmanager
import ctypes
import unittest
from unittest import mock
import weakref

from loki_agent import windows_api as api


class OverlappedPipeTests(unittest.IsolatedAsyncioTestCase):
    @contextmanager
    def calls(self, *, pending=False, terminal=0, cancel_error=0, short=False):
        state = {'error': 0, 'buffer': None, 'record': None, 'written': bytearray()}
        submitted, cancelled, released = asyncio.Event(), asyncio.Event(), asyncio.Event()
        if not pending:
            released.set()

        def submit(handle, buffer, size, count, record):
            self.assertEqual(handle, 77)
            state.update(buffer=weakref.ref(buffer), record=ctypes.cast(record, ctypes.c_void_p).value, size=size)
            submitted.set()
            if pending:
                state['error'] = 997
                return False
            return True

        def result(handle, record, count, wait):
            self.assertEqual(handle, 77)
            self.assertFalse(wait)
            self.assertEqual(ctypes.cast(record, ctypes.c_void_p).value, state['record'])
            if not released.is_set():
                state['error'] = 996
                return False
            if terminal:
                state['error'] = terminal
                return False
            buffer = state['buffer']()
            self.assertIsNotNone(buffer, 'native operation lost its submitted buffer')
            amount = min(state['size'], 2) if short else min(state['size'], 6)
            if short:
                state['written'].extend(ctypes.string_at(buffer, amount))
            else:
                ctypes.memmove(buffer, b'reply\n', amount)
            ctypes.cast(count, ctypes.POINTER(ctypes.wintypes.DWORD))[0] = amount
            return True

        def cancel(handle, record):
            self.assertEqual(handle, 77)
            self.assertEqual(ctypes.cast(record, ctypes.c_void_p).value, state['record'])
            cancelled.set()
            state['error'] = cancel_error
            return not cancel_error

        functions = {'ReadFile': submit, 'WriteFile': submit,
                     'GetOverlappedResult': result, 'CancelIoEx': cancel}
        with mock.patch.object(api, 'bind', side_effect=lambda library, name, *types: functions[name]), \
                mock.patch.object(ctypes, 'get_last_error', side_effect=lambda: state['error'], create=True):
            yield [state, submitted, cancelled, released]

    async def test_read_uses_the_existing_overlapped_layout_and_exact_completion(self):
        with self.calls():
            self.assertEqual(await api.read_pipe(77, 6), b'reply\n')

    async def test_write_completes_every_short_transfer(self):
        with self.calls(short=True) as [state, _submitted, _cancelled, _released]:
            await api.write_pipe(77, b'abcdef')
            self.assertEqual(state['written'], b'abcdef')

    async def test_broken_read_pipe_is_eof(self):
        with self.calls(terminal=109):
            self.assertEqual(await api.read_pipe(77), b'')

    async def test_unexpected_completion_failure_is_not_eof(self):
        with self.calls(terminal=5), self.assertRaises(api.WindowsApiError) as caught:
            await api.read_pipe(77)
        self.assertEqual(caught.exception.status, 5)

    async def _cancel_pending(self, cancel_error=0):
        with self.calls(pending=True, terminal=995, cancel_error=cancel_error) as [
                state, submitted, cancelled, released]:
            task = asyncio.create_task(api.read_pipe(77))
            try:
                await submitted.wait()
                task.cancel()
                await cancelled.wait()
                self.assertFalse(task.done())
                self.assertIsNotNone(state['buffer']())
                task.cancel()
                released.set()
                with self.assertRaises(api.WindowsApiError if cancel_error else asyncio.CancelledError) as caught:
                    await task
                if cancel_error:
                    self.assertEqual(caught.exception.status, cancel_error)
            finally:
                released.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_repeated_cancellation_keeps_buffer_until_kernel_completion(self):
        await self._cancel_pending()

    async def test_failed_cancel_still_joins_completion_and_reports_the_error(self):
        await self._cancel_pending(cancel_error=5)
