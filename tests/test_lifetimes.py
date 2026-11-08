"""Cleanup ownership and error priority, without elapsed-time assumptions."""

import asyncio
import unittest

from loki_agent.lifetimes import finish_cleanup


class CleanupLifetimeTests(unittest.IsolatedAsyncioTestCase):
    async def _cancel_during_cleanup(self, *, preserve_failure=False, cleanup_failure=None):
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = OSError('creation failed')

        async def cleanup():
            started.set()
            try:
                await release.wait()
                if cleanup_failure is not None:
                    raise cleanup_failure
            finally:
                finished.set()

        async def caller():
            if preserve_failure:
                try:
                    raise original
                except OSError:
                    await finish_cleanup(cleanup(), preserve_failure=True)
                    raise
            await finish_cleanup(cleanup())

        task = asyncio.create_task(caller())
        try:
            await started.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertFalse(finished.is_set())
            task.cancel()
            release.set()
            expected = (type(cleanup_failure) if cleanup_failure is not None else
                        OSError if preserve_failure else asyncio.CancelledError)
            with self.assertRaises(expected) as caught:
                await task
            if cleanup_failure is not None:
                self.assertIs(caught.exception, cleanup_failure)
            elif preserve_failure:
                self.assertIs(caught.exception, original)
            self.assertTrue(finished.is_set())
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_repeated_cancellation_waits_for_actual_cleanup(self):
        await self._cancel_during_cleanup()

    async def test_creation_failure_survives_later_caller_cancellation(self):
        await self._cancel_during_cleanup(preserve_failure=True)

    async def test_cleanup_failure_is_not_hidden_by_creation_failure_or_cancellation(self):
        await self._cancel_during_cleanup(
            preserve_failure=True, cleanup_failure=ValueError('retirement failed'))

    async def test_cancelled_waiter_cannot_detach_the_shared_cleanup_owner(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def cleanup():
            started.set()
            await release.wait()
            return 'retired'

        owner = asyncio.create_task(cleanup())
        first = asyncio.create_task(finish_cleanup(owner))
        second = asyncio.create_task(finish_cleanup(owner))
        try:
            await started.wait()
            first.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertEqual(await second, 'retired')
            self.assertFalse(owner.cancelled())
        finally:
            release.set()
            await asyncio.gather(first, second, owner, return_exceptions=True)

    async def test_cancelled_cleanup_is_not_mistaken_for_success(self):
        async def cleanup():
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

        with self.assertRaises(asyncio.CancelledError):
            await finish_cleanup(cleanup(), preserve_failure=True)
