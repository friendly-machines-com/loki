"""Shared fixture observations borrow producers and retire only their own waits."""

import asyncio
import unittest
from unittest import mock

from process_lifecycle_fixtures import observe_fixture, running_fixture_task


class ObserveFixtureTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_wait_observes_front_exit(self):
        async with running_fixture_task(mock.AsyncMock()()) as owner:
            before = asyncio.all_tasks()
            with self.assertRaisesRegex(AssertionError, 'operation ended before expected fixture event'):
                await observe_fixture(asyncio.Event().wait(), owner)
            self.assertFalse(asyncio.all_tasks() - before)

    async def test_event_wait_propagates_front_failure(self):
        failure = OSError('front failed while awaiting an event')
        async with running_fixture_task(mock.AsyncMock(side_effect=failure)()) as owner:
            before = asyncio.all_tasks()
            with self.assertRaises(OSError) as caught:
                await observe_fixture(asyncio.Event().wait(), owner)
            self.assertIs(caught.exception, failure)
            self.assertFalse(asyncio.all_tasks() - before)

    async def test_event_wait_joins_its_observer_without_cancelling_owners(self):
        async with running_fixture_task(asyncio.Event().wait()) as owner:
            operation = asyncio.get_running_loop().create_future()
            event = asyncio.Event()
            before = asyncio.all_tasks()
            event.set()
            self.assertTrue(await observe_fixture(event.wait(), owner, operation))
            self.assertFalse(operation.done())
            self.assertFalse(owner.done())
            self.assertFalse(asyncio.all_tasks() - before)

    async def test_cancelled_event_wait_joins_only_its_observer(self):
        async with running_fixture_task(asyncio.Event().wait()) as owner:
            operation = asyncio.get_running_loop().create_future()
            before = asyncio.all_tasks()
            waiting = asyncio.create_task(observe_fixture(asyncio.Event().wait(), owner, operation))
            try:
                await asyncio.sleep(0)
                waiting.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiting
                self.assertFalse(operation.done())
                self.assertFalse(owner.done())
                self.assertFalse(asyncio.all_tasks() - before)
            finally:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)

    async def test_event_wait_rejects_reply_before_event(self):
        async with running_fixture_task(asyncio.Event().wait()) as owner:
            operation = asyncio.get_running_loop().create_future()
            before = asyncio.all_tasks()
            operation.set_result('reply before event')
            with self.assertRaisesRegex(AssertionError, 'reply before event'):
                await observe_fixture(asyncio.Event().wait(), owner, operation)
            self.assertEqual(operation.result(), 'reply before event')
            self.assertFalse(owner.done())
            self.assertFalse(asyncio.all_tasks() - before)
