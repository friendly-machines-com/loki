"""Completion of cleanup which still owns resources after caller cancellation."""

import asyncio


async def finish_cleanup(cleanup, *, preserve_failure=False):
    """Own and join cleanup, then propagate cancellation or its actual failure.

    Process creation/retirement and job-family shutdown must not abandon an
    unpublished child or a live observer when the caller is cancelled again.
    asyncio.shield alone only protects the child task: it still interrupts the
    caller before cleanup completes. Keep that one task owned through every
    cancellation, without a deadline or a second waiter against its resources.
    A coroutine becomes one owned task; an existing shared cleanup task is
    joined without replacing or cancelling it. This is for resource retirement,
    not a borrowed sampling/request operation whose lifetime belongs elsewhere.
    Failed-creation handlers use preserve_failure so another caller cancellation
    cannot replace their original error. Actual cleanup failure still propagates.
    """
    task = asyncio.ensure_future(cleanup)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        result = task.result()
        if preserve_failure:
            return result
        raise
