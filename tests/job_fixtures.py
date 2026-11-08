"""Real retained-directory and process-creator ownership for job fixtures.

Fixtures outside frontends supply acquired roots and retire the manager before
releasing those roots. Creator injection observes the actual launch operation.
"""

from contextlib import AsyncExitStack, asynccontextmanager

from loki_agent import host_process, jobs, private_files


def process_creator():
    async def create(*command, **options):
        return await host_process.create_exec(*command, **options)
    return create


def job_manager(base_dir, *, cleanup, async_cleanup=None, runner=None):
    directory = private_files.open_directory_tree(base_dir)
    # A real Session supplies this borrowed root. Tests must release it only
    # after the manager's work and narrower session-directory ownership end.
    cleanup(private_files.close, directory)
    manager = jobs.JobManager(base_dir, directory=directory,
                              create_exec=process_creator())
    if runner is not None:
        # The synchronous test keeps this Runner open through cleanup, so
        # async resources retire on the same loop that created them.
        cleanup(runner.run, manager.aclose())
    elif async_cleanup is not None:
        async_cleanup(manager.aclose)
    else:
        cleanup(manager.close)
    return manager


@asynccontextmanager
async def managed_jobs(base_dir):
    async with AsyncExitStack() as cleanup:
        yield job_manager(base_dir, cleanup=cleanup.callback,
                          async_cleanup=cleanup.push_async_callback)
