"""Real selected-root/launcher inputs for jobs invoked outside a frontend.

Unittest's dispatcher is not a sanctioned Loki executable. These fixtures
supply that missing caller identity and own real state-directory references;
they do not disable the ordinary-child namespace or substitute a process.
"""

from contextlib import AsyncExitStack, asynccontextmanager
import sys

from loki_entrypoints import entrypoint
from loki_agent import host_process, jobs, private_files


def bootstrap_command():
    return [sys.executable, entrypoint('loki'), '--tool-exec']


def process_creator(directory):
    async def create(*command, **options):
        return await host_process.create_exec(
            *command, state_directory=directory,
            tool_command=bootstrap_command() if sys.platform.startswith('linux') else None,
            **options)
    return create


def job_manager(base_dir, *, cleanup, async_cleanup=None, runner=None):
    directory = private_files.open_directory_tree(base_dir)
    # A real Session supplies this borrowed root. Tests must release it only
    # after the manager's work and narrower session-directory ownership end.
    cleanup(private_files.close, directory)
    manager = jobs.JobManager(base_dir, directory=directory,
                              create_exec=process_creator(directory))
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
