"""Real runtime/worker environments and observations of their owned resources.

The tests run on the host platform. Only the caller's executable identity is
substituted: unittest is not a Loki frontend. Launch, containment, delegation,
waiting and resource release remain the production implementations.

Nothing here patches a close function or asks the kernel about a number after a
close. A number is not an identity: it can be bound again, so querying it proves
neither that the original object was released nor which object is there now.
Ownership is read from the owner's own state instead.
"""

import asyncio
from contextlib import ExitStack, asynccontextmanager
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

from loki_entrypoints import configure_container, entrypoint
from loki_agent import credential_storages, credential_supervisors, paths
from loki_agent import runtime_isolation
from loki_agent.credentials import CredentialStore

if os.name == 'nt':
    from loki_agent.windows_subprocesses import BaseSubprocessTransport


class ProcessResources:
    """Observe production launch ownership; public fixture drivers use cleanup only."""

    def __init__(self, process):
        self.process = process
        self.wait = process.wait
        self.transport = getattr(process, '_transport', None)
        # Keep the actual native owner, whose leader and family releases have
        # different lifetimes. The transport retains its cached exit result.
        self.native = (getattr(self.transport, '_proc', None)
                       if os.name == 'nt' and isinstance(self.transport, BaseSubprocessTransport) else None)
        self.pipes = []
        if os.name == 'posix' and self.transport is not None:
            for fd in [0, 1, 2]:
                pipe = self.transport.get_pipe_transport(fd)
                if pipe is not None:
                    self.pipes.append(pipe.get_extra_info('pipe'))

    def assert_released(self, case):
        case.assertIsNotNone(
            self.process.returncode, 'shutdown did not observe the child exit')
        if self.transport is None:
            case.assertIsNone(self.process.job)
            case.assertFalse(self.process.information.hProcess)
            case.assertFalse(self.process.information.hThread)
        elif os.name == 'nt':
            case.assertIsInstance(self.transport, BaseSubprocessTransport)
            case.assertTrue(self.transport._exit_task.done())
            case.assertFalse(self.transport._exit_task.cancelled())
            if self.native is not None:
                # This retirement scope still owns both the leader and its job.
                case.assertIsNone(self.native.job)
                case.assertFalse(self.native.information.hProcess)
                case.assertFalse(self.native.information.hThread)
        else:
            case.assertTrue(self.transport.is_closing())
            for pipe in self.pipes:
                case.assertTrue(pipe.closed)

    async def cleanup(self):
        # Backstop for a failing assertion or broken production cleanup. This
        # runs after the ownership assertions, so it cannot make them pass.
        process = self.process
        if process.returncode is None:
            process.kill()
            await self.wait()
        if self.transport is not None:
            self.transport.close()
        else:
            process.close()


class LifecycleObservations:
    def __init__(self, supervisor, workspace, launcher):
        self.launcher = launcher
        self.supervisor = supervisor
        self.workspace = workspace
        self.processes = []
        self.delegations = []
        self.ready = asyncio.Event()
        self.tasks_before = asyncio.all_tasks()

    def record_process(self, process):
        self.processes.append(ProcessResources(process))
        return process

    async def assert_released(self, case):
        # Let already-scheduled pipe connection_lost callbacks finish.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        case.assertTrue(self.processes)
        for resources in self.processes:
            resources.assert_released(case)
        for delegation in self.delegations:
            case.assertIsNone(delegation.owner_parent)
            case.assertIsNone(delegation.owner_child)
            case.assertIsNone(delegation.credential_child)
            server = delegation.credential_server
            case.assertTrue(server._reader_task.done())
            case.assertIsNotNone(server._writer_close_task)
            case.assertTrue(server._writer_close_task.done())
            case.assertFalse(server._writer_close_task.cancelled())
            case.assertIsNone(server._writer_close_task.exception())
            if os.name == 'nt':
                case.assertEqual(server._writer._endpoint.handles(), [])
                for operation in [server._writer._reading, server._writer._writing]:
                    if operation is not None:
                        case.assertTrue(operation.done())
            else:
                case.assertTrue(server._writer.is_closing())
        pending = asyncio.all_tasks() - self.tasks_before
        case.assertFalse(pending, f'shutdown left pending tasks: {pending}')


@asynccontextmanager
async def process_lifecycle(name):
    with tempfile.TemporaryDirectory(prefix='loki-lifecycle-') as root, ExitStack() as stack:
        workspace = os.path.join(root, 'workspace')
        os.mkdir(workspace)
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith('LOKI_')
                       and not any(key.endswith(suffix)
                                   for suffix in ['_KEY', '_TOKEN', '_PAT'])}
        environment.update({
            'HOME': root,
            'XDG_CONFIG_HOME': os.path.join(root, 'config'),
            'XDG_STATE_HOME': os.path.join(root, 'state'),
            'LOKI_PROVIDER': 'dummy',
            'LOKI_API_BASE': 'http://dummy.invalid/v1',
            'LOKI_MODEL': 'dummy-model',
            'LOKI_DUMMY_REPLY': 'lifecycle complete',
        })
        launcher = entrypoint(name)
        configure_container(environment, workspace)
        stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
        # Model the actual frontend caller without changing the interpreter's
        # global sys module or substituting a different executable/dispatcher.
        frozen = os.name == 'nt' and launcher.lower().endswith('.exe')
        identity = SimpleNamespace(
            executable=launcher if frozen else sys.executable,
            argv=[launcher], frozen=frozen, platform=sys.platform,
            stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        stack.enter_context(mock.patch.object(runtime_isolation, 'sys', identity))
        if os.name == 'nt':
            from loki_agent import windows_runtime, windows_subprocesses
            stack.enter_context(mock.patch.object(windows_runtime, 'sys', identity))
            stack.enter_context(mock.patch.object(windows_subprocesses, 'sys', identity))
        supervisor = credential_supervisors.CredentialSupervisor(
            CredentialStore(environment),
            credential_storages.JsonCredentialStorage(paths.credential_directory()))
        observations = LifecycleObservations(supervisor, workspace, launcher)
        delegate = supervisor.delegate

        async def record_delegation(*args, **kwargs):
            delegation = await delegate(*args, **kwargs)
            observations.delegations.append(delegation)
            answer = delegation.credential_server._answer

            async def observe_handshake(message):
                await answer(message)
                if message.get('method') == 'describe':
                    observations.ready.set()

            delegation.credential_server._answer = observe_handshake
            return delegation

        stack.enter_context(mock.patch.object(supervisor, 'delegate', new=record_delegation))
        try:
            yield observations
        finally:
            try:
                for delegation in observations.delegations:
                    await delegation.close()
            finally:
                try:
                    for resources in observations.processes:
                        await resources.cleanup()
                finally:
                    supervisor.storage.close()
