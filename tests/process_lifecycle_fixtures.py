"""Real runtime/worker environments and observations of their owned resources.

The tests run on the host platform. Only the caller's executable identity is
substituted: unittest is not a Loki frontend. Launch, containment, delegation,
waiting and resource release remain the production implementations.
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


class ProcessResources:
    def __init__(self, process, closed):
        self.process = process
        self.wait = process.wait
        self.closed = closed
        self.close_offset = len(closed)
        self.handles = []
        self.pipes = []
        self.transport = getattr(process, '_transport', None)
        if os.name == 'nt':
            native = process
            if self.transport is not None:
                worker = self.transport._proc
                native = worker._contained
                self.handles.extend((worker.stdin.fileno(), worker.stdout.fileno()))
            self.handles.extend((native.job, native.information.hThread,
                                 native.information.hProcess))
        elif self.transport is not None:
            for fd in (0, 1, 2):
                pipe = self.transport.get_pipe_transport(fd)
                if pipe is not None:
                    self.pipes.append(pipe.get_extra_info('pipe'))

    def assert_released(self, case):
        case.assertIsNotNone(self.process.returncode)
        if os.name == 'nt':
            releases = self.closed[self.close_offset:]
            for handle in self.handles:
                case.assertEqual(releases.count(handle), 1,
                                 f'owned handle {handle} not released exactly once')
            if self.transport is not None:
                case.assertTrue(self.transport._exit_task.done())
                case.assertFalse(self.transport._exit_task.cancelled())
        else:
            # wait() must have reaped the child, not merely noticed its death.
            with case.assertRaises(ChildProcessError):
                os.waitpid(self.process.pid, os.WNOHANG)
            case.assertTrue(self.transport.is_closing())
            for pipe in self.pipes:
                case.assertTrue(pipe.closed)

    async def cleanup(self):
        # Backstop for a failing assertion or broken production cleanup. This
        # runs AFTER the ownership assertions, so it cannot make them pass.
        process = self.process
        if process.returncode is None:
            process.kill()
            await asyncio.wait_for(self.wait(), 5)
        if self.transport is not None:
            self.transport.close()
        if os.name == 'nt':
            from loki_agent import windows_api
            for handle in self.handles:
                if handle not in self.closed[self.close_offset:]:
                    windows_api.close_handle(handle)


class LifecycleObservations:
    def __init__(self, supervisor, workspace, launcher, closed):
        self.launcher = launcher
        self.supervisor = supervisor
        self.workspace = workspace
        self.closed = closed
        self.processes = []
        self.delegations = []
        self.ready = asyncio.Event()
        self.tasks_before = asyncio.all_tasks()

    def record_process(self, process):
        self.processes.append(ProcessResources(process, self.closed))
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
                       and not key.endswith(('_KEY', '_TOKEN', '_PAT'))}
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
        identity = SimpleNamespace(
            executable=launcher if os.name == 'nt' else sys.executable,
            argv=[launcher], frozen=os.name == 'nt', stderr=sys.stderr)
        stack.enter_context(mock.patch.object(runtime_isolation, 'sys', identity))
        closed = []
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            from loki_agent import windows_api, windows_runtime, windows_subprocesses
            stack.enter_context(mock.patch.object(windows_runtime, 'sys', identity))
            stack.enter_context(mock.patch.object(windows_subprocesses, 'sys', identity))
            close_handle = windows_api.close_handle
            get_handle_information = windows_api.bind(
                'kernel32', 'GetHandleInformation', wintypes.BOOL,
                ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))

            def close(handle):
                close_handle(handle)
                flags = wintypes.DWORD()
                if (get_handle_information(handle, ctypes.byref(flags))
                        or ctypes.get_last_error() != 6):  # ERROR_INVALID_HANDLE
                    raise AssertionError(f'kernel handle {handle} was not released')
                closed.append(handle)

            stack.enter_context(mock.patch.object(windows_api, 'close_handle', side_effect=close))
        supervisor = credential_supervisors.CredentialSupervisor(
            CredentialStore(environment),
            credential_storages.JsonCredentialStorage(paths.credential_directory()))
        observations = LifecycleObservations(supervisor, workspace, launcher, closed)
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
                for resources in observations.processes:
                    await resources.cleanup()
