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
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, closing, suppress
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


async def observe_fixture(observation, owner, *finished):
    """Own a fixture observation, borrowing the operations that can invalidate it.

    ACP, terminal and launch fixtures wait for events, socket membership or file
    replies while the producer is deliberately paused. Joining that producer
    cannot establish readiness; a deadline cannot establish failure. Race the
    actual observation against its owners, and retire only our observer task.
    Real-process lifecycle witnesses below separately own their exit waiters.
    """
    owners = [owner, *finished]
    waiting = asyncio.create_task(observation)
    try:
        await asyncio.wait([waiting, *owners], return_when=asyncio.FIRST_COMPLETED)
        if waiting.done():
            return waiting.result()
        for owner in owners:
            if owner.done():
                raise AssertionError(
                    f'operation ended before expected fixture event: {owner.result()!r}')
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


@asynccontextmanager
async def running_fixture_task(awaitable):
    """Own one created actor; observe_fixture instead borrows its actors.

    Assert expected errors/cancellation inside the scope, before retirement
    consumes the outcome. Callers retain their own release and channel order.
    """
    task = asyncio.create_task(awaitable)
    try:
        yield task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def close_fixture_connection(task, writer):
    """Retire accepted work, including cancellation before its first step.

    asyncio.Server owns listeners, not handler tasks or their streams. The
    accepting fixture registers this with its existing AsyncExitStack before
    yielding to the handler; an observer which only borrows tasks cannot own
    this release. Inspect task and stream failures instead of caching errors
    inside a handler whose final close can itself fail.
    """
    try:
        writer.close()
    finally:
        if not task.done():
            task.cancel()
        try:
            outcome, = await asyncio.gather(task, return_exceptions=True)
            if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                raise outcome
        finally:
            # Abort can race peer reset. A handler's own close failure remains
            # its task outcome; unrelated late close failures still propagate.
            with suppress(ConnectionError):
                await writer.wait_closed()


async def observe_fixture_state(condition, owner, *finished):
    """Observe synchronous fixture state through the same borrowed owners.

    Immediate-input and loopback-server fixtures acknowledge admission inline,
    not through an awaitable event. Poll that existing state rather than add a
    second acknowledgement channel; observe_fixture owns the polling task and
    observes producer failure without cancelling the borrowed operations.
    """
    async def observed():
        while not condition():
            await asyncio.sleep(0)
    await observe_fixture(observed(), owner, *finished)


def observe_process_exit(case, pid):
    """Hold the live kernel object; PID reuse/zombie presence is not liveness."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        from loki_agent import windows_api
        opening = windows_api.bind('kernel32', 'OpenProcess', ctypes.c_void_p,
                                   wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        handle = opening(0x00100000, False, pid)
        case.assertTrue(handle, f'cannot observe live process {pid}: {ctypes.get_last_error()}')
        case.addCleanup(windows_api.close_handle, handle)

        def exited():
            return windows_api.wait_for_single_object(handle, 0) == 0
    elif sys.platform.startswith('linux'):
        import select
        if hasattr(os, 'pidfd_open'):
            fd = os.pidfd_open(pid)
        else:
            import ctypes
            # PyPy lacks this binding; retain the same kernel identity via libc.
            opening = ctypes.CDLL(None, use_errno=True).pidfd_open
            opening.argtypes, opening.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
            fd = opening(pid, 0)
            if fd < 0:
                raise OSError(ctypes.get_errno(), 'pidfd_open failed', pid)
        case.addCleanup(os.close, fd)
        poll = select.poll()
        poll.register(fd, select.POLLIN)

        def exited():
            return bool(poll.poll(0))
    else:
        import select
        queue = select.kqueue()
        case.addCleanup(queue.close)
        queue.control([select.kevent(pid, filter=select.KQ_FILTER_PROC,
                                     flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                                     fflags=select.KQ_NOTE_EXIT)], 0, 0)
        ended = False

        def exited():
            nonlocal ended
            # kqueue delivers the exit once; retain that terminal observation.
            ended = ended or bool(queue.control(None, 1, 0))
            return ended
    case.assertFalse(exited(), f'process {pid} was already dead')
    return exited


def windows_process_environment(pid, parent_pid):
    """Observe a live same-user test child through a held read-only handle.

    The test owns and keeps both processes alive; verified PID/parent fields
    reject an unrelated selection. This uses no debug privilege or target code.
    The PEB prefix follows winternl.h; Environment follows CommandLine in the
    native parameters ABI (phnt/ntrtl.h). An unsanitized owning-worker control
    must independently detect its injected value before absence is meaningful.
    """
    import ctypes
    from ctypes import wintypes
    from loki_agent import windows_api as api

    pointer = ctypes.c_void_p

    class Basic(ctypes.Structure):
        _fields_ = [('status', ctypes.c_int32), ('peb', pointer),
                    ('affinity', ctypes.c_size_t), ('priority', ctypes.c_int32),
                    ('pid', ctypes.c_size_t), ('parent', ctypes.c_size_t)]

    class Peb(ctypes.Structure):
        _fields_ = [('flags', ctypes.c_ubyte * 4), ('reserved', pointer * 2),
                    ('loader', pointer), ('parameters', pointer)]

    class String(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint16), ('maximum', ctypes.c_uint16), ('buffer', pointer)]

    class Parameters(ctypes.Structure):
        _fields_ = [('reserved', ctypes.c_ubyte * 16), ('pointers', pointer * 10),
                    ('image', String), ('command', String), ('environment', pointer)]

    opening = api.bind('kernel32', 'OpenProcess', pointer, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    query = api.bind('ntdll', 'NtQueryInformationProcess', ctypes.c_int32,
                     pointer, ctypes.c_int, pointer, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG))
    reading = api.bind('kernel32', 'ReadProcessMemory', wintypes.BOOL,
                       pointer, pointer, pointer, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t))
    handle = opening(0x0400 | 0x0010, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        basic, returned = Basic(), wintypes.ULONG()
        status = query(handle, 0, ctypes.byref(basic), ctypes.sizeof(basic), ctypes.byref(returned))
        if status < 0:
            raise OSError('NtQueryInformationProcess: 0x%x' % (status & 0xffffffff))
        if returned.value != ctypes.sizeof(basic) or basic.pid != pid or basic.parent != parent_pid:
            raise AssertionError('process observation identity/ABI mismatch')

        def read(address, size):
            data, count = ctypes.create_string_buffer(size), ctypes.c_size_t()
            if not reading(handle, address, data, size, ctypes.byref(count)):
                raise ctypes.WinError(ctypes.get_last_error())
            if count.value != size:
                raise AssertionError('short native environment observation')
            return data.raw

        peb = Peb.from_buffer_copy(read(basic.peb, ctypes.sizeof(Peb)))
        parameters = Parameters.from_buffer_copy(read(peb.parameters, ctypes.sizeof(Parameters)))
        if not parameters.environment:
            raise AssertionError('missing native environment pointer')
        data = bytearray()
        # A malformed native view must fail, never manufacture an empty report.
        for offset in range(0, 131072, 2):
            data.extend(read(parameters.environment + offset, 2))
            if data.endswith(b'\0\0\0\0'):
                return data.decode('utf-16-le').encode('utf-8')
        raise AssertionError('unterminated controlled environment')
    finally:
        api.close_handle(handle)


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
                # Leader retirement does not imply local family-Job release.
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

    async def wait_until_ready(self, original_wait):
        readiness = asyncio.create_task(self.ready.wait())
        exited = asyncio.create_task(original_wait())
        try:
            await asyncio.wait([readiness, exited], return_when=asyncio.FIRST_COMPLETED)
            if exited.done():
                code = await exited
                raise AssertionError(f'runtime exited with {code} before the credential-handshake witness')
        finally:
            for task in [readiness, exited]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(readiness, exited, return_exceptions=True)

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
            case.assertIsNotNone(server._close_task)
            case.assertTrue(server._close_task.done())
            case.assertFalse(server._close_task.cancelled())
            case.assertIsNone(server._close_task.exception())
            case.assertTrue(server.closed)
            case.assertTrue(server._writer.is_closing())
            if os.name == 'nt':
                case.assertEqual(server._writer._endpoint.handles(), [])
                for operation in [server._writer._reading, server._writer._writing]:
                    if operation is not None:
                        case.assertTrue(operation.done())
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
        storage = stack.enter_context(closing(
            credential_storages.JsonCredentialStorage(paths.credential_directory())))
        supervisor = credential_supervisors.CredentialSupervisor(
            CredentialStore(environment), storage)
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
            # One failed revocation/retirement must not skip the other owners.
            # The outer stack closes storage only after every async release.
            async with AsyncExitStack() as cleanup:
                for resources in reversed(observations.processes):
                    cleanup.push_async_callback(resources.cleanup)
                for delegation in reversed(observations.delegations):
                    cleanup.push_async_callback(delegation.close)
