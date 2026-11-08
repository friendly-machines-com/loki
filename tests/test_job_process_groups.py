"""Termination follows a job's group even after its leader has been reaped."""

import asyncio
import contextlib
import json
import os
import signal
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from job_fixtures import job_manager
from loki_agent import host_process, jobs as job_runtime
from process_lifecycle_fixtures import ProcessResources, observe_fixture


_DESCENDANT = """
import os, signal, socket, sys
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(sys.argv[1])
sock.sendall(f'child{sys.argv[2]} {os.getpid()} {os.getpgrp()}\\n'.encode())
while data := sock.recv(1):
    sock.sendall(data)
"""

_GRACEFUL_DESCENDANT = """
import os, signal, socket, sys
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(sys.argv[1])
def finish(*_args):
    sock.sendall(b'stopping\\n')
    sock.recv(1)
    sock.sendall(b'clean\\n')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, finish)
signal.signal(signal.SIGINT, finish)
sock.sendall(f'child{sys.argv[2]} {os.getpid()} {os.getpgrp()}\\n'.encode())
sock.recv(1)
"""

_LEADER = """
import os, select, signal, socket, subprocess, sys
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)
children = [subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1], str(n)])
            for n in range(2)]
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(sys.argv[1])
sock.sendall(f'leader {os.getpid()} {os.getpgrp()}\\n'.encode())
control = int(sys.argv[-1]) if sys.argv[3] == 'owner' else sock
# Only controller input/EOF releases this leader. Polling its actual children
# makes failed membership observable through that same leader, without a timer
# standing in for readiness or an additional failure channel.
while not select.select([control], [], [], .01)[0]:
    for child in children:
        if child.poll() is not None:
            raise RuntimeError('descendant exited before controller release: ' + str(child.returncode))
if sys.argv[3] == 'owner':
    while os.read(int(sys.argv[-1]), 1):
        pass
else:
    sock.recv(1)
"""


class _Group:
    """Each member's socket witnesses readiness and death, including zombies.

    The independent backstop kills the group even if production cleanup forgets
    it after leader exit. Direct-child-only fixtures cannot clean up this case.
    Readiness, replies and teardown have no default watchdog: execution limits
    belong only to the root driver's explicit user option.
    """

    async def __aenter__(self):
        async with contextlib.AsyncExitStack() as scope:
            self.directory = tempfile.TemporaryDirectory(prefix='loki-group-')
            scope.enter_context(self.directory)
            self.manager = job_manager(os.path.join(self.directory.name, 'jobs'),
                                       cleanup=scope.callback, async_cleanup=scope.push_async_callback)
            self.path = os.path.join(self.directory.name, 'observer')
            self.accepted = asyncio.Queue()
            self.writers = []
            self.connections = {}
            self.facts = {}
            self.job = None
            self.task = None

            def accept(reader, writer):
                self.writers.append(writer)
                self.accepted.put_nowait([reader, writer])

            server = await asyncio.start_unix_server(accept, self.path)
            self.server = await scope.enter_async_context(server)
            self.job_scope = scope.pop_all()
            return self

    @contextlib.contextmanager
    def grace_clock(self):
        """Exercise real grace-loop logic without racing this test controller.

        These tests pause actors to inspect ownership and signal ordering. Only
        the job module's grace clock is controlled; real asyncio functions and
        the deliberately requested 20ms execution timeout still use real time.
        Patching the event loop's own clock would instead freeze its I/O timers.
        """
        clock = SimpleNamespace(now=0.0, entered=asyncio.Queue())
        api = SimpleNamespace(**vars(asyncio))
        api.get_running_loop = lambda: SimpleNamespace(time=lambda: clock.now)
        original = self.manager._wait_for_group

        async def entered(job, grace_s):
            clock.entered.put_nowait(grace_s)
            return await original(job, grace_s)

        with mock.patch.object(job_runtime, 'asyncio', api), \
                mock.patch.object(self.manager, '_wait_for_group', new=entered):
            yield clock

    def command(self, mode='socket', descendant=_DESCENDANT):
        return [sys.executable, '-c', _LEADER, self.path, descendant, mode]

    async def ready(self):
        # The fixture does not release the leader before membership is known.
        # Observe its real exit, and the foreground owner where present, so a
        # failed spawn/actor cannot leave an empty accept queue waiting forever.
        exited = asyncio.create_task(self.job.process.wait())
        owners = [exited] if self.task is None else [exited, self.task]
        try:
            for _ in range(3):
                reader, writer = await observe_fixture(self.accepted.get(), *owners)
                line = await observe_fixture(reader.readline(), *owners)
                name, pid, pgid = line.decode().split()
                self.connections[name] = [reader, writer]
                self.facts[name] = [int(pid), int(pgid)]
            if set(self.connections) != {'leader', 'child0', 'child1'}:
                raise AssertionError(f'missing group members: {self.facts}')
            for pid, pgid in self.facts.values():
                if pgid != self.job.pgid:
                    raise AssertionError(f'{pid} was not in job group {self.job.pgid}')
            self.resources = ProcessResources(self.job.process)
        finally:
            exited.cancel()
            await asyncio.gather(exited, return_exceptions=True)

    async def ping_descendants(self):
        for name in ['child0', 'child1']:
            reader, writer = self.connections[name]
            writer.write(b'p')
            await writer.drain()
            reply = await reader.readexactly(1)
            if reply != b'p':
                raise AssertionError(f'{name} did not answer')

    async def leader_exited(self):
        await self.job.process.wait()
        await self.connections['leader'][0].read()

    async def all_dead(self):
        # A zombie has closed its sockets. kill(pid, 0) cannot distinguish it
        # from a live descendant and would give a misleading test result.
        await asyncio.gather(*[
            reader.read() for reader, _writer in self.connections.values()])

    async def __aexit__(self, *exc):
        try:
            if self.job is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.job.pgid, signal.SIGKILL)
                await self.job.process.wait()
            if self.task is not None:
                if not self.task.done():
                    self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            await self.manager.aclose()
        finally:
            self.server.close()
            try:
                for writer in self.writers:
                    writer.close()
                await asyncio.gather(*[writer.wait_closed() for writer in self.writers],
                                     return_exceptions=True)
            finally:
                await self.job_scope.aclose()


@unittest.skipUnless(os.name == 'posix', 'POSIX process-group contract')
class JobProcessGroupTests(unittest.IsolatedAsyncioTestCase):
    async def test_readiness_observes_descendant_failure_before_connect(self):
        async with _Group() as group:
            group.job = await group.manager.run_background_exec(group.command(
                descendant='raise RuntimeError("injected descendant startup failure")'))
            with self.assertRaisesRegex(AssertionError, 'operation ended before expected fixture event'):
                await group.ready()
            self.assertNotEqual(await group.job.process.wait(), 0)
            self.assertIn('injected descendant startup failure',
                          group.manager._spool_tail(group.job, 'stderr.log'))

    async def test_soft_stop_then_force_after_leader_exit(self):
        async with _Group() as group:
            manager = group.manager
            job = group.job = await manager.run_background_exec(group.command())
            await group.ready()
            self.assertIn('Sent SIGTERM', manager.stop_job(job.id))
            await group.leader_exited()
            await group.ping_descendants()
            manager.list_jobs()
            self.assertEqual(job.status, 'stopping')
            self.assertEqual(job.exit_code, -signal.SIGTERM)
            self.assertIsNone(job.finished_at)
            self.assertIn('already stopping', manager.stop_job(job.id))
            self.assertIn('Sent SIGKILL', manager.stop_job(job.id, force=True))
            await group.all_dead()
            manager.job_status(job.id)
            self.assertEqual(job.status, 'stopped')
            self.assertFalse(job.group_active)
            with open(job.metadata_path, encoding='utf-8') as stream:
                metadata = json.load(stream)
            # The exit code still describes the leader, not its descendants.
            self.assertEqual(metadata['exit_code'], -signal.SIGTERM)
            self.assertEqual(metadata['status'], 'stopped')
            with mock.patch.object(host_process, 'signal_group') as send:
                self.assertIn('not running', manager.stop_job(job.id, force=True))
                send.assert_not_called()
            group.resources.assert_released(self)

    async def test_foreground_cleanup_escalates_after_leader_exit(self):
        for cause in ['timeout', 'cancel', 'unwind', 'pre_cancel', 'during_cleanup']:
            with self.subTest(cause=cause):
                tasks_before = asyncio.all_tasks()
                async with _Group() as group:
                    manager = group.manager
                    cancel_event = asyncio.Event()
                    spawned = asyncio.Event()
                    release = asyncio.Event()
                    waiting = asyncio.Event()
                    real_spawn = manager._spawn

                    async def held_spawn(*args, **kwargs):
                        group.job = await real_spawn(*args, **kwargs)
                        spawned.set()
                        await release.wait()
                        waiting.set()
                        return group.job

                    timeout = 20 if cause in ['timeout', 'during_cleanup'] else None
                    with group.grace_clock() as clock, \
                            mock.patch.object(manager, '_spawn', side_effect=held_spawn):
                        group.task = asyncio.create_task(manager.run_foreground(
                            group.command(), 'group', timeout, cancel_event=cancel_event))
                        await observe_fixture(spawned.wait(), group.task)
                        await group.ready()
                        if cause == 'pre_cancel':
                            cancel_event.set()
                        release.set()
                        await observe_fixture(waiting.wait(), group.task)
                        if cause == 'cancel':
                            cancel_event.set()
                        elif cause == 'unwind':
                            group.task.cancel()
                        grace = await observe_fixture(clock.entered.get(), group.task)
                        if cause == 'during_cleanup':
                            await group.leader_exited()
                            await group.ping_descendants()
                            group.task.cancel()
                            await asyncio.sleep(0)
                            group.task.cancel()
                        clock.now += grace + 1
                        if cause in ['unwind', 'during_cleanup']:
                            with self.assertRaises(asyncio.CancelledError):
                                await group.task
                        else:
                            _job, outcome, _stdout, _stderr = await group.task
                            self.assertEqual(outcome,
                                             'timed_out' if cause == 'timeout' else 'cancelled')
                    await group.all_dead()
                    expected = 'timed_out' if cause == 'timeout' else 'cancelled'
                    self.assertEqual(group.job.status, expected)
                    self.assertFalse(group.job.group_active)
                    leader_signal = (signal.SIGTERM if cause in ['timeout', 'during_cleanup']
                                     else signal.SIGINT)
                    self.assertEqual(group.job.exit_code, -leader_signal)
                    group.resources.assert_released(self)
                    self.assertFalse(asyncio.all_tasks() - tasks_before)

    async def test_descendants_get_the_grace_period_after_leader_exit(self):
        async with _Group() as group:
            manager = group.manager
            real_spawn = manager._spawn
            spawned = asyncio.Event()
            cancel_event = asyncio.Event()

            async def capture_spawn(*args, **kwargs):
                group.job = await real_spawn(*args, **kwargs)
                spawned.set()
                return group.job

            with group.grace_clock() as clock, \
                    mock.patch.object(manager, '_spawn', side_effect=capture_spawn):
                group.task = asyncio.create_task(manager.run_foreground(
                    group.command(descendant=_GRACEFUL_DESCENDANT), 'group', None,
                    cancel_event=cancel_event))
                await observe_fixture(spawned.wait(), group.task)
                await group.ready()
                cancel_event.set()
                await group.leader_exited()
                grace = await observe_fixture(clock.entered.get(), group.task)
                for name in ['child0', 'child1']:
                    reader, writer = group.connections[name]
                    self.assertEqual(await reader.readline(), b'stopping\n')
                    self.assertFalse(group.task.done())
                    writer.write(b'x')
                    self.assertEqual(await reader.readline(), b'clean\n')
                await group.all_dead()
                clock.now += grace + 1
                _job, outcome, _stdout, _stderr = await group.task
            await group.all_dead()
            self.assertEqual(outcome, 'cancelled')
            group.resources.assert_released(self)

    async def test_owned_shutdown_includes_groups_with_exited_leaders(self):
        for mode in ['owner', 'socket', 'already_exited']:
            with self.subTest(mode=mode):
                async with _Group() as group:
                    manager = group.manager
                    job = group.job = await manager.run_background_exec(
                        group.command(mode), session_owned=True, subagent=True)
                    await group.ready()
                    if mode == 'already_exited':
                        group.connections['leader'][1].write(b'x')
                        await group.leader_exited()
                        manager.list_jobs()
                        await group.ping_descendants()
                        self.assertTrue(job.group_active)
                        self.assertIsNone(job.finished_at)
                        self.assertFalse(job.subagent_slot)
                        self.assertEqual(manager._active_subagents, 0)
                    closing = None
                    try:
                        with group.grace_clock() as clock:
                            closing = asyncio.create_task(manager.close_session_owned())
                            owner_grace = await observe_fixture(clock.entered.get(), closing)
                            if mode != 'socket':
                                await group.leader_exited()
                            clock.now += owner_grace + 1
                            termination_grace = await observe_fixture(clock.entered.get(), closing)
                            await group.leader_exited()
                            clock.now += termination_grace + 1
                            await closing
                    finally:
                        # The real clock is restored before any failing controller
                        # joins teardown, so a paused fake clock cannot strand it.
                        if closing is not None:
                            if not closing.done():
                                closing.cancel()
                            await asyncio.gather(closing, return_exceptions=True)
                    await group.all_dead()
                    self.assertEqual(job.status, 'owner_closed')
                    self.assertEqual(job.exit_code, -signal.SIGTERM if mode == 'socket' else 0)
                    self.assertFalse(job.group_active)
                    self.assertFalse(job.subagent_slot)
                    self.assertEqual(manager._active_subagents, 0)
                    with mock.patch.object(host_process, 'signal_group') as send:
                        await manager.close_session_owned()
                        send.assert_not_called()
                    group.resources.assert_released(self)

    async def test_normal_completion_does_not_kill_surviving_descendants(self):
        async with _Group() as group:
            manager = group.manager
            real_spawn = manager._spawn
            spawned = asyncio.Event()

            async def capture_spawn(*args, **kwargs):
                group.job = await real_spawn(*args, **kwargs)
                spawned.set()
                return group.job

            with mock.patch.object(manager, '_spawn', side_effect=capture_spawn):
                group.task = asyncio.create_task(manager.run_foreground(
                    group.command(), 'group', None))
                await observe_fixture(spawned.wait(), group.task)
                await group.ready()
                group.connections['leader'][1].write(b'x')
                job, outcome, _stdout, _stderr = await group.task
            self.assertEqual(outcome, 'completed')
            self.assertEqual(job.exit_code, 0)
            await group.ping_descendants()
            self.assertIn('Sent SIGKILL', manager.stop_job(job.id, force=True))
            await group.all_dead()

    async def test_force_retires_even_a_zombie_only_group(self):
        async with contextlib.AsyncExitStack() as job_scope:
            directory = job_scope.enter_context(tempfile.TemporaryDirectory())
            manager = job_manager(directory, cleanup=job_scope.callback,
                                  async_cleanup=job_scope.push_async_callback)
            job = mock.Mock(group_active=True, pgid=123)
            job.process.wait = mock.AsyncMock(return_value=-signal.SIGTERM)
            with mock.patch.object(host_process, 'group_exists', return_value=True) as exists, \
                    mock.patch.object(host_process, 'signal_group') as send:
                code = await manager._terminate_process_group(job, signal.SIGTERM, 0)
                self.assertEqual(code, -signal.SIGTERM)
                self.assertFalse(job.group_active)
                self.assertEqual(send.call_args_list, [
                    mock.call(job.process, job.pgid, signal.SIGTERM),
                    mock.call(job.process, job.pgid, host_process.FORCE)])
                exists.reset_mock()
                send.reset_mock()
                await manager._terminate_process_group(job, signal.SIGTERM, 0)
                exists.assert_not_called()
                send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
