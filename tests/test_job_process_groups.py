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

from loki_agent import host_process, jobs
from process_lifecycle_fixtures import ProcessResources


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
import os, signal, socket, subprocess, sys
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)
for n in range(2):
    subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1], str(n)])
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(sys.argv[1])
sock.sendall(f'leader {os.getpid()} {os.getpgrp()}\\n'.encode())
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
    """

    async def __aenter__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='loki-group-')
        self.manager = jobs.JobManager(os.path.join(self.directory.name, 'jobs'))
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

        self.server = await asyncio.start_unix_server(accept, self.path)
        return self

    def command(self, mode='socket', descendant=_DESCENDANT):
        return [sys.executable, '-c', _LEADER, self.path, descendant, mode]

    async def ready(self):
        for _ in range(3):
            reader, writer = await asyncio.wait_for(self.accepted.get(), 5)
            line = await asyncio.wait_for(reader.readline(), 5)
            name, pid, pgid = line.decode().split()
            self.connections[name] = [reader, writer]
            self.facts[name] = [int(pid), int(pgid)]
        if set(self.connections) != {'leader', 'child0', 'child1'}:
            raise AssertionError(f'missing group members: {self.facts}')
        for pid, pgid in self.facts.values():
            if pgid != self.job.pgid:
                raise AssertionError(f'{pid} was not in job group {self.job.pgid}')
        self.resources = ProcessResources(self.job.process)

    async def ping_descendants(self):
        for name in ['child0', 'child1']:
            reader, writer = self.connections[name]
            writer.write(b'p')
            await writer.drain()
            reply = await asyncio.wait_for(reader.readexactly(1), 5)
            if reply != b'p':
                raise AssertionError(f'{name} did not answer')

    async def leader_exited(self):
        await asyncio.wait_for(self.job.process.wait(), 5)
        await asyncio.wait_for(self.connections['leader'][0].read(), 5)

    async def all_dead(self):
        # A zombie has closed its sockets. kill(pid, 0) cannot distinguish it
        # from a live descendant and would give a misleading test result.
        await asyncio.wait_for(asyncio.gather(*[
            reader.read() for reader, _writer in self.connections.values()]), 5)

    async def __aexit__(self, *exc):
        try:
            if self.job is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.job.pgid, signal.SIGKILL)
                await asyncio.wait_for(self.job.process.wait(), 5)
            if self.task is not None:
                if not self.task.done():
                    self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            await self.manager.aclose()
        finally:
            self.server.close()
            for writer in self.writers:
                writer.close()
            await asyncio.gather(*[writer.wait_closed() for writer in self.writers],
                                 return_exceptions=True)
            await self.server.wait_closed()
            self.directory.cleanup()


@unittest.skipUnless(os.name == 'posix', 'POSIX process-group contract')
class JobProcessGroupTests(unittest.IsolatedAsyncioTestCase):
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
                    with mock.patch.object(manager, '_spawn', side_effect=held_spawn):
                        group.task = asyncio.create_task(manager.run_foreground(
                            group.command(), 'group', timeout, cancel_event=cancel_event))
                        await asyncio.wait_for(spawned.wait(), 5)
                        await group.ready()
                        if cause == 'pre_cancel':
                            cancel_event.set()
                        release.set()
                        await asyncio.wait_for(waiting.wait(), 5)
                        if cause == 'cancel':
                            cancel_event.set()
                        elif cause == 'unwind':
                            group.task.cancel()
                        elif cause == 'during_cleanup':
                            await group.leader_exited()
                            await group.ping_descendants()
                            group.task.cancel()
                            await asyncio.sleep(0)
                            group.task.cancel()
                        if cause in ['unwind', 'during_cleanup']:
                            with self.assertRaises(asyncio.CancelledError):
                                await asyncio.wait_for(group.task, 8)
                        else:
                            _job, outcome, _stdout, _stderr = await asyncio.wait_for(
                                group.task, 8)
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

            with mock.patch.object(manager, '_spawn', side_effect=capture_spawn):
                group.task = asyncio.create_task(manager.run_foreground(
                    group.command(descendant=_GRACEFUL_DESCENDANT), 'group', None,
                    cancel_event=cancel_event))
                await asyncio.wait_for(spawned.wait(), 5)
                await group.ready()
                cancel_event.set()
                await group.leader_exited()
                for name in ['child0', 'child1']:
                    reader, writer = group.connections[name]
                    self.assertEqual(await asyncio.wait_for(reader.readline(), 5), b'stopping\n')
                    self.assertFalse(group.task.done())
                    writer.write(b'x')
                    self.assertEqual(await asyncio.wait_for(reader.readline(), 5), b'clean\n')
                _job, outcome, _stdout, _stderr = await asyncio.wait_for(group.task, 8)
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
                    await asyncio.wait_for(manager.close_session_owned(), 8)
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
                await asyncio.wait_for(spawned.wait(), 5)
                await group.ready()
                group.connections['leader'][1].write(b'x')
                job, outcome, _stdout, _stderr = await asyncio.wait_for(group.task, 5)
            self.assertEqual(outcome, 'completed')
            self.assertEqual(job.exit_code, 0)
            await group.ping_descendants()
            self.assertIn('Sent SIGKILL', manager.stop_job(job.id, force=True))
            await group.all_dead()

    async def test_force_retires_even_a_zombie_only_group(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = jobs.JobManager(directory)
            self.addCleanup(manager.close)
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
