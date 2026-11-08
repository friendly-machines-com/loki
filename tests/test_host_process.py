"""Tests for the spawn/signal seam.

Each platform asserts its own contract: POSIX detaches into a new session and
signals the group by ``killpg``; Windows gives the child its own process group
(``CREATE_NEW_PROCESS_GROUP``), identifies the group by its leader pid, and
stops it by terminating (``FORCE``) or, where a console is shared, by
``CTRL_BREAK_EVENT``.
"""

import asyncio
import errno
import os
import signal
import subprocess
import sys
import types
import unittest
from unittest import mock

from loki_agent import host_process, streams
from process_lifecycle_fixtures import observe_fixture


class ProcessRetirementTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipe_close_failure_still_closes_siblings_and_joins_transport(self):
        failure = OSError('stdin close failed')
        entered = asyncio.Event()
        calls = []
        pipes = {fd: mock.Mock() for fd in [0, 1, 2]}
        for fd, pipe in pipes.items():
            pipe.is_closing.return_value = False
            pipe.close.side_effect = lambda fd=fd: calls.append(fd)

        def abort():
            calls.append(0)
            raise failure

        def close_transport():
            calls.append('process')
            entered.set()

        pipes[0].abort.side_effect = abort
        transport = mock.Mock()
        transport.get_pipe_transport.side_effect = pipes.get
        transport.close.side_effect = close_transport
        protocol = streams.ProcessProtocol()
        protocol.connection_made(transport)
        process = streams.Process(transport, protocol)
        closing = asyncio.create_task(process.aclose())
        try:
            await observe_fixture(entered.wait(), closing)
            self.assertEqual(calls, [0, 1, 2, 'process'])
            self.assertFalse(closing.done())
            protocol.connection_lost(None)
            with self.assertRaises(OSError) as caught:
                await closing
            self.assertIs(caught.exception, failure)
            transport.close.assert_called_once_with()
        finally:
            protocol.connection_lost(None)
            await asyncio.gather(closing, return_exceptions=True)


class PosixSpawnTests(unittest.TestCase):
    def test_the_child_is_detached_into_its_own_session(self):
        kwargs = host_process.spawn_kwargs()

        if os.name != "posix":
            # Windows gives the child its own process group, which is what a
            # console control event can be addressed to.
            self.assertEqual(
                kwargs, {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP})
            return
        self.assertIs(kwargs["start_new_session"], True)
        self.assertTrue(callable(kwargs["preexec_fn"]))


class ProcessGroupTests(unittest.TestCase):
    def test_a_live_pid_reports_its_process_group(self):
        proc = types.SimpleNamespace(pid=os.getpid())

        if os.name != "posix":
            # No queryable process-group id; the group is the leader's pid.
            self.assertEqual(
                host_process.process_group(proc, os.getpid()), os.getpid())
            return
        self.assertEqual(
            host_process.process_group(proc, os.getpid()),
            os.getpgid(os.getpid()))

    def test_an_unqueryable_pid_falls_back_to_itself(self):
        missing = 2 ** 30
        proc = types.SimpleNamespace(pid=missing)

        self.assertEqual(host_process.process_group(proc, missing), missing)


class GroupExistenceTests(unittest.TestCase):
    def test_liveness_uses_the_group_not_the_exited_leader(self):
        proc = types.SimpleNamespace(pid=123, returncode=0)
        if os.name != 'posix':
            self.assertFalse(host_process.group_exists(proc, 123))
            proc.returncode = None
            self.assertTrue(host_process.group_exists(proc, 123))
            return
        with mock.patch.object(os, 'killpg') as probe:
            self.assertTrue(host_process.group_exists(proc, 123))
            probe.assert_called_once_with(123, 0)

    @unittest.skipUnless(os.name == 'posix', 'POSIX group probing')
    def test_only_a_missing_group_is_reported_as_gone(self):
        proc = types.SimpleNamespace(pid=123, returncode=0)
        with mock.patch.object(os, 'killpg', side_effect=ProcessLookupError(errno.ESRCH, 'gone')):
            self.assertFalse(host_process.group_exists(proc, 123))
        with mock.patch.object(os, 'killpg', side_effect=PermissionError(errno.EPERM, 'denied')):
            with self.assertRaises(PermissionError):
                host_process.group_exists(proc, 123)


class SignalTests(unittest.TestCase):
    def _running_child(self):
        kwargs = ({"start_new_session": True} if os.name == "posix"
                  else host_process.spawn_kwargs())
        # These signal tests own termination; the actor cannot self-exit while
        # the controller is descheduled. Cleanup kills and joins the same child.
        proc = subprocess.Popen(
            [sys.executable, "-c", "import threading; threading.Event().wait()"], **kwargs)
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        return proc

    def test_the_signal_reaches_the_process_group(self):
        proc = self._running_child()
        group = host_process.process_group(proc, proc.pid)

        if os.name != "posix":
            # No signals; a forced stop terminates the process.
            host_process.signal_group(proc, group, host_process.FORCE)
            self.assertEqual(proc.wait(), 1)
            return
        host_process.signal_group(proc, group, signal.SIGKILL)

        self.assertEqual(proc.wait(), -signal.SIGKILL)

    def test_a_forced_stop_maps_to_sigkill_without_naming_it(self):
        proc = self._running_child()

        host_process.signal_group(
            proc, host_process.process_group(proc, proc.pid), host_process.FORCE)

        if os.name != "posix":
            self.assertEqual(proc.wait(), 1)
            return
        self.assertEqual(proc.wait(), -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
