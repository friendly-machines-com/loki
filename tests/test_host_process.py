"""Tests for the spawn/signal seam.

Each platform asserts its own contract: POSIX detaches into a new session and
signals the group by ``killpg``; Windows gives the child its own process group
(``CREATE_NEW_PROCESS_GROUP``), identifies the group by its leader pid, and
stops it by terminating (``FORCE``) or, where a console is shared, by
``CTRL_BREAK_EVENT``.
"""

import os
import signal
import subprocess
import sys
import types
import unittest

from loki_agent import host_process


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


class SignalTests(unittest.TestCase):
    def _running_child(self):
        kwargs = ({"start_new_session": True} if os.name == "posix"
                  else host_process.spawn_kwargs())
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        return proc

    def test_the_signal_reaches_the_process_group(self):
        proc = self._running_child()
        group = host_process.process_group(proc, proc.pid)

        if os.name != "posix":
            # No signals; a forced stop terminates the process.
            host_process.signal_group(proc, group, host_process.FORCE)
            self.assertEqual(proc.wait(timeout=5), 1)
            return
        host_process.signal_group(proc, group, signal.SIGKILL)

        self.assertEqual(proc.wait(timeout=5), -signal.SIGKILL)

    def test_a_forced_stop_maps_to_sigkill_without_naming_it(self):
        proc = self._running_child()

        host_process.signal_group(
            proc, host_process.process_group(proc, proc.pid), host_process.FORCE)

        if os.name != "posix":
            self.assertEqual(proc.wait(timeout=5), 1)
            return
        self.assertEqual(proc.wait(timeout=5), -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
