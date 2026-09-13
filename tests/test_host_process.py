"""Tests for the spawn/signal seam.

The Windows branch is not exercised here -- it needs a console and a real
``CTRL_BREAK_EVENT`` -- so these cover the POSIX contract that the seam has to
keep: a detached session, an unblocked signal mask, a queryable group, and a
group that actually receives the signal.
"""

import os
import signal
import subprocess
import sys
import types
import unittest

from loki_agent import host_process


@unittest.skipUnless(os.name == "posix", "POSIX seam")
class PosixSpawnTests(unittest.TestCase):
    def test_the_child_is_detached_into_its_own_session(self):
        kwargs = host_process.spawn_kwargs()

        self.assertIs(kwargs["start_new_session"], True)
        self.assertTrue(callable(kwargs["preexec_fn"]))


@unittest.skipUnless(os.name == "posix", "POSIX seam")
class ProcessGroupTests(unittest.TestCase):
    def test_a_live_pid_reports_its_process_group(self):
        proc = types.SimpleNamespace(pid=os.getpid())

        self.assertEqual(
            host_process.process_group(proc, os.getpid()),
            os.getpgid(os.getpid()))

    def test_an_unqueryable_pid_falls_back_to_itself(self):
        missing = 2 ** 30
        proc = types.SimpleNamespace(pid=missing)

        self.assertEqual(host_process.process_group(proc, missing), missing)


@unittest.skipUnless(os.name == "posix", "POSIX seam")
class SignalTests(unittest.TestCase):
    def test_the_signal_reaches_the_process_group(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True)
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)

        host_process.signal_group(
            proc, host_process.process_group(proc, proc.pid), signal.SIGKILL)

        self.assertEqual(proc.wait(timeout=5), -signal.SIGKILL)

    def test_a_forced_stop_maps_to_sigkill_without_naming_it(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True)
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)

        host_process.signal_group(
            proc, host_process.process_group(proc, proc.pid), host_process.FORCE)

        self.assertEqual(proc.wait(timeout=5), -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
