"""Spawning and signalling child processes, per host OS.

``JobManager`` drives one state machine on both; what differs is how a child is
detached from this process, how its process group is identified, and how that
group is stopped.

The Windows notes here are not translations of the POSIX calls:

* There is no ``preexec_fn`` -- ``subprocess`` refuses it outright -- so a
  child runs none of our code between fork and exec.  The POSIX side uses it
  only to unblock the signal mask the parent handed down.
* ``CREATE_NEW_PROCESS_GROUP`` gives the child its own group, which is what
  ``CTRL_BREAK_EVENT`` is addressed to.  Windows has no signal that can be
  delivered to an arbitrary process.
* There is no ``SIGTERM``/``SIGKILL``.  A cooperative stop is
  ``CTRL_BREAK_EVENT``; a forced stop terminates the process.
* ``terminate()`` reaches the direct child only.  Killing grandchildren needs a
  Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, which is not
  implemented here yet.
"""

from __future__ import annotations

import os
import signal
import subprocess

if os.name == "posix":
    def spawn_kwargs():
        """Put the child in its own session, with its signal mask unblocked."""
        return {"start_new_session": True,
                "preexec_fn": _unblock_child_signals}

    def _unblock_child_signals():
        # Runs between fork and exec, where only async-signal-safe calls belong.
        signal.pthread_sigmask(signal.SIG_UNBLOCK,
                               [signal.SIGINT, signal.SIGTERM])

    def process_group(proc, pid):
        """The child's process group, falling back to its own pid."""
        try:
            return os.getpgid(pid)
        except (AttributeError, OSError):
            return pid

    def signal_group(proc, pgid, signum):
        """Signal the child's whole process group, so its shell children follow."""
        os.killpg(pgid or proc.pid, signum)

else:
    def spawn_kwargs():
        """Give the child its own process group, for CTRL_BREAK_EVENT."""
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

    def process_group(proc, pid):
        # Windows has no queryable process group id; a new group is addressed
        # by the id of the process that leads it.
        return pid

    def signal_group(proc, pgid, signum):
        """Stop the child; Windows has no signals.

        A forced stop terminates it.  Anything else asks a cooperating child to
        stop with ``CTRL_BREAK_EVENT``, falling back to terminating it when the
        child cannot be reached that way.
        """
        if signum == signal.SIGKILL:
            proc.terminate()
            return
        try:
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        except OSError:
            proc.terminate()
