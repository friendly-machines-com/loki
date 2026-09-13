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
* ``SIGKILL`` does not exist.  Callers request a forced stop with ``FORCE``;
  the seam maps it to ``SIGKILL`` on POSIX and to a terminate on Windows.
* ``terminate()`` reaches the direct child only.  Killing grandchildren needs a
  Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, which is not
  implemented here yet.
"""

from __future__ import annotations

import os
import signal
import subprocess

# ``signal.SIGKILL`` is absent on Windows, so a forced stop is named here and
# mapped per platform instead of being spelled with a signal number.
FORCE = "force"


def label(signum) -> str:
    """Human-readable name for a stop request, without naming a value Windows lacks."""
    if signum == FORCE:
        return "SIGKILL" if os.name == "posix" else "terminate"
    return signum.name


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
        os.killpg(pgid or proc.pid,
                  signal.SIGKILL if signum == FORCE else signum)

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
        if signum == FORCE:
            _terminate(proc)
            return
        try:
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        except OSError:
            _terminate(proc)

    def _terminate(proc):
        # Callers distinguish "the process is gone" from "the signal failed",
        # and Windows reports the former as an ordinary OSError.  Translate an
        # exited process so that distinction survives the seam.
        try:
            proc.terminate()
        except ProcessLookupError:
            raise
        except OSError as error:
            if proc.returncode is not None:
                raise ProcessLookupError(error.errno) from error
            raise
