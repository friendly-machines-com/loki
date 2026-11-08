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
* Native local creation retains a non-inheritable Job Object reference beyond
  leader/stream completion. It owns later family stops, not normal-completion
  shutdown; the local job has no kill-on-close limit.
"""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import subprocess
import sys

from .lifetimes import finish_cleanup
from . import streams

# ``signal.SIGKILL`` is absent on Windows, so a forced stop is named here and
# mapped per platform instead of being spelled with a signal number.
FORCE = "force"


def label(signum) -> str:
    """Human-readable name for a stop request, without naming a value Windows lacks."""
    if signum == FORCE:
        return "SIGKILL" if sys.platform != "win32" else "terminate"
    return signum.name


async def create_exec(*command, owning=False, before_resume=None, **kwargs):
    """Preserve the caller's image/cwd; only trusted callers select authority."""
    if sys.platform == 'win32':
        from . import windows_subprocesses
        return await windows_subprocesses.create_local_process(
            *command, owning=owning, before_resume=before_resume, **kwargs)
    # POSIX cannot defer execution until admission is checked. Refuse early,
    # then recheck after startup inside the existing unpublished-child rollback.
    if before_resume is not None:
        before_resume()
    return await create_posix_process(*command, before_publish=before_resume, **kwargs)


def release_group(process):
    """Release family signalling ownership without terminating descendants."""
    if sys.platform == 'win32' and process is not None:
        close = getattr(process, 'close_family', None)
        if close is not None:
            close()


async def create_posix_process(*command, on_cancel=None, after_spawn=None,
                               before_publish=None, stdin=None, stdout=None, stderr=None,
                               limit=64 * 1024, **options):
    """Own creation through optional startup checks, until the caller takes over.

    Runtime delegation needs immediate revocation on cancellation; jobs and
    hooks need startup/publication checks on the same process owner, not
    separate pipe-draining retirement implementations. Local
    Session/Job admission runs synchronously after these awaits: rejection must
    retire the unpublished child here, not hand it to a revoked caller.
    """
    loop = asyncio.get_running_loop()

    async def create():
        # close_fds still preserves inherited stdio slots. A None Python stream
        # does not mean its fd stayed closed: PyPy/libffi imports were observed
        # reusing stdout's slot for an inheritable library file, making the child
        # fail when it flushed that read-only file as stdout. Clear inheritance,
        # not the fd itself: the importing library still owns the parent's fd.
        # Explicitly supplied child stdio must remain untouched.
        for fd, selected, stream in [
                [0, stdin, sys.stdin], [1, stdout, sys.stdout],
                [2, stderr, sys.stderr]]:
            if selected is None and stream is None:
                try:
                    os.set_inheritable(fd, False)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise
        transport, protocol = await loop.subprocess_exec(
            lambda: streams.ProcessProtocol(limit=limit),
            # The loop API defaults to pipes; the public subprocess helper
            # inherits missing streams. Terminal runtimes require the latter.
            *command, stdin=stdin, stdout=stdout, stderr=stderr, **options)
        return streams.Process(transport, protocol)

    creation = asyncio.create_task(create())
    process = None
    try:
        process = await asyncio.shield(creation)
        if after_spawn is not None:
            await after_spawn(process)
        if before_publish is not None:
            before_publish()
        return process
    except BaseException as error:
        try:
            if isinstance(error, asyncio.CancelledError):
                try:
                    if on_cancel is not None:
                        on_cancel()
                finally:
                    creation.cancel()
        finally:
            async def retire():
                selected_process = process
                if selected_process is None:
                    try:
                        selected_process = await creation
                    except (asyncio.CancelledError, Exception):
                        return
                try:
                    # The target may already have execed and forked before
                    # startup publication. Only a group created by this launch
                    # is ours to retire; hooks otherwise keep their direct-child
                    # lifetime contract.
                    if options.get('start_new_session'):
                        try:
                            os.killpg(selected_process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                finally:
                    # Close our pipe ends, then reap. Waiting for pipe EOF
                    # would let a surviving writer stall cancellation forever.
                    await selected_process.aclose()

            await finish_cleanup(retire(), preserve_failure=True)
        raise


if sys.platform != "win32":
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

    def group_exists(proc, pgid):
        """The group can outlive its leader; only ESRCH proves it is gone."""
        try:
            os.killpg(pgid or proc.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def signal_group(proc, pgid, signum):
        """Signal the child's whole process group, so its shell children follow."""
        os.killpg(pgid or proc.pid,
                  signal.SIGKILL if signum == FORCE else signum)

else:
    def spawn_kwargs():
        """Give the child its own process group, for CTRL_BREAK_EVENT."""
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # pytype: disable=module-attr

    def process_group(proc, pid):
        # Windows has no queryable process group id; a new group is addressed
        # by the id of the process that leads it.
        return pid

    def group_exists(proc, pgid):
        family_exists = getattr(proc, 'family_exists', None)
        return family_exists() if family_exists is not None else proc.returncode is None

    def signal_group(proc, pgid, signum):
        """Stop the child; Windows has no signals.

        A forced stop terminates it.  Anything else asks a cooperating child to
        stop with ``CTRL_BREAK_EVENT``, falling back to terminating it when the
        child cannot be reached that way.
        """
        if not group_exists(proc, pgid):
            raise ProcessLookupError("child family has exited")
        if signum == FORCE or proc.returncode is not None:
            _terminate(proc)
            return
        try:
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)  # pytype: disable=module-attr
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
            if not group_exists(proc, None):
                raise ProcessLookupError(error.errno) from error
            raise
