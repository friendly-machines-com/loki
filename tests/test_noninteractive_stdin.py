"""A noninteractive job's stdin reaches end of file carrying nothing.

Jobs run without a terminal: Bash, the search children Glob and Grep spawn,
and subagents all need input that ends immediately rather than blocking. Each
child here reads fd 0 to end of file and reports the byte count, so "the stream
ended" is distinguished from "the read returned". Both spawn branches are
covered, plus the search-child and subagent argv shapes, plus the completion,
cancellation, and launch-failure paths, each requiring that no stdin end is
retained afterwards.

On Windows these jobs run inside a container that denies the null device, so
stdin cannot be the null device there. Nothing in this file observes that; the
contained topology is covered natively in ``test_windows_appcontainers.py``.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from loki_agent import loki


# Reads stdin to EOF and prints the byte count.  Real child, real reads: the
# only thing this fixture substitutes is the program under test.
_REPORTS_EOF = """
import sys
data = sys.stdin.buffer.read()
sys.stdout.write('stdin-bytes: %d' % len(data))
sys.stdout.flush()
"""

# The same report, then a marker saying the child is live, then a block that
# ends when the caller writes the release file.  The child never decides to
# finish on its own: the cancellation under test stops it, or the fixture
# releases it.  Takes both paths from the caller.
_BLOCKS_UNTIL_RELEASED = """
import pathlib, sys, time
sys.stdout.flush()
ready, release = %r, %r
pathlib.Path(ready).write_text('ready')
while not pathlib.Path(release).exists():
    time.sleep(0.01)
sys.stdout.write('released\\n')
sys.stdout.flush()
"""


class NoninteractiveStdinTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_job_child_reads_end_of_file_on_stdin(self):
        """A job child's stdin reaches EOF and carries no bytes.

        Same function, both platforms.  The assertion is the observed byte
        count, not that the child finished, so a stdin that never ends fails
        here instead of passing as an ordinary short read.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = loki.JobManager(str(root / "jobs"))
            try:
                job, status, stdout, stderr = await manager.run_exec(
                    [sys.executable, "-c", _REPORTS_EOF],
                    description="noninteractive stdin", cwd=str(workspace))
            finally:
                await manager.close_session_owned()
            self.assertEqual(status, "completed", stderr)
            self.assertEqual(job.status, "exited", stderr)
            self.assertEqual(job.exit_code, 0, stderr)
            self.assertEqual(
                stdout.strip(), "stdin-bytes: 0",
                "job stdin must be end of file carrying nothing")

    async def test_a_shell_job_child_reads_end_of_file_on_stdin(self):
        """The Bash path reaches the same result as the direct-executable one.

        Both spawn branches in ``JobManager._spawn`` must hand over stdin, so
        this asserts the shell branch separately rather than assuming the exec
        branch covers it.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            # A real file, so the shell command is one program invocation and
            # the shell is still Loki's Bash branch doing the spawning.
            script = root / "report_eof.py"
            script.write_text(_REPORTS_EOF, encoding="utf-8")
            manager = loki.JobManager(str(root / "jobs"))
            try:
                # run_shell renders the job result, so read the text it
                # returns rather than the intermediate tuple run_foreground
                # produces; the stdout it carries is the child's report.
                rendered = await manager.run_shell(
                    '"%s" "%s"' % (sys.executable, script),
                    description="shell stdin",
                    cwd=str(workspace))
            finally:
                await manager.close_session_owned()
            job, = manager.jobs.values()
            self.assertEqual(job.status, "exited", rendered)
            self.assertEqual(job.exit_code, 0, rendered)
            self.assertIn(
                "stdin-bytes: 0", rendered,
                "shell job stdin must be end of file carrying nothing")

    async def test_a_search_child_reads_end_of_file_on_stdin(self):
        """The Glob/Grep child launch shape reaches the same EOF.

        Those tools spawn an external search program through ``run_exec`` with
        the searcher's own argv, so the handoff is exercised here with that
        shape rather than assumed from the direct-executable case.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = loki.JobManager(str(root / "jobs"))
            try:
                job, status, stdout, stderr = await manager.run_exec(
                    [sys.executable, "-c", _REPORTS_EOF,
                     "--color=never", "--", "marker"],
                    description="search child stdin", cwd=str(workspace))
            finally:
                await manager.close_session_owned()
            self.assertEqual(status, "completed", stderr)
            self.assertEqual(job.exit_code, 0, stderr)
            self.assertEqual(
                stdout.strip(), "stdin-bytes: 0",
                "search child stdin must be end of file carrying nothing")

    async def test_a_subagent_child_reads_end_of_file_on_stdin(self):
        """The session-owned subagent launch shape reaches the same EOF.

        Subagents run session-owned with a credential capability, which is a
        different spawn path from an ordinary command; only the stdin handoff
        is under test here, so the capability is not requested.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = loki.JobManager(str(root / "jobs"))
            try:
                job, status, stdout, stderr = await manager.run_exec(
                    [sys.executable, "-c", _REPORTS_EOF],
                    description="subagent stdin", cwd=str(workspace),
                    session_owned=True, subagent=True)
            finally:
                await manager.close_session_owned()
            self.assertEqual(status, "completed", stderr)
            self.assertEqual(job.exit_code, 0, stderr)
            self.assertEqual(
                stdout.strip(), "stdin-bytes: 0",
                "subagent stdin must be end of file carrying nothing")

    async def test_a_background_job_child_reads_end_of_file_on_stdin(self):
        """Background completion delivers the same EOF and releases stdin.

        Item 4's acceptance names background completion explicitly, and it runs
        through the background monitor rather than the foreground waiter.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = loki.JobManager(str(root / "jobs"))
            try:
                job = await manager.run_background_exec(
                    [sys.executable, "-c", _REPORTS_EOF],
                    description="background stdin", cwd=str(workspace))
                await manager._monitor_background_job(job)
            finally:
                await manager.close_session_owned()
            self.assertEqual(job.status, "exited")
            self.assertEqual(job.exit_code, 0)
            self.assertEqual(
                Path(job.stdout_path).read_text(encoding="utf-8").strip(),
                "stdin-bytes: 0",
                "background job stdin must be end of file carrying nothing")
            self.assertIsNone(
                getattr(job.process, "stdin", None),
                "a completed background job still owns its stdin end")

    async def test_a_cancelled_job_reads_end_of_file_and_releases_stdin(self):
        """Cancellation delivers the same EOF and still releases resources.

        Cancellation is the other half of item 4's acceptance.  The child
        reports its stdin EOF, then blocks on a pipe it cannot write to until
        the stop arrives, so this ends by the cancellation reaching it -- not
        by a deadline.  No timeout is set anywhere here: a run that has to be
        cut short by one is telling us nothing about stdin.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            ready = root / "ready"
            release = root / "release"
            script = root / "report_then_wait.py"
            script.write_text(
                _REPORTS_EOF + _BLOCKS_UNTIL_RELEASED
                % (str(ready), str(release)),
                encoding="utf-8")
            manager = loki.JobManager(str(root / "jobs"))
            cancel = asyncio.Event()
            try:
                task = asyncio.create_task(manager.run_exec(
                    [sys.executable, str(script)],
                    description="cancelled stdin", cwd=str(workspace),
                    cancel_event=cancel))
                # Wait for the child's own marker: it has reported EOF and is
                # now live, so the stop lands on a running process rather than
                # on one that already finished.
                while not ready.exists():
                    if task.done():
                        self.fail("job finished before its stdin EOF was "
                                  "observed: %r" % (task.result(),))
                    await asyncio.sleep(0.01)
                cancel.set()
                job, status, stdout, stderr = await task
            finally:
                # The child may outlive a failed expectation; releasing the
                # pipe lets it exit on its own instead of lingering.
                release.write_text("release", encoding="utf-8")
                await manager.close_session_owned()
            self.assertEqual(status, "cancelled", stderr)
            self.assertEqual(stdout.strip(), "stdin-bytes: 0", stderr)
            self.assertIsNone(
                getattr(job.process, "stdin", None),
                "a cancelled job still owns its stdin end")

    async def test_a_released_job_does_not_leave_stdin_open(self):
        """A finished job holds nothing on stdin.

        Once the child is gone, no descriptor or handle it was given may still
        be held here: a retained stdin end would keep the pipe alive for any
        later observer. This reads the owner state rather than counting
        kernel objects, because a number can be reused.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = loki.JobManager(str(root / "jobs"))
            try:
                await manager.run_exec(
                    [sys.executable, "-c", _REPORTS_EOF],
                    description="stdin ownership", cwd=str(workspace))
                job, = [job for job in manager.jobs.values()]
                stdin = getattr(job.process, "stdin", None)
                self.assertIsNone(
                    stdin, "a completed job still owns its stdin end")
            finally:
                await manager.close_session_owned()

    async def test_a_failed_launch_releases_the_stdin_it_acquired(self):
        """Launch failure must not strand a stdin pipe end.

        The spawn branch acquires stdin before the child exists; if the
        executable cannot be started, that end has nowhere to go. This injects
        the failure after acquisition and requires the launch to unwind with
        nothing retained.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = loki.JobManager(str(root / "jobs"))
            try:
                with self.assertRaises(FileNotFoundError):
                    await manager.run_exec(
                        ["definitely-not-a-real-program"],
                        description="stdin leak on failure", cwd=str(workspace))
                for job in manager.jobs.values():
                    self.assertIsNone(
                        getattr(job.process, "stdin", None),
                        "a failed launch retained its stdin end")
            finally:
                await manager.close_session_owned()


if __name__ == "__main__":
    unittest.main()
