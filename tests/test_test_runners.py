"""One root driver: real parallel workers, consistent output and lifetime."""

import asyncio
import io
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import run_tests


class TimerTests(unittest.TestCase):
    def test_no_diagnostic_timer_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(run_tests.stall_seconds())

    def test_diagnostic_interval_requires_an_explicit_finite_value(self):
        for value in ["2", "0.25"]:
            with self.subTest(value=value), mock.patch.dict(os.environ, {
                    "LOKI_SUITE_STALL_SECONDS": value}, clear=True):
                self.assertEqual(run_tests.stall_seconds(), float(value))
        for value in ["", "invalid", "0", "-1", "inf", "nan"]:
            with self.subTest(value=value), mock.patch.dict(os.environ, {
                    "LOKI_SUITE_STALL_SECONDS": value}, clear=True):
                self.assertIsNone(run_tests.stall_seconds())

    def test_worker_command_preserves_the_selected_interpreter_and_isolation(self):
        with mock.patch.object(run_tests.sys, "flags", mock.Mock(isolated=True)):
            command = run_tests._worker_command(Path("example.py"), Path("tests"), ["-q"])
        self.assertEqual(command[0], sys.executable)
        self.assertIn("-I", command)
        self.assertIn(str(run_tests.DRIVER), command)
        self.assertIn("--_worker", command)


class DriverProcessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.tests = self.root / "tests"
        self.tests.mkdir()
        self.work = self.root / "work"
        self.work.mkdir()
        self.env = dict(os.environ)
        self.env.pop("LOKI_SUITE_STALL_SECONDS", None)

    def write(self, name, source):
        path = self.tests / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path

    def command(self, *arguments, driver=None, isolated=False):
        command = [sys.executable]
        if isolated:
            command.append("-I")
        command.extend([
            "-u", str(driver or run_tests.DRIVER), "-s", str(self.tests), *arguments])
        return command

    def run_driver(self, *arguments, env=None):
        return subprocess.run(
            self.command(*arguments), env=env or self.env,
            capture_output=True, text=True, encoding="utf-8")

    def test_parallel_files_overlap_without_shared_interpreter_state(self):
        # Each file announces admission before waiting for the other. Serial
        # execution cannot satisfy this barrier; a deadline would additionally
        # impose an unrelated startup-speed requirement on healthy parallel runs.
        for name, other in [["a", "b"], ["b", "a"]]:
            self.write(f"test_{name}.py", f"""
                import pathlib, sys, time, unittest
                root = pathlib.Path({str(self.root)!r})
                class Parallel(unittest.TestCase):
                    def test_overlap(self):
                        self.assertFalse(hasattr(sys, 'other_file_state'))
                        sys.other_file_state = True
                        (root / {name!r}).write_text('ready')
                        while not (root / {other!r}).exists():
                            time.sleep(0.01)
                        print({('OVERLAP-' + name)!r}, flush=True)
            """)
        result = self.run_driver("-j", "2")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("OVERLAP-a", output)
        self.assertIn("OVERLAP-b", output)
        self.assertIn("PASS: test_a.py", output)
        self.assertIn("PASS: test_b.py", output)

    def test_filename_and_unittest_filters_are_forwarded(self):
        self.write("test_selected.py", """
            import unittest
            class Selected(unittest.TestCase):
                def test_kept(self):
                    print('KEPT')
                def test_filtered_out(self):
                    self.fail('must not run')
        """)
        self.write("test_ignored.py", "raise RuntimeError('file must not load')\n")
        result = self.run_driver("-p", "test_selected.py", "-v", "-k", "kept")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("KEPT", output)
        self.assertIn("Ran 1 test", output)
        self.assertNotIn("test_filtered_out", output)
        self.assertNotIn("test_ignored.py", output)

    def test_nested_same_named_files_run_once_each(self):
        for name in ["a", "b"]:
            self.write(f"{name}/__init__.py", "")
            self.write(f"{name}/test_same.py", f"""
                import unittest
                class Same(unittest.TestCase):
                    def test_once(self):
                        print({('ONCE-' + name)!r}, flush=True)
            """)
        result = self.run_driver("-p", "test_same.py", "-j", "2")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertEqual(output.count("ONCE-a"), 1)
        self.assertEqual(output.count("ONCE-b"), 1)

    def test_sibling_fixtures_and_module_setup_are_preserved(self):
        self.write("helpers.py", "VALUE = 'SIBLING-FIXTURE'\n")
        self.write("test_fixture.py", """
            import helpers, unittest
            ready = False
            def setUpModule():
                global ready
                ready = True
            class Fixture(unittest.TestCase):
                def test_setup(self):
                    self.assertTrue(ready)
                    print(helpers.VALUE)
        """)
        result = self.run_driver()
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("SIBLING-FIXTURE", output)

    def test_assertion_import_and_abnormal_exit_fail_the_run(self):
        sources = {
            "assertion": """
                import unittest
                class Bad(unittest.TestCase):
                    def test_bad(self):
                        self.fail('ASSERTION-FAILURE')
            """,
            "import": "raise RuntimeError('IMPORT-FAILURE')\n",
            "exit": "import os\nos._exit(7)\n",
        }
        for name, source in sources.items():
            with self.subTest(name=name):
                self.write("test_bad.py", source)
                result = self.run_driver()
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 1, output)
                self.assertIn("FAIL: test_bad.py", output)
                self.assertIn("1 of 1 test files failed", output)
                if name != "exit":
                    self.assertIn(name.upper() + "-FAILURE", output)

    def test_empty_selection_and_invalid_jobs_fail(self):
        result = self.run_driver("-p", "nothing.py")
        self.assertEqual(result.returncode, 1)
        self.assertIn("No test files match", result.stdout)
        result = self.run_driver("-j", "0")
        self.assertEqual(result.returncode, 2)

    def test_diagnostic_timer_does_not_kill_a_healthy_slow_test(self):
        # This timer must deliberately fire. Keep the worker alive until its
        # actual dump is observed, not for an assumed sufficient sleep interval.
        release = os.path.join(self.root, 'release-diagnostic')
        self.write("test_slow.py", f"""
            import os, time, unittest
            class Slow(unittest.TestCase):
                def test_slow(self):
                    while not os.path.exists({release!r}):
                        time.sleep(0.01)
                    print('SLOW-TEST-FINISHED')
        """)
        env = dict(self.env, LOKI_SUITE_STALL_SECONDS="0.04")
        process = subprocess.Popen(
            self.command(), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8')
        try:
            observed = ''
            while 'Timeout' not in observed:
                line = process.stdout.readline()
                self.assertTrue(line, observed)
                observed += line
            with open(release, 'a'):
                pass
            tail, _ = process.communicate()
            output = observed + tail
            self.assertEqual(process.returncode, 0, output)
            self.assertIn("SLOW-TEST-FINISHED", output)
            self.assertIn("Timeout", output)
            self.assertNotIn("explicit debug timeout exceeded", output)
        finally:
            with open(release, 'a'):
                pass
            if process.poll() is None:
                process.communicate()
            process.stdout.close()

    def test_explicit_execution_timeout_fails_without_claiming_a_hang(self):
        # The limit includes worker startup and fixture imports: no output is
        # promised before it expires. An indefinite import cannot finish normally
        # merely because the supervising process was descheduled for a long time.
        self.write("test_blocked.py", """
            import threading
            threading.Event().wait()
        """)
        result = self.run_driver("--timeout", "2")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertIn("explicit debug timeout exceeded (2s)", output)
        self.assertIn("1 of 1 test files failed", output)
        self.assertNotIn("stuck", output)

    def test_explicit_timeout_preserves_observed_output_and_joins_worker(self):
        # The CLI case checks the real timer. Here expiry follows observed output,
        # not elapsed startup time; the worker, pipe delivery, termination and join
        # remain real. Observing completion too makes startup failures fail promptly.
        path = self.write("test_output.py", """
            import sys, threading, unittest
            class Blocked(unittest.TestCase):
                def test_blocked(self):
                    print('PARTIAL-OUTPUT', end='', flush=True)
                    print('PARTIAL-ERROR', end='', file=sys.stderr, flush=True)
                    threading.Event().wait()
        """)
        stdout = io.StringIO()
        stderr = io.StringIO()
        output = run_tests._Output()

        async def exercise():
            ready = asyncio.Event()
            delivered = {1: "", 2: ""}
            write = output.write

            def observe(label, text, descriptor):
                # Readiness observes delivery, not rendering: losing captured text
                # must fail the final assertions rather than stall this observer.
                delivered[descriptor] += text
                write(label, text, descriptor)
                if ("PARTIAL-OUTPUT" in delivered[1]
                        and "PARTIAL-ERROR" in delivered[2]):
                    ready.set()

            async def expire_after_output(finished, timeout):
                self.assertEqual(timeout, 2)
                observation = asyncio.create_task(ready.wait())
                try:
                    await asyncio.wait(
                        [finished, observation], return_when=asyncio.FIRST_COMPLETED)
                    self.assertFalse(finished.done(), stdout.getvalue() + stderr.getvalue())
                    self.assertTrue(ready.is_set())
                    finished.cancel()  # Match wait_for cancelling the shield, not its owner.
                    raise TimeoutError
                finally:
                    observation.cancel()
                    await asyncio.gather(observation, return_exceptions=True)

            with mock.patch.object(run_tests.sys, "stdout", stdout), \
                    mock.patch.object(run_tests.sys, "stderr", stderr), \
                    mock.patch.dict(os.environ, self.env, clear=True), \
                    mock.patch.object(output, "write", side_effect=observe), \
                    mock.patch.object(run_tests.asyncio, "wait_for", side_effect=expire_after_output) as expiry, \
                    mock.patch.object(run_tests._WorkerProcess, "close", autospec=True,
                                      side_effect=run_tests._WorkerProcess.close) as close:
                result = await run_tests._run_file(path, self.tests, [], 2, output)
            expiry.assert_called_once()
            close.assert_called_once()
            worker = close.call_args.args[0]
            self.assertTrue(worker.finished.done())
            self.assertFalse(worker.finished.cancelled())
            self.assertEqual(worker.finished.result(), worker.transport.get_returncode())
            self.assertNotEqual(worker.finished.result(), 0)
            self.assertTrue(worker.transport.is_closing())
            return result

        self.assertEqual(asyncio.run(exercise()), 1)
        self.assertIn("PARTIAL-OUTPUT", stdout.getvalue())
        self.assertIn("PARTIAL-ERROR", stderr.getvalue())
        self.assertIn("FAIL: test_output.py", stdout.getvalue())
        self.assertIn("explicit debug timeout exceeded (2s)", stdout.getvalue())

    def test_partial_progress_is_visible_before_worker_finishes(self):
        ready = self.root / "ready"
        release = self.root / "release"
        self.write("test_live.py", f"""
            import pathlib, time, unittest
            class Live(unittest.TestCase):
                def test_live(self):
                    print('VISIBLE-BEFORE-END', flush=True)
                    pathlib.Path({str(ready)!r}).touch()
                    while not pathlib.Path({str(release)!r}).exists():
                        time.sleep(0.01)
        """)
        # Output and the release handshake observe this worker's real lifetime;
        # an unrelated execution budget would only impose another startup limit.
        process = subprocess.Popen(
            self.command(), env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8")
        try:
            observed = ""
            while "VISIBLE-BEFORE-END" not in observed:
                line = process.stdout.readline()
                self.assertTrue(line, observed)
                observed += line
            self.assertIsNone(process.poll())
            self.assertFalse(release.exists())
            release.touch()
            tail, _ = process.communicate()
            self.assertEqual(process.returncode, 0, observed + tail)
        finally:
            release.touch()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()

    @unittest.skipUnless(os.name == "posix", "SIGINT worker cleanup is a POSIX check")
    def test_interrupt_terminates_and_joins_workers(self):
        release = self.root / "interrupt-release"
        # Normal completion must come from the driver's interrupt cleanup, not
        # a finite actor sleep. The controller also releases the actor on an
        # assertion failure, so killing the driver cannot orphan an endless wait.
        self.write("test_wait.py", f"""
            import os, time, unittest
            class Wait(unittest.TestCase):
                def test_wait(self):
                    print('WORKER-READY', flush=True)
                    while not os.path.exists({str(release)!r}):
                        time.sleep(0.01)
        """)
        process = subprocess.Popen(
            self.command(), env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8")
        try:
            observed = ""
            while "WORKER-READY" not in observed:
                line = process.stdout.readline()
                self.assertTrue(line, observed)
                observed += line
            process.send_signal(signal.SIGINT)
            tail, _ = process.communicate()
            self.assertEqual(process.returncode, 130, observed + tail)
            self.assertNotIn("All test files passed", tail)
            self.assertNotIn("Task was destroyed", tail)
            self.assertNotIn("unclosed transport", tail)
        finally:
            with open(release, "a"):
                pass
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            # communicate() already consumed/closed the pipe on the success
            # path; only the failing scope can still need an EOF drain.
            if process.stdout.closed:
                process.wait()
            else:
                process.communicate()
            process.stdout.close()

    def test_staged_isolated_bootstrap_preserves_working_dir_and_test_activation(self):
        # The Windows probes cannot be replaced with ordinary discovery: their
        # checked --standard-user bootstrap activates native cases. This fixture
        # exercises the same staged-driver/argument/working-dir route under -I;
        # it does not pretend to verify Windows token or ACL behavior on Unix.
        driver = self.root / "run_tests.py"
        shutil.copyfile(run_tests.DRIVER, driver)
        path = self.write("test_probe.py", f"""
            import os, pathlib, sys, unittest
            @unittest.skip('bootstrap required')
            class Native(unittest.TestCase):
                def test_activated(self):
                    self.assertEqual(os.getcwd(), {str(self.work)!r})
                    self.assertTrue(sys.flags.isolated)
                    print('NATIVE-BOOTSTRAP-ACTIVATED')
            if __name__ == '__main__':
                if sys.argv[1:] != ['--standard-user', 'expected-sid']:
                    raise RuntimeError('identity argument rejected')
                del sys.argv[1:]
                Native.__unittest_skip__ = False
                unittest.main(verbosity=2)
        """)
        # Preserve the real probe layout: test file and driver at stage root.
        staged = self.root / path.name
        shutil.copyfile(path, staged)
        path.unlink()
        result = subprocess.run(
            [sys.executable, "-I", "-u", str(driver), "-s", str(self.root),
             "-p", staged.name, "--", "--standard-user", "expected-sid"],
            cwd=self.work, env=self.env,
            capture_output=True, text=True, encoding="utf-8")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertEqual(output.count("NATIVE-BOOTSTRAP-ACTIVATED"), 1)
        self.assertIn("Ran 1 test", output)
        self.assertNotIn("skipped=1", output)


class WorkerTimerTests(unittest.TestCase):
    def test_worker_default_has_no_timer_and_explicit_diagnostics_are_cleaned_up(self):
        root = Path(__file__).resolve().parent
        worker = mock.Mock()
        worker.result.wasSuccessful.return_value = True
        for interval in [None, 2.0]:
            with self.subTest(interval=interval), \
                    mock.patch.object(run_tests, "_configure_output"), \
                    mock.patch.object(run_tests.sys, "path", list(sys.path)), \
                    mock.patch.object(run_tests.importlib, "import_module", return_value=mock.Mock()), \
                    mock.patch.object(run_tests.unittest, "main", return_value=worker), \
                    mock.patch.object(run_tests.faulthandler, "enable"), \
                    mock.patch.object(run_tests, "stall_seconds", return_value=interval), \
                    mock.patch.object(run_tests.faulthandler, "dump_traceback_later") as dump, \
                    mock.patch.object(run_tests.faulthandler, "cancel_dump_traceback_later") as cancel:
                self.assertEqual(run_tests._run_worker(root / "test_example.py", root, []), 0)
                if interval is None:
                    dump.assert_not_called()
                    cancel.assert_not_called()
                else:
                    dump.assert_called_once_with(interval, repeat=True, exit=False)
                    cancel.assert_called_once_with()
