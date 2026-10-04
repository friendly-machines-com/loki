#!/usr/bin/env python3
"""The local and CI test driver: parallel files, one interpreter per file.

    python3 run_tests.py
    python3 run_tests.py -p 'test_acp*' -j 4 -v -k prompt

No execution deadline or diagnostic timer is enabled by default.
LOKI_SUITE_STALL_SECONDS opts into repeating worker stack dumps; it never
changes a test's verdict. --timeout SECONDS explicitly opts into a per-file
debug execution limit. CI budgets belong in the workflow, not this driver.

Workers execute this same script's private --_worker mode. There is no second
runner or CI-specific execution path. Selected files are imported exactly
once per worker, with the tests directory on sys.path just as in unittest
discovery. Windows standard-user probes retain their test-file bootstrap:
it validates the token and enables the native cases before unittest runs.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import faulthandler
import importlib
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parent
DRIVER = Path(__file__).resolve()
TEST_DIR = ROOT / "tests"


def stall_seconds() -> float | None:
    """An explicit diagnostic interval, never a default execution limit."""
    value = os.environ.get("LOKI_SUITE_STALL_SECONDS")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds > 0 and math.isfinite(seconds) else None


def positive_seconds(value):
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number") from error
    if seconds <= 0 or not math.isfinite(seconds):
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return seconds


def _configure_output():
    # Workers must flush the in-progress test name, not wait for its newline.
    # One encoding also covers isolated Windows probe interpreters whose
    # sanitized environment otherwise supplies a different locale encoding.
    for stream in [sys.stdout, sys.stderr]:
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace",
                               write_through=True)
        except (AttributeError, ValueError):
            pass


def _run_worker(path, start_directory, arguments):
    path = path.resolve()
    relative = path.relative_to(start_directory)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(start_directory))
    _configure_output()
    faulthandler.enable()
    interval = stall_seconds()
    if interval is not None:
        faulthandler.dump_traceback_later(interval, repeat=True, exit=False)
    try:
        if "--standard-user" in arguments:
            # This is a test bootstrap, not an alternative suite runner. Its
            # SID check and AppContainer unskip hook must run under -I in the
            # staged interpreter; ordinary import/discovery would skip them.
            sys.argv = [str(path), *arguments]
            try:
                runpy.run_path(str(path), run_name="__main__")
            except SystemExit as error:
                if error.code is None:
                    return 0
                if isinstance(error.code, int):
                    return error.code
                print(error.code, file=sys.stderr)
                return 1
            return 0
        module_name = ".".join(relative.with_suffix("").parts)
        module = importlib.import_module(module_name)
        program = unittest.main(
            module=module, argv=[str(path), *arguments],
            verbosity=2, exit=False)
        return 0 if program.result.wasSuccessful() else 1
    finally:
        if interval is not None:
            faulthandler.cancel_dump_traceback_later()


class _Output:
    """Tag live chunks without buffering partial progress or stack dumps."""

    def __init__(self):
        self.sources = {1: None, 2: None}

    def write(self, label, text, descriptor):
        if not text:
            return
        stream = sys.stdout if descriptor == 1 else sys.stderr
        source = self.sources[descriptor]
        if source is not None and source != label:
            stream.write("\n")
            source = None
        for fragment in text.splitlines(keepends=True):
            if source is None:
                stream.write(f"[{label}] ")
            stream.write(fragment)
            source = None if fragment.endswith("\n") or fragment.endswith("\r") else label
        self.sources[descriptor] = source
        stream.flush()

    def note(self, text):
        for descriptor in [1, 2]:
            if self.sources[descriptor] is not None:
                stream = sys.stdout if descriptor == 1 else sys.stderr
                stream.write("\n")
                stream.flush()
                self.sources[descriptor] = None
        print(text, flush=True)


class _WorkerProcess(asyncio.SubprocessProtocol):
    """Own one worker's process, output pipes, and completion signal."""

    def __init__(self, loop, output, label):
        self.transport = None
        self.finished = loop.create_future()
        self.output = output
        self.label = label
        self.error = None
        self.decoders = {
            descriptor: codecs.getincrementaldecoder("utf-8")("backslashreplace")
            for descriptor in [1, 2]
        }

    def connection_made(self, transport):
        self.transport = transport

    def pipe_data_received(self, descriptor, data):
        try:
            self.output.write(self.label, self.decoders[descriptor].decode(data), descriptor)
        except Exception as error:
            self.error = error
            self.transport.close()

    def pipe_connection_lost(self, descriptor, error):
        if error is not None:
            self.error = error
        try:
            self.output.write(
                self.label, self.decoders[descriptor].decode(b"", final=True), descriptor)
        except Exception as error:
            self.error = error

    def connection_lost(self, error):
        if error is not None:
            self.error = error
        if not self.finished.done():
            self.finished.set_result(
                self.transport.get_returncode() if self.transport is not None else None)

    def close(self):
        if self.transport is None:
            return
        try:
            # The live leader owns this POSIX process group. Do not signal a
            # possibly reused group ID after the leader has exited. Closing
            # the transport always releases the pipes, including pipes a
            # descendant kept open, so debug cancellation cannot hang on EOF.
            if os.name == "posix" and self.transport.get_returncode() is None:
                try:
                    os.killpg(self.transport.get_pid(), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            self.transport.close()


def _worker_command(path, start_directory, arguments):
    command = [sys.executable]
    if sys.flags.isolated:
        command.append("-I")
    command.extend([
        "-u", str(DRIVER), "--_worker", str(path),
        "-s", str(start_directory), "--", *arguments])
    return command


async def _run_file(path, start_directory, arguments, timeout, output):
    loop = asyncio.get_running_loop()
    label = path.relative_to(start_directory).as_posix()
    worker = _WorkerProcess(loop, output, label)
    started = time.monotonic()
    code = 1
    detail = ""
    try:
        options = {"start_new_session": True} if os.name == "posix" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        # Standard-user probes require their supervisor's writable work dir;
        # ordinary suite workers use the checkout root. The choice is an
        # explicit bootstrap argument, never an implicit CI/environment flag.
        cwd = os.getcwd() if "--standard-user" in arguments else str(ROOT)
        await loop.subprocess_exec(
            lambda: worker, *_worker_command(path, start_directory, arguments),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=cwd, **options)
        if timeout is None:
            code = await asyncio.shield(worker.finished)
        else:
            try:
                code = await asyncio.wait_for(asyncio.shield(worker.finished), timeout)
            except TimeoutError:
                detail = f"; explicit debug timeout exceeded ({timeout:g}s)"
        if worker.error is not None:
            code = 1
            detail = f"; output failed: {worker.error}"
    except OSError as error:
        detail = f"; could not launch: {error}"
    finally:
        worker.close()
        if worker.transport is not None:
            await asyncio.shield(worker.finished)
    passed = code == 0 and not detail
    outcome = "PASS" if passed else "FAIL"
    output.note(f"{outcome}: {label} ({time.monotonic() - started:.2f}s){detail}")
    return 0 if passed else 1


async def _run_parallel(paths, start_directory, arguments, jobs, timeout):
    output = _Output()
    output.note(f"Selected {len(paths)} test files; running with up to {jobs} workers.")
    slots = asyncio.Semaphore(jobs)

    async def run(path):
        async with slots:
            return await _run_file(path, start_directory, arguments, timeout, output)

    tasks = [asyncio.create_task(run(path)) for path in paths]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    failures = sum(results)
    output.note(f"{failures} of {len(paths)} test files failed." if failures else
                "All test files passed!")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run test files in parallel, identically locally and in CI.",
        allow_abbrev=False)
    parser.add_argument("-p", "--pattern", default="test_*.py",
                        help="test filename pattern (default: %(default)s)")
    parser.add_argument("-s", "--start-directory", type=Path, default=TEST_DIR,
                        help="test directory (default: checkout's tests directory)")
    parser.add_argument("-j", "--jobs", type=int,
                        default=min(32, (os.cpu_count() or 1) + 4),
                        help="maximum parallel file workers (default: %(default)s)")
    parser.add_argument("--timeout", type=positive_seconds,
                        help="explicit per-file debug limit in seconds (default: unlimited)")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    options, arguments = parser.parse_known_args(argv)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if options.jobs < 1:
        parser.error("--jobs must be at least 1")
    start_directory = options.start_directory.resolve()
    if not start_directory.is_dir():
        parser.error(f"test directory does not exist: {start_directory}")
    if options._worker is not None:
        return _run_worker(options._worker, start_directory, arguments)
    paths = sorted({path.resolve() for path in start_directory.rglob(options.pattern)
                    if path.is_file() and path.suffix == ".py" and path.stem.isidentifier()})
    if not paths:
        print(f"No test files match {options.pattern!r} in {start_directory}")
        return 1
    if "--standard-user" in arguments and len(paths) != 1:
        parser.error("--standard-user requires selecting exactly one probe file")
    _configure_output()
    try:
        return asyncio.run(_run_parallel(
            paths, start_directory, arguments, options.jobs, options.timeout))
    except KeyboardInterrupt:
        print("Test run interrupted.", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
