"""Run the full suite so a stalled test reports its own stack.

The Windows build leg has wedged inside a test and been canceled with no
traceback: unittest writes a test's name before it runs and only reaches a
newline when the result follows, so the log ended at a bare
``test_x (...) ... `` and the ten-minute job budget was the only signal.  Two
things fix that here:

* stderr is written through, so the in-progress test name reaches the log
  before that test finishes;
* ``faulthandler.dump_traceback_later`` prints every thread's stack if the
  suite is still running after the stall timeout, so a hang names its frame
  instead of costing one job timeout per attempt.

The timeout is a stall detector, not a test timeout: a healthy but slow suite
prints nothing it did not need.

Usage: ``python -u tests/run_suite.py`` from the repository root.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import unittest


STALL_SECONDS_DEFAULT = 300.0


def stall_seconds() -> float:
    try:
        return float(os.environ.get(
            "LOKI_SUITE_STALL_SECONDS", STALL_SECONDS_DEFAULT))
    except ValueError:
        return STALL_SECONDS_DEFAULT


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        # The progress line has no newline until the result follows; without
        # this the running test's name is invisible while it is running.
        sys.stderr.reconfigure(write_through=True)
    except (AttributeError, ValueError):
        # A replaced or detached stderr is not worth failing the run over;
        # the faulthandler dump below still works.
        pass
    faulthandler.enable()
    faulthandler.dump_traceback_later(
        stall_seconds(), repeat=True, exit=False)
    suite = unittest.TestLoader().discover(os.path.join(root, "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
