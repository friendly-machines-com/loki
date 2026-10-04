"""Run the full suite so a stalled test reports its own stack.

The Windows build leg has wedged inside a test and been canceled with no
traceback: unittest writes a test's name before it runs and only reaches a
newline when the result follows, so the log ended at a bare
``test_x (...) ... `` and the ten-minute job budget was the only signal.  Two
things fix that here:

* stderr is written through, so the in-progress test name reaches the log
  before that test finishes;
* when explicitly enabled with a positive, finite
  ``LOKI_SUITE_STALL_SECONDS``, ``faulthandler.dump_traceback_later`` prints
  every thread's stack after that interval, so a hang names its frame.

No timer is enabled by default and execution may run indefinitely. The
opt-in timer is diagnostic only: it never kills the suite or changes its
verdict.

Usage: ``python -u tests/run_suite.py`` from the repository root.
"""

from __future__ import annotations

import faulthandler
import math
import os
import sys
import unittest


def stall_seconds() -> float | None:
    """No timer unless an explicit positive, finite debug interval is set."""
    value = os.environ.get("LOKI_SUITE_STALL_SECONDS")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds > 0 and math.isfinite(seconds) else None


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
    interval = stall_seconds()
    if interval is not None:
        faulthandler.dump_traceback_later(interval, repeat=True, exit=False)
    try:
        suite = unittest.TestLoader().discover(os.path.join(root, "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    finally:
        if interval is not None:
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    raise SystemExit(main())
