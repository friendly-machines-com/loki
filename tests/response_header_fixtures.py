"""Test-owned observation storage for modules exercising inference runtimes.

unittest runs setUpModule even when selecting just one class or test. Redirect
new stores as well as replacing the already-created default session: changing
XDG_STATE_HOME alone cannot relocate a store constructed before the fixture.
No production save hooks are disabled, and the process environment is untouched.
"""

import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from loki_agent import loki, response_headers
from loki_agent.sessions import Session


@contextlib.contextmanager
def isolated_response_headers():
    with tempfile.TemporaryDirectory(prefix="loki-test-status-") as directory:
        path = str(Path(directory) / "response-headers.json")
        with mock.patch.object(response_headers, "snapshot_path", return_value=path):
            with mock.patch.object(loki, "_DEFAULT_SESSION", Session()):
                yield path


def setUpModule():
    fixture = isolated_response_headers()
    fixture.__enter__()
    unittest.addModuleCleanup(fixture.__exit__, None, None, None)
