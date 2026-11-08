"""Test-owned observation storage for modules exercising inference runtimes.

unittest runs setUpModule even when selecting just one class or test. Redirect
new stores as well as replacing the already-created default session: changing
XDG_STATE_HOME alone cannot relocate a store constructed before the fixture.
No production save hooks are disabled, and the process environment is untouched.
"""

import contextlib
import os
import tempfile
import unittest
from unittest import mock

from loki_agent import loki, response_headers
from loki_agent.sessions import Session


@contextlib.contextmanager
def isolated_response_headers(*, runner=None):
    with tempfile.TemporaryDirectory(prefix="loki-test-status-") as directory:
        path = os.path.join(directory, "response-headers.json")
        with mock.patch.object(response_headers, "snapshot_path", return_value=path):
            session = Session()
            with contextlib.closing(session.response_headers):
                # Core unit workflows have no frontend caller identity. Select
                # the real sanctioned test dispatcher; fresh frontend processes
                # remain unpatched and exercise their own reentry selection.
                with mock.patch.object(loki, "_DEFAULT_SESSION", session):
                    try:
                        yield path
                    finally:
                        # QuestionGuard owns a live Runner and real Bash jobs;
                        # retire its Session before Store, directory and Runner.
                        if runner is not None:
                            runner.run(session.close_state())


def setUpModule():
    unittest.enterModuleContext(isolated_response_headers())
