"""Tests for the test-side entrypoint helpers.

These helpers decide which executable a test launches and what a Windows child
needs to find its container; both are easy to get subtly wrong in a way only
Windows would show.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import loki_entrypoints


class EntrypointTests(unittest.TestCase):
    def test_the_posix_entrypoint_is_the_checkout_script(self):
        if os.name == "nt":
            self.skipTest("the packaged executable is the Windows entrypoint")
        self.assertEqual(loki_entrypoints.entrypoint("loki"),
                         os.path.join(loki_entrypoints.ROOT, "loki.py"))
        self.assertEqual(loki_entrypoints.entrypoint("loki-acp"),
                         os.path.join(loki_entrypoints.ROOT, "loki-acp"))


class SeedContainerLedgerTests(unittest.TestCase):
    """A Windows child that keeps its own state directory needs the ledger.

    The gate looks the ledger up under the runtime's state directory; a test
    that relocates XDG_STATE_HOME must carry the configuration with it.
    """

    def test_nothing_is_written_off_windows(self):
        with tempfile.TemporaryDirectory() as state, \
                mock.patch.object(loki_entrypoints, "_on_windows",
                                  return_value=False):
            loki_entrypoints.seed_container_ledger({"XDG_STATE_HOME": state})

            self.assertEqual(os.listdir(state), [])

    def test_a_windows_child_state_receives_the_ledger(self):
        blob = {"version": 1, "workspaces": {"key": {"profile": "p"}}}
        with tempfile.TemporaryDirectory() as state, \
                mock.patch.object(loki_entrypoints, "_on_windows",
                                  return_value=True), \
                mock.patch("loki_agent.windows_state.load_ledger",
                           return_value=blob):
            loki_entrypoints.seed_container_ledger({"XDG_STATE_HOME": state})

            with open(os.path.join(state, "loki", "windows-setup.json"),
                      encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), blob)

    def test_the_real_state_directory_is_left_alone(self):
        # No XDG_STATE_HOME means the child already looks where the ledger is.
        with mock.patch.object(loki_entrypoints, "_on_windows",
                               return_value=True), \
                mock.patch("loki_agent.windows_state.load_ledger") as load:
            loki_entrypoints.seed_container_ledger({})
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
