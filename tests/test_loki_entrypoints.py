"""Tests for the test-side entrypoint helper.

It decides which executable a test launches; getting that subtly wrong only
shows up on Windows, where the checkout script cannot be executed at all.
"""

import os
import unittest

import loki_entrypoints


class EntrypointTests(unittest.TestCase):
    def test_the_posix_entrypoint_is_the_checkout_script(self):
        if os.name == "nt":
            self.skipTest("the packaged executable is the Windows entrypoint")
        self.assertEqual(loki_entrypoints.entrypoint("loki"),
                         os.path.join(loki_entrypoints.ROOT, "loki.py"))
        self.assertEqual(loki_entrypoints.entrypoint("loki-acp"),
                         os.path.join(loki_entrypoints.ROOT, "loki-acp"))


if __name__ == "__main__":
    unittest.main()
