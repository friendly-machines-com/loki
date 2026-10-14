"""Isolate user configuration for workflows that create conversations.

These workflows use default instructions, not the developer's personal
SYSTEM.md. Isolate discovery while keeping the real configuration loader.
"""

import os
import tempfile
import unittest
from unittest import mock


def setUpModule():
    directory = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(directory.cleanup)
    patch = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": directory.name})
    patch.start()
    unittest.addModuleCleanup(patch.stop)
