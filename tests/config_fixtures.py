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
    # The core can be imported before this module-scoped setup runs. Its
    # captured skill directory must use the same isolated configuration root.
    from loki_agent import loki, paths
    config = mock.patch.object(loki, 'LOKI_CONFIG_DIR', paths.loki_config_dir())
    config.start()
    unittest.addModuleCleanup(config.stop)
