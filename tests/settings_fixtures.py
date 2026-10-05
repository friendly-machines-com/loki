"""Built-in UI preferences for workflows that do not test settings loading.

No settings file is discovered or read. The guards also catch a reference to
an unpatched loader, rather than silently falling back to personal settings.
Settings and configured-presentation tests keep their real, explicit files.
"""

import contextlib
import unittest
from unittest import mock

from loki_agent import settings


@contextlib.contextmanager
def default_settings():
    with mock.patch.object(
            settings, "load_settings",
            new=mock.AsyncMock(return_value=settings.Settings())) as load, \
            mock.patch.object(settings, "_user_path", side_effect=AssertionError(
                "workflow test attempted to discover user settings")), \
            mock.patch.object(settings, "_read_ini", side_effect=AssertionError(
                "workflow test attempted to read a settings file")):
        yield load


def setUpModule():
    fixture = default_settings()
    fixture.__enter__()
    unittest.addModuleCleanup(fixture.__exit__, None, None, None)
