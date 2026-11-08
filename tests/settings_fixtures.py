"""Default configuration for workflows that do not test configuration loading.

The single loader is mocked, so neither preferences nor hooks discover personal
settings. An unpatched alias attempting discovery still trips the path guard.
Tests with explicit configuration files retain the real parser/decoder.
"""

import contextlib
import unittest
from unittest import mock

from loki_agent import private_files, settings


def write_user_settings(config_home, text):
    """Seed real frontend fixtures through the selected config directory.

    Hook startup tests and ACP recovery wire tests must exercise discovery of
    the actual user file, not a patched loader. Retain the supplied directory
    and create literal children through native primitives, without assembling
    another pathname from the caller's VFS key. The fixture owns the temporary
    config home; each acquired directory closes before that home is removed.
    """
    root = private_files.open_directory(config_home)
    try:
        write_user_settings_at(root, text)
    finally:
        private_files.close(root)


def write_user_settings_at(root, text):
    """Seed the same file through a borrowed root, including literal-~ tests.

    The filename-lookup journey has already selected its directory natively;
    reopening a constructed pathname would defeat the operand under test.
    Only the acquired child directory/file belong to this helper.
    """
    private_files.create_directory_at(root, "loki")
    directory = private_files.open_directory_at(root, "loki")
    try:
        file = private_files.create_exclusive_at(directory, "settings.ini", 0o600)
        with private_files.binary_file(file, "wb") as target:
            target.write(text.encode("utf-8"))
    finally:
        private_files.close(directory)


@contextlib.contextmanager
def default_settings():
    with mock.patch.object(
            settings, "load_settings",
            new=mock.AsyncMock(return_value=settings.Settings())) as load, \
            mock.patch.object(settings, "user_path", side_effect=AssertionError(
                "workflow test attempted to discover user settings")):
        yield load


def setUpModule():
    unittest.enterModuleContext(default_settings())
