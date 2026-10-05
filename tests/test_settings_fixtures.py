"""Workflow preferences never discover settings, and fixture cleanup restores loading."""

import unittest
from unittest import mock

from loki_agent import settings
from settings_fixtures import default_settings


class DefaultSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_without_discovery_or_file_reads(self):
        original_load = settings.load_settings
        original_path = settings._user_path
        original_read = settings._read_ini
        report = mock.Mock()
        with default_settings() as load:
            self.assertEqual(await settings.load_settings(on_error=report), settings.Settings())
            load.assert_awaited_once_with(on_error=report)
            report.assert_not_called()
            # An alias captured before the fixture must not bypass isolation.
            with self.assertRaisesRegex(AssertionError, "discover user settings"):
                await original_load()
            with self.assertRaisesRegex(AssertionError, "read a settings file"):
                await original_load(files=["unexpected-settings.ini"])
        self.assertIs(settings.load_settings, original_load)
        self.assertIs(settings._user_path, original_path)
        self.assertIs(settings._read_ini, original_read)

    async def test_body_failure_restores_the_real_loader(self):
        original_load = settings.load_settings
        original_path = settings._user_path
        original_read = settings._read_ini
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with default_settings():
                raise RuntimeError("body failed")
        self.assertIs(settings.load_settings, original_load)
        self.assertIs(settings._user_path, original_path)
        self.assertIs(settings._read_ini, original_read)
