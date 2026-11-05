"""Workflow configuration does not discover personal settings; cleanup restores loading."""
import unittest
from unittest import mock

from loki_agent import settings
from settings_fixtures import default_settings


class DefaultSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_without_discovery(self):
        original_load = settings.load_settings
        original_path = settings.user_path
        report = mock.Mock()
        with default_settings() as load:
            self.assertEqual(
                await settings.load_settings(on_error=report),
                settings.Settings())
            load.assert_awaited_once_with(on_error=report)
            report.assert_not_called()
            self.assertEqual((await settings.load_settings()).hooks, [])
            # An alias captured before the fixture must not bypass isolation.
            with self.assertRaisesRegex(AssertionError, "discover user settings"):
                await original_load()
        self.assertIs(settings.load_settings, original_load)
        self.assertIs(settings.user_path, original_path)

    async def test_body_failure_restores_the_real_loader(self):
        original_load = settings.load_settings
        original_path = settings.user_path
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with default_settings():
                raise RuntimeError("body failed")
        self.assertIs(settings.load_settings, original_load)
        self.assertIs(settings.user_path, original_path)
