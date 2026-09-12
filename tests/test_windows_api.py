"""Portable checks for the Windows declarations and the platform path choice.

A wrong GUID layout, flag value or structure size is invisible on the machine
that runs it -- it surfaces as a failure somewhere else, on someone else's
Windows box.  The pure declarations are therefore checked here, on every
platform; only the calls themselves need Windows.  The configuration and state
locations are checked here too, for the same reason: which platform-specific
location they resolve to is decided by branches that never run on this host.
"""

import ctypes
import os
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import paths
from loki_agent import windows_api


class GuidTests(unittest.TestCase):
    def test_documented_guid_parses_to_its_documented_fields(self):
        guid = windows_api.guid_from_text(windows_api.FOLDERID_LOCAL_APP_DATA)

        self.assertEqual(guid.data1, 0xF1B32785)
        self.assertEqual(guid.data2, 0x6FBA)
        self.assertEqual(guid.data3, 0x4FCF)
        self.assertEqual(
            bytes(guid.data4), bytes.fromhex("9d557b8e7f157091"))

    def test_memory_layout_is_little_endian_for_the_first_three_groups(self):
        # The text groups are numbers; the little-endian byte order belongs to
        # the layout.  Getting this backwards activated a different CLSID in
        # the COM fixture, so pin the bytes, not just the fields.
        guid = windows_api.guid_from_text(
            "{3EB685DB-65F9-4CF6-A03A-E3EF65729F3D}")

        self.assertEqual(ctypes.sizeof(guid), 16)
        self.assertEqual(
            bytes(guid), bytes.fromhex("db85b63ef965f64ca03ae3ef65729f3d"))

    def test_malformed_guid_is_rejected(self):
        for text in ("", "nonsense", "{F1B32785-6FBA-4FCF}",
                     "{F1B32785-6FBA-4FCF-9D55-7B8E7F15709Z}"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                windows_api.guid_from_text(text)


class FlagValueTests(unittest.TestCase):
    def test_flag_values_match_the_reference(self):
        # https://learn.microsoft.com/en-us/windows/win32/api/shlobj_core/ne-shlobj_core-known_folder_flag
        self.assertEqual(windows_api.KF_FLAG_DEFAULT, 0x00000000)
        self.assertEqual(windows_api.KF_FLAG_NO_PACKAGE_REDIRECTION, 0x00010000)


class DeclarationSafetyTests(unittest.TestCase):
    def test_same_signature_may_be_declared_again(self):
        key = ("test-library", "Repeated")
        windows_api._record_signature(
            *key, ctypes.c_long, (ctypes.c_void_p,))
        windows_api._record_signature(
            *key, ctypes.c_long, (ctypes.c_void_p,))

    def test_conflicting_redeclaration_is_refused(self):
        # ctypes keeps one function object per symbol, so a second, different
        # signature would silently replace the first one's marshalling.
        key = ("test-library", "Conflicting")
        windows_api._record_signature(
            *key, ctypes.c_long, (ctypes.c_void_p,))
        with self.assertRaises(windows_api.WindowsApiError):
            windows_api._record_signature(
                *key, ctypes.c_long, (ctypes.c_int,))


class OffPlatformTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "needs a non-Windows host")
    def test_known_folder_refuses_off_windows(self):
        with self.assertRaises(windows_api.WindowsUnavailableError):
            windows_api.known_folder(windows_api.FOLDERID_LOCAL_APP_DATA)


class ConfigBaseDirectoryTests(unittest.TestCase):
    def test_explicit_override_wins_on_windows_too(self):
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(windows_api, "known_folder") as known:
            self.assertEqual(
                paths.config_base_directory({"XDG_CONFIG_HOME": "/configuration"}),
                "/configuration")
        known.assert_not_called()

    def test_windows_uses_the_local_app_data_known_folder(self):
        base = os.path.join(tempfile.gettempdir(), "Local")
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(windows_api, "known_folder",
                                  return_value=base) as known:
            self.assertEqual(paths.config_base_directory({}), base)

        known.assert_called_once_with(
            windows_api.FOLDERID_LOCAL_APP_DATA,
            windows_api.KF_FLAG_NO_PACKAGE_REDIRECTION)

    def test_redirected_local_app_data_is_refused(self):
        redirected = "\\\\server\\share\\Local"
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(windows_api, "known_folder",
                                  return_value=redirected):
            with self.assertRaisesRegex(
                    windows_api.WindowsApiError, "redirected"):
                paths.config_base_directory({})

    def test_credential_directory_follows_the_base_directory(self):
        base = os.path.join(tempfile.gettempdir(), "Local")
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(windows_api, "known_folder",
                                  return_value=base):
            self.assertEqual(
                paths.credential_directory({}),
                os.path.join(base, "loki", "credentials"))


class StateBaseDirectoryTests(unittest.TestCase):
    def test_explicit_xdg_state_home_wins(self):
        self.assertEqual(
            paths.state_base_directory({"XDG_STATE_HOME": "/state"}),
            "/state")

    def test_posix_default_is_local_state(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(
                paths.state_base_directory({}),
                os.path.expanduser("~/.local/state"))

    def test_windows_has_no_config_state_split(self):
        # Windows has no XDG state directory, so state shares the application
        # data root with configuration rather than inventing a second tree.
        base = os.path.join(tempfile.gettempdir(), "Local")
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(windows_api, "known_folder",
                                  return_value=base):
            self.assertEqual(paths.state_base_directory({}), base)
            self.assertEqual(paths.config_base_directory({}), base)

    def test_loki_state_dir_appends_the_app_directory(self):
        self.assertEqual(
            paths.loki_state_dir({"XDG_STATE_HOME": "/state"}),
            os.path.join("/state", "loki"))


if __name__ == "__main__":
    unittest.main()
