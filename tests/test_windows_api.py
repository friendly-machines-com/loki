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
from ctypes import wintypes
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


class ExtendedPathTests(unittest.TestCase):
    """GetFinalPathNameByHandleW returns a prefixed path; comparisons do not.

    The prefix literals are easy to get a backslash wrong, and the mistake
    changes nothing off Windows, so the shapes are pinned on every host.
    """

    def test_the_extended_length_prefix_is_stripped(self):
        self.assertEqual(
            windows_api._strip_extended_prefix('\\\\?\\C:\\work'),
            'C:\\work')

    def test_a_unc_share_keeps_its_double_leading_backslash(self):
        self.assertEqual(
            windows_api._strip_extended_prefix('\\\\?\\UNC\\server\\share'),
            '\\\\server\\share')

    def test_a_plain_path_is_returned_unchanged(self):
        self.assertEqual(
            windows_api._strip_extended_prefix('C:\\work'), 'C:\\work')


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
            windows_api.KnownFolderFlags.KF_FLAG_NO_PACKAGE_REDIRECTION)

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


class ContainerDeclarationTests(unittest.TestCase):
    def test_security_capabilities_layout_is_two_pointers_and_two_dwords(self):
        pointer = ctypes.sizeof(ctypes.c_void_p)
        capabilities = windows_api.SecurityCapabilities
        self.assertEqual(ctypes.sizeof(capabilities), 2 * pointer + 8)
        self.assertEqual(capabilities.AppContainerSid.offset, 0)
        self.assertEqual(capabilities.Capabilities.offset, pointer)
        self.assertEqual(capabilities.CapabilityCount.offset, 2 * pointer)
        self.assertEqual(capabilities.Reserved.offset, 2 * pointer + 4)

    def test_process_information_layout_follows_the_handles(self):
        pointer = ctypes.sizeof(ctypes.c_void_p)
        information = windows_api.ProcessInformation
        self.assertEqual(ctypes.sizeof(information), 2 * pointer + 8)
        self.assertEqual(information.hProcess.offset, 0)
        self.assertEqual(information.hThread.offset, pointer)
        self.assertEqual(information.dwProcessId.offset, 2 * pointer)
        self.assertEqual(information.dwThreadId.offset, 2 * pointer + 4)

    def test_startup_info_ex_carries_the_attribute_list_last(self):
        pointer = ctypes.sizeof(ctypes.c_void_p)
        self.assertEqual(windows_api.StartupInfo.cb.offset, 0)
        self.assertEqual(
            windows_api.StartupInfoEx.lpAttributeList.offset,
            ctypes.sizeof(windows_api.StartupInfo))
        # hStdInput, hStdOutput and hStdError are the final three pointers.
        self.assertEqual(
            windows_api.StartupInfo.hStdInput.offset + 3 * pointer,
            ctypes.sizeof(windows_api.StartupInfo))
        # 104 bytes on x64, 68 on x86.  Pinning the size is what actually
        # catches a member added or dropped in the middle.
        self.assertEqual(ctypes.sizeof(windows_api.StartupInfo),
                         9 * pointer + 32)


class TokenInspectionTests(unittest.TestCase):
    @staticmethod
    def _fake_get_info(writes):
        def get_info(token, info_class, buffer, length, returned):
            writes.append(info_class)
            if info_class == windows_api.TOKEN_IS_APP_CONTAINER_CLASS:
                if buffer:
                    ctypes.cast(buffer, ctypes.POINTER(wintypes.BOOL))[0] = True
            else:
                ctypes.cast(returned, ctypes.POINTER(wintypes.DWORD))[0] = (
                    ctypes.sizeof(ctypes.c_void_p))
                if buffer:
                    ctypes.cast(
                        buffer, ctypes.POINTER(ctypes.c_void_p))[0] = 0x1234
            return True
        return get_info

    def test_app_container_flag_is_read_from_the_token(self):
        writes = []
        with mock.patch.object(windows_api, "bind",
                               return_value=self._fake_get_info(writes)):
            self.assertTrue(windows_api.token_is_app_container(42))
        self.assertEqual(writes, [windows_api.TOKEN_IS_APP_CONTAINER_CLASS])

    def test_app_container_sid_is_read_from_the_returned_pointer(self):
        writes = []
        with mock.patch.object(windows_api, "bind",
                               return_value=self._fake_get_info(writes)), \
                mock.patch.object(windows_api, "sid_text",
                                  return_value="S-1-15-2-1") as convert:
            sid = windows_api.token_app_container_sid(42)

        self.assertEqual(sid, "S-1-15-2-1")
        self.assertEqual(
            writes, [windows_api.TOKEN_APP_CONTAINER_SID_CLASS] * 2)
        convert.assert_called_once_with(0x1234)

    def test_a_null_package_sid_is_an_error(self):
        def get_info(token, info_class, buffer, length, returned):
            ctypes.cast(returned, ctypes.POINTER(wintypes.DWORD))[0] = (
                ctypes.sizeof(ctypes.c_void_p))
            return True

        with mock.patch.object(windows_api, "bind", return_value=get_info):
            with self.assertRaises(windows_api.WindowsApiError):
                windows_api.token_app_container_sid(42)


class AppContainerLaunchTests(unittest.TestCase):
    def test_launch_carries_security_capabilities_and_leaves_the_child_suspended(self):
        seen = {"attributes": []}

        def convert_sid(string_sid, out):
            seen["sid"] = string_sid
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 0xABCD
            return True

        def initialize(attribute_list, count, flags, size):
            seen.setdefault("counts", []).append(count)
            ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 256
            return attribute_list is not None

        def update(attribute_list, flags, attribute, value, size, previous,
                   returned):
            seen["attributes"].append((flags, attribute, value, size))
            return True

        def delete(attribute_list):
            seen["deleted"] = True

        def local_free(pointer):
            seen["freed"] = True

        def create(application, command_line, process_attributes,
                   thread_attributes, inherit, flags, environment, directory,
                   startup, information):
            seen["flags"] = flags
            seen["inherit"] = inherit
            seen["command_line"] = command_line.value
            process = ctypes.cast(
                information,
                ctypes.POINTER(windows_api.ProcessInformation)).contents
            process.hProcess = 1
            process.hThread = 2
            process.dwProcessId = 3
            return True

        fakes = {
            "ConvertStringSidToSidW": convert_sid,
            "LocalFree": local_free,
            "InitializeProcThreadAttributeList": initialize,
            "UpdateProcThreadAttribute": update,
            "DeleteProcThreadAttributeList": delete,
            "CreateProcessW": create,
        }
        with mock.patch.object(
                windows_api, "bind",
                side_effect=lambda library, symbol, *rest: fakes[symbol]), \
                mock.patch.object(
                    windows_api, "drive_environment_entries",
                    return_value=[]):
            information = windows_api.create_process_in_app_container(
                "loki.exe", ["--runtime", "x"], "S-1-15-2-1")

        self.assertEqual(information.dwProcessId, 3)
        self.assertEqual(seen["sid"], "S-1-15-2-1")
        self.assertTrue(seen["deleted"])
        self.assertTrue(seen["freed"])
        self.assertFalse(seen["inherit"])
        self.assertTrue(
            seen["flags"] & windows_api.CreateProcessFlags.CREATE_SUSPENDED)
        self.assertTrue(
            seen["flags"]
            & windows_api.CreateProcessFlags.EXTENDED_STARTUPINFO_PRESENT)
        self.assertIn("loki.exe", seen["command_line"])

        capabilities_calls = [
            call for call in seen["attributes"]
            if call[1] == windows_api.PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES]
        self.assertEqual(len(capabilities_calls), 1)
        flags, attribute, value, size = capabilities_calls[0]
        self.assertEqual(flags, 0, "dwFlags must precede the attribute id")
        self.assertEqual(size, ctypes.sizeof(windows_api.SecurityCapabilities))
        capabilities = ctypes.cast(
            value, ctypes.POINTER(windows_api.SecurityCapabilities)).contents
        self.assertEqual(capabilities.AppContainerSid, 0xABCD)
        self.assertEqual(capabilities.CapabilityCount, 0)
        self.assertEqual(seen["counts"], [1, 1],
                         "the list is sized for exactly the attributes set")

    def test_inherited_handles_size_the_attribute_list_for_two(self):
        seen = {"attributes": [], "counts": []}

        def convert_sid(string_sid, out):
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 0xABCD
            return True

        def initialize(attribute_list, count, flags, size):
            seen["counts"].append(count)
            ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 256
            return attribute_list is not None

        def update(attribute_list, flags, attribute, value, size, previous,
                   returned):
            seen["attributes"].append((flags, attribute))
            return True

        def delete(attribute_list):
            pass

        def local_free(pointer):
            pass

        def create(application, command_line, process_attributes,
                   thread_attributes, inherit, flags, environment, directory,
                   startup, information):
            seen["inherit"] = inherit
            return True

        fakes = {
            "ConvertStringSidToSidW": convert_sid,
            "LocalFree": local_free,
            "InitializeProcThreadAttributeList": initialize,
            "UpdateProcThreadAttribute": update,
            "DeleteProcThreadAttributeList": delete,
            "CreateProcessW": create,
        }
        with mock.patch.object(
                windows_api, "bind",
                side_effect=lambda library, symbol, *rest: fakes[symbol]), \
                mock.patch.object(
                    windows_api, "drive_environment_entries",
                    return_value=[]):
            windows_api.create_process_in_app_container(
                "loki.exe", ["--runtime"], "S-1-15-2-1",
                inherited_handles=[0x11])

        self.assertEqual(seen["counts"], [2, 2])
        self.assertTrue(seen["inherit"])
        self.assertEqual(
            [attribute for _, attribute in seen["attributes"]],
            [windows_api.PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
             windows_api.PROC_THREAD_ATTRIBUTE_HANDLE_LIST])

    def test_pseudoconsole_adds_a_third_attribute(self):
        seen = {"attributes": [], "counts": []}

        def convert_sid(string_sid, out):
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 0xABCD
            return True

        def initialize(attribute_list, count, flags, size):
            seen["counts"].append(count)
            ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 256
            return attribute_list is not None

        def update(attribute_list, flags, attribute, value, size, previous,
                   returned):
            seen["attributes"].append((flags, attribute))
            return True

        def create(application, command_line, process_attributes,
                   thread_attributes, inherit, flags, environment, directory,
                   startup, information):
            return True

        fakes = {
            "ConvertStringSidToSidW": convert_sid,
            "LocalFree": lambda pointer: None,
            "InitializeProcThreadAttributeList": initialize,
            "UpdateProcThreadAttribute": update,
            "DeleteProcThreadAttributeList": lambda attribute_list: None,
            "CreateProcessW": create,
        }
        with mock.patch.object(
                windows_api, "bind",
                side_effect=lambda library, symbol, *rest: fakes[symbol]), \
                mock.patch.object(
                    windows_api, "drive_environment_entries",
                    return_value=[]):
            windows_api.create_process_in_app_container(
                "loki.exe", ["--runtime"], "S-1-15-2-1",
                inherited_handles=[0x11], pseudoconsole=0x1234)

        self.assertEqual(seen["counts"], [3, 3],
                         "the list is sized for the package SID, the handle "
                         "list and the pseudoconsole")
        self.assertEqual(
            [attribute for _, attribute in seen["attributes"]],
            [windows_api.PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
             windows_api.PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
             windows_api.PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE])

    def test_environment_block_carries_the_drive_entries_first(self):
        seen = {}

        def convert_sid(string_sid, out):
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 0xABCD
            return True

        def initialize(attribute_list, count, flags, size):
            ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 256
            return attribute_list is not None

        def update(*arguments):
            return True

        def create(application, command_line, process_attributes,
                   thread_attributes, inherit, flags, environment, directory,
                   startup, information):
            seen["environment"] = bytes(environment)
            process = ctypes.cast(
                information,
                ctypes.POINTER(windows_api.ProcessInformation)).contents
            process.hProcess = 1
            process.hThread = 2
            process.dwProcessId = 3
            return True

        fakes = {
            "ConvertStringSidToSidW": convert_sid,
            "LocalFree": lambda pointer: None,
            "InitializeProcThreadAttributeList": initialize,
            "UpdateProcThreadAttribute": update,
            "DeleteProcThreadAttributeList": lambda attribute_list: None,
            "CreateProcessW": create,
        }

        def run(drive_entries, current_directory):
            with mock.patch.object(
                    windows_api, "bind",
                    side_effect=lambda library, symbol, *rest: fakes[symbol]), \
                    mock.patch.object(
                        windows_api, "drive_environment_entries",
                        return_value=drive_entries):
                windows_api.create_process_in_app_container(
                    "loki.exe", ["--runtime"], "S-1-15-2-1",
                    environment={"Path": "C:\\bin"},
                    current_directory=current_directory)
            # wchar_t is 2 bytes on Windows and 4 on this host.
            width = ctypes.sizeof(ctypes.c_wchar)
            encoding = "utf-16-le" if width == 2 else "utf-32-le"
            return seen["environment"].decode(encoding)

        # A drive entry the parent already holds is carried over.
        text = run(["=C:=C:\\work"], None)
        self.assertTrue(text.startswith("=C:=C:\\work\0"), repr(text))
        self.assertIn("Path=C:\\bin", text)
        # With none to carry, the child's own drive is set from its current
        # directory -- the block element the reference says must be supplied.
        text = run([], "D:\\work\\ws")
        self.assertTrue(text.startswith("=D:=D:\\work\\ws\0"), repr(text))
        self.assertIn("Path=C:\\bin", text)


class FileAccessTests(unittest.TestCase):
    def test_a_successful_open_returns_the_handle(self):
        with mock.patch.object(windows_api, "bind",
                               return_value=lambda *arguments: 77):
            self.assertEqual(
                windows_api.open_with_access(
                    "/x", windows_api.AccessMask.GENERIC_READ),
                77)

    def test_an_invalid_handle_raises_with_the_win32_status(self):
        # ``ctypes.get_last_error`` exists only on Windows, so it is created
        # for the duration of the test rather than imported.
        with mock.patch.object(
                windows_api, "bind",
                return_value=lambda *arguments: windows_api.INVALID_HANDLE_VALUE), \
                mock.patch.object(windows_api.ctypes, "get_last_error",
                                  return_value=windows_api.ERROR_ACCESS_DENIED,
                                  create=True):
            with self.assertRaises(windows_api.WindowsApiError) as caught:
                windows_api.open_with_access(
                    "/x", windows_api.AccessMask.GENERIC_READ)
        self.assertEqual(caught.exception.status,
                         windows_api.ERROR_ACCESS_DENIED)

    def test_the_identity_open_asks_for_attribute_read_only(self):
        # ``GetFinalPathNameByHandleW`` fails on a handle opened with no access,
        # so the identity open must request FILE_READ_ATTRIBUTES -- but not
        # GENERIC_READ or READ_CONTROL, which the workspace's Modify grant does
        # not include and which the contained runtime therefore lacks.
        calls = {}

        def create_file(path, access, share, *rest):
            calls.update(access=access, share=share)
            return 77

        with mock.patch.object(windows_api, "bind", return_value=create_file):
            self.assertEqual(windows_api.open_directory_handle("/x"), 77)
        self.assertEqual(calls["access"],
                         windows_api.AccessMask.FILE_READ_ATTRIBUTES)
        self.assertEqual(calls["share"], windows_api.FILE_SHARE_ALL)
        self.assertEqual(calls["access"] & windows_api.AccessMask.READ_CONTROL,
                         0)
        self.assertEqual(calls["access"] & windows_api.AccessMask.GENERIC_READ,
                         0)


class HandleRelativeFileDeclarationTests(unittest.TestCase):
    """Layout and values the handle-relative credential operations rely on."""

    def test_unicode_string_lengths_are_bytes_and_the_buffer_is_a_pointer(self):
        pointer = ctypes.sizeof(ctypes.c_void_p)
        self.assertEqual(windows_api.UnicodeString.Length.offset, 0)
        self.assertEqual(windows_api.UnicodeString.MaximumLength.offset, 2)
        self.assertEqual(windows_api.UnicodeString.Buffer.offset, pointer)
        self.assertEqual(ctypes.sizeof(windows_api.UnicodeString), 2 * pointer)

    def test_object_attributes_puts_the_root_directory_after_the_length(self):
        # RootDirectory is what makes an open relative; a member added or
        # dropped in the middle would shift it and every field after it.
        pointer = ctypes.sizeof(ctypes.c_void_p)
        attributes = windows_api.ObjectAttributes
        self.assertEqual(attributes.Length.offset, 0)
        self.assertEqual(attributes.RootDirectory.offset, pointer)
        self.assertEqual(attributes.ObjectName.offset, 2 * pointer)
        self.assertEqual(attributes.Attributes.offset, 3 * pointer)
        self.assertEqual(attributes.SecurityDescriptor.offset, 4 * pointer)
        self.assertEqual(
            attributes.SecurityQualityOfService.offset, 5 * pointer)
        self.assertEqual(ctypes.sizeof(attributes), 6 * pointer)

    def test_io_status_block_is_a_status_then_a_size(self):
        pointer = ctypes.sizeof(ctypes.c_void_p)
        self.assertEqual(windows_api.IoStatusBlock.Status.offset, 0)
        self.assertEqual(windows_api.IoStatusBlock.Information.offset, pointer)
        self.assertEqual(ctypes.sizeof(windows_api.IoStatusBlock), 2 * pointer)

    def test_file_rename_information_carries_a_root_and_a_counted_name(self):
        pointer = ctypes.sizeof(ctypes.c_void_p)
        rename = windows_api.FileRenameInformation
        self.assertEqual(rename.ReplaceIfExists.offset, 0)
        self.assertEqual(rename.RootDirectory.offset, pointer)
        self.assertEqual(rename.FileNameLength.offset, 2 * pointer)
        self.assertEqual(rename.FileName.offset, 2 * pointer + 4)

    def test_by_handle_file_information_is_all_fixed_width(self):
        # All c_uint32, so the layout is the same on every host -- which is the
        # point of not using wintypes.DWORD, that is c_ulong (8 bytes) on LP64.
        information = windows_api.ByHandleFileInformation
        self.assertEqual(ctypes.sizeof(information), 52)
        self.assertEqual(information.dwFileAttributes.offset, 0)
        self.assertEqual(information.ftCreationTime.offset, 4)
        self.assertEqual(information.ftLastAccessTime.offset, 12)
        self.assertEqual(information.ftLastWriteTime.offset, 20)
        self.assertEqual(information.dwVolumeSerialNumber.offset, 28)
        self.assertEqual(information.nFileSizeHigh.offset, 32)
        self.assertEqual(information.nFileSizeLow.offset, 36)
        self.assertEqual(information.nNumberOfLinks.offset, 40)
        self.assertEqual(information.nFileIndexHigh.offset, 44)
        self.assertEqual(information.nFileIndexLow.offset, 48)
        self.assertEqual(ctypes.sizeof(windows_api.FileTime), 8)

    def test_full_sids_pass_through_and_aliases_resolve(self):
        # A full SID is already canonical.  Off Windows an alias resolves
        # through the documented table (on Windows the API is authoritative),
        # which is what makes the privacy predicate testable on this host.
        self.assertEqual(
            windows_api.canonical_sid("S-1-5-21-1-2-3-1001"),
            "S-1-5-21-1-2-3-1001")
        self.assertEqual(windows_api.canonical_sid("S-1-3-4"), "S-1-3-4")
        self.assertEqual(windows_api.canonical_sid("SY"), "S-1-5-18")
        self.assertEqual(windows_api.canonical_sid("BA"), "S-1-5-32-544")
        self.assertEqual(windows_api.canonical_sid("OW"), "S-1-3-4")
        self.assertEqual(windows_api.canonical_sid("WD"), "S-1-1-0")


class PipeDeclarationTests(unittest.TestCase):
    """The anonymous-pipe primitive the credential transport is built on."""

    def test_security_attributes_layout_is_two_pointers_and_a_dword(self):
        # bInheritHandle must be a fixed-width DWORD, not wintypes.BOOL (which
        # is c_long and therefore 8 bytes on an LP64 host).  SECURITY_ATTRIBUTES
        # is DWORD, LPVOID, BOOL: the 8-byte-aligned pointer pushes the flag to
        # 2*pointer and the structure to 3*pointer.
        pointer = ctypes.sizeof(ctypes.c_void_p)
        attributes = windows_api.SecurityAttributes
        self.assertEqual(attributes.nLength.offset, 0)
        self.assertEqual(attributes.nLength.size, 4)
        self.assertEqual(attributes.lpSecurityDescriptor.offset, pointer)
        self.assertEqual(attributes.bInheritHandle.offset, 2 * pointer)
        self.assertEqual(attributes.bInheritHandle.size, 4)
        self.assertEqual(ctypes.sizeof(attributes), 3 * pointer)

    def test_create_pipe_returns_the_two_ends_and_requests_inheritance(self):
        seen = {}

        def create(read_out, write_out, attributes, size):
            structure = ctypes.cast(
                attributes,
                ctypes.POINTER(windows_api.SecurityAttributes)).contents
            seen["inherit"] = structure.bInheritHandle
            seen["length"] = structure.nLength
            seen["size"] = size
            ctypes.cast(read_out, ctypes.POINTER(ctypes.c_void_p))[0] = 0x11
            ctypes.cast(write_out, ctypes.POINTER(ctypes.c_void_p))[0] = 0x22
            return True

        with mock.patch.object(windows_api, "bind", return_value=create):
            self.assertEqual(windows_api.create_pipe(), (0x11, 0x22))
        self.assertEqual(seen["inherit"], 1)
        self.assertEqual(seen["length"],
                         ctypes.sizeof(windows_api.SecurityAttributes))
        self.assertEqual(seen["size"], 0)

    def test_create_pipe_without_inheritance_clears_the_flag(self):
        seen = {}

        def create(read_out, write_out, attributes, size):
            seen["inherit"] = ctypes.cast(
                attributes,
                ctypes.POINTER(windows_api.SecurityAttributes)
            ).contents.bInheritHandle
            ctypes.cast(read_out, ctypes.POINTER(ctypes.c_void_p))[0] = 0
            ctypes.cast(write_out, ctypes.POINTER(ctypes.c_void_p))[0] = 0
            return True

        with mock.patch.object(windows_api, "bind", return_value=create):
            windows_api.create_pipe(inherit=False)
        self.assertEqual(seen["inherit"], 0)

    def test_create_pipe_failure_raises_with_the_win32_status(self):
        with mock.patch.object(
                windows_api, "bind",
                return_value=lambda *arguments: False), \
                mock.patch.object(windows_api.ctypes, "get_last_error",
                                  return_value=6, create=True):
            with self.assertRaises(windows_api.WindowsApiError) as caught:
                windows_api.create_pipe()
        self.assertEqual(caught.exception.status, 6)

    def test_clear_handle_inheritance_clears_only_that_flag(self):
        seen = []

        def set_information(handle, mask, flags):
            seen.append((handle, mask, flags))
            return True

        with mock.patch.object(windows_api, "bind",
                               return_value=set_information):
            windows_api.clear_handle_inheritance(0x33)
        self.assertEqual(
            seen, [(0x33, windows_api.HandleFlags.HANDLE_FLAG_INHERIT, 0)])

    def test_set_handle_information_failure_raises_with_the_win32_status(self):
        with mock.patch.object(
                windows_api, "bind",
                return_value=lambda *arguments: False), \
                mock.patch.object(windows_api.ctypes, "get_last_error",
                                  return_value=6, create=True):
            with self.assertRaises(windows_api.WindowsApiError) as caught:
                windows_api.set_handle_information(
                    1, windows_api.HandleFlags.HANDLE_FLAG_INHERIT, 0)
        self.assertEqual(caught.exception.status, 6)


class RangeLockDeclarationTests(unittest.TestCase):
    def test_overlapped_states_the_offset_the_lock_starts_at(self):
        pointer = ctypes.sizeof(ctypes.c_void_p)
        overlapped = windows_api.Overlapped
        self.assertEqual(overlapped.Internal.offset, 0)
        self.assertEqual(overlapped.InternalHigh.offset, pointer)
        self.assertEqual(overlapped.Offset.offset, 2 * pointer)
        self.assertEqual(overlapped.OffsetHigh.offset, 2 * pointer + 4)
        self.assertEqual(overlapped.hEvent.offset, 2 * pointer + 8)
        self.assertEqual(ctypes.sizeof(overlapped), 3 * pointer + 8)


if __name__ == "__main__":
    unittest.main()
