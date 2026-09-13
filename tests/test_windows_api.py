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


class ContainerDeclarationTests(unittest.TestCase):
    def test_token_information_classes_match_the_reference(self):
        # TOKEN_INFORMATION_CLASS
        self.assertEqual(windows_api.TOKEN_IS_APP_CONTAINER_CLASS, 29)
        self.assertEqual(windows_api.TOKEN_APP_CONTAINER_SID_CLASS, 31)

    def test_process_access_and_launch_values_match_the_reference(self):
        self.assertEqual(windows_api.PROCESS_QUERY_LIMITED_INFORMATION, 0x1000)
        self.assertEqual(windows_api.CREATE_SUSPENDED, 0x00000004)
        self.assertEqual(windows_api.EXTENDED_STARTUPINFO_PRESENT, 0x00080000)
        # Values from the tested launch in tests/test_windows_appcontainers.py.
        self.assertEqual(
            windows_api.PROC_THREAD_ATTRIBUTE_HANDLE_LIST, 0x20002)
        self.assertEqual(
            windows_api.PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, 0x20009)

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

    def test_file_access_values_match_the_reference(self):
        self.assertEqual(windows_api.GENERIC_READ, 0x80000000)
        self.assertEqual(windows_api.GENERIC_WRITE, 0x40000000)
        self.assertEqual(windows_api.WRITE_DAC, 0x00040000)
        self.assertEqual(windows_api.FILE_FLAG_BACKUP_SEMANTICS, 0x02000000)
        self.assertEqual(windows_api.OPEN_EXISTING, 3)
        self.assertEqual(windows_api.ERROR_ACCESS_DENIED, 5)
        self.assertEqual(windows_api.INVALID_HANDLE_VALUE,
                         ctypes.c_void_p(-1).value)

    def test_access_masks_match_the_winnt_values(self):
        # These decide what a DACL grant means and what the containment probe
        # asks for; a wrong digit is invisible on the host that writes it.
        self.assertEqual(windows_api.FILE_GENERIC_READ, 0x00120089)
        self.assertEqual(windows_api.FILE_GENERIC_WRITE, 0x00120116)
        self.assertEqual(windows_api.FILE_GENERIC_EXECUTE, 0x001200A0)
        self.assertEqual(windows_api.FILE_ALL_ACCESS, 0x001F01FF)
        self.assertEqual(windows_api.FILE_WRITE_DATA, 0x00000002)
        self.assertEqual(windows_api.FILE_APPEND_DATA, 0x00000004)
        self.assertEqual(windows_api.DELETE, 0x00010000)
        self.assertEqual(windows_api.READ_CONTROL, 0x00020000)
        self.assertEqual(windows_api.WRITE_OWNER, 0x00080000)
        self.assertEqual(windows_api.SYNCHRONIZE, 0x00100000)
        self.assertEqual(windows_api.ACCESS_SYSTEM_SECURITY, 0x01000000)
        self.assertEqual(windows_api.ERROR_FILE_NOT_FOUND, 2)
        self.assertEqual(windows_api.ERROR_PATH_NOT_FOUND, 3)


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
        self.assertTrue(seen["flags"] & windows_api.CREATE_SUSPENDED)
        self.assertTrue(
            seen["flags"] & windows_api.EXTENDED_STARTUPINFO_PRESENT)
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
        with mock.patch.object(
                windows_api, "bind",
                side_effect=lambda library, symbol, *rest: fakes[symbol]), \
                mock.patch.object(
                    windows_api, "drive_environment_entries",
                    return_value=["=C:=C:\\work"]):
            windows_api.create_process_in_app_container(
                "loki.exe", ["--runtime"], "S-1-15-2-1",
                environment={"Path": "C:\\bin"})

        # wchar_t is 2 bytes on Windows and 4 on this host; the drive entry
        # precedes the named ones.
        width = ctypes.sizeof(ctypes.c_wchar)
        encoding = "utf-16-le" if width == 2 else "utf-32-le"
        text = seen["environment"].decode(encoding)
        self.assertTrue(text.startswith("=C:=C:\\work\0"), repr(text))
        self.assertIn("Path=C:\\bin", text)


class FileAccessTests(unittest.TestCase):
    def test_a_successful_open_returns_the_handle(self):
        with mock.patch.object(windows_api, "bind",
                               return_value=lambda *arguments: 77):
            self.assertEqual(
                windows_api.open_with_access("/x", windows_api.GENERIC_READ),
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
                windows_api.open_with_access("/x", windows_api.GENERIC_READ)
        self.assertEqual(caught.exception.status,
                         windows_api.ERROR_ACCESS_DENIED)


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

    def test_handle_relative_values_match_the_reference(self):
        self.assertEqual(windows_api.OBJ_CASE_INSENSITIVE, 0x40)
        self.assertEqual(windows_api.FILE_NON_DIRECTORY_FILE, 0x40)
        self.assertEqual(windows_api.FILE_SYNCHRONOUS_IO_NONALERT, 0x20)
        self.assertEqual(windows_api.FILE_OPEN_REPARSE_POINT, 0x00200000)
        self.assertEqual(windows_api.FILE_OPEN, 1)
        self.assertEqual(windows_api.FILE_CREATE, 2)
        self.assertEqual(windows_api.FILE_OPEN_IF, 3)
        self.assertEqual(windows_api.FILE_ATTRIBUTE_DIRECTORY, 0x10)
        self.assertEqual(windows_api.FILE_ATTRIBUTE_REPARSE_POINT, 0x400)
        self.assertEqual(windows_api.FILE_ATTRIBUTE_NORMAL, 0x80)
        self.assertEqual(windows_api.FILE_READ_ATTRIBUTES, 0x80)
        self.assertEqual(windows_api.FILE_RENAME_INFORMATION, 10)
        self.assertEqual(windows_api.FILE_DISPOSITION_INFO, 4)
        self.assertEqual(windows_api.OWNER_SECURITY_INFORMATION, 0x1)

    def test_ntstatus_values_match_the_reference(self):
        self.assertEqual(windows_api.STATUS_OBJECT_NAME_NOT_FOUND, 0xC0000034)
        self.assertEqual(windows_api.STATUS_OBJECT_PATH_NOT_FOUND, 0xC000003A)
        self.assertEqual(windows_api.STATUS_OBJECT_NAME_COLLISION, 0xC0000035)
        self.assertEqual(windows_api.STATUS_ACCESS_DENIED, 0xC0000022)
        self.assertEqual(windows_api.STATUS_NOT_A_DIRECTORY, 0xC0000103)

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

    def test_handle_inheritance_flag_matches_the_reference(self):
        self.assertEqual(windows_api.HANDLE_FLAG_INHERIT, 0x00000001)

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
            seen, [(0x33, windows_api.HANDLE_FLAG_INHERIT, 0)])

    def test_set_handle_information_failure_raises_with_the_win32_status(self):
        with mock.patch.object(
                windows_api, "bind",
                return_value=lambda *arguments: False), \
                mock.patch.object(windows_api.ctypes, "get_last_error",
                                  return_value=6, create=True):
            with self.assertRaises(windows_api.WindowsApiError) as caught:
                windows_api.set_handle_information(
                    1, windows_api.HANDLE_FLAG_INHERIT, 0)
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

    def test_lock_values_match_the_reference(self):
        self.assertEqual(windows_api.LOCKFILE_FAIL_IMMEDIATELY, 0x1)
        self.assertEqual(windows_api.LOCKFILE_EXCLUSIVE_LOCK, 0x2)
        self.assertEqual(windows_api.ERROR_LOCK_VIOLATION, 33)


if __name__ == "__main__":
    unittest.main()
