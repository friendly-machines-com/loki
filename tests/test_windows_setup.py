"""Portable tests for the Windows setup editor's logic and rules.

The Tk editor and the ACL/profile calls need Windows and are deliberately not
covered here.  Everything that *decides* something -- path rules, warnings,
naming, SDDL composition, the ledger, the plan diff, and the platform gate --
is pure, and is checked on every platform because a mistake there either grants
too much or refuses something legitimate.
"""

import contextlib
import ctypes
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from directory_fixtures import directory_alias
from windows_state_fixtures import directory_identity_patch, directory_handle, real_directory_locations

from loki_agent import windows_acl
from loki_agent import windows_api
from loki_agent import windows_containers
from loki_agent import windows_setup
from loki_agent import windows_state


# The canonical path of an object is asked of the kernel about an open handle
# (windows_state._final_path), which only exists on Windows.  These modules test
# the ledger, plan and gate logic on every host, so they substitute the
# platform's canonicalisation with its Python equivalent; the real call is
# exercised by the Windows legs.


def setUpModule():
    unittest.enterModuleContext(directory_identity_patch())


PACKAGE_SID = "S-1-15-2-1-2-3-4-5-6-7-8"
OTHER_SID = "S-1-5-21-1-2-3-1004"


class AccessLevelTests(unittest.TestCase):
    def test_read_is_generic_read_and_execute(self):
        # Execute on a directory is traverse, which known-path access needs.
        self.assertEqual(windows_state.access_sddl(windows_setup.Access.READ),
                         "FRFX")

    def test_write_uses_modify_not_all_access(self):
        # FA would include WRITE_DAC, letting a tool rewrite the DACL and lock
        # the owner out of their own directory.
        self.assertEqual(
            windows_state.access_sddl(windows_setup.Access.READ_WRITE),
            "0x1301BF")


class SddlEditingTests(unittest.TestCase):
    EXISTING = "D:PAI(A;OICI;FA;;;SY)(A;OICI;FA;;;" + OTHER_SID + ")"

    def test_add_appends_an_inheritable_ace_and_keeps_existing_ones(self):
        result = windows_setup.add_package_ace(
            self.EXISTING, PACKAGE_SID, windows_setup.Access.READ)

        self.assertTrue(result.startswith(self.EXISTING))
        self.assertEqual(result, self.EXISTING + "(A;OICI;FRFX;;;%s)"
                         % PACKAGE_SID)
        self.assertTrue(windows_acl.names_package(result, PACKAGE_SID))

    def test_add_rejects_something_that_is_not_a_dacl(self):
        with self.assertRaises(windows_api.WindowsApiError):
            windows_setup.add_package_ace("S:PAI", PACKAGE_SID,
                                          windows_setup.Access.READ)

    def test_remove_deletes_only_that_package_and_is_idempotent_afterwards(self):
        added = windows_setup.add_package_ace(
            self.EXISTING, PACKAGE_SID, windows_setup.Access.READ_WRITE)

        removed = windows_setup.remove_package_aces(added, PACKAGE_SID)

        self.assertEqual(removed, self.EXISTING)
        self.assertFalse(windows_acl.names_package(removed, PACKAGE_SID))
        self.assertEqual(
            windows_setup.remove_package_aces(removed, PACKAGE_SID), removed)

    def test_remove_leaves_other_packages_alone(self):
        both = windows_setup.add_package_ace(
            windows_setup.add_package_ace(
                self.EXISTING, PACKAGE_SID, windows_setup.Access.READ),
            OTHER_SID, windows_setup.Access.READ)

        removed = windows_setup.remove_package_aces(both, PACKAGE_SID)

        self.assertFalse(windows_acl.names_package(removed, PACKAGE_SID))
        self.assertTrue(windows_acl.names_package(removed, OTHER_SID))


class PackageAccessTests(unittest.TestCase):
    """The DACL must be judged on rights, not on naming the SID.

    Every case here is legal SDDL that the old substring test reported as a
    grant (or, on a protected tree, as a superset of its contents).  These are
    the false passes a start-up gate must not have.
    """

    READ = windows_state.Access.READ
    READ_WRITE = windows_state.Access.READ_WRITE

    def dacl(self, ace):
        return f"D:PAI{ace}"

    def test_access_mask_matches_the_sddl_levels(self):
        self.assertEqual(windows_state.access_mask(self.READ), 0x1200A9)
        self.assertEqual(windows_state.access_mask(self.READ_WRITE), 0x1301BF)

    def test_an_allow_ace_covering_the_level_is_a_grant(self):
        self.assertTrue(windows_state.grants_access(
            self.dacl(f"(A;OICI;FRFX;;;{PACKAGE_SID})"), PACKAGE_SID, self.READ))
        self.assertTrue(windows_state.grants_access(
            self.dacl(f"(A;OICI;0x1301BF;;;{PACKAGE_SID})"),
            PACKAGE_SID, self.READ_WRITE))

    def test_a_deny_ace_names_the_sid_without_granting_it(self):
        sddl = self.dacl(f"(D;OICI;FA;;;{PACKAGE_SID})")

        self.assertTrue(windows_acl.names_package(sddl, PACKAGE_SID))
        self.assertEqual(windows_acl.package_access(sddl, PACKAGE_SID), 0)
        self.assertFalse(
            windows_state.grants_access(sddl, PACKAGE_SID, self.READ))

    def test_a_deny_clears_the_bits_it_covers(self):
        # Allow the level, deny only DELETE: read survives, read-write does not
        # because Modify includes DELETE.
        sddl = self.dacl(
            f"(A;OICI;0x1301BF;;;{PACKAGE_SID})(D;;SD;;;{PACKAGE_SID})")

        self.assertTrue(
            windows_state.grants_access(sddl, PACKAGE_SID, self.READ))
        self.assertFalse(
            windows_state.grants_access(sddl, PACKAGE_SID, self.READ_WRITE))

    def test_a_mask_smaller_than_the_level_is_not_a_grant(self):
        # FILE_WRITE_ATTRIBUTES alone names the SID and grants no data access.
        sddl = self.dacl(f"(A;OICI;0x00000100;;;{PACKAGE_SID})")

        self.assertFalse(
            windows_state.grants_access(sddl, PACKAGE_SID, self.READ))

    def test_an_inherit_only_allow_does_not_apply_to_the_object(self):
        sddl = self.dacl(f"(A;OICIIO;FRFX;;;{PACKAGE_SID})")

        self.assertEqual(windows_acl.package_access(sddl, PACKAGE_SID), 0)
        self.assertFalse(
            windows_state.grants_access(sddl, PACKAGE_SID, self.READ))

    def test_package_allow_still_counts_an_inherit_only_allow(self):
        # It grants the tree's children, so the protected-tree check must fail.
        sddl = self.dacl(f"(A;OICIIO;FRFX;;;{PACKAGE_SID})")

        self.assertEqual(
            windows_acl.package_allow(sddl, PACKAGE_SID), 0x1200A9)

    def test_package_allow_ignores_a_deny_only_dacl(self):
        sddl = self.dacl(f"(D;OICI;FA;;;{PACKAGE_SID})")

        self.assertEqual(windows_acl.package_allow(sddl, PACKAGE_SID), 0)

    def test_read_does_not_cover_read_write(self):
        sddl = self.dacl(f"(A;OICI;FRFX;;;{PACKAGE_SID})")

        self.assertFalse(windows_state.grants_access(
            sddl, PACKAGE_SID, self.READ_WRITE))

    def test_modify_covers_read(self):
        sddl = self.dacl(f"(A;OICI;0x1301BF;;;{PACKAGE_SID})")

        self.assertTrue(
            windows_state.grants_access(sddl, PACKAGE_SID, self.READ))

    def test_an_ace_for_another_sid_grants_nothing(self):
        sddl = self.dacl(f"(A;OICI;FA;;;{OTHER_SID})")

        self.assertEqual(windows_acl.package_access(sddl, PACKAGE_SID), 0)
        self.assertFalse(
            windows_state.grants_access(sddl, PACKAGE_SID, self.READ))

    def test_an_inherited_allow_still_applies_to_the_object(self):
        sddl = self.dacl(f"(A;OICIID;FRFX;;;{PACKAGE_SID})")

        self.assertTrue(
            windows_state.grants_access(sddl, PACKAGE_SID, self.READ))

    def test_hex_and_named_masks_agree(self):
        named = self.dacl(f"(A;OICI;FR;;;{PACKAGE_SID})")
        hexed = self.dacl(f"(A;OICI;0x120089;;;{PACKAGE_SID})")

        self.assertEqual(windows_acl.package_access(named, PACKAGE_SID),
                         windows_acl.package_access(hexed, PACKAGE_SID))

    def test_named_file_masks_match_the_winnt_values(self):
        for code, mask in (("FR", 0x120089), ("FW", 0x120116),
                           ("FX", 0x1200A0), ("FA", 0x1F01FF)):
            with self.subTest(code=code):
                self.assertEqual(
                    windows_acl.package_access(
                        self.dacl(f"(A;OICI;{code};;;{PACKAGE_SID})"),
                        PACKAGE_SID),
                    mask)

    def test_an_object_ace_without_guids_is_evaluated(self):
        sddl = self.dacl(f"(OA;OICI;FRFX;;;{PACKAGE_SID})")

        self.assertTrue(
            windows_state.grants_access(sddl, PACKAGE_SID, self.READ))

    def test_an_object_ace_with_guids_is_refused_not_ignored(self):
        guid = "{11111111-2222-3333-4444-555555555555}"
        sddl = self.dacl(f"(OA;OICI;FRFX;{guid};;{PACKAGE_SID})")

        with self.assertRaises(windows_api.WindowsApiError):
            windows_acl.package_access(sddl, PACKAGE_SID)

    def test_an_unknown_rights_code_raises_rather_than_guessing(self):
        sddl = self.dacl(f"(A;OICI;ZZ;;;{PACKAGE_SID})")

        with self.assertRaises(windows_api.WindowsApiError):
            windows_state.grants_access(sddl, PACKAGE_SID, self.READ)


class ProtectedPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.config = os.path.join(self.root, "loki")
        self.credentials = os.path.join(self.config, "credentials")
        self.state = os.path.join(self.root, "state")
        os.makedirs(self.credentials)
        os.mkdir(self.state)
        os.mkdir(self.root + os.sep + 'elsewhere')
        self.enterContext(real_directory_locations())
        for target, attribute in ((self.config, "loki_config_dir"),
                                  (self.state, "loki_state_dir"),
                                  (self.credentials, "credential_directory")):
            patch = mock.patch.object(
                windows_setup.paths, attribute, return_value=target)
            patch.start()
            self.addCleanup(patch.stop)

    def test_refuses_a_grant_that_covers_the_credential_directory(self):
        self.assertTrue(windows_state.protected_path_errors(self.root))

    def test_refuses_the_credential_directory_itself(self):
        self.assertTrue(
            windows_state.protected_path_errors(self.credentials))

    def test_refuses_a_grant_inside_a_protected_tree(self):
        selected = self.config + os.sep + 'sessions'
        os.mkdir(selected)
        self.assertTrue(windows_state.protected_path_errors(selected))

    def test_refuses_the_runtime_tree(self):
        package = os.path.dirname(os.path.abspath(windows_setup.__file__))
        self.assertTrue(windows_state.protected_path_errors(package))

    def test_configuration_is_readable_but_not_writable(self):
        self.assertEqual(windows_state.protected_path_errors(
            self.config, windows_state.Access.READ), [])
        self.assertTrue(windows_state.protected_path_errors(
            self.config, windows_state.Access.READ_WRITE))
        for path in [self.root, self.credentials, self.state]:
            with self.subTest(path=path):
                self.assertTrue(windows_state.protected_path_errors(
                    path, windows_state.Access.READ))

    def test_overlapping_config_and_state_are_rejected(self):
        child = self.config + os.sep + 'state'
        os.mkdir(child)
        for state in [self.config, child, self.root]:
            with self.subTest(state=state), mock.patch.object(
                    windows_setup.paths, 'loki_state_dir', return_value=state):
                self.assertTrue(windows_state.configuration_path_errors())
                self.assertTrue(windows_state.protected_path_errors(
                    self.config, windows_state.Access.READ))

    def test_alias_of_state_cannot_be_granted_as_configuration(self):
        locate = windows_state._directory_location
        state_object = locate(self.state)
        with mock.patch.object(windows_state, '_directory_location', side_effect=lambda path:
                               state_object if path == self.config else locate(path)):
            self.assertTrue(windows_state.protected_path_errors(
                self.config, windows_state.Access.READ))

    def test_allows_an_unrelated_directory(self):
        self.assertEqual(
            windows_state.protected_path_errors(
                os.path.join(self.root, "elsewhere")), [])


@unittest.skipUnless(sys.platform == 'win32', 'native Windows object containment')
class NativeObjectPolicyTests(unittest.TestCase):
    def setUp(self):
        self.root = self.enterContext(tempfile.TemporaryDirectory(prefix='loki-object-policy-'))
        self.enterContext(real_directory_locations())
        handle = windows_api.open_with_access(
            self.root, windows_api.AccessMask.FILE_READ_ATTRIBUTES | windows_api.AccessMask.FILE_WRITE_ATTRIBUTES,
            flags=windows_api.FileFlags.FILE_FLAG_BACKUP_SEMANTICS)
        try:
            setter = windows_api.bind('kernel32', 'SetFileInformationByHandle', ctypes.wintypes.BOOL,
                                      ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.wintypes.DWORD)
            enabled = ctypes.c_uint32(1)
            if not setter(handle, 23, ctypes.byref(enabled), ctypes.sizeof(enabled)):
                error = ctypes.get_last_error()
                if error in [1, 50]:
                    self.skipTest(f'this filesystem has no case-sensitive directories: {error}')
                raise ctypes.WinError(error)
        finally:
            windows_api.close_handle(handle)
        self.upper, self.lower, self.other = [self.root + os.sep + leaf for leaf in ['Case', 'case', 'other']]
        for path in [self.upper, self.lower, self.other]:
            os.mkdir(path)
        self.credentials = self.upper + os.sep + 'credentials'
        os.mkdir(self.credentials)

    @contextlib.contextmanager
    def layout(self, state):
        with contextlib.ExitStack() as cleanup:
            for name, path in [['loki_config_dir', self.upper], ['loki_state_dir', state],
                               ['credential_directory', self.credentials]]:
                cleanup.enter_context(mock.patch.object(windows_state.paths, name, return_value=path))
            yield

    def test_case_distinct_config_and_state_do_not_overlap(self):
        with self.layout(self.lower):
            self.assertEqual(windows_state.configuration_path_errors(), [])
            self.assertEqual(windows_state.protected_path_errors(
                self.lower, windows_state.Access.READ_WRITE, origin='state'), [])

    def test_actual_handle_not_its_label_selects_the_containment_verdict(self):
        with self.layout(self.other):
            handle = windows_api.open_directory_for_acl(self.lower)
            try:
                self.assertEqual(windows_state.handle_path_errors(handle, self.upper), [])
            finally:
                windows_api.close_handle(handle)
            handle = windows_api.open_directory_for_acl(self.upper)
            try:
                self.assertTrue(windows_state.handle_path_errors(handle, self.lower))
            finally:
                windows_api.close_handle(handle)

    def test_state_acl_update_refuses_a_case_lookalike_sibling_object(self):
        child = self.lower + os.sep + 'child.txt'
        with open(child, 'w', encoding='utf-8') as stream:
            stream.write('not managed state')
        backend = windows_setup.WindowsBackend({})
        root = windows_api.open_directory_for_acl(self.upper)
        try:
            with mock.patch.object(windows_setup, 'state_objects', return_value=[child]), \
                    mock.patch.object(windows_api, 'set_handle_dacl') as write:
                with self.assertRaises(windows_api.WindowsApiError):
                    backend._grant_state(PACKAGE_SID, windows_api.current_user_sid(), root)
            write.assert_not_called()
        finally:
            windows_api.close_handle(root)


class HandlePathTests(unittest.TestCase):
    """The check bound to the ACL write asks the write's own handle."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.config = os.path.join(self.root, "loki")
        self.credentials = os.path.join(self.config, "credentials")
        self.state = os.path.join(self.root, "state")
        os.makedirs(self.credentials)
        os.mkdir(self.state)
        os.mkdir(self.root + os.sep + 'elsewhere')
        self.enterContext(real_directory_locations())
        for target, attribute in ((self.config, "loki_config_dir"),
                                  (self.state, "loki_state_dir"),
                                  (self.credentials, "credential_directory")):
            patch = mock.patch.object(
                windows_setup.paths, attribute, return_value=target)
            patch.start()
            self.addCleanup(patch.stop)

    def test_refuses_the_credential_directory_through_the_handle(self):
        with directory_handle(self.credentials) as handle:
            errors = windows_state.handle_path_errors(handle, self.credentials)
        self.assertTrue(errors)

    def test_config_read_exception_is_bound_to_the_actual_handle(self):
        with directory_handle(self.config) as handle:
            self.assertEqual(windows_state.handle_path_errors(
                handle, self.config, windows_state.Access.READ), [])
        with directory_handle(self.credentials) as handle:
            self.assertTrue(windows_state.handle_path_errors(
                handle, self.config, windows_state.Access.READ))

    def test_allows_an_unrelated_directory_through_the_handle(self):
        elsewhere = self.root + os.sep + 'elsewhere'
        with directory_handle(elsewhere) as handle:
            self.assertEqual(windows_state.handle_path_errors(handle, elsewhere), [])


class GrantHandleTests(unittest.TestCase):
    """A grant reads and writes the DACL through the handle it was given."""

    EXISTING = "D:PAI(A;OICI;FA;;;SY)(A;OICI;FA;;;" + OTHER_SID + ")"

    def setUp(self):
        self.backend = windows_setup.WindowsBackend({})

    def test_reads_and_writes_the_same_handle(self):
        handle = object()
        with mock.patch.object(
                windows_api, "handle_dacl_sddl",
                return_value=self.EXISTING) as read, \
                mock.patch.object(
                    windows_containers, "set_handle_dacl_sddl") as write:
            self.backend._grant_path(
                handle, "/work", windows_setup.Access.READ_WRITE, PACKAGE_SID)
        read.assert_called_once_with(handle)
        self.assertIs(write.call_args.args[0], handle)

    def test_an_unchanged_dacl_is_not_rewritten(self):
        handle = object()
        unchanged = windows_setup.add_package_ace(
            self.EXISTING, PACKAGE_SID, windows_setup.Access.READ)
        with mock.patch.object(
                windows_api, "handle_dacl_sddl", return_value=unchanged), \
                mock.patch.object(
                    windows_containers, "set_handle_dacl_sddl") as write:
            self.backend._grant_path(
                handle, "/work", windows_setup.Access.READ, PACKAGE_SID)
        write.assert_not_called()

    def test_refuses_an_object_with_no_dacl(self):
        with mock.patch.object(
                windows_api, "handle_dacl_sddl", return_value=None):
            with self.assertRaises(windows_api.WindowsApiError):
                self.backend._grant_path(
                    object(), "/work", windows_setup.Access.READ, PACKAGE_SID)


class SecretWarningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = temporary.name
        self.enterContext(real_directory_locations())
        patch = mock.patch.object(
            windows_setup.os.path, "expanduser", return_value=self.home)
        patch.start()
        self.addCleanup(patch.stop)

    def test_warns_only_about_existing_covered_locations(self):
        os.makedirs(os.path.join(self.home, ".ssh"))

        warnings = windows_state.covered_secret_warnings(self.home)

        self.assertEqual(len(warnings), 1)
        self.assertIn(".ssh", warnings[0])

    def test_is_silent_when_nothing_is_there(self):
        self.assertEqual(
            windows_state.covered_secret_warnings(self.home), [])

    def test_is_silent_for_an_unrelated_directory(self):
        os.makedirs(os.path.join(self.home, ".ssh"))
        elsewhere = os.path.join(self.home, "projects")

        self.assertEqual(
            windows_state.covered_secret_warnings(elsewhere), [])


class NamingTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(real_directory_locations())

    def test_workspace_key_is_shared_by_native_aliases(self):
        with tempfile.TemporaryDirectory() as root:
            child = os.path.join(root, "project")
            os.makedirs(child)
            link = os.path.join(root, "link")
            directory_alias(child, link)

            self.assertEqual(
                windows_state.workspace_key(child),
                windows_state.workspace_key(link))
            self.assertEqual(
                windows_state.workspace_key(child),
                windows_state.workspace_key(child + os.sep))
            self.assertEqual(
                windows_state.workspace_key(child),
                windows_state.workspace_key(
                    os.path.join(child, "..", "project")))

    def test_profile_name_is_deterministic_and_usable(self):
        with tempfile.TemporaryDirectory() as root:
            project, other = root + os.sep + 'project', root + os.sep + 'other'
            os.mkdir(project)
            os.mkdir(other)
            first = windows_state.profile_name_for(project)
            self.assertEqual(first, windows_state.profile_name_for(project))
            self.assertNotEqual(first, windows_state.profile_name_for(other))
            self.assertTrue(windows_api.app_container_profile_name_is_usable(first))

    def test_directory_rename_keeps_the_workspace_key_and_profile(self):
        with tempfile.TemporaryDirectory() as root:
            original, renamed = root + os.sep + 'original', root + os.sep + 'renamed'
            os.mkdir(original)
            key = windows_state.workspace_key(original)
            profile = windows_state.profile_name_for(original)
            os.rename(original, renamed)
            self.assertEqual(windows_state.workspace_key(renamed), key)
            self.assertEqual(windows_state.profile_name_for(renamed), profile)

    def test_case_distinct_objects_have_distinct_workspace_keys_and_profiles(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = root + os.sep + 'Case', root + os.sep + 'case'
            os.mkdir(first)
            try:
                os.mkdir(second)
            except FileExistsError:
                self.skipTest('this filesystem does not create case-distinct directories')
            self.assertNotEqual(windows_state.workspace_key(first), windows_state.workspace_key(second))
            self.assertNotEqual(windows_state.profile_name_for(first), windows_state.profile_name_for(second))


class LedgerTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            location = os.path.join(root, "nested", "ledger.json")
            blob = {"version": windows_state.LEDGER_VERSION,
                    "workspaces": {"k": {"workspace": "/p",
                                         "profile": "Loki.Workspace.x",
                                         "grants": []}}}

            windows_setup.save_ledger(blob, location)

            self.assertEqual(windows_setup.load_ledger(location), blob)

    def test_missing_or_corrupt_ledger_is_empty(self):
        with tempfile.TemporaryDirectory() as root:
            missing = os.path.join(root, "absent.json")
            self.assertEqual(
                windows_setup.load_ledger(missing)["workspaces"], {})
            corrupt = os.path.join(root, "corrupt.json")
            with open(corrupt, "w", encoding="utf-8") as stream:
                json.dump({"workspaces": "not a mapping"}, stream)
            self.assertEqual(
                windows_setup.load_ledger(corrupt)["workspaces"], {})

    def test_recorded_workspace_label_preserves_the_callers_operand(self):
        raw = 'relative/link/../workspace//'
        entry = windows_state.ledger_entry(raw, 'profile', [])
        self.assertEqual(entry['workspace'], raw)


class PlanTests(unittest.TestCase):
    def definition(self, *grants):
        return windows_setup.Definition("/p", list(grants))

    def grant(self, path, access=windows_setup.Access.READ, origin="user"):
        return windows_setup.Grant(path, access, origin)

    def test_add_remove_and_level_changes(self):
        definition = self.definition(
            self.grant("/one"),
            self.grant("/two", windows_setup.Access.READ_WRITE))
        previous = windows_setup.ledger_entry(
            "/p", "Loki.Workspace.x",
            [self.grant("/two"),
             self.grant("/gone", origin="workspace")])

        plan = windows_setup.build_plan(definition, previous)

        kinds = sorted((change.kind, change.path) for change in plan.changes)
        self.assertEqual(kinds, [("add", "/one"),
                                 ("level", "/two"),
                                 ("remove", "/gone")])

    def test_no_changes_when_everything_matches(self):
        definition = self.definition(self.grant("/one"))
        previous = windows_setup.ledger_entry("/p", "x", [self.grant("/one")])

        self.assertEqual(
            windows_setup.build_plan(definition, previous).changes, [])

    def test_first_setup_can_plan_missing_config_without_creating_it(self):
        with tempfile.TemporaryDirectory() as root, real_directory_locations():
            # A workspace grant must not cover the protected config/state trees.
            workspace = root + os.sep + 'workspace'
            os.mkdir(workspace)
            config, state = root + os.sep + 'config', root + os.sep + 'state'
            credentials = config + os.sep + 'credentials'
            definition = windows_state.Definition(workspace, [
                windows_state.Grant(workspace, windows_state.Access.READ_WRITE, 'workspace'),
                windows_state.Grant(config, windows_state.Access.READ, 'config')])
            with mock.patch.object(windows_state.paths, "loki_config_dir", return_value=config), \
                    mock.patch.object(windows_state.paths, "loki_state_dir", return_value=state), \
                    mock.patch.object(windows_state.paths, "credential_directory",
                                      return_value=credentials):
                plan = windows_state.build_plan(definition, None)
            self.assertTrue(plan.applicable, plan.errors)
            self.assertEqual({change.path for change in plan.changes}, {workspace, config})
            self.assertFalse(os.path.exists(config))
            self.assertFalse(os.path.exists(state))

    def test_protected_paths_are_errors_and_block_apply(self):
        with tempfile.TemporaryDirectory() as root, real_directory_locations():
            credentials = root + os.sep + 'credentials'
            os.mkdir(credentials)
            selected = windows_state._directory_location(credentials)
            roots = {'configuration': None, 'state': None, 'credentials': selected}
            with mock.patch.object(windows_state, '_root_locations', return_value=roots):
                definition = windows_state.Definition(root, [self.grant(root)])
                plan = windows_setup.build_plan(definition, None)
        self.assertFalse(plan.applicable)
        self.assertTrue(plan.errors)

    def test_warnings_are_collected_for_user_grants_only(self):
        with mock.patch.object(
                windows_state, "covered_secret_warnings",
                side_effect=lambda path: [f"covers {path}"]) as warning:
            plan = windows_setup.build_plan(
                self.definition(self.grant("/user-dir"),
                                self.grant("/auto", origin="workspace")),
                None)

        self.assertEqual(plan.warnings, ["covers /user-dir"])
        warning.assert_called_once_with("/user-dir")


class ManagedStatePolicyTests(unittest.TestCase):
    def setUp(self):
        aliases = {'SY': 'S-1-5-18', 'BA': 'S-1-5-32-544'}
        patch = mock.patch.object(windows_api, 'canonical_sid',
                                  side_effect=lambda sid: aliases.get(sid, sid))
        patch.start()
        self.addCleanup(patch.stop)

    def errors(self, ace):
        return windows_state.state_acl_errors('D:P' + ace, OTHER_SID, {PACKAGE_SID})

    def test_runtime_modify_is_allowed_but_setup_authority_is_not(self):
        self.assertEqual(self.errors(f'(A;OICI;0x1301BF;;;{PACKAGE_SID})'), [])
        for rights in ['FA', 'GA', '0x10000000', 'WD', 'WO', '0x40']:
            with self.subTest(rights=rights):
                self.assertTrue(self.errors(f'(A;OICI;{rights};;;{PACKAGE_SID})'))

    def test_no_write_restriction_alternative_may_mutate_state(self):
        for sid in [windows_state.tool_role_sid(PACKAGE_SID),
                    *windows_state.TOOL_COMPATIBILITY_SIDS, 'S-1-5-5-0-99']:
            for rights in ['FW', 'GA', '0x40000000', '0x10000000', '0x2', 'SD', 'WD', 'WO']:
                for flags in ['', 'OICIIO', 'ID']:
                    with self.subTest(sid=sid, rights=rights, flags=flags):
                        self.assertTrue(self.errors(f'(A;{flags};{rights};;;{sid})'))
            self.assertEqual(self.errors(f'(A;OICI;FRFX;;;{sid})'), [])

    def test_inventory_excludes_only_root_setup_publication_files(self):
        with tempfile.TemporaryDirectory() as root:
            job = os.path.join(root, 'job')
            os.mkdir(job)
            for path in [os.path.join(root, 'windows-setup.json'),
                         os.path.join(root, '.setup-pending'),
                         os.path.join(job, 'windows-setup.json'),
                         os.path.join(job, 'stdout.log')]:
                with open(path, 'w'):
                    pass
            self.assertEqual(set(windows_state.state_objects(root)),
                             {root, job, os.path.join(job, 'windows-setup.json'),
                              os.path.join(job, 'stdout.log')})

    def test_inventory_refuses_a_setup_directory(self):
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, 'windows-setup.json'))
            with self.assertRaises(windows_api.WindowsApiError):
                list(windows_state.state_objects(root))

    def test_revocation_removes_the_tool_grant_even_if_the_package_grant_is_already_gone(self):
        role = windows_state.tool_role_sid(PACKAGE_SID)
        backend = windows_setup.WindowsBackend({})
        with mock.patch.object(windows_api, 'dacl_sddl', return_value=f'D:P(A;OICI;0x1301BF;;;{role})'), \
                mock.patch.object(windows_containers, 'set_dacl_sddl') as write:
            backend._ungrant_path('/work', PACKAGE_SID)
        write.assert_called_once_with('/work', 'D:P')

    def test_writable_workspace_grants_both_package_and_tool_role(self):
        backend = windows_setup.WindowsBackend({})
        with mock.patch.object(windows_api, 'handle_dacl_sddl', return_value='D:P'), \
                mock.patch.object(windows_containers, 'set_handle_dacl_sddl') as write:
            backend._grant_path(123, '/work', windows_state.Access.READ_WRITE, PACKAGE_SID)
        sddl = write.call_args.args[1]
        self.assertTrue(windows_state.grants_access(sddl, PACKAGE_SID, windows_state.Access.READ_WRITE))
        self.assertTrue(windows_state.grants_access(
            sddl, windows_state.tool_role_sid(PACKAGE_SID), windows_state.Access.READ_WRITE))


class AutomaticGrantTests(unittest.TestCase):
    def test_automatic_grants_include_read_only_configuration(self):
        grants = windows_setup.automatic_grants("/work")

        by_origin = {}
        for grant in grants:
            by_origin.setdefault(grant.origin, []).append(grant)
        self.assertEqual(sorted(by_origin), ["config", "runtime", "state", "workspace"])
        self.assertEqual(by_origin["config"], [windows_setup.Grant(
            windows_setup.paths.loki_config_dir(), windows_setup.Access.READ, "config")])
        self.assertEqual(by_origin["workspace"][0].access,
                         windows_setup.Access.READ_WRITE)
        for grant in by_origin["runtime"]:
            self.assertEqual(grant.access, windows_setup.Access.READ)
        # No scratch grant: TEMP stays inside the writable workspace, not beside
        # credentials under configuration. Configuration is shared read-only;
        # the credential directory's protected DACL must block inheritance from
        # that readable parent. A writable scratch grant there would introduce
        # mutation rights into a tree whose integrity the runtime must preserve.
        self.assertEqual(windows_state.protected_path_errors("/work"), [])

    def test_definition_is_automatic_plus_recorded_user_grants(self):
        previous = windows_setup.ledger_entry(
            "/work", "x", [windows_setup.Grant("/extra",
                                               windows_setup.Access.READ,
                                               "user")])

        definition = windows_setup.definition_for("/work", previous)

        origins = [grant.origin for grant in definition.grants]
        self.assertEqual(origins.count("user"), 1)
        self.assertIn("workspace", origins)


class PlatformGateTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "needs a non-Windows host")
    def test_posix_refuses_the_setup_entrypoint_entirely(self):
        # Even --help: the flag vocabulary must not exist on POSIX, so an
        # unknown option fails instead of quietly doing nothing.
        for argument in [["--help"], ["--edit", "/work"], ["--verify", "/w"]]:
            with self.subTest(argument=argument):
                self.assertEqual(windows_setup.main(argument), 2)

    def test_unavailable_backend_refuses_every_operation(self):
        backend = windows_setup.UnavailableBackend()
        for call in (lambda: backend.apply(None, None),
                     lambda: backend.verify(None),
                     lambda: backend.uninstall({})):
            with self.subTest(call=call), self.assertRaises(
                    windows_api.WindowsUnavailableError):
                call()

    def test_uninstall_message_names_the_whole_harness(self):
        message = windows_setup.UNINSTALL_MESSAGE
        self.assertIn("entire Loki coding agent harness", message)
        self.assertIn("Your files are not deleted", message)


class FakeBackend:
    """Records what the editor asked for and returns minimal checks."""

    def __init__(self):
        self.applied = []
        self.verified = []
        self.uninstalled = []

    def apply(self, plan, definition):
        self.applied.append((plan, definition))
        return [windows_setup.Check("profile", "pass", "ok")]

    def verify(self, definition):
        self.verified.append(definition)
        return [windows_setup.Check("contained probe", "untested", "probe")]

    def uninstall(self, blob):
        self.uninstalled.append(blob)
        blob["workspaces"] = {}
        return [windows_setup.Check("ledger", "pass", "cleared")]


class NativeCleanupClassificationTests(unittest.TestCase):
    def test_optional_root_cannot_disappear_after_cleanup_failure(self):
        failure = windows_api.WindowsResourceError('CloseHandle failed', status=windows_api.ERROR_FILE_NOT_FOUND)
        with mock.patch.object(windows_state, '_directory_location', side_effect=failure):
            with self.assertRaises(windows_api.WindowsResourceError) as caught:
                windows_state._optional_location('configured-operand')
        self.assertIs(caught.exception, failure)

    def test_automatic_preflight_cannot_accept_cleanup_failure_as_missing(self):
        failure = windows_api.WindowsResourceError('CloseHandle failed', status=windows_api.ERROR_PATH_NOT_FOUND)
        with mock.patch.object(windows_state.paths, 'loki_config_dir', return_value='configured-operand'), \
                mock.patch.object(windows_state, '_directory_location', side_effect=failure):
            errors = windows_state.protected_path_errors(
                'configured-operand', windows_state.Access.READ, origin='config')
        self.assertTrue(errors)
        self.assertIn('CloseHandle failed', errors[0])

    def test_cleanup_failure_cannot_create_a_pending_default_identity(self):
        failure = windows_api.WindowsResourceError('CloseHandle failed', status=windows_api.ERROR_PATH_NOT_FOUND)
        grant = windows_state.Grant('configured-operand', windows_state.Access.READ, 'config')
        with mock.patch.object(windows_state.paths, 'loki_config_dir', return_value=grant.path), \
                mock.patch.object(windows_state, '_directory_identity', side_effect=failure):
            with self.assertRaises(windows_api.WindowsResourceError) as caught:
                windows_state._grant_identity(grant)
        self.assertIs(caught.exception, failure)

    def test_profile_cleanup_failure_cannot_trigger_an_already_exists_fallback(self):
        failure = windows_api.WindowsResourceError(
            'LocalFree failed', status=windows_containers.PROFILE_ALREADY_EXISTS)
        with mock.patch.object(windows_containers, 'create_app_container_profile', side_effect=failure), \
                mock.patch.object(windows_api, 'derive_app_container_sid') as derive:
            with self.assertRaises(windows_api.WindowsResourceError) as caught:
                windows_setup.WindowsBackend({})._ensure_profile('profile')
        self.assertIs(caught.exception, failure)
        derive.assert_not_called()


class ConfigureCommandTests(unittest.TestCase):
    """--configure applies the editor's plan without importing Tk.

    It is the non-interactive setup path: a CI job or headless caller runs it,
    and the entrypoint's gate then finds a ledger entry that verify accepts.
    """

    def configure(self, arguments, backend=None):
        backend = FakeBackend() if backend is None else backend
        # The plan's protected-path rule resolves Loki's real directories, and
        # those are Windows known folders under the patched platform; pin them
        # so this stays a host-independent test of the command, not of paths.
        with mock.patch.object(windows_setup.sys, "platform", "win32"), \
                mock.patch.object(windows_setup, "load_ledger",
                                  return_value={"workspaces": {}}), \
                mock.patch.object(windows_setup, "backend_for",
                                  return_value=backend), \
                mock.patch.object(windows_state.paths, "loki_config_dir",
                                  return_value="/protected/config"), \
                mock.patch.object(windows_state.paths, "loki_state_dir",
                                  return_value="/protected/state"), \
                mock.patch.object(windows_state.paths, "credential_directory",
                                  return_value="/protected/credentials"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = windows_setup.main(["--configure", *arguments])
        return code, output.getvalue(), backend

    def test_configure_applies_the_same_automatic_grants_as_the_editor(self):
        def programs(*, automatic_read=False):
            return ['/runtime-a', '/runtime-b'] + ([] if automatic_read else ['/editor'])

        with mock.patch.object(windows_state, 'program_directories', side_effect=programs):
            code, output, backend = self.configure(["/work"])

        self.assertEqual(code, 0)
        self.assertIn("pass: profile", output)
        self.assertEqual(len(backend.applied), 1)
        _plan, definition = backend.applied[0]
        self.assertEqual(definition.workspace, "/work")
        self.assertEqual([grant.origin for grant in definition.grants],
                         ["workspace", "config", "state", "runtime", "runtime"])
        self.assertEqual([grant.path for grant in definition.grants if grant.origin == 'runtime'],
                         ['/runtime-a', '/runtime-b'])

    def test_a_failing_check_is_a_nonzero_exit(self):
        class Failing(FakeBackend):
            def apply(self, plan, definition):
                return [windows_setup.Check("profile", "fail", "denied")]

        code, output, _backend = self.configure(["/work"], Failing())

        self.assertEqual(code, 1)
        self.assertIn("fail: profile denied", output)

    def test_a_missing_workspace_is_a_usage_error(self):
        with mock.patch.object(windows_setup.sys, "platform", "win32"), \
                mock.patch.object(windows_setup, "load_ledger",
                                  return_value={"workspaces": {}}), \
                contextlib.redirect_stderr(io.StringIO()) as error:
            code = windows_setup.main(["--configure"])

        self.assertEqual(code, 2)
        self.assertIn("needs a workspace", error.getvalue())


class DescribePlanTests(unittest.TestCase):
    def plan(self, *changes, warnings=(), errors=()):
        return windows_setup.Plan("/p", "profile", list(changes),
                                  list(warnings), list(errors))

    def change(self, kind, path, access=windows_setup.Access.READ):
        return windows_state.Change(kind, path, access)

    def test_describes_each_change_with_its_level(self):
        text = windows_setup.describe_plan(self.plan(
            self.change("add", "/new", windows_setup.Access.READ_WRITE),
            self.change("remove", "/gone"),
            self.change("level", "/changed")))

        self.assertIn("will grant /new (read and write)", text)
        self.assertIn("will stop granting /gone (read)", text)
        self.assertIn("will change /changed (read)", text)

    def test_says_so_when_nothing_changes(self):
        self.assertIn("no changes to grants",
                      windows_setup.describe_plan(self.plan()))

    def test_lists_warnings_after_the_changes(self):
        text = windows_setup.describe_plan(
            self.plan(self.change("add", "/x"), warnings=["covers /x/.ssh"]))

        self.assertIn("Warnings:", text)
        self.assertIn("! covers /x/.ssh", text)
        self.assertLess(text.index("will grant"), text.index("Warnings:"))


class EditorModelTests(unittest.TestCase):
    def model(self, ledger=None, backend=None, workspace="/work"):
        return windows_setup.EditorModel(
            {"workspaces": {}} if ledger is None else ledger,
            workspace, backend or FakeBackend())

    def user_grant(self, path, access=windows_setup.Access.READ):
        return windows_setup.Grant(path, access, "user")

    def user_ledger(self, path="/extra", access=windows_setup.Access.READ):
        entry = windows_setup.ledger_entry(
            "/work", "p", [self.user_grant(path, access)])
        return {"workspaces": {windows_setup.workspace_key("/work"): entry}}

    def test_rows_show_level_path_and_origin(self):
        model = self.model(self.user_ledger())

        rows = model.rows()

        self.assertIn("read: /extra", rows)
        self.assertTrue(any("[workspace]" in row for row in rows))
        self.assertTrue(any("[runtime]" in row for row in rows))

    def test_user_rows_are_editable_and_automatic_rows_are_not(self):
        model = self.model(self.user_ledger())

        automatic = [i for i, grant in enumerate(model.grants)
                     if grant.origin != "user"]
        user = [i for i, grant in enumerate(model.grants)
                if grant.origin == "user"]
        self.assertTrue(automatic)
        for index in automatic:
            self.assertIn("shared automatically", model.edit_refusal(index))
        self.assertEqual(len(user), 1)
        self.assertIsNone(model.edit_refusal(user[0]))

    def test_adding_a_grant_makes_the_model_dirty_and_reload_discards_it(self):
        model = self.model()
        self.assertFalse(model.dirty)

        model.add("/new", windows_setup.Access.READ)

        self.assertTrue(model.dirty)
        self.assertIn("read: /new", model.rows())

        model.reload(model.workspace)

        self.assertFalse(model.dirty)
        self.assertNotIn("read: /new", model.rows())

    def test_level_change_and_removal_edit_the_selected_row(self):
        model = self.model(self.user_ledger())
        index = [i for i, grant in enumerate(model.grants)
                 if grant.origin == "user"][0]

        model.set_level(index, windows_setup.Access.READ_WRITE)
        self.assertIn("read and write: /extra", model.rows())

        model.remove(index)
        self.assertNotIn("read and write: /extra", model.rows())
        self.assertTrue(model.dirty)

    def test_apply_and_verify_delegate_to_the_backend(self):
        backend = FakeBackend()
        model = self.model(backend=backend)
        model.add("/new", windows_setup.Access.READ)

        checks = model.apply()
        model.verify()

        self.assertEqual(checks[0].status, "pass")
        self.assertEqual(len(backend.applied), 1)
        plan, definition = backend.applied[0]
        self.assertIs(definition, model.definition)
        self.assertEqual(plan.profile,
                         windows_state.profile_name_for("/work"))
        self.assertEqual(backend.verified, [model.definition])

    def test_uninstall_delegates_and_reloads(self):
        ledger = self.user_ledger()
        backend = FakeBackend()
        model = self.model(ledger, backend)

        model.uninstall()

        self.assertEqual(backend.uninstalled, [ledger])
        self.assertFalse(model.dirty)

    def test_describe_is_the_plan_text(self):
        model = self.model()
        model.add("/new", windows_setup.Access.READ)

        self.assertIn("will grant /new (read)", model.describe())


@unittest.skipUnless(sys.platform == "win32",
                     "Tk widgets need Windows (there is no display server)")
class EditorWidgetTests(unittest.TestCase):
    """Widget smoke tests: does each button do what it claims?

    Only the wiring is covered -- the rules live in EditorModelTests, and the
    appearance stays a visual check.
    """

    def setUp(self):
        tk, _, _ = windows_setup._tk()
        self.tk = tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.filedialog = mock.Mock()
        self.messagebox = mock.Mock()
        self.messagebox.askokcancel.return_value = True
        self.messagebox.askyesnocancel.return_value = True
        self.backend = FakeBackend()

    def editor(self, ledger=None, workspace="/work"):
        model = windows_setup.EditorModel(
            {"workspaces": {}} if ledger is None else ledger,
            workspace, self.backend)
        return windows_setup.Editor(self.root, model, self.tk,
                                    self.filedialog, self.messagebox)

    def user_ledger(self, path="/extra"):
        entry = windows_setup.ledger_entry(
            "/work", "p", [windows_setup.Grant(path,
                                               windows_setup.Access.READ,
                                               "user")])
        return {"workspaces": {windows_setup.workspace_key("/work"): entry}}

    def rows(self, editor):
        return list(editor.listing.get(0, "end"))

    def select(self, editor, index):
        editor.listing.selection_clear(0, "end")
        editor.listing.selection_set(index)
        self.root.update_idletasks()

    def test_rows_render_each_grant_with_its_level(self):
        self.assertIn("read: /extra", self.rows(self.editor(
            self.user_ledger())))

    def test_add_appends_the_chosen_path_at_the_chosen_level(self):
        self.filedialog.askdirectory.return_value = "/chosen"
        self.messagebox.askyesnocancel.return_value = False  # read only
        editor = self.editor()

        editor.add()

        self.assertIn("read: /chosen", self.rows(editor))
        self.assertTrue(editor.model.dirty)

    def test_modify_refuses_an_automatic_grant_with_a_reason(self):
        editor = self.editor()
        automatic = [i for i, grant in enumerate(editor.model.grants)
                     if grant.origin != "user"][0]
        self.select(editor, automatic)
        before = editor.model.grants[automatic].access

        editor.modify()

        self.messagebox.showinfo.assert_called_once()
        self.assertIn("shared automatically",
                      self.messagebox.showinfo.call_args.args[1])
        self.assertIs(editor.model.grants[automatic].access, before)

    def test_delete_removes_a_user_grant_after_confirmation(self):
        editor = self.editor(self.user_ledger())
        index = [i for i, grant in enumerate(editor.model.grants)
                 if grant.origin == "user"][0]
        self.select(editor, index)

        editor.delete()

        self.assertNotIn("read: /extra", self.rows(editor))

    def test_apply_hands_the_definition_to_the_backend(self):
        editor = self.editor(self.user_ledger())

        editor.apply()

        self.assertEqual(editor.status, 0)
        self.assertEqual(len(self.backend.applied), 1)

    def test_apply_refuses_a_protected_path_without_calling_the_backend(self):
        with mock.patch.object(windows_state, "_runtime_trees",
                               return_value=["/protected"]):
            editor = self.editor()
            editor.model.add("/protected", windows_setup.Access.READ)

            editor.apply()

        self.messagebox.showerror.assert_called_once()
        self.assertEqual(self.backend.applied, [])
        self.assertEqual(editor.status, 1)

    def test_switching_workspace_discards_changes_only_after_asking(self):
        self.filedialog.askdirectory.return_value = "/other"
        self.messagebox.askyesnocancel.return_value = False  # discard
        editor = self.editor()
        editor.model.add("/new", windows_setup.Access.READ)

        editor.browse()

        self.messagebox.askyesnocancel.assert_called_once()
        self.assertEqual(editor.model.workspace, "/other")
        self.assertFalse(editor.model.dirty)

    def test_switching_workspace_stays_put_when_the_user_cancels(self):
        self.filedialog.askdirectory.return_value = "/other"
        self.messagebox.askyesnocancel.return_value = None  # cancel
        editor = self.editor()
        editor.model.add("/new", windows_setup.Access.READ)

        editor.browse()

        self.assertEqual(editor.model.workspace, "/work")
        self.assertEqual(self.filedialog.askdirectory.call_count, 0)


if __name__ == "__main__":
    unittest.main()
