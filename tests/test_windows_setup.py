"""Portable tests for the Windows setup editor's logic and rules.

The Tk editor and the ACL/profile calls need Windows and are deliberately not
covered here.  Everything that *decides* something -- path rules, warnings,
naming, SDDL composition, the ledger, the plan diff, and the platform gate --
is pure, and is checked on every platform because a mistake there either grants
too much or refuses something legitimate.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import windows_acl
from loki_agent import windows_api
from loki_agent import windows_setup
from loki_agent import windows_state


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
        self.assertTrue(windows_state.protected_path_errors(
            os.path.join(self.config, "sessions")))

    def test_refuses_the_runtime_tree(self):
        package = os.path.dirname(os.path.abspath(windows_setup.__file__))
        self.assertTrue(windows_state.protected_path_errors(package))

    def test_allows_an_unrelated_directory(self):
        self.assertEqual(
            windows_state.protected_path_errors(
                os.path.join(self.root, "elsewhere")), [])


class SecretWarningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = temporary.name
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
    def test_canonical_form_collapses_spellings_of_one_directory(self):
        with tempfile.TemporaryDirectory() as root:
            child = os.path.join(root, "project")
            os.makedirs(child)
            link = os.path.join(root, "link")
            os.symlink(child, link)

            self.assertEqual(
                windows_state.canonical_workspace(child),
                windows_state.canonical_workspace(link))
            self.assertEqual(
                windows_state.canonical_workspace(child),
                windows_state.canonical_workspace(child + os.sep))
            self.assertEqual(
                windows_state.canonical_workspace(child),
                windows_state.canonical_workspace(
                    os.path.join(child, "..", "project")))

    def test_profile_name_is_deterministic_and_usable(self):
        first = windows_state.profile_name_for("/tmp/project")
        second = windows_state.profile_name_for("/tmp/project")

        self.assertEqual(first, second)
        self.assertNotEqual(first, windows_state.profile_name_for("/tmp/other"))
        self.assertTrue(
            windows_api.app_container_profile_name_is_usable(first))


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

    def test_protected_paths_are_errors_and_block_apply(self):
        with tempfile.TemporaryDirectory() as root:
            credentials = os.path.join(root, "credentials")
            with mock.patch.object(
                    windows_state, "_runtime_trees", return_value=[credentials]):
                plan = windows_setup.build_plan(
                    self.definition(self.grant(root)), None)

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


class AutomaticGrantTests(unittest.TestCase):
    def test_automatic_grants_are_the_workspace_and_toolchain(self):
        grants = windows_setup.automatic_grants("/work")

        by_origin = {grant.origin: grant for grant in grants}
        self.assertEqual(sorted(by_origin), ["toolchain", "workspace"])
        self.assertEqual(by_origin["workspace"].access,
                         windows_setup.Access.READ_WRITE)
        self.assertEqual(by_origin["toolchain"].access,
                         windows_setup.Access.READ)
        # No scratch grant: TEMP points inside the workspace, and a directory of
        # our own under the configuration tree would put a package ACE beside
        # the credential directory.
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
        for argument in (["--help"], ["--edit", "/work"], ["--verify", "/w"]):
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
        code, output, backend = self.configure(["/work"])

        self.assertEqual(code, 0)
        self.assertIn("pass: profile", output)
        self.assertEqual(len(backend.applied), 1)
        _plan, definition = backend.applied[0]
        self.assertEqual(definition.workspace, "/work")
        self.assertEqual([grant.origin for grant in definition.grants],
                         ["workspace", "toolchain"])

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


class ContainerWorkspaceTests(unittest.TestCase):
    """A configured workspace covers the directories beneath it."""

    @staticmethod
    def ledger(*workspaces):
        return {"workspaces": {
            windows_state.workspace_key(workspace): {"profile": "p", "grants": [{}]}
            for workspace in workspaces}}

    def test_the_nearest_configured_ancestor_covers_a_directory(self):
        blob = self.ledger("/work", "/work/tools")

        self.assertEqual(
            windows_state.container_workspace("/work/tools/project", blob),
            windows_state.workspace_key("/work/tools"))

    def test_a_shared_text_prefix_is_not_an_ancestor(self):
        self.assertIsNone(
            windows_state.container_workspace("/workshop", self.ledger("/work")))

    def test_an_unconfigured_directory_has_no_container(self):
        self.assertIsNone(
            windows_state.container_workspace("/elsewhere", self.ledger("/work")))


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
        self.assertTrue(any("[toolchain]" in row for row in rows))

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
