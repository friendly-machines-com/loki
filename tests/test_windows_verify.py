"""Portable tests for the containment probe's decision logic.

The CreateFileW and token calls need Windows; what is checked here is the
mapping from their results to checks.  The cases that matter are the ones that
could make a broken container look fine: a granted denial must fail, an
unexpected error must not pass, and an unreachable workspace must not make the
denials look successful by making them vacuous.
"""

import os
import unittest
from unittest import mock

from loki_agent import paths
from loki_agent import windows_api
from loki_agent import windows_state
from loki_agent import windows_verify


PACKAGE = "S-1-15-2-1-2-3-4"
OTHER = "S-1-5-21-1-2-3-1004"
WORKSPACE = "/workspace"
GRANTED = "/granted"
CREDENTIALS = "/credentials"
CREDENTIAL_FILE = os.path.join(CREDENTIALS, paths.CREDENTIAL_FILE_NAME)
CREDENTIAL_LOCK = os.path.join(CREDENTIALS, paths.CREDENTIAL_LOCK_FILE_NAME)
CONFIG = "/config"
STATE = "/state"
WORKSPACE_RW = windows_api.GENERIC_READ | windows_api.GENERIC_WRITE

# A protected-tree DACL that names no package SID.
NO_PACKAGE = f"D:PAI(A;OICI;FA;;;{OTHER})"

_DENIED = object()


class ProbeContainmentTests(unittest.TestCase):
    def setUp(self):
        self.attempts = []
        self.outcomes = {}
        self.closed = []
        patches = [
            mock.patch.object(windows_api, "current_process_handle",
                              return_value=1),
            mock.patch.object(windows_api, "open_process_token",
                              return_value=2),
            mock.patch.object(windows_api, "close_handle",
                              side_effect=self.closed.append),
            mock.patch.object(windows_api, "token_is_app_container",
                              return_value=True),
            mock.patch.object(windows_api, "token_app_container_sid",
                              return_value=PACKAGE),
            mock.patch.object(windows_api, "derive_app_container_sid",
                              return_value=PACKAGE),
            mock.patch.object(windows_verify, "profile_name_for",
                              return_value="profile"),
            mock.patch.object(paths, "credential_directory",
                              return_value=CREDENTIALS),
            mock.patch.object(windows_api, "open_with_access",
                              side_effect=self._open),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _open(self, path, desired_access):
        self.attempts.append((path, desired_access))
        outcome = self.outcomes.get((path, desired_access), _DENIED)
        if outcome is _DENIED:
            raise windows_api.WindowsApiError(
                "denied", status=windows_api.ERROR_ACCESS_DENIED)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def allow(self, path, access):
        self.outcomes[(path, access)] = 99

    def deny(self, path, access, status=windows_api.ERROR_ACCESS_DENIED):
        self.outcomes[(path, access)] = windows_api.WindowsApiError(
            "denied", status=status)

    def happy(self):
        # Everything not named here defaults to ACCESS_DENIED, including the
        # credential file's read, which is also how the probe learns the file
        # exists rather than being absent.
        self.allow(WORKSPACE, WORKSPACE_RW)

    def checks(self):
        return {check.name: check
                for check in windows_verify.probe_containment(WORKSPACE)}

    def test_happy_path_reports_every_check_as_passing(self):
        self.happy()

        checks = self.checks()

        self.assertEqual(set(checks), {
            "AppContainer", "package SID", "workspace reachable",
            "credentials unlistable", "cannot create credentials",
            "credential directory not deletable",
            "credential directory DACL not rewritable",
            "credential directory owner not rewritable",
            "credential file unreadable", "credential file not writable",
            "credential file not appendable", "credential file not deletable",
            "credential file DACL not rewritable",
            "credential file owner not rewritable",
            "credential lock unreadable", "credential lock not writable",
            "credential lock not appendable", "credential lock not deletable",
            "credential lock DACL not rewritable",
            "credential lock owner not rewritable", "cannot rewrite a DACL"})
        for name, check in checks.items():
            self.assertEqual(check.status, "pass", (name, check))

    def test_the_attempts_are_the_expected_paths_and_rights(self):
        self.happy()

        self.checks()

        self.assertEqual(self.attempts, [
            (WORKSPACE, WORKSPACE_RW),
            (CREDENTIALS, windows_api.GENERIC_READ),
            (CREDENTIALS, windows_api.FILE_WRITE_DATA),
            (CREDENTIALS, windows_api.DELETE),
            (CREDENTIALS, windows_api.WRITE_DAC),
            (CREDENTIALS, windows_api.WRITE_OWNER),
            (CREDENTIAL_FILE, windows_api.GENERIC_READ),
            (CREDENTIAL_FILE, windows_api.GENERIC_WRITE),
            (CREDENTIAL_FILE, windows_api.FILE_APPEND_DATA),
            (CREDENTIAL_FILE, windows_api.DELETE),
            (CREDENTIAL_FILE, windows_api.WRITE_DAC),
            (CREDENTIAL_FILE, windows_api.WRITE_OWNER),
            (CREDENTIAL_LOCK, windows_api.GENERIC_READ),
            (CREDENTIAL_LOCK, windows_api.GENERIC_WRITE),
            (CREDENTIAL_LOCK, windows_api.FILE_APPEND_DATA),
            (CREDENTIAL_LOCK, windows_api.DELETE),
            (CREDENTIAL_LOCK, windows_api.WRITE_DAC),
            (CREDENTIAL_LOCK, windows_api.WRITE_OWNER),
            (WORKSPACE, windows_api.WRITE_DAC)])

    def test_an_opened_handle_and_the_token_are_closed(self):
        self.happy()

        self.checks()

        self.assertIn(99, self.closed)
        self.assertIn(2, self.closed)

    def test_a_normal_token_fails_the_identity_check(self):
        self.happy()

        with mock.patch.object(windows_api, "token_is_app_container",
                               return_value=False):
            checks = self.checks()

        self.assertEqual(checks["AppContainer"].status, "fail")

    def test_a_different_package_sid_fails_the_identity_check(self):
        self.happy()

        with mock.patch.object(windows_api, "token_app_container_sid",
                               return_value="S-1-15-2-9-9"):
            checks = self.checks()

        self.assertEqual(checks["package SID"].status, "fail")

    def test_a_listable_credential_directory_fails(self):
        self.happy()
        self.allow(CREDENTIALS, windows_api.GENERIC_READ)

        self.assertEqual(
            self.checks()["credentials unlistable"].status, "fail")

    def test_a_creatable_credential_directory_fails(self):
        self.happy()
        self.allow(CREDENTIALS, windows_api.FILE_WRITE_DATA)

        self.assertEqual(
            self.checks()["cannot create credentials"].status, "fail")

    def test_a_deletable_credential_directory_fails(self):
        self.happy()
        self.allow(CREDENTIALS, windows_api.DELETE)

        self.assertEqual(
            self.checks()["credential directory not deletable"].status,
            "fail")

    def test_a_rewritable_credential_directory_dacl_fails(self):
        self.happy()
        self.allow(CREDENTIALS, windows_api.WRITE_DAC)

        self.assertEqual(
            self.checks()["credential directory DACL not rewritable"].status,
            "fail")

    def test_a_rewritable_credential_directory_owner_fails(self):
        self.happy()
        self.allow(CREDENTIALS, windows_api.WRITE_OWNER)

        self.assertEqual(
            self.checks()["credential directory owner not rewritable"].status,
            "fail")

    def test_a_granted_credential_read_fails(self):
        self.happy()
        self.allow(CREDENTIAL_FILE, windows_api.GENERIC_READ)

        self.assertEqual(
            self.checks()["credential file unreadable"].status, "fail")

    def test_a_granted_credential_write_fails(self):
        self.happy()
        self.allow(CREDENTIAL_FILE, windows_api.GENERIC_WRITE)

        self.assertEqual(
            self.checks()["credential file not writable"].status, "fail")

    def test_a_granted_credential_append_fails(self):
        self.happy()
        self.allow(CREDENTIAL_FILE, windows_api.FILE_APPEND_DATA)

        self.assertEqual(
            self.checks()["credential file not appendable"].status, "fail")

    def test_a_granted_credential_delete_fails(self):
        self.happy()
        self.allow(CREDENTIAL_FILE, windows_api.DELETE)

        self.assertEqual(
            self.checks()["credential file not deletable"].status, "fail")

    def test_a_granted_credential_dacl_write_fails(self):
        self.happy()
        self.allow(CREDENTIAL_FILE, windows_api.WRITE_DAC)

        self.assertEqual(
            self.checks()["credential file DACL not rewritable"].status,
            "fail")

    def test_a_granted_credential_owner_write_fails(self):
        self.happy()
        self.allow(CREDENTIAL_FILE, windows_api.WRITE_OWNER)

        self.assertEqual(
            self.checks()["credential file owner not rewritable"].status,
            "fail")

    def test_a_readable_credential_lock_fails(self):
        self.happy()
        self.allow(CREDENTIAL_LOCK, windows_api.GENERIC_READ)

        self.assertEqual(
            self.checks()["credential lock unreadable"].status, "fail")

    def test_a_granted_credential_lock_write_fails(self):
        self.happy()
        self.allow(CREDENTIAL_LOCK, windows_api.GENERIC_WRITE)

        self.assertEqual(
            self.checks()["credential lock not writable"].status, "fail")

    def test_a_deletable_credential_lock_fails(self):
        self.happy()
        self.allow(CREDENTIAL_LOCK, windows_api.DELETE)

        self.assertEqual(
            self.checks()["credential lock not deletable"].status, "fail")

    def test_an_absent_credential_lock_skips_the_direction_probes(self):
        self.happy()
        self.deny(CREDENTIAL_LOCK, windows_api.GENERIC_READ,
                  status=windows_api.ERROR_FILE_NOT_FOUND)

        checks = self.checks()

        self.assertEqual(checks["credential lock"].status, "pass")
        self.assertNotIn("credential lock unreadable", checks)
        self.assertEqual(
            [attempt for attempt in self.attempts
             if attempt[0] == CREDENTIAL_LOCK],
            [(CREDENTIAL_LOCK, windows_api.GENERIC_READ)])

    def test_an_absent_credential_file_skips_the_direction_probes(self):
        self.happy()
        self.deny(CREDENTIAL_FILE, windows_api.GENERIC_READ,
                  status=windows_api.ERROR_FILE_NOT_FOUND)

        checks = self.checks()

        self.assertEqual(checks["credential file"].status, "pass")
        self.assertNotIn("credential file unreadable", checks)
        self.assertNotIn("credential file not writable", checks)
        # With nothing to read, the create denial is what carries the claim.
        self.assertEqual(checks["cannot create credentials"].status, "pass")
        self.assertEqual(
            [attempt for attempt in self.attempts
             if attempt[0] == CREDENTIAL_FILE],
            [(CREDENTIAL_FILE, windows_api.GENERIC_READ)])

    def test_an_unexpected_credential_file_error_is_not_a_pass(self):
        self.happy()
        self.deny(CREDENTIAL_FILE, windows_api.GENERIC_READ, status=87)

        self.assertEqual(
            self.checks()["credential file unreadable"].status, "fail")

    def test_a_granted_write_dac_fails(self):
        self.happy()
        self.allow(WORKSPACE, windows_api.WRITE_DAC)

        self.assertEqual(
            self.checks()["cannot rewrite a DACL"].status, "fail")

    def test_an_unreachable_workspace_fails_the_positive_control(self):
        self.happy()
        self.deny(WORKSPACE, WORKSPACE_RW)

        checks = self.checks()

        self.assertEqual(checks["workspace reachable"].status, "fail")
        # The denials are still reported on their own merits.
        self.assertEqual(checks["credentials unlistable"].status, "pass")

    def test_an_unexpected_error_is_not_reported_as_a_pass(self):
        self.happy()
        self.deny(CREDENTIALS, windows_api.GENERIC_READ, status=87)

        self.assertEqual(
            self.checks()["credentials unlistable"].status, "fail")

    def test_an_unopenable_token_fails(self):
        self.happy()

        with mock.patch.object(windows_api, "open_process_token",
                               side_effect=windows_api.WindowsApiError("no")):
            checks = self.checks()

        self.assertEqual(checks["process token"].status, "fail")


class VerifyContainerTests(unittest.TestCase):
    """The uncontained inventory must read rights, not SID strings.

    ``verify_container`` is the start-up gate: a false pass here accepts a
    container the runtime then treats as contained.  Each descriptor below is
    legal SDDL that the old substring check reported as a pass.
    """

    READ = windows_state.Access.READ
    READ_WRITE = windows_state.Access.READ_WRITE

    def setUp(self):
        self.descriptors = {}
        patches = [
            mock.patch.object(windows_verify, "profile_name_for",
                              return_value="profile"),
            mock.patch.object(windows_api, "derive_app_container_sid",
                              return_value=PACKAGE),
            mock.patch.object(windows_api, "dacl_sddl",
                              side_effect=lambda path: self.descriptors.get(
                                  path, NO_PACKAGE)),
            mock.patch.object(paths, "credential_directory",
                              return_value=CREDENTIALS),
            mock.patch.object(paths, "loki_config_dir", return_value=CONFIG),
            mock.patch.object(paths, "loki_state_dir", return_value=STATE),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def check(self, path, access):
        checks = windows_verify.verify_container(
            WORKSPACE, [windows_state.Grant(path, access, "user")])
        return {check.name: check for check in checks}

    def test_read_write_grant_and_clean_trees_pass(self):
        self.descriptors[GRANTED] = (
            f"D:PAI(A;OICI;0x1301BF;;;{PACKAGE})")

        checks = self.check(GRANTED, self.READ_WRITE)

        for name, check in checks.items():
            self.assertEqual(check.status, "pass", (name, check))

    def test_a_deny_ace_is_not_a_grant(self):
        self.descriptors[GRANTED] = f"D:PAI(D;OICI;FA;;;{PACKAGE})"

        checks = self.check(GRANTED, self.READ)

        self.assertEqual(checks[f"grant {GRANTED}"].status, "fail")

    def test_a_rights_mask_smaller_than_the_level_fails(self):
        self.descriptors[GRANTED] = (
            f"D:PAI(A;OICI;0x00000100;;;{PACKAGE})")

        checks = self.check(GRANTED, self.READ)

        self.assertEqual(checks[f"grant {GRANTED}"].status, "fail")

    def test_an_inherit_only_allow_fails(self):
        self.descriptors[GRANTED] = f"D:PAI(A;OICIIO;FRFX;;;{PACKAGE})"

        checks = self.check(GRANTED, self.READ)

        self.assertEqual(checks[f"grant {GRANTED}"].status, "fail")

    def test_read_only_does_not_satisfy_a_read_write_grant(self):
        self.descriptors[GRANTED] = f"D:PAI(A;OICI;FRFX;;;{PACKAGE})"

        checks = self.check(GRANTED, self.READ_WRITE)

        self.assertEqual(checks[f"grant {GRANTED}"].status, "fail")

    def test_a_deny_ace_on_a_protected_tree_is_a_pass(self):
        # A deny is the safest possible entry; it must not be reported as a
        # package grant.
        self.descriptors[GRANTED] = f"D:PAI(A;OICI;0x1301BF;;;{PACKAGE})"
        self.descriptors[CREDENTIALS] = f"D:PAI(D;OICI;FA;;;{PACKAGE})"

        checks = self.check(GRANTED, self.READ_WRITE)

        self.assertEqual(checks["credentials"].status, "pass")

    def test_an_allow_ace_on_a_protected_tree_fails(self):
        self.descriptors[GRANTED] = f"D:PAI(A;OICI;FRFX;;;{PACKAGE})"
        self.descriptors[CONFIG] = f"D:PAI(A;OICI;FRFX;;;{PACKAGE})"

        checks = self.check(GRANTED, self.READ)

        self.assertEqual(checks["config"].status, "fail")

    def test_an_inherit_only_allow_on_a_protected_tree_fails(self):
        # It grants the tree's children, so "holds no access on the directory"
        # must not be read as containment.
        self.descriptors[GRANTED] = f"D:PAI(A;OICI;FRFX;;;{PACKAGE})"
        self.descriptors[STATE] = f"D:PAI(A;OICIIO;FA;;;{PACKAGE})"

        checks = self.check(GRANTED, self.READ)

        self.assertEqual(checks["state"].status, "fail")

    def test_an_unknown_rights_code_fails_closed(self):
        self.descriptors[GRANTED] = f"D:PAI(A;OICI;ZZ;;;{PACKAGE})"

        checks = self.check(GRANTED, self.READ)

        self.assertEqual(checks[f"grant {GRANTED}"].status, "fail")


if __name__ == "__main__":
    unittest.main()
