"""Portable tests for the containment probe's decision logic.

The CreateFileW and token calls need Windows; what is checked here is the
mapping from their results to checks.  The cases that matter are the ones that
could make a broken container look fine: a granted denial must fail, an
unexpected error must not pass, and an unreachable workspace must not make the
denials look successful by making them vacuous.
"""

import unittest
from unittest import mock

from loki_agent import paths
from loki_agent import windows_api
from loki_agent import windows_verify


PACKAGE = "S-1-15-2-1-2-3-4"
WORKSPACE = "/workspace"
CREDENTIALS = "/credentials"
WORKSPACE_RW = windows_api.GENERIC_READ | windows_api.GENERIC_WRITE

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
        self.allow(WORKSPACE, WORKSPACE_RW)
        self.deny(CREDENTIALS, windows_api.GENERIC_READ)
        self.deny(WORKSPACE, windows_api.WRITE_DAC)

    def checks(self):
        return {check.name: check
                for check in windows_verify.probe_containment(WORKSPACE)}

    def test_happy_path_reports_every_check_as_passing(self):
        self.happy()

        checks = self.checks()

        self.assertEqual(set(checks), {
            "AppContainer", "package SID", "workspace reachable",
            "credentials unreadable", "cannot rewrite a DACL"})
        for name, check in checks.items():
            self.assertEqual(check.status, "pass", (name, check))

    def test_the_attempts_are_the_expected_paths_and_rights(self):
        self.happy()

        self.checks()

        self.assertEqual(self.attempts, [
            (WORKSPACE, WORKSPACE_RW),
            (CREDENTIALS, windows_api.GENERIC_READ),
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

    def test_readable_credentials_fail(self):
        self.happy()
        self.allow(CREDENTIALS, windows_api.GENERIC_READ)

        self.assertEqual(
            self.checks()["credentials unreadable"].status, "fail")

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
        self.assertEqual(checks["credentials unreadable"].status, "pass")

    def test_an_unexpected_error_is_not_reported_as_a_pass(self):
        self.happy()
        self.deny(CREDENTIALS, windows_api.GENERIC_READ, status=87)

        self.assertEqual(
            self.checks()["credentials unreadable"].status, "fail")

    def test_an_unopenable_token_fails(self):
        self.happy()

        with mock.patch.object(windows_api, "open_process_token",
                               side_effect=windows_api.WindowsApiError("no")):
            checks = self.checks()

        self.assertEqual(checks["process token"].status, "fail")


if __name__ == "__main__":
    unittest.main()
