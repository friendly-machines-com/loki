import json
import os
import subprocess
import sys
import unittest
from unittest import mock

from loki_agent import process_protections


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakePrctl:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []
        self.restype = None

    def __call__(self, *args):
        self.calls.append(tuple(
            value.value if hasattr(value, "value") else value
            for value in args))
        return next(self.results)


class ProcessProtectionTests(unittest.TestCase):
    def test_non_linux_makes_no_security_claim(self):
        with mock.patch.object(
                process_protections.sys, "platform", "openbsd7"), \
                mock.patch.object(
                    process_protections.ctypes, "CDLL") as load:
            protected = (
                process_protections.protect_credential_process())

        self.assertFalse(protected)
        load.assert_not_called()

    def test_linux_set_verify_contract_and_fail_closed_errors(self):
        for results, error in (
                ([0, 0], None),
                ([-1], "PR_SET_DUMPABLE failed"),
                ([0, -1], "unexpected state -1"),
                ([0, 1], "unexpected state 1"),
                ([0, 2], "unexpected state 2")):
            with self.subTest(results=results):
                prctl = FakePrctl(results)
                libc = mock.Mock(prctl=prctl)
                with mock.patch.object(process_protections.sys,
                                       "platform", "linux"), \
                        mock.patch.object(process_protections.ctypes, "CDLL",
                                          return_value=libc), \
                        mock.patch.object(process_protections.ctypes,
                                          "get_errno", return_value=13):
                    if error is None:
                        self.assertTrue(
                            process_protections.protect_credential_process())
                    else:
                        with self.assertRaisesRegex(
                                process_protections.ProcessProtectionError,
                                error):
                            process_protections.protect_credential_process()
                expected = [(4, 0, 0, 0, 0)]  # SET_DUMPABLE must clear it.
                if len(results) == 2:
                    expected.append((3, 0, 0, 0, 0))  # GET_DUMPABLE
                self.assertEqual(prctl.calls, expected)

    def test_real_process_reports_protection(self):
        if os.name == "nt":
            # Windows has no prctl; the protection is the process object's
            # DACL, so a child applies it and reports its own process DACL.
            code = (
                "import json\n"
                "from loki_agent import process_protections, windows_api\n"
                "process_protections.protect_credential_process()\n"
                "print(json.dumps({'dacl': windows_api.handle_dacl_sddl(\n"
                "    windows_api.current_process_handle(),\n"
                "    windows_api.SE_KERNEL_OBJECT)}))\n"
            )
            process = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            dacl = json.loads(process.stdout)["dacl"]
            self.assertIsNotNone(dacl)
            # Only the owner, SYSTEM and the Administrators hold full control;
            # the inherited Everyone/Users grants are gone.
            self.assertIn("SY", dacl)
            self.assertIn("BA", dacl)
            self.assertNotIn("WD", dacl)
            self.assertNotIn("BU", dacl)
            return
        if not sys.platform.startswith("linux"):
            self.skipTest("native dumpability witness requires Linux")
        code = (
            "import ctypes, json\n"
            "from loki_agent import process_protections\n"
            "libc = ctypes.CDLL(None)\n"
            "assert libc.prctl(4, 1, 0, 0, 0) == 0\n"
            "before = libc.prctl(3, 0, 0, 0, 0)\n"
            "assert before == 1\n"
            "assert process_protections.protect_credential_process()\n"
            "after = libc.prctl(3, 0, 0, 0, 0)\n"
            "assert process_protections.protect_credential_process()\n"
            "print(json.dumps([before, after,\n"
            "    libc.prctl(3, 0, 0, 0, 0)]))\n"
        )

        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
