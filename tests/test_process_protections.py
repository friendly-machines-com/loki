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

    def test_linux_sets_and_verifies_non_dumpable(self):
        prctl = FakePrctl([0, 0])
        libc = mock.Mock(prctl=prctl)
        with mock.patch.object(
                process_protections.sys, "platform", "linux"), \
                mock.patch.object(
                    process_protections.ctypes, "CDLL",
                    return_value=libc):
            protected = (
                process_protections.protect_credential_process())

        self.assertTrue(protected)
        self.assertEqual(
            [call[0] for call in prctl.calls],
            [
                process_protections._PR_SET_DUMPABLE,
                process_protections._PR_GET_DUMPABLE,
            ],
        )

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "Linux prctl contract")
    def test_real_linux_process_reports_non_dumpable(self):
        code = (
            "import ctypes\n"
            "from loki_agent import process_protections\n"
            "process_protections.protect_credential_process()\n"
            "libc = ctypes.CDLL(None)\n"
            "print(libc.prctl(3, 0, 0, 0, 0))\n"
        )

        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "0")


if __name__ == "__main__":
    unittest.main()
