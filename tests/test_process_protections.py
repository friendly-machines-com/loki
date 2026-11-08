import json
import os
import subprocess
import sys
import unittest
from unittest import mock

from loki_agent import process_protections, windows_api


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakePrctl:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []
        self.restype = None

    def __call__(self, *args):
        self.calls.append(list(
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
                expected = [[4, 0, 0, 0, 0]]  # SET_DUMPABLE must clear it.
                if len(results) == 2:
                    expected.append([3, 0, 0, 0, 0])  # GET_DUMPABLE
                self.assertEqual(prctl.calls, expected)

    def test_windows_current_objects_and_future_default_have_independent_owners(self):
        for valid in [True, False]:
            with self.subTest(valid=valid), \
                    mock.patch.object(process_protections.sys, 'platform', 'win32'), \
                    mock.patch.object(windows_api, 'current_user_sid', return_value='owner'), \
                    mock.patch.object(windows_api, 'current_process_handle', return_value=1), \
                    mock.patch.object(windows_api, 'bind', return_value=lambda: 2), \
                    mock.patch.object(windows_api, 'set_handle_dacl') as protect, \
                    mock.patch.object(windows_api, 'open_process_token', return_value=3) as open_token, \
                    mock.patch.object(windows_api, 'set_token_default_dacl', side_effect=(
                        None if valid else windows_api.WindowsApiError('default refused'))) as install, \
                    mock.patch.object(windows_api, 'close_handle') as close:
                if valid:
                    self.assertTrue(process_protections.protect_credential_process())
                else:
                    with self.assertRaises(process_protections.ProcessProtectionError):
                        process_protections.protect_credential_process()
                open_token.assert_called_once_with(
                    1, windows_api.TokenAccess.TOKEN_QUERY | windows_api.TokenAccess.TOKEN_ADJUST_DEFAULT)
                install.assert_called_once_with(3, process_protections.private_object_dacl('owner'))
                close.assert_called_once_with(3)  # Both current-object handles are borrowed pseudohandles.
                self.assertEqual([call.args[0] for call in protect.call_args_list], [1, 2] if valid else [1])

    def test_real_process_reports_protection(self):
        if sys.platform == "win32":
            # Observe this direct caller and a genuinely later thread. The
            # suspended-child tests separately establish creation-time security.
            code = (
                "import ctypes, json, threading\n"
                "from loki_agent import process_protections, windows_api as api\n"
                "process_protections.protect_credential_process()\n"
                "current_thread = api.bind('kernel32', 'GetCurrentThread', ctypes.c_void_p)\n"
                "threads = [api.handle_dacl_sddl(current_thread(), api.SE_KERNEL_OBJECT)]\n"
                "worker = threading.Thread(target=lambda: threads.append(\n"
                "    api.handle_dacl_sddl(current_thread(), api.SE_KERNEL_OBJECT)))\n"
                "worker.start(); worker.join()\n"
                "token = api.open_process_token(api.current_process_handle())\n"
                "try:\n"
                "    result = {'owner': api.current_user_sid(), 'threads': threads,\n"
                "              'default': api.acl_sddl(api.token_default_dacl(token)),\n"
                "              'dacl': api.handle_dacl_sddl(api.current_process_handle(), api.SE_KERNEL_OBJECT)}\n"
                "finally: api.close_handle(token)\n"
                "print(json.dumps(result))\n"
            )
            process = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            from loki_agent import windows_acl
            result = json.loads(process.stdout)
            self.assertEqual(len(result['threads']), 2)
            expected = {windows_api.canonical_sid(sid) for sid in [result['owner'], 'SY', 'BA']}
            for dacl in [result['dacl'], result['default'], *result['threads']]:
                self.assertIsNotNone(dacl)
                self.assertEqual({windows_api.canonical_sid(sid) for sid in
                                  windows_acl.allow_trustees(dacl, include_inherit_only=True)}, expected)
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
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
