"""Landlock UAPI selection, rule contents, and ownership without a live domain."""

import contextlib
import ctypes
import errno
import os
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import landlock_bindings


class SyscallNumberTests(unittest.TestCase):
    def test_shared_and_mips_syscall_numbers(self):
        # Expected values from Linux's tables, not from the resolver's data.
        for expected, builds in [
                ((444, 446), [('x86_64-linux-gnu', 8), ('i386-linux-gnu', 4), ('i686-linux-gnu', 4),
                              ('aarch64-linux-gnu', 8), ('arm-linux-gnueabihf', 4), ('armeb-linux-gnueabi', 4),
                              ('riscv64-linux-gnu', 8), ('riscv32-linux-gnu', 4), ('s390-linux-gnu', 4),
                              ('hppa-linux-gnu', 8), ('powerpc64le-linux-gnu', 8), ('loongarch64-linux-gnu', 8)]),
                ((4444, 4446), [('mipsel-linux-gnu', 4), ('mipsisa32r6-linux-musl', 4)]),
                ((6444, 6446), [('mips64el-linux-gnuabin32', 4)]),
                ((5444, 5446), [('mipsisa64r6el-linux-gnuabi64', 8), ('mips64-linux-muslabi64', 8)])]:
            for tag, width in builds:
                with self.subTest(tag=tag, width=width):
                    self.assertEqual(landlock_bindings._syscall_numbers(tag, width), expected)

    def test_no_guess_for_excluded_missing_or_inconsistent_abi(self):
        for tag, width in [('', 8), ('x86_64-linux-gnux32', 4), ('x86_64-linux-gnu', 4),
                           ('arm-linux-gnu', 4), ('alpha-linux-gnu', 8), ('mips64el-linux-gnu', 4),
                           ('mips64el-linux-gnuabi64', 4), ('mips64el-linux-gnuabin32', 8),
                           ('unknown-linux-gnu', 8)]:
            with self.subTest(tag=tag, width=width), self.assertRaises(OSError) as result:
                landlock_bindings._syscall_numbers(tag, width)
            self.assertEqual(result.exception.errno, errno.ENOTSUP)


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux Landlock binding')
class ScopeInstallationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(landlock_bindings, '_syscall_numbers', return_value=(444, 446)))
        self.syscall = self.enterContext(mock.patch.object(ctypes, 'CDLL')).return_value.syscall

    def test_abi_probe_refusal_does_not_create_ruleset(self):
        for returned, error, message in [(5, errno.ENOTSUP, 'ABI 6 or later'),
                                         (-1, errno.ENOSYS, 'availability check failed'),
                                         (-1, errno.EOPNOTSUPP, 'availability check failed')]:
            self.syscall.reset_mock()
            self.syscall.return_value = returned
            with self.subTest(returned=returned, error=error), mock.patch.object(ctypes, 'get_errno', return_value=error):
                with self.assertRaisesRegex(OSError, message) as result:
                    landlock_bindings.restrict_abstract_unix_sockets()
                self.assertEqual(result.exception.errno, error)
                self.assertEqual(self.syscall.call_count, 1)

    def test_scope_only_and_ruleset_fd_closed_on_success_and_refusal(self):
        for allowed in [True, False]:
            with self.subTest(allowed=allowed), tempfile.TemporaryFile() as handle:
                fd = os.dup(handle.fileno())
                try:
                    self.syscall.reset_mock()
                    self.syscall.side_effect = [6, fd, 0 if allowed else -1]
                    if allowed:
                        landlock_bindings.restrict_abstract_unix_sockets()
                    else:
                        with mock.patch.object(ctypes, 'get_errno', return_value=errno.EPERM):
                            with self.assertRaisesRegex(PermissionError, 'enforcement failed'):
                                landlock_bindings.restrict_abstract_unix_sockets()
                    probe, create, restrict = self.syscall.call_args_list
                    self.assertEqual((probe.args[0], probe.args[2].value, probe.args[3].value), (444, 0, 1))
                    self.assertIsNone(probe.args[1].value)
                    self.assertEqual((create.args[0], create.args[2].value, create.args[3].value), (444, 24, 0))
                    attr = ctypes.cast(create.args[1], ctypes.POINTER(landlock_bindings._Ruleset)).contents
                    self.assertEqual((attr.handled_access_fs, attr.handled_access_net, attr.scoped), (0, 0, 1))
                    self.assertEqual((restrict.args[0], restrict.args[1].value, restrict.args[2].value), (446, fd, 0))
                    with self.assertRaises(OSError) as closed:
                        os.fstat(fd)
                    self.assertEqual(closed.exception.errno, errno.EBADF)
                finally:
                    with contextlib.suppress(OSError):
                        os.close(fd)
