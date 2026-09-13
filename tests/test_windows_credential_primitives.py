"""Native exercises of the production Windows credential primitives.

The portable tests pin the declarations and the logic around the calls; this is
the only place the calls themselves execute, so the diagnostics are printed and
the assertions state what a stock Windows box actually does.

This is a separate file from ``test_windows_primitives`` because that file is
copied and run on its own by the standard-user probe, which stages it without
the package beside it; this file needs ``loki_agent`` importable.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from loki_agent import _credential_files_windows  # noqa: E402
from loki_agent import credential_files  # noqa: E402
from loki_agent import windows_acl  # noqa: E402
from loki_agent import windows_api  # noqa: E402


@unittest.skipUnless(os.name == 'nt', 'requires native Windows Python')
class CredentialPrimitiveTests(unittest.TestCase):
    """Run the production credential primitives on a real Windows volume."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / 'credentials'
        self.directory.mkdir(mode=0o700)

    def test_well_known_aliases_canonicalize_to_full_sids(self):
        # A DACL read back spells well-known trustees as SDDL aliases; the
        # privacy check compares full SIDs, so the conversion is load-bearing.
        self.assertEqual(windows_api.canonical_sid('SY'), 'S-1-5-18')
        self.assertEqual(windows_api.canonical_sid('BA'), 'S-1-5-32-544')
        self.assertEqual(windows_api.canonical_sid('OW'), 'S-1-3-4')
        self.assertEqual(
            windows_api.canonical_sid('S-1-5-21-1-2-3-1001'),
            'S-1-5-21-1-2-3-1001')

    def test_a_private_directory_is_owned_and_unshared(self):
        dacl = windows_api.dacl_sddl(str(self.directory))
        print(json.dumps({
            'operation': 'mkdir_0o700_dacl', 'sddl': dacl,
            'owner': windows_api.current_user_sid(),
            'trustees': sorted(windows_acl.allow_trustees(dacl))}), flush=True)
        facts = credential_files.describe_path(str(self.directory))
        self.assertTrue(facts.directory)
        self.assertFalse(facts.reparse_point)
        self.assertTrue(facts.owned_by_current_user)
        # If this fails, the DACL Windows gives a private directory names a
        # trustee the privacy allowlist does not -- a fact to fix, not hide.
        self.assertFalse(facts.group_or_other_access)

    def test_create_read_replace_and_unlink_round_trip(self):
        directory = _credential_files_windows.open_directory(str(self.directory))
        self.addCleanup(_credential_files_windows.close, directory)

        first = _credential_files_windows.create_exclusive_at(
            directory, 'tokens.json', 0o600)
        # Exclusive create: the second must refuse, mapped from the NTSTATUS
        # name collision to FileExistsError.
        with self.assertRaises(FileExistsError):
            _credential_files_windows.create_exclusive_at(
                directory, 'tokens.json', 0o600)
        _credential_files_windows.write(first, b'{"secret": 1}')
        _credential_files_windows.fsync(first)
        _credential_files_windows.close(first)

        reader = _credential_files_windows.open_read_at(directory, 'tokens.json')
        try:
            facts = _credential_files_windows.describe(reader)
            self.assertTrue(facts.regular)
            self.assertFalse(facts.group_or_other_access)
            self.assertEqual(facts.size, len(b'{"secret": 1}'))
            self.assertEqual(
                _credential_files_windows.read(reader, 1024), b'{"secret": 1}')
        finally:
            _credential_files_windows.close(reader)

        with self.assertRaises(FileNotFoundError):
            _credential_files_windows.open_read_at(directory, 'absent.json')

        # A reader holding the destination open must not block publication;
        # this is why every open requests shared read/write/delete.
        reader = _credential_files_windows.open_read_at(directory, 'tokens.json')
        try:
            temporary = _credential_files_windows.create_exclusive_at(
                directory, '.tokens.json.tmp', 0o600)
            _credential_files_windows.write(temporary, b'{"secret": 2}')
            _credential_files_windows.fsync(temporary)
            _credential_files_windows.close(temporary)
            _credential_files_windows.replace_at(
                directory, '.tokens.json.tmp', 'tokens.json')
        finally:
            _credential_files_windows.close(reader)

        reader = _credential_files_windows.open_read_at(directory, 'tokens.json')
        try:
            self.assertEqual(
                _credential_files_windows.read(reader, 1024), b'{"secret": 2}')
        finally:
            _credential_files_windows.close(reader)

        _credential_files_windows.unlink_at(directory, 'tokens.json')
        with self.assertRaises(FileNotFoundError):
            _credential_files_windows.open_read_at(directory, 'tokens.json')

    def test_operations_stay_bound_to_the_retained_directory(self):
        directory = _credential_files_windows.open_directory(str(self.directory))
        self.addCleanup(_credential_files_windows.close, directory)
        handle = _credential_files_windows.create_exclusive_at(
            directory, 'tokens.json', 0o600)
        _credential_files_windows.write(handle, b'real')
        _credential_files_windows.close(handle)

        # Rename the directory and recreate its old name with a decoy: the
        # retained handle must still reach the original object.
        os.rename(self.directory, self.root / 'moved')
        self.directory.mkdir(mode=0o700)
        decoy_directory = _credential_files_windows.open_directory(
            str(self.directory))
        decoy = _credential_files_windows.create_exclusive_at(
            decoy_directory, 'tokens.json', 0o600)
        _credential_files_windows.write(decoy, b'decoy')
        _credential_files_windows.close(decoy)
        _credential_files_windows.close(decoy_directory)

        reader = _credential_files_windows.open_read_at(directory, 'tokens.json')
        try:
            self.assertEqual(
                _credential_files_windows.read(reader, 1024), b'real')
        finally:
            _credential_files_windows.close(reader)

    def test_a_directory_reparse_point_is_refused(self):
        real = self.root / 'real'
        real.mkdir()
        link = self.root / 'link'
        result = subprocess.run(
            ['cmd.exe', '/d', '/c', 'mklink', '/J',
             str(link).replace('/', '\\'), str(real).replace('/', '\\')],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.addCleanup(lambda: os.rmdir(link))

        self.assertTrue(credential_files.describe_path(str(link)).reparse_point)
        # A handle to the junction would still traverse it as a root, so
        # open_directory must refuse the link itself.
        with self.assertRaises(credential_files.CredentialStorageError):
            _credential_files_windows.open_directory(str(link))

    def test_fsync_flushes_a_file_and_is_a_no_op_for_a_directory(self):
        directory = _credential_files_windows.open_directory(str(self.directory))
        try:
            # A directory cannot be flushed; the primitive must not raise.
            _credential_files_windows.fsync(directory)
            handle = _credential_files_windows.create_exclusive_at(
                directory, 'flush.json', 0o600)
            try:
                _credential_files_windows.write(handle, b'x')
                _credential_files_windows.fsync(handle)
            finally:
                _credential_files_windows.close(handle)
        finally:
            _credential_files_windows.close(directory)


if __name__ == '__main__':
    unittest.main()
