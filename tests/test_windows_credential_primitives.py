"""Native exercises of Loki's production Windows primitives.

The portable tests pin the declarations and the logic around the calls; this is
the only place the calls themselves execute, so the diagnostics are printed and
the assertions state what a stock Windows box actually does.  It covers the
credential-storage primitives and the handle-bound setup grant primitives.

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

from loki_agent import _private_files_windows  # noqa: E402
from loki_agent import credential_storages  # noqa: E402
from loki_agent import private_files  # noqa: E402
from loki_agent import windows_acl  # noqa: E402
from loki_agent import windows_api  # noqa: E402
from loki_agent import windows_containers  # noqa: E402
from loki_agent import windows_state  # noqa: E402


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
        facts = private_files.describe_path(str(self.directory))
        self.assertTrue(facts.directory)
        self.assertFalse(facts.reparse_point)
        self.assertTrue(facts.owned_by_current_user)
        # If this fails, the DACL Windows gives a private directory names a
        # trustee the privacy allowlist does not -- a fact to fix, not hide.
        self.assertFalse(facts.group_or_other_access)

    def test_create_read_replace_and_unlink_round_trip(self):
        directory = _private_files_windows.open_directory(str(self.directory))
        self.addCleanup(_private_files_windows.close, directory)

        first = _private_files_windows.create_exclusive_at(
            directory, 'tokens.json', 0o600)
        # Exclusive create: the second must refuse, mapped from the NTSTATUS
        # name collision to FileExistsError.
        with self.assertRaises(FileExistsError):
            _private_files_windows.create_exclusive_at(
                directory, 'tokens.json', 0o600)
        _private_files_windows.write(first, b'{"secret": 1}')
        _private_files_windows.fsync(first)
        _private_files_windows.close(first)

        reader = _private_files_windows.open_read_at(directory, 'tokens.json')
        try:
            facts = _private_files_windows.describe(reader)
            self.assertTrue(facts.regular)
            self.assertFalse(facts.group_or_other_access)
            self.assertEqual(facts.size, len(b'{"secret": 1}'))
            self.assertEqual(
                _private_files_windows.read(reader, 1024), b'{"secret": 1}')
        finally:
            _private_files_windows.close(reader)

        with self.assertRaises(FileNotFoundError):
            _private_files_windows.open_read_at(directory, 'absent.json')

        # Windows refuses a rename that would replace a destination another
        # handle holds open (STATUS_ACCESS_DENIED), so publication closes the
        # destination first.  The storage reads and closes the JSON before it
        # writes, which is this same ordering; a reader must not be left holding
        # tokens.json across the replace.
        reader = _private_files_windows.open_read_at(directory, 'tokens.json')
        _private_files_windows.close(reader)
        temporary = _private_files_windows.create_exclusive_at(
            directory, '.tokens.json.tmp', 0o600)
        _private_files_windows.write(temporary, b'{"secret": 2}')
        _private_files_windows.fsync(temporary)
        _private_files_windows.close(temporary)
        _private_files_windows.replace_at(
            directory, '.tokens.json.tmp', 'tokens.json')

        reader = _private_files_windows.open_read_at(directory, 'tokens.json')
        try:
            self.assertEqual(
                _private_files_windows.read(reader, 1024), b'{"secret": 2}')
        finally:
            _private_files_windows.close(reader)

        _private_files_windows.unlink_at(directory, 'tokens.json')
        with self.assertRaises(FileNotFoundError):
            _private_files_windows.open_read_at(directory, 'tokens.json')

    def test_operations_stay_bound_to_the_retained_directory(self):
        directory = _private_files_windows.open_directory(str(self.directory))
        self.addCleanup(_private_files_windows.close, directory)
        handle = _private_files_windows.create_exclusive_at(
            directory, 'tokens.json', 0o600)
        _private_files_windows.write(handle, b'real')
        _private_files_windows.close(handle)

        # Rename the directory and recreate its old name with a decoy: the
        # retained handle must still reach the original object.
        os.rename(self.directory, self.root / 'moved')
        self.directory.mkdir(mode=0o700)
        decoy_directory = _private_files_windows.open_directory(
            str(self.directory))
        decoy = _private_files_windows.create_exclusive_at(
            decoy_directory, 'tokens.json', 0o600)
        _private_files_windows.write(decoy, b'decoy')
        _private_files_windows.close(decoy)
        _private_files_windows.close(decoy_directory)

        reader = _private_files_windows.open_read_at(directory, 'tokens.json')
        try:
            self.assertEqual(
                _private_files_windows.read(reader, 1024), b'real')
        finally:
            _private_files_windows.close(reader)

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

        self.assertTrue(private_files.describe_path(str(link)).reparse_point)
        # A handle to the junction would still traverse it as a root, so
        # The primitive opens the link itself (FILE_OPEN_REPARSE_POINT) and
        # returns its handle; refusing it is the caller's decision, made from
        # describe() on that handle.
        handle = _private_files_windows.open_directory(str(link))
        try:
            self.assertTrue(_private_files_windows.describe(handle).reparse_point)
        finally:
            _private_files_windows.close(handle)
        with self.assertRaises(private_files.CredentialStorageError):
            credential_storages.JsonCredentialStorage(str(link))._open_directory()

    def test_fsync_flushes_a_file_and_is_a_no_op_for_a_directory(self):
        directory = _private_files_windows.open_directory(str(self.directory))
        try:
            # A directory cannot be flushed; the primitive must not raise.
            _private_files_windows.fsync(directory)
            handle = _private_files_windows.create_exclusive_at(
                directory, 'flush.json', 0o600)
            try:
                _private_files_windows.write(handle, b'x')
                _private_files_windows.fsync(handle)
            finally:
                _private_files_windows.close(handle)
        finally:
            _private_files_windows.close(directory)


@unittest.skipUnless(os.name == 'nt', 'requires native Windows Python')
class GrantHandlePrimitiveTests(unittest.TestCase):
    """Native checks of the handle-bound setup grant primitives.

    The portable tests in ``test_windows_setup`` substitute these calls; this is
    where ``open_directory_for_acl``, ``handle_dacl_sddl`` and
    ``set_handle_dacl_sddl`` actually run.  A grant binds the containment check,
    the DACL read and the DACL write to one handle so a junction swapped into
    the operand cannot move the decision or the write to a different object.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / 'target'
        self.target.mkdir()
        self.package = windows_api.derive_app_container_sid('loki-grant-probe')

    def test_the_handle_keeps_naming_its_directory_across_a_rename(self):
        # The handle binds the check, the DACL read and the DACL write to the
        # object, not to the name.  Withholding FILE_SHARE_DELETE does not stop
        # a directory being renamed -- measured: os.rename succeeds while the
        # handle is open -- so what closes the check/apply race is that the
        # handle still names the object the check judged.  Rename the target,
        # put a decoy at the name the handle was opened with, then write a DACL
        # through the handle: the moved original must change, the decoy must
        # not.
        handle = windows_api.open_directory_for_acl(str(self.target))
        try:
            self.assertEqual(
                os.path.normcase(windows_api.final_path_from_handle(handle)),
                os.path.normcase(str(self.target)))
            current = windows_api.handle_dacl_sddl(handle)
            moved = self.root / 'moved'
            os.rename(self.target, moved)
            self.target.mkdir()
            updated = windows_state.add_package_ace(
                current, self.package, windows_state.Access.READ_WRITE)
            windows_containers.set_handle_dacl_sddl(handle, updated)
            self.assertEqual(
                os.path.normcase(windows_api.final_path_from_handle(handle)),
                os.path.normcase(str(moved)))
        finally:
            windows_api.close_handle(handle)
        moved_dacl = windows_api.dacl_sddl(str(moved))
        decoy_dacl = windows_api.dacl_sddl(str(self.target))
        print(json.dumps({'operation': 'dacl_after_rename',
                          'moved': moved_dacl, 'decoy': decoy_dacl}),
              flush=True)
        self.assertTrue(windows_acl.names_package(moved_dacl, self.package))
        self.assertFalse(windows_acl.names_package(decoy_dacl, self.package))

    def test_the_identity_open_reaches_the_final_path_uncontained(self):
        # The deciding experiment for the contained gate's
        # "GetFinalPathNameByHandleW sizing failed: 5".  The runtime opens its
        # workspace with FILE_READ_ATTRIBUTES only (open_directory_handle), not
        # with the grant handle's READ_CONTROL|WRITE_DAC.  If this call
        # succeeds here, outside any container, the access mask is not the
        # cause and the AppContainer token is; if it raises, the status in the
        # message names the right the query needs.
        handle = windows_api.open_directory_handle(str(self.target))
        try:
            final = windows_api.final_path_from_handle(handle)
        finally:
            windows_api.close_handle(handle)
        print(json.dumps({'operation': 'identity_open_final_path',
                          'final': final}), flush=True)
        self.assertEqual(
            os.path.normcase(final), os.path.normcase(str(self.target)))

    def test_a_grant_reads_and_writes_the_dacl_of_its_handle(self):
        handle = windows_api.open_directory_for_acl(str(self.target))
        try:
            current = windows_api.handle_dacl_sddl(handle)
            updated = windows_state.add_package_ace(
                current, self.package, windows_state.Access.READ_WRITE)
            windows_containers.set_handle_dacl_sddl(handle, updated)
        finally:
            windows_api.close_handle(handle)
        on_disk = windows_api.dacl_sddl(str(self.target))
        print(json.dumps({'operation': 'grant_through_handle', 'sddl': on_disk}),
              flush=True)
        self.assertTrue(windows_acl.names_package(on_disk, self.package))

    def test_a_protected_dacl_stays_protected_through_the_handle(self):
        # The handle setter must carry the same protected-flag rule as the
        # pathname setter, or re-applying a grant could drop DACL protection.
        protected = "D:P(A;OICI;FA;;;%s)" % windows_api.current_user_sid()
        handle = windows_api.open_directory_for_acl(str(self.target))
        try:
            windows_containers.set_handle_dacl_sddl(handle, protected)
        finally:
            windows_api.close_handle(handle)
        header = windows_api.dacl_sddl(str(self.target)).split('(', 1)[0]
        self.assertIn('P', header)


if __name__ == '__main__':
    unittest.main()
