"""Setup decision/recovery tests. ACL calls are substituted, not native evidence."""

import ctypes
import os
import tempfile
import types
import unittest
from unittest import mock

from loki_agent import windows_api as api
from loki_agent import windows_containers as containers
from loki_agent import windows_setup as setup
from loki_agent import windows_state as state
from loki_agent import windows_verify as verify


# windows_state._final_path asks the kernel about an open handle, which only
# exists on Windows; this module tests the ledger and recovery logic on every
# host, so it substitutes the platform's canonicalisation.  The real call is
# exercised by the Windows legs.
_final_path_patch = None


def setUpModule():
    global _final_path_patch
    _final_path_patch = mock.patch.object(
        state, '_final_path',
        side_effect=lambda path: os.path.realpath(path))
    _final_path_patch.start()


def tearDownModule():
    _final_path_patch.stop()


SID = 'S-1-15-2-1-2-3'


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.workspace = os.path.join(self.root, 'work')
        self.removed = os.path.join(self.root, 'removed')
        self.new = os.path.join(self.root, 'new')
        self.location = os.path.join(self.root, 'ledger.json')
        self.key = state.workspace_key(self.workspace)
        self.profile = state.profile_name_for(self.workspace)
        self.old = [state.Grant(self.workspace, state.Access.READ_WRITE, 'workspace'),
                    state.Grant(self.removed, state.Access.READ_WRITE, 'user')]
        self.ledger = {'version': 1, 'workspaces': {
            self.key: state.ledger_entry(self.workspace, self.profile, self.old)}}
        self.backend = setup.WindowsBackend(self.ledger, self.location)
        self.descriptors = {g.path: state.add_package_ace('D:AI(A;OICI;FA;;;SY)', SID, g.access)
                            for g in self.old}
        self.descriptors[self.new] = 'D:AI(A;OICI;FA;;;SY)'
        patches = [
            mock.patch.object(api, 'current_user_sid', return_value='user'),
            mock.patch.object(self.backend, '_private_directory'),
            mock.patch.object(self.backend, '_ensure_profile', return_value=SID),
            mock.patch.object(api, 'dacl_sddl', side_effect=self.descriptors.__getitem__),
            mock.patch.object(containers, 'set_dacl_sddl', side_effect=self.descriptors.__setitem__),
            # The grant is checked, read and written through one handle; this
            # host has no handles, so the path itself stands in for it.
            mock.patch.object(api, 'final_path_from_handle', side_effect=lambda handle: handle),
            mock.patch.object(api, 'open_directory_for_acl', side_effect=lambda path: path),
            mock.patch.object(api, 'close_handle'),
            mock.patch.object(api, 'handle_dacl_sddl', side_effect=self.descriptors.__getitem__),
            mock.patch.object(containers, 'set_handle_dacl_sddl', side_effect=lambda handle, sddl: self.descriptors.__setitem__(handle, sddl)),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def definition(self):
        return state.Definition(self.workspace, [
            state.Grant(self.workspace, state.Access.READ, 'workspace'),
            state.Grant(self.new, state.Access.READ, 'user')])

    def apply(self):
        definition = self.definition()
        plan = state.build_plan(definition, self.ledger['workspaces'][self.key])
        return self.backend.apply(plan, definition)

    def test_remove_downgrade_and_repeat(self):
        self.assertTrue(all(c.status == 'pass' for c in self.apply()))
        self.assertNotIn(SID, self.descriptors[self.removed])
        self.assertNotIn('0x1301BF', self.descriptors[self.workspace])
        self.assertEqual(self.descriptors[self.workspace].count(SID), 1)
        before = dict(self.descriptors)
        self.assertTrue(all(c.status == 'pass' for c in self.apply()))
        self.assertEqual(before, self.descriptors)
        self.assertFalse(self.ledger['workspaces'][self.key].get('pending'))

    def test_failed_apply_keeps_recovery_paths_and_blocks_startup(self):
        with mock.patch.object(self.backend, '_grant_path', side_effect=api.WindowsApiError('grant failed')):
            checks = self.apply()
        self.assertTrue(any(c.status == 'fail' for c in checks))
        persisted = state.load_ledger(self.location)
        entry = persisted['workspaces'][self.key]
        self.assertTrue(entry['pending'])
        self.assertEqual({g.path for g in state.entry_grants(entry)},
                         {self.workspace, self.removed, self.new})
        with mock.patch.object(verify, 'verify_container') as inventory:
            self.assertEqual(verify.verify_workspace(persisted, self.workspace)[0].status, 'fail')
        inventory.assert_not_called()
        self.assertTrue(all(c.status == 'pass' for c in self.apply()))

    def test_final_save_failure_retains_pending_ledger_in_memory_and_on_disk(self):
        real_save = state.save_ledger
        calls = 0

        def save(blob, path):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('disk full')
            real_save(blob, path)
        with mock.patch.object(setup, 'save_ledger', side_effect=save):
            checks = self.apply()
        self.assertTrue(any(c.status == 'fail' for c in checks))
        self.assertTrue(self.ledger['workspaces'][self.key]['pending'])
        self.assertTrue(state.load_ledger(self.location)['workspaces'][self.key]['pending'])

    def test_uninstall_keeps_failed_entry_and_does_not_delete_profile(self):
        with mock.patch.object(self.backend, '_ungrant_path', side_effect=OSError('denied')), \
                mock.patch.object(containers, 'delete_app_container_profile') as delete, \
                mock.patch.object(api, 'derive_app_container_sid', return_value=SID):
            checks = self.backend.uninstall(self.ledger)
        delete.assert_not_called()
        self.assertTrue(any(c.status == 'fail' for c in checks))
        self.assertTrue(self.ledger['workspaces'][self.key]['pending'])
        self.assertEqual(state.entry_grants(self.ledger['workspaces'][self.key]), self.old)

    def test_backend_rechecks_private_paths_without_gui(self):
        definition = state.Definition(self.workspace, [
            state.Grant(setup.paths.credential_directory(), state.Access.READ, 'user')])
        plan = state.Plan(self.workspace, self.profile, [], [], [])
        with mock.patch.object(self.backend, '_ensure_profile') as create:
            checks = self.backend.apply(plan, definition)
        create.assert_not_called()
        self.assertEqual(checks[0].status, 'fail')


class RulesTests(unittest.TestCase):
    def test_read_only_code_grant_is_allowed_but_write_is_not(self):
        code = os.path.dirname(setup.__file__)
        self.assertEqual(state.protected_path_errors(code, state.Access.READ), [])
        self.assertTrue(state.protected_path_errors(code, state.Access.READ_WRITE))

    def test_alias_to_private_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            private = os.path.join(root, 'private')
            alias = os.path.join(root, 'alias')
            os.mkdir(private)
            os.symlink(private, alias)
            with mock.patch.object(state, '_runtime_trees', return_value=[private]):
                self.assertTrue(state.protected_path_errors(alias, state.Access.READ))

    def test_reload_does_not_duplicate_automatic_grants(self):
        expected = len(setup.automatic_grants('/work'))
        definition = setup.definition_for('/work', None)
        for _ in range(3):
            entry = state.ledger_entry('/work', 'profile', definition.grants)
            definition = setup.definition_for('/work', entry)
            self.assertEqual(len(definition.grants), expected)

    def test_replacing_allow_preserves_deny_and_other_principals(self):
        original = f'D:AI(D;;FW;;;{SID})(A;OICI;0x1301BF;;;{SID})(A;OICIID;FR;;;SY)'
        updated = state.add_package_ace(original, SID, state.Access.READ)
        self.assertEqual(updated,
                         f'D:AI(D;;FW;;;{SID})(A;OICI;FRFX;;;{SID})(A;OICIID;FR;;;SY)')
        self.assertEqual(state.remove_package_aces(updated, SID),
                         f'D:AI(D;;FW;;;{SID})(A;OICIID;FR;;;SY)')

    def test_inherited_package_allow_is_not_silently_removed(self):
        with self.assertRaises(api.WindowsApiError):
            state.remove_package_aces(f'D:AI(A;OICIID;FR;;;{SID})', SID)

    def test_unsupported_sddl_is_refused(self):
        for sddl in ('D:NO_ACCESS_CONTROL', 'S:(A;;FR;;;SY)', 'D:(XA;;FR;;;SY;(condition))'):
            with self.subTest(sddl=sddl), self.assertRaises(api.WindowsApiError):
                state.add_package_ace(sddl, SID, state.Access.READ)

    def test_failed_apply_does_not_report_gui_success(self):
        editor = setup.Editor.__new__(setup.Editor)
        editor.status = 0
        editor.messagebox = mock.Mock()
        editor.model = types.SimpleNamespace(
            plan=lambda: state.Plan('/work', 'profile', [], [], []),
            describe=lambda: 'change', apply=lambda: [state.Check('profile', 'fail')])
        self.assertFalse(editor._apply_current())
        self.assertEqual(editor.status, 1)
        editor.messagebox.showinfo.assert_not_called()

    def test_failed_uninstall_does_not_report_gui_success(self):
        editor = setup.Editor.__new__(setup.Editor)
        editor.status = 0
        editor.messagebox = mock.Mock()
        editor.model = types.SimpleNamespace(
            uninstall=lambda: [state.Check('ungrant', 'fail')])
        editor.uninstall()
        self.assertEqual(editor.status, 1)
        editor.messagebox.showerror.assert_called_once()
        editor.messagebox.showinfo.assert_not_called()


class InheritanceTests(unittest.TestCase):
    def test_only_protected_sddl_requests_disabling_inheritance(self):
        for sddl, expected in [('D:AI(A;;FR;;;SY)', 4), ('D:P(A;;FR;;;SY)', 4 | 0x80000000)]:
            set_named = mock.Mock(return_value=0)

            def convert(text, revision, output, length):
                ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 10
                return 1

            def get_dacl(descriptor, present, output, defaulted):
                ctypes.cast(present, ctypes.POINTER(ctypes.wintypes.BOOL))[0] = True
                ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 11
                return 1
            functions = {
                'ConvertStringSecurityDescriptorToSecurityDescriptorW': convert,
                'GetSecurityDescriptorDacl': get_dacl,
                'SetNamedSecurityInfoW': set_named,
                'LocalFree': lambda *args: None,
            }
            with self.subTest(sddl=sddl), mock.patch.object(
                    containers, 'bind', side_effect=lambda library, symbol, *args: functions[symbol]):
                containers.set_dacl_sddl('/work', sddl)
            self.assertEqual(set_named.call_args.args[2], expected)

    def test_handle_setter_requests_disabling_inheritance_like_the_named_one(self):
        # The handle-relative grant must carry the same protected-flag rule as
        # the pathname one, or a re-applied grant could drop DACL protection.
        for sddl, expected in [('D:AI(A;;FR;;;SY)', 4), ('D:P(A;;FR;;;SY)', 4 | 0x80000000)]:
            set_info = mock.Mock(return_value=0)

            def convert(text, revision, output, length):
                ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 10
                return 1

            def get_dacl(descriptor, present, output, defaulted):
                ctypes.cast(present, ctypes.POINTER(ctypes.wintypes.BOOL))[0] = True
                ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 11
                return 1
            functions = {
                'ConvertStringSecurityDescriptorToSecurityDescriptorW': convert,
                'GetSecurityDescriptorDacl': get_dacl,
                'SetSecurityInfo': set_info,
                'LocalFree': lambda *args: None,
            }
            handle = object()
            with self.subTest(sddl=sddl), mock.patch.object(
                    containers, 'bind', side_effect=lambda library, symbol, *args: functions[symbol]):
                containers.set_handle_dacl_sddl(handle, sddl)
            self.assertEqual(set_info.call_args.args[0], handle)
            self.assertEqual(set_info.call_args.args[2], expected)
