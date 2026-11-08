"""Setup decision/recovery tests. ACL calls are substituted, not native evidence."""

from contextlib import contextmanager
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
        self.config = os.path.join(self.root, 'config')
        self.state = os.path.join(self.root, 'state')
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
            mock.patch.object(setup.paths, 'loki_config_dir', return_value=self.config),
            mock.patch.object(setup.paths, 'loki_state_dir', return_value=self.state),
            mock.patch.object(setup.paths, 'credential_directory',
                              return_value=os.path.join(self.config, 'credentials')),
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

    def test_shared_config_reads_survive_another_workspace_reapply_and_removal(self):
        other_workspace = os.path.join(self.root, 'other-workspace')
        other_sid = 'S-1-15-2-9-8-7'
        self.descriptors[self.config] = setup.private_dacl_sddl('user')
        self.descriptors[other_workspace] = setup.private_dacl_sddl('user')
        first = state.Definition(self.workspace, [
            state.Grant(self.workspace, state.Access.READ_WRITE, 'workspace'),
            state.Grant(self.config, state.Access.READ, 'config')])
        second = state.Definition(other_workspace, [
            state.Grant(other_workspace, state.Access.READ_WRITE, 'workspace'),
            state.Grant(self.config, state.Access.READ, 'config')])

        def apply(definition, sid):
            previous = self.ledger['workspaces'].get(state.workspace_key(definition.workspace))
            with mock.patch.object(self.backend, '_ensure_profile', return_value=sid):
                checks = self.backend.apply(state.build_plan(definition, previous), definition)
            self.assertTrue(all(check.status == 'pass' for check in checks), checks)

        apply(first, SID)
        apply(second, other_sid)
        apply(first, SID)
        self.assertTrue(state.grants_access(self.descriptors[self.config], SID, state.Access.READ))
        self.assertTrue(state.grants_access(
            self.descriptors[self.config], other_sid, state.Access.READ))
        self.backend._private_directory.assert_any_call(
            os.path.join(self.config, 'credentials'), 'user')
        self.assertNotIn(mock.call(self.config, 'user'),
                         self.backend._private_directory.call_args_list)
        with mock.patch.object(api, 'derive_app_container_sid', return_value=SID), \
                mock.patch.object(containers, 'delete_app_container_profile'):
            checks = self.backend.uninstall({'workspaces': {
                self.key: self.ledger['workspaces'][self.key]}})
        self.assertTrue(all(check.status == 'pass' for check in checks), checks)
        self.assertNotIn(SID, self.descriptors[self.config])
        self.assertTrue(state.grants_access(
            self.descriptors[self.config], other_sid, state.Access.READ))
        self.assertIn(state.workspace_key(other_workspace), self.ledger['workspaces'])

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
    @contextmanager
    def _dacl_calls(self, *, conversion=True, descriptor=10, readable=True, present=True, dacl=11, status=0):
        # Substitute only the native ABI, not inheritance/ownership decisions.
        # Both pathname policies and the shared handle primitive must release
        # their converted descriptor without consuming the caller's handle.
        def convert(text, revision, output, length):
            output._obj.value = descriptor
            return conversion

        def get_dacl(descriptor, present_out, output, defaulted):
            present_out._obj.value = present
            output._obj.value = dacl
            return readable

        calls = {
            'ConvertStringSecurityDescriptorToSecurityDescriptorW': mock.Mock(side_effect=convert),
            'GetSecurityDescriptorDacl': mock.Mock(side_effect=get_dacl),
            'SetNamedSecurityInfoW': mock.Mock(return_value=status),
            'SetSecurityInfo': mock.Mock(return_value=status),
            'LocalFree': mock.Mock(return_value=None),
        }

        def bind(library, symbol, *args):
            return calls[symbol]

        with mock.patch.object(containers, 'bind', side_effect=bind), \
                mock.patch.object(api, 'bind', side_effect=bind), \
                mock.patch.object(api.ctypes, 'get_last_error', return_value=5, create=True), \
                mock.patch.object(api, 'close_handle') as close:
            yield calls
            close.assert_not_called()

    def test_only_protected_sddl_requests_disabling_inheritance(self):
        for sddl, expected in [
                ('D:AI(A;;FR;;;SY)', api.SecurityInformation.DACL_SECURITY_INFORMATION),
                ('D:P(A;;FR;;;SY)', api.SecurityInformation.DACL_SECURITY_INFORMATION
                 | api.SecurityInformation.PROTECTED_DACL_SECURITY_INFORMATION)]:
            with self.subTest(sddl=sddl), self._dacl_calls() as calls:
                containers.set_dacl_sddl('/work', sddl)
            self.assertEqual(calls['SetNamedSecurityInfoW'].call_args.args[2], expected)
            calls['LocalFree'].assert_called_once()
            self.assertEqual(calls['LocalFree'].call_args.args[0].value, 10)

    def test_handle_setter_requests_disabling_inheritance_like_the_named_one(self):
        for sddl, expected in [
                ('D:AI(A;;FR;;;SY)', api.SecurityInformation.DACL_SECURITY_INFORMATION),
                ('D:P(A;;FR;;;SY)', api.SecurityInformation.DACL_SECURITY_INFORMATION
                 | api.SecurityInformation.PROTECTED_DACL_SECURITY_INFORMATION)]:
            handle = object()
            with self.subTest(sddl=sddl), self._dacl_calls() as calls:
                containers.set_handle_dacl_sddl(handle, sddl)
            applied = calls['SetSecurityInfo'].call_args.args
            self.assertIs(applied[0], handle)
            self.assertEqual(applied[1], api.SE_FILE_OBJECT)
            self.assertEqual(applied[2], expected)
            calls['SetNamedSecurityInfoW'].assert_not_called()
            calls['LocalFree'].assert_called_once()
            self.assertEqual(calls['LocalFree'].call_args.args[0].value, 10)

    def test_kernel_handle_keeps_dacl_only_behavior(self):
        with self._dacl_calls() as calls:
            api.set_handle_dacl(object(), 'D:P(A;;FR;;;SY)', object_type=api.SE_KERNEL_OBJECT)
        applied = calls['SetSecurityInfo'].call_args.args
        self.assertEqual(applied[1], api.SE_KERNEL_OBJECT)
        self.assertEqual(applied[2], api.SecurityInformation.DACL_SECURITY_INFORMATION)

    def test_shared_named_setter_does_not_acquire_setup_protection_policy(self):
        with self._dacl_calls() as calls:
            api.set_named_dacl('/work', 'D:P(A;;FR;;;SY)')
        self.assertEqual(calls['SetNamedSecurityInfoW'].call_args.args[2],
                         api.SecurityInformation.DACL_SECURITY_INFORMATION)
        calls['SetSecurityInfo'].assert_not_called()

    def test_handle_conversion_failure_does_not_free_unacquired_output(self):
        with self._dacl_calls(conversion=False) as calls:
            with self.assertRaises(api.WindowsApiError):
                api.set_handle_dacl(object(), 'D:P(A;;FR;;;SY)')
        calls['GetSecurityDescriptorDacl'].assert_not_called()
        calls['SetSecurityInfo'].assert_not_called()
        calls['LocalFree'].assert_not_called()

    def test_null_descriptor_is_not_queried_applied_or_freed(self):
        for setter, target in [[api.set_handle_dacl, object()],
                               [api.set_named_dacl, '/work'],
                               [containers.set_dacl_sddl, '/work']]:
            with self.subTest(setter=setter.__name__), self._dacl_calls(descriptor=None) as calls:
                with self.assertRaisesRegex(api.WindowsApiError, 'null'):
                    setter(target, 'D:P(A;;FR;;;SY)')
            for name in ['GetSecurityDescriptorDacl', 'SetSecurityInfo', 'SetNamedSecurityInfoW', 'LocalFree']:
                calls[name].assert_not_called()

    def _assert_handle_dacl_refused(self, **options):
        with self._dacl_calls(**options) as calls:
            with self.assertRaises(api.WindowsApiError):
                api.set_handle_dacl(object(), 'D:P(A;;FR;;;SY)')
        calls['SetSecurityInfo'].assert_not_called()
        calls['LocalFree'].assert_called_once()
        self.assertEqual(calls['LocalFree'].call_args.args[0].value, 10)

    def test_handle_query_failure_releases_the_descriptor_without_applying(self):
        self._assert_handle_dacl_refused(readable=False)

    def test_absent_handle_dacl_is_refused_without_applying(self):
        self._assert_handle_dacl_refused(present=False)

    def test_null_handle_dacl_is_refused_without_applying(self):
        self._assert_handle_dacl_refused(dacl=0)

    def test_handle_application_failure_preserves_status_and_releases_the_descriptor(self):
        handle = object()
        with self._dacl_calls(status=5) as calls:
            with self.assertRaises(api.WindowsApiError) as refused:
                api.set_handle_dacl(handle, 'D:P(A;;FR;;;SY)')
        self.assertEqual(refused.exception.status, 5)
        self.assertIs(calls['SetSecurityInfo'].call_args.args[0], handle)
        calls['SetNamedSecurityInfoW'].assert_not_called()
        calls['LocalFree'].assert_called_once()
        self.assertEqual(calls['LocalFree'].call_args.args[0].value, 10)
