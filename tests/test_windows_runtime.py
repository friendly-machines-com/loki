"""Portable startup-gate tests; native launch still needs Windows validation."""

import asyncio
from contextlib import ExitStack, contextmanager
import ctypes
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock
from windows_state_fixtures import directory_identity_patch

from loki_agent import windows_runtime as runtime
from loki_agent import windows_api as api
from loki_agent.windows_state import Check
from loki_agent.runtime_isolations import RuntimeIsolationError


# The canonical path of an object is asked of the kernel about an open handle
# (windows_state._final_path), which only exists on Windows.  These modules test
# the ledger, plan and gate logic on every host, so they substitute the
# platform's canonicalisation with its Python equivalent; the real call is
# exercised by the Windows legs.


def setUpModule():
    unittest.enterModuleContext(directory_identity_patch())


class NativeDiagnosticTests(unittest.TestCase):
    def test_independent_acl_diagnostic_uses_native_modules_and_literal_filenames(self):
        from loki_entrypoints import _powershell_sddl
        result = types.SimpleNamespace(returncode=0, stdout='D:PAI\n', stderr='')
        path = r"C:\users\O'Hara\project"
        caller_environment = {'SystemRoot': r'C:\Windows', 'PSModulePath': r'C:\PowerShell7\Modules'}
        with mock.patch.dict(os.environ, caller_environment, clear=True), \
                mock.patch('loki_entrypoints.subprocess.run', return_value=result) as run:
            self.assertEqual(_powershell_sddl(path), 'D:PAI')
        self.assertEqual(run.call_args.args[0][-1],
                         "(Get-Acl -LiteralPath 'C:\\users\\O''Hara\\project').Sddl")
        environment = run.call_args.kwargs['env']
        self.assertNotIn('PSMODULEPATH', [name.upper() for name in environment])
        self.assertEqual([value for name, value in environment.items() if name.upper() == 'SYSTEMROOT'],
                         [r'C:\Windows'])
        self.assertNotIn('timeout', run.call_args.kwargs)


class GateTests(unittest.TestCase):
    def test_empty_or_unfinished_checks_refuse(self):
        for checks in [[], [Check('probe', 'untested')], [Check('probe', 'fail')]]:
            with self.subTest(checks=checks), self.assertRaises(RuntimeIsolationError):
                runtime.require_checks(checks)

    def test_missing_ledger_refuses_before_inventory(self):
        with mock.patch.object(runtime.state, 'load_ledger', return_value={}), \
                mock.patch.object(runtime.windows_verify, 'verify_workspace') as verify:
            with self.assertRaisesRegex(RuntimeIsolationError, 'run loki-setup'):
                runtime.configured_workspace([])
        verify.assert_not_called()

    def test_recorded_workspace_is_checked(self):
        workspace = os.getcwd()
        entry = {'profile': runtime.state.profile_name_for(workspace),
                 'grants': [{'path': workspace, 'access': 'read-write',
                             'origin': 'workspace'}]}
        ledger = {'workspaces': {runtime.state.workspace_key(workspace): entry}}
        with mock.patch.object(runtime.state, 'load_ledger', return_value=ledger), \
                mock.patch.object(runtime.windows_verify, 'verify_workspace',
                                  return_value=[Check('inventory', 'pass')]) as verify:
            self.assertEqual(runtime.configured_workspace(['--shell-cwd', workspace]),
                             workspace)
        verify.assert_called_once_with(ledger, workspace)

    def test_required_workspace_gates_the_same_ledger(self):
        # The ACP front resolves the workspace from the session cwd, not a
        # command line; the gate it passes must be the same one.
        workspace = os.getcwd()
        entry = {'profile': runtime.state.profile_name_for(workspace),
                 'grants': [{'path': workspace, 'access': 'read-write',
                             'origin': 'workspace'}]}
        ledger = {'workspaces': {runtime.state.workspace_key(workspace): entry}}
        with mock.patch.object(runtime.state, 'load_ledger', return_value=ledger), \
                mock.patch.object(runtime.windows_verify, 'verify_workspace',
                                  return_value=[Check('inventory', 'pass')]) as verify:
            self.assertEqual(runtime.required_workspace(workspace), workspace)
        verify.assert_called_once_with(ledger, workspace)

    def test_required_workspace_keeps_the_raw_operand_and_one_selected_identity(self):
        workspace = 'relative/link/../workspace//'
        key = 'native-directory-identity'
        entry = {'profile': runtime.state.profile_name_for_key(key),
                 'grants': [{'path': workspace, 'access': 'read-write', 'origin': 'workspace'}]}
        ledger = {'workspaces': {key: entry}}
        with mock.patch.object(runtime.state, '_directory_identity', return_value=key) as selected, \
                mock.patch.object(runtime.state, 'load_ledger', return_value=ledger), \
                mock.patch.object(runtime.windows_verify, 'verify_workspace',
                                  return_value=[Check('inventory', 'pass')]) as verify:
            self.assertEqual(runtime.required_workspace(workspace), workspace)
        selected.assert_called_once_with(workspace)
        verify.assert_called_once_with(ledger, workspace)

    def test_required_workspace_without_ledger_refuses(self):
        with mock.patch.object(runtime.state, 'load_ledger', return_value={}), \
                mock.patch.object(runtime.windows_verify, 'verify_workspace') as verify:
            with self.assertRaisesRegex(RuntimeIsolationError, 'run loki-setup'):
                runtime.required_workspace(os.getcwd())
        verify.assert_not_called()

    def test_runtime_requires_expected_workspace(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(runtime.windows_verify, 'probe_containment') as probe:
            with self.assertRaises(RuntimeIsolationError):
                runtime.verify_runtime()
        probe.assert_not_called()

    def test_runtime_requires_expected_package_sid(self):
        # The child compares its token against the SID the launcher verified it
        # against; with no SID there is nothing to compare, so it refuses
        # rather than deriving one (which the container cannot do).
        with mock.patch.dict(
                os.environ, {runtime.WORKSPACE_ENV: '/work'}, clear=True), \
                mock.patch.object(runtime.windows_verify, 'probe_containment') as probe:
            with self.assertRaisesRegex(
                    RuntimeIsolationError, 'package SID'):
                runtime.verify_runtime()
        probe.assert_not_called()

    def test_runtime_runs_probe_and_refuses_denial_failure(self):
        with mock.patch.dict(
                os.environ, {runtime.WORKSPACE_ENV: '/work',
                             runtime.PACKAGE_ENV: 'S-1-15-2-1-2-3'}), \
                mock.patch.object(runtime.windows_verify, 'probe_containment',
                                  return_value=[Check('credentials', 'fail')]) as probe:
            with self.assertRaises(RuntimeIsolationError):
                runtime.verify_runtime()
        probe.assert_called_once_with('/work', 'S-1-15-2-1-2-3')


class ScratchTests(unittest.TestCase):
    """Scratch lives in the workspace, and startup recreates it."""

    def test_startup_creates_and_recreates_the_directory(self):
        with tempfile.TemporaryDirectory() as workspace:
            directory = runtime.scratch_directory(workspace)
            self.assertEqual(
                directory, os.path.join(workspace, '.loki', 'tmp'))
            with mock.patch.dict(
                    os.environ, {runtime.WORKSPACE_ENV: workspace}):
                runtime.ensure_runtime_temp()
                self.assertTrue(os.path.isdir(directory))
                # The tools that use it may delete it between runs.
                shutil.rmtree(directory)
                runtime.ensure_runtime_temp()
                self.assertTrue(os.path.isdir(directory))
            # Startup never removes it afterwards.
            self.assertTrue(os.path.isdir(directory))

    def test_startup_requires_the_launch_workspace(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                    RuntimeIsolationError, 'launch workspace'):
                runtime.ensure_runtime_temp()

    def test_startup_fails_closed_when_the_directory_cannot_be_made(self):
        with mock.patch.dict(
                os.environ, {runtime.WORKSPACE_ENV: '/work'}), \
                mock.patch.object(runtime.os, 'makedirs',
                                  side_effect=OSError('denied')):
            with self.assertRaisesRegex(
                    RuntimeIsolationError, 'scratch directory'):
                runtime.ensure_runtime_temp()


class EntrypointTests(unittest.TestCase):
    def test_failed_isolation_stops_before_process_protection(self):
        from loki_agent import __main__ as entry
        with mock.patch.object(
                entry.runtime_isolation, 'isolate_runtime',
                side_effect=RuntimeIsolationError('denied')) as isolate, \
                mock.patch.object(entry, 'protect_credential_process') as protect:
            with self.assertRaises(RuntimeIsolationError):
                entry._protect_runtime()
        isolate.assert_called_once_with(None)
        protect.assert_not_called()

    def test_isolation_precedes_process_protection(self):
        from loki_agent import __main__ as entry
        calls = []

        def isolate(directory=None):
            calls.append('isolate')

        def scratch():
            calls.append('scratch')

        def protect():
            calls.append('protect')
            return True

        with mock.patch.object(entry.runtime_isolation, 'isolate_runtime',
                               side_effect=isolate), \
                mock.patch.object(entry.runtime_isolation,
                                  'prepare_runtime_scratch',
                                  side_effect=scratch), \
                mock.patch.object(entry, 'protect_credential_process',
                                  side_effect=protect):
            entry._protect_runtime()

        self.assertEqual(calls, ['isolate', 'scratch', 'protect'])


@unittest.skipUnless(sys.platform == "win32",
                     "Windows selection inside the runtime_isolation seam")
class IsolationSeamTests(unittest.TestCase):
    def test_isolate_runtime_verifies_the_token(self):
        from loki_agent import runtime_isolation
        with mock.patch.object(runtime_isolation.windows_runtime,
                               'verify_runtime') as verify:
            runtime_isolation.isolate_runtime()
        verify.assert_called_once_with()

    def test_verify_contained_runtime_rechecks_the_token(self):
        from loki_agent import runtime_isolation
        with mock.patch.object(runtime_isolation.windows_runtime,
                               'verify_runtime') as verify:
            runtime_isolation.verify_contained_runtime()
        verify.assert_called_once_with()

    def test_prepare_runtime_scratch_creates_the_workspace_directory(self):
        from loki_agent import runtime_isolation
        with mock.patch.object(runtime_isolation.windows_runtime,
                               'ensure_runtime_temp') as ensure:
            runtime_isolation.prepare_runtime_scratch()
        ensure.assert_called_once_with()

    def test_runtime_cwd_is_the_supervisor_cwd_not_the_workspace(self):
        # The terminal runtime's ambient cwd is inherited from the
        # supervisor, as the POSIX spawn inherits it; the workspace crosses
        # only as the container key, never as the process's directory.  This is
        # the Windows side of the cwd half of the isolation property; the POSIX
        # side is
        # test_runtime_isolations.LinuxIsolationTests
        # .test_runtime_rebinds_cwd_through_credential_cover, where a mount
        # cover is what makes relative opens miss the credential tree.
        import asyncio

        from loki_agent import runtime_isolation

        class Delegation:
            owner_child = (7,)
            credential_child = (9,)

            def child_arguments(self):
                return []

        with mock.patch.object(runtime_isolation.host_ipc, 'handles',
                               side_effect=lambda end: list(end)), \
                mock.patch.object(runtime_isolation.windows_runtime,
                                  'launch', return_value=mock.Mock()) as launch:
            asyncio.run(runtime_isolation.start_runtime(
                '/installed/loki', ['--headless'], '/recorded/work',
                {'SAFE': 'value'}, Delegation()))
        self.assertEqual(launch.call_args.args[0], '/installed/loki')
        self.assertEqual(launch.call_args.args[1],
                         ['--runtime', '--', '--headless'])
        self.assertEqual(launch.call_args.args[3], '/recorded/work')
        # ``environment`` is positional too: launch(executable, arguments,
        # environment, workspace, inherited, ...).
        self.assertEqual(launch.call_args.args[2], {'SAFE': 'value'})
        self.assertEqual(launch.call_args.kwargs['current_directory'],
                         os.getcwd())

    def test_worker_cwd_is_the_front_cwd_not_the_session_workspace(self):
        # The invariant this pins: the worker's ACTUAL cwd is inherited from
        # the front, exactly as the POSIX spawn inherits it by omission.  The
        # session cwd gates and keys the container; it must never become the
        # ambient directory the contained process resolves against.
        import asyncio

        from loki_agent import host_ipc
        from loki_agent import runtime_isolation
        from loki_agent import windows_subprocesses

        class Delegation:
            owner_child = (7,)
            credential_child = (9,)

            def child_arguments(self):
                return ['--session-owner-fd', 'r=7']

        with mock.patch.object(runtime_isolation.windows_runtime,
                               'required_workspace',
                               return_value='/recorded/work') as gate, \
                mock.patch.object(host_ipc, 'handles',
                                  side_effect=lambda end: list(end)), \
                mock.patch.object(windows_subprocesses,
                                  'create_worker_process',
                                  new=mock.AsyncMock(return_value='worker')) as spawn:
            worker = asyncio.run(runtime_isolation.start_worker(
                '/session/cwd', {'SAFE': 'value'}, Delegation()))
        gate.assert_called_once_with('/session/cwd')
        self.assertEqual(worker, 'worker')
        # The workspace crosses as the container key only; the ambient cwd is
        # the front's, exactly as the POSIX spawn inherits it by omission.
        spawn.assert_awaited_once_with(
            workspace='/recorded/work',
            environment={'SAFE': 'value'},
            arguments=['--worker', '--session-owner-fd', 'r=7'],
            inherited_handles=[7, 9],
            current_directory=os.getcwd())


class LaunchTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, contained, default_installed=True, capabilities=None, stdio=None,
                       retirement=None, creation_failure=False):
        events = []
        information = api.ProcessInformation(11, 12, 13, 14)

        def private_default(*args):
            events.append('private-default')
            if not default_installed:
                raise api.WindowsApiError('private default refused')
            return b'private-acl'

        def duplicate(source_process, source, target, output, rights, inherit, flags):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = source + 100
            return 1

        def native_bind(library, symbol, *signature):
            if symbol == 'CreateJobObjectW':
                return lambda *args: 21
            if symbol == 'DuplicateHandle':
                return duplicate
            if symbol == 'WaitForSingleObject':
                return lambda *args: 258 if retirement is not None and not retirement.exited.is_set() else 0

            def call(*args):
                events.append(symbol)
                return 1
            return call

        def create_child(*args, **options):
            output = options['information']
            ctypes.memmove(ctypes.byref(output), ctypes.byref(information), ctypes.sizeof(information))
            if creation_failure:
                raise KeyboardInterrupt('interrupted native buffer cleanup')
            return output

        def terminate(*args):
            events.append('terminate')
            if retirement is not None:
                retirement.terminated.set()

        def closed(handle):
            if retirement is not None and handle in [11, 12]:
                self.assertTrue(retirement.exited.is_set(), 'released an unobserved child')
                retirement.closed.append(handle)

        with mock.patch.dict('sys.modules', {'msvcrt': types.SimpleNamespace(
                get_osfhandle=lambda fd: fd + 30)}), \
                mock.patch.object(api, 'bind', side_effect=native_bind), \
                mock.patch.object(api, 'derive_app_container_sid', return_value='package'), \
                mock.patch.object(api, 'current_process_handle', return_value=1), \
                mock.patch.object(api, 'current_user_sid', return_value='S-1-5-21-1'), \
                mock.patch.object(api, 'set_token_default_dacl', side_effect=private_default), \
                mock.patch.object(api, 'create_process_in_app_container',
                                  side_effect=create_child) as create, \
                mock.patch.object(api, 'open_process_token', return_value=22), \
                mock.patch.object(api, 'token_is_app_container', return_value=contained), \
                mock.patch.object(api, 'token_app_container_sid', return_value='package'), \
                mock.patch.object(api, 'token_capabilities', return_value=(
                    [[sid, api.SE_GROUP_ENABLED] for sid in sorted(runtime._NETWORK_CAPABILITIES)]
                    if capabilities is None else capabilities)), \
                mock.patch.object(api, 'resume_thread',
                                  side_effect=lambda *args: events.append('resume')), \
                mock.patch.object(api, 'terminate_process',
                                  side_effect=terminate), \
                mock.patch.object(api, 'close_handle', side_effect=closed) as close:
            success = contained and default_installed and capabilities is None and not creation_failure
            if success:
                process = await runtime.launch(
                    'loki.py', ['--runtime'], {'SAFE': 'value'},
                    '/work', [40, 41], stdio=stdio, current_directory='/work')
                await process.wait()
                process.close()
            else:
                failure = (KeyboardInterrupt if creation_failure else RuntimeIsolationError
                           if not contained or capabilities is not None else api.WindowsApiError)
                with self.assertRaises(failure):
                    await runtime.launch(
                        'loki.py', ['--runtime'], {'SAFE': 'value'},
                        '/work', [40, 41], stdio=stdio, current_directory='/work')
            scratch = os.path.join('/work', runtime.SCRATCH_DIRECTORY)
            expected = {
                'SAFE': 'value', runtime.WORKSPACE_ENV: '/work',
                runtime.PACKAGE_ENV: 'package',
                'TEMP': scratch, 'TMP': scratch}
            # The AppContainer start resolves the package's state under
            # %LOCALAPPDATA% from the block it is given, so the launch supplies
            # it when the caller's environment does not (measured on Windows:
            # tests/test_windows_runtime.LaunchEnvironmentTests).
            if os.environ.get('LOCALAPPDATA'):
                expected['LOCALAPPDATA'] = os.environ['LOCALAPPDATA']
            self.assertEqual(create.call_args.kwargs['environment'], expected)
            self.assertEqual(create.call_args.kwargs['current_directory'],
                             '/work')
            self.assertEqual(create.call_args.kwargs['capability_sids'],
                             sorted(runtime._NETWORK_CAPABILITIES))
            self.assertEqual(create.call_args.kwargs['object_security'],
                             'D:P(A;;GA;;;S-1-5-21-1)(A;;GA;;;SY)(A;;GA;;;BA)')
            self.assertEqual(create.call_args.kwargs['inherited_handles'],
                             [40, 41, 130, 131, 132] if stdio is None else [40, 41, 132, *stdio])
            self.assertEqual(create.call_args.kwargs['standard_handles'],
                             [130, 131, 132] if stdio is None else [*stdio, 132])
            # Stdio remains borrowed; only our duplicate and native owners close.
            self.assertCountEqual([call.args[0] for call in close.call_args_list],
                                  [*([22] if not creation_failure else []),
                                   *([130, 131, 132] if stdio is None else [132]), 21, 12, 11])
            self.assertIn(mock.call(11), close.call_args_list)
            self.assertIn(mock.call(12), close.call_args_list)
            self.assertIn(mock.call(21), close.call_args_list)
        return events

    async def test_verified_child_is_assigned_before_resume(self):
        events = await self.exercise(True)
        self.assertLess(events.index('private-default'), events.index('AssignProcessToJobObject'))
        self.assertLess(events.index('AssignProcessToJobObject'), events.index('resume'))
        self.assertNotIn('terminate', events)

    async def test_wrong_token_is_terminated_without_resume(self):
        events = await self.exercise(False)
        self.assertIn('terminate', events)
        self.assertNotIn('resume', events)

    async def test_missing_capabilities_are_terminated_without_resume(self):
        events = await self.exercise(True, capabilities=[])
        self.assertIn('terminate', events)
        self.assertNotIn('resume', events)

    async def test_extra_capabilities_are_terminated_without_resume(self):
        enabled = [[sid, api.SE_GROUP_ENABLED] for sid in sorted(runtime._NETWORK_CAPABILITIES)]
        events = await self.exercise(True, capabilities=enabled + [['unexpected-capability', api.SE_GROUP_ENABLED]])
        self.assertIn('terminate', events)
        self.assertNotIn('resume', events)

    async def test_disabled_capabilities_are_terminated_without_resume(self):
        events = await self.exercise(True, capabilities=[[sid, 0] for sid in sorted(runtime._NETWORK_CAPABILITIES)])
        self.assertIn('terminate', events)
        self.assertNotIn('resume', events)

    async def test_private_default_failure_terminates_without_resume(self):
        events = await self.exercise(True, default_installed=False)
        self.assertIn('terminate', events)
        self.assertNotIn('resume', events)

    async def test_piped_child_receives_caller_handles_and_duplicated_stderr(self):
        await self.exercise(True, stdio=(50, 51))

    async def _held_failed_launch(self, creation_failure=False):
        retirement = types.SimpleNamespace(terminated=asyncio.Event(), exited=asyncio.Event(), closed=[])
        task = asyncio.create_task(self.exercise(False, retirement=retirement, creation_failure=creation_failure))
        try:
            await retirement.terminated.wait()
            self.assertEqual(retirement.closed, [])
            task.cancel()
            retirement.exited.set()
            events = await task
            self.assertEqual(retirement.closed, [12, 11])
            self.assertNotIn('resume', events)
        finally:
            retirement.exited.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_failed_initial_launch_retains_ownership_through_later_cancellation(self):
        await self._held_failed_launch()

    async def test_interrupted_native_cleanup_uses_the_same_owned_retirement(self):
        await self._held_failed_launch(creation_failure=True)


class LocalLaunchTests(unittest.TestCase):
    @contextmanager
    def calls(self, owning, mismatch=None):
        original, compatible, private = b'original-acl', b'compatible-acl', b'private-acl'
        info = api.ProcessInformation(101, 102, 103, 104)
        seen = {'sets': [], 'closed': [], 'job_events': [], 'ui_flags': 0}

        def job_call(library, name, *signature):
            if name == 'CreateJobObjectW':
                return lambda *args: 88
            if name == 'AssignProcessToJobObject':
                def assign(job, process):
                    seen['job_events'].append(['assign', job, process])
                    if seen.get('fail_assignment'):
                        raise OSError('assignment refused')
                    return 1
                return assign
            if name == 'SetInformationJobObject':
                def restrict(job, kind, value, size):
                    seen['ui_flags'] = ctypes.cast(value, ctypes.POINTER(ctypes.c_uint32))[0]
                    seen['job_events'].append(['restrict', job, kind, seen['ui_flags']])
                    return 1
                return restrict
            if name == 'QueryInformationJobObject':
                def query(job, kind, output, size, length):
                    ctypes.cast(output, ctypes.POINTER(ctypes.c_uint32))[0] = seen.get('reported_ui_flags', seen['ui_flags'])
                    return 1
                return query
            if name == 'IsProcessInJob':
                def member(process, job, output):
                    ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))[0] = process == 101 or seen.get('writer_in_leaf', False)
                    return 1
                return member
            raise AssertionError('unexpected native call: ' + name)

        role = runtime.state.tool_role_sid('S-1-15-2-1')
        original_sids = [['S-1-5-21-2', 0]]
        filtered_sids = [*original_sids, [role, 7]]
        values = {
            'bind': job_call,
            'current_process_handle': lambda: 99,
            'open_process_token': mock.Mock(side_effect=[44, 55, 66]),
            'current_user_sid': lambda: 'S-1-5-21-1',
            'token_is_app_container': lambda token: not (mismatch == 'container' and token == 66),
            'token_app_container_sid': lambda token: 'S-1-15-2-9' if mismatch == 'package' and token == 66 else 'S-1-15-2-1',
            'token_integrity_level': lambda token: 'S-1-16-4096',
            'token_restricting_sids': lambda token: original_sids if owning or token == 44 else filtered_sids,
            'token_capabilities': lambda token: [] if token == 66 and mismatch == 'capabilities' else [
                [sid, api.SE_GROUP_ENABLED] for sid in sorted(runtime._NETWORK_CAPABILITIES)],
            'token_logon_sid': lambda token: 'S-1-5-5-0-99',
            'create_write_restricted_token': mock.Mock(return_value=77),
            'token_default_dacl': lambda token: original if token == 44 else private if token == 66 and owning else compatible,
            'duplicate_primary_token': lambda *args: 77,
            'acl_sddl': lambda acl: 'D:existing-grants',
            'set_token_default_dacl': lambda *args: seen['sets'].append(list(args)) or (private if args[0] == 66 else compatible),
            'close_handle': seen['closed'].append,
            'create_process_suspended': mock.Mock(return_value=info),
            'resume_thread': mock.Mock(),
        }
        with ExitStack() as cleanup:
            cleanup.enter_context(mock.patch.object(runtime.os, 'getcwd', return_value='/trusted'))
            for name, function in values.items():
                cleanup.enter_context(mock.patch.object(api, name, side_effect=function))
            seen['create'] = api.create_process_suspended
            seen['filter'] = api.create_write_restricted_token
            seen['resume'] = api.resume_thread
            yield seen

    def test_roles_preserve_cwd_image_restrictions_and_parent_default(self):
        for owning in [True, False]:
            with self.subTest(owning=owning), self.calls(owning) as seen:
                process = runtime.ContainedProcess(None)
                cwd = '/trusted' if owning else '/workspace'
                runtime.prepare_local(process, ['/chosen/image.exe', '--subagent'], {'SAFE': 'yes'},
                                      cwd, [21, 22, 23, 31], [21, 22, 23], owning=owning)
                self.assertEqual(process.pid, 103)
                call = seen['create'].call_args
                self.assertEqual(list(call.args), ['/chosen/image.exe', ['--subagent']])
                self.assertEqual(call.kwargs['current_directory'], cwd)
                self.assertEqual(call.kwargs['primary_token'], 77)
                self.assertFalse(call.kwargs['search_path'])
                self.assertEqual(call.kwargs['inherited_handles'], [21, 22, 23, 31])
                compatible = 'D:existing-grants(A;;GA;;;S-1-15-2-1)'
                if not owning:
                    role = runtime.state.tool_role_sid('S-1-15-2-1')
                    compatible += f'(A;;GA;;;{role})'
                    self.assertEqual(seen['filter'].call_args.args[0], 55)
                    self.assertEqual(seen['filter'].call_args.args[1],
                                     [role, *runtime.state.TOOL_COMPATIBILITY_SIDS, 'S-1-5-5-0-99'])
                else:
                    seen['filter'].assert_not_called()
                self.assertEqual(seen['sets'][0], [77, compatible])
                private = 'D:P(A;;GA;;;S-1-5-21-1)(A;;GA;;;SY)(A;;GA;;;BA)'
                self.assertEqual(seen['sets'][1:], [[66, private]] if owning else [])
                self.assertNotIn(44, [entry[0] for entry in seen['sets']])
                self.assertCountEqual(seen['closed'], [44, 55, 66, 77])
                seen['resume'].assert_not_called()

    def test_generic_role_policy_does_not_replace_the_callers_explicit_directory(self):
        # Loki's trusted call site supplies its process cwd. This generic layer
        # must forward that value, not derive another directory from the role.
        with self.calls(True) as seen:
            runtime.prepare_local(runtime.ContainedProcess(None), ['/chosen/loki.exe'], {},
                                  '/explicit-trusted-directory', [], [1, 2, 3], owning=True)
            self.assertEqual(seen['create'].call_args.kwargs['current_directory'], '/explicit-trusted-directory')

    def test_identity_failure_retains_the_suspended_child_for_async_retirement(self):
        self._identity_failure(False, 'package')

    def test_only_bare_external_commands_use_windows_command_lookup(self):
        """Qualification chooses a native launch parameter; it must never rewrite the image."""
        cases = [
            ['cmd.exe', True],
            [r'.\cmd.exe', False],
            ['tools/cmd.exe', False],
            [r'C:cmd.exe', False],
            [r'C:\tools\cmd.exe', False],
            [r'\tools\cmd.exe', False],
            [r'\\server\share\cmd.exe', False],
        ]
        for image, search in cases:
            for owning in [False, True]:
                with self.subTest(image=image, owning=owning), self.calls(owning) as seen:
                    runtime.prepare_local(runtime.ContainedProcess(None), [image, '/c', 'exit'], {},
                                          '/workspace', [], [1, 2, 3], owning=owning)
                    self.assertEqual(seen['create'].call_args.args[0], image)
                    self.assertEqual(seen['create'].call_args.kwargs['search_path'],
                                     search and not owning)

    def _identity_failure(self, owning, mismatch):
        with self.calls(owning, mismatch=mismatch) as seen:
            process = runtime.ContainedProcess(None)
            with self.assertRaises(api.WindowsApiError):
                runtime.prepare_local(process, ['/chosen/loki.exe'], {}, '/trusted', [], [1, 2, 3],
                                      owning=owning)
            self.assertEqual(process.information.hProcess, 101)
            self.assertNotIn(101, seen['closed'])
            self.assertNotIn(102, seen['closed'])
            seen['resume'].assert_not_called()
            if mismatch == 'package':
                self.assertTrue(api.token_is_app_container(66))
                self.assertNotEqual(api.token_app_container_sid(66), api.token_app_container_sid(44))

    def test_owning_child_package_mismatch_retains_the_suspended_child(self):
        self._identity_failure(True, 'package')

    def test_tool_child_capability_mismatch_retains_the_suspended_child(self):
        self._identity_failure(False, 'capabilities')

    def test_owning_child_capability_mismatch_retains_the_suspended_child(self):
        self._identity_failure(True, 'capabilities')

    def test_tool_child_losing_appcontainer_identity_retains_the_suspended_child(self):
        self._identity_failure(False, 'container')

    def test_owning_child_losing_appcontainer_identity_retains_the_suspended_child(self):
        self._identity_failure(True, 'container')

    def test_only_ordinary_children_receive_verified_ui_limits_before_execution(self):
        # Local creation must keep the writer out of the child's UI domain.
        # This portable contract checks ordering; the native fixture separately
        # checks exact enclosing-job membership and actual window operations.
        for owning in [True, False]:
            with self.subTest(owning=owning), self.calls(owning) as seen:
                process = runtime.ContainedProcess(None)
                try:
                    runtime.prepare_local(
                        process, ['/chosen/image.exe'], {}, '/workspace', [], [1, 2, 3],
                        owning=owning)
                    self.assertEqual(seen['job_events'], [['assign', 88, 101]] if owning else [
                        ['assign', 88, 101], ['restrict', 88, 4, 1]])
                    seen['resume'].assert_not_called()
                finally:
                    process.close()

    def test_ui_assignment_failure_keeps_the_child_available_for_retirement(self):
        with self.calls(False) as seen:
            seen['fail_assignment'] = True
            process = runtime.ContainedProcess(None)
            try:
                with self.assertRaisesRegex(OSError, 'assignment refused'):
                    runtime.prepare_local(
                        process, ['/chosen/image.exe'], {}, '/workspace', [], [1, 2, 3],
                        owning=False)
                # An empty leaf job cannot terminate the suspended child.
                # Close it locally; the factory must still retire the child.
                self.assertIsNone(process.job)
                self.assertEqual(seen['closed'].count(88), 1)
                self.assertEqual(process.information.hProcess, 101)
                self.assertNotIn(101, seen['closed'])
                seen['resume'].assert_not_called()
            finally:
                process.close()

    def test_ui_readback_or_writer_membership_mismatch_never_publishes_the_job(self):
        for failure in [{'reported_ui_flags': 0}, {'writer_in_leaf': True}]:
            with self.subTest(failure=failure), self.calls(False) as seen:
                seen.update(failure)
                process = runtime.ContainedProcess(None)
                try:
                    with self.assertRaisesRegex(api.WindowsApiError, 'UI boundary mismatch'):
                        runtime.prepare_local(
                            process, ['/chosen/image.exe'], {}, '/workspace', [], [1, 2, 3],
                            owning=False)
                    self.assertIsNone(process.job)
                    self.assertEqual(seen['closed'].count(88), 1)
                    self.assertEqual(process.information.hProcess, 101)
                    self.assertNotIn(101, seen['closed'])
                    seen['resume'].assert_not_called()
                finally:
                    process.close()


class FamilyAccountingTests(unittest.TestCase):
    def _close_failure(self, failed_handle):
        owner = runtime.ContainedProcess(api.ProcessInformation(71, 72, 73, 0), job=88)
        failure = api.WindowsApiError('CloseHandle failed', status=6)
        attempted = []

        def close(handle):
            attempted.append(handle)
            if handle == failed_handle:
                raise failure

        with mock.patch.object(api, 'close_handle', side_effect=close):
            with self.assertRaises(api.WindowsApiError) as caught:
                owner.close()
            self.assertIs(caught.exception, failure)
            self.assertEqual(attempted, [88, 72, 71])
            self.assertIsNone(owner.job)
            self.assertFalse(owner.information.hThread)
            self.assertFalse(owner.information.hProcess)
            owner.close()
        self.assertEqual(attempted, [88, 72, 71])

    def test_family_close_failure_still_attempts_both_leader_handles_once(self):
        self._close_failure(88)

    def test_thread_close_failure_still_attempts_process_release_once(self):
        self._close_failure(72)

    def test_leader_release_retains_the_original_family_and_cached_result(self):
        owner = runtime.ContainedProcess(api.ProcessInformation(71, 72, 73, 0), job=88)
        owner.returncode = 2
        with mock.patch.object(api, 'close_handle') as close:
            owner.release_leader()
            self.assertEqual(owner.job, 88)
            self.assertEqual(owner.returncode, 2)
            self.assertFalse(owner.information.hProcess)
            self.assertFalse(owner.information.hThread)
            owner.close_family()
            owner.close()
        self.assertEqual(close.call_args_list, [mock.call(72), mock.call(71), mock.call(88)])

    def test_family_release_does_not_close_a_live_leader(self):
        owner = runtime.ContainedProcess(api.ProcessInformation(71, 72, 73, 0), job=88)
        with mock.patch.object(api, 'close_handle') as close:
            owner.close_family()
        close.assert_called_once_with(88)
        self.assertIsNone(owner.job)
        self.assertIsNone(owner.returncode)
        self.assertEqual([owner.information.hProcess, owner.information.hThread], [71, 72])

    def test_active_membership_not_leader_exit_determines_liveness(self):
        def query(job, kind, output, size, length):
            self.assertEqual([job, kind, size], [88, 1, 48])
            output._obj.active = active
            return True

        for active in [0, 2]:
            with self.subTest(active=active), mock.patch.object(api, 'bind', return_value=query):
                self.assertEqual(runtime.job_has_processes(88), bool(active))
        with mock.patch.object(api, 'bind', return_value=lambda *args: False), \
                mock.patch.object(ctypes, 'get_last_error', return_value=5, create=True):
            with self.assertRaises(api.WindowsApiError):
                runtime.job_has_processes(88)  # Query denial is not family absence.


class LocalRetirementTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_joins_exit_before_closing_handles(self):
        entered, exited = asyncio.Event(), asyncio.Event()
        events = []
        process = mock.Mock(information=api.ProcessInformation(71, 72, 73, 0))

        async def wait():
            entered.set()
            await exited.wait()
            events.append('exit')
            return 1

        process.wait = wait
        process.terminate.side_effect = lambda: events.append('terminate')
        process.close.side_effect = lambda: events.append('close')
        task = asyncio.create_task(runtime.retire_local(process))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        self.assertEqual(events, ['terminate'])
        exited.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(events, ['terminate', 'exit', 'close'])


@unittest.skipUnless(sys.platform == "win32",
                     "the contained launch is Windows-only")
class LaunchEnvironmentTests(unittest.TestCase):
    """Reproduce the contained launch with controlled environments.

    The pty tests fail ``CreateProcessW failed: 203`` when the TUI front starts
    its runtime, and the environment that launch is given is the suspect.  This
    runs the real ``windows_runtime.launch`` with the reduced environment those
    tests use, plus one variant adding each variable the broker restores from
    its profile, and records start/winerror per variant.  Recorded, not
    asserted: a failure here is the measurement, and the assertion belongs to
    whatever fix follows.
    """

    def test_environment_variants(self):
        import json
        from loki_entrypoints import child_environment, configure_container
        record = {"probe": "launch-environment-203"}
        with tempfile.TemporaryDirectory(prefix="loki-launch-env-") as root:
            workspace = os.path.join(root, "workspace")
            os.makedirs(workspace)
            reduced = child_environment(
                HOME=root,
                XDG_CONFIG_HOME=os.path.join(root, "config"),
                XDG_STATE_HOME=os.path.join(root, "state"),
                PATH=os.environ.get("PATH", ""))
            configure_container(reduced, workspace)
            variants = {"reduced": dict(reduced)}
            for name in ["USERPROFILE", "LOCALAPPDATA", "APPDATA",
                         "PROGRAMDATA", "PROGRAMFILES"]:
                if os.environ.get(name):
                    variants["with_" + name] = {**reduced, name: os.environ[name]}
            variants["full_process"] = dict(os.environ)
            for label, environment in variants.items():
                try:
                    process = runtime.launch(
                        sys.executable, ["-c", "pass"], environment,
                        workspace, [], current_directory=os.getcwd())
                except Exception as error:
                    record[label] = {
                        "started": False,
                        "winerror": getattr(error, "winerror", None),
                        "error": type(error).__name__,
                        "message": str(error)}
                else:
                    record[label] = {"started": True}
                    process.close()
        print(json.dumps(record), flush=True)
        self.assertTrue(record)
