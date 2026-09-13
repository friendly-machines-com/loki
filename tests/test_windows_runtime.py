"""Portable startup-gate tests; native launch still needs Windows validation."""

import ctypes
import os
import sys
import types
import unittest
from unittest import mock

from loki_agent import windows_runtime as runtime
from loki_agent import windows_api as api
from loki_agent.windows_state import Check
from loki_agent.runtime_isolations import RuntimeIsolationError


class GateTests(unittest.TestCase):
    def test_empty_or_unfinished_checks_refuse(self):
        for checks in ([], [Check('probe', 'untested')], [Check('probe', 'fail')]):
            with self.subTest(checks=checks), self.assertRaises(RuntimeIsolationError):
                runtime.require_checks(checks)

    def test_missing_ledger_refuses_before_inventory(self):
        with mock.patch.object(runtime.state, 'load_ledger', return_value={}), \
                mock.patch.object(runtime.windows_verify, 'verify_workspace') as verify:
            with self.assertRaisesRegex(RuntimeIsolationError, 'run loki-setup'):
                runtime.configured_workspace([])
        verify.assert_not_called()

    def test_recorded_workspace_is_checked(self):
        workspace = runtime.state.canonical_workspace(os.getcwd())
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

    def test_runtime_requires_expected_workspace(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(runtime.windows_verify, 'probe_containment') as probe:
            with self.assertRaises(RuntimeIsolationError):
                runtime.verify_runtime()
        probe.assert_not_called()

    def test_runtime_runs_probe_and_refuses_denial_failure(self):
        with mock.patch.dict(os.environ, {runtime.WORKSPACE_ENV: '/work'}), \
                mock.patch.object(runtime.windows_verify, 'probe_containment',
                                  return_value=[Check('credentials', 'fail')]) as probe:
            with self.assertRaises(RuntimeIsolationError):
                runtime.verify_runtime()
        probe.assert_called_once_with('/work')


class EntrypointTests(unittest.TestCase):
    def test_failed_isolation_stops_before_process_protection(self):
        from loki_agent import __main__ as entry
        with mock.patch.object(
                entry.runtime_isolation, 'isolate_runtime',
                side_effect=RuntimeIsolationError('denied')) as isolate, \
                mock.patch.object(entry, 'protect_credential_process') as protect:
            with self.assertRaises(RuntimeIsolationError):
                entry._protect_runtime()
        isolate.assert_called_once_with()
        protect.assert_not_called()

    def test_isolation_precedes_process_protection(self):
        from loki_agent import __main__ as entry
        calls = []

        def isolate():
            calls.append('isolate')

        def protect():
            calls.append('protect')
            return True

        with mock.patch.object(entry.runtime_isolation, 'isolate_runtime',
                               side_effect=isolate), \
                mock.patch.object(entry, 'protect_credential_process',
                                  side_effect=protect):
            entry._protect_runtime()

        self.assertEqual(calls, ['isolate', 'protect'])


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


class LaunchTests(unittest.TestCase):
    def exercise(self, contained):
        events = []
        information = api.ProcessInformation(11, 12, 13, 14)

        def duplicate(source_process, source, target, output, rights, inherit, flags):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = source + 100
            return 1

        def native_bind(library, symbol, *signature):
            if symbol == 'CreateJobObjectW':
                return lambda *args: 21
            if symbol == 'DuplicateHandle':
                return duplicate
            if symbol == 'WaitForSingleObject':
                return lambda *args: 0
            def call(*args):
                events.append(symbol)
                return 1
            return call

        with mock.patch.dict('sys.modules', {'msvcrt': types.SimpleNamespace(
                get_osfhandle=lambda fd: fd + 30)}), \
                mock.patch.object(api, 'bind', side_effect=native_bind), \
                mock.patch.object(api, 'derive_app_container_sid', return_value='package'), \
                mock.patch.object(api, 'current_process_handle', return_value=1), \
                mock.patch.object(api, 'create_process_in_app_container',
                                  return_value=information) as create, \
                mock.patch.object(api, 'open_process_token', return_value=22), \
                mock.patch.object(api, 'token_is_app_container', return_value=contained), \
                mock.patch.object(api, 'token_app_container_sid', return_value='package'), \
                mock.patch.object(api, 'resume_thread',
                                  side_effect=lambda *args: events.append('resume')), \
                mock.patch.object(api, 'terminate_process',
                                  side_effect=lambda *args: events.append('terminate')), \
                mock.patch.object(api, 'close_handle') as close:
            if contained:
                process = runtime.launch('loki.py', ['--runtime'], {'SAFE': 'value'},
                                         '/work', [40, 41])
                process.close()
            else:
                with self.assertRaises(RuntimeIsolationError):
                    runtime.launch('loki.py', ['--runtime'], {'SAFE': 'value'},
                                   '/work', [40, 41])
            self.assertEqual(create.call_args.kwargs['environment'],
                             {'SAFE': 'value', runtime.WORKSPACE_ENV: '/work'})
            self.assertEqual(create.call_args.kwargs['inherited_handles'],
                             [40, 41, 130, 131, 132])
            self.assertIn(mock.call(11), close.call_args_list)
            self.assertIn(mock.call(12), close.call_args_list)
            self.assertIn(mock.call(21), close.call_args_list)
        return events

    def test_verified_child_is_assigned_before_resume(self):
        events = self.exercise(True)
        self.assertLess(events.index('AssignProcessToJobObject'), events.index('resume'))
        self.assertNotIn('terminate', events)

    def test_wrong_token_is_terminated_without_resume(self):
        events = self.exercise(False)
        self.assertIn('terminate', events)
        self.assertNotIn('resume', events)
