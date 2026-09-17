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

    def test_required_workspace_gates_the_same_ledger(self):
        # The ACP front resolves the workspace from the session cwd, not a
        # command line; the gate it passes must be the same one.
        workspace = runtime.state.canonical_workspace(os.getcwd())
        entry = {'profile': runtime.state.profile_name_for(workspace),
                 'grants': [{'path': workspace, 'access': 'read-write',
                             'origin': 'workspace'}]}
        ledger = {'workspaces': {runtime.state.workspace_key(workspace): entry}}
        with mock.patch.object(runtime.state, 'load_ledger', return_value=ledger), \
                mock.patch.object(runtime.windows_verify, 'verify_workspace',
                                  return_value=[Check('inventory', 'pass')]) as verify:
            self.assertEqual(runtime.required_workspace(workspace), workspace)
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

    def test_runtime_cwd_is_the_supervisor_cwd_not_the_workspace(self):
        # The terminal runtime's ambient cwd is inherited from the
        # supervisor, as the POSIX spawn inherits it; the workspace crosses
        # only as the container key, never as the process's directory.
        import asyncio

        from loki_agent import runtime_isolation

        class Delegation:
            owner_child = (7,)
            credential_child = (9,)

            def child_arguments(self):
                return []

        with mock.patch.object(runtime_isolation.host_ipc, 'handles',
                               side_effect=lambda end: tuple(end)), \
                mock.patch.object(runtime_isolation.windows_runtime,
                                  'launch', return_value=mock.Mock()) as launch:
            asyncio.run(runtime_isolation.start_runtime(
                '/installed/loki', ['--headless'], '/recorded/work',
                {'SAFE': 'value'}, Delegation()))
        self.assertEqual(launch.call_args.args[0], '/installed/loki')
        self.assertEqual(launch.call_args.args[1],
                         ['--runtime', '--', '--headless'])
        self.assertEqual(launch.call_args.args[3], '/recorded/work')
        self.assertEqual(launch.call_args.kwargs['environment'],
                         {'SAFE': 'value'})
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

        class Delegation:
            def child_arguments(self):
                return ['--session-owner-fd', 'r=7']

        front = (0x30, 0x31)
        child = (0x32, 0x33)

        with mock.patch.object(runtime_isolation.windows_runtime,
                               'required_workspace',
                               return_value='/recorded/work') as gate, \
                mock.patch.object(runtime_isolation.host_ipc,
                                  'worker_stdio',
                                  return_value=(front, child)) as pipes, \
                mock.patch.object(host_ipc, 'handles',
                                  side_effect=lambda end: tuple(end)), \
                mock.patch.object(host_ipc, 'WorkerStdio',
                                  return_value=mock.Mock()) as streams, \
                mock.patch.object(runtime_isolation.windows_runtime,
                                  'launch',
                                  return_value=mock.Mock()) as launch:
            worker = asyncio.run(runtime_isolation.start_worker(
                '/session/cwd', {'SAFE': 'value'}, Delegation()))
        gate.assert_called_once_with('/session/cwd')
        pipes.assert_called_once_with()
        streams.assert_called_once_with(*front)
        self.assertIs(worker._process, launch.return_value)
        self.assertEqual(launch.call_args.args[0], sys.argv[0])
        self.assertEqual(launch.call_args.args[1],
                         ['--worker', '--session-owner-fd', 'r=7'])
        self.assertEqual(launch.call_args.kwargs['stdio'], child)
        self.assertEqual(launch.call_args.kwargs['current_directory'],
                         os.getcwd())
        # The workspace crosses as the container key only.
        self.assertEqual(launch.call_args.args[3], '/recorded/work')


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
                                         '/work', [40, 41],
                                         current_directory='/work')
                process.close()
            else:
                with self.assertRaises(RuntimeIsolationError):
                    runtime.launch('loki.py', ['--runtime'], {'SAFE': 'value'},
                                   '/work', [40, 41],
                                   current_directory='/work')
            self.assertEqual(create.call_args.kwargs['environment'],
                             {'SAFE': 'value', runtime.WORKSPACE_ENV: '/work'})
            self.assertEqual(create.call_args.kwargs['current_directory'],
                             '/work')
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

    def exercise_stdio(self):
        """A piped child: caller-owned stdin/stdout, duplicated stderr only."""
        closed = []
        information = api.ProcessInformation(11, 12, 13, 14)

        def duplicate(source_process, source, target, output, rights, inherit, flags):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = source + 100
            return 1

        def native_bind(library, symbol, *signature):
            if symbol == 'CreateJobObjectW':
                return lambda *args: 21
            if symbol == 'DuplicateHandle':
                return duplicate
            return lambda *args: 1

        with mock.patch.dict('sys.modules', {'msvcrt': types.SimpleNamespace(
                get_osfhandle=lambda fd: fd + 30)}), \
                mock.patch.object(api, 'bind', side_effect=native_bind), \
                mock.patch.object(api, 'derive_app_container_sid', return_value='package'), \
                mock.patch.object(api, 'current_process_handle', return_value=1), \
                mock.patch.object(api, 'create_process_in_app_container',
                                  return_value=information) as create, \
                mock.patch.object(api, 'open_process_token', return_value=22), \
                mock.patch.object(api, 'token_is_app_container', return_value=True), \
                mock.patch.object(api, 'token_app_container_sid', return_value='package'), \
                mock.patch.object(api, 'resume_thread'), \
                mock.patch.object(api, 'close_handle',
                                  side_effect=closed.append):
            process = runtime.launch('loki.py', ['--worker'], {'SAFE': 'value'},
                                     '/work', [40, 41], stdio=(50, 51),
                                     current_directory=os.getcwd())
            process.close()
            self.assertEqual(create.call_args.kwargs['inherited_handles'],
                             [40, 41, 132, 50, 51])
            self.assertEqual(create.call_args.kwargs['standard_handles'],
                             (50, 51, 132))
            # The caller's cwd decision is forwarded unchanged: the launch
            # never substitutes the workspace (or anything else) for it.
            self.assertEqual(create.call_args.kwargs['current_directory'],
                             os.getcwd())
        # Only the launch's own handles are closed: the inspected token (22)
        # and its stderr duplicate (fd 2 -> 32 -> 132); the caller's stdio
        # handles 50 and 51 are not.  The process close then releases the
        # job, thread and process handles.
        self.assertEqual(closed, [22, 132, 21, 12, 11])

    def test_piped_child_receives_caller_handles_and_duplicated_stderr(self):
        self.exercise_stdio()
