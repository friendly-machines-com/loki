"""The same terminal ownership contracts against each host's native backend."""

import contextlib
import copy
import unittest
from unittest import mock

from loki_agent import terminals
from terminal_resource_fixtures import ModeEnvironment


class TerminalModeOwnershipTests(unittest.TestCase):
    def test_output_lifetime_survives_inner_input_mode_exit(self):
        with ModeEnvironment().installed() as environment:
            with terminals.terminal_output_mode():
                outer_state = copy.deepcopy(environment.state)
                with terminals.TerminalMode(123, enabled=True):
                    self.assertNotEqual(environment.state, outer_state)
                self.assertEqual(environment.state, outer_state)
            self.assertEqual(environment.state, environment.initial)

    def test_normal_exit_restores_every_changed_setting(self):
        with ModeEnvironment().installed() as environment:
            mode = terminals.TerminalMode(123, enabled=True)
            with mode:
                self.assertNotEqual(environment.state, environment.initial)
            self.assertEqual(environment.state, environment.initial)
            writes = list(environment.writes)
            mode.restore()
            self.assertEqual(environment.writes, writes)

    def test_partial_setup_rolls_back_at_each_native_setter(self):
        with ModeEnvironment().installed() as environment:
            for key in environment.initial:
                for position in ('before', 'after', 'reported'):
                    with self.subTest(setting=key, failure=position):
                        mode = terminals.TerminalMode(123, enabled=True)
                        environment.failures.add((key, 'setup', position))
                        try:
                            with self.assertRaises(OSError):
                                mode.__enter__()
                            self.assertEqual(environment.state, environment.initial)
                        finally:
                            environment.failures.clear()
                            mode.restore()

    def test_failed_restore_keeps_ownership_and_restores_other_settings(self):
        with ModeEnvironment().installed() as environment:
            for key in environment.initial:
                with self.subTest(setting=key):
                    mode = terminals.TerminalMode(123, enabled=True)
                    mode.__enter__()
                    environment.failures.add((key, 'restore', 'before'))
                    try:
                        with self.assertRaises(OSError):
                            mode.restore()
                        for other in environment.initial:
                            if other != key:
                                self.assertEqual(environment.state[other], environment.initial[other])
                    finally:
                        environment.failures.clear()
                        mode.restore()
                    self.assertEqual(environment.state, environment.initial)
                    writes = list(environment.writes)
                    mode.restore()
                    self.assertEqual(environment.writes, writes)

    def test_all_restore_failures_are_attempted_and_remain_visible(self):
        with ModeEnvironment().installed() as environment:
            mode = terminals.TerminalMode(123, enabled=True)
            mode.__enter__()
            environment.failures.update((key, 'restore', 'before') for key in environment.initial)
            try:
                with self.assertRaises(OSError) as caught:
                    mode.restore()
                messages = []
                error = caught.exception
                while error is not None:
                    messages.append(str(error))
                    error = error.__context__
                for key in environment.initial:
                    self.assertIn(f'injected {key} restore', messages)
            finally:
                environment.failures.clear()
                mode.restore()
            self.assertEqual(environment.state, environment.initial)

    def test_rollback_failure_keeps_the_original_error_and_can_be_retried(self):
        with ModeEnvironment().installed() as environment:
            key = next(iter(environment.initial))
            mode = terminals.TerminalMode(123, enabled=True)
            environment.failures.update(((key, 'setup', 'after'), (key, 'restore', 'before')))
            try:
                with self.assertRaises(OSError) as caught:
                    mode.__enter__()
                self.assertIsInstance(caught.exception.__context__, OSError)
            finally:
                environment.failures.clear()
                mode.restore()
            self.assertEqual(environment.state, environment.initial)


class FrontendTerminalOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_output_mode_encloses_overlay_setup_and_teardown(self):
        from loki_agent import terminal_frontend as frontend

        for fails in (False, True):
            with self.subTest(frontend_failure=fails):
                actions = []
                mode = mock.MagicMock()
                mode.__enter__.side_effect = lambda: actions.append('output enter')
                mode.__exit__.side_effect = lambda *args: actions.append('output exit')

                async def run(args):
                    actions.append('run')
                    if fails:
                        raise OSError('frontend failed')
                    return 0

                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(frontend.signal, 'signal'))
                    if hasattr(frontend.signal, 'pthread_sigmask'):
                        stack.enter_context(mock.patch.object(frontend.signal, 'pthread_sigmask'))
                    stack.enter_context(mock.patch.object(frontend, 'configure_tool_hook_pipeline'))
                    stack.enter_context(mock.patch.object(frontend, 'terminal_output_mode', return_value=mode))
                    stack.enter_context(mock.patch.object(frontend, 'current_chat_log_path', return_value=None))
                    stack.enter_context(mock.patch.object(
                        frontend, 'current_session', return_value=mock.Mock(job_manager=None)))
                    stack.enter_context(mock.patch.object(
                        frontend, 'initialize_terminal_overlay',
                        side_effect=lambda *args: actions.append('overlay enter')))
                    stack.enter_context(mock.patch.object(
                        frontend, 'restore_terminal_overlay',
                        side_effect=lambda *args: actions.append('overlay exit')))
                    stack.enter_context(mock.patch.object(frontend, 'async_main', new=run))
                    if fails:
                        with self.assertRaisesRegex(OSError, 'frontend failed'):
                            await frontend._run_frontend([])
                    else:
                        self.assertEqual(await frontend._run_frontend([]), 0)
                self.assertEqual(actions, ['output enter', 'overlay enter', 'run', 'overlay exit', 'output exit'])
