"""The same terminal ownership contracts against each host's native backend."""

import asyncio
import contextlib
import copy
import os
import unittest
from unittest import mock

from loki_agent import terminals
from terminal_resource_fixtures import ModeEnvironment, ReaderEnvironment


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


class ByteReaderOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def assert_reusable(self, environment):
        reader = terminals.AsyncByteReader(environment.read_fd)
        async with reader:
            os.write(environment.write_fd, b'usable')
            self.assertEqual(await asyncio.wait_for(reader.read(), 2), b'usable')
        environment.assert_released(self)

    async def test_success_and_repeated_cleanup_preserve_borrowed_input(self):
        with ReaderEnvironment(asyncio.get_running_loop()).installed() as environment:
            reader = terminals.AsyncByteReader(environment.read_fd)
            await reader.__aenter__()
            await reader.__aexit__(None, None, None)
            await reader.__aexit__(None, None, None)
            environment.assert_released(self)
            await self.assert_reusable(environment)

    async def test_failed_start_releases_acquired_resources(self):
        with ReaderEnvironment(asyncio.get_running_loop()).installed() as environment:
            for stage in environment.acquisitions:
                with self.subTest(stage=stage):
                    reader = terminals.AsyncByteReader(environment.read_fd)
                    environment.failures.add(stage)
                    try:
                        with self.assertRaises((OSError, RuntimeError)):
                            await reader.__aenter__()
                        environment.assert_released(self)
                    finally:
                        environment.failures.clear()
                        await reader.__aexit__(None, None, None)
                    await self.assert_reusable(environment)

    async def test_failed_start_with_failed_rollback_retains_ownership_for_retry(self):
        with ReaderEnvironment(asyncio.get_running_loop()).installed() as environment:
            reader = terminals.AsyncByteReader(environment.read_fd)
            environment.exit_gate.clear()
            environment.failures.update((environment.acquisitions[-1], 'stop'))
            try:
                with self.assertRaises((OSError, RuntimeError)) as caught:
                    await reader.__aenter__()
                self.assertIsNotNone(caught.exception.__context__)
            finally:
                environment.failures.clear()
                environment.exit_gate.set()
                await reader.__aexit__(None, None, None)
            environment.assert_released(self)
            await self.assert_reusable(environment)

    async def test_failed_stop_retains_resources_until_retry(self):
        with ReaderEnvironment(asyncio.get_running_loop()).installed() as environment:
            for stage in environment.releases:
                with self.subTest(stage=stage):
                    reader = terminals.AsyncByteReader(environment.read_fd)
                    await reader.__aenter__()
                    if stage in ('stop', 'signal'):
                        environment.exit_gate.clear()
                    environment.failures.add(stage)
                    try:
                        with self.assertRaises((OSError, RuntimeError)):
                            await reader.__aexit__(None, None, None)
                        os.fstat(environment.read_fd)
                        if stage == 'stop':
                            if os.name == 'posix':
                                self.assertTrue(environment.registered)
                                with mock.patch.object(terminals.os, 'read', return_value=b'late') as read:
                                    reader._on_readable()
                                read.assert_not_called()
                            else:
                                self.assertTrue(environment.events)
                                self.assertTrue(any(thread.is_alive() for thread, _ in environment.threads))
                                source = reader._windows_reader
                                source._post(b'late')
                            await asyncio.sleep(0)
                            self.assertTrue(reader.queue.empty())
                    finally:
                        environment.failures.clear()
                        environment.exit_gate.set()
                        await reader.__aexit__(None, None, None)
                    environment.assert_released(self)
                    await self.assert_reusable(environment)


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


class InputSessionOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_paths_restore_independent_resources_and_retry_only_pending_ones(self):
        cases = (
            ('mode_enter',), ('reader_enter',), ('producer_create',),
            ('producer_exit',), ('reader_exit',), ('mode_exit',),
            ('reader_exit', 'mode_exit'), ('reader_enter', 'mode_exit'),
            ('producer_exit', 'reader_exit', 'mode_exit'),
        )
        for failures in cases:
            with self.subTest(failures=failures):
                faults = set(failures)
                active = set()
                actions = []
                started = asyncio.Event()
                rejected_coroutines = []

                def step(resource, entering):
                    action = resource + ('_enter' if entering else '_exit')
                    actions.append(action)
                    if entering:
                        active.add(resource)
                    if action in faults:
                        raise RuntimeError('injected ' + action)
                    if not entering:
                        active.discard(resource)

                mode = mock.MagicMock()
                mode.__enter__.side_effect = lambda: step('mode', True)
                mode.__exit__.side_effect = lambda *args: step('mode', False)
                reader = mock.Mock()
                reader.__aenter__ = mock.AsyncMock(side_effect=lambda: step('reader', True))
                reader.__aexit__ = mock.AsyncMock(side_effect=lambda *args: step('reader', False))

                async def produce():
                    active.add('producer')
                    started.set()
                    try:
                        await asyncio.Future()
                    finally:
                        active.discard('producer')
                        actions.append('producer_exit')
                        if 'producer_exit' in faults:
                            raise RuntimeError('injected producer_exit')

                def reject_task(coroutine):
                    rejected_coroutines.append(coroutine)
                    raise RuntimeError('injected producer_create')

                session = terminals.InputSession(fd=123)
                session.interactive = True
                session.reader = reader
                session._produce = produce
                with mock.patch.object(terminals, 'TerminalMode', return_value=mode):
                    if any(name.endswith('_enter') for name in faults):
                        with self.assertRaisesRegex(RuntimeError, 'injected'):
                            await session.__aenter__()
                    elif 'producer_create' in faults:
                        with (
                            mock.patch.object(terminals.asyncio, 'create_task', side_effect=reject_task),
                            self.assertRaisesRegex(RuntimeError, 'producer_create'),
                        ):
                            await session.__aenter__()
                        self.assertTrue(all(coro.cr_frame is None for coro in rejected_coroutines))
                    else:
                        await session.__aenter__()
                        await asyncio.wait_for(started.wait(), 1)
                        with self.assertRaisesRegex(RuntimeError, 'injected'):
                            await session.__aexit__(None, None, None)
                    expected = {name.removesuffix('_exit') for name in faults if name in ('reader_exit', 'mode_exit')}
                    self.assertEqual(active, expected)
                    if 'reader_exit' in actions and 'mode_exit' in actions:
                        self.assertLess(actions.index('reader_exit'), actions.index('mode_exit'))
                    faults.clear()
                    previous = len(actions)
                    await session.__aexit__(None, None, None)
                    expected_actions = [name + '_exit' for name in ('reader', 'mode') if name in expected]
                    self.assertEqual(actions[previous:], expected_actions)
                    self.assertFalse(active)
                    previous = list(actions)
                    await session.__aexit__(None, None, None)
                    self.assertEqual(actions, previous)

    async def test_body_cancellation_closes_the_input_session(self):
        with ReaderEnvironment(asyncio.get_running_loop()).installed() as environment:
            session = terminals.InputSession(fd=environment.read_fd)
            session.interactive = False
            entered = asyncio.Event()

            async def run():
                async with session:
                    entered.set()
                    await asyncio.Future()

            task = asyncio.create_task(run())
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            environment.assert_released(self)
            self.assertIsNone(session._producer)
            await session.__aexit__(None, None, None)


class KeyReaderOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_reader_shutdown_remains_owned_by_key_reader(self):
        loop = asyncio.get_running_loop()
        with ReaderEnvironment(loop).installed() as environment:
            reader = terminals.AsyncKeyReader(environment.read_fd)
            await reader.__aenter__()
            environment.exit_gate.clear()
            environment.failures.add('stop')
            try:
                with self.assertRaises((OSError, RuntimeError)):
                    await reader.__aexit__(None, None, None)
            finally:
                environment.failures.clear()
                environment.exit_gate.set()
                await reader.__aexit__(None, None, None)
            environment.assert_released(self)
