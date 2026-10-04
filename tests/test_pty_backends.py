"""PTY stream completion, exit-status ownership, and native cleanup."""

import errno
import os
import signal
import types
import unittest
from unittest import mock

from loki_agent import pty_backend, windows_api


@unittest.skipUnless(os.name == 'posix', 'POSIX PTY backend')
class PosixPtyTests(unittest.TestCase):
    def test_reaping_preserves_failure_and_does_not_signal_reused_pid(self):
        handle = pty_backend._PosixPty(123, 456)
        status = object()
        with (
            mock.patch.object(os, 'waitpid', return_value=(123, status)) as wait,
            mock.patch.object(os, 'waitstatus_to_exitcode', return_value=2),
            mock.patch.object(os, 'killpg') as kill,
        ):
            self.assertEqual(handle.poll(), 2)
            self.assertEqual(handle.wait(), 2)
            self.assertEqual(handle.poll(), 2)
            handle.terminate()
        wait.assert_called_once_with(123, os.WNOHANG)
        kill.assert_not_called()

    def test_teardown_kills_owned_group_and_reaps_without_grace_period(self):
        handle = pty_backend._PosixPty(123, 456)
        status = object()
        with (
            mock.patch.object(os, 'waitpid', side_effect=[(0, status), (123, status)]) as wait,
            mock.patch.object(os, 'waitstatus_to_exitcode', return_value=-signal.SIGKILL),
            mock.patch.object(os, 'killpg') as kill,
        ):
            handle.terminate()
            handle.terminate()
        kill.assert_called_once_with(123, signal.SIGKILL)
        self.assertEqual(wait.call_args_list, [mock.call(123, os.WNOHANG), mock.call(123, 0)])

    def test_immediate_abort_also_handles_a_not_yet_established_group(self):
        handle = pty_backend._PosixPty(123, 456)
        status = object()
        with (
            mock.patch.object(os, 'waitpid', side_effect=[(0, status), (123, status)]),
            mock.patch.object(os, 'waitstatus_to_exitcode', return_value=-signal.SIGKILL),
            mock.patch.object(os, 'killpg', side_effect=ProcessLookupError),
            mock.patch.object(os, 'kill') as kill,
        ):
            handle.terminate()
        kill.assert_called_once_with(123, signal.SIGKILL)
        self.assertEqual(handle.exit_code, -signal.SIGKILL)

    def test_read_maps_only_closed_slave_to_eof(self):
        handle = pty_backend._PosixPty(123, 456)
        with mock.patch.object(os, 'read', side_effect=OSError(errno.EIO, 'closed slave')):
            self.assertEqual(handle.read(1024), b'')
        with mock.patch.object(os, 'read', side_effect=OSError(errno.EBADF, 'bad descriptor')):
            with self.assertRaises(OSError) as error:
                handle.read(1024)
            self.assertEqual(error.exception.errno, errno.EBADF)

    def test_close_releases_master_once(self):
        handle = pty_backend._PosixPty(123, 456)
        with mock.patch.object(os, 'close') as close:
            handle.close()
            handle.close()
        close.assert_called_once_with(456)


class WindowsPtyTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.multiple(
            windows_api, create_pipe=mock.DEFAULT, pseudoconsole_create=mock.DEFAULT,
            create_process_with_pseudoconsole=mock.DEFAULT,
            pseudoconsole_release=mock.DEFAULT, pseudoconsole_close=mock.DEFAULT,
            close_handle=mock.DEFAULT, read_file=mock.DEFAULT,
            wait_for_single_object=mock.DEFAULT, get_exit_code_process=mock.DEFAULT,
            terminate_process=mock.DEFAULT,
        )
        self.api = patcher.start()
        self.addCleanup(patcher.stop)
        self.closed = []
        self.api['create_pipe'].side_effect = [
            ['input-read', 'input-write'], ['output-read', 'output-write']]
        self.api['pseudoconsole_create'].return_value = 'conpty'
        self.api['create_process_with_pseudoconsole'].return_value = types.SimpleNamespace(
            hProcess='process', hThread='thread')
        self.api['close_handle'].side_effect = self.closed.append
        self.api['pseudoconsole_close'].side_effect = self.closed.append
        self.api['wait_for_single_object'].return_value = windows_api.WAIT_OBJECT_0
        self.api['get_exit_code_process'].return_value = 0

    def spawn(self):
        return pty_backend._windows_spawn(['/installed/loki', '--help'], 80, 24, {}, '/workspace')

    def test_release_precedes_capture_and_close_follows_terminal_eof(self):
        def release(hpc):
            self.assertEqual(hpc, 'conpty')
            self.assertCountEqual(self.closed, ['output-write', 'input-read'])
        self.api['pseudoconsole_release'].side_effect = release
        handle = self.spawn()
        self.api['get_exit_code_process'].return_value = 2
        self.assertEqual(handle.poll(), 2)
        # The root has exited, but ConPTY can still have final/descendant output.
        self.api['read_file'].side_effect = [
            b'late output', b'\x1b[0m',
            windows_api.WindowsApiError('closed pipe', status=windows_api.ERROR_BROKEN_PIPE)]
        self.assertEqual(handle.read(1024), b'late output')
        self.assertEqual(handle.read(1024), b'\x1b[0m')
        self.assertEqual(handle.read(1024), b'')
        self.api['pseudoconsole_close'].assert_not_called()
        self.assertEqual(handle.wait(), 2)
        handle.close()
        handle.close()
        self.assertCountEqual(self.closed, [
            'output-write', 'input-read', 'output-read', 'input-write',
            'conpty', 'thread', 'process'])
        self.api['pseudoconsole_close'].assert_called_once_with('conpty')
        self.assertLess(self.closed.index('output-read'), self.closed.index('conpty'))

    def test_read_failure_is_not_eof(self):
        handle = self.spawn()
        error = windows_api.WindowsApiError('read denied', status=windows_api.ERROR_ACCESS_DENIED)
        self.api['read_file'].side_effect = error
        with self.assertRaises(windows_api.WindowsApiError) as caught:
            handle.read(1024)
        self.assertIs(caught.exception, error)
        handle.close()

    def test_failed_acquisitions_release_every_acquired_resource(self):
        cases = [
            ['first-pipe', []],
            ['second-pipe', ['input-read', 'input-write']],
            ['conpty', ['output-write', 'input-read', 'output-read', 'input-write']],
            ['process', ['output-write', 'input-read', 'output-read', 'input-write', 'conpty']],
            ['release', ['output-write', 'input-read', 'output-read', 'input-write',
                         'conpty', 'thread', 'process']],
        ]
        for stage, expected in cases:
            with self.subTest(stage=stage):
                self.closed.clear()
                for name in ['pseudoconsole_create', 'create_process_with_pseudoconsole',
                             'pseudoconsole_release']:
                    self.api[name].side_effect = None
                self.api['create_process_with_pseudoconsole'].return_value = types.SimpleNamespace(
                    hProcess='process', hThread='thread')
                failure = OSError('injected setup failure')
                pipes = [['input-read', 'input-write'], ['output-read', 'output-write']]
                if stage == 'first-pipe':
                    pipes[0] = failure
                elif stage == 'second-pipe':
                    pipes[1] = failure
                else:
                    operation = {'conpty': 'pseudoconsole_create',
                                 'process': 'create_process_with_pseudoconsole',
                                 'release': 'pseudoconsole_release'}[stage]
                    self.api[operation].side_effect = failure
                self.api['create_pipe'].side_effect = pipes
                self.api['wait_for_single_object'].side_effect = [
                    windows_api.WAIT_TIMEOUT, windows_api.WAIT_OBJECT_0]
                self.api['terminate_process'].reset_mock()
                with self.assertRaisesRegex(OSError, 'injected setup failure'):
                    self.spawn()
                self.assertCountEqual(self.closed, expected)
                if 'conpty' in expected:
                    self.assertLess(self.closed.index('output-read'), self.closed.index('conpty'))
                if stage == 'release':
                    self.api['terminate_process'].assert_called_once_with('process')
                else:
                    self.api['terminate_process'].assert_not_called()

    def test_close_failure_keeps_failed_resource_and_releases_the_rest(self):
        handle = self.spawn()

        def close(native_handle):
            if native_handle == 'output-read':
                raise OSError('injected close failure')
            self.closed.append(native_handle)
        self.api['close_handle'].side_effect = close
        with self.assertRaisesRegex(OSError, 'injected close failure'):
            handle.close()
        self.assertCountEqual(self.closed, [
            'output-write', 'input-read', 'input-write', 'conpty', 'thread', 'process'])
        self.api['close_handle'].side_effect = self.closed.append
        handle.close()
        self.assertEqual(self.closed[-1], 'output-read')
        self.api['pseudoconsole_close'].assert_called_once_with('conpty')

    def test_termination_failure_is_suppressed_only_when_exit_won_the_race(self):
        handle = self.spawn()
        self.api['wait_for_single_object'].side_effect = [
            windows_api.WAIT_TIMEOUT, windows_api.WAIT_OBJECT_0, windows_api.WAIT_OBJECT_0]
        self.api['terminate_process'].side_effect = windows_api.WindowsApiError('already exited')
        handle.terminate()
        handle.close()

    def test_live_child_termination_failure_is_reported(self):
        handle = self.spawn()
        self.api['wait_for_single_object'].return_value = windows_api.WAIT_TIMEOUT
        self.api['terminate_process'].side_effect = windows_api.WindowsApiError('terminate denied')
        try:
            with self.assertRaisesRegex(windows_api.WindowsApiError, 'terminate denied'):
                handle.terminate()
        finally:
            handle.close()
        self.assertCountEqual(self.closed, [
            'input-read', 'input-write', 'output-read', 'output-write',
            'conpty', 'process', 'thread'])

    def test_wait_failure_is_not_reported_as_exit(self):
        handle = self.spawn()
        self.api['wait_for_single_object'].return_value = windows_api.WAIT_TIMEOUT
        with self.assertRaisesRegex(windows_api.WindowsApiError, 'wait failed'):
            handle.wait()
        handle.close()


class WindowsReleaseBindingTests(unittest.TestCase):
    def test_release_reports_hresult_failure(self):
        native = mock.Mock(return_value=-1)
        with mock.patch.object(windows_api, 'bind', return_value=native):
            with self.assertRaisesRegex(windows_api.WindowsApiError, 'ReleasePseudoConsole failed'):
                windows_api.pseudoconsole_release('conpty')
        native.assert_called_once_with('conpty')

    def test_missing_release_reports_test_harness_requirement(self):
        with mock.patch.object(windows_api, 'bind', side_effect=AttributeError('missing API')):
            with self.assertRaisesRegex(windows_api.WindowsUnavailableError, 'Windows 11 24H2'):
                windows_api.pseudoconsole_release('conpty')


if __name__ == '__main__':
    unittest.main()
