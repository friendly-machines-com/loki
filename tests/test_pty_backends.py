"""PTY stream completion, exit-status ownership, and native cleanup."""

import ctypes
import errno
import os
import signal
import sys
import unittest
from unittest import mock

from loki_agent import host_ipc, pty_backend, windows_api


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


@unittest.skipUnless(sys.platform == 'win32', 'Windows ConPTY factory ABI')
class WindowsPtyTests(unittest.TestCase):
    def setUp(self):
        self.creator = windows_api.create_process_suspended
        patcher = mock.patch.multiple(
            windows_api, pseudoconsole_create=mock.DEFAULT,
            create_process_suspended=mock.DEFAULT,
            pseudoconsole_release=mock.DEFAULT, pseudoconsole_close=mock.DEFAULT,
            close_handle=mock.DEFAULT, read_file=mock.DEFAULT,
            wait_for_single_object=mock.DEFAULT, get_exit_code_process=mock.DEFAULT,
            terminate_process=mock.DEFAULT, resume_thread=mock.DEFAULT,
        )
        self.api = patcher.start()
        self.addCleanup(patcher.stop)
        pipe_patcher = mock.patch.object(host_ipc, 'private_pipe')
        self.api['private_pipe'] = pipe_patcher.start()
        self.addCleanup(pipe_patcher.stop)
        self.closed = []
        self.api['private_pipe'].side_effect = [
            ['input-read', 'input-write'], ['output-read', 'output-write']]
        self.hpc = ctypes.c_void_p(33)
        self.api['pseudoconsole_create'].return_value = self.hpc
        self.api['create_process_suspended'].side_effect = self.create
        self.api['close_handle'].side_effect = self.closed.append
        self.api['pseudoconsole_close'].side_effect = lambda hpc: self.closed.append('conpty')
        self.api['wait_for_single_object'].return_value = windows_api.WAIT_OBJECT_0
        self.api['get_exit_code_process'].return_value = 0

    def create(self, executable, arguments, *, information, **options):
        self.assertEqual([executable, arguments], ['/installed/loki', ['--help']])
        self.assertEqual(options, {'pseudoconsole': 33, 'standard_handles': [None, None, None],
                                   'environment': {}, 'current_directory': '/workspace'})
        self.assertFalse(information.hProcess)
        self.assertFalse(information.hThread)
        information.hProcess, information.hThread = 11, 22

    def spawn(self):
        return pty_backend._windows_spawn(['/installed/loki', '--help'], 80, 24, {}, '/workspace')

    def test_release_precedes_capture_and_close_follows_terminal_eof(self):
        def resume(thread):
            self.assertEqual(thread, 22)
            self.assertEqual(self.closed, ['output-write', 'input-read'])

        def release(hpc):
            self.assertIs(hpc, self.hpc)
            self.assertCountEqual(self.closed, ['output-write', 'input-read'])
            self.api['resume_thread'].assert_called_once_with(22)
        self.api['resume_thread'].side_effect = resume
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
            'conpty', 22, 11])
        self.api['pseudoconsole_close'].assert_called_once_with(self.hpc)
        self.assertLess(self.closed.index('output-read'), self.closed.index('conpty'))

    def test_read_failure_is_not_eof(self):
        handle = self.spawn()
        error = windows_api.WindowsApiError('read denied', status=windows_api.ERROR_ACCESS_DENIED)
        self.api['read_file'].side_effect = error
        with self.assertRaises(windows_api.WindowsApiError) as caught:
            handle.read(1024)
        self.assertIs(caught.exception, error)
        handle.close()

    def test_created_child_is_retired_when_attribute_cleanup_is_interrupted(self):
        failure = KeyboardInterrupt()

        def initialize(attributes, count, flags, size):
            size._obj.value = 256
            return attributes is not None

        def create(*args):
            information = args[-1]._obj
            information.hProcess, information.hThread = 11, 22
            return True

        calls = {'InitializeProcThreadAttributeList': initialize,
                 'UpdateProcThreadAttribute': mock.Mock(return_value=True),
                 'DeleteProcThreadAttributeList': mock.Mock(side_effect=failure),
                 'CreateProcessW': create,
                 'ConvertStringSidToSidW': mock.Mock(), 'LocalFree': mock.Mock()}
        self.api['create_process_suspended'].side_effect = self.creator
        self.api['wait_for_single_object'].side_effect = [windows_api.WAIT_TIMEOUT, windows_api.WAIT_OBJECT_0]
        with mock.patch.object(windows_api, 'bind', side_effect=lambda lib, name, *args: calls[name]), \
                mock.patch.object(ctypes, 'get_last_error', return_value=122, create=True):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.spawn()
        self.assertIs(caught.exception, failure)
        self.api['terminate_process'].assert_called_once_with(11)
        self.api['wait_for_single_object'].assert_has_calls([
            mock.call(11, 0), mock.call(11, windows_api.INFINITE)])
        self.assertIn(11, self.closed)
        self.assertIn(22, self.closed)

    def _assert_failed_spawn(self, stage, expected, *, acquired=False):
        failure = OSError('injected setup failure')
        pipes = [['input-read', 'input-write'], ['output-read', 'output-write']]
        if stage == 'first-pipe':
            pipes[0] = failure
        elif stage == 'second-pipe':
            pipes[1] = failure
        else:
            operation = {'conpty': 'pseudoconsole_create', 'process': 'create_process_suspended',
                         'resume': 'resume_thread', 'release': 'pseudoconsole_release'}[stage]
            self.api[operation].side_effect = failure
        self.api['private_pipe'].side_effect = pipes

        def wait(process, budget):
            self.assertEqual(process, 11)
            if budget == windows_api.INFINITE:
                self.closed.append('wait')
                return windows_api.WAIT_OBJECT_0
            self.assertEqual(budget, 0)
            return windows_api.WAIT_TIMEOUT

        self.api['wait_for_single_object'].side_effect = wait
        self.api['terminate_process'].side_effect = lambda process: self.closed.append('terminate')
        with self.assertRaises(OSError) as caught:
            self.spawn()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.closed, expected)
        if acquired:
            self.api['terminate_process'].assert_called_once_with(11)
            self.api['wait_for_single_object'].assert_has_calls([
                mock.call(11, 0), mock.call(11, windows_api.INFINITE)])
        else:
            self.api['terminate_process'].assert_not_called()
            self.api['wait_for_single_object'].assert_not_called()
        if stage != 'release':
            self.api['pseudoconsole_release'].assert_not_called()

    def test_first_pipe_failure_acquires_nothing(self):
        self._assert_failed_spawn('first-pipe', [])

    def test_second_pipe_failure_releases_the_first_pair(self):
        self._assert_failed_spawn('second-pipe', ['input-read', 'input-write'])

    def test_console_creation_failure_releases_both_pairs(self):
        self._assert_failed_spawn('conpty', ['output-write', 'input-read', 'output-read', 'input-write'])

    def test_process_creation_failure_does_not_poll_empty_output(self):
        self._assert_failed_spawn('process', ['output-write', 'input-read', 'output-read', 'input-write', 'conpty'])

    def test_resume_failure_retires_the_acquired_child_before_owned_resources(self):
        self._assert_failed_spawn(
            'resume', ['output-write', 'input-read', 'terminate', 'wait',
                       'output-read', 'input-write', 'conpty', 22, 11], acquired=True)

    def test_session_release_failure_retires_the_acquired_child_before_owned_resources(self):
        self._assert_failed_spawn(
            'release', ['output-write', 'input-read', 'terminate', 'wait',
                        'output-read', 'input-write', 'conpty', 22, 11], acquired=True)

    def test_close_failure_consumes_once_and_releases_the_rest(self):
        handle = self.spawn()
        failure = windows_api.WindowsApiError('injected close failure', status=6)

        def close(native_handle):
            if native_handle == 'output-read':
                raise failure
            self.closed.append(native_handle)
        self.api['close_handle'].side_effect = close
        with self.assertRaises(windows_api.WindowsApiError) as caught:
            handle.close()
        self.assertIs(caught.exception, failure)
        self.assertCountEqual(self.closed, [
            'output-write', 'input-read', 'input-write', 'conpty', 22, 11])
        attempts = list(self.api['close_handle'].call_args_list)
        self.assertEqual(attempts.count(mock.call('output-read')), 1)
        handle.close()
        self.assertEqual(self.api['close_handle'].call_args_list, attempts)
        self.assertIsNone(handle.output_read)
        self.assertIsNone(handle.input_write)
        self.assertIsNone(handle.hpc)
        self.assertIsNone(handle.information)
        self.api['pseudoconsole_close'].assert_called_once_with(self.hpc)

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
            'conpty', 11, 22])

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
