"""Conversation storage follows workspace ownership, never process or /cd cwd."""

from contextlib import redirect_stderr
import io
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from config_fixtures import setUpModule as set_up_config
from response_header_fixtures import setUpModule as set_up_response_headers
from loki_entrypoints import child_environment, configure_container, entrypoint
from loki_agent import loki, savefiles, sessions, settings, terminal_frontend


def setUpModule():
    set_up_config()
    set_up_response_headers()


class WorkspaceHistoryTests(unittest.IsolatedAsyncioTestCase):
    def session(self, workspace):
        session = sessions.Session(shell_cwd=workspace)
        self.addCleanup(session.response_headers.close)
        self.enterContext(mock.patch.object(loki, '_DEFAULT_SESSION', session))
        return session

    async def test_allocation_listing_and_resume_keep_workspace_after_cd(self):
        process_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tools:
            session = self.session(workspace)
            with mock.patch.object(loki.paths, 'chat_log_dir_for', wraps=loki.paths.chat_log_dir_for) as directory:
                path = loki.new_chat_log_path()
                directory.assert_called_once_with(workspace)
            loki.new_chat_log(path)
            loki.save_chat_log()
            loki.change_shell_cwd(tools)
            session.session_state['workspace_cwd'] = 'untrusted saved value'
            loki.mark_chat_log_dirty()
            loki.save_chat_log()
            self.assertEqual(session.workspace_cwd, workspace)
            entries = savefiles.chat_log_entries(loki.chat_log_directory())
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].path, path)
            self.assertEqual(loki.resolve_chat_log_path(entries[0].name), path)
            summaries = await savefiles.list_saved_sessions(loki.chat_log_directory())
            self.assertEqual([item['path'] for item in summaries], [path])
            self.assertEqual(summaries[0]['cwd'], tools)
            loki.load_chat_log(path)
            self.assertEqual(session.shell_cwd, tools)
            self.assertEqual(session.workspace_cwd, workspace)
            self.assertEqual(loki.resolve_chat_log_path(entries[0].name), path)
            with mock.patch.object(loki.paths, 'chat_log_dir_for', wraps=loki.paths.chat_log_dir_for) as directory:
                loki.new_chat_log_path()
                directory.assert_called_once_with(workspace)
        self.assertEqual(os.getcwd(), process_cwd)

    async def test_explicit_resume_operand_keeps_fixed_lookup_context_verbatim(self):
        session = self.session('opaque/../workspace//.')
        session.shell_cwd = 'different tool directory'
        operand = 'link/../logs/chat-saved.json'
        with mock.patch.object(loki, '_resolve_path', return_value=mock.sentinel.selected_log) as resolve:
            self.assertIs(loki.resolve_chat_log_path(operand), mock.sentinel.selected_log)
        resolve.assert_called_once_with(operand, loki.STARTUP_CWD)

    async def test_cli_captures_workspace_key_before_tool_or_saved_cwd_can_change_it(self):
        session = self.session(os.getcwd())
        workspace = 'opaque/../workspace//.'

        def change(key):
            self.assertIs(key, workspace)
            session.shell_cwd = 'tool lookup spelling'

        with mock.patch.object(loki, 'change_shell_cwd', side_effect=change), \
                mock.patch.object(terminal_frontend, 'build_config_from_env', return_value=object()), \
                mock.patch.object(terminal_frontend, 'apply_runtime_config'), \
                mock.patch.object(terminal_frontend, 'current_model', return_value='dummy'), \
                mock.patch.object(terminal_frontend.subagents, 'run_cli_async', new_callable=mock.AsyncMock,
                                  return_value=0):
            self.assertEqual(await terminal_frontend.async_main(
                ['--shell-cwd', workspace, '--headless'], settings.Settings()), 0)
        self.assertIs(session.workspace_cwd, workspace)
        self.assertEqual(session.shell_cwd, 'tool lookup spelling')

    async def test_empty_cli_workspace_fails_without_rebinding_history(self):
        session = self.session(os.getcwd())
        workspace = session.workspace_cwd
        with redirect_stderr(io.StringIO()), mock.patch.object(loki, 'change_shell_cwd') as change:
            result = await terminal_frontend.async_main(['--shell-cwd', ''], settings.Settings())
        self.assertEqual(result, 2)
        change.assert_not_called()
        self.assertEqual(session.workspace_cwd, workspace)


class WorkspaceHistoryProcessTests(unittest.TestCase):
    def test_real_terminal_save_picker_and_named_resume_use_selected_workspace(self):
        with tempfile.TemporaryDirectory() as control, tempfile.TemporaryDirectory() as workspace, \
                tempfile.TemporaryDirectory(dir=workspace) as tools, \
                tempfile.TemporaryDirectory() as config, tempfile.TemporaryDirectory() as state:
            launch_cwd = control
            if os.name == 'nt':
                # Query the actual packaged image's parent, not a project-derived
                # pathname or a text-normalized installation root. Setup grants
                # this bundle read-only access to its contained runtime.
                from loki_agent import windows_api
                image = windows_api.open_with_access(entrypoint('loki'), windows_api.AccessMask.FILE_READ_ATTRIBUTES)
                try:
                    parent = windows_api.open_parent_directory(image)
                    self.assertIsNotNone(parent)
                    try:
                        launch_cwd = windows_api.kernel_path_from_handle(parent)
                    finally:
                        windows_api.close_handle(parent)
                finally:
                    windows_api.close_handle(image)
            with os.scandir(launch_cwd) as entries:
                before = {entry.name for entry in entries}
            env = child_environment(
                PATH=os.environ.get('PATH', ''), TERM='dumb', HOME=config, USERPROFILE=config,
                XDG_CONFIG_HOME=config, XDG_STATE_HOME=state,
                LOKI_PROVIDER='dummy', LOKI_API_BASE='http://dummy.invalid/v1', LOKI_MODEL='dummy',
                LOKI_DUMMY_REPLY='workspace reply')
            configure_container(env, workspace)
            try:
                command = [entrypoint('loki'), '--shell-cwd', workspace]
                change = '/cd "' + tools + '"\n/quit\n'
                first = subprocess.run(command, cwd=launch_cwd, env=env, input='workspace history witness\n' + change,
                                       capture_output=True, text=True)
                self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
                with os.scandir(workspace) as entries:
                    workspace_state = next(entry.path for entry in entries if entry.name == '.loki')
                with os.scandir(workspace_state) as entries:
                    history = next(entry.path for entry in entries if entry.name == 'chats')
                logs = savefiles.chat_log_entries(history)
                self.assertEqual(len(logs), 1)
                with open(logs[0].path, encoding='utf-8') as source:
                    blob = json.load(source)
                self.assertEqual(blob['session_state']['shell_cwd'], tools)
                self.assertNotIn('workspace_cwd', blob['session_state'])
                with os.scandir(tools) as entries:
                    self.assertNotIn('.loki', {entry.name for entry in entries})
                for arguments, stdin in [[['--resume'], '1\n' + change],
                                         [['-r', logs[0].name], change]]:
                    result = subprocess.run([*command, *arguments], cwd=launch_cwd, env=env,
                                            input=stdin, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('workspace history witness', result.stdout)
                    self.assertIn('workspace reply', result.stdout)
                    self.assertEqual([entry.path for entry in savefiles.chat_log_entries(history)], [logs[0].path])
                with os.scandir(launch_cwd) as entries:
                    self.assertEqual({entry.name for entry in entries}, before)
            finally:
                if os.name == 'nt':
                    result = subprocess.run([entrypoint('loki-setup'), '--uninstall'], cwd=launch_cwd,
                                            env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
