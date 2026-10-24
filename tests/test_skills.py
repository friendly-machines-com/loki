"""Skill discovery and loading through both sanctioned frontend processes."""

import asyncio
import copy
import json
import os
import tempfile
import unittest
from unittest import mock

from config_fixtures import setUpModule as set_up_config
from response_header_fixtures import setUpModule as set_up_response_headers
from loki_entrypoints import child_environment, configure_container, entrypoint
from process_lifecycle_fixtures import ProcessResources
from test_acps import _ACPFrontFixture
from test_pty_ui import _read_to_exit, _read_until, _SgrStreamTracker

from loki_agent import connections, formats, loki, paths, pty_backend, terminal_frontend
from loki_agent.acp_worker import Worker
from loki_agent.credentials import CredentialStore
from loki_agent.sessions import Session


_NAME = 'pack:inner'
_DESCRIPTION = 'Explains the selected project tree'
_ARGUMENT = 'explain this tree without changing it?'
_BODY = '# Inner skill\nUse the selected project as read-only context.\n'


def setUpModule():
    set_up_config()
    set_up_response_headers()


class SkillFrontendTests(_ACPFrontFixture, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='loki-skill-fronts-')
        self.addCleanup(temporary.cleanup)
        self.workspace = os.path.join(temporary.name, 'workspace')
        config = os.path.join(temporary.name, 'config')
        state = os.path.join(temporary.name, 'state')
        os.mkdir(self.workspace)
        self.environment = child_environment(
            PATH=os.environ.get('PATH', ''), TERM='dumb', HOME=temporary.name,
            USERPROFILE=temporary.name, XDG_CONFIG_HOME=config, XDG_STATE_HOME=state,
            LOKI_PROVIDER='dummy', LOKI_API_BASE='http://dummy.invalid/v1',
            LOKI_MODEL='dummy-model', LOKI_DUMMY_REPLY='Skill journey complete.')
        self.config = paths.loki_config_dir(self.environment)
        self.directory = os.path.join(self.config, 'skills', 'pack', 'inner')
        self.filename = os.path.join(self.directory, 'SKILL.md')
        os.makedirs(self.directory)
        self.write_skill(_DESCRIPTION)
        configure_container(self.environment, self.workspace)

    def write_skill(self, description):
        with open(self.filename, 'w', encoding='utf-8') as stream:
            stream.write('---\ndescription: ' + description + '\n---\n' + _BODY)

    def environment_for(self, name):
        environment = dict(self.environment)
        environment['LOKI_DUMMY_TOOL_CALL'] = json.dumps({
            'name': 'Skill', 'arguments': {'skill': name, 'args': _ARGUMENT}})
        return environment

    def saved_result(self, name):
        with os.scandir(paths.chat_log_dir_for(self.workspace)) as directory:
            logs = [entry.path for entry in directory
                    if entry.name.startswith('chat-') and entry.name.endswith('.json')]
        self.assertEqual(len(logs), 1, logs)
        with open(logs[0], encoding='utf-8') as stream:
            blob = json.load(stream)
        formats.validate_events(blob['events'])
        calls = [call for call in formats.response_tool_calls(blob['events'])
                 if formats.tool_call_name(call) == 'Skill']
        self.assertEqual(len(calls), 1)
        self.assertEqual(formats.tool_call_input(calls[0]), {'skill': name, 'args': _ARGUMENT})
        results = [event for event in blob['events']
                   if event.get('type') == 'tool_result' and event.get('name') == 'Skill']
        self.assertEqual(len(results), 1)
        return results[0]

    def assert_loaded(self, result):
        self.assertFalse(result['is_error'], result)
        content = formats.item_text(result)
        self.assertIn(_BODY, content)
        self.assertIn('Args: ' + _ARGUMENT, content)
        self.assertIn('Base directory for this skill: ' + self.directory, content)

    async def terminal(self, name):
        process = await asyncio.create_subprocess_exec(
            entrypoint('loki'), '--dangerously-skip-permissions',
            cwd=self.workspace, env=self.environment_for(name),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        resources = ProcessResources(process)
        self.addAsyncCleanup(resources.cleanup)
        stdout, stderr = await process.communicate(('/' + name + ' ' + _ARGUMENT + '\n/quit\n').encode())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(process.returncode, 0, stderr.decode(errors='replace'))
        resources.assert_released(self)
        return stdout.decode(errors='replace'), stderr.decode(errors='replace')

    async def acp(self, name):
        front, diagnostics = await self._front(self.environment_for(name), self.workspace)
        await self._request(front, 1, 'initialize', {'protocolVersion': 1})
        opened, _preceding = await self._request(front, 2, 'session/new', {'cwd': self.workspace})
        session_id = opened['sessionId']
        message = json.loads(await front.stdout.readline())
        self.assertEqual(message['method'], 'session/update')
        update = message['params']['update']
        self.assertEqual(update['sessionUpdate'], 'available_commands_update')
        advertised = update['availableCommands']
        reply, messages = await self._request(front, 3, 'session/prompt', {
            'sessionId': session_id,
            'prompt': [{'type': 'text', 'text': '/' + name + ' ' + _ARGUMENT}],
        })
        self.assertEqual(reply, {'stopReason': 'end_turn'})
        await self._finish(front, diagnostics, session_id)
        return advertised, messages

    async def test_terminal_announces_catalogue_and_loads_namespaced_skill(self):
        """A terminal user must see valid IDs and load one through the real tool loop."""
        stdout, stderr = await self.terminal(_NAME)
        self.assertIn('Available skills:', stdout)
        self.assertIn('/' + _NAME + ': ' + _DESCRIPTION, stdout)
        self.assertNotIn('Available skills:', stderr)
        self.assertIn('Skill journey complete.', stdout)
        self.assert_loaded(self.saved_result(_NAME))

    def test_terminal_pty_announces_and_invokes_skill_before_natural_exit(self):
        """The real TTY must reveal skills before input and execute the user's request."""
        environment = self.environment_for(_NAME)
        environment['TERM'] = 'xterm'
        handle = pty_backend.spawn_pty([entrypoint('loki')], env=environment, cwd=self.workspace)
        captured = bytearray()
        try:
            _read_until(handle, captured, 'User: !')
            visible = _SgrStreamTracker()
            visible.feed(bytes(captured))
            self.assertIn('/' + _NAME + ': ' + _DESCRIPTION, ''.join(visible.text))
            handle.write(b'\x7f' + ('/' + _NAME + ' ' + _ARGUMENT).encode() + b'\r')
            handle.write(b'\x7f/quit\r')
            code, output = _read_to_exit(handle, captured)
            self.assertEqual(code, 0, output)
            self.assert_loaded(self.saved_result(_NAME))
        finally:
            try:
                handle.terminate()
            finally:
                handle.close()

    async def test_acp_announces_same_catalogue_and_loads_namespaced_skill(self):
        """ACP advertisement and actual loading must identify the same file as terminal."""
        advertised, _messages = await self.acp(_NAME)
        entries = [entry for entry in advertised if entry['name'] == _NAME]
        self.assertEqual(entries, [{'name': _NAME, 'description': _DESCRIPTION}])
        self.assert_loaded(self.saved_result(_NAME))

    async def test_terminal_records_unknown_skill_as_failed_tool_result(self):
        """An unregistered identifier must not become a filename or a successful load."""
        await self.terminal('../outside')
        result = self.saved_result('../outside')
        self.assertTrue(result['is_error'], result)
        self.assertIn('unavailable', formats.item_text(result))
        self.assertNotIn(_BODY, formats.item_text(result))

    async def test_acp_records_unknown_skill_as_failed_tool_result(self):
        """ACP must report the same failed load instead of silently accepting its text."""
        _advertised, messages = await self.acp('../outside')
        result = self.saved_result('../outside')
        self.assertTrue(result['is_error'], result)
        self.assertIn('unavailable', formats.item_text(result))
        self.assertTrue(any(message.get('params', {}).get('update', {}).get('status') == 'failed'
                            for message in messages), messages)

    def install_session(self):
        session = Session(shell_cwd=self.workspace)
        self.addCleanup(session.response_headers.close)
        self.enterContext(mock.patch.object(loki, '_DEFAULT_SESSION', session))
        self.enterContext(mock.patch.object(loki, 'LOKI_CONFIG_DIR', self.config))
        return session

    async def test_model_catalogue_is_fresh_per_turn_without_synthetic_conversation(self):
        """All request routes advertise skills without rewriting user text or continuation history."""
        for frontend in ['terminal', 'acp', 'headless']:
            with self.subTest(frontend=frontend):
                self.write_skill(_DESCRIPTION)
                session = self.install_session()
                session.runtime_config = connections.build_config_from_env(
                    credentials=CredentialStore(self.environment))
                transcript = [formats.message_item('user', '/' + _NAME + ' ' + _ARGUMENT)]
                session.transcript_items = transcript
                original_user = copy.deepcopy(transcript[0])
                received = []
                descriptions = []

                async def model(items, tools=None, *arguments, **kwargs):
                    received.append(copy.deepcopy(items))
                    descriptions.append(next(tool['function']['description'] for tool in tools
                                             if tool.get('function', {}).get('name') == 'Skill'))
                    if len(received) == 1:
                        self.write_skill('Changed after first request')
                        return [formats.tool_call_item('load', 'Skill', {'skill': _NAME, 'args': _ARGUMENT})]
                    return [formats.message_item('assistant', 'complete')]

                async def run_turn():
                    if frontend == 'terminal':
                        await terminal_frontend.run_terminal_turn_async(transcript)
                    elif frontend == 'acp':
                        await Worker(session, lambda _message: None)._run_turn(
                            lambda _event: None, loki.capture_turn_settings())
                    else:
                        await loki.run_tool_loop_async(transcript, allowed={'Skill'})

                with mock.patch.object(loki, 'async_chat_completion', new=model), \
                        mock.patch.object(terminal_frontend, 'async_chat_completion', new=model), \
                        mock.patch.object(terminal_frontend, '_terminal_agent_event'):
                    await run_turn()
                    self.assertEqual(received[0], [original_user])
                    self.assertEqual(received[1], transcript[:-1])
                    self.assertEqual(transcript[0], original_user)
                    self.assertEqual([event['type'] for event in transcript],
                                     ['message', 'model_response', 'tool_result', 'model_response'])
                    self.assertIn(_NAME, descriptions[0])
                    self.assertIn(_DESCRIPTION, descriptions[0])
                    self.assertEqual(descriptions[0], descriptions[1])
                    transcript.append(formats.message_item('user', 'Another task'))
                    descriptions.clear()
                    received.clear()

                    async def next_turn(items, tools=None, *arguments, **kwargs):
                        received.append(copy.deepcopy(items))
                        descriptions.append(next(tool['function']['description'] for tool in tools
                                                 if tool.get('function', {}).get('name') == 'Skill'))
                        return [formats.message_item('assistant', 'next')]

                    with mock.patch.object(loki, 'async_chat_completion', new=next_turn), \
                            mock.patch.object(terminal_frontend, 'async_chat_completion', new=next_turn):
                        await run_turn()
                    self.assertEqual(formats.item_text(received[0][-1]), 'Another task')
                    self.assertIn('Changed after first request', descriptions[0])

    async def test_loaded_skill_does_not_grant_edit_permissions_in_explore(self):
        """Loading instructions is read-only; their contents cannot authorize a later Write."""
        session = self.install_session()
        session.agent_mode = 'explore'
        transcript = [formats.message_item('user', 'Read the skill and change files?')]
        replies = [
            [formats.tool_call_item('load', 'Skill', {'skill': _NAME}),
             formats.tool_call_item('write', 'Write', {'file_path': 'unexpected', 'content': 'not allowed'})],
            [formats.message_item('assistant', 'complete')],
        ]

        async def model(items, *, codex_turn_state):
            return replies.pop(0)

        await loki.run_tool_loop_async(transcript, allowed=loki.EXPLORE_TOOLS, chat_fn=model)
        results = {event['call_id']: event for event in transcript if event.get('type') == 'tool_result'}
        self.assertFalse(results['load']['is_error'])
        self.assertTrue(results['write']['is_error'])
        self.assertEqual(os.listdir(self.workspace), [])
