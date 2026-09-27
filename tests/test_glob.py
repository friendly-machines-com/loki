import asyncio
import pathlib
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock

from loki_agent import loki


class GlobTests(unittest.TestCase):
    def test_flags_and_empty_result_guidance(self):
        for hidden in (False, True):
            for no_ignore in (False, True):
                with self.subTest(hidden=hidden, no_ignore=no_ignore):
                    manager = types.SimpleNamespace(run_exec=mock.AsyncMock(
                        return_value=(types.SimpleNamespace(exit_code=1),
                                      'completed', '', '')))
                    with mock.patch.object(loki, '_find_rg_binary', return_value='rg'), \
                            mock.patch.object(loki, 'current_job_manager', return_value=manager):
                        result = loki.run_glob('nothing', tempfile.gettempdir(),
                                               hidden=hidden, no_ignore=no_ignore)
                    args = manager.run_exec.call_args.args[0]
                    self.assertEqual('--hidden' in args, hidden)
                    self.assertEqual('--no-ignore' in args, no_ignore)
                    self.assertEqual('hidden=true' in result, not hidden)
                    self.assertEqual('no_ignore=true' in result, not no_ignore)
                    self.assertTrue(result.startswith('No files matched'))

    def test_handler_forwarding_and_defaults(self):
        for options in ({}, {'hidden': True, 'no_ignore': True}):
            args = {'pattern': '*.py', 'path': '/workspace', **options}
            expected = {'hidden': options.get('hidden', False),
                        'no_ignore': options.get('no_ignore', False)}
            with mock.patch.object(loki, 'run_glob', return_value='ok') as run:
                self.assertEqual(loki._handle_glob(args), 'ok')
                run.assert_called_once_with('*.py', '/workspace', **expected)
            cancel = asyncio.Event()
            with mock.patch.object(loki, 'run_glob_async', new_callable=mock.AsyncMock) as run:
                run.return_value = 'ok'
                self.assertEqual(asyncio.run(loki._handle_glob_async(
                    args, {'cancel_event': cancel})), 'ok')
                run.assert_awaited_once_with('*.py', '/workspace',
                                            cancel_event=cancel, **expected)

    def test_schema(self):
        definition = next(t['function'] for t in loki.TOOLS
                          if t['function']['name'] == 'Glob')
        for name in ('hidden', 'no_ignore'):
            prop = definition['parameters']['properties'][name]
            self.assertEqual(prop['type'], 'boolean')
            self.assertIs(prop['default'], False)
            self.assertNotIn(name, definition['parameters']['required'])

    @unittest.skipUnless(shutil.which('rg'), 'ripgrep required')
    def test_hidden_and_ignored_directory_with_slash_pattern(self):
        async def run_exec(args, **kwargs):
            # Model the job manager running in the fixture workspace.
            result = subprocess.run(args, capture_output=True, text=True, cwd=directory)
            return (types.SimpleNamespace(exit_code=result.returncode),
                    'completed', result.stdout, result.stderr)

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / '.ignore').write_text('/.loki/\n')
            chats = root / '.loki' / 'chats'
            chats.mkdir(parents=True)
            target = chats / 'chat.json'
            target.write_text('{}')
            manager = types.SimpleNamespace(run_exec=run_exec)
            with mock.patch.object(loki, 'current_job_manager', return_value=manager):
                for hidden in (False, True):
                    for no_ignore in (False, True):
                        with self.subTest(hidden=hidden, no_ignore=no_ignore):
                            result = loki.run_glob('.loki/**/*', directory,
                                                   hidden=hidden, no_ignore=no_ignore)
                            self.assertEqual(str(target) in result, hidden and no_ignore)
                result = loki.run_glob('chat-*.json', str(chats))
                self.assertTrue(result.startswith('No files matched'))
                result = loki.run_glob('*.json', str(chats))
                self.assertIn(str(target), result)


if __name__ == '__main__':
    unittest.main()
