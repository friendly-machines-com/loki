import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import loki
from loki_agent.sessions import Session


class GlobTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('rg'), 'ripgrep required')
    def test_advertised_options_discover_read_and_reap_real_jobs(self):
        definition = next(tool['function'] for tool in loki.TOOLS
                          if tool['function']['name'] == 'Glob')
        for name in ('hidden', 'no_ignore'):
            prop = definition['parameters']['properties'][name]
            self.assertEqual(prop['type'], 'boolean')
            self.assertIs(prop['default'], False)
            self.assertNotIn(name, definition['parameters']['required'])

        for asynchronous in (False, True):
            for options in ({}, {'hidden': False, 'no_ignore': False},
                            {'hidden': False, 'no_ignore': True},
                            {'hidden': True, 'no_ignore': False},
                            {'hidden': True, 'no_ignore': True}):
                with self.subTest(asynchronous=asynchronous, options=options), \
                        tempfile.TemporaryDirectory() as directory:
                    workspace = Path(directory) / 'workspace'
                    workspace.mkdir()
                    contents = {
                        '.ignore': '/ignored/\n/.loki/\n',
                        'visible/artifact.txt': 'visible artifact',
                        '.hidden/artifact.txt': 'hidden artifact',
                        'ignored/artifact.txt': 'ignored artifact',
                        '.loki/chats/artifact.txt': 'hidden ignored artifact',
                        '.loki/chats/chat.json': '{"artifact":"chat"}',
                    }
                    for index, (name, text) in enumerate(contents.items()):
                        path = workspace / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(text, encoding='utf-8')
                        # Independent, distinct times make ordering observable.
                        stamp = 1_000_000_000 + index
                        os.utime(path, (stamp, stamp))
                    manager = loki.JobManager(str(Path(directory) / 'jobs'))
                    session = Session(shell_cwd=str(workspace),
                                      job_manager=manager)
                    hidden = options.get('hidden', False)
                    no_ignore = options.get('no_ignore', False)
                    expected = ['visible/artifact.txt']
                    if hidden:
                        expected += ['.hidden/artifact.txt']
                    if no_ignore:
                        expected += ['ignored/artifact.txt']
                        if hidden:
                            expected += ['.loki/chats/artifact.txt']
                    order = list(contents)
                    expected.sort(key=order.index, reverse=True)

                    def search(pattern, path=workspace):
                        arguments = {'pattern': pattern, 'path': str(path),
                                     **options}
                        if asynchronous:
                            return asyncio.run(loki._handle_glob_async(
                                arguments, {'cancel_event': asyncio.Event()}))
                        return loki._handle_glob(arguments)

                    with mock.patch.object(loki, '_DEFAULT_SESSION', session):
                        # A positive glob can override ignore/hidden rules
                        # for matching files. Test directory traversal with a
                        # leaf-only glob that does not whitelist directories.
                        result = search('*.txt')
                        lines = result.splitlines()
                        self.assertEqual(lines[1:4], [
                            f'num_files: {len(expected)}',
                            'truncated: false', '[filenames]'])
                        self.assertEqual(lines[4:],
                                         [str(workspace / name)
                                          for name in expected])
                        for name in expected:
                            observed = loki.run_read(str(workspace / name))
                            for line in contents[name].splitlines():
                                self.assertIn(line, observed)

                        # Explicit slash patterns still require both switches;
                        # explicitly naming the hidden root does not.
                        chat = workspace / '.loki' / 'chats' / 'chat.json'
                        slash = search('.loki/**/*')
                        if hidden and no_ignore:
                            self.assertEqual(slash.splitlines()[4:],
                                             [str(chat),
                                              str(chat.parent / 'artifact.txt')])
                        else:
                            self.assertTrue(slash.startswith('No files matched'))
                        explicit = search('*.json', chat.parent)
                        self.assertEqual(explicit.splitlines()[4:], [str(chat)])
                        empty = search('chat-*.json', chat.parent)
                        self.assertTrue(empty.startswith('No files matched'))
                        self.assertEqual('hidden=true' in empty, not hidden)
                        self.assertEqual('no_ignore=true' in empty, not no_ignore)

                    self.assertEqual(len(manager.jobs), 4)
                    for job in manager.jobs.values():
                        self.assertIsNotNone(job.process)
                        self.assertIn(job.process.returncode, (0, 1))
                        self.assertEqual(job.exit_code, job.process.returncode)
                        self.assertEqual(job.status, 'exited')
                        metadata = json.loads(
                            Path(job.metadata_path).read_text(encoding='utf-8'))
                        self.assertEqual(metadata['exit_code'], job.exit_code)
                        self.assertEqual(metadata['cwd'], str(workspace))
                        self.assertEqual(metadata['status'], 'exited')
                        if os.name == 'posix':
                            self.assertTrue(job.process._transport.is_closing())

    @unittest.skipUnless(shutil.which('rg'), 'ripgrep required')
    def test_cancel_reaches_a_running_search_and_reaps_its_child(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = loki.JobManager(str(Path(directory) / 'jobs'))
            session = Session(shell_cwd=directory, job_manager=manager)

            async def scenario():
                cancel = asyncio.Event()
                run_exec = manager.run_exec

                async def gated_search(argv, **kwargs):
                    # Substitute only the external search program with a
                    # readiness-gated child; dispatch, jobs and cancellation
                    # stay real. This makes active cancellation deterministic.
                    return await run_exec([
                        sys.executable, '-u', '-c',
                        "import time; print('search-ready'); time.sleep(30)"],
                        **kwargs)

                with mock.patch.object(manager, 'run_exec', gated_search):
                    search = asyncio.create_task(loki._handle_glob_async(
                        {'pattern': '*.txt', 'path': directory},
                        {'cancel_event': cancel}))
                    try:
                        async def ready():
                            while True:
                                for job in manager.jobs.values():
                                    if (job.process is not None
                                            and Path(job.stdout_path).exists()
                                            and 'search-ready' in Path(
                                                job.stdout_path).read_text()):
                                        return job
                                await asyncio.sleep(0.01)

                        job = await asyncio.wait_for(ready(), 5)
                        self.assertIsNone(job.process.returncode)
                        self.assertFalse(search.done())
                        cancel.set()
                        result = await asyncio.wait_for(search, 6)
                        self.assertEqual(
                            result, 'Error: glob search was cancelled by the user')
                        self.assertIsNotNone(job.process.returncode)
                        self.assertEqual(job.status, 'cancelled')
                        metadata = json.loads(Path(job.metadata_path).read_text())
                        self.assertEqual(metadata['status'], 'cancelled')
                    finally:
                        cancel.set()
                        search.cancel()
                        await asyncio.gather(search, return_exceptions=True)
                        # Backstop after evidence assertions, also covering a
                        # broken cancellation-forwarding implementation.
                        for job in manager.jobs.values():
                            if job.process is not None:
                                if job.process.returncode is None:
                                    job.process.kill()
                                await asyncio.wait_for(job.process.wait(), 3)

            with mock.patch.object(loki, '_DEFAULT_SESSION', session):
                asyncio.run(scenario())


if __name__ == '__main__':
    unittest.main()
