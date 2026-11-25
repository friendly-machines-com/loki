import asyncio
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from job_fixtures import captured_result, job_manager
from response_header_fixtures import setUpModule  # noqa: F401 - unittest hook
from loki_agent import loki, process_outputs, tool_contracts
from loki_agent.sessions import Session
from loki_agent import files as file_operations


class GlobTests(unittest.TestCase):
    def test_advertised_options_discover_read_and_reap_real_jobs(self):
        definition = next(tool['function'] for tool in loki.TOOLS
                          if tool['function']['name'] == 'Glob')
        for name in ['hidden', 'no_ignore']:
            prop = definition['parameters']['properties'][name]
            self.assertEqual(prop['type'], 'boolean')
            self.assertIs(prop['default'], False)
            self.assertNotIn(name, definition['parameters']['required'])

        for options in [{}, {'hidden': False, 'no_ignore': False},
                        {'hidden': False, 'no_ignore': True},
                        {'hidden': True, 'no_ignore': False},
                        {'hidden': True, 'no_ignore': True}]:
            with self.subTest(options=options), \
                    tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as job_scope:
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
                manager = job_manager(str(Path(directory) / 'jobs'), cleanup=job_scope.callback)
                session = Session(shell_cwd=str(workspace),
                                  job_manager=manager)
                self.addCleanup(session.response_headers.close)
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

                run_exec = manager.run_exec

                def external_search(argv, **kwargs):
                    # Assert the requested search, then have a child emit the
                    # records ripgrep would return. The assertions below still
                    # require newest-first output; a real search tests it too.
                    pattern, search_root = argv[-2:]
                    flags = ['--hidden'] if hidden else []
                    if no_ignore:
                        flags.append('--no-ignore')
                    self.assertEqual(argv, [
                        'test-search-program', '--files', '--color=never', '--sortr', 'modified', '--null',
                        *flags, '--glob', pattern, search_root])
                    if pattern == '*.txt':
                        matches = [str(workspace / name) for name in expected]
                    elif pattern == '.loki/**/*':
                        matches = ([str(workspace / '.loki/chats/chat.json'),
                                    str(workspace / '.loki/chats/artifact.txt')]
                                   if hidden and no_ignore else [])
                    elif pattern == '*.json':
                        matches = [str(Path(search_root) / 'chat.json')]
                    else:
                        self.assertEqual(pattern, 'chat-*.json')
                        matches = []
                    return run_exec([
                        sys.executable, '-c',
                        'import json, os, sys\n'
                        'names = json.loads(sys.argv[1])\n'
                        'for name in names:\n'
                        '    sys.stdout.buffer.write(os.fsencode(name) + b"\\0")\n'
                        'sys.exit(0 if names else 1)',
                        json.dumps(matches)], **kwargs)

                def search(pattern, path=workspace):
                    arguments = {'pattern': pattern, 'path': str(path),
                                 **options}
                    result = asyncio.run(loki.dispatch_tool_async(
                        'Glob', arguments, extra_context={'cancel_event': asyncio.Event()}))
                    self.assertTrue(result['ok'], result)
                    return result['content']

                with mock.patch.object(loki, '_DEFAULT_SESSION', session), \
                        mock.patch.object(file_operations.executables, 'RIPGREP', 'test-search-program'), \
                        mock.patch.object(manager, 'run_exec', external_search):
                    # Fixture responses exercise option forwarding, ordering
                    # and Read without depending on an installed search tool.
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

                    # Exercise empty results and explicit-root requests too.
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
                    with open(job.metadata_path, encoding='utf-8') as stream:
                        metadata = json.load(stream)
                    self.assertEqual(metadata['exit_code'], job.exit_code)
                    self.assertEqual(metadata['cwd'], str(workspace))
                    self.assertEqual(metadata['status'], 'exited')
                    if os.name == 'posix':
                        self.assertTrue(job.process._transport.is_closing())

    def test_missing_search_targets_are_checked_by_the_external_program(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as job_scope:
            target = str(Path(directory) / 'missing')
            manager = job_manager(str(Path(directory) / 'jobs'), cleanup=job_scope.callback)
            session = Session(shell_cwd=directory, job_manager=manager)
            self.addCleanup(session.response_headers.close)
            for name in ['Glob', 'Grep']:
                with self.subTest(tool=name):
                    execute = mock.Mock(return_value=captured_result(mock.Mock(exit_code=2), err=b'target unavailable'))
                    with mock.patch.object(loki, '_DEFAULT_SESSION', session), \
                            mock.patch.object(file_operations.executables, 'RIPGREP', 'test-search-program'), \
                            mock.patch.object(manager, 'run_exec', execute):
                        result = asyncio.run(loki.dispatch_tool_async(name, {'pattern': 'needle', 'path': target}))
                    execute.assert_called_once()
                    self.assertEqual(execute.call_args.args[0][-1], target)
                    self.assertFalse(result['ok'], result)
                    self.assertIn('ripgrep failed with exit code 2\ntarget unavailable', result['content'])

    def test_cancel_reaches_a_running_search_and_reaps_its_child(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as job_scope:
            manager = job_manager(str(Path(directory) / 'jobs'), cleanup=job_scope.callback)
            session = Session(shell_cwd=directory, job_manager=manager)
            self.addCleanup(session.response_headers.close)

            async def scenario():
                cancel = asyncio.Event()
                run_exec = manager.run_exec

                def gated_search(argv, **kwargs):
                    # Substitute only the external search program with a
                    # readiness-gated child; dispatch, jobs and cancellation
                    # stay real. This makes active cancellation deterministic.
                    return run_exec([
                        sys.executable, '-u', '-c',
                        "import time\nprint('search-ready')\nwhile True: time.sleep(1)"],
                        **kwargs)

                with mock.patch.object(manager, 'run_exec', gated_search), \
                        mock.patch.object(file_operations.executables, 'RIPGREP', 'test-search-program'):
                    search = asyncio.create_task(loki.dispatch_tool_async(
                        'Glob', {'pattern': '*.txt', 'path': directory},
                        extra_context={'cancel_event': cancel}))
                    try:
                        async def ready():
                            while True:
                                for job in manager.jobs.values():
                                    if job.process is not None and os.path.exists(job.stdout_path):
                                        with open(job.stdout_path) as stream:
                                            if 'search-ready' in stream.read():
                                                return job
                                self.assertFalse(search.done(), 'search ended before readiness')
                                await asyncio.sleep(0.01)

                        job = await ready()
                        self.assertIsNone(job.process.returncode)
                        self.assertFalse(search.done())
                        cancel.set()
                        result = await search
                        self.assertFalse(result['ok'], result)
                        self.assertIn('glob search was cancelled by the user', result['content'])
                        self.assertIsNotNone(job.process.returncode)
                        self.assertEqual(job.status, 'cancelled')
                        with open(job.metadata_path) as stream:
                            metadata = json.load(stream)
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
                                await job.process.wait()

            with mock.patch.object(loki, '_DEFAULT_SESSION', session):
                asyncio.run(scenario())


@unittest.skipUnless(shutil.which(file_operations.executables.RIPGREP), 'requires ripgrep')
class SearchPreviewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.workspace = self.enterContext(tempfile.TemporaryDirectory())
        state = self.enterContext(tempfile.TemporaryDirectory())
        self.manager = job_manager(state, cleanup=self.addCleanup, async_cleanup=self.addAsyncCleanup)
        session = Session(shell_cwd=self.workspace, job_manager=self.manager)
        self.addCleanup(session.response_headers.close)
        self.enterContext(mock.patch.object(loki, '_DEFAULT_SESSION', session))
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop('RIPGREP_CONFIG_PATH', None)

    def write_fixture(self, data=b'', *, prefix=None):
        with tempfile.NamedTemporaryFile(dir=self.workspace, prefix=prefix, suffix='.txt', delete=False) as file:
            file.write(data)
        return file.name

    async def search(self, tool, arguments, *, preview_chars=None):
        run_exec = self.manager.run_exec

        def run_ripgrep(argv, **kwargs):
            # Make the unsorted-by-time control deterministic. Glob's own
            # sort must override this order, which puts the oldest file first.
            return run_exec([argv[0], '--sort=path', *argv[1:]], **kwargs)

        limit = tool_contracts.BASH_MAX_OUTPUT_CHARS if preview_chars is None else preview_chars
        with (mock.patch.object(self.manager, 'run_exec', run_ripgrep),
              mock.patch.object(tool_contracts, 'BASH_MAX_OUTPUT_CHARS', limit)):
            result = await loki.dispatch_tool_async(tool, arguments)
        self.assertTrue(result['ok'], result)
        return result['content']

    async def test_glob_counts_all_files_and_keeps_newest_when_the_preview_is_cut(self):
        paths = []
        for index in range(102):
            name = self.write_fixture(prefix=f'ranked-{index:03d}-')
            stamp = 1_700_000_000 + 100 * index
            os.utime(name, times=(stamp, stamp))
            paths.append(name)

        complete_page = await self.search('Glob', {'pattern': '*.txt'})
        self.assertIn('num_files: 102\ntruncated: true\n', complete_page)
        self.assertNotIn('[output truncated]', complete_page)
        self.assertEqual(complete_page.split('\n[filenames]\n', 1)[1], '\n'.join(reversed(paths[2:])))

        short_page = await self.search('Glob', {'pattern': '*.txt'}, preview_chars=len(paths[-1]))
        self.assertIn('num_files: 102\ntruncated: true\n[output truncated]\n', short_page)
        self.assertEqual(short_page.split('\n[filenames]\n', 1)[1], paths[-1])

    async def test_grep_offsets_from_the_start_even_across_a_large_record(self):
        first = b'needle ' + b'x' * (128 * 1024)
        later = ''.join(f'needle row-{index}\n' for index in range(1, 30)).encode('utf-8')
        source = self.write_fixture(first + b'\n' + later)
        arguments = {'pattern': 'needle', 'path': source, 'output_mode': 'content', 'offset': 1}
        match = f'{source}:2:needle row-1'
        page = await self.search('Grep', {**arguments, 'head_limit': 1}, preview_chars=len(match))
        self.assertIn('num_entries: 30\napplied_offset: 1\napplied_limit: 1\ntruncated: true\n', page)
        self.assertNotIn('[output truncated]', page)
        self.assertEqual(page.split('\n[results]\n', 1)[1], match)

        unlimited = await self.search('Grep', {**arguments, 'head_limit': 0}, preview_chars=len(match))
        self.assertIn('applied_limit: 0\ntruncated: true\n[output truncated]\n', unlimited)
        self.assertEqual(unlimited.split('\n[results]\n', 1)[1], match)

    async def test_grep_keeps_prefix_of_oversized_selected_record(self):
        for width in [16, 128 * 1024]:
            with self.subTest(width=width):
                source = self.write_fixture(b'needle ' + b'x' * width + b'\nneedle later\n')
                prefix = f'{source}:1:needle xxx'
                page = await self.search('Grep', {
                    'pattern': 'needle', 'path': source, 'output_mode': 'content', 'head_limit': 2,
                }, preview_chars=len(prefix))
                self.assertIn('num_entries: 2\napplied_offset: 0\napplied_limit: 2\ntruncated: true\n', page)
                self.assertIn('\n[output truncated]\n[results]\n', page)
                self.assertEqual(page.split('\n[results]\n', 1)[1], prefix)

    async def test_search_rejects_premature_eof_and_closes_capture(self):
        source = self.write_fixture(b'needle one\nneedle two\n')
        capture = None
        read = process_outputs.capture_chunks

        def read_short_capture(stream):
            nonlocal capture
            capture = stream
            size = os.fstat(stream.fileno()).st_size
            chunks = read(stream)
            # Report a larger size for this sample only. The real reader will
            # then observe EOF before reaching it, after yielding some results.
            with mock.patch.object(process_outputs.os, 'fstat', return_value=mock.Mock(st_size=size + 1)):
                first = next(chunks)
            yield first
            yield from chunks

        with mock.patch.object(process_outputs, 'capture_chunks', read_short_capture):
            result = await loki.dispatch_tool_async('Grep', {
                'pattern': 'needle', 'path': source, 'output_mode': 'content',
            })
        self.assertFalse(result['ok'], result)
        self.assertIn('Search output ended before it could be read in full.', result['content'])
        self.assertNotIn('[results]', result['content'])
        self.assertIsNotNone(capture)
        self.assertTrue(capture.closed)
        self.assertEqual(next(iter(self.manager.jobs.values())).exit_code, 0)

    async def test_cancelling_a_page_read_releases_its_capture(self):
        witness = b'needle reading this capture'
        source = self.write_fixture(witness + b' ' + b'x' * (256 * 1024) + b'\n')
        cancel = asyncio.Event()
        capture = None
        read = process_outputs.capture_chunks

        def read_until_cancel(stream):
            nonlocal capture
            for data in read(stream):
                if witness in data:
                    capture = stream
                    asyncio.get_running_loop().call_soon(cancel.set)
                yield data

        with mock.patch.object(process_outputs, 'capture_chunks', read_until_cancel):
            result = await loki.dispatch_tool_async('Grep', {
                'pattern': 'needle', 'path': source, 'output_mode': 'content',
            }, extra_context={'cancel_event': cancel})
        self.assertFalse(result['ok'], result)
        self.assertIn('grep search was cancelled by the user', result['content'])
        self.assertIsNotNone(capture)
        self.assertTrue(capture.closed)
        self.assertEqual(next(iter(self.manager.jobs.values())).exit_code, 0)

    @unittest.skipUnless(os.name == 'posix', 'requires newlines and colons in filenames')
    async def test_grep_filename_and_count_pages_treat_a_newline_name_as_one_entry(self):
        source = self.write_fixture(b'needle twice: needle\n', prefix='name:\nwith-newline-')
        glob = await self.search('Glob', {'pattern': '*.txt'})
        self.assertIn('num_files: 1\ntruncated: false\n', glob)
        self.assertEqual(glob.split('\n[filenames]\n', 1)[1], source)
        for mode, expected in [('files_with_matches', source), ('count', f'{source}:2')]:
            with self.subTest(mode=mode):
                args = {'pattern': 'needle', 'path': source, 'output_mode': mode, 'head_limit': 1}
                page = await self.search('Grep', args)
                self.assertIn('num_entries: 1\n', page)
                self.assertIn('truncated: false\n', page)
                self.assertEqual(page.split('\n[results]\n', 1)[1], expected)
                self.assertTrue((await self.search('Grep', {**args, 'offset': 1})).startswith('No matches'))


if __name__ == '__main__':
    unittest.main()
