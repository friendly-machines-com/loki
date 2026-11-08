"""Ordinary-child authority and unpublished POSIX launch ownership.

The Linux cases execute both sanctioned dispatchers. The transport regression
keeps an independent stdout writer alive until after cancellation assertions;
closing that witness first would conceal an EOF-draining cleanup bug.
"""

import asyncio
import contextlib
import errno
import json
import os
import shlex
import shutil
import signal
import sys
import tempfile
import unittest
from unittest import mock

from loki_entrypoints import child_environment, entrypoint
from loki_agent import credential_storages, host_process, private_files, response_headers, settings, tool_runtime
from process_lifecycle_fixtures import ProcessResources


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux ordinary-child mount view')
class OrdinaryStateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='loki-tool-state-')
        self.addCleanup(temporary.cleanup)
        self.state = temporary.name + os.sep + 'state'
        self.workspace = temporary.name + os.sep + 'workspace'
        os.mkdir(self.state)
        os.mkdir(self.workspace)
        self.directory = private_files.open_directory_tree(self.state)
        self.addCleanup(private_files.close, self.directory)
        self.other = self.state + os.sep + 'other-job.json'
        with open(self.other, 'w', encoding='ascii') as stream:
            stream.write('owner-original')

    async def spawn(self, *command, frontend='loki', **options):
        return await host_process.create_exec(
            *command, state_directory=self.directory,
            tool_command=[sys.executable, entrypoint(frontend), '--tool-exec'], **options)

    async def test_reads_and_live_owner_publication_are_allowed_but_mutation_is_denied(self):
        code = r'''
import json, os, stat, sys
state = sys.argv[1]
print('ready', file=sys.stderr, flush=True)
assert sys.stdin.buffer.read(1) == b'g'
result = {'read': open(state + os.sep + 'other-job.json').read(),
          'later': open(state + os.sep + 'later').read(), 'cwd': os.getcwd(), 'directories': []}
for name, operation in [
        ['write', lambda: open(state + os.sep + 'other-job.json', 'w').close()],
        ['create', lambda: open(state + os.sep + 'created', 'w').close()],
        ['unlink', lambda: os.unlink(state + os.sep + 'other-job.json')],
        ['chmod', lambda: os.chmod(state + os.sep + 'other-job.json', 0o600)],
        ['cwd-write', lambda: open('other-job.json', 'w').close()]]:
    try: operation()
    except OSError as error: result[name] = error.errno
    else: result[name] = 0
for name in os.listdir('/proc/self/fd'):
    try:
        if int(name) >= 3 and stat.S_ISDIR(os.fstat(int(name)).st_mode): result['directories'].append(name)
    except OSError: pass
os.write(1, b'abcdef')
os.lseek(1, 0, os.SEEK_SET)
os.ftruncate(1, 3)
print(json.dumps(result), file=sys.stderr, flush=True)
'''
        for frontend in ['loki', 'loki-acp']:
            with self.subTest(frontend=frontend), open(self.state + os.sep + 'own-output', 'w+b') as capture:
                process = await self.spawn(
                    sys.executable, '-c', code, self.state, frontend=frontend,
                    cwd=self.state, stdin=asyncio.subprocess.PIPE,
                    stdout=capture, stderr=asyncio.subprocess.PIPE)
                try:
                    self.assertEqual(await process.stderr.readline(), b'ready\n')
                    with open(self.other, 'w', encoding='ascii') as stream:
                        stream.write('owner-update')
                    with open(self.state + os.sep + 'later', 'w', encoding='ascii') as stream:
                        stream.write('published-after-start')
                    process.stdin.write(b'g')
                    await process.stdin.drain()
                    process.stdin.close()
                    await process.stdin.wait_closed()
                    errors = await process.stderr.read()
                    await process.wait()
                    self.assertEqual(process.returncode, 0, errors)
                    report = json.loads(errors)
                    self.assertEqual(report['read'], 'owner-update')
                    self.assertEqual(report['later'], 'published-after-start')
                    self.assertEqual(report['cwd'], self.state)
                    self.assertEqual(report['directories'], [])
                    for operation in ['write', 'create', 'unlink', 'chmod', 'cwd-write']:
                        self.assertEqual(report[operation], errno.EROFS, report)
                    capture.seek(0)
                    self.assertEqual(capture.read(), b'abc')
                finally:
                    await process.aclose()
        # A trusted owning launch has the same command interface but retains
        # its writer role; neither its argv nor the lifetime flag selects it.
        process = await self.spawn(
            sys.executable, '-c', 'import sys; open(sys.argv[1], "w").write("owning-writer")',
            self.other, owning=True, cwd=self.workspace)
        try:
            self.assertEqual(await process.wait(), 0)
            with open(self.other, encoding='ascii') as stream:
                self.assertEqual(stream.read(), 'owning-writer')
        finally:
            await process.aclose()

    async def test_real_runtime_preserves_a_credential_cover_nested_under_the_state_view(self):
        config = self.workspace + os.sep + 'config'
        state = config + os.sep + 'loki'
        credentials = state + os.sep + 'credentials'
        # Use the real store's private creation and validation, not fabricated
        # setup state. Its retained handle is not part of the launch fixture.
        storage = credential_storages.JsonCredentialStorage(credentials)
        storage.close()
        secret = credentials + os.sep + 'fixture-secret'
        with open(secret, 'w', encoding='ascii') as stream:
            stream.write('FAKE-SECRET')
        with open(state + os.sep + 'readable', 'w', encoding='ascii') as stream:
            stream.write('public-state')
        report_path = self.workspace + os.sep + 'overlap-report.json'
        script = self.workspace + os.sep + 'overlap.py'
        with open(script, 'w', encoding='ascii') as stream:
            stream.write('import json, sys\n'
                         'result = {"read": open(sys.argv[1]).read()}\n'
                         'try: open(sys.argv[1], "w").close()\n'
                         'except OSError as error: result["write"] = error.errno\n'
                         'else: result["write"] = 0\n'
                         'try: open(sys.argv[2]).read()\n'
                         'except OSError as error: result["secret"] = error.errno\n'
                         'else: result["secret"] = 0\n'
                         'with open(sys.argv[3], "w") as out: json.dump(result, out)\n')
        # Overlapping POSIX roots are valid operands. A nonrecursive state bind
        # must not drop the inherited credential mount and reveal this file.
        environment = child_environment(
            HOME=self.workspace, PATH=os.environ.get('PATH', ''),
            XDG_CONFIG_HOME=config, XDG_STATE_HOME=config,
            LOKI_PROVIDER='dummy', LOKI_API_BASE='http://dummy.invalid/v1', LOKI_MODEL='dummy')
        command = shlex.join([sys.executable, script, state + os.sep + 'readable', secret, report_path])
        process = await asyncio.create_subprocess_exec(
            entrypoint('loki'), '--dangerously-skip-permissions',
            cwd=self.workspace, env=environment, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        resources = ProcessResources(process)
        try:
            output, errors = await process.communicate(('!' + command + '\n/quit\n').encode())
            self.assertEqual(process.returncode, 0, errors)
            self.assertTrue(os.path.exists(report_path), [output, errors])
            with open(report_path, encoding='ascii') as stream:
                self.assertEqual(json.load(stream), {'read': 'public-state', 'write': errno.EROFS,
                                                     'secret': errno.EACCES})
            with open(secret, encoding='ascii') as stream:
                self.assertEqual(stream.read(), 'FAKE-SECRET')
        finally:
            # This stdlib driver is fixture-owned, not Loki's worker transport.
            await resources.cleanup()

    async def test_second_exec_keeps_only_explicit_descriptor_loans(self):
        script = 'import os,sys; assert os.read(int(sys.argv[1]),4)==b"loan"; print("borrowed")'
        for frontend in ['loki', 'loki-acp']:
            with self.subTest(frontend=frontend), open(self.workspace + os.sep + 'loan', 'w+b') as loan:
                loan.write(b'loan')
                loan.flush()
                loan.seek(0)
                process = await self.spawn(
                    sys.executable, '-c', script, str(loan.fileno()), frontend=frontend,
                    pass_fds=[loan.fileno()], stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE)
                try:
                    output, errors, _code = await asyncio.gather(
                        process.stdout.read(), process.stderr.read(), process.wait())
                    self.assertEqual(process.returncode, 0, errors)
                    self.assertEqual(output.strip(), b'borrowed')
                    self.assertFalse(os.get_inheritable(loan.fileno()))
                finally:
                    await process.aclose()

    async def _assert_missing_descriptor_manifest_is_refused(self, frontend):
        marker = self.workspace + os.sep + 'legacy-target-ran'
        reader, writer = os.pipe()
        process = None
        try:
            # Normal producers always add the list. Invoke the real receiver
            # directly to verify that an old-format launch cannot execute.
            process = await host_process.create_posix_process(
                sys.executable, entrypoint(frontend), '--tool-exec',
                str(self.directory), str(writer), '--cwd', self.workspace, '--',
                sys.executable, '-c', 'import sys; open(sys.argv[1], "w").close()', marker,
                pass_fds=[self.directory, writer], stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            os.close(writer)
            writer = None
            _output, errors, _code = await asyncio.gather(
                process.stdout.read(), process.stderr.read(), process.wait())
            self.assertFalse(os.path.exists(marker), 'launch without a descriptor list executed the target')
            self.assertEqual(process.returncode, 2, errors)
            with os.fdopen(reader, 'r', encoding='ascii') as status:
                reader = None
                report = json.load(status)
            self.assertEqual(report['errno'], errno.EIO)
            self.assertEqual(report['stage'], 'state')
            self.assertIn('inherited descriptors are missing', report['message'])
        finally:
            if process is not None:
                await process.aclose()
            for descriptor in [reader, writer]:
                if descriptor is not None:
                    os.close(descriptor)

    async def test_terminal_bootstrap_refuses_missing_descriptor_manifest(self):
        await self._assert_missing_descriptor_manifest_is_refused('loki')

    async def test_acp_bootstrap_refuses_missing_descriptor_manifest(self):
        await self._assert_missing_descriptor_manifest_is_refused('loki-acp')

    async def test_target_exec_and_cwd_errors_keep_the_parent_spawn_contract(self):
        missing = self.workspace + os.sep + 'missing'
        for command, cwd, filename in [
                [[missing], self.workspace, missing],
                [[sys.executable, '-c', 'raise SystemExit(0)'], missing, missing]]:
            with self.subTest(command=command, cwd=cwd):
                with self.assertRaises(FileNotFoundError) as caught:
                    await self.spawn(*command, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                     stderr=asyncio.subprocess.PIPE)
                self.assertEqual(caught.exception.errno, errno.ENOENT)
                self.assertEqual(caught.exception.filename, filename)

    async def test_bad_state_input_never_retries_the_target_without_protection(self):
        marker = self.workspace + os.sep + 'ran'
        selected = os.open(self.other, os.O_RDONLY | os.O_CLOEXEC)
        try:
            with self.assertRaises(OSError):
                await host_process.create_exec(
                    sys.executable, '-c', 'import sys; open(sys.argv[1], "w").close()', marker,
                    state_directory=selected,
                    tool_command=[sys.executable, entrypoint('loki'), '--tool-exec'],
                    cwd=self.workspace, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE)
        finally:
            os.close(selected)
        self.assertFalse(os.path.exists(marker))

    async def test_spawn_options_cannot_replace_the_trusted_bootstrap(self):
        marker = self.workspace + os.sep + 'unexpected-command'
        for options in [{'shell': True}, {'executable': sys.executable},
                        {'close_fds': False}, {'restore_signals': False}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                await self.spawn(
                    sys.executable, '-c', 'import sys; open(sys.argv[1], "w").close()', marker,
                    cwd=self.workspace, **options)
        self.assertFalse(os.path.exists(marker))

    async def test_lazy_manager_uses_the_session_root_after_its_name_moves(self):
        # Import the core under test-owned locations, then supply an explicit
        # Session: a unit runner has no sanctioned frontend identity of its own.
        with mock.patch.dict(os.environ, {'XDG_STATE_HOME': self.workspace}):
            from loki_agent import loki
        store = response_headers.Store(self.state + os.sep + 'response-headers.json')
        session = loki.Session(response_headers=store)
        moved = self.workspace + os.sep + 'moved-state'
        os.rename(self.state, moved)
        os.mkdir(self.state)
        try:
            with mock.patch.object(loki, '_DEFAULT_SESSION', session), \
                    mock.patch.object(loki, 'LOKI_JOB_STATE_DIR', self.state + os.sep + 'jobs'), \
                    mock.patch.object(loki, '_tool_bootstrap_command', return_value=[
                        sys.executable, entrypoint('loki'), '--tool-exec']):
                manager = loki.current_job_manager()
                code = ('import sys\ntry: open(sys.argv[1], "w").close()\n'
                        'except OSError as error: print(error.errno)\nelse: print(0)')
                job, _status, output, errors = await manager.run_foreground(
                    [sys.executable, '-c', code, moved + os.sep + 'other-job.json'],
                    'selected-root check', None, shell=False, cwd=self.workspace)
                self.assertEqual(job.exit_code, 0, errors)
                self.assertEqual(output.strip(), str(errno.EROFS))
                self.assertTrue(os.path.isdir(moved + os.sep + 'jobs' + os.sep + manager.session_id))
                self.assertFalse(os.path.exists(self.state + os.sep + 'jobs'))
        finally:
            await session.close_state()

    async def test_all_configured_hook_phases_use_the_explicit_root_without_xdg(self):
        script = self.workspace + os.sep + 'hook.py'
        with open(script, 'w', encoding='ascii') as stream:
            stream.write('import json, os, sys\n'
                         'payload = json.load(sys.stdin)\n'
                         'try: open(sys.argv[1], "w").close()\n'
                         'except OSError as error: number = error.errno\n'
                         'else: number = 0\n'
                         'with open(sys.argv[2] + os.sep + payload["event"] + ".json", "w") as out:\n'
                         '    json.dump({"errno": number, "xdg": os.environ.get("XDG_STATE_HOME"), "cwd": os.getcwd()}, out)\n'
                         'print("{}")\n')
        command = [sys.executable, script, self.other, self.workspace]
        sections = ['pre_tool_call', 'pre_tool_gate', 'post_tool_call', 'turn_end', 'turn_error']
        # Decode actual user configuration, then bind its commands to the
        # explicit creator. Shell-context cwd deliberately names the protected
        # root: only configured cwd may choose the target's working directory.
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=self.workspace, delete=False) as resource:
            for phase in sections:
                resource.write(f'[hook.{phase}.{phase}]\ncwd = {self.workspace}\n')
                for index, argument in enumerate(command):
                    resource.write(f'command.{index} = {argument}\n')
            filename = resource.name
        configuration = await settings.load_settings(files=[filename], strict=True)
        pipeline = tool_runtime.build_hook_pipeline(configuration.hooks, create_exec=self.spawn)
        invocation = tool_runtime.ToolInvocation('c', 'TodoRead', {}, {}, {},
                                                 {'type': 'object'}, cwd=self.state)
        prepared = await pipeline.prepare(invocation)
        self.assertIsNone(prepared.denied_reason)
        await pipeline.finish(prepared, tool_runtime.ToolOutcome('ok', True, True, 'ok'))
        await pipeline.finish_turn({'event': 'turn_end', 'cwd': self.state})
        decision = await pipeline.recover_turn(
            {'event': 'turn_error', 'cwd': self.state}, cancel_event=asyncio.Event())
        self.assertIsNone(decision.delay_seconds)
        for name in sections:
            with open(self.workspace + os.sep + name + '.json', encoding='ascii') as stream:
                self.assertEqual(json.load(stream), {'errno': errno.EROFS, 'xdg': None, 'cwd': self.workspace})
        with open(self.other, encoding='ascii') as stream:
            self.assertEqual(stream.read(), 'owner-original')

    async def test_second_exec_preserves_subprocess_signal_restoration(self):
        shell = shutil.which('sh')
        self.assertIsNotNone(shell)
        original = signal.signal(signal.SIGPIPE, signal.SIG_IGN)
        self.addCleanup(signal.signal, signal.SIGPIPE, original)
        reports = []
        for owning in [True, False]:
            process = await self.spawn(
                shell, '-c', 'kill -PIPE $$; printf survived', owning=owning, cwd=self.workspace,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                output, errors, code = await asyncio.gather(
                    process.stdout.read(), process.stderr.read(), process.wait())
                reports.append([code, output, errors])
            finally:
                await process.aclose()
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[0][0], -signal.SIGPIPE)


@unittest.skipUnless(sys.platform.startswith('linux'), 'native ordinary and owning POSIX creation')
class CreatorRevocationTests(unittest.IsolatedAsyncioTestCase):
    async def _assert_revoked_creation(self, *, owning, broker_loss, via_hook):
        from loki_agent import authentications, credential_capabilities, loki

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                response_headers, 'snapshot_path',
                return_value=directory + os.sep + 'response-headers.json'):
            async with contextlib.AsyncExitStack() as cleanup:
                session = loki.Session()
                cleanup.callback(session.response_headers.close)
                if broker_loss:
                    server, child = await credential_capabilities.CredentialCapabilityServer.create(
                        authentications.CredentialBroker())
                    cleanup.push_async_callback(server.close)
                    client = await credential_capabilities.CredentialClient.from_fd(child)
                    cleanup.push_async_callback(client.close)
                    session.credential_authority = client

                command = [sys.executable, '-c', 'import sys; sys.stdin.buffer.read(1)']
                with mock.patch.object(loki, '_DEFAULT_SESSION', session), mock.patch.object(
                        loki, '_tool_bootstrap_command',
                        return_value=[sys.executable, 'loki.py', '--tool-exec']), mock.patch.object(
                        loki, 'TOOL_HOOK_PIPELINE'):
                    create = loki._tool_process_creator()
                    if via_hook:
                        configuration = settings.Settings(hooks=[settings.HookSettings(
                            phase='pre_tool_gate', hook_id='revoked', command=command,
                            cwd=directory, on_error='deny')])
                        loki.configure_tool_hook_pipeline(configuration)
                        pipeline = loki.TOOL_HOOK_PIPELINE
                original = host_process.create_posix_process
                created = []
                tasks_before = asyncio.all_tasks()

                async def observed(*arguments, **options):
                    startup = options.pop('after_spawn', None)

                    async def ready(process):
                        resources = ProcessResources(process)
                        created.append(resources)
                        cleanup.push_async_callback(process.aclose)
                        cleanup.push_async_callback(resources.cleanup)
                        if startup is not None:
                            # Ordinary launches must complete their real protected
                            # second exec before this control revokes the owner.
                            await startup(process)
                        if broker_loss:
                            await server.close()
                            await client.wait_closed()
                        else:
                            session.begin_close()

                    return await original(*arguments, after_spawn=ready, **options)

                with mock.patch.object(host_process, 'create_posix_process', side_effect=observed):
                    if via_hook:
                        # Exercise the installed Settings -> runtime capability
                        # binding, not just the creator in isolation. Revocation
                        # during its real startup must become a denied gate,
                        # with the unpublished process already retired.
                        invocation = tool_runtime.ToolInvocation(
                            'revoked', 'Example', {}, {}, {}, {'type': 'object'}, cwd=session.shell_cwd)
                        prepared = await pipeline.prepare(invocation)
                        self.assertIn('This conversation has closed', prepared.denied_reason)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'This conversation has closed'):
                            await create(
                                *command, owning=owning, stdin=asyncio.subprocess.PIPE,
                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                start_new_session=True)
                self.assertFalse(session.active)
                self.assertEqual(len(created), 1)
                # Observe retirement before the fixture's backstop can hide a
                # leaked child, pipe, or creation/cleanup task.
                created[0].assert_released(self)
                self.assertFalse(asyncio.all_tasks() - tasks_before)

    async def test_session_closure_during_creation_retires_the_unpublished_child(self):
        for owning, via_hook in [[False, False], [True, False], [False, True]]:
            with self.subTest(owning=owning, via_hook=via_hook):
                await self._assert_revoked_creation(owning=owning, broker_loss=False, via_hook=via_hook)

    async def test_broker_loss_during_creation_retires_the_unpublished_child(self):
        for owning, via_hook in [[False, False], [True, False], [False, True]]:
            with self.subTest(owning=owning, via_hook=via_hook):
                await self._assert_revoked_creation(owning=owning, broker_loss=True, via_hook=via_hook)


@unittest.skipUnless(sys.platform.startswith('linux'), 'native /proc stdout-writer control')
class StartupRetirementTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_retires_child_before_releasing_an_extra_writer(self):
        ready = asyncio.Event()
        observed = []

        async def startup(process):
            observed.append(process)
            self.assertEqual(await process.stdout.readline(), b'ready\n')
            ready.set()
            await asyncio.Future()

        operation = asyncio.create_task(host_process.create_posix_process(
            sys.executable, '-c', 'import threading; print("ready", flush=True); threading.Event().wait()',
            after_spawn=startup, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True))
        writer = None
        readiness = asyncio.create_task(ready.wait())
        try:
            done, _pending = await asyncio.wait([readiness, operation], return_when=asyncio.FIRST_COMPLETED)
            if operation in done:
                self.fail(f'startup completed before the control point: {await operation!r}')
            process = observed[0]
            resources = ProcessResources(process)
            writer = os.open(f'/proc/{process.pid}/fd/1', os.O_WRONLY | os.O_CLOEXEC)
            operation.cancel()
            await asyncio.sleep(0)
            operation.cancel()
            # Join actual retirement while our extra writer is still held.
            # Neither elapsed time nor an arbitrary number of callback turns
            # proves that cleanup is independent of protocol EOF.
            with self.assertRaises(asyncio.CancelledError):
                await operation
            self.assertTrue(operation.done(), 'retirement is still waiting for the held stdout writer')
            self.assertIsNotNone(process.returncode)
            resources.assert_released(self)
        finally:
            readiness.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await readiness
            if writer is not None:
                os.close(writer)
            if not operation.done():
                operation.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await operation
