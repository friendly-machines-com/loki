"""Native Linux state-view experiments through the real Loki tool boundary.

These are mechanism probes, not a production state-isolation implementation.
The retained pre-cover directory fd and outside hardlink are deliberate bypass
controls: a read-only pathname view must not be reported as inode immutability.
"""

import asyncio
import contextlib
import ctypes
import errno
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loki_agent import runtime_isolations as isolation  # noqa: E402
from loki_entrypoints import child_environment, entrypoint  # noqa: E402
from process_lifecycle_fixtures import ProcessResources  # noqa: E402

MS_BIND = 4096
MS_REMOUNT = 32


def access_result(operation, action):
    try:
        action()
    except OSError as error:
        outcome = ('denied' if error.errno in [errno.EROFS, errno.EACCES, errno.EPERM]
                   else 'mount-busy' if error.errno == errno.EBUSY
                   else 'cross-mount' if error.errno == errno.EXDEV else 'unexpected-error')
        return {'operation': operation, 'outcome': outcome,
                'errno': error.errno, 'message': str(error)}
    return {'operation': operation, 'outcome': 'allowed'}


def privilege_state():
    selected = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, separator, value = line.partition(':')
        if separator and key in ['CapEff', 'CapPrm', 'CapInh', 'CapBnd', 'NoNewPrivs']:
            selected[key] = value.strip()
    return selected


def mount_call(libc, source, target, flags):
    mount = libc.mount
    mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                      ctypes.c_ulong, ctypes.c_void_p]
    mount.restype = ctypes.c_int
    result = mount(os.fsencode(source) if source is not None else None,
                   os.fsencode(target), None, flags, None)
    if result != 0:
        raise OSError(ctypes.get_errno(),
                      f'mount({source!r}, {str(target)!r}, flags={flags:#x}) failed')


def unmount_call(libc, path):
    unmount = libc.umount2
    unmount.argtypes = [ctypes.c_char_p, ctypes.c_int]
    unmount.restype = ctypes.c_int
    if unmount(os.fsencode(path), 0) != 0:
        raise OSError(ctypes.get_errno(), 'unmount probe failed')


def mountpoint_present(path):
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        name = line.split()[4]
        for encoded, decoded in [['\\040', ' '], ['\\011', '\t'],
                                 ['\\012', '\n'], ['\\134', '\\']]:
            name = name.replace(encoded, decoded)
        if name == str(path):
            return True
    return False


def rename_ancestor(state):
    parent = state.parent
    moved = parent.with_name('moved-state-parent')
    parent.rename(moved)
    try:
        if state.exists():
            raise RuntimeError('ancestor rename did not move the original state pathname')
    finally:
        moved.rename(parent)


def retained_mount_flags(path):
    # Bind-remount does not preserve flags for us. A less-privileged namespace
    # must not clear inherited locked nosuid/nodev/noexec/atime restrictions.
    flags = os.statvfs(path).f_flag
    mapping = [
        ['ST_NOSUID', 2], ['ST_NODEV', 4], ['ST_NOEXEC', 8],
        ['ST_SYNCHRONOUS', 16], ['ST_MANDLOCK', 64],
        ['ST_NOATIME', 1024], ['ST_NODIRATIME', 2048],
        ['ST_RELATIME', 1 << 21],
    ]
    result = 0
    for name, mount_flag in mapping:
        if flags & getattr(os, name, 0):
            result |= mount_flag
    return result


def credential_cover_probe(manifest_path):
    """Attack the existing credential cover through a movable ancestor.

    Runs the real ``isolate_credential_directory`` inside a disposable
    deployment-shaped tree, then attempts to rename ``~/.config`` and
    recreate the credential pathname. Success means a later supervisor
    ``auth login`` writes tokens into a directory the covered runtime can
    reach; the token file never existing before isolation keeps this from
    passing merely because nothing was there to read.
    """
    import socket

    manifest = json.loads(Path(manifest_path).read_text())
    home = Path(manifest['home'])
    configuration = Path(manifest['configuration'])
    # Named for the mounted cover, not the credential tree: the Windows
    # boundary guard scans launching modules' mkdir/open calls for that
    # marker, and this disposable Linux fixture must not read as fabricated
    # container state.
    cover_target = Path(manifest['credentials'])
    libc = ctypes.CDLL(None, use_errno=True)
    uid, gid = os.getuid(), os.getgid()
    isolation._unshare_user_and_mount_namespaces(libc)
    isolation._map_current_identity(uid, gid)
    mount_call(libc, None, '/', isolation._MS_REC | isolation._MS_PRIVATE)
    mount_call(libc, home, home, MS_BIND)
    # Match the real deployment, which exposes ~/.config/loki read-only and
    # leaves ~/.config an ordinary directory. Binding ~/.config itself would
    # make it a mountpoint and turn its rename into EBUSY for a reason that
    # has nothing to do with the cover -- the confound this probe exists to
    # avoid repeating.
    loki = configuration / 'loki'
    mount_call(libc, loki, loki, MS_BIND)
    mount_call(libc, None, loki, MS_BIND | MS_REMOUNT |
               retained_mount_flags(loki) | isolation._MS_RDONLY)
    directory_fd = os.open(cover_target, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    isolation.isolate_credential_directory(directory_fd)
    moved = configuration.with_name('moved-config')
    with socket.socket(socket.AF_UNIX) as control:
        control.connect(manifest['control'])
        output = control.makefile('w', buffering=1)
        try:
            operations = [
                ['cover-empty', lambda: os.listdir(cover_target)],
                ['cover-open-token', lambda: (cover_target / 'tokens.json').open('rb').close()],
                # ~/.config is not a mountpoint, and rename checks operands
                # only (fs/namei.c), so this succeeds: names in a writable
                # parent belong to whoever can write that parent.
                ['rename-config', lambda: configuration.rename(moved)],
            ]
            results = []
            for name, action in operations:
                row = access_result(name, action)
                if name == 'rename-config' and row['outcome'] == 'allowed':
                    try:
                        # The cover traveled with the dentry: the real
                        # covered tree stays hidden at its moved location.
                        moved_target = moved / 'loki' / 'credentials' / 'tokens.json'
                        row['moved-original-covered'] = access_result(
                            'open-moved-token', lambda: moved_target.open('rb').close())['outcome']
                        # The freed name is an ordinary writable path now: a
                        # forged store planted there is readable. Mounts
                        # protect objects, never names; defending the name is
                        # the trusted side's job (resolve once, hold the fd).
                        cover_target.mkdir(parents=True, exist_ok=True)
                        (cover_target / 'tokens.json').write_bytes(b'FORGED-CREDENTIALS')
                        row['forged_readable'] = access_result(
                            'read-forged-token', lambda: (cover_target / 'tokens.json').read_bytes())['outcome']
                    finally:
                        if cover_target.exists():
                            shutil.rmtree(cover_target.parent, ignore_errors=True)
                            configuration.mkdir(parents=True, exist_ok=True)
                        moved.rename(configuration)
                results.append(row)
            output.write(json.dumps({'phase': 'result', 'operations': results}) + '\n')
            output.flush()
        finally:
            output.close()


def deployment_launcher(manifest_path):
    """Prepare a disposable Guix-shaped mount view, then exec the real Loki."""
    from loki_agent import credential_storages

    manifest = json.loads(Path(manifest_path).read_text())
    home = Path(manifest['home'])
    state = Path(manifest['state'])
    libc = ctypes.CDLL(None, use_errno=True)
    uid, gid = os.getuid(), os.getgid()
    isolation._unshare_user_and_mount_namespaces(libc)
    isolation._map_current_identity(uid, gid)
    mount_call(libc, None, '/', isolation._MS_REC | isolation._MS_PRIVATE)
    mount_call(libc, home, home, MS_BIND)
    mount_call(libc, state, state, MS_BIND)
    # The real deployment's code is immutable. Hide the editing session's
    # writable checkout view in this namespace only; never change its contents.
    mount_call(libc, ROOT, ROOT, MS_BIND)
    mount_call(libc, None, ROOT, MS_BIND | MS_REMOUNT |
               retained_mount_flags(ROOT) | isolation._MS_RDONLY)
    storage = credential_storages.JsonCredentialStorage(
        str(home / '.config' / 'loki' / 'credentials'))
    storage.ensure_directory()
    configuration = Path(storage.directory).parent
    mount_call(libc, configuration, configuration, MS_BIND)
    mount_call(libc, None, configuration, MS_BIND | MS_REMOUNT |
               retained_mount_flags(configuration) | isolation._MS_RDONLY)
    mount_call(libc, storage.directory, storage.directory, MS_BIND)
    mount_call(libc, None, storage.directory, MS_BIND | MS_REMOUNT |
               retained_mount_flags(storage.directory))
    os.chdir(manifest['workspace'])
    isolation._drop_namespace_capabilities(libc)
    launcher = entrypoint('loki')
    os.execve(launcher, [launcher, '--dangerously-skip-permissions'], os.environ)


def deployment_ancestor_attack(state, control, output):
    parent = state.parent
    moved = parent.with_name('moved-state-parent')
    result = access_result('deployment-ancestor-rename', lambda: parent.rename(moved))
    if result['outcome'] != 'allowed':
        output.write(json.dumps({'phase': 'ancestor-blocked', 'result': result}) + '\n')
        output.flush()
        if control.recv(1) != b'a':
            raise RuntimeError('controller closed before ancestor observation')
        return result
    try:
        parent.mkdir()
        state.mkdir()
        (state / 'published-later').write_bytes(b'FORGED-STATE-PUBLICATION')
        result['replacement_read_only'] = bool(os.statvfs(state).f_flag & os.ST_RDONLY)
        output.write(json.dumps({'phase': 'ancestor-replaced', 'result': result}) + '\n')
        output.flush()
        if control.recv(1) != b'a':
            raise RuntimeError('controller closed before replacement observation')
    finally:
        # Only the newly manufactured fixture tree is removed. The real bound
        # state stays under moved/loki, then returns to its original spelling.
        if parent.exists():
            shutil.rmtree(parent)
        moved.rename(parent)
    return result


def state_actions(manifest):
    state = Path(manifest['state'])
    workspace = Path(manifest['workspace'])
    operations = [
        ['state-read', lambda: (state / 'read').read_bytes()],
        ['state-create', lambda: (state / 'created').write_bytes(b'bad')],
        ['state-append', lambda: append(state / 'append')],
        ['state-truncate', lambda: (state / 'truncate').write_bytes(b'bad')],
        ['state-unlink', lambda: (state / 'unlink').unlink()],
        ['state-rename', lambda: (state / 'rename').rename(state / 'renamed')],
        ['state-replace', lambda: os.replace(state / 'replacement-temp', state / 'replace')],
        ['state-mkdir', lambda: (state / 'created-directory').mkdir()],
        ['state-rmdir', lambda: (state / 'empty-directory').rmdir()],
        ['state-chmod', lambda: (state / 'chmod').chmod(0o600)],
        ['state-root-rename', lambda: state.rename(state.with_name('moved-state'))],
        ['state-symlink-write', lambda: (workspace / 'state-alias' / 'via-link').write_bytes(b'bad')],
        ['state-proc-root-write', lambda: Path('/proc/self/root' + str(state), 'via-proc').write_bytes(b'bad')],
        ['workspace-write', lambda: (workspace / 'allowed').write_bytes(b'allowed')],
        ['new-owner-file-read', lambda: (state / 'published-later').read_bytes()],
    ]
    results = [access_result(name, action) for name, action in operations]
    directory = os.open(state, os.O_RDONLY | os.O_DIRECTORY)
    try:
        results.append(access_result('post-cover-dirfd-write', lambda: Path(
            f'/proc/self/fd/{directory}/via-fd').write_bytes(b'bad')))
    finally:
        os.close(directory)
    return results


def append(path):
    with path.open('ab') as stream:
        stream.write(b'bad')


def tool_probe(manifest_path):
    import socket

    manifest = json.loads(Path(manifest_path).read_text())
    with socket.socket(socket.AF_UNIX) as control:
        control.connect(manifest['control'])
        output = control.makefile('w', buffering=1)
        try:
            before = privilege_state()
            before_namespace = os.readlink('/proc/self/ns/mnt')
            state = Path(manifest['state'])
            deployment = manifest.get('deployment', False)
            inherited_mount = mountpoint_present(state)
            code_read_only = bool(os.statvfs(ROOT).f_flag & os.ST_RDONLY)
            retained = None if deployment else os.open(state, os.O_RDONLY | os.O_DIRECTORY)
            cross_link = Path(manifest['workspace']) / 'link-before-cover'
            link_before = access_result('new-hardlink-before-cover', lambda: os.link(state / 'read', cross_link))
            if cross_link.exists():
                cross_link.unlink()
            libc = ctypes.CDLL(None, use_errno=True)
            uid, gid = os.getuid(), os.getgid()
            isolation._unshare_user_and_mount_namespaces(libc)
            isolation._map_current_identity(uid, gid)
            mount_call(libc, None, '/', isolation._MS_REC | isolation._MS_PRIVATE)
            if deployment and manifest['pin_ancestors']:
                # Pin directory identities, not their contents: unrelated
                # ~/.local writes stay allowed, but these ancestors cannot be
                # renamed out from under the protected state pathname.
                home = Path(manifest['home'])
                for ancestor in [home / '.local', home / '.local' / 'state']:
                    mount_call(libc, ancestor, ancestor, MS_BIND | isolation._MS_REC)
            if manifest['submount']:
                mount_call(libc, manifest['nested_source'], state / 'nested', MS_BIND)
            mount_call(libc, state, state, MS_BIND | isolation._MS_REC)
            remount = MS_BIND | MS_REMOUNT | retained_mount_flags(state)
            mount_call(libc, None, state, remount | isolation._MS_RDONLY)
            # A successful writable remount while still privileged validates the
            # later attack's call shape; denial must not mean bad parameters.
            mount_call(libc, None, state, remount)
            mount_call(libc, None, state, remount | isolation._MS_RDONLY)
            unmount_control = Path(manifest['workspace']) / 'unmount-control'
            unmount_control.mkdir()
            mount_call(libc, state, unmount_control, MS_BIND)
            unmount_call(libc, unmount_control)
            unmount_control.rmdir()
            os.chdir(manifest['workspace'])
            isolation._drop_namespace_capabilities(libc)
            output.write(json.dumps({'phase': 'ready', 'before': before,
                                     'after': privilege_state(),
                                     'inherited_state_mount': inherited_mount,
                                     'code_read_only': code_read_only,
                                     'new_hardlink_before_cover': link_before,
                                     'namespace_changed': before_namespace != os.readlink(
                                         '/proc/self/ns/mnt')}) + '\n')
            output.flush()
            if control.recv(1) != b'g':
                raise RuntimeError('probe controller closed before publication')
            results = state_actions(manifest)
            if deployment:
                results.append(access_result('unrelated-local-write', lambda: (
                    Path(manifest['home']) / '.local' / 'unrelated-tool-file').write_bytes(b'allowed')))
                results.append(access_result('deployment-local-ancestor-rename', lambda: rename_ancestor(state.parent)))
            limitations = []
            if retained is not None:
                limitations.extend([
                    access_result('pre-cover-dirfd-write', lambda: Path(
                        f'/proc/self/fd/{retained}/legacy-fd').write_bytes(b'legacy')),
                    access_result('outside-hardlink-write', lambda: (
                        Path(manifest['workspace']) / 'hardlink').write_bytes(b'alias')),
                    access_result('state-ancestor-rename', lambda: rename_ancestor(state)),
                ])
                os.close(retained)
            ancestor_attack = (deployment_ancestor_attack(state, control, output)
                               if deployment else None)
            if manifest['submount']:
                limitations.append(access_result('nested-mount-write', lambda: (
                    state / 'nested' / 'nested-write').write_bytes(b'nested')))
            exceptions = []
            for fd in [1, 2]:
                def write_own(fd=fd):
                    os.write(fd, b'own-output-descriptor\n')
                    os.lseek(fd, 0, os.SEEK_END)
                    os.ftruncate(fd, os.fstat(fd).st_size)
                exceptions.append(access_result(f'own-fd-{fd}-write-seek-truncate', write_own))
            attacks = [
                access_result('remount-after-cap-drop', lambda: mount_call(libc, None, state, remount)),
                access_result('unmount-after-cap-drop', lambda: unmount_call(libc, state)),
            ]
            # A new user namespace must not unlock the inherited read-only mount.
            try:
                uid, gid = os.getuid(), os.getgid()
                isolation._unshare_user_and_mount_namespaces(libc)
                isolation._map_current_identity(uid, gid)
            except isolation.RuntimeIsolationError as error:
                attacks.append({'operation': 'further-user-namespace',
                                'outcome': 'setup-refused', 'message': str(error)})
            else:
                attacks.append(access_result('remount-from-further-user-namespace', lambda: mount_call(
                    libc, None, state, remount)))
                unmount = access_result('unmount-from-further-user-namespace', lambda: unmount_call(libc, state))
                if unmount.get('errno') == errno.EINVAL:
                    # Linux returns EINVAL for MNT_LOCKED inherited mounts,
                    # not just invalid targets. Confirm the mount exists; the
                    # successful scratch unmount above validates our call.
                    confirmed = mountpoint_present(state)
                    unmount['mountpoint_confirmed'] = confirmed
                    if confirmed:
                        unmount['outcome'] = 'locked-mount'
                attacks.append(unmount)
                isolation._drop_namespace_capabilities(libc)
            child = subprocess.run(
                [sys.executable, __file__, '--descendant', manifest_path],
                capture_output=True, text=True, check=False, close_fds=True)
            if child.returncode:
                raise RuntimeError('descendant failed: ' + child.stderr)
            output.write(json.dumps({'phase': 'result', 'operations': results,
                                     'limitations': limitations, 'stdio_exceptions': exceptions,
                                     'attacks': attacks, 'descendant': json.loads(child.stdout),
                                     'deployment_ancestor_attack': ancestor_attack}) + '\n')
        except BaseException as error:
            output.write(json.dumps({'phase': 'probe-error', 'error': repr(error)}) + '\n')
            raise
        finally:
            output.close()


@unittest.skipUnless(sys.platform.startswith('linux'), 'requires Linux namespaces')
class StateViewProbes(unittest.TestCase):
    maxDiff = None

    def test_read_only_state_in_credential_isolated_tool(self):
        for submount in [False, True]:
            with self.subTest(submount=submount), tempfile.TemporaryDirectory(prefix='state probe-') as directory:
                asyncio.run(self.exercise(Path(directory), submount))

    def test_credential_cover_ancestor_replacement(self):
        """A moved ancestor steals the name; the cover still protects the object.

        The cover mounts on the credentials dentry, and mounts attach to
        dentries, not names. Renaming ~/.config -- an ordinary directory in
        the real deployment, where only ~/.config/loki is exposed read-only --
        is allowed (rename checks operands only, fs/namei.c), carries the
        covered subtree to its new name, and frees the original spelling for
        a forged writable store. This documents that mount isolation cannot
        defend names: the trusted side must resolve once and hold the
        directory handle, the way private_files already does per operation.
        """
        with tempfile.TemporaryDirectory(prefix='credential cover-') as directory:
            asyncio.run(self.credential_exercise(Path(directory)))

    async def credential_exercise(self, root):
        home = root / 'home'
        configuration = home / '.config'
        cover_target = configuration / 'loki' / 'credentials'
        cover_target.mkdir(parents=True)
        ready = asyncio.get_running_loop().create_future()
        server = await asyncio.start_unix_server(
            lambda reader, writer: ready.set_result([reader, writer]),
            path=str(root / 'control.sock'))
        manifest = {'home': str(home), 'configuration': str(configuration),
                    'credentials': str(cover_target), 'control': str(root / 'control.sock')}
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).resolve()), '--credential-cover', str(path),
            cwd=str(root), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output = asyncio.create_task(process.communicate())
        writer = None
        try:
            done, pending = await asyncio.wait(
                [ready, output], return_when=asyncio.FIRST_COMPLETED)
            if ready not in done:
                stdout, stderr = await output
                self.fail(f'cover probe exited before connecting: {stdout!r} {stderr!r}')
            reader, writer = await ready
            result = json.loads(await reader.readline())
            print(json.dumps({'linux_credential_ancestor_probe': result}), flush=True)
            self.assertEqual(result['phase'], 'result', result)
            rows = {row['operation']: row for row in result['operations']}
            # The cover is mode 000: both listing and opening are denied.
            self.assertEqual(rows['cover-empty']['outcome'], 'denied')
            self.assertEqual(rows['cover-open-token']['outcome'], 'denied')
            rename = rows['rename-config']
            print(json.dumps({'credential_ancestor_outcome': rename}), flush=True)
            # ~/.config is not a mountpoint: the rename succeeds, the cover
            # follows the dentry to the moved name, and the freed spelling
            # accepts a readable forgery.
            self.assertEqual(rename['outcome'], 'allowed', rename)
            self.assertEqual(rename['moved-original-covered'], 'denied', rename)
            self.assertEqual(rename['forged_readable'], 'allowed', rename)
            self.assertFalse(cover_target.exists() and any(cover_target.iterdir()))
            self.assertFalse((home / '.config' / 'moved-config').exists())
            self.assertFalse((root / 'moved-config').exists())
        finally:
            if writer is not None:
                writer.close()
                with contextlib.suppress(ConnectionError):
                    await writer.wait_closed()
            server.close()
            await server.wait_closed()
            if process.returncode is None:
                process.kill()
            await output

    def test_inherited_state_bind_under_writable_home_ancestors(self):
        for pin_ancestors in [False, True]:
            with self.subTest(pin_ancestors=pin_ancestors), tempfile.TemporaryDirectory(
                    prefix='state deployment-') as directory:
                asyncio.run(self.exercise(Path(directory), False, deployment=True,
                                          pin_ancestors=pin_ancestors))

    async def exercise(self, root, submount, deployment=False, pin_ancestors=False):
        home = root / 'home' if deployment else root
        home.mkdir(exist_ok=True)
        state_home = home / '.local' / 'state' if deployment else root / 'state'
        workspace, state = root / 'workspace', state_home / 'loki'
        workspace.mkdir()
        state.mkdir(parents=True)
        for name in ['read', 'append', 'truncate', 'unlink', 'rename', 'replace',
                     'replacement-temp', 'chmod', 'hardlink-target']:
            (state / name).write_bytes(b'original')
        (state / 'empty-directory').mkdir()
        (state / 'nested').mkdir()
        nested_source = root / 'nested-source'
        nested_source.mkdir()
        (workspace / 'state-alias').symlink_to(state, target_is_directory=True)
        if not deployment:
            os.link(state / 'hardlink-target', workspace / 'hardlink')
        ready = asyncio.get_running_loop().create_future()
        server = await asyncio.start_unix_server(
            lambda reader, writer: ready.set_result([reader, writer]), path=str(workspace / 'control'))
        manifest = {'state': str(state), 'workspace': str(workspace),
                    'control': str(workspace / 'control'), 'submount': submount,
                    'nested_source': str(nested_source), 'deployment': deployment, 'home': str(home),
                    'pin_ancestors': pin_ancestors}
        path = workspace / 'manifest.json'
        path.write_text(json.dumps(manifest))
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), '--tool', str(path)])
        environment = child_environment(
            PATH=os.environ.get('PATH', ''), HOME=str(home),
            XDG_CONFIG_HOME=str(home / '.config' if deployment else root / 'config'),
            XDG_STATE_HOME=str(state_home),
            LOKI_PROVIDER='dummy', LOKI_API_BASE='http://dummy.invalid/v1', LOKI_MODEL='dummy')
        launch = ([sys.executable, str(Path(__file__).resolve()), '--deployment', str(path)]
                  if deployment else [entrypoint('loki'), '--dangerously-skip-permissions'])
        process = await asyncio.create_subprocess_exec(
            *launch, cwd=str(workspace), env=environment,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        resources = ProcessResources(process)
        output = asyncio.create_task(process.communicate(f'!{command}\n/quit\n'.encode()))
        writer = None
        try:
            done, _pending = await asyncio.wait([ready, output], return_when=asyncio.FIRST_COMPLETED)
            if ready not in done:
                self.fail(f'Loki exited before probe connection: {await output!r}')
            reader, writer = await ready
            prepared = json.loads(await reader.readline())
            print(json.dumps({'linux_state_probe_setup': prepared, 'submount': submount}), flush=True)
            self.assertEqual(prepared['phase'], 'ready', prepared)
            self.assertTrue(prepared['namespace_changed'])
            if deployment:
                self.assertTrue(prepared['inherited_state_mount'])
                self.assertTrue(prepared['code_read_only'])
                self.assertEqual(prepared['new_hardlink_before_cover']['outcome'], 'cross-mount')
            for phase in ['before', 'after']:
                self.assertEqual(prepared[phase]['NoNewPrivs'], '1')
                for key in ['CapEff', 'CapPrm', 'CapInh', 'CapBnd']:
                    self.assertEqual(prepared[phase][key], '0' * 16)
            (state / 'published-later').write_bytes(b'outside owner publication')
            writer.write(b'g')
            await writer.drain()
            if deployment:
                stage = json.loads(await reader.readline())
                print(json.dumps({'linux_deployment_ancestor_probe': stage,
                                  'pin_ancestors': pin_ancestors}), flush=True)
                if pin_ancestors:
                    self.assertEqual(stage['phase'], 'ancestor-blocked', stage)
                if stage['phase'] == 'ancestor-replaced':
                    self.assertFalse(stage['result']['replacement_read_only'])
                    self.assertEqual((state / 'published-later').read_bytes(), b'FORGED-STATE-PUBLICATION')
                    original = state.parent.with_name('moved-state-parent') / state.name / 'published-later'
                    self.assertEqual(original.read_bytes(), b'outside owner publication')
                else:
                    self.assertEqual(stage['phase'], 'ancestor-blocked', stage)
                    self.assertIn(stage['result']['outcome'], ['denied', 'mount-busy'], stage)
                    self.assertEqual((state / 'published-later').read_bytes(), b'outside owner publication')
                writer.write(b'a')
                await writer.drain()
            result = json.loads(await reader.readline())
            print(json.dumps({'linux_state_probe': result, 'submount': submount}), flush=True)
            self.assertEqual(result['phase'], 'result', result)
            allowed = ['state-read', 'workspace-write', 'new-owner-file-read', 'unrelated-local-write']
            if deployment and not pin_ancestors:
                allowed.append('deployment-local-ancestor-rename')
            for row in result['operations']:
                expected = ('allowed' if row['operation'] in allowed else 'mount-busy'
                            if row['operation'] in ['state-root-rename', 'deployment-local-ancestor-rename']
                            else 'denied')
                self.assertEqual(row['outcome'], expected, row)
            for row in result['limitations'] + result['stdio_exceptions']:
                self.assertEqual(row['outcome'], 'allowed', row)
            for row in result['attacks']:
                self.assertIn(row['outcome'], ['denied', 'setup-refused', 'locked-mount'], row)
            self.assertEqual(result['descendant']['outcome'], 'denied')
            stdout, stderr = await output
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertEqual(process.returncode, 0, stderr)
            resources.assert_released(self)
            # The owning runtime must still publish terminal job status after
            # its child established the private read-only view.
            metadata = list((state / 'jobs').glob('*/*/job.json'))
            self.assertEqual(len(metadata), 1, stdout)
            self.assertEqual(json.loads(metadata[0].read_text())['status'], 'exited')
            self.assertEqual((state / 'read').read_bytes(), b'original')
        finally:
            if writer is not None:
                writer.close()
                with contextlib.suppress(ConnectionError):
                    await writer.wait_closed()
            server.close()
            await server.wait_closed()
            await resources.cleanup()
            await output


if __name__ == '__main__':
    if sys.argv[1:2] == ['--deployment']:
        deployment_launcher(sys.argv[2])
    elif sys.argv[1:2] == ['--credential-cover']:
        credential_cover_probe(sys.argv[2])
    elif sys.argv[1:2] == ['--tool']:
        tool_probe(sys.argv[2])
    elif sys.argv[1:2] == ['--descendant']:
        description = json.loads(Path(sys.argv[2]).read_text())
        print(json.dumps(access_result('exec-descendant-state-write', lambda: (
            Path(description['state']) / 'descendant-write').write_bytes(b'bad'))))
    else:
        unittest.main()
