"""Linux ordinary-tool bootstrap and its bounded second-exec error channel.

There is no launch service: the sanctioned Loki image initializes one child,
then execs the caller's command in that same PID. The state-root loan ends
before that exec; only stdio and the caller's explicit descriptor loans continue.
This mode creates no credential authority.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
import errno
import json
import os
import signal
import sys

from . import host_ipc


_STATUS_LIMIT = 4096
_EXEC_SIGNALS = ['SIGPIPE', 'SIGXFZ', 'SIGXFSZ']


def _status_pipe():
    raw_read, raw_write = os.pipe()
    read = None
    try:
        # A pipe can occupy absent stdio slots. Move both ends above 2 before
        # spawning; a private startup channel must never become a standard stream.
        read = host_ipc.duplicate_for_child(raw_read)
        try:
            write = host_ipc.duplicate_for_child(raw_write)
        except BaseException:
            os.close(read)
            raise
        return read, write
    finally:
        os.close(raw_read)
        os.close(raw_write)


async def _read_status(fd):
    loop = asyncio.get_running_loop()
    result = loop.create_future()
    data = bytearray()
    os.set_blocking(fd, False)

    def readable():
        if result.done():
            return
        try:
            chunk = os.read(fd, _STATUS_LIMIT + 1 - len(data))
            data.extend(chunk)
            if len(data) > _STATUS_LIMIT:
                raise OSError(errno.EIO, 'command startup report is too large')
            if not chunk:
                result.set_result(bytes(data))
        except BlockingIOError:
            return
        except BaseException as error:
            result.set_exception(error)

    loop.add_reader(fd, readable)
    try:
        return await result
    finally:
        # Remove readiness ownership before the outer owner closes/reuses fd.
        loop.remove_reader(fd)


async def create_exec(tool_command, state_directory, command, **options):
    from . import host_process

    if not command:
        raise ValueError('command must not be empty')
    # These options would replace/reinterpret the first exec, which must be
    # the trusted bootstrap. Actual producers already supply an explicit argv.
    if options.get('shell') or options.get('executable') is not None:
        raise ValueError('command launches require an explicit argv image')
    cwd = options.pop('cwd', None)
    if options.pop('restore_signals', True) is not True:
        raise ValueError('command launches require default signal restoration')
    inherited = list(options.pop('pass_fds', []))
    if options.get('close_fds', True) is not True:
        raise ValueError('command launches require explicit inherited descriptors')
    options['close_fds'] = True
    with ExitStack() as cleanup:
        root = host_ipc.duplicate_for_child(state_directory)
        cleanup.callback(os.close, root)
        reader, writer = _status_pipe()
        cleanup.callback(os.close, reader)

        def close_writer():
            nonlocal writer
            selected, writer = writer, None
            if selected is not None:
                os.close(selected)

        cleanup.callback(close_writer)
        arguments = [*tool_command, str(root), str(writer), '--inherit-fds', json.dumps(inherited)]
        if cwd is not None:
            arguments.extend(['--cwd', os.fsdecode(cwd)])
        arguments.extend(['--', *command])

        async def ready(process):
            # Keeping our copy open would withhold success EOF forever. The
            # child re-arms CLOEXEC after its first exec, before the target exec.
            close_writer()
            data = await _read_status(reader)
            if not data:
                return
            try:
                report = json.loads(data)
                number = report['errno']
                message = report['message']
                stage = report['stage']
                if (not isinstance(number, int) or isinstance(number, bool)
                        or not 0 < number < 4096 or not isinstance(message, str)
                        or stage not in ['state', 'cwd', 'exec']):
                    raise ValueError('invalid command startup report')
            except (ValueError, KeyError, TypeError) as error:
                raise OSError(errno.EIO, 'invalid command startup report') from error
            filename = cwd if stage == 'cwd' else command[0] if stage == 'exec' else None
            raise OSError(number, message, filename)

        # Launcher resolution uses the owner's fixed cwd. The original raw
        # tool cwd is applied only after the new view; resolving it before the
        # cover could retain writable access underneath the protected mount.
        return await host_process.create_posix_process(
            *arguments, after_spawn=ready,
            pass_fds=[*inherited, root, writer], **options)


def main(arguments) -> int:
    root = writer = None
    stage = 'state'
    try:
        if len(arguments) < 4:
            raise ValueError('command startup descriptors are missing')
        root, writer = int(arguments[0]), int(arguments[1])
        if root < 3 or writer < 3 or root == writer:
            root = writer = None
            raise ValueError('command startup descriptors are invalid')
        # pass_fds makes the first exec possible; the second must not leak this
        # channel, and successful exec must give the waiting parent EOF.
        os.set_inheritable(writer, False)
        os.set_inheritable(root, False)
        cwd = None
        index = 2
        # Without the producer's list, import-created descriptors cannot be
        # distinguished from explicit loans at the second exec.
        if arguments[index:index + 1] != ['--inherit-fds']:
            raise ValueError('command inherited descriptors are missing')
        inherited = json.loads(arguments[index + 1])
        if not isinstance(inherited, list) or any(type(fd) is not int or fd < 0 for fd in inherited):
            raise ValueError('command inherited descriptors are invalid')
        index += 2
        if arguments[index:index + 1] == ['--cwd']:
            cwd = arguments[index + 1]
            index += 2
        if arguments[index:index + 1] != ['--'] or not arguments[index + 1:]:
            raise ValueError('command is missing')
        command = arguments[index + 1:]
        from . import runtime_isolations
        selected, root = root, None
        runtime_isolations.isolate_state_directory(selected)
        stage = 'cwd'
        if cwd is not None:
            os.chdir(cwd)
        # Python startup ignores these signals, unlike the default dispositions
        # normally restored by subprocess immediately before the actual command.
        for name in _EXEC_SIGNALS:
            if hasattr(signal, name):
                number = getattr(signal, name)
                signal.signal(number, signal.SIG_DFL)
        # First-exec close_fds cannot cover this interpreter's own imports:
        # PyPy/libffi leaves an inheritable library fd, even above stdio. Keep
        # library-owned resources usable here, but lend only the caller's
        # explicit descriptors to the target. Absent stdio can be reused too.
        streams = [sys.stdin, sys.stdout, sys.stderr]
        for name in os.listdir('/proc/self/fd'):
            fd = int(name)
            keep = streams[fd] is not None if fd < 3 else fd in inherited
            try:
                os.set_inheritable(fd, keep)
            except OSError as error:
                if error.errno != errno.EBADF:  # The enumeration's own fd is already closed.
                    raise
        stage = 'exec'
        os.execvpe(command[0], command, os.environ)
    except BaseException as error:
        if writer is not None:
            number = getattr(error, 'errno', None) or errno.EIO
            message = getattr(error, 'strerror', None) or str(error)
            # Even astral characters expand to at most twelve ASCII bytes in
            # JSON. This bound keeps the entire error below the pipe's limit.
            report = json.dumps({'errno': number, 'message': message[:256], 'stage': stage}).encode('ascii')
            try:
                os.write(writer, report)
            except OSError:
                pass
        else:
            print('Could not start command: invalid startup arguments', file=sys.stderr)
    finally:
        for fd in [root, writer]:
            if fd is not None:
                os.close(fd)
    # Do not flush a Python wrapper backed by a resource that reused absent
    # stdio. All meaningful failure information already went to the private pipe.
    os._exit(2)
