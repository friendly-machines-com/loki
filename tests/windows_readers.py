"""Native reader cancellation checks, staged beside the AppContainer fixture.

The child uses the real console reader and a private primary-token default.
ConPTY supplies one key without a newline: a line-mode read then remains
pending, so shutdown must cancel I/O rather than just wake an idle waiter.
"""

import asyncio
from contextlib import ExitStack
import ctypes as C
from ctypes import wintypes as W
import json
from pathlib import Path
import sys
import threading
import time

H = C.c_void_p
TOKEN_DEFAULT_DACL = 6


def default_dacl_snapshot(token):
    from loki_agent import windows_api as api
    get = api.bind('advapi32', 'GetTokenInformation', W.BOOL,
                   H, C.c_int, H, W.DWORD, C.POINTER(W.DWORD))
    size = W.DWORD()
    get(token, TOKEN_DEFAULT_DACL, None, 0, C.byref(size))
    if not size.value:
        raise C.WinError(C.get_last_error())
    buffer = C.create_string_buffer(size.value)
    if not get(token, TOKEN_DEFAULT_DACL, buffer, len(buffer), C.byref(size)):
        raise C.WinError(C.get_last_error())
    acl = H.from_buffer(buffer).value
    if not acl:
        raise RuntimeError('missing token default DACL')
    # The returned pointer is into this buffer. Retain it for restoration and
    # compare ACL bytes, not addresses allocated afresh by each query. AclSize
    # is the 16-bit field after the ACL revision and reserved bytes.
    return {'buffer': buffer, 'bytes': C.string_at(acl, C.c_uint16.from_address(acl + 2).value)}


async def _child(path):
    from loki_agent import handle_reader, windows_acl, windows_api as api
    reader = None
    query = H()
    token = api.open_process_token(api.current_process_handle())
    try:
        original = default_dacl_snapshot(token)
    except BaseException:
        api.close_handle(token)
        raise
    result = {'status': 'failed'}
    entered = threading.Event()
    duplicate_thread = handle_reader._DuplicateCurrentThread
    read_file = handle_reader._ReadFile
    cancelled = []
    cancel_io = handle_reader._CancelSynchronousIo

    def duplicate():
        handle = duplicate_thread()
        try:
            current = api.bind('kernel32', 'GetCurrentThread', H)()
            copy = api.bind('kernel32', 'DuplicateHandle', W.BOOL,
                            H, H, H, C.POINTER(H), W.DWORD, W.BOOL, W.DWORD)
            # QUERY_INFORMATION is instrumentation, not a production right.
            # Its handle lets this test verify real pending I/O instead of
            # treating an announcement before ReadFile as proof of blocking.
            process = api.current_process_handle()
            if not copy(process, current, process, C.byref(query), 0x40, False, 0):
                raise C.WinError(C.get_last_error())
            result['thread_dacl'] = api.handle_dacl_sddl(current, api.SE_KERNEL_OBJECT)
            return handle
        except BaseException:
            api.close_handle(handle)
            raise

    def read(*args):
        entered.set()
        ok = read_file(*args)
        result['read_error'] = 0 if ok else C.get_last_error()
        return ok

    def cancel(handle):
        ok = cancel_io(handle)
        cancelled.append(bool(ok))
        return ok

    handle_reader._DuplicateCurrentThread = duplicate
    handle_reader._ReadFile = read
    handle_reader._CancelSynchronousIo = cancel
    try:
        get_std = api.bind('kernel32', 'GetStdHandle', H, W.DWORD)
        incoming = get_std(W.DWORD(-10).value)
        get_mode = api.bind('kernel32', 'GetConsoleMode', W.BOOL, H, C.POINTER(W.DWORD))
        set_mode = api.bind('kernel32', 'SetConsoleMode', W.BOOL, H, W.DWORD)
        mode = W.DWORD()
        if not get_mode(incoming, C.byref(mode)) or not set_mode(incoming, mode.value | 2):
            raise C.WinError(C.get_last_error())
        reader = handle_reader.HandleReader(0, asyncio.get_running_loop(), asyncio.Queue())
        reader.start()
        path.with_suffix('.ready').write_text('send one non-newline key')
        pending_io = api.bind('kernel32', 'GetThreadIOPendingFlag', W.BOOL, H, C.POINTER(W.BOOL))
        while True:
            pending = W.BOOL()
            if not pending_io(query, C.byref(pending)):
                raise C.WinError(C.get_last_error())
            if entered.is_set() and pending.value:
                break
            await asyncio.sleep(.005)
        result['pending_read'] = True
        reader.stop()
        allowed = {api.canonical_sid(sid) for sid in [api.current_user_sid(), 'SY', 'BA']}
        actual = {api.canonical_sid(sid) for sid in windows_acl.allow_trustees(result['thread_dacl'])}
        result['default_unchanged'] = default_dacl_snapshot(token)['bytes'] == original['bytes']
        result['released'] = reader.thread is None and reader.stop_event is None and reader._cancel_thread is None
        if (result['read_error'] != 995 or not any(cancelled) or actual != allowed
                or not result['default_unchanged'] or not result['released']):
            raise AssertionError('private console reader contract failed')
        result['status'] = 'passed'
    except Exception as error:
        result['error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        with ExitStack() as cleanup:
            cleanup.callback(path.write_text, json.dumps(result))
            cleanup.callback(setattr, handle_reader, '_DuplicateCurrentThread', duplicate_thread)
            cleanup.callback(setattr, handle_reader, '_ReadFile', read_file)
            cleanup.callback(setattr, handle_reader, '_CancelSynchronousIo', cancel_io)
            cleanup.callback(api.close_handle, token)
            if query.value:
                cleanup.callback(api.close_handle, query)
            if reader is not None:
                reader.stop()
    return 0 if result['status'] == 'passed' else 1


def child(path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return asyncio.run(_child(path))


def probe(native, limits_type, sid, workspace, output, actor_script):
    sys.path.insert(0, str(Path(actor_script).resolve().parent))
    from loki_agent import host_ipc, windows_api as api
    handles = []
    information = None
    console = job = None
    captured = bytearray()
    try:
        input_read, input_write = host_ipc.private_pipe(inherit=False)
        handles.extend([input_read, input_write])
        output_read, output_write = host_ipc.private_pipe(inherit=False)
        handles.extend([output_read, output_write])
        console = api.pseudoconsole_create(80, 24, input_read, output_write)
        job = native.check(native.create_job(None, None))
        limits = limits_type()
        limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE, not a UI restriction.
        native.check(native.set_job(job, 9, C.byref(limits), C.sizeof(limits)))
        report_path = workspace / 'private-reader-child.json'
        information = api.create_process_in_app_container(
            sys.executable, ['-I', '-u', str(actor_script), '--private-reader-child', str(report_path)],
            native.sid(sid), current_directory=str(workspace),
            standard_handles=[None, None, None], pseudoconsole=console.value)
        # Null stdio suppresses inherited redirected parent handles; otherwise
        # the child's fd 0 can be a log/NUL handle instead of ConPTY's console.
        for value in [input_read, output_write]:
            api.close_handle(value)
            handles.remove(value)
        token = api.open_process_token(information.hProcess, 0x88)
        descriptor = H()
        try:
            if not api.token_is_app_container(token) or api.token_app_container_sid(token) != native.sid(sid):
                raise AssertionError('private reader child token mismatch')
            native.check(native.convert_sd(
                f'D:P(A;;GA;;;{api.current_user_sid()})(A;;GA;;;SY)(A;;GA;;;BA)',
                1, C.byref(descriptor), None))
            present, defaulted, acl = W.BOOL(), W.BOOL(), H()
            native.check(native.get_dacl(descriptor, C.byref(present), C.byref(acl), C.byref(defaulted)))
            set_default = api.bind('advapi32', 'SetTokenInformation', W.BOOL, H, C.c_int, H, W.DWORD)
            value = H(acl.value)
            if not set_default(token, TOKEN_DEFAULT_DACL, C.byref(value), C.sizeof(value)):
                raise C.WinError(C.get_last_error())
        finally:
            if descriptor.value:
                native.local_free(descriptor)
            api.close_handle(token)
        native.check(native.assign_job(job, information.hProcess))
        api.resume_thread(information.hThread)
        sent = False
        while native.wait(information.hProcess, 0) == 258:
            if report_path.with_suffix('.ready').exists() and not sent:
                native.write_all(input_write, b'x')
                sent = True
            available = W.DWORD()
            if native.peek_pipe(output_read, None, 0, None, C.byref(available), None) and available.value:
                captured.extend(api.read_file(output_read, min(available.value, 65536)))
            time.sleep(.005)
        code = C.c_uint32()
        native.check(native.exit_code(information.hProcess, C.byref(code)))
        result = json.loads(report_path.read_text())
        print(json.dumps({'private_console_reader': result}), flush=True)
        if code.value != 0 or result['status'] != 'passed' or not sent:
            raise AssertionError('private console cancellation failed: %r' % result)
        return result
    finally:
        with ExitStack() as cleanup:
            # Preserve output as data, never replay its terminal controls.
            cleanup.callback(Path(output).write_text, captured.decode(errors='replace'))
            if console is not None:
                cleanup.callback(api.pseudoconsole_close, console)
            # These callbacks run before ClosePseudoConsole: teardown cannot
            # wait for this same controller to drain its own output reader.
            for value in handles:
                cleanup.callback(api.close_handle, value)
            if information:
                cleanup.callback(api.close_handle, information.hProcess)
                cleanup.callback(api.close_handle, information.hThread)
            if job:
                api.close_handle(job)
            if information:
                # Setup can fail before job assignment. Explicitly terminate
                # that suspended child too; closing an empty job cannot kill it.
                if native.wait(information.hProcess, 0) == 258:
                    native.check(native.terminate(information.hProcess, 1))
                native.wait(information.hProcess, 0xffffffff)
