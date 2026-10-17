"""Native owning-child compatibility regression, not an activation of policy.

The shared creation primitive is exercised inside an unfiltered AppContainer.
Only a per-launch token copy gets a package-compatible creation default. The
live parent stays private, initial objects are private at creation, and the
actual child token's future-object default is checked before resume.
"""

import asyncio
from contextlib import ExitStack
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import sys
import threading
import traceback

H = C.c_void_p


def _api():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from loki_agent import windows_api
    return windows_api


def _private(sddl, owner):
    from loki_agent import windows_acl
    api = _api()
    actual = {api.canonical_sid(sid) for sid in windows_acl.allow_trustees(sddl or '')}
    return actual == {api.canonical_sid(sid) for sid in [owner, 'SY', 'BA']}


def _acl_bytes(native, sddl):
    descriptor = H()
    native.check(native.convert_sd(sddl, 1, C.byref(descriptor), None))
    try:
        present, defaulted, acl = W.BOOL(), W.BOOL(), H()
        native.check(native.get_dacl(descriptor, C.byref(present), C.byref(acl), C.byref(defaulted)))
        if not present.value or not acl.value:
            raise AssertionError('fixture requires a non-null ACL')
        return C.string_at(acl, C.c_uint16.from_address(acl.value + 2).value)
    finally:
        native.local_free(descriptor)


def _open_handles(api):
    # Enumerate live handles rather than probing already-closed ones: Windows
    # can raise STATUS_INVALID_HANDLE instead of returning a BOOL for that.
    class Entry(C.Structure):
        _fields_ = [('handle', H), ('handle_count', C.c_size_t), ('pointer_count', C.c_size_t),
                    ('access', C.c_uint32), ('kind', C.c_uint32),
                    ('attributes', C.c_uint32), ('reserved', C.c_uint32)]

    query = api.bind(
        'ntdll', 'NtQueryInformationProcess', C.c_long,
        H, C.c_int, H, W.DWORD, C.POINTER(W.DWORD))
    size = 65536
    while True:
        data, needed = C.create_string_buffer(size), W.DWORD()
        status = query(api.current_process_handle(), 51, data, len(data), C.byref(needed))
        if status >= 0:
            break
        if status & 0xffffffff not in [0xc0000004, 0xc0000023]:
            raise AssertionError('handle snapshot failed: 0x%x' % (status & 0xffffffff))
        size = max(size * 2, needed.value)
    count = C.c_size_t.from_buffer(data).value
    offset = 2 * C.sizeof(C.c_size_t)
    if offset + count * C.sizeof(Entry) > len(data):
        raise AssertionError('short handle snapshot')
    return {Entry.from_buffer(data, offset + index * C.sizeof(Entry)).handle for index in range(count)}


def launch(native, tokens, directory, actor_script, depth, *, package=None, job=None, fault=None):
    """Exercise the shared primitive and retain ownership through setup/abort."""
    import msvcrt
    api = _api()
    owner = api.current_user_sid()
    private = f'D:P(A;;GA;;;{owner})(A;;GA;;;SY)(A;;GA;;;BA)'
    current = api.current_process_handle()
    owned = []
    information = None
    result = {'resumed': False}
    parent = api.open_process_token(current, api.TokenAccess.TOKEN_QUERY)
    before = None
    try:
        before = api.token_default_dacl(parent)
        if before is None:
            raise AssertionError('parent token has a null default ACL')
        identity = tokens.snapshot(parent)
        token = None
        if package is None:
            source = api.open_process_token(current, api.TokenAccess.TOKEN_DUPLICATE)
            owned.append(source)
            token = api.duplicate_primary_token(
                source, api.TokenAccess.TOKEN_QUERY | api.TokenAccess.TOKEN_DUPLICATE |
                api.TokenAccess.TOKEN_ASSIGN_PRIMARY | api.TokenAccess.TOKEN_ADJUST_DEFAULT)
            owned.append(token)
            if tokens.snapshot(token) != identity:
                raise AssertionError('owning token copy changed authority')
            api.set_token_default_dacl(token, private + f'(A;;GA;;;{identity["package"]})')
            if api.token_default_dacl(parent) != before:
                raise AssertionError('copy preparation changed the parent default')
        directory.mkdir()
        output = directory / 'actor.json'
        empty = directory / 'stdin'
        empty.write_bytes(b'')
        with empty.open('rb') as inp, (directory / 'stdout').open('wb') as out:
            handles = [msvcrt.get_osfhandle(f.fileno()) for f in [inp, out]]
            for handle in handles:
                os.set_handle_inheritable(handle, True)
            try:
                information = api.create_process_in_app_container(
                    sys.executable, ['-I', '-u', str(actor_script), '--owning-child', str(output), str(depth)],
                    package, current_directory=str(directory), inherited_handles=handles,
                    standard_handles=[handles[0], handles[1], handles[1]],
                    primary_token=token, object_security=private + 'S:(ML;;NW;;;LW)')
                owned.extend([information.hProcess, information.hThread])
            finally:
                for handle in handles:
                    os.set_handle_inheritable(handle, False)
            if fault == 'after-create':
                raise asyncio.CancelledError()
            for handle in [information.hProcess, information.hThread]:
                if not _private(api.handle_dacl_sddl(handle, api.SE_KERNEL_OBJECT), owner):
                    raise AssertionError('initial object was not created private')
            child_token = api.open_process_token(
                information.hProcess, api.TokenAccess.TOKEN_QUERY | api.TokenAccess.TOKEN_ADJUST_DEFAULT)
            owned.append(child_token)
            child_identity = tokens.snapshot(child_token)
            expected_package = package if package is not None else identity['package']
            if (not child_identity['app'] or child_identity['package'] != expected_package or
                    (package is None and child_identity != identity)):
                raise AssertionError('owning child authority mismatch')
            api.set_token_default_dacl(child_token, private)
            if fault == 'after-default':
                raise asyncio.CancelledError()
            if api.token_default_dacl(child_token) != _acl_bytes(native, private):
                raise AssertionError('child future default readback mismatch')
            if job is not None:
                native.check(native.assign_job(job, information.hProcess))
            inside = W.BOOL()
            native.check(native.in_job(information.hProcess, job, C.byref(inside)))
            if not inside.value:
                raise AssertionError('owning child lost its lifetime job')
            if fault == 'before-resume':
                raise asyncio.CancelledError()
            api.resume_thread(information.hThread)
            result['resumed'] = True
            if native.wait(information.hProcess, 0xffffffff) != 0:
                raise AssertionError('owning child wait failed')
            code = api.get_exit_code_process(information.hProcess)
        if code != 0:
            raise AssertionError('owning child exit %d: %s' % (code, (directory / 'stdout').read_text(errors='replace')))
        result['actor'] = json.loads(output.read_text())
        if result['actor']['status'] != 'passed':
            raise AssertionError('owning child regression failed: %r' % result['actor'])
    except asyncio.CancelledError:
        if fault is None:
            raise
        result['cancelled'] = True
    finally:
        with ExitStack() as cleanup:
            cleanup.callback(api.close_handle, parent)
            # Register all closers before cleanup can fail. The child is killed
            # and reaped while its handles are still held, including pre-job
            # failures where closing the enclosing job would not yet help.
            for handle in owned:
                cleanup.callback(api.close_handle, handle)
            if information is not None:
                if native.wait(information.hProcess, 0) == 258:
                    api.terminate_process(information.hProcess)
                    result['terminated_before_close'] = True
                if native.wait(information.hProcess, 0xffffffff) != 0:
                    raise AssertionError('child cleanup wait failed')
                result['reaped_before_close'] = True
            if before is not None:
                result['parent_default_unchanged'] = api.token_default_dacl(parent) == before
                if not result['parent_default_unchanged']:
                    raise AssertionError('owning launch changed the parent default')
    # Snapshot immediately after closing, before another launch can reuse a
    # handle value. Native initialization caches are not launch-owned handles.
    remaining = _open_handles(api)
    for handle in [parent, *owned]:
        value = handle.value if isinstance(handle, H) else handle
        if value in remaining:
            raise AssertionError('launch-owned handle was not released')
    result['closed_handles'] = len(owned) + 1
    if fault is not None and (result['resumed'] or output.exists() or not result.get('terminated_before_close')):
        raise AssertionError('cancelled suspended child executed or survived')
    return result


def child(path, depth, native, tokens, actor_script):
    api = _api()
    result = {'status': 'failed'}
    token = api.open_process_token(api.current_process_handle())
    try:
        identity = tokens.snapshot(token)
        owner = identity['user']
        if not identity['app'] or identity['restricting_sids']:
            raise AssertionError('expected an unfiltered owning AppContainer')
        expected = _acl_bytes(native, f'D:P(A;;GA;;;{owner})(A;;GA;;;SY)(A;;GA;;;BA)')
        before = api.token_default_dacl(token)
        if before != expected:
            raise AssertionError('child did not enter with a private default')
        current_thread = api.bind('kernel32', 'GetCurrentThread', H)
        threads = [api.handle_dacl_sddl(current_thread(), api.SE_KERNEL_OBJECT)]

        def inspect():
            threads.append(api.handle_dacl_sddl(current_thread(), api.SE_KERNEL_OBJECT))

        worker = threading.Thread(target=inspect)
        worker.start()
        worker.join()
        callback_type = C.WINFUNCTYPE(W.DWORD, H)

        @callback_type
        def inspect_native(unused):
            inspect()
            return 0

        create_thread = api.bind('kernel32', 'CreateThread', H,
                                 H, C.c_size_t, H, H, W.DWORD, C.POINTER(W.DWORD))
        tid = W.DWORD()
        native_thread = native.check(create_thread(None, 0, C.cast(inspect_native, H), None, 0, C.byref(tid)))
        try:
            if native.wait(native_thread, 0xffffffff) != 0:
                raise AssertionError('native thread did not finish')
        finally:
            api.close_handle(native_thread)
        if len(threads) != 3 or not all(_private(sddl, owner) for sddl in threads):
            raise AssertionError('initial or future thread is not private')
        result.update(identity=identity, self_token_query=True, threads=threads)
        if depth:
            result['recursive'] = launch(native, tokens, path.parent / 'recursive', actor_script, depth - 1)
        if depth == 2:
            result['cancellations'] = []
            for stage in ['after-create', 'after-default', 'before-resume']:
                result['cancellations'].append(launch(
                    native, tokens, path.parent / stage, actor_script, 0, fault=stage))
        result['default_unchanged'] = api.token_default_dacl(token) == before
        if not result['default_unchanged']:
            raise AssertionError('recursive launch changed the owning default')
        result['status'] = 'passed'
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        api.close_handle(token)
        path.write_text(json.dumps(result))
    return 0 if result['status'] == 'passed' else 1


def probe(native, tokens, limits_type, sid, workspace, actor_script):
    job = native.check(native.create_job(None, None))
    try:
        limits = limits_type()
        limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE; no UI limit on owners.
        native.check(native.set_job(job, 9, C.byref(limits), C.sizeof(limits)))
        result = launch(native, tokens, workspace / 'owning-children', actor_script, 2,
                        package=native.sid(sid), job=job)
        print(json.dumps({'owning_child_compatibility': result}), flush=True)
        return result
    finally:
        native.check(native.close(job))
