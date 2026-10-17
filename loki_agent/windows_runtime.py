# pytype: skip-file
"""Windows runtime launch and startup gate. No profile or ACL mutation.

The supervisor checks configuration, creates a suspended AppContainer process,
checks its token, assigns a kill-on-close job, then resumes it. The child runs
its own access probe before importing the frontend. These checks cover startup
preconditions, not arbitrary escape routes or future ACL changes.
"""

import asyncio
import ctypes
from contextlib import ExitStack, contextmanager
import logging
import os
import sys
from ctypes import wintypes

from . import windows_api as api
from . import windows_state as state
from . import windows_verify
from .runtime_isolations import RuntimeIsolationError

WORKSPACE_ENV = "LOKI_CONTAINER_WORKSPACE"
# The package SID the launcher verified the child's token against, for the
# child's own gate to compare against -- see verify_runtime.
PACKAGE_ENV = "LOKI_CONTAINER_PACKAGE_SID"

# The contained runtime's scratch directory, relative to the workspace.  It is
# inside the granted tree on purpose; see ensure_runtime_temp.
SCRATCH_DIRECTORY = os.path.join(".loki", "tmp")

logger = logging.getLogger(__name__)


@contextmanager
def worker_stdout_null():
    """Lend a write-only NUL handle to one explicit worker launch."""
    import msvcrt

    fd = os.open(os.devnull, os.O_WRONLY | os.O_BINARY)
    try:
        handle = msvcrt.get_osfhandle(fd)
        os.set_handle_inheritable(handle, True)
        yield handle
    finally:
        os.close(fd)


@contextmanager
def inherited_stdout_null(handle):
    """Own the inherited handle, transferring it to the CRT exactly once."""
    import msvcrt

    try:
        api.clear_handle_inheritance(handle)
        fd = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
    except BaseException:
        api.close_handle(handle)
        raise
    try:
        yield fd
    finally:
        os.close(fd)


def require_checks(checks):
    failures = [f"{check.name}: {check.detail}" for check in checks
                if check.status != "pass"]
    if not checks or failures:
        raise RuntimeIsolationError("Windows containment check failed: "
                                    + "; ".join(failures or ["no checks"]))


def configured_workspace(arguments):
    # Supervisor-side only: use the frontend's parser rather than mistaking a
    # prompt value for a workspace option. The child's gate does not call this.
    from .arguments import parse_cli_args
    workspace = os.getcwd()
    options, _ = parse_cli_args(arguments)
    for option, value in options:
        if option == "--shell-cwd":
            workspace = value
    return required_workspace(workspace)


def required_workspace(path):
    """Gate a launch on the container recorded for ``path``.

    The ACP front knows the workspace as the session cwd, not as a command
    line, so its worker launch resolves the same ledger entry, profile and
    read-only verification the terminal gate uses -- and refuses identically
    when no container is configured.

    This runs uncontained, so asking the kernel to identify the workspace is
    allowed; a path that cannot be identified is a refusal like any other, not
    a raw Windows error.
    """
    try:
        workspace = state.canonical_workspace(path)
        profile = state.profile_name_for(workspace)
        key = state.workspace_key(workspace)
    except api.WindowsApiError as error:
        raise RuntimeIsolationError(
            f"Cannot identify the workspace {path!r}: {error}") from error
    ledger = state.load_ledger()
    entry = ledger.get("workspaces", {}).get(key)
    if not isinstance(entry, dict) or not entry.get("grants"):
        raise RuntimeIsolationError(
            f"No Windows container configured for {workspace!r}; run loki-setup.")
    if entry.get("profile") != profile:
        raise RuntimeIsolationError("Windows container ledger profile mismatch")
    require_checks(windows_verify.verify_workspace(ledger, workspace))
    return workspace


def verify_runtime():
    workspace = os.environ.get(WORKSPACE_ENV)
    if not workspace:
        raise RuntimeIsolationError("Missing Windows container launch workspace")
    # The launcher already checked this process's token against this SID before
    # resuming it.  Re-deriving it here would need GetFinalPathNameByHandleW on
    # the workspace, which an AppContainer token is denied.
    package = os.environ.get(PACKAGE_ENV)
    if not package:
        raise RuntimeIsolationError("Missing Windows container package SID")
    require_checks(windows_verify.probe_containment(workspace, package))


def scratch_directory(workspace):
    """This runtime's scratch directory, inside the granted workspace.

    The workspace is the only tree the AppContainer may write, so scratch goes
    there rather than in the user's profile TEMP -- which holds no package
    grant (as do its USERPROFILE and Windows-directory fallbacks) and is shared
    with every other process running as that user.  ``launch`` points TEMP and
    TMP here; nothing else in Loki uses this path.
    """
    return os.path.join(workspace, SCRATCH_DIRECTORY)


def report_runtime_access(workspace, directory, error):
    """Report, from inside the container, what it meets where it must write.

    The workspace's DACL and label and this process's own integrity level are
    the inputs that decide whether a contained write can succeed, and none of
    them is visible from outside.  Recorded, not asserted: one line on stderr,
    which is this process's log channel.
    """
    import json

    report = {
        "runtime-access": "scratch",
        "directory": directory,
        "created": os.path.isdir(directory),
        "error": (None if error is None
                  else f"{type(error).__name__}: {error}"),
    }
    state = os.path.join(workspace, SCRATCH_DIRECTORY.split(os.sep)[0])
    for name, path in (("workspace", workspace), ("state", state)):
        entry = {"exists": os.path.exists(path)}
        for key, call in (("dacl", api.dacl_sddl), ("label", api.label_sddl)):
            try:
                entry[key] = call(path)
            except Exception as problem:  # noqa: BLE001 - recording only
                entry[key] = f"<{type(problem).__name__}: {problem}>"
        report[name] = entry
    try:
        token = api.open_process_token(api.current_process_handle())
        try:
            report["integrity"] = api.token_integrity_level(token)
        finally:
            api.close_handle(token)
    except Exception as problem:  # noqa: BLE001 - recording only
        report["integrity"] = f"<{type(problem).__name__}: {problem}>"
    print(json.dumps(report), file=sys.stderr, flush=True)


def report_write_failure(operation, path, error, traceback_text=None):
    # The error text names a path, not the call that used it.
    if traceback_text:
        print(traceback_text, file=sys.stderr, flush=True)
    """Report a refused write from inside the container, beside the paths.

    A refusal is the one moment the descriptor, this process's integrity level
    and the failing call are visible together.  The descriptor is objective,
    but whether this process can read it and what it is granted are not, so the
    report records the exception's own winerror and path, the descriptors it
    could read, its integrity level, and whether a fresh child of that parent
    can be created at all -- which separates "this parent refuses creation"
    from "something else about this call was refused".
    """
    import json

    report = {
        "runtime-access": operation,
        "path": path,
        "winerror": getattr(error, "winerror", None),
        "errno": getattr(error, "errno", None),
        "error": f"{type(error).__name__}: {error}",
    }
    parent = os.path.dirname(path or "")
    for name, candidate in (("target", path), ("parent", parent),
                            ("grandparent", os.path.dirname(parent))):
        if candidate is None:
            continue
        entry = {"path": candidate, "exists": os.path.exists(candidate)}
        for key, call in (("dacl", api.dacl_sddl), ("label", api.label_sddl)):
            try:
                entry[key] = call(candidate)
            except Exception as problem:  # noqa: BLE001 - recording only
                entry[key] = f"<{type(problem).__name__}: {problem}>"
        report[name] = entry
    report["can_create_child"] = _try_create_child(parent)
    if path:
        # The directory the runtime must create inside is the one whose label
        # differs from the workspace's, so ask about it separately.
        report["can_create_child_in_target"] = _try_create_child(path)
        report["makedirs_child"] = _attempt(lambda: os.makedirs(
            os.path.join(path, "chats"), exist_ok=True))
        report["stat"] = _attempt(lambda: os.stat(path))
        report["isdir"] = os.path.isdir(path)
        report["mkdir"] = _attempt(lambda: os.mkdir(path))
        report["makedirs_exist_ok"] = _attempt(
            lambda: os.makedirs(path, exist_ok=True))
    try:
        token = api.open_process_token(api.current_process_handle())
        try:
            report["integrity"] = api.token_integrity_level(token)
        finally:
            api.close_handle(token)
    except Exception as problem:  # noqa: BLE001 - recording only
        report["integrity"] = f"<{type(problem).__name__}: {problem}>"
    print(json.dumps(report), file=sys.stderr, flush=True)


def _attempt(operation):
    """Run ``operation`` and describe the outcome, for the failure report."""
    try:
        operation()
    except OSError as error:
        return f"{type(error).__name__}: {error}"
    return "ok"


def _try_create_child(parent):
    """Whether this process can create a fresh child of ``parent`` right now."""
    if not parent or not os.path.isdir(parent):
        return "no-parent"
    probe = os.path.join(parent, f"loki-write-probe-{os.getpid()}")
    try:
        os.mkdir(probe)
    except OSError as error:
        return f"refused: {type(error).__name__}: {error}"
    try:
        os.rmdir(probe)
    except OSError:
        pass
    return "created"


def ensure_runtime_temp():
    """Create the scratch directory if it is missing.

    The directory is writable by the very tools that use it, so it can be
    deleted between runs; startup recreates it.  Loki never removes it or its
    contents -- that is the user's business.
    """
    workspace = os.environ.get(WORKSPACE_ENV)
    if not workspace:
        raise RuntimeIsolationError("Missing Windows container launch workspace")
    directory = scratch_directory(workspace)
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as error:
        report_runtime_access(workspace, directory, error)
        raise RuntimeIsolationError(
            f"could not create the runtime scratch directory "
            f"{directory!r}: {error}") from error
    report_runtime_access(workspace, directory, None)


# Layouts used by the native fixture in tests/test_windows_appcontainers.py.
class _BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                ('flags', ctypes.c_uint32), ('minimum_ws', ctypes.c_size_t),
                ('maximum_ws', ctypes.c_size_t), ('active', ctypes.c_uint32),
                ('affinity', ctypes.c_size_t), ('priority', ctypes.c_uint32),
                ('scheduling', ctypes.c_uint32)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', _BasicLimits), ('io', ctypes.c_uint64 * 6),
                ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]


def _checked(result, operation):
    if not result:
        raise api.WindowsApiError(f"{operation}: {ctypes.get_last_error()}")
    return result


class ContainedProcess:
    """Own a native child and, for supervised runtimes, its lifetime job.

    Local launches retain the existing enclosing job rather than introducing a
    second descendant lifecycle. Their root handles release after exit; runtime
    supervisors/worker transports retain job ownership until explicit close.
    """

    def __init__(self, information, job=None):
        self.information = information
        self.job = job
        self.pid = information.dwProcessId if information is not None else None
        self.returncode = None

    async def wait(self):
        wait = api.bind("kernel32", "WaitForSingleObject", wintypes.DWORD,
                        ctypes.c_void_p, wintypes.DWORD)
        get_code = api.bind("kernel32", "GetExitCodeProcess", wintypes.BOOL,
                            ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))
        while self.returncode is None:
            result = wait(self.information.hProcess, 0)
            if result == 0:
                code = wintypes.DWORD()
                _checked(get_code(self.information.hProcess, ctypes.byref(code)),
                         "GetExitCodeProcess")
                self.returncode = code.value
            elif result == 258:  # WAIT_TIMEOUT; do not block the broker loop.
                await asyncio.sleep(0.02)
            else:
                raise api.WindowsApiError(f"WaitForSingleObject: {result}")
        if self.job is None:
            self.close()
        return self.returncode

    def terminate(self):
        if self.job is not None:
            terminate = api.bind("kernel32", "TerminateJobObject", wintypes.BOOL,
                                 ctypes.c_void_p, wintypes.UINT)
            _checked(terminate(self.job, 1), "TerminateJobObject")
        elif self.information is not None and self.information.hProcess:
            try:
                api.terminate_process(self.information.hProcess)
            except api.WindowsApiError as error:
                status = error.status or ctypes.get_last_error()
                if api.wait_for_single_object(self.information.hProcess, 0) == 0:
                    raise ProcessLookupError('child has exited') from error
                raise OSError(None, str(error), None, status) from error
        elif self.returncode is not None:
            raise ProcessLookupError('child has exited')

    kill = terminate

    def close(self):
        # A local job retains the runtime's existing enclosing job. Closing a
        # root handle must not kill descendants on normal foreground completion.
        # Supervisor-owned jobs retain their existing kill-on-close lifetime.
        if self.job is not None:
            api.close_handle(self.job)
            self.job = None
        if self.information is not None:
            for field in ['hThread', 'hProcess']:
                handle = getattr(self.information, field)
                if handle:
                    api.close_handle(handle)
                    setattr(self.information, field, None)


def duplicate_standard_handle(value, fd):
    """Lend an inheritable stdio copy without changing a borrowed CRT handle."""
    import msvcrt
    source = fd if value is None else value if isinstance(value, int) else value.fileno()
    duplicate = api.bind('kernel32', 'DuplicateHandle', wintypes.BOOL,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD,
                         wintypes.BOOL, wintypes.DWORD)
    process = api.current_process_handle()
    result = ctypes.c_void_p()
    _checked(duplicate(process, msvcrt.get_osfhandle(source), process,
                       ctypes.byref(result), 0, True, 2), 'DuplicateHandle')
    return result.value


def prepare_local(process, command, environment, current_directory, handles, standard_handles, *,
                  owning, creationflags=0):
    """Prepare an existing local child owner; never infer cwd or role from argv.

    Owning children retain the parent's future-object policy exactly. Ordinary
    children need a compatible creation default for their own libraries' pipes.
    Tools get a write-restricted token whose role is derived from the actual
    package. Owning launches never remove an inherited restriction. Neither
    path changes the live parent's default; only preopened output handles grant
    a tool writes to state objects that its token cannot reopen for mutation.
    """
    if creationflags & ~int(api.CreateProcessFlags.CREATE_NEW_PROCESS_GROUP):
        raise ValueError('unsupported local process creation flags')
    with ExitStack() as cleanup:
        source = api.open_process_token(api.current_process_handle(), api.TokenAccess.TOKEN_QUERY)
        cleanup.callback(api.close_handle, source)
        contained = api.token_is_app_container(source)
        package = api.token_app_container_sid(source) if contained else None
        restricting = api.token_restricting_sids(source)
        integrity = api.token_integrity_level(source)
        original = api.token_default_dacl(source)
        primary = None
        expected_restricting = restricting
        expected_default = original
        owner = api.current_user_sid()
        security = f'D:P(A;;GA;;;{owner})(A;;GA;;;SY)(A;;GA;;;BA)'
        if contained:
            if original is None:
                raise api.WindowsApiError('the parent token has no creation ACL')
            launch_rights = (api.TokenAccess.TOKEN_ASSIGN_PRIMARY | api.TokenAccess.TOKEN_DUPLICATE |
                             api.TokenAccess.TOKEN_QUERY | api.TokenAccess.TOKEN_ADJUST_DEFAULT)
            role = state.tool_role_sid(package)
            already_tool = role in [sid for sid, attributes in restricting]
            actual = api.open_process_token(api.current_process_handle(),
                                            api.TokenAccess.TOKEN_DUPLICATE if owning or already_tool else launch_rights)
            cleanup.callback(api.close_handle, actual)
            if owning or already_tool:
                # Duplicating an already restricted primary preserves its
                # restrictions. It is not an unfiltered retry of a failed filter.
                primary = api.duplicate_primary_token(actual, launch_rights)
            else:
                primary = api.create_write_restricted_token(
                    actual, [role, *state.TOOL_COMPATIBILITY_SIDS, api.token_logon_sid(source)])
            cleanup.callback(api.close_handle, primary)
            expected_restricting = api.token_restricting_sids(primary)
            if not owning and role not in [sid for sid, attributes in expected_restricting]:
                raise api.WindowsApiError('the tool token lacks its write restriction')
            # Preserve existing role/logon/deny entries. Only this launch token
            # gets compatibility grants for its own future objects and token.
            compatible = api.acl_sddl(original) + f'(A;;GA;;;{package})'
            if not owning:
                compatible += f'(A;;GA;;;{role})'
            api.set_token_default_dacl(primary, compatible)
            if not owning:
                expected_default = api.token_default_dacl(primary)
        process.information = api.create_process_suspended(
            command[0], command[1:], current_directory=current_directory,
            inherited_handles=handles, environment=environment, standard_handles=standard_handles,
            primary_token=primary, object_security=security, creationflags=creationflags,
            # This is ordinary external-command lookup, from the creator's
            # actual cwd. Loki entrypoints must keep their selected image.
            search_path=not owning and not os.path.dirname(command[0]))
        process.pid = process.information.dwProcessId
        rights = api.TokenAccess.TOKEN_QUERY
        if contained and owning:
            rights |= api.TokenAccess.TOKEN_ADJUST_DEFAULT
        child = api.open_process_token(process.information.hProcess, rights)
        cleanup.callback(api.close_handle, child)
        if (api.token_is_app_container(child) != contained or
                (contained and api.token_app_container_sid(child) != package) or
                api.token_integrity_level(child) != integrity or
                api.token_restricting_sids(child) != expected_restricting):
            raise api.WindowsApiError('suspended child identity mismatch')
        if contained and owning:
            api.set_token_default_dacl(child, original)
        if api.token_default_dacl(child) != expected_default or api.token_default_dacl(source) != original:
            raise api.WindowsApiError('suspended child creation ACL mismatch')


async def retire_local(process):
    """Join unpublished-child teardown; JobManager cannot own it before return."""
    if process.information is None:
        return

    async def finish():
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        await process.wait()
        process.close()

    task = asyncio.create_task(finish())
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
        task.result()
        raise


def launch(executable, arguments, environment, workspace, inherited_handles,
           stdio=None, current_directory=None):
    """Start the contained process, with piped or inherited standard handles.

    ``stdio`` is ``(stdin_handle, stdout_handle)`` for children that speak a
    protocol on their standard streams (the ACP worker): both handles must
    already be inheritable and are named in the handle list, and the caller
    owns them -- the launch never closes a handle it did not duplicate.
    Without ``stdio`` the parent's own console handles are duplicated, as
    before.  Either way ``stderr`` is a fresh inheritable duplicate of this
    process's stderr: it is the log channel, inherited rather than piped.

    ``current_directory`` is the child's initial cwd; ``None`` passes NULL
    to ``CreateProcessW`` (inherit).  ``workspace`` names the AppContainer
    the child runs in: the profile whose package SID confines it, and the
    ``LOKI_CONTAINER_WORKSPACE`` value its own startup gate verifies.
    """
    import msvcrt

    package = api.derive_app_container_sid(state.profile_name_for(workspace))
    child_environment = dict(environment)
    child_environment[WORKSPACE_ENV] = workspace
    child_environment[PACKAGE_ENV] = package
    # LOCALAPPDATA is needed in this block: with the terminal tests' reduced
    # environment the launch is refused with ERROR_ENVVAR_NOT_FOUND (203), and
    # with LOCALAPPDATA added the same launch starts.  Measured on Windows by
    # tests/test_windows_runtime.LaunchEnvironmentTests.
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data and "LOCALAPPDATA" not in child_environment:
        child_environment["LOCALAPPDATA"] = local_app_data
    # Scratch inside the granted workspace: the profile TEMP holds no package
    # grant and is shared with every other process running as the user.  The
    # contained runtime recreates the directory if it is missing.
    scratch = scratch_directory(workspace)
    child_environment["TEMP"] = scratch
    child_environment["TMP"] = scratch
    # The image is this process's own interpreter or executable, never the
    # string the caller used to start it.  A source script is passed to that
    # interpreter by the name given, and the child's cwd is the caller's, so
    # the name resolves the same way for both.
    if not getattr(sys, "frozen", False):
        arguments = [executable, *arguments]
    executable = sys.executable
    logger.debug(
        "runtime launch: executable=%r arguments=%r sys.argv[0]=%r "
        "sys.executable=%r frozen=%r",
        executable, arguments, sys.argv[0], sys.executable,
        getattr(sys, "frozen", False))

    create_job = api.bind("kernel32", "CreateJobObjectW", ctypes.c_void_p,
                          ctypes.c_void_p, wintypes.LPCWSTR)
    set_job = api.bind("kernel32", "SetInformationJobObject", wintypes.BOOL,
                       ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    assign = api.bind("kernel32", "AssignProcessToJobObject", wintypes.BOOL,
                      ctypes.c_void_p, ctypes.c_void_p)
    duplicate = api.bind("kernel32", "DuplicateHandle", wintypes.BOOL,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD,
                         wintypes.BOOL, wintypes.DWORD)
    current = api.current_process_handle()
    job = _checked(create_job(None, None), "CreateJobObjectW")
    information = None
    duplicates = []
    handed_off = False
    try:
        limits = _ExtendedLimits()
        limits.basic.flags = api.JobObjectLimits.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        _checked(set_job(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)),
                 "SetInformationJobObject")
        # Duplicate stdio rather than changing inheritance on the parent's fds.
        for fd in [0, 1, 2] if stdio is None else (2,):
            handle = ctypes.c_void_p()
            _checked(duplicate(current, msvcrt.get_osfhandle(fd), current,
                               ctypes.byref(handle), 0, True, 2), "DuplicateHandle")
            duplicates.append(handle.value)
        standard_handles = stdio + (duplicates[-1],) if stdio is not None \
            else duplicates
        information = api.create_process_in_app_container(
            executable, arguments, package,
            current_directory=current_directory,
            inherited_handles=[*inherited_handles, *duplicates, *(
                stdio if stdio is not None else ())],
            environment=child_environment, standard_handles=standard_handles)
        token = api.open_process_token(information.hProcess)
        try:
            if (not api.token_is_app_container(token)
                    or api.token_app_container_sid(token) != package):
                raise RuntimeIsolationError("Suspended runtime token mismatch")
        finally:
            api.close_handle(token)
        _checked(assign(job, information.hProcess), "AssignProcessToJobObject")
        api.resume_thread(information.hThread)
        process = ContainedProcess(information, job)
        handed_off = True
        return process
    finally:
        for handle in duplicates:
            api.close_handle(handle)
        if not handed_off:
            try:
                if information is not None:
                    api.terminate_process(information.hProcess)
                    wait = api.bind("kernel32", "WaitForSingleObject", wintypes.DWORD,
                                    ctypes.c_void_p, wintypes.DWORD)
                    result = wait(information.hProcess, 5000)
                    if result != 0:
                        raise api.WindowsApiError(f"Runtime cleanup wait: {result}")
            finally:
                if information is not None:
                    api.close_handle(information.hThread)
                    api.close_handle(information.hProcess)
                api.close_handle(job)
