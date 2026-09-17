"""Windows runtime launch and startup gate. No profile or ACL mutation.

The supervisor checks configuration, creates a suspended AppContainer process,
checks its token, assigns a kill-on-close job, then resumes it. The child runs
its own access probe before importing the frontend. These checks cover startup
preconditions, not arbitrary escape routes or future ACL changes.
"""

import asyncio
import ctypes
import logging
import os
import sys
from ctypes import wintypes

from . import windows_api as api
from . import windows_state as state
from . import windows_verify
from .runtime_isolations import RuntimeIsolationError

WORKSPACE_ENV = "LOKI_CONTAINER_WORKSPACE"

# The contained runtime's scratch directory, relative to the workspace.  It is
# inside the granted tree on purpose; see ensure_runtime_temp.
SCRATCH_DIRECTORY = os.path.join(".loki", "tmp")

logger = logging.getLogger(__name__)


def require_checks(checks):
    failures = [f"{check.name}: {check.detail}" for check in checks
                if check.status != "pass"]
    if not checks or failures:
        raise RuntimeIsolationError("Windows containment check failed: "
                                    + "; ".join(failures or ["no checks"]))


def configured_workspace(arguments):
    # Supervisor-side only: use the frontend's parser rather than mistaking a
    # prompt value for a workspace option. The child's gate does not call this.
    from .terminal_frontend import parse_cli_args
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
    """
    workspace = state.canonical_workspace(path)
    ledger = state.load_ledger()
    entry = ledger.get("workspaces", {}).get(state.workspace_key(workspace))
    if not isinstance(entry, dict) or not entry.get("grants"):
        raise RuntimeIsolationError(
            f"No Windows container configured for {workspace!r}; run loki-setup.")
    if entry.get("profile") != state.profile_name_for(workspace):
        raise RuntimeIsolationError("Windows container ledger profile mismatch")
    require_checks(windows_verify.verify_workspace(ledger, workspace))
    return workspace


def verify_runtime():
    workspace = os.environ.get(WORKSPACE_ENV)
    if not workspace:
        raise RuntimeIsolationError("Missing Windows container launch workspace")
    require_checks(windows_verify.probe_containment(workspace))


def scratch_directory(workspace):
    """This runtime's scratch directory, inside the granted workspace.

    The workspace is the only tree the AppContainer may write, so scratch goes
    there rather than in the user's profile TEMP -- which holds no package
    grant (as do its USERPROFILE and Windows-directory fallbacks) and is shared
    with every other process running as that user.  ``launch`` points TEMP and
    TMP here; nothing else in Loki uses this path.
    """
    return os.path.join(workspace, SCRATCH_DIRECTORY)


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
        raise RuntimeIsolationError(
            f"could not create the runtime scratch directory "
            f"{directory!r}: {error}") from error


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
    """Own the native process and job handles until the supervisor closes them."""

    def __init__(self, information, job):
        self.information = information
        self.job = job
        self.pid = information.dwProcessId
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
        return self.returncode

    def terminate(self):
        terminate = api.bind("kernel32", "TerminateJobObject", wintypes.BOOL,
                             ctypes.c_void_p, wintypes.UINT)
        _checked(terminate(self.job, 1), "TerminateJobObject")

    kill = terminate

    def close(self):
        # Closing the job also kills descendants left after the root exits.
        api.close_handle(self.job)
        api.close_handle(self.information.hThread)
        api.close_handle(self.information.hProcess)


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
    # Scratch inside the granted workspace: the profile TEMP holds no package
    # grant and is shared with every other process running as the user.  The
    # contained runtime recreates the directory if it is missing.
    scratch = scratch_directory(workspace)
    child_environment["TEMP"] = scratch
    child_environment["TMP"] = scratch
    # The image is this process's own interpreter or executable, never the
    # string the caller used to start it: in a frozen build sys.argv[0] can be
    # a relative name or a symlink, and the child's cwd is whatever the
    # caller stated, which need not be where that string would resolve.  A
    # source script is passed to the interpreter as an absolute path.
    if not getattr(sys, "frozen", False):
        arguments = [os.path.abspath(executable), *arguments]
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
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        _checked(set_job(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)),
                 "SetInformationJobObject")
        # Duplicate stdio rather than changing inheritance on the parent's fds.
        for fd in (0, 1, 2) if stdio is None else (2,):
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
