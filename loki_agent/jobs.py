"""Tracked subprocess lifetime, captured streams, and job records.

Callers select logical cwd and delegated authority. This module owns the
spawn/exit/cancellation state machine and its resources, not session policy.
"""

import asyncio
import contextlib
import json
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass, field

from . import credential_capabilities, executables, files, host_ipc, host_process
from . import private_files
from . import process_outputs, tool_contracts, tool_runtime


def _bash_output(stdout: str, stderr: str, exit_code: int | None,
                 status: str = "completed", no_output_expected: bool = False):
    parts = [f"status: {status}"]
    if exit_code is not None:
        parts.append(f"exit_code: {exit_code}")
    if no_output_expected:
        parts.append("no_output_expected: true")
    return process_outputs.ProcessOutput("\n".join(parts), stdout, stderr)


def _shell_result(output, *, structured, ok=True, content=None):
    text = (process_outputs.truncate_text(output.render(), tool_contracts.BASH_MAX_OUTPUT_CHARS)
            if content is None else content)
    if structured:
        return tool_runtime.tool_result(ok, text, process_output=output.to_dict())
    return text if ok else tool_runtime.tool_result(False, text)


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _decode_spooled_output(data: bytes) -> str:
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('utf-8', errors='replace')


def _read_spool_tail(directory_fd, name: str,
                     max_chars: int = tool_contracts.JOB_TAIL_CHARS) -> str:
    """Read the last ``max_chars`` of one spool, relative to its directory.

    The read resolves from the retained session handle, never a pathname, and
    is bounded: it positions at most ``max_chars * 4`` bytes from the end
    (UTF-8 may use multiple bytes per character; trimmed after decoding so the
    character cap is stable).
    """
    if max_chars <= 0:
        return ''
    try:
        fd = private_files.open_read_at(directory_fd, name)
    except FileNotFoundError:
        return ""
    try:
        facts = private_files.information(fd)
        if not facts.regular or facts.reparse_point:
            raise OSError("job spool is not a regular file")
        start = max(0, facts.size - max_chars * 4)
        if start:
            private_files.seek(fd, start, os.SEEK_SET)
        text = _decode_spooled_output(private_files.read(fd, max_chars * 4 + 1))
    finally:
        private_files.close(fd)
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


@dataclass
class Job:
    id: str
    command: str
    argv: list[str] | None
    shell: bool
    description: str
    background: bool
    spool_dir: str
    stdout_path: str
    stderr_path: str
    metadata_path: str
    started_at_iso: str
    cwd: str | None = None
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    pid: int | None = None
    pgid: int | None = None
    # A leader's exit does not retire its group. This in-memory ownership ends
    # on observed absence or a successful forced stop; pgid remains for display.
    group_active: bool = field(default=False, repr=False)
    status: str = "starting"
    exit_code: int | None = None
    signal: int | None = None
    # Why the job never ran or why Loki ended it, when that is a failure.
    error: str | None = None
    finished_at: float | None = None
    finished_at_iso: str | None = None
    timeout_ms: int | None = None
    session_owned: bool = False
    # A descriptor on POSIX and a socket on Windows; see host_ipc.
    owner_signal_fd: object | None = field(default=None, repr=False)
    credential_capability: credential_capabilities.CredentialCapabilityServer | None = field(
        default=None, repr=False)
    # Live resource accounting only. This is deliberately absent from
    # job.json: after process exit it has no user-visible or recovery meaning.
    subagent_slot: bool = field(default=False, repr=False)


class SubagentCapacityError(RuntimeError):
    """The process already owns its maximum number of live subagents."""


class _JobRevokedDuringLaunch(RuntimeError):
    """The session revoked the job while its process was being created."""


class JobManager:
    """Own subprocess job state on one asyncio event loop.

    Every in-memory state transition in this class is synchronous and contains
    no ``await``. It therefore completes before another event-loop task can
    observe or modify the jobs table. Keep await points outside those
    transitions instead of adding thread locks to event-loop-owned state.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.session_id = str(uuid.uuid4())
        self.session_dir = os.path.join(base_dir, self.session_id)
        # One retained handle to the session directory, opened at construction
        # and held for the manager's lifetime: every spool and
        # metadata access resolves from this dentry, never from a pathname, so
        # a renamed state ancestor cannot redirect job storage.  Path strings
        # on Job remain for display and job.json only.
        self._session_fd = None
        self._closed = False
        self._monitors = set()
        self.jobs = {}
        self._counter = 0
        self._active_subagents = 0
        # A local synchronous observer, bound only while a frontend owns it.
        self.on_change = None
        self._session_directory()

    def close(self) -> None:
        if any(not task.done() for task in self._monitors):
            raise RuntimeError("job monitors must be joined with aclose()")
        self._closed = True
        if self._session_fd is not None:
            fd, self._session_fd = self._session_fd, None
            private_files.close(fd)

    def __del__(self):
        if getattr(self, '_session_fd', None) is not None:
            fd, self._session_fd = self._session_fd, None
            with contextlib.suppress(OSError):
                private_files.close(fd)

    async def aclose(self):
        if self._closed:
            return
        try:
            await self.close_session_owned()
        finally:
            tasks = list(self._monitors)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.close()

    def _session_directory(self):
        """The retained session-directory handle, creating the tree once."""
        if self._closed:
            raise ValueError("job storage is closed")
        if self._session_fd is None:
            base_fd = private_files.open_directory_tree(self.base_dir)
            try:
                try:
                    private_files.create_directory_at(base_fd, self.session_id)
                except OSError as error:
                    raise OSError(
                        f"could not create job session directory: {error}") from error
                self._session_fd = private_files.open_directory_at(
                    base_fd, self.session_id)
            finally:
                private_files.close(base_fd)
        return self._session_fd

    def _job_directory(self, job_id: str, *, create=False):
        """A transient handle reached only through the retained session."""
        session_fd = self._session_directory()
        if create:
            private_files.create_directory_at(session_fd, job_id)
        return private_files.open_directory_at(session_fd, job_id)

    @staticmethod
    def _open_spool_stream(directory, name):
        return private_files.binary_file(
            private_files.create_exclusive_at(directory, name, 0o600), 'wb')

    def _append_spool(self, job, name, data):
        directory = self._job_directory(job.id)
        try:
            fd = private_files.open_append_at(directory, name)
            try:
                facts = private_files.information(fd)
                if not facts.regular or facts.reparse_point:
                    raise OSError("job spool is not a regular file")
                view = memoryview(data)
                while view:
                    count = private_files.write(fd, view)
                    if count <= 0:
                        raise OSError("spool write made no progress")
                    view = view[count:]
            finally:
                private_files.close(fd)
        finally:
            private_files.close(directory)

    def _spool_tail(self, job, name, max_chars=tool_contracts.JOB_TAIL_CHARS):
        try:
            directory = self._job_directory(job.id)
        except FileNotFoundError:
            return ''
        try:
            return _read_spool_tail(directory, name, max_chars)
        finally:
            private_files.close(directory)

    def _start_monitor(self, job):
        operation = self._monitor_background_job(job)
        try:
            task = asyncio.create_task(operation)
        except BaseException:
            operation.close()
            raise
        self._monitors.add(task)
        task.add_done_callback(self._monitors.discard)

    def _reserve_subagent_slot(self, job: Job):
        # This check-and-increment contains no await and JobManager belongs to
        # one event loop, so concurrent tasks cannot overbook the limit.
        if self._active_subagents >= tool_contracts.MAX_ACTIVE_DIRECT_SUBAGENTS:
            raise SubagentCapacityError(
                "maximum active direct subagents reached "
                f"({tool_contracts.MAX_ACTIVE_DIRECT_SUBAGENTS})")
        self._active_subagents += 1
        # The flag means "this job holds a slot", not "this is a subagent job":
        # the release path keys off it, and a job is registered before its slot
        # is reserved, so a refusal must not look like something to release.
        job.subagent_slot = True

    def _release_subagent_slot(self, job: Job):
        # More than one observer can notice an exit. Ownership moves off the
        # job before decrementing so the operation is idempotent.
        if not getattr(job, "subagent_slot", False):
            return
        job.subagent_slot = False
        self._active_subagents -= 1

    def _next_job_id(self) -> str:
        self._counter += 1
        return str(self._counter)

    def _job_dir(self, job_id: str) -> str:
        return os.path.join(self.session_dir, job_id)

    def _job_metadata(self, job: Job) -> dict:
        return {
            "id": job.id,
            "command": job.command,
            "cwd": job.cwd,
            "argv": job.argv,
            "shell": job.shell,
            "description": job.description,
            "background": job.background,
            "pid": job.pid,
            "pgid": job.pgid,
            "status": job.status,
            "exit_code": job.exit_code,
            "signal": job.signal,
            "error": job.error,
            "started_at": job.started_at_iso,
            "finished_at": job.finished_at_iso,
            "timeout_ms": job.timeout_ms,
            "session_owned": job.session_owned,
            "stdout_path": job.stdout_path,
            "stderr_path": job.stderr_path,
        }

    def _write_metadata(self, job: Job):
        try:
            directory = self._job_directory(job.id)
            try:
                files.atomic_write_text_at(directory, 'job.json', json.dumps(self._job_metadata(job), indent=2) + '\n')
            finally:
                private_files.close(directory)
        finally:
            # State has already changed, even if persistence failed. Display
            # errors must neither change the job outcome nor mask an IO error.
            if self.on_change is not None:
                try:
                    self.on_change()
                except Exception:
                    logging.getLogger(__name__).exception("Job status observer failed")

    @staticmethod
    def _close_owner_signal(job: Job):
        end = job.owner_signal_fd
        job.owner_signal_fd = None
        if end is None:
            return
        try:
            host_ipc.close_end(end)
        except OSError:
            pass

    @staticmethod
    def _revoke_credential_capability(job: Job):
        capability = job.credential_capability
        if capability is not None:
            capability.close_now()

    @staticmethod
    async def _close_credential_capability(job: Job):
        """Revoke a job capability immediately, then finish async cleanup."""
        capability = job.credential_capability
        if capability is None:
            return
        capability.close_now()
        try:
            await capability.close()
        finally:
            if job.credential_capability is capability:
                job.credential_capability = None

    def _finalize_job(self, job: Job, status: str, error: str | None = None):
        """Bring a job to a recorded state, whichever path it took.

        Every exit from ``_spawn`` after the job exists comes through here, so a
        launch that never happened is still a job a caller can observe instead
        of an exception with no trace. Release the live-child reservation here
        too: a launch failure may never reach _record_exit.
        """
        job.status = status
        if error is not None:
            job.error = error
        job.finished_at = time.time()
        job.finished_at_iso = _now_iso()
        self._close_owner_signal(job)
        self._revoke_credential_capability(job)
        self._release_subagent_slot(job)
        self._write_metadata(job)

    def _record_exit(self, job: Job, exit_code: int):
        # Status records why Loki ended the job; exit_code/signal record
        # how the child process actually terminated.
        job.signal = -exit_code if exit_code < 0 else None
        job.exit_code = exit_code
        self._close_owner_signal(job)
        self._revoke_credential_capability(job)
        # The capacity limit counts direct subagent children, not descendants
        # left in their groups. Leader exit releases that reservation now.
        self._release_subagent_slot(job)
        if self._group_exists(job):
            # Keep a surviving group signalable, including after a soft stop.
            # Normal foreground completion still returns the leader's result;
            # it does not implicitly kill or wait for background descendants.
            self._write_metadata(job)
            return
        if job.status in [
                "cancelled", "timed_out", "stopped", "owner_closed",
                "failed"]:
            # Preserve the user-visible reason even after wait() reports the
            # eventual process exit code.
            status = job.status
        elif job.status == "owner_closing":
            status = "owner_closed"
        elif job.status == "stopping":
            status = "stopped"
        elif exit_code < 0:
            status = "signaled"
        else:
            status = "exited"
        self._finalize_job(job, status)

    def _refresh_job(self, job: Job):
        if job.status not in ["running", "stopping"]:
            return
        exit_code = job.process.returncode
        if exit_code is not None:
            self._record_exit(job, exit_code)

    async def _spawn(self, command, display_command: str, description: str,
                     background: bool, timeout_ms: int | None, shell: bool,
                     env: dict | None = None, cwd: str | None = None,
                     session_owned: bool = False,
                     credential_refs=None, credential_authority=None, credential_destinations=None, subagent: bool = False) -> Job:
        if session_owned and shell:
            raise ValueError(
                "session-owned jobs require a direct executable")
        if credential_refs is not None and shell:
            raise ValueError(
                "credential capabilities require a direct executable")
        if subagent and not session_owned:
            raise ValueError("subagents must be session-owned")
        job_id = self._next_job_id()
        spool_dir = self._job_dir(job_id)
        stdout_path = os.path.join(spool_dir, "stdout.log")
        stderr_path = os.path.join(spool_dir, "stderr.log")
        metadata_path = os.path.join(spool_dir, "job.json")
        job = Job(
            id=job_id,
            command=display_command,
            argv=None if shell else list(command),
            shell=shell,
            description=description,
            background=background,
            spool_dir=spool_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata_path=metadata_path,
            started_at_iso=_now_iso(),
            cwd=cwd or os.getcwd(),
            timeout_ms=timeout_ms,
            session_owned=session_owned,
        )
        # Registered before the first step that can fail.  A job that never
        # launches is then a recorded failure rather than a missing record, so
        # anything watching self.jobs sees an outcome instead of waiting for
        # one; see _finalize_job.  It may be observed in "starting" briefly,
        # before the launch resolves to "running" or "failed".
        self.jobs[job.id] = job

        owner_parent = None
        owner_child = None
        credential_child = None
        credential_server = None
        parent_handed_off = False
        capability_handed_off = False
        spawn_command = command
        proc = None
        job_dir_fd = None
        try:
            if subagent:
                # Reserved inside the failure boundary, immediately before the
                # first operation that can yield to another task: a refusal
                # here is a recorded failed job, not one stuck in "starting".
                self._reserve_subagent_slot(job)
            # Channel acquisition belongs inside the same cleanup boundary as
            # process creation. A failed relay setup must not leave either end
            # of the already-created owner channel live.
            if session_owned:
                owner_parent, owner_child = host_ipc.owner_channel()
                spawn_command = [
                    *command,
                    "--session-owner-fd",
                    str(host_ipc.reference(owner_child)),
                ]
            if credential_refs is not None:
                authority = credential_authority
                if authority is None:
                    raise RuntimeError(
                        "cannot delegate credentials without an authority")
                destinations = credential_destinations
                credential_server, credential_child = await (
                    credential_capabilities.CredentialCapabilityServer.create(
                        authority, credential_refs, destinations=destinations))
                spawn_command = [
                    *spawn_command,
                    "--credential-capability-fd",
                    str(host_ipc.reference(credential_child)),
                ]

            job_dir_fd = self._job_directory(job_id, create=True)
            with self._open_spool_stream(job_dir_fd, 'stdout.log') as stdout_file, \
                    self._open_spool_stream(job_dir_fd, 'stderr.log') as stderr_file:
                # A job gets no input, so its stdin is a pipe with no writer
                # rather than the null device: the AppContainer that confines
                # the Windows runtime denies that device, so a DEVNULL stdin
                # cannot be delivered there at all.  The parent copy is
                # released as soon as the child owns the end, on the failure
                # path too.
                stdin, release_stdin = host_ipc.eof_stdin()
                try:
                    if shell:
                        # shell rejects session_owned/credential_refs above, so
                        # spawn_command is still the original command string.
                        proc = await asyncio.create_subprocess_shell(
                            str(spawn_command),
                            executable=executables.BASH,
                            stdin=stdin,
                            stdout=stdout_file,
                            stderr=stderr_file,
                            close_fds=True,
                            env=env,
                            cwd=job.cwd,
                            **host_process.spawn_kwargs(),
                        )
                    else:
                        child_ends = [
                            end for end in [owner_child, credential_child]
                            if end is not None]
                        proc = await asyncio.create_subprocess_exec(
                            *spawn_command,
                            stdin=stdin,
                            stdout=stdout_file,
                            stderr=stderr_file,
                            close_fds=True,
                            env=env,
                            cwd=job.cwd,
                            **host_process.spawn_kwargs(),
                            **host_ipc.spawn_kwargs(child_ends),
                        )
                finally:
                    # The child duplicated or inherited the end at launch, so
                    # this copy is no longer needed either way.
                    release_stdin()

            # Release the transient directory inside the launch failure
            # boundary, before publishing a live child to its caller.
            directory, job_dir_fd = job_dir_fd, None
            private_files.close(directory)

            # The session may have revoked this job while it was launching.
            # Publishing it now would leave a live child nobody owns.
            if job.status != "starting":
                raise _JobRevokedDuringLaunch(
                    f"job was revoked while launching (status={job.status})")
            job.process = proc
            job.owner_signal_fd = owner_parent
            parent_handed_off = True
            job.credential_capability = credential_server
            capability_handed_off = True
            job.pid = proc.pid
            job.pgid = host_process.process_group(proc, proc.pid)
            job.group_active = True
            job.status = "running"
            self._write_metadata(job)
            return job
        except BaseException as error:
            if proc is not None:
                # A child exists, so end it before recording anything. An
                # unowned live process is the one outcome this must not leave.
                with contextlib.suppress(Exception):
                    host_process.signal_group(
                        proc, getattr(job, "pgid", None) or proc.pid,
                        host_process.FORCE)
                    job.group_active = False
                with contextlib.suppress(Exception):
                    await proc.wait()
            revoked = isinstance(
                error, (asyncio.CancelledError, _JobRevokedDuringLaunch))
            try:
                self._finalize_job(
                    job, "cancelled" if revoked else "failed",
                    None if revoked else f"{type(error).__name__}: {error}")
            except Exception:
                # Recording the failure must not replace the failure itself.
                pass
            if not revoked:
                try:
                    self._append_spool(
                        job, 'stderr.log',
                        f"\n[launch failed: {type(error).__name__}: "
                        f"{error}]\n".encode('utf-8'))
                except OSError:
                    pass
            raise
        finally:
            # Each independent owner must release its resource even if another
            # close fails. The session handle is retained; only this job's
            # transient lookup handle belongs to this launch.
            async with contextlib.AsyncExitStack() as cleanup:
                if credential_server is not None and not capability_handed_off:
                    cleanup.push_async_callback(credential_server.close)
                if owner_parent is not None and not parent_handed_off:
                    cleanup.callback(host_ipc.close_end, owner_parent)
                if credential_child is not None:
                    cleanup.callback(host_ipc.close_end, credential_child)
                if owner_child is not None:
                    cleanup.callback(host_ipc.close_end, owner_child)
                if job_dir_fd is not None:
                    cleanup.callback(private_files.close, job_dir_fd)

    async def _wait_for_job(self, job: Job) -> int:
        return await job.process.wait()

    async def _monitor_background_job(self, job: Job):
        try:
            exit_code = await self._wait_for_job(job)
            self._record_exit(job, exit_code)
        except Exception as e:
            # Background monitor failures cannot be returned synchronously, so
            # record them in the same stderr spool the caller already inspects.
            job.status = "monitor_error"
            job.finished_at = time.time()
            job.finished_at_iso = _now_iso()
            self._close_owner_signal(job)
            self._revoke_credential_capability(job)
            # A broken waiter does not prove that the child exited. Retain its
            # slot and fail closed instead of admitting an extra live child.
            self._append_spool(job, 'stderr.log',
                               f'\n[job monitor error: {type(e).__name__}: {e}]\n'.encode('utf-8'))
            self._write_metadata(job)
        finally:
            await self._close_credential_capability(job)

    async def run_foreground(self, command, display_command: str, timeout_ms: int | None,
                             description: str = "", shell: bool = False,
                             output_chars: int = tool_contracts.BASH_MAX_OUTPUT_CHARS,
                             env: dict | None = None, cwd: str | None = None,
                             cancel_event: asyncio.Event | None = None,
                             session_owned: bool = False,
                             credential_refs=None, credential_authority=None,
                             credential_destinations=None, subagent: bool = False):
        job = await self._spawn(command, display_command, description, False, timeout_ms, shell,
                                env=env, cwd=cwd,
                                session_owned=session_owned,
                                credential_refs=credential_refs,
                                credential_authority=credential_authority,
                                credential_destinations=credential_destinations,
                                subagent=subagent)

        exit_task = asyncio.create_task(self._wait_for_job(job))
        cancel_task = None
        try:
            # Cancellation may already have arrived while _spawn awaited.
            interrupted = cancel_event is not None and cancel_event.is_set()
            if not interrupted:
                if cancel_event is not None:
                    cancel_task = asyncio.create_task(cancel_event.wait())
                watchers = {exit_task} if cancel_task is None else {
                    exit_task, cancel_task}
                done, _pending = await asyncio.wait(
                    watchers,
                    timeout=timeout_ms / 1000 if timeout_ms is not None else None,
                    return_when=asyncio.FIRST_COMPLETED)
                # Cancellation wins a simultaneous finish. The watcher's later
                # cancellation in finally is not evidence of user intent.
                interrupted = cancel_event is not None and cancel_event.is_set()
                if not interrupted and exit_task in done:
                    self._record_exit(job, exit_task.result())
                    await self._close_credential_capability(job)
                    return job, "completed", self._spool_tail(job, 'stdout.log', output_chars), self._spool_tail(job, 'stderr.log', output_chars)

            exit_code = await self._terminate_process_group(
                job, signal.SIGINT if interrupted else signal.SIGTERM, 2.0)
            status = "cancelled" if interrupted else "timed_out"
            job.status = status
            self._record_exit(job, exit_code)
            await self._close_credential_capability(job)
            return job, status, self._spool_tail(job, 'stdout.log', output_chars), self._spool_tail(job, 'stderr.log', output_chars)
        except BaseException:
            # Cover the entire ownership scope, including cancellation during
            # pre-wait cleanup or escalation, not just the initial wait.
            exit_code = await self._terminate_process_group(
                job, signal.SIGINT, 2.0)
            job.status = "cancelled"
            self._record_exit(job, exit_code)
            await self._close_credential_capability(job)
            raise
        finally:
            tasks = [task for task in [cancel_task, exit_task]
                     if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run_shell(self, command: str, timeout: int | None = None, description: str = "",
                        run_in_background: bool = False,
                        cancel_event: asyncio.Event | None = None,
                        cwd: str | None = None, *, structured=False):
        if command is None:
            return "Error: command is required"
        if command.strip() == "":
            return _shell_result(
                _bash_output("", "", 0, no_output_expected=True), structured=structured)

        timeout_ms = int(timeout) if timeout else tool_contracts.BASH_DEFAULT_TIMEOUT_MS
        timeout_ms = min(timeout_ms, tool_contracts.BASH_MAX_TIMEOUT_MS)
        cwd = cwd or os.getcwd()

        if run_in_background:
            job = await self._spawn(command, command, description, True, None, True, cwd=cwd)
            try:
                self._start_monitor(job)
            except RuntimeError:
                # If no loop can accept the monitor task, the job still exists
                # with stdout/stderr/metadata paths for manual inspection.
                pass
            header = "\n".join([
                f"Started background job {job.id}",
                f"cwd: {job.cwd}",
                f"pid: {job.pid}",
                f"pgid: {job.pgid}",
                f"status: {job.status}",
                f"stdout: {job.stdout_path}",
                f"stderr: {job.stderr_path}",
            ])
            return _shell_result(
                process_outputs.ProcessOutput(header, "", ""),
                structured=structured, content=header)

        job, status, stdout, stderr = await self.run_foreground(
            command, command, timeout_ms, description=description, shell=True,
            cwd=cwd, cancel_event=cancel_event)
        if status == "cancelled":
            message = (
                "Tool call interrupted by user (SIGINT to the job's process group)")
            return _shell_result(
                process_outputs.ProcessOutput(message, stdout, stderr),
                structured=structured, ok=False,
                content=message + (f"; partial output:\n{stdout}" if stdout else ""))
        if status == "timed_out":
            if stderr:
                stderr += "\n"
            stderr += f"command timed out after {timeout_ms}ms"
            return _shell_result(
                _bash_output(stdout, stderr, job.exit_code, status="timed_out"),
                structured=structured, ok=False)

        return _shell_result(
            _bash_output(stdout, stderr, job.exit_code), structured=structured)

    async def run_exec(self, argv: list[str], timeout_ms: int | None = None, description: str = "",
                       output_chars: int = tool_contracts.BASH_MAX_OUTPUT_CHARS,
                       env: dict | None = None, cwd: str | None = None,
                       cancel_event: asyncio.Event | None = None,
                       session_owned: bool = False,
                       credential_refs=None, credential_authority=None,
                       credential_destinations=None, subagent: bool = False):
        if not argv:
            raise ValueError("argv must not be empty")
        return await self.run_foreground(argv, " ".join(argv), timeout_ms,
                                         description=description, shell=False,
                                         output_chars=output_chars, env=env,
                                         cwd=cwd or os.getcwd(),
                                         cancel_event=cancel_event,
                                         session_owned=session_owned,
                                         credential_refs=credential_refs,
                                         credential_authority=credential_authority,
                                         credential_destinations=credential_destinations,
                                         subagent=subagent)

    async def run_background_exec(self, argv: list[str], description: str = "",
                                  env: dict | None = None, cwd: str | None = None,
                                  session_owned: bool = False,
                                  credential_refs=None, credential_authority=None,
                                  credential_destinations=None, subagent: bool = False) -> Job:
        if not argv:
            raise ValueError("argv must not be empty")
        job = await self._spawn(argv, " ".join(argv), description, True, None, False,
                                env=env, cwd=cwd or os.getcwd(),
                                session_owned=session_owned,
                                credential_refs=credential_refs,
                                credential_authority=credential_authority,
                                credential_destinations=credential_destinations,
                                subagent=subagent)
        self._start_monitor(job)
        return job

    @staticmethod
    def _group_exists(job: Job) -> bool:
        if not job.group_active:
            return False
        if host_process.group_exists(job.process, job.pgid):
            return True
        job.group_active = False
        return False

    @staticmethod
    def _signal_process_group(job: Job, signum) -> bool:
        if job.process is None or not job.group_active:
            return False
        try:
            host_process.signal_group(job.process, job.pgid, signum)
        except ProcessLookupError:
            job.group_active = False
            return False
        if signum == host_process.FORCE:
            # Non-child zombies may keep a PGID present until their adopter
            # reaps them. Once SIGKILL is delivered, do not wait forever for
            # that or later signal a recycled historical PGID.
            job.group_active = False
        return True

    async def _wait_for_group(self, job: Job, grace_s: float) -> bool:
        deadline = asyncio.get_running_loop().time() + grace_s
        while self._group_exists(job):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.05, remaining))
        return True

    async def _terminate_process_group(self, job: Job, signum, grace_s: float,
                                       owner_grace_s: float = 0) -> int:
        async def finish():
            if not (owner_grace_s and await self._wait_for_group(job, owner_grace_s)):
                if self._signal_process_group(job, signum):
                    if not await self._wait_for_group(job, grace_s):
                        self._signal_process_group(job, host_process.FORCE)
            return await job.process.wait()

        # A cancelled caller must join its teardown before relinquishing the
        # group. Repeated cancellation must not strand the escalation task.
        task = asyncio.create_task(finish())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    pass
            task.result()
            raise

    async def _finish_owned_job(self, job: Job):
        try:
            exit_code = await self._terminate_process_group(
                job, signal.SIGTERM, 2.0, owner_grace_s=2.0)
        except asyncio.CancelledError:
            # The termination helper joined teardown before propagating this.
            if job.finished_at is None:
                self._record_exit(job, job.process.returncode)
            await self._close_credential_capability(job)
            raise
        if job.finished_at is None:
            self._record_exit(job, exit_code)
        await self._close_credential_capability(job)

    async def close_session_owned(self):
        """End and reap every live process owned by this Loki session."""
        owned = []
        cleanup = []
        for job in list(self.jobs.values()):
            self._refresh_job(job)
            if not job.session_owned:
                continue
            if job.process is None:
                if job.status == "starting":
                    # Launch still in flight: revoke it so it cannot publish a
                    # live child nobody owns. _spawn re-checks after launching.
                    job.status = "owner_closed"
                self._close_owner_signal(job)
                self._revoke_credential_capability(job)
                cleanup.append(job)
                continue
            if job.process.returncode is not None and not self._group_exists(job):
                self._close_owner_signal(job)
                self._revoke_credential_capability(job)
                cleanup.append(job)
                continue
            job.status = "owner_closing"
            self._write_metadata(job)
            # EOF asks a cooperating Loki child to cancel its current work.
            # Process-group signals below enforce the ownership boundary if
            # the child is stuck or does not understand the owner channel.
            self._close_owner_signal(job)
            self._revoke_credential_capability(job)
            owned.append(job)
        if cleanup:
            await asyncio.gather(
                *(self._close_credential_capability(job)
                  for job in cleanup))
        if owned:
            await asyncio.gather(
                *(self._finish_owned_job(job) for job in owned))

    def _get_job(self, job_id: str) -> Job | None:
        job = self.jobs.get(str(job_id))
        if job is not None:
            self._refresh_job(job)
        return job

    def list_jobs(self, *, show_all=True) -> str:
        jobs = list(self.jobs.values())
        for job in jobs:
            self._refresh_job(job)
        if not show_all:
            jobs = [job for job in jobs
                    if job.status in ["running", "starting", "failed", "signaled", "timed_out"]]
        # _now_iso() emits fixed-width UTC timestamps. _next_job_id() stores
        # decimal counter IDs as text for job keys and paths; int(id) orders
        # that counter numerically, so "10" follows "2" at equal timestamps.
        jobs.sort(key=lambda job: (job.started_at_iso, int(job.id)))
        if not jobs:
            return ("No jobs." if show_all
                    else "No running, starting, or failed jobs.")
        lines = ["Jobs:"]
        for job in jobs:
            lines.append(
                f"{job.id}. status={job.status} pid={job.pid} exit={job.exit_code} "
                f"started={job.started_at_iso} cwd={job.cwd!r} "
                f"command={job.command!r}"
            )
        return "\n".join(lines)

    def job_status(self, job_id: str, tail_chars: int = tool_contracts.JOB_TAIL_CHARS, *, structured=False):
        job = self._get_job(job_id)
        if job is None:
            return f"Error: unknown job id {job_id!r}"
        stdout = self._spool_tail(job, 'stdout.log', tail_chars)
        stderr = self._spool_tail(job, 'stderr.log', tail_chars)
        parts = [
            f"job_id: {job.id}",
            f"status: {job.status}",
            f"pid: {job.pid}",
            f"pgid: {job.pgid}",
            f"exit_code: {job.exit_code}",
            f"signal: {job.signal}",
            f"cwd: {job.cwd}",
            f"started_at: {job.started_at_iso}",
            f"finished_at: {job.finished_at_iso}",
            f"stdout_path: {job.stdout_path}",
            f"stderr_path: {job.stderr_path}",
        ]
        output = process_outputs.ProcessOutput(
            "\n".join(parts), stdout, stderr, shell=job.shell, tail=True)
        text = output.render()
        if structured and job.shell:
            return tool_runtime.tool_result(True, text, process_output=output.to_dict())
        return text

    def stop_job(self, job_id: str, force: bool = False) -> str:
        job = self._get_job(job_id)
        if job is None:
            return f"Error: unknown job id {job_id!r}"
        # This whole transition contains no await, so the event-loop reaper
        # cannot run between signal dispatch and the new state being recorded.
        if job.status not in ["running", "stopping"]:
            return f"Job {job.id} is not running (status={job.status})."
        if job.status == "stopping" and not force:
            return (
                f"Job {job.id} is already stopping. "
                "Use force=true to send SIGKILL.")
        sig = host_process.FORCE if force else signal.SIGTERM
        if not self._signal_process_group(job, sig):
            self._refresh_job(job)
            return f"Job {job.id} is no longer running."
        job.status = "stopping"
        self._write_metadata(job)
        return (f"Sent {host_process.label(sig)} to job {job.id} "
                f"(pgid={job.pgid}).")
