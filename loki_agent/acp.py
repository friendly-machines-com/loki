"""ACP front process: transport owner and isolated-session registry.

Each live ACP session owns one worker process.  A :class:`WorkerChannel` is
the sole reader of that process's stdout and multiplexes replies by request
id.  This is the essential concurrency invariant: prompts and cancellation
may overlap, but two coroutines must never race to read the same byte stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import sys
import uuid

from . import acps, credential_supervisors, savefiles
from .credentials import CredentialStore
from .loki import CHAT_LOG_DIR

PROTOCOL_VERSION = 1
AGENT_INFO = {"name": "loki", "title": "Loki", "version": "0.1"}

SESSION_METHODS = (
    "session/prompt",
    "session/cancel",
    "session/set_config_option",
)

WORKER_COMMAND = [sys.argv[0], "--worker"]


class WorkerChannel:
    """One request multiplexer around one worker subprocess."""

    def __init__(self, session_id: str, process: asyncio.subprocess.Process,
                 forward, credential_delegation=None):
        self.session_id = session_id
        self.process = process
        self.forward = forward
        self.credential_delegation = credential_delegation
        self._pending: dict[str, asyncio.Future] = {}
        self._next_request_id = 0
        self._write_lock = asyncio.Lock()
        self._closed = False
        self._reader_task = asyncio.create_task(
            self._read_messages(),
            name=f"acp-worker-reader-{session_id}",
        )

    async def request(self, method: str, params: dict):
        if self._closed or self.process.returncode is not None:
            raise acps.TransportError(
                f"worker for {self.session_id} is not running")
        self._next_request_id += 1
        request_id = (
            f"front-{self.session_id}-{self._next_request_id}")
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            message = acps.request(request_id, method, params)
            encoded = (
                json.dumps(message, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            async with self._write_lock:
                if self._closed or self.process.stdin is None:
                    raise acps.TransportError(
                        f"worker for {self.session_id} is closed")
                self.process.stdin.write(encoded)
                await self.process.stdin.drain()
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def _read_messages(self):
        failure = None
        try:
            while True:
                raw = await self.process.stdout.readline()
                if not raw:
                    return_code = await self.process.wait()
                    failure = acps.TransportError(
                        f"worker for {self.session_id} exited "
                        f"with status {return_code}")
                    break
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    failure = acps.TransportError(
                        f"worker for {self.session_id} emitted invalid "
                        f"JSON: {error}")
                    break
                if not isinstance(message, dict):
                    failure = acps.TransportError(
                        f"worker for {self.session_id} emitted a "
                        "non-object message")
                    break

                request_id = message.get("id")
                if request_id is not None:
                    future = self._pending.get(str(request_id))
                    if future is None:
                        # Internal worker replies are never client messages.
                        # A late reply can legitimately arrive after its
                        # caller was cancelled; discard it.
                        continue
                    if "error" in message:
                        error = message.get("error") or {}
                        future.set_exception(acps.TransportError(
                            str(error.get("message") or "worker error"),
                            code=error.get("code", acps.INTERNAL_ERROR),
                        ))
                    else:
                        future.set_result(message.get("result"))
                    continue

                # Only notifications/requests belong on the outward channel.
                if message.get("method"):
                    self.forward(message)
        except asyncio.CancelledError:
            failure = acps.TransportError(
                f"worker channel for {self.session_id} closed")
            raise
        except (BrokenPipeError, ConnectionError, OSError) as error:
            failure = acps.TransportError(
                f"worker channel for {self.session_id} failed: {error}")
        finally:
            self._closed = True
            if self.credential_delegation is not None:
                self.credential_delegation.revoke_now()
                await self.credential_delegation.close()
                self.credential_delegation = None
            failure = failure or acps.TransportError(
                f"worker for {self.session_id} closed")
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(failure)

    async def close(self):
        if not self._closed:
            self._closed = True
            if self.process.stdin is not None:
                self.process.stdin.close()
                with contextlib.suppress(
                        BrokenPipeError, ConnectionError, OSError):
                    await self.process.stdin.wait_closed()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=2)
        except asyncio.TimeoutError:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if not self._reader_task.done():
            self._reader_task.cancel()
        with contextlib.suppress(
                asyncio.CancelledError, acps.TransportError):
            await self._reader_task
        if self.credential_delegation is not None:
            await self.credential_delegation.close()
            self.credential_delegation = None


class Front:
    def __init__(self, read, write, credentials: CredentialStore):
        self.read = read
        self.write = write
        self.credential_supervisor = (
            credential_supervisors.CredentialSupervisor(credentials))
        self.environment = self.credential_supervisor.environment
        self.credentials = self.credential_supervisor.inventory
        self.credential_broker = self.credential_supervisor.broker
        self.workers: dict[str, WorkerChannel] = {}
        self._next_worker_id = 0
        self._tasks: set[asyncio.Task] = set()

    async def run(self):
        try:
            async for message in self.read():
                await self.handle(message)
        finally:
            tasks = list(self._tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            channels = list(self.workers.values())
            self.workers.clear()
            if channels:
                await asyncio.gather(
                    *(channel.close() for channel in channels),
                    return_exceptions=True,
                )

    def _start_task(self, coroutine, *, name: str):
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def handle(self, message: dict):
        method = message.get("method")
        request_id = message.get("id")
        if method is None:
            return
        if request_id is None and method not in (
                "session/cancel", "session/close"):
            return
        if method == "session/prompt":
            # The transport loop remains free to route cancellation.
            self._start_task(
                self._answer(message),
                name=f"acp-client-request-{request_id}",
            )
            return
        if request_id is None:
            self._start_task(
                self._dispatch_notification(
                    method, message.get("params") or {}),
                name=f"acp-client-notification-{method}",
            )
            return
        await self._answer(message)

    async def _dispatch_notification(self, method: str, params: dict):
        try:
            await self.dispatch(method, params)
        except Exception as error:
            print(
                f"ACP notification {method!r} failed: {error!r}",
                file=sys.stderr,
            )

    async def _answer(self, message: dict):
        method = message.get("method")
        request_id = message.get("id")
        try:
            result = await self.dispatch(method, message.get("params") or {})
        except acps.TransportError as error:
            self.write(acps.response(
                request_id,
                error={"code": getattr(error, "code", acps.INTERNAL_ERROR),
                       "message": str(error)}))
            return
        except Exception as error:
            self.write(acps.response(
                request_id,
                error={"code": acps.INTERNAL_ERROR, "message": str(error)}))
            return
        self.write(acps.response(request_id, result=result))

    async def dispatch(self, method: str, params: dict):
        if method == "initialize":
            return self.initialize(params)
        if method == "session/new":
            return await self.new_session(params)
        if method == "session/load":
            return await self.load_session(params)
        if method == "session/list":
            return self.list_sessions(params)
        if method == "session/close":
            return await self.close_session(params)
        if method in SESSION_METHODS:
            return await self.forward_to_worker(method, params)
        raise acps.TransportError(
            f"method not found: {method}", code=acps.METHOD_NOT_FOUND)

    def initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentInfo": AGENT_INFO,
            "authMethods": [],
            "agentCapabilities": {
                "loadSession": True,
                "promptCapabilities": {
                    "image": False,
                    "audio": False,
                    "embeddedContext": False,
                },
                "mcpCapabilities": {
                    "http": False,
                    "sse": False,
                },
                "sessionCapabilities": {
                    "list": {},
                    "close": {},
                },
            },
        }

    @staticmethod
    def _working_directory(params: dict) -> str:
        cwd = params.get("cwd") or os.getcwd()
        if not isinstance(cwd, str) or not os.path.isabs(cwd):
            raise acps.TransportError(
                "session cwd must be an absolute path",
                code=acps.INVALID_PARAMS,
            )
        if not os.path.isdir(cwd):
            raise acps.TransportError(
                f"session cwd is not a directory: {cwd}",
                code=acps.INVALID_PARAMS,
            )
        return cwd

    @staticmethod
    def _validate_session_setup(params: dict):
        mcp_servers = params.get("mcpServers", [])
        if not isinstance(mcp_servers, list):
            raise acps.TransportError(
                "mcpServers must be an array",
                code=acps.INVALID_PARAMS,
            )
        if mcp_servers:
            raise acps.TransportError(
                "this Loki ACP adapter does not support MCP servers",
                code=acps.INVALID_PARAMS,
            )
        additional = params.get("additionalDirectories", [])
        if not isinstance(additional, list):
            raise acps.TransportError(
                "additionalDirectories must be an array",
                code=acps.INVALID_PARAMS,
            )
        if additional:
            raise acps.TransportError(
                "additionalDirectories were not advertised and are "
                "not supported",
                code=acps.INVALID_PARAMS,
            )

    async def _open_worker(self, *, cwd: str, resume=None,
                           session_id: str | None = None,
                           replay: bool = False) -> tuple[str, dict]:
        if session_id is None:
            self._next_worker_id += 1
            session_id = f"loki-{uuid.uuid4()}"
        if session_id in self.workers:
            raise acps.TransportError(
                f"session {session_id!r} is already active",
                code=acps.INVALID_PARAMS,
            )
        delegation = await self.credential_supervisor.delegate()
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *WORKER_COMMAND,
                *delegation.child_arguments(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,
                close_fds=True,
                pass_fds=delegation.child_fds(),
                env=self.environment,
                # ACP uses pipes, not a terminal. A new session prevents an
                # inherited controlling terminal from becoming an escape
                # channel through TIOCSTI or terminal-generated signals.
                start_new_session=True,
            )
        finally:
            if process is None:
                await delegation.close()
            else:
                delegation.child_spawned()
        channel = WorkerChannel(
            session_id, process, self.write, delegation)
        self.workers[session_id] = channel
        try:
            reply = await channel.request(
                "session/open",
                {
                    "sessionId": session_id,
                    "cwd": cwd,
                    "resume": resume,
                    "replay": replay,
                },
            )
        except BaseException:
            self.workers.pop(session_id, None)
            await channel.close()
            raise
        return session_id, reply or {}

    async def new_session(self, params: dict) -> dict:
        # session/new is intentionally fresh.  Resumption has its own
        # session/load operation and cannot accidentally alias an old log.
        self._validate_session_setup(params)
        session_id, worker_reply = await self._open_worker(
            cwd=self._working_directory(params))
        result = {"sessionId": session_id}
        config_options = worker_reply.get("configOptions")
        if config_options:
            result["configOptions"] = config_options
        return result

    async def load_session(self, params: dict) -> dict:
        saved_id = params.get("sessionId")
        if not isinstance(saved_id, str) or not saved_id:
            raise acps.TransportError(
                "session/load requires sessionId",
                code=acps.INVALID_PARAMS,
            )
        self._validate_session_setup(params)
        _, worker_reply = await self._open_worker(
            cwd=self._working_directory(params),
            resume=saved_id,
            session_id=saved_id,
            replay=params.get("replay", True) is not False,
        )
        result = {}
        config_options = worker_reply.get("configOptions")
        if config_options:
            result["configOptions"] = config_options
        return result

    def list_sessions(self, params: dict) -> dict:
        cwd_filter = params.get("cwd")
        if cwd_filter is not None:
            if (not isinstance(cwd_filter, str)
                    or not os.path.isabs(cwd_filter)):
                raise acps.TransportError(
                    "session/list cwd must be an absolute path",
                    code=acps.INVALID_PARAMS,
                )
            cwd_filter = os.path.realpath(cwd_filter)
        entries = []
        for path in savefiles.filtered_chat_log_paths("", CHAT_LOG_DIR):
            try:
                with open(path, "r", encoding="utf-8") as file_obj:
                    blob = json.load(file_obj)
                modified = os.path.getmtime(path)
            except (OSError, json.JSONDecodeError):
                continue
            state = blob.get("session_state") if isinstance(blob, dict) else {}
            cwd = (state or {}).get("cwd") or (state or {}).get("shell_cwd")
            if not cwd:
                continue
            if (cwd_filter is not None
                    and os.path.realpath(cwd) != cwd_filter):
                continue
            entries.append({
                "sessionId": (
                    os.path.basename(path)[len("chat-"):-len(".json")]),
                "cwd": cwd,
                "updatedAt": datetime.datetime.fromtimestamp(
                    modified, datetime.timezone.utc).isoformat(),
            })
        return {"sessions": entries}

    async def close_session(self, params: dict) -> dict:
        session_id = params.get("sessionId")
        channel = self.workers.pop(session_id, None)
        if channel is None:
            raise acps.TransportError(
                f"unknown session {session_id!r}",
                code=acps.INVALID_PARAMS,
            )
        await channel.close()
        return {}

    async def forward_to_worker(self, method: str, params: dict):
        session_id = params.get("sessionId")
        channel = self.workers.get(session_id)
        if channel is None:
            raise acps.TransportError(
                f"unknown session {session_id!r}",
                code=acps.INVALID_PARAMS,
            )
        return await channel.request(method, params)
