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

from . import __version__, acps, credential_supervisors, savefiles
from .connections import (
    ConnectionDescriptor,
    ConnectionDescriptorError,
    connection_display_fields,
)
from .credentials import CredentialStore
from .loki import CHAT_LOG_DIR

PROTOCOL_VERSION = 1
AGENT_INFO = {
    "name": "loki",
    "title": "Loki",
    "version": __version__,
}

SESSION_METHODS = (
    "session/prompt",
    "session/cancel",
    "session/set_config_option",
)

RESTORE_METHODS = (
    "session/load",
    "session/resume",
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

    async def request(
            self, method: str, params: dict,
            forwarded: asyncio.Event | None = None):
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
            if forwarded is not None:
                forwarded.set()
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
    def __init__(
            self, read, write, credentials: CredentialStore,
            credential_storage=None):
        self.read = read
        self.write = write
        self.credential_supervisor = (
            credential_supervisors.CredentialSupervisor(
                credentials, credential_storage))
        self.environment = self.credential_supervisor.environment
        self.credentials = self.credential_supervisor.inventory
        self.credential_broker = self.credential_supervisor.broker
        self.workers: dict[str, WorkerChannel] = {}
        self._opening_sessions: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._client_requests: dict[str, asyncio.Future] = {}
        self._next_client_request_id = 0
        self._client_supports_form_elicitation = False

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
            failure = acps.TransportError("ACP client connection closed")
            for future in list(self._client_requests.values()):
                if not future.done():
                    future.set_exception(failure)
            self._client_requests.clear()
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
            self._resolve_client_response(message)
            return
        if request_id is None and method not in (
                "session/cancel", "session/close"):
            return
        if method == "session/prompt":
            forwarded = asyncio.Event()
            self._start_task(
                self._answer(message, forwarded=forwarded),
                name=f"acp-client-request-{request_id}",
            )
            # Preserve client order only until the request reaches the worker;
            # its long-running reply remains concurrent with cancellation.
            await forwarded.wait()
            return
        if method in RESTORE_METHODS:
            # The transport loop must remain free to route cancellation and
            # responses to reverse requests such as elicitation/create.
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

    async def _answer(
            self, message: dict, forwarded: asyncio.Event | None = None):
        method = message.get("method")
        request_id = message.get("id")
        try:
            result = await self.dispatch(
                method,
                message.get("params") or {},
                request_id=request_id,
                forwarded=forwarded,
            )
        except acps.TransportError as error:
            self.write(acps.response(
                request_id,
                error={"code": error.code, "message": str(error)}))
            return
        except Exception as error:
            self.write(acps.response(
                request_id,
                error={"code": acps.INTERNAL_ERROR, "message": str(error)}))
            return
        finally:
            if forwarded is not None:
                forwarded.set()
        self.write(acps.response(request_id, result=result))

    async def dispatch(
            self, method: str, params: dict, request_id=None,
            forwarded: asyncio.Event | None = None):
        if method == "initialize":
            return self.initialize(params)
        if method == "session/new":
            return await self.new_session(params)
        if method in RESTORE_METHODS:
            return await self.restore_session(
                method, params, request_id=request_id)
        if method == "session/list":
            return self.list_sessions(params)
        if method == "session/close":
            return await self.close_session(params)
        if method in SESSION_METHODS:
            return await self.forward_to_worker(
                method, params, forwarded=forwarded)
        raise acps.TransportError(
            f"method not found: {method}", code=acps.METHOD_NOT_FOUND)

    def initialize(self, params: dict) -> dict:
        client_capabilities = params.get("clientCapabilities")
        if not isinstance(client_capabilities, dict):
            client_capabilities = {}
        elicitation = client_capabilities.get("elicitation")
        self._client_supports_form_elicitation = (
            isinstance(elicitation, dict)
            and isinstance(elicitation.get("form"), dict)
        )
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
                    "resume": {},
                    "close": {},
                },
            },
        }

    def _resolve_client_response(self, message: dict) -> None:
        request_id = message.get("id")
        future = self._client_requests.get(str(request_id))
        if future is None or future.done():
            return
        if "error" in message:
            error = message.get("error") or {}
            future.set_exception(acps.TransportError(
                str(error.get("message") or "ACP client request failed"),
                code=error.get("code", acps.INTERNAL_ERROR),
            ))
        elif "result" in message:
            future.set_result(message.get("result"))
        else:
            future.set_exception(acps.TransportError(
                "ACP client response has neither result nor error",
                code=acps.INVALID_PARAMS,
            ))

    async def _request_client(self, method: str, params: dict):
        self._next_client_request_id += 1
        request_id = f"loki-{self._next_client_request_id}"
        future = asyncio.get_running_loop().create_future()
        self._client_requests[request_id] = future
        try:
            self.write(acps.request(request_id, method, params))
            return await future
        finally:
            self._client_requests.pop(request_id, None)

    async def _authorize_saved_connection(
            self, descriptor: ConnectionDescriptor,
            restore_request_id) -> None:
        if not self._client_supports_form_elicitation:
            raise acps.TransportError(
                "restoring a saved network connection requires an ACP "
                "client with form elicitation support",
                code=acps.INVALID_PARAMS,
            )
        facts = "\n".join(
            f"{label}: {json.dumps(value, ensure_ascii=True)}"
            for label, value in connection_display_fields(descriptor)
        )
        result = await self._request_client("elicitation/create", {
            # A restore has not committed a session yet, so ACP requires
            # request scope rather than a fabricated active-session scope.
            "requestId": restore_request_id,
            "mode": "form",
            "message": (
                "Authorize Loki to use this saved connection?\n" + facts),
            "requestedSchema": {
                "type": "object",
                "properties": {
                    "authorize": {
                        "type": "boolean",
                        "title": "Use saved connection",
                        "description": (
                            "Allow this resumed session to make requests "
                            "using the connection shown above."),
                        "default": False,
                    },
                },
                "required": ["authorize"],
            },
        })
        content = (
            result.get("content") if isinstance(result, dict) else None)
        accepted = (
            isinstance(result, dict)
            and result.get("action") == "accept"
            and isinstance(content, dict)
            and content.get("authorize") is True
        )
        if not accepted:
            raise acps.TransportError(
                "saved connection authorization was not accepted",
                code=acps.INVALID_PARAMS,
            )

    @staticmethod
    def _working_directory(params: dict) -> str:
        cwd = params.get("cwd")
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

    async def _open_worker(self, *, cwd: str, open_method: str,
                           session_id: str | None = None,
                           restore_request_id=None) -> tuple[str, dict]:
        if session_id is None:
            session_id = f"loki-{uuid.uuid4()}"
        if (session_id in self.workers
                or session_id in self._opening_sessions):
            raise acps.TransportError(
                f"session {session_id!r} is already active",
                code=acps.INVALID_PARAMS,
            )
        self._opening_sessions.add(session_id)
        channel = None
        try:
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
                    # ACP uses pipes, not a terminal. A new session prevents
                    # an inherited controlling terminal from becoming an
                    # escape channel through TIOCSTI or terminal signals.
                    start_new_session=True,
                )
            finally:
                if process is None:
                    await delegation.close()
                else:
                    delegation.child_spawned()
            channel = WorkerChannel(
                session_id, process, self.write, delegation)
            prepared = await channel.request(
                "session/prepare_open",
                {
                    "sessionId": session_id,
                    "cwd": cwd,
                    "openMethod": open_method,
                },
            )
            raw_descriptor = (prepared or {}).get(
                "authorizationConnection")
            if raw_descriptor is not None:
                try:
                    descriptor = ConnectionDescriptor.from_dict(
                        raw_descriptor)
                except ConnectionDescriptorError as error:
                    raise acps.TransportError(
                        f"worker returned an invalid connection: {error}"
                    ) from error
                await self._authorize_saved_connection(
                    descriptor, restore_request_id)
            reply = await channel.request("session/commit_open", {})
            # Publication is the commit point. Before this assignment no
            # prompt, config change, or close request can reach the worker.
            self.workers[session_id] = channel
            return session_id, reply or {}
        except BaseException:
            if channel is not None:
                await channel.close()
            raise
        finally:
            self._opening_sessions.discard(session_id)

    async def new_session(self, params: dict) -> dict:
        # session/new is intentionally fresh. Restoration has separate
        # session/load and session/resume operations and cannot alias this.
        self._validate_session_setup(params)
        session_id, worker_reply = await self._open_worker(
            cwd=self._working_directory(params),
            open_method="session/new",
        )
        result = {"sessionId": session_id}
        config_options = worker_reply.get("configOptions")
        if config_options:
            result["configOptions"] = config_options
        return result

    async def restore_session(
            self, method: str, params: dict, request_id=None) -> dict:
        """Restore one saved session with the method's ACP replay semantics."""
        saved_id = params.get("sessionId")
        if not isinstance(saved_id, str) or not saved_id:
            raise acps.TransportError(
                f"{method} requires sessionId",
                code=acps.INVALID_PARAMS,
            )
        self._validate_session_setup(params)
        _, worker_reply = await self._open_worker(
            cwd=self._working_directory(params),
            open_method=method,
            session_id=saved_id,
            restore_request_id=request_id,
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

    async def forward_to_worker(
            self, method: str, params: dict,
            forwarded: asyncio.Event | None = None):
        session_id = params.get("sessionId")
        channel = self.workers.get(session_id)
        if channel is None:
            raise acps.TransportError(
                f"unknown session {session_id!r}",
                code=acps.INVALID_PARAMS,
            )
        return await channel.request(
            method, params, forwarded=forwarded)
