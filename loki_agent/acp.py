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
import logging
import os
import sys
import uuid

from . import (
    __version__,
    acps,
    credential_supervisors,
    endpoint_pins,
    runtime_isolation,
    savefiles,
)
from .connections import (
    ConnectionDescriptor,
    ConnectionDescriptorError,
    connection_display_fields,
)
from .credentials import CredentialStore
from .loki import CHAT_LOG_DIR, chat_log_dir_for
from .runtime_isolation import RuntimeIsolationError

logger = logging.getLogger(__name__)

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

# Bounds for one model-authored question. The question and its options are
# model output rendered by the client, so their size is capped before the
# front will carry them.
MAX_ASK_QUESTION_CHARS = 4000
MAX_ASK_OPTIONS = 8
MAX_ASK_LABEL_CHARS = 200
MAX_ASK_DESCRIPTION_CHARS = 1000


class WorkerChannel:
    """One request multiplexer around one worker subprocess."""

    def __init__(self, session_id: str, process: asyncio.subprocess.Process,
                 forward, credential_delegation=None, reverse_handler=None):
        self.session_id = session_id
        self.process = process
        self.forward = forward
        self.reverse_handler = reverse_handler
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

    async def respond(self, request_id, result=None, error=None):
        """Answer one worker-initiated reverse request.

        Best effort by design: a worker that is closing or gone cannot
        receive the answer, and its pending request future dies with the
        reader's failure fanout.
        """
        encoded = (
            json.dumps(acps.response(request_id, result=result, error=error),
                       ensure_ascii=False) + "\n").encode("utf-8")
        try:
            async with self._write_lock:
                if self._closed or self.process.stdin is None:
                    return
                self.process.stdin.write(encoded)
                await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionError, OSError):
            pass

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
                method = message.get("method")
                if method is not None:
                    if request_id is not None:
                        # A worker-initiated request. The front owns the only
                        # sanctioned path back to the client, so reverse
                        # requests dispatch there instead of forward.
                        if self.reverse_handler is not None:
                            self.reverse_handler(message)
                        continue
                    # Only notifications belong on the outward channel.
                    self.forward(message)
                    continue
                if request_id is not None:
                    future = self._pending.get(str(request_id))
                    if future is None or future.done():
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
        # The platform process is released only after the reader has stopped:
        # on Windows this closes the stdio threads, the front pipe handles
        # and the job whose close kills any descendants.  POSIX subprocesses
        # own their transport and need nothing here.
        runtime_isolation.close_runtime_process(self.process)
        if self.credential_delegation is not None:
            await self.credential_delegation.close()
            self.credential_delegation = None


class SessionOperations:
    """Front-owned lifetime of one session, including its provisional worker.

    There is no prompt queue here. A configuration/open operation excludes new
    ordinary requests for this session; conflicting requests fail explicitly.
    The object, rather than its reusable session ID, owns tasks and approvals.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = "opening"
        self.channel = None
        self.delegation = None
        self.changing = False
        self.tasks: set[asyncio.Task] = set()
        # Ask-the-user operations awaiting the client. session/cancel
        # cancels them so an abandoned question cannot outlive its turn.
        self.pending_asks: set[asyncio.Task] = set()
        self.client_requests: set[str] = set()
        self.forwarded = asyncio.Event()
        self.forwarded.set()
        self.cleanup = None


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
        self._sessions: dict[str, SessionOperations] = {}
        self._connected = True
        self._tasks: set[asyncio.Task] = set()
        self._client_requests: dict[str, asyncio.Future] = {}
        self._next_client_request_id = 0
        self._client_supports_form_elicitation = False

    @property
    def workers(self) -> dict[str, WorkerChannel]:
        """Published channels only; ownership lives in _sessions."""
        return {session_id: owner.channel
                for session_id, owner in self._sessions.items()
                if owner.state == "active"}

    async def run(self):
        try:
            async for message in self.read():
                self.handle(message)
        finally:
            await self.shutdown()

    async def shutdown(self):
        self._connected = False
        owners = list(self._sessions.values())
        for owner in owners:
            self._begin_close(owner)
        cleanups = {owner.cleanup for owner in owners}
        tasks = list(self._tasks - cleanups)
        # _begin_close already cancelled session operations. A second cancel
        # could interrupt their acquisition-failure cleanup while it unwinds.
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        for future in self._client_requests.values():
            future.cancel()
        self._client_requests.clear()
        # Cleanup tasks own worker/delegation release. Cancelling them alongside
        # requests could abandon an unpublished worker or its credential pipe.
        await asyncio.gather(*tasks, *cleanups, return_exceptions=True)

    def _start_task(self, coroutine, *, name: str, owner=None):
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if owner is not None:
            owner.tasks.add(task)
            task.add_done_callback(owner.tasks.discard)
        return task

    def _error(self, message, error):
        if not self._connected:
            return
        if message.get("id") is None:
            print(f"ACP notification {message.get('method')!r} failed: "
                  f"{error!r}", file=sys.stderr)
        else:
            self.write(acps.response(message["id"], error={
                "code": getattr(error, "code", acps.INTERNAL_ERROR),
                "message": str(error),
            }))

    def _reserve_session(self, session_id):
        if not isinstance(session_id, str) or not session_id:
            raise acps.TransportError(
                "session open requires sessionId", code=acps.INVALID_PARAMS)
        if session_id in self._sessions:
            raise acps.TransportError(
                f"session {session_id!r} is already active",
                code=acps.INVALID_PARAMS)
        owner = SessionOperations(session_id)
        self._sessions[session_id] = owner
        return owner

    def _check_owner(self, owner):
        if (self._sessions.get(owner.session_id) is not owner
                or owner.state == "closing"):
            raise acps.TransportError("session operation was closed")
        channel = owner.channel
        if channel is not None and (
                channel._closed or channel.process.returncode is not None):
            raise acps.TransportError("session worker is not running")

    def handle(self, message: dict):
        """Route/admit synchronously; never await operation work in the reader.

        The same input stream carries approval answers and user requests. Even
        waiting for a forwarding milestone here could starve lifecycle controls.
        Admission reserves the session before scheduling, so buffered requests
        cannot race past an approval operation whose task has not started yet.
        """
        method = message.get("method")
        request_id = message.get("id")
        if method is None:
            self._resolve_client_response(message)
            return
        if not self._connected:
            return
        if request_id is None and method not in [
                "session/cancel", "session/close"]:
            return
        params = message.get("params") or {}
        owner = None
        predecessor = None
        forwarded = None
        changing = False
        try:
            if not isinstance(params, dict):
                raise acps.TransportError(
                    "request params must be an object", code=acps.INVALID_PARAMS)
            if method == "session/new" or method in RESTORE_METHODS:
                self._validate_session_setup(params)
                self._working_directory(params)
                session_id = (f"loki-{uuid.uuid4()}"
                              if method == "session/new"
                              else params.get("sessionId"))
                owner = self._reserve_session(session_id)
                changing = True
            elif method in SESSION_METHODS or method == "session/close":
                session_id = params.get("sessionId")
                owner = self._sessions.get(session_id)
                if owner is None or owner.state == "closing":
                    raise acps.TransportError(
                        f"unknown session {session_id!r}",
                        code=acps.INVALID_PARAMS)
                if method == "session/close":
                    self._begin_close(owner)
                    self._start_task(
                        self._answer(message, owner=owner),
                        name=f"acp-client-close-{request_id}")
                    return
                if method != "session/cancel" and (
                        owner.state == "opening" or owner.changing):
                    raise acps.TransportError(
                        "session configuration or approval is pending; "
                        "request was not executed",
                        code=acps.INVALID_PARAMS)
                if owner.state != "active":
                    raise acps.TransportError(
                        f"unknown session {session_id!r}",
                        code=acps.INVALID_PARAMS)
                predecessor = owner.forwarded
                if method == "session/prompt":
                    forwarded = asyncio.Event()
                    owner.forwarded = forwarded
                changing = method == "session/set_config_option"
            if changing:
                owner.changing = True
            task = self._start_task(
                self._operate(message, owner, predecessor, forwarded, changing),
                name=f"acp-client-request-{request_id}", owner=owner)
            task.add_done_callback(
                lambda task: self._cancelled_before_start(task, message, forwarded))
        except Exception as error:
            self._error(message, error)

    def _cancelled_before_start(self, task, message, forwarded):
        # Cancelling a task before its first scheduling never enters its try/
        # finally. Supply that request's missing response and forwarding release.
        # _operate handles an entered task's cancellation and returns normally,
        # so this callback cannot send a second response for that case.
        if task.cancelled():
            if forwarded is not None:
                forwarded.set()
            self._error(message, acps.TransportError(
                "session operation was closed"))

    async def _operate(self, message, owner, predecessor, forwarded, changing):
        try:
            # Reuse the existing prompt-forwarded boundary, but wait in the
            # operation, not the reader. A following effort update/cancel must
            # reach the worker after the prompt whose snapshot it follows.
            # This is not a queue behind approval: those requests were rejected
            # synchronously at admission above.
            if predecessor is not None:
                await predecessor.wait()
            if owner is not None:
                self._check_owner(owner)
            await self._answer(message, forwarded=forwarded, owner=owner)
        except asyncio.CancelledError:
            self._error(message, acps.TransportError(
                "session operation was closed"))
            return
        except Exception as error:
            self._error(message, error)
        finally:
            if forwarded is not None:
                forwarded.set()
            if changing:
                owner.changing = False
            if owner is not None and owner.state == "opening":
                # An open which failed before _open_worker (e.g. validation)
                # must not leave an ID reserved after its request completes.
                if self._sessions.get(owner.session_id) is owner:
                    del self._sessions[owner.session_id]

    def _begin_close(self, owner):
        if owner.cleanup is not None:
            return owner.cleanup
        # Invalidation is synchronous: an already-buffered acceptance must not
        # publish a worker or write a pin after this close wins admission.
        owner.state = "closing"
        if owner.delegation is not None:
            owner.delegation.revoke_now()
        for request_id in owner.client_requests:
            future = self._client_requests.pop(request_id, None)
            if future is not None:
                future.cancel()
        tasks = list(owner.tasks)
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        owner.cleanup = self._start_task(
            self._finish_close(owner, tasks),
            name=f"acp-session-close-{owner.session_id}")
        return owner.cleanup

    async def _finish_close(self, owner, tasks):
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            if owner.channel is not None:
                await owner.channel.close()
            elif owner.delegation is not None:
                await owner.delegation.close()
        finally:
            if self._sessions.get(owner.session_id) is owner:
                del self._sessions[owner.session_id]

    def _worker_finished(self, owner, task):
        # A dead worker must also release a caller awaiting the *client*, not
        # just the worker request futures resolved by WorkerChannel itself.
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error("ACP worker reader failed: %r", error)
        if owner.state != "closing":
            self._begin_close(owner)

    async def _answer(
            self, message: dict, forwarded: asyncio.Event | None = None,
            owner=None):
        method = message.get("method")
        request_id = message.get("id")
        try:
            result = await self.dispatch(
                method,
                message.get("params") or {},
                request_id=request_id,
                forwarded=forwarded,
                owner=owner,
            )
        except Exception as error:
            self._error(message, error)
            return
        finally:
            if forwarded is not None:
                forwarded.set()
        if not self._connected or request_id is None:
            return
        commands = None
        if isinstance(result, dict):
            commands = result.pop("lokiCommands", None)
        self.write(acps.response(request_id, result=result))
        if commands:
            # Ordering is the point: the client learns the session from the
            # reply above, then receives the commands advertised for it.
            session_id = (
                result.get("sessionId") if isinstance(result, dict) else None)
            if not isinstance(session_id, str) or not session_id:
                session_id = (message.get("params") or {}).get("sessionId")
            if isinstance(session_id, str) and session_id:
                self.write(acps.notification("session/update", {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "available_commands_update",
                        "availableCommands": commands,
                    },
                }))

    async def dispatch(
            self, method: str, params: dict, request_id=None,
            forwarded: asyncio.Event | None = None, owner=None):
        if method == "initialize":
            return self.initialize(params)
        if method == "session/new":
            return await self.new_session(params, owner=owner)
        if method in RESTORE_METHODS:
            return await self.restore_session(
                method, params, request_id=request_id, owner=owner)
        if method == "session/list":
            return self.list_sessions(params)
        if method == "session/close":
            return await self.close_session(params, owner=owner)
        if method in SESSION_METHODS:
            return await self.forward_to_worker(
                method, params, forwarded=forwarded,
                request_id=request_id, owner=owner)
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
            error = message.get("error")
            if not isinstance(error, dict):
                future.set_exception(acps.TransportError(
                    "ACP client response has an invalid error object",
                    code=acps.INVALID_PARAMS))
                return
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

    async def _request_client(self, method: str, params: dict, *, owner):
        self._check_owner(owner)
        self._next_client_request_id += 1
        request_id = f"loki-{self._next_client_request_id}"
        future = asyncio.get_running_loop().create_future()
        self._client_requests[request_id] = future
        owner.client_requests.add(request_id)
        try:
            self.write(acps.request(request_id, method, params))
            result = await future
            self._check_owner(owner)
            return result
        finally:
            self._client_requests.pop(request_id, None)
            owner.client_requests.discard(request_id)

    def _worker_request(self, owner, message: dict):
        """Admit one worker-initiated reverse request (reader callback).

        The reader must not await: the ask lives as an owned task so
        lifecycle controls keep flowing while a question is open.
        """
        request_id = message.get("id")
        if request_id is None:
            return
        method = message.get("method")
        if method == "session/request_input":
            task = self._start_task(
                self._run_request_input(
                    request_id, message.get("params") or {}, owner=owner),
                name=f"acp-worker-ask-{request_id}", owner=owner)
            owner.pending_asks.add(task)
            task.add_done_callback(owner.pending_asks.discard)
            return
        self._start_task(
            self._refuse_worker_request(
                owner.channel, request_id, method),
            name=f"acp-worker-refused-{request_id}", owner=owner)

    async def _refuse_worker_request(self, channel, request_id, method):
        # A worker bug must surface as an answer, not as a hung model tool
        # call awaiting a reply that will never come.
        if channel is not None:
            await channel.respond(request_id, error={
                "code": acps.METHOD_NOT_FOUND,
                "message": f"front does not accept {method!r} from workers",
            })

    async def _run_request_input(self, request_id, params, *, owner):
        channel = owner.channel
        try:
            result = await self._ask_user_via_elicitation(params, owner=owner)
        except asyncio.CancelledError:
            if channel is not None:
                await channel.respond(
                    request_id, result={"action": "cancelled"})
            raise
        except acps.TransportError as error:
            await channel.respond(request_id, error={
                "code": error.code, "message": str(error)})
            return
        await channel.respond(request_id, result=result)

    async def _ask_user_via_elicitation(self, params: dict, *, owner) -> dict:
        """Ask the user one model-authored question through the client.

        Worker asks are their own class, disjoint from the front-owned
        security elicitations: always session-scoped and form mode, with
        the schema built here from validated fields. A worker can never
        send the user to a URL, leave its session scope, or re-shape an
        approval form.
        """
        self._check_owner(owner)
        if not self._client_supports_form_elicitation:
            raise acps.TransportError(
                "this ACP client cannot answer model questions: no form "
                "elicitation support",
                code=acps.INVALID_PARAMS)
        forbidden = sorted({"requestId", "mode", "url", "elicitationId",
                            "sessionId"} & set(params))
        if forbidden:
            raise acps.TransportError(
                "model questions cannot set " + ", ".join(forbidden)
                + "; scope and mode belong to the front",
                code=acps.INVALID_PARAMS)
        question = params.get("question")
        if (not isinstance(question, str) or not question.strip()
                or len(question) > MAX_ASK_QUESTION_CHARS):
            raise acps.TransportError(
                "model question must be a nonempty string of at most "
                f"{MAX_ASK_QUESTION_CHARS} characters",
                code=acps.INVALID_PARAMS)
        multi_select = params.get("multiSelect", False)
        if multi_select not in [True, False]:
            raise acps.TransportError(
                "model question multiSelect must be a boolean",
                code=acps.INVALID_PARAMS)
        raw_options = params.get("options")
        if (not isinstance(raw_options, list)
                or not 2 <= len(raw_options) <= MAX_ASK_OPTIONS):
            raise acps.TransportError(
                f"model question requires 2 to {MAX_ASK_OPTIONS} options",
                code=acps.INVALID_PARAMS)
        options = []
        for option in raw_options:
            if not isinstance(option, dict):
                raise acps.TransportError(
                    "each model question option must be an object",
                    code=acps.INVALID_PARAMS)
            label = option.get("label")
            if (not isinstance(label, str) or not label.strip()
                    or len(label) > MAX_ASK_LABEL_CHARS):
                raise acps.TransportError(
                    "model question option labels must be nonempty strings"
                    f" of at most {MAX_ASK_LABEL_CHARS} characters",
                    code=acps.INVALID_PARAMS)
            description = option.get("description")
            if description is not None and (
                    not isinstance(description, str)
                    or len(description) > MAX_ASK_DESCRIPTION_CHARS):
                raise acps.TransportError(
                    "model question option descriptions must be strings of "
                    f"at most {MAX_ASK_DESCRIPTION_CHARS} characters",
                    code=acps.INVALID_PARAMS)
            options.append((label, description))
        # The answer value is the label itself, so equal labels would make
        # an answer ambiguous.
        labels = [label for label, _ in options]
        if len(set(labels)) != len(labels):
            raise acps.TransportError(
                "model question option labels must be unique",
                code=acps.INVALID_PARAMS)
        choices = []
        for label, description in options:
            choice = {"const": label, "title": label}
            if description is not None:
                choice["description"] = description
            choices.append(choice)
        if multi_select:
            answer_schema = {
                "type": "array",
                "title": "Answer",
                "items": {"anyOf": choices},
            }
        else:
            answer_schema = {
                "type": "string",
                "title": "Answer",
                "oneOf": choices,
            }
        elicitation = {
            "sessionId": owner.session_id,
            "mode": "form",
            "message": question,
            "requestedSchema": {
                "type": "object",
                "properties": {
                    "answer": answer_schema,
                    "custom": {
                        "type": "string",
                        "title": "Other",
                        "description": (
                            "Type your own answer instead (optional)"),
                    },
                },
                "required": [],
            },
        }
        result = await self._request_client(
            "elicitation/create", elicitation, owner=owner)
        if not isinstance(result, dict):
            raise acps.TransportError(
                "ACP client answer was not an object",
                code=acps.INVALID_PARAMS)
        action = result.get("action")
        if action == "decline":
            # A decline is an answer ("no"), not an abandoned question:
            # the turn continues and the model learns the user refused.
            return {"action": "declined"}
        if action != "accept":
            # cancel and any unknown action dismiss the question; the turn
            # keeps running without an answer.
            return {"action": "cancelled"}
        content = result.get("content")
        if not isinstance(content, dict):
            raise acps.TransportError(
                "ACP client answer was missing its content object",
                code=acps.INVALID_PARAMS)
        custom = content.get("custom")
        if isinstance(custom, str) and custom.strip():
            custom = custom.strip()
        else:
            custom = None
        picks = content.get("answer")
        if multi_select:
            if picks is not None and not (
                    isinstance(picks, list)
                    and all(isinstance(item, str) for item in picks)):
                raise acps.TransportError(
                    "ACP client multi-select answer must be an array of "
                    "strings",
                    code=acps.INVALID_PARAMS)
            answer = [item for item in (picks or []) if item]
            if not answer and custom is not None:
                answer = [custom]
            if not answer:
                raise acps.TransportError(
                    "ACP client answer selected nothing",
                    code=acps.INVALID_PARAMS)
            answered = {"action": "answered", "answer": answer}
            if custom is not None and answer != [custom]:
                answered["custom"] = custom
            return answered
        if picks is not None and not isinstance(picks, str):
            raise acps.TransportError(
                "ACP client answer must be a string",
                code=acps.INVALID_PARAMS)
        if picks:
            answered = {"action": "answered", "answer": picks}
            if custom is not None:
                # The typed text supplements, never replaces, a picked
                # option.
                answered["custom"] = custom
            return answered
        if custom is not None:
            return {"action": "answered", "answer": custom}
        raise acps.TransportError(
            "ACP client answer contained no answer",
            code=acps.INVALID_PARAMS)

    async def _authorize_saved_connection(
            self, descriptor: ConnectionDescriptor,
            restore_request_id, working_directory=None, *, owner) -> None:
        if not self._client_supports_form_elicitation:
            raise acps.TransportError(
                "restoring a saved network connection requires an ACP "
                "client with form elicitation support",
                code=acps.INVALID_PARAMS,
            )
        display = list(connection_display_fields(descriptor))
        if working_directory:
            # The workspace is part of what the client approves: it is where
            # every tool reads and writes for this session.
            display.append(("Working directory", working_directory))
        facts = "\n".join(
            f"{label}: {json.dumps(value, ensure_ascii=True)}"
            for label, value in display
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
        }, owner=owner)
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
                           restore_request_id=None, owner=None) -> tuple[str, dict]:
        if owner is None:
            owner = self._reserve_session(
                session_id or f"loki-{uuid.uuid4()}")
        session_id = owner.session_id
        self._check_owner(owner)
        channel = None
        try:
            delegation = await self.credential_supervisor.delegate()
            # Close may arrive during spawn, before a WorkerChannel exists.
            # The provisional owner must already be able to revoke authority.
            owner.delegation = delegation
            process = None
            try:
                # The platform seam spawns the worker contained where the
                # platform has a container (Windows AppContainer) and piped
                # on stdio everywhere; the front never launches it directly.
                process = await runtime_isolation.start_worker(
                    cwd, self.environment, delegation)
            except RuntimeIsolationError as error:
                raise acps.TransportError(
                    f"could not start worker: {error}",
                    code=acps.INTERNAL_ERROR,
                ) from error
            finally:
                if process is None:
                    await delegation.close()
                else:
                    delegation.child_spawned()
            channel = WorkerChannel(
                session_id, process, self.write, delegation,
                reverse_handler=lambda message: self._worker_request(
                    owner, message))
            owner.channel = channel
            channel._reader_task.add_done_callback(
                lambda task: self._worker_finished(owner, task))
            self._check_owner(owner)
            prepared = await channel.request(
                "session/prepare_open",
                {
                    "sessionId": session_id,
                    "cwd": cwd,
                    "openMethod": open_method,
                    # The worker advertises its ask-the-user tool only when
                    # the client can actually answer it, so the model is
                    # never given a question tool that always fails.
                    "formElicitation": self._client_supports_form_elicitation,
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
                    descriptor, restore_request_id, cwd, owner=owner)
            self._check_owner(owner)
            reply = await channel.request("session/commit_open", {})
            self._check_owner(owner)
            # Publication is the commit point. Close can invalidate even a
            # provisional owner, but ordinary requests cannot reach it yet.
            owner.state = "active"
            return session_id, reply or {}
        except BaseException:
            owner.state = "closing"
            try:
                if channel is not None:
                    await channel.close()
            finally:
                if (owner.cleanup is None
                        and self._sessions.get(session_id) is owner):
                    del self._sessions[session_id]
            raise

    async def new_session(self, params: dict, owner=None) -> dict:
        # session/new is intentionally fresh. Restoration has separate
        # session/load and session/resume operations and cannot alias this.
        self._validate_session_setup(params)
        session_id, worker_reply = await self._open_worker(
            cwd=self._working_directory(params),
            open_method="session/new",
            owner=owner,
        )
        result = {"sessionId": session_id}
        config_options = worker_reply.get("configOptions")
        if config_options:
            result["configOptions"] = config_options
        commands = worker_reply.get("lokiCommands")
        if commands:
            result["lokiCommands"] = commands
        return result

    async def restore_session(
            self, method: str, params: dict, request_id=None, owner=None) -> dict:
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
            owner=owner,
        )
        result = {}
        config_options = worker_reply.get("configOptions")
        if config_options:
            result["configOptions"] = config_options
        commands = worker_reply.get("lokiCommands")
        if commands:
            result["lokiCommands"] = commands
        return result

    def list_sessions(self, params: dict) -> dict:
        requested = params.get("cwd")
        cwd_filter = None
        if requested is not None:
            if (not isinstance(requested, str)
                    or not os.path.isabs(requested)):
                raise acps.TransportError(
                    "session/list cwd must be an absolute path",
                    code=acps.INVALID_PARAMS,
                )
            # Case-folded names are compared, nothing is asked of the
            # filesystem: Loki never resolves or normalises an operand.
            cwd_filter = os.path.normcase(requested)
        # Sessions live in the workspace the client names; with no filter, in
        # this process's own workspace, the directory it was started in.
        root = (chat_log_dir_for(requested) if requested is not None
                else CHAT_LOG_DIR)
        entries = []
        for path in savefiles.filtered_chat_log_paths("", root):
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
                    and os.path.normcase(cwd) != cwd_filter):
                continue
            entries.append({
                "sessionId": (
                    os.path.basename(path)[len("chat-"):-len(".json")]),
                "cwd": cwd,
                "updatedAt": datetime.datetime.fromtimestamp(
                    modified, datetime.timezone.utc).isoformat(),
            })
        return {"sessions": entries}

    async def close_session(self, params: dict, owner=None) -> dict:
        session_id = params.get("sessionId")
        if owner is None:
            owner = self._sessions.get(session_id)
        if owner is None:
            raise acps.TransportError(
                f"unknown session {session_id!r}",
                code=acps.INVALID_PARAMS,
            )
        # EOF can cancel the close request itself; worker/delegation cleanup
        # must finish independently of whether its reply can still be delivered.
        await asyncio.shield(self._begin_close(owner))
        return {}

    async def forward_to_worker(
            self, method: str, params: dict,
            forwarded: asyncio.Event | None = None,
            request_id=None, owner=None):
        session_id = params.get("sessionId")
        if owner is None:
            owner = self._sessions.get(session_id)
        if owner is None or owner.state != "active":
            raise acps.TransportError(
                f"unknown session {session_id!r}",
                code=acps.INVALID_PARAMS,
            )
        self._check_owner(owner)
        channel = owner.channel
        if method == "session/cancel":
            # An in-flight model question belongs to the turn being
            # cancelled. Answer it "cancelled" and abandon the client
            # elicitation; a late client answer resolves no future.
            for task in list(owner.pending_asks):
                if not task.cancelling():
                    task.cancel()
        if method == "session/set_config_option":
            # A catalog endpoint decides where a static credential is sent.
            # The worker knows which pair a config value selects; the front
            # owns the client channel, so the approval is asked here and the
            # worker's own check then sees an approved pair.
            await self._approve_config_endpoint(
                channel, params, request_id, owner=owner)
        self._check_owner(owner)
        return await channel.request(
            method, params, forwarded=forwarded)

    async def _approve_config_endpoint(
            self, channel, params: dict, request_id=None, *, owner) -> None:
        """Ask the client to approve a catalog endpoint+credential pair."""
        selection = await channel.request(
            "session/describe_config_selection", params)
        if not isinstance(selection, dict) or not selection:
            return
        if not self._client_supports_form_elicitation:
            raise acps.TransportError(
                "switching to this provider requires an ACP client with "
                "form elicitation support, so its endpoint can be approved",
                code=acps.INVALID_PARAMS,
            )
        facts = [("Endpoint", selection.get("endpoint")),
                 ("Credential", selection.get("credential"))]
        if selection.get("changed"):
            facts = [
                ("Approved endpoint", selection.get("approvedEndpoint")),
                ("Approved credential", selection.get("approvedCredential")),
                *facts,
            ]
        result = await self._request_client("elicitation/create", {
            "requestId": (
                request_id if request_id is not None
                else f"approve-endpoint-{channel.session_id}"),
            "mode": "form",
            "message": (
                "Send this credential to this endpoint?\n"
                + "\n".join(
                    f"{label}: {json.dumps(value, ensure_ascii=True)}"
                    for label, value in facts)),
            "requestedSchema": {
                "type": "object",
                "properties": {
                    "approve": {
                        "type": "boolean",
                        "title": "Use this endpoint and credential",
                        "description": (
                            "Approve sending the named credential to the "
                            "named endpoint for this provider."),
                        "default": False,
                    },
                },
                "required": ["approve"],
            },
        }, owner=owner)
        content = (
            result.get("content") if isinstance(result, dict) else None)
        approved = (
            isinstance(result, dict)
            and result.get("action") == "accept"
            and isinstance(content, dict)
            and content.get("approve") is True
        )
        if not approved:
            raise acps.TransportError(
                "the provider endpoint was not approved",
                code=acps.INVALID_PARAMS,
            )
        # Consent belongs to this operation, not any replacement session using
        # the same ID. No await separates the lifetime check and durable write.
        self._check_owner(owner)
        endpoint_pins.record(
            str(selection.get("providerId")),
            str(selection.get("endpoint")),
            str(selection.get("credential")),
        )
