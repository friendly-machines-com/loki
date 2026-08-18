"""ACP front process: transport owner, session registry, worker spawner.

One front process per configured agent (what a client like Zed spawns).
It answers the session-free methods itself -- initialize, session/new,
session/list -- and routes every per-session line to the worker process
it spawned for that session.  Each worker is one single-session Loki,
so conversations are isolated the way the terminal front-end is.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from . import acps, savefiles
from .loki import CHAT_LOG_DIR

PROTOCOL_VERSION = 1
AGENT_INFO = {"name": "loki", "title": "Loki", "version": "0.1"}

SESSION_METHODS = (
    "session/prompt",
    "session/cancel",
    "session/load",
    "session/resume",
    "session/set_config_option",
)

WORKER_COMMAND = [sys.executable, "-m", "loki_agent.acp_worker_main"]


class Front:
    def __init__(self, read, write):
        self.read = read
        self.write = write
        self.workers: dict[str, asyncio.subprocess.Process] = {}
        self._next_worker_id = 0

    async def run(self):
        async for message in self.read():
            await self.handle(message)

    async def handle(self, message: dict):
        method = message.get("method")
        request_id = message.get("id")
        if method is None:
            return
        if request_id is None and method != "session/cancel":
            return  # other notifications are not for us
        if method == "session/prompt":
            # Long-running: dispatch concurrently so a session/cancel
            # arriving from the client while the turn runs is still read
            # and routed instead of queueing behind the prompt.
            asyncio.get_running_loop().create_task(self._answer(message))
            return
        if request_id is None:
            # cancel-as-notification: route it, answer nothing.
            params = message.get("params") or {}
            asyncio.get_running_loop().create_task(
                self.dispatch(method, params))
            return
        await self._answer(message)

    async def _answer(self, message: dict):
        method = message.get("method")
        request_id = message.get("id")
        try:
            result = await self.dispatch(method, message.get("params") or {})
        except Exception as error:
            self.write(acps.response(
                request_id,
                error={"code": acps.INTERNAL_ERROR, "message": str(error)}))
            return
        if result is not None:
            self.write(acps.response(request_id, result=result))

    async def dispatch(self, method: str, params: dict):
        if method == "initialize":
            return self.initialize(params)
        if method == "session/new":
            return await self.new_session(params)
        if method == "session/list":
            return self.list_sessions(params)
        if method in SESSION_METHODS:
            return await self.forward_to_worker(method, params)
        raise acps.TransportError(f"method not found: {method}")

    # -- session-free methods ----------------------------------------------

    def initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentInfo": AGENT_INFO,
            "authMethods": [],
            "agentCapabilities": {
                "loadSession": True,
                "promptCapabilities": {},
            },
        }

    async def new_session(self, params: dict) -> dict:
        cwd = params.get("cwd") or os.getcwd()
        self._next_worker_id += 1
        session_id = f"loki-{os.getpid()}-{self._next_worker_id}"
        process = await asyncio.create_subprocess_exec(
            *WORKER_COMMAND,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # inherit: worker logs land in our stderr (ACP log channel)
        )
        self.workers[session_id] = process
        worker_reply = await self._worker_request(
            session_id, "session/open",
            {"sessionId": session_id, "resume": params.get("resume")})
        result = {"sessionId": session_id}
        config_options = (worker_reply or {}).get("configOptions")
        if config_options:
            result["configOptions"] = config_options
        return result

    def list_sessions(self, params: dict) -> dict:
        import datetime
        entries = []
        for path in savefiles.filtered_chat_log_paths("", CHAT_LOG_DIR):
            try:
                with open(path, "r", encoding="utf-8") as file_obj:
                    blob = json.load(file_obj)
            except (OSError, json.JSONDecodeError):
                continue
            state = blob.get("session_state") if isinstance(blob, dict) else {}
            cwd = (state or {}).get("cwd") or (state or {}).get("shell_cwd")
            if not cwd:
                continue  # SessionInfo requires cwd; logs without one stay loadable
            entries.append({
                "sessionId": os.path.basename(path),
                "cwd": cwd,
                "updatedAt": datetime.datetime.fromtimestamp(
                    os.path.getmtime(path),
                    datetime.timezone.utc).isoformat(),
            })
        return {"sessions": entries}

    # -- routing -----------------------------------------------------------

    async def forward_to_worker(self, method: str, params: dict):
        session_id = params.get("sessionId")
        if session_id not in self.workers:
            raise acps.TransportError(f"unknown session {session_id!r}")
        return await self._worker_request(session_id, method, params)

    async def _worker_request(self, session_id: str, method: str,
                              params: dict):
        process = self.workers[session_id]
        request_id = f"front-{method}-{session_id}"
        line = json.dumps(acps.request(request_id, method, params))
        process.stdin.write(line.encode("utf-8"))
        process.stdin.write(b"\n")
        await process.stdin.drain()
        # Read lines until the reply with our id arrives; worker
        # notifications (session/update) are forwarded to the client.
        while True:
            raw = await process.stdout.readline()
            if not raw:
                raise acps.TransportError(f"worker for {session_id} exited")
            message = json.loads(raw.decode("utf-8"))
            if message.get("id") == request_id:
                if "error" in message:
                    raise acps.TransportError(
                        str(message["error"].get("message")))
                return message.get("result")
            self.write(message)  # forward notifications/requests verbatim
