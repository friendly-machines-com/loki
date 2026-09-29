"""Optional local bridge transport; no frontend or conversation ownership."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import socket
import struct
from dataclasses import dataclass


logger = logging.getLogger(__name__)
VERSION = 1


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    frame_bytes: int = 1024 * 1024
    prompt_bytes: int = 64 * 1024
    pending_inputs: int = 32
    events: int = 256
    buffer_bytes: int = 8 * 1024 * 1024
    completed_inputs: int = 256
    connect_timeout: float = 3
    write_timeout: float = 5
    retry_min: float = 1
    retry_max: float = 30


def encode(message, limits):
    data = (json.dumps(message, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")
    if len(data) > limits.frame_bytes:
        raise ProtocolError("bridge frame exceeds size limit")
    return data


async def read_message(reader, limits):
    try:
        raw = await reader.readline()
    except ValueError as error:
        raise ProtocolError("bridge frame exceeds size limit") from error
    if not raw:
        raise ConnectionError("bridge disconnected")
    if len(raw) > limits.frame_bytes or not raw.endswith(b"\n"):
        raise ProtocolError("invalid bridge framing")
    try:
        message = json.loads(raw.decode("utf-8"), parse_constant=_bad_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ProtocolError("invalid bridge JSON") from error
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ProtocolError("bridge message requires an object with a type")
    return message


def _bad_constant(value):
    raise ValueError("non-finite JSON number")


class BridgeTransport:
    """One reconnecting connection. Callbacks only enqueue work, never run it.

    The owning runtime opens the socket (non-inheritable by default). This is
    not credential IPC and grants no broker capability. A pathname/peer UID is
    a local trust boundary, not protection against tools with the same UID.
    """

    def __init__(self, path, *, register, receive, limits=None, peer_uid=None,
                 report=None):
        self.path = path
        self.register = register
        self.receive = receive
        self.limits = limits or Limits()
        self.peer_uid = os.getuid() if peer_uid is None else peer_uid
        self.report = report or logger.warning
        self.connected = False
        self._queue = asyncio.Queue(maxsize=self.limits.events)
        self._bytes = 0
        self._task = None
        self._writer = None
        self._warned = False

    async def __aenter__(self):
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def send(self, message):
        if not self.connected:
            return False
        data = encode(message, self.limits)
        if (self._queue.full()
                or self._bytes + len(data) > self.limits.buffer_bytes):
            # Session events remain in its bounded journal for reconnect replay.
            self.connected = False
            self._writer.close()
            return False
        self._bytes += len(data)
        self._queue.put_nowait(data)
        return True

    async def flush(self):
        if self.connected:
            try:
                await asyncio.wait_for(
                    self._queue.join(), self.limits.write_timeout)
            except TimeoutError:
                pass

    async def close(self):
        self.connected = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def _check_peer(self, writer):
        if hasattr(socket, "SO_PEERCRED"):
            credentials = writer.get_extra_info("socket").getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED,
                struct.calcsize("iII"))
            _pid, uid, _gid = struct.unpack("iII", credentials)
            if uid != self.peer_uid:
                raise ProtocolError("unexpected bridge peer UID")

    async def _write(self, writer, data):
        writer.write(data)
        await asyncio.wait_for(writer.drain(), self.limits.write_timeout)

    async def _write_messages(self, writer):
        while True:
            data = await self._queue.get()
            try:
                await self._write(writer, data)
            finally:
                self._bytes -= len(data)
                self._queue.task_done()

    async def _read_messages(self, reader):
        while True:
            self.receive(await read_message(reader, self.limits))

    async def _connection(self):
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(
                self.path, limit=self.limits.frame_bytes),
            self.limits.connect_timeout)
        self._writer = writer
        tasks = []
        try:
            self._check_peer(writer)
            await self._write(writer, encode(
                {"type": "hello", "version": VERSION}, self.limits))
            hello = await asyncio.wait_for(
                read_message(reader, self.limits), self.limits.connect_timeout)
            if (hello != {"type": "hello", "version": VERSION}
                    or type(hello.get("version")) is not int):
                raise ProtocolError("unsupported bridge handshake/version")
            # New events queue while registration/replay are being written.
            # Snapshot creation is synchronous, so none fall between the two.
            self.connected = True
            for message in self.register():
                await self._write(writer, encode(message, self.limits))
            if self._warned:
                self.report("Bridge connection restored.")
                self._warned = False
            tasks = [asyncio.create_task(self._write_messages(writer)),
                     asyncio.create_task(self._read_messages(reader))]
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            self.connected = False
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
            self._bytes = 0
            writer.close()
            try:
                await asyncio.wait_for(
                    writer.wait_closed(), self.limits.write_timeout)
            except (OSError, TimeoutError):
                pass
            self._writer = None

    async def _run(self):
        delay = self.limits.retry_min
        while True:
            try:
                await self._connection()
            except (OSError, ValueError, ConnectionError, TimeoutError) as error:
                if not self._warned:
                    self.report(f"Bridge unavailable: {error}; retrying.")
                    self._warned = True
            await asyncio.sleep(delay * random.uniform(0.8, 1.2))
            delay = min(self.limits.retry_max, delay * 2)
