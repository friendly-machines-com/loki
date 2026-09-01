"""Async credential capabilities over anonymous Unix socket pairs.

A delegated process receives only its end of a fresh socket pair.  Nested
subagents never inherit that upstream descriptor: their parent serves a new
socket through a restricted relay.  This makes delegation recursive without
requiring nonportable descriptor passing or a pathname-addressed service.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket

from .authentications import (
    CredentialAuthority,
    CredentialError,
    CredentialLease,
    CredentialRef,
    CredentialUnavailable,
    RefreshIndeterminateError,
    RefreshPermanentError,
    RefreshTransientError,
)


CAPABILITY_MAX_MESSAGE_BYTES = 64 * 1024


class CapabilityError(CredentialError):
    pass


def _endpoint_from_fd(fd: int) -> socket.socket:
    """Take synchronous ownership of an inherited capability descriptor.

    This is deliberately not async. Delegated-process startup races capability
    initialization against owner revocation; ownership must transfer before
    that race can cancel the initialization task, or a task canceled before
    its first instruction could leave the inherited credential FD open.
    """
    try:
        os.set_inheritable(fd, False)
        endpoint = socket.socket(fileno=fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    endpoint.setblocking(False)
    return endpoint


async def _streams_from_endpoint(endpoint: socket.socket):
    try:
        return await asyncio.open_connection(
            sock=endpoint,
            limit=CAPABILITY_MAX_MESSAGE_BYTES,
        )
    except BaseException:
        endpoint.close()
        raise


def _encode(message: dict) -> bytes:
    data = json.dumps(
        message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(data) + 1 > CAPABILITY_MAX_MESSAGE_BYTES:
        raise CapabilityError("credential capability message is too large")
    return data + b"\n"


def _decode(raw: bytes) -> dict:
    if len(raw) > CAPABILITY_MAX_MESSAGE_BYTES:
        raise CapabilityError("credential capability message is too large")
    try:
        message = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CapabilityError(
            "credential capability emitted invalid JSON") from error
    if not isinstance(message, dict):
        raise CapabilityError(
            "credential capability message must be an object")
    return message


def _safe_error_text(error: Exception) -> str:
    """Return capability diagnostics that cannot contain credential values."""
    if isinstance(error, CapabilityError):
        return str(error)
    if isinstance(error, CredentialUnavailable):
        return str(error)
    if isinstance(error, RefreshIndeterminateError):
        return (
            "credential refresh state is indeterminate; log in again")
    if isinstance(error, RefreshPermanentError):
        return "credential refresh failed permanently; log in again"
    if isinstance(error, RefreshTransientError):
        return "credential refresh failed transiently"
    return "credential request failed"


class _CredentialClientInitializer:
    """Own a capability socket until its async transport takes ownership."""

    def __init__(self, client_class, endpoint: socket.socket):
        self._client_class = client_class
        self._endpoint = endpoint

    async def connect(self):
        endpoint = self._endpoint
        try:
            reader, writer = await _streams_from_endpoint(endpoint)
        finally:
            # open_connection() either transferred ownership to its transport
            # or _streams_from_endpoint() closed the socket on failure.
            self._endpoint = None
        return await self._client_class._from_streams(reader, writer)

    def close_now(self):
        endpoint = self._endpoint
        self._endpoint = None
        if endpoint is not None:
            endpoint.close()


class CredentialClient:
    """Multiplex async credential requests over one delegated capability."""

    def __init__(self, reader: asyncio.StreamReader,
                 writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._closed = False
        self._available: frozenset[CredentialRef] = frozenset()
        self._reader_task = asyncio.create_task(
            self._read_messages(), name="credential-capability-client")

    @classmethod
    def from_fd(cls, fd: int):
        """Return an initialization awaitable after taking ownership of FD."""
        initializer = cls.take_fd(fd)

        async def connect():
            try:
                return await initializer.connect()
            finally:
                initializer.close_now()

        return connect()

    @classmethod
    def take_fd(cls, fd: int) -> _CredentialClientInitializer:
        """Take FD now so a later task cancellation cannot leak it."""
        return _CredentialClientInitializer(
            cls, _endpoint_from_fd(fd))

    @classmethod
    async def _from_streams(
            cls, reader, writer) -> "CredentialClient":
        client = cls(reader, writer)
        try:
            raw_refs = await client._request("describe", {})
            if not isinstance(raw_refs, list):
                raise CapabilityError(
                    "credential capability describe result must be an array")
            client._available = frozenset(
                CredentialRef.decode(value) for value in raw_refs)
            return client
        except BaseException as error:
            await client.close()
            if isinstance(error, ValueError):
                raise CapabilityError(
                    "credential capability returned an invalid reference"
                ) from error
            raise

    def available(self) -> frozenset[CredentialRef]:
        return self._available

    async def lease(
            self, credential: CredentialRef,
            rejected_generation: int | None = None) -> CredentialLease:
        result = await self._request("lease", {
            "credential": credential.encode(),
            "rejected_generation": rejected_generation,
        })
        try:
            lease = CredentialLease.from_wire(result)
        except ValueError as error:
            raise CapabilityError(
                "credential capability returned an invalid lease") from error
        if lease.credential != credential:
            raise CapabilityError(
                "credential capability returned the wrong credential")
        return lease

    async def _request(self, method: str, params: dict):
        if self._closed:
            raise CapabilityError("credential capability is closed")
        self._next_id += 1
        request_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            encoded = _encode({
                "id": request_id,
                "method": method,
                "params": params,
            })
            async with self._write_lock:
                if self._closed:
                    raise CapabilityError(
                        "credential capability is closed")
                self._writer.write(encoded)
                await self._writer.drain()
            return await future
        except (BrokenPipeError, ConnectionError, OSError) as error:
            if not future.done():
                future.cancel()
            else:
                with contextlib.suppress(
                        asyncio.CancelledError, CapabilityError):
                    future.exception()
            raise CapabilityError(
                "credential capability is closed") from error
        finally:
            self._pending.pop(request_id, None)

    async def _read_messages(self):
        failure = None
        try:
            while True:
                raw = await self._reader.readline()
                if not raw:
                    failure = CapabilityError(
                        "credential capability closed by its owner")
                    break
                message = _decode(raw)
                request_id = message.get("id")
                if not isinstance(request_id, int):
                    raise CapabilityError(
                        "credential capability response has no integer id")
                future = self._pending.get(request_id)
                if future is None or future.done():
                    # A caller may be cancelled while its request completes.
                    continue
                error = message.get("error")
                if error is not None:
                    future.set_exception(CapabilityError(str(error)))
                elif "result" in message:
                    future.set_result(message["result"])
                else:
                    future.set_exception(CapabilityError(
                        "credential capability response has no result"))
        except asyncio.CancelledError:
            failure = CapabilityError("credential capability closed")
            raise
        except Exception as error:
            failure = (
                error if isinstance(error, CapabilityError)
                else CapabilityError(
                    f"credential capability failed: {error}"))
        finally:
            self._closed = True
            failure = failure or CapabilityError(
                "credential capability closed")
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(failure)

    async def close(self):
        self.close_now()
        with contextlib.suppress(
                BrokenPipeError, ConnectionError, OSError):
            await self._writer.wait_closed()
        await asyncio.gather(self._reader_task, return_exceptions=True)

    async def wait_closed(self):
        """Wait for owner revocation without canceling the shared reader."""
        await asyncio.shield(self._reader_task)

    def close_now(self):
        """Revoke the capability synchronously during process state changes."""
        self._closed = True
        self._writer.close()
        if not self._reader_task.done():
            self._reader_task.cancel()


class CredentialCapabilityServer:
    """Serve a fixed subset of an authority through one socket endpoint."""

    def __init__(
            self, authority: CredentialAuthority,
            allowed: frozenset[CredentialRef],
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter):
        if not allowed.issubset(authority.available()):
            raise ValueError(
                "delegated credentials exceed the upstream authority")
        self.authority = authority
        self.allowed = allowed
        self._reader = reader
        self._writer = writer
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._requests: set[asyncio.Task] = set()
        self._closed = False
        self._reader_task = asyncio.create_task(
            self._read_messages(), name="credential-capability-server")

    @classmethod
    async def create(
            cls, authority: CredentialAuthority,
            allowed=None
    ) -> tuple["CredentialCapabilityServer", int]:
        permitted = frozenset(
            authority.available() if allowed is None else allowed)
        if not permitted.issubset(authority.available()):
            raise ValueError(
                "delegated credentials exceed the upstream authority")
        parent_socket, child_socket = socket.socketpair()
        parent_socket.set_inheritable(False)
        child_socket.set_inheritable(False)
        parent_socket.setblocking(False)
        try:
            reader, writer = await asyncio.open_connection(
                sock=parent_socket,
                limit=CAPABILITY_MAX_MESSAGE_BYTES,
            )
            server = cls(authority, permitted, reader, writer)
            return server, child_socket.detach()
        except BaseException:
            parent_socket.close()
            child_socket.close()
            raise

    async def _read_messages(self):
        try:
            while True:
                raw = await self._reader.readline()
                if not raw:
                    break
                message = _decode(raw)
                task = asyncio.create_task(
                    self._answer(message),
                    name="credential-capability-request",
                )
                self._requests.add(task)
                task.add_done_callback(self._requests.discard)
        except (CapabilityError, ConnectionError, OSError, ValueError):
            pass
        finally:
            await self._close_writer()
            requests = list(self._requests)
            for task in requests:
                task.cancel()
            if requests:
                await asyncio.gather(*requests, return_exceptions=True)

    async def _answer(self, message: dict):
        request_id = message.get("id")
        if not isinstance(request_id, int):
            return
        try:
            result = await self._dispatch(
                message.get("method"), message.get("params"))
            response = {"id": request_id, "result": result}
        except Exception as error:
            # The authority may wrap a third-party response or future storage
            # backend. Never trust its exception text across the boundary:
            # arbitrary diagnostics can contain a refresh token.
            response = {
                "id": request_id,
                "error": _safe_error_text(error),
            }
        try:
            encoded = _encode(response)
            async with self._write_lock:
                if self._closed:
                    return
                self._writer.write(encoded)
                await self._writer.drain()
        except (BrokenPipeError, ConnectionError, OSError):
            pass

    async def _dispatch(self, method, params):
        if method == "describe":
            return [
                credential.encode()
                for credential in sorted(self.allowed)
            ]
        if method != "lease" or not isinstance(params, dict):
            raise CapabilityError("unsupported credential capability request")
        credential = CredentialRef.decode(params.get("credential"))
        if credential not in self.allowed:
            raise CapabilityError(
                f"credential {credential.encode()!r} is not delegated")
        rejected = params.get("rejected_generation")
        if (rejected is not None
                and (not isinstance(rejected, int)
                     or isinstance(rejected, bool))):
            raise CapabilityError(
                "rejected credential generation must be an integer or null")
        lease = await self.authority.lease(
            credential, rejected_generation=rejected)
        return lease.to_wire()

    async def _close_writer(self):
        async with self._close_lock:
            self._closed = True
            self._writer.close()
            with contextlib.suppress(
                    BrokenPipeError, ConnectionError, OSError):
                await self._writer.wait_closed()

    async def close(self):
        self.close_now()
        await asyncio.gather(self._reader_task, return_exceptions=True)
        requests = list(self._requests)
        for task in requests:
            task.cancel()
        if requests:
            await asyncio.gather(*requests, return_exceptions=True)
        await self._close_writer()

    def close_now(self):
        """Revoke a child capability without awaiting task cleanup."""
        if not self._closed:
            self._closed = True
            self._writer.close()
        if not self._reader_task.done():
            self._reader_task.cancel()
        for task in list(self._requests):
            task.cancel()


class CredentialRelay(CredentialCapabilityServer):
    """A freshly restricted child capability backed by an upstream client.

    The wire protocol is identical to a root broker's capability server. The
    distinct name documents the recursive trust boundary at nested-subagent
    call sites: a child receives a new socket and cannot inherit or widen its
    parent's upstream capability.
    """
