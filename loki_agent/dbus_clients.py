#!/usr/bin/env python3
"""
Zero-dependency, pure Python standard library asyncio D-Bus client for inhibiting
system standby/sleep via org.freedesktop.login1 (systemd-logind / elogind).

Specifications & Standards referenced:
1. Inhibitor Locks Specification:
   https://www.freedesktop.org/wiki/Software/systemd/inhibit/
   https://systemd.io/INHIBITOR_LOCKS/
   man 5 org.freedesktop.login1
2. D-Bus Specification:
   https://dbus.freedesktop.org/doc/dbus-specification.html
3. POSIX.1g recvmsg / SCM_RIGHTS file descriptor passing:
   https://man7.org/linux/man-pages/man7/unix.7.html
"""

import asyncio
import collections
import os
import socket
import struct
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Tuple


class DbusError(RuntimeError):
    """
    Raised when the bus daemon or target service returns an ERROR message (type 3).
    Ref: D-Bus Spec paragraph "Message Protocol: Message Types"
    """
    def __init__(self, name: str, message: str):
        super().__init__(f"{name}: {message}")
        self.name = name
        self.message = message


class AsyncDbusClient:
    """
    Non-blocking, event-loop-native D-Bus client over a Unix domain stream socket.
    Ref: D-Bus Spec paragraph "Transports: UNIX domain sockets"
    """

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._recv_buf = bytearray()
        self._fd_queue: collections.deque[int] = collections.deque()
        self._serial = 1

    @classmethod
    async def connect(cls, socket_path: Optional[str] = None) -> "AsyncDbusClient":
        """
        Connects to the system bus socket and initializes authentication and Hello.
        """
        if socket_path is None:
            socket_path = (
                "/run/dbus/system_bus_socket"
                if os.path.exists("/run/dbus/system_bus_socket")
                else "/var/run/dbus/system_bus_socket"
            )

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        sock.setblocking(False)  # Native non-blocking integration with asyncio loop

        client = cls(sock)
        await client._authenticate()
        await client._hello()
        return client

    async def _read_from_socket(self) -> None:
        """
        Asynchronously awaits socket readability via loop.add_reader and pulls
        incoming chunks and SCM_RIGHTS file descriptors.
        Ref: POSIX recvmsg(2), Linux unix(7) SCM_RIGHTS.
        """
        loop = asyncio.get_running_loop()
        fd = self.sock.fileno()

        while True:
            try:
                # 4096 buffer for payload, 256 for ancillary SCM_RIGHTS data
                chunk, ancdata, _, _ = self.sock.recvmsg(4096, 256)
                break
            except (BlockingIOError, InterruptedError):
                # Socket not ready yet; yield to event loop until readable
                fut = loop.create_future()

                def _on_readable() -> None:
                    if not fut.done():
                        fut.set_result(None)

                loop.add_reader(fd, _on_readable)
                try:
                    await fut
                finally:
                    loop.remove_reader(fd)

        if not chunk:
            raise ConnectionResetError("D-Bus socket closed by peer")

        for level, ctype, data in ancdata:
            if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                for i in range(0, len(data), 4):
                    passed_fd = struct.unpack("i", data[i : i + 4])[0]
                    self._fd_queue.append(passed_fd)

        self._recv_buf.extend(chunk)

    async def _read_sasl_line(self) -> bytes:
        """
        Reads a single CRLF-terminated line for the SASL profile without losing buffered bytes.
        Ref: D-Bus Spec paragraph "Authentication Handshake: Protocol"
        """
        while b"\r\n" not in self._recv_buf:
            await self._read_from_socket()
        idx = self._recv_buf.index(b"\r\n")
        line = bytes(self._recv_buf[:idx])
        del self._recv_buf[: idx + 2]
        return line

    async def _authenticate(self) -> None:
        """
        Executes SASL EXTERNAL auth and negotiates Unix FD passing asynchronously.
        Ref: D-Bus Spec paragraph "Authentication Handshake: The EXTERNAL mechanism"
        Ref: D-Bus Spec paragraph "Authentication Handshake: Command NEGOTIATE_UNIX_FD"
        Ref: D-Bus Spec paragraph "Authentication Handshake: Command BEGIN"
        """
        loop = asyncio.get_running_loop()
        uid_hex = str(os.getuid()).encode("ascii").hex().encode("ascii")

        await loop.sock_sendall(self.sock, b"\0AUTH EXTERNAL " + uid_hex + b"\r\n")
        line = await self._read_sasl_line()
        if not line.startswith(b"OK"):
            raise ConnectionError(f"D-Bus SASL EXTERNAL failed: {line.decode(errors='replace')}")

        await loop.sock_sendall(self.sock, b"NEGOTIATE_UNIX_FD\r\n")
        line = await self._read_sasl_line()
        if not line.startswith(b"AGREE_UNIX_FD"):
            raise ConnectionError("D-Bus broker refused UNIX FD passing (NEGOTIATE_UNIX_FD)")

        await loop.sock_sendall(self.sock, b"BEGIN\r\n")

    @staticmethod
    def _encode_str(s: str) -> bytes:
        """
        Encodes a D-Bus STRING (type 's') or OBJECT_PATH (type 'o').
        Ref: D-Bus Spec paragraph "Marshalling: Basic Types"
        """
        data = s.encode("utf-8")
        return struct.pack("<I", len(data)) + data + b"\x00"

    @classmethod
    def _encode_field(cls, code: int, sig: str, val: str) -> bytes:
        """
        Encodes an entry in the header fields array: struct { BYTE, VARIANT } -> (yv).
        Ref: D-Bus Spec paragraph "Header fields"
        """
        entry = bytearray([code, len(sig)]) + sig.encode("ascii") + b"\x00"
        if sig in ("s", "o"):
            entry += b"\x00" * (-len(entry) % 4) + cls._encode_str(val)
        elif sig == "g":
            val_b = val.encode("ascii")
            entry += bytes([len(val_b)]) + val_b + b"\x00"
        return bytes(entry)

    def _build_call_message(
        self,
        dest: str,
        path: str,
        iface: str,
        member: str,
        sig: str = "",
        args: Optional[list[str]] = None,
    ) -> Tuple[int, bytes]:
        """
        Assembles a binary D-Bus METHOD_CALL message.
        Ref: D-Bus Spec paragraph "Message Protocol: Message Format"
        """
        body = bytearray()
        if args:
            for s in args:
                body += b"\x00" * (-len(body) % 4) + self._encode_str(s)

        fields = [
            (1, "o", path),
            (2, "s", iface),
            (3, "s", member),
            (6, "s", dest),
        ]
        if sig:
            fields.append((8, "g", sig))

        fields_data = bytearray()
        for code, s_type, val in fields:
            fields_data += b"\x00" * (-len(fields_data) % 8) + self._encode_field(code, s_type, val)

        serial = self._serial
        self._serial += 1

        header = bytearray(
            struct.pack("<BBBBIII", ord("l"), 1, 0, 1, len(body), serial, len(fields_data))
        ) + fields_data
        header += b"\x00" * (-len(header) % 8)

        return serial, bytes(header + body)

    async def _recv_message(self) -> Tuple[int, int, bytes, bytes]:
        """
        Reads one full D-Bus message frame from the stream asynchronously.
        Ref: D-Bus Spec paragraph "Message Protocol: Message Format"
        """
        while len(self._recv_buf) < 16:
            await self._read_from_socket()

        endian = self._recv_buf[0]
        endian_fmt = "<" if endian == ord("l") else ">"

        msg_type = self._recv_buf[1]
        body_len, serial, fields_len = struct.unpack(
            f"{endian_fmt}III", self._recv_buf[4:16]
        )

        header_fields_padded = fields_len + (-fields_len % 8)
        total_len = 16 + header_fields_padded + body_len

        while len(self._recv_buf) < total_len:
            await self._read_from_socket()

        msg_raw = self._recv_buf[:total_len]
        del self._recv_buf[:total_len]

        fields_raw = msg_raw[16 : 16 + fields_len]
        body_raw = msg_raw[16 + header_fields_padded : total_len]

        return msg_type, serial, fields_raw, body_raw

    async def _call_and_wait_reply(
        self,
        dest: str,
        path: str,
        iface: str,
        member: str,
        sig: str = "",
        args: Optional[list[str]] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Sends a METHOD_CALL and awaits matching METHOD_RETURN (2) or ERROR (3).
        Silently discards asynchronous signals (type 4, like NameAcquired).
        Ref: D-Bus Spec paragraph "Message Protocol: Message Types"
        """
        loop = asyncio.get_running_loop()
        serial, msg = self._build_call_message(dest, path, iface, member, sig, args)
        await loop.sock_sendall(self.sock, msg)

        while True:
            msg_type, reply_serial, fields_raw, body_raw = await self._recv_message()

            if msg_type == 4:  # DBUS_MESSAGE_TYPE_SIGNAL
                continue

            if msg_type == 3:  # DBUS_MESSAGE_TYPE_ERROR
                err_msg = ""
                if len(body_raw) >= 4:
                    str_len = struct.unpack("<I", body_raw[:4])[0]
                    err_msg = body_raw[4 : 4 + str_len].decode("utf-8", errors="replace")
                raise DbusError("D-Bus Error received", err_msg or repr(body_raw))

            if msg_type == 2:  # DBUS_MESSAGE_TYPE_METHOD_RETURN
                return fields_raw, body_raw

            raise DbusError("UnexpectedMessageType", f"Expected METHOD_RETURN (2), got {msg_type}")

    async def _hello(self) -> None:
        """
        Registers connection on the message bus.
        Ref: D-Bus Spec paragraph "Message Bus Starting Services: org.freedesktop.DBus.Hello"
        """
        await self._call_and_wait_reply(
            dest="org.freedesktop.DBus",
            path="/org/freedesktop/DBus",
            iface="org.freedesktop.DBus",
            member="Hello",
        )

    async def call_inhibit(self, what: str, who: str, why: str, mode: str) -> int:
        """
        Invokes org.freedesktop.login1.Manager.Inhibit asynchronously to acquire the lock.
        Ref: Freedesktop.org Inhibitor Locks Spec
             Interface: org.freedesktop.login1.Manager
             Method: Inhibit(s what, s who, s why, s mode) -> (h fd)
        """
        await self._call_and_wait_reply(
            dest="org.freedesktop.login1",
            path="/org/freedesktop/login1",
            iface="org.freedesktop.login1.Manager",
            member="Inhibit",
            sig="ssss",
            args=[what, who, why, mode],
        )

        if not self._fd_queue:
            raise RuntimeError(
                "D-Bus returned METHOD_RETURN but no file descriptor was received via SCM_RIGHTS."
            )

        return self._fd_queue.popleft()

    def close(self) -> None:
        while self._fd_queue:
            try:
                os.close(self._fd_queue.popleft())
            except OSError:
                pass
        try:
            self.sock.close()
        except OSError:
            pass


@asynccontextmanager
async def prevent_standby(
    who: str = "loki_agent",
    why: str = "Task in progress",
    what: str = "idle:sleep",
    mode: str = "block",
) -> AsyncIterator[int]:
    """
    Async context manager that acquires and holds an inhibitor lock.
    Ref: https://systemd.io/INHIBITOR_LOCKS/
    Yields the integer file descriptor. Releasing happens on exit.
    """
    client = await AsyncDbusClient.connect()
    lock_fd = -1
    try:
        lock_fd = await client.call_inhibit(what=what, who=who, why=why, mode=mode)
        yield lock_fd
    finally:
        if lock_fd >= 0:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        client.close()


async def main() -> None:
    print("Connecting to system bus and acquiring inhibitor lock (async)...")
    async with prevent_standby(who="loki_agent", why="Executing async pipeline") as fd:
        print(f"Lock active! Acquired file descriptor: {fd}")
        print("Run 'elogind-inhibit --list' or 'systemd-inhibit --list' in another terminal to verify.")
        await asyncio.sleep(20)

    print("Lock released cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
