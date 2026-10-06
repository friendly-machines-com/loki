#!/usr/bin/env python3
"""
Zero-dependency, pure Python standard library asyncio standby inhibitor.
Prefers XDG Desktop Portal (Session Bus, sandbox-friendly) with automatic
fallback to systemd-logind/elogind (System Bus).

Specifications & Standards referenced:
1. XDG Desktop Portal - Inhibit API (Version 3):
   https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.Inhibit.html
   Method: org.freedesktop.portal.Inhibit.Inhibit(s window, u flags, a{sv} options) -> (o handle)
   Release: org.freedesktop.portal.Request.Close() on the returned handle.
2. Inhibitor Locks Specification (logind/elogind):
   https://systemd.io/INHIBITOR_LOCKS/
   man 5 org.freedesktop.login1
   Method: org.freedesktop.login1.Manager.Inhibit(s what, s who, s why, s mode) -> (h fd)
3. D-Bus Specification:
   https://dbus.freedesktop.org/doc/dbus-specification.html
"""

import asyncio
import collections
import os
import socket
import struct
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Tuple


class DbusError(RuntimeError):
    """Raised when D-Bus returns an ERROR message (type 3)."""
    def __init__(self, name: str, message: str):
        super().__init__(f"{name}: {message}")
        self.name = name
        self.message = message


class AsyncDBusWireClient:
    """Minimal wire-level D-Bus stream client over a Unix socket."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._recv_buf = bytearray()
        self._fd_queue: collections.deque[int] = collections.deque()
        self._serial = 1

    @classmethod
    async def connect(cls, socket_path: str) -> "AsyncDBusWireClient":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        sock.setblocking(False)

        client = cls(sock)
        await client._authenticate()
        await client._hello()
        return client

    async def _read_from_socket(self) -> None:
        loop = asyncio.get_running_loop()
        fd = self.sock.fileno()

        while True:
            try:
                chunk, ancdata, _, _ = self.sock.recvmsg(4096, 256)
                break
            except (BlockingIOError, InterruptedError):
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
                    self._fd_queue.append(struct.unpack("i", data[i: i + 4])[0])

        self._recv_buf.extend(chunk)

    async def _read_sasl_line(self) -> bytes:
        while b"\r\n" not in self._recv_buf:
            await self._read_from_socket()
        idx = self._recv_buf.index(b"\r\n")
        line = bytes(self._recv_buf[:idx])
        del self._recv_buf[: idx + 2]
        return line

    async def _authenticate(self) -> None:
        loop = asyncio.get_running_loop()
        uid_hex = str(os.getuid()).encode("ascii").hex().encode("ascii")

        await loop.sock_sendall(self.sock, b"\0AUTH EXTERNAL " + uid_hex + b"\r\n")
        line = await self._read_sasl_line()
        if not line.startswith(b"OK"):
            raise ConnectionError(f"D-Bus SASL failed: {line.decode(errors='replace')}")

        await loop.sock_sendall(self.sock, b"NEGOTIATE_UNIX_FD\r\n")
        line = await self._read_sasl_line()
        if not line.startswith(b"AGREE_UNIX_FD"):
            raise ConnectionError("D-Bus broker refused UNIX FD passing")

        await loop.sock_sendall(self.sock, b"BEGIN\r\n")

    @staticmethod
    def _encode_str(s: str) -> bytes:
        data = s.encode("utf-8")
        return struct.pack("<I", len(data)) + data + b"\x00"

    @classmethod
    def _encode_field(cls, code: int, sig: str, val: str) -> bytes:
        entry = bytearray([code, len(sig)]) + sig.encode("ascii") + b"\x00"
        if sig in ["s", "o"]:
            entry += b"\x00" * (-len(entry) % 4) + cls._encode_str(val)
        elif sig == "g":
            val_b = val.encode("ascii")
            entry += bytes([len(val_b)]) + val_b + b"\x00"
        return bytes(entry)

    def _build_header(self, dest: str, path: str, iface: str, member: str, sig: str, body_len: int) -> bytes:
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
            struct.pack("<BBBBIII", ord("l"), 1, 0, 1, body_len, serial, len(fields_data))
        ) + fields_data
        header += b"\x00" * (-len(header) % 8)
        return bytes(header)

    async def _recv_message(self) -> Tuple[int, int, bytes, bytes]:
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

        fields_raw = msg_raw[16: 16 + fields_len]
        body_raw = msg_raw[16 + header_fields_padded: total_len]

        return msg_type, serial, fields_raw, body_raw

    async def call_method(
        self,
        dest: str,
        path: str,
        iface: str,
        member: str,
        sig: str = "",
        body: bytes = b"",
    ) -> bytes:
        """Sends method call and waits for return, skipping asynchronous signals (type 4)."""
        loop = asyncio.get_running_loop()
        header = self._build_header(dest, path, iface, member, sig, len(body))
        await loop.sock_sendall(self.sock, header + body)

        while True:
            msg_type, _, _, body_raw = await self._recv_message()

            if msg_type == 4:  # Discard asynchronous signals (e.g. NameAcquired)
                continue

            if msg_type == 3:  # D-Bus ERROR
                err_msg = ""
                if len(body_raw) >= 4:
                    str_len = struct.unpack("<I", body_raw[:4])[0]
                    err_msg = body_raw[4: 4 + str_len].decode("utf-8", errors="replace")
                raise DbusError("D-Bus Error received", err_msg or repr(body_raw))

            if msg_type == 2:  # METHOD_RETURN
                return body_raw

            raise DbusError("UnexpectedMessageType", f"Expected METHOD_RETURN (2), got {msg_type}")

    async def _hello(self) -> None:
        await self.call_method(
            dest="org.freedesktop.DBus",
            path="/org/freedesktop/DBus",
            iface="org.freedesktop.DBus",
            member="Hello",
        )

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


# ---------------------------------------------------------------------------
# Backend 1: XDG Desktop Portal (Session Bus)
# ---------------------------------------------------------------------------
class PortalInhibitor:
    """
    Inhibits standby via org.freedesktop.portal.Inhibit on the session bus.
    Works inside Flatpak and graphical Wayland/X11 desktops.
    """
    def __init__(self, client: AsyncDBusWireClient):
        self.client = client
        self.request_handle: Optional[str] = None

    @classmethod
    async def try_connect(cls) -> Optional["PortalInhibitor"]:
        # Resolve session bus socket from environment or systemd user runtime
        sock_path = None
        addr = os.environ.get("DBUS_SESSION_BUS_ADDRESS")
        if addr:
            for part in addr.split(";"):
                if part.startswith("unix:"):
                    params = dict(kv.split("=", 1) for kv in part[5:].split(",") if "=" in kv)
                    if "path" in params:
                        sock_path = params["path"]
                    elif "abstract" in params:
                        sock_path = "\0" + params["abstract"]

        if not sock_path:
            uid = os.getuid()
            user_bus = f"/run/user/{uid}/bus"
            if os.path.exists(user_bus):
                sock_path = user_bus

        if not sock_path:
            return None

        try:
            client = await AsyncDBusWireClient.connect(sock_path)
            return cls(client)
        except (OSError, ConnectionError):
            return None

    async def inhibit(self, why: str) -> str:
        """
        Calls org.freedesktop.portal.Inhibit.Inhibit(s window, u flags, a{sv} options).
        Flags: 4 = Suspend, 8 = Idle -> 12 = Both (idle:sleep).
        Options: {'reason': Variant('s', why)}
        """
        # Pack options vardict: a{sv} containing {"reason": <"s", why>}
        # Struct {sv} starts 8-byte aligned at array start
        reason_bytes = AsyncDBusWireClient._encode_str("reason")  # 11 bytes
        var_sig = b"\x01s\x00"                                    # 3 bytes (11 + 3 = 14)
        pad = b"\x00" * (-14 % 4)                                 # 2 bytes pad to 4-byte boundary
        val_bytes = AsyncDBusWireClient._encode_str(why)
        dict_entry = reason_bytes + var_sig + pad + val_bytes

        # Body: string window ("") + uint32 flags (12) + array length + dict_entry
        body = AsyncDBusWireClient._encode_str("")                  # 5 bytes
        body += b"\x00" * (-len(body) % 4) + struct.pack("<I", 12)  # flags=12 (Suspend|Idle)
        body += b"\x00" * (-len(body) % 4) + struct.pack("<I", len(dict_entry)) + dict_entry

        reply_body = await self.client.call_method(
            dest="org.freedesktop.portal.Desktop",
            path="/org/freedesktop/portal/desktop",
            iface="org.freedesktop.portal.Inhibit",
            member="Inhibit",
            sig="sua{sv}",
            body=body,
        )

        # Reply returns object path 'o' of the request handle
        if len(reply_body) >= 4:
            path_len = struct.unpack("<I", reply_body[:4])[0]
            self.request_handle = reply_body[4: 4 + path_len].decode("utf-8")
            return self.request_handle

        raise RuntimeError("Portal returned invalid request handle")

    async def release(self) -> None:
        """Releases the lock by calling org.freedesktop.portal.Request.Close()."""
        if self.request_handle:
            try:
                await self.client.call_method(
                    dest="org.freedesktop.portal.Desktop",
                    path=self.request_handle,
                    iface="org.freedesktop.portal.Request",
                    member="Close",
                )
            except Exception:
                pass
            self.request_handle = None
        self.client.close()


# ---------------------------------------------------------------------------
# Backend 2: systemd-logind / elogind (System Bus)
# ---------------------------------------------------------------------------
class LogindInhibitor:
    """
    Inhibits standby via org.freedesktop.login1.Manager.Inhibit on the system bus.
    Used on headless setups, SSH sessions, or when portals are unavailable.
    """
    def __init__(self, client: AsyncDBusWireClient):
        self.client = client
        self.lock_fd: int = -1

    @classmethod
    async def connect(cls) -> "LogindInhibitor":
        path = (
            "/run/dbus/system_bus_socket"
            if os.path.exists("/run/dbus/system_bus_socket")
            else "/var/run/dbus/system_bus_socket"
        )
        client = await AsyncDBusWireClient.connect(path)
        return cls(client)

    async def inhibit(self, who: str, why: str, what: str = "idle:sleep", mode: str = "block") -> int:
        body = bytearray()
        for s in [what, who, why, mode]:
            body += b"\x00" * (-len(body) % 4) + AsyncDBusWireClient._encode_str(s)

        await self.client.call_method(
            dest="org.freedesktop.login1",
            path="/org/freedesktop/login1",
            iface="org.freedesktop.login1.Manager",
            member="Inhibit",
            sig="ssss",
            body=bytes(body),
        )

        if not self.client._fd_queue:
            raise RuntimeError("No file descriptor received via SCM_RIGHTS from logind")

        self.lock_fd = self.client._fd_queue.popleft()
        return self.lock_fd

    async def release(self) -> None:
        if self.lock_fd >= 0:
            try:
                os.close(self.lock_fd)
            except OSError:
                pass
            self.lock_fd = -1
        self.client.close()


# ---------------------------------------------------------------------------
# Public Unified Context Manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def prevent_standby(
    who: str = "loki_agent",
    why: str = "Task in progress",
    what: str = "idle:sleep",
    mode: str = "block",
) -> AsyncIterator[str]:
    """
    Acquires an inhibitor lock. Prefers XDG Desktop Portal (Session Bus).
    Falls back automatically to systemd-logind/elogind (System Bus).

    Yields a string describing the active lock token or file descriptor.
    """
    inhibitor = None
    lock_info = ""

    # 1. Prefer XDG Desktop Portal (Flatpak / Desktop Session)
    portal = await PortalInhibitor.try_connect()
    if portal:
        try:
            handle = await portal.inhibit(why=why)
            inhibitor = portal
            lock_info = f"XDG Desktop Portal (handle: {handle})"
        except Exception:
            portal.client.close()
            inhibitor = None

    # 2. Fallback to systemd-logind / elogind (System Bus)
    if inhibitor is None:
        logind = await LogindInhibitor.connect()
        fd = await logind.inhibit(who=who, why=why, what=what, mode=mode)
        inhibitor = logind
        lock_info = f"logind/elogind (fd: {fd})"

    try:
        yield lock_info
    finally:
        await inhibitor.release()


# ---------------------------------------------------------------------------
# Verification Entrypoint
# ---------------------------------------------------------------------------
async def main() -> None:
    print("Selecting backend and acquiring standby inhibition...")
    async with prevent_standby(who="loki_agent", why="Running test harness") as info:
        print(f"Lock active using: {info}")
        print("Holding for 20 seconds. Standby and idle are inhibited.")
        await asyncio.sleep(20)

    print("Released cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
