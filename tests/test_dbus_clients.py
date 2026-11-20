import os
import unittest
from unittest import mock

from loki_agent.dbus_clients import AsyncDBusWireClient, PortalInhibitor


class SessionBusAddressTests(unittest.IsolatedAsyncioTestCase):
    async def test_escaped_path_reaches_connect_as_original_bytes(self):
        address = ("unix:path=/tmp/link/../bus%20%2c%3B%3D%252F%ff,"
                   "guid=0123456789abcdef0123456789abcdef")
        with (mock.patch.dict(os.environ, {"DBUS_SESSION_BUS_ADDRESS": address}),
              mock.patch.object(AsyncDBusWireClient, "connect") as connect):
            await PortalInhibitor.try_connect()
        connect.assert_awaited_once_with(b"/tmp/link/../bus ,;=%2F\xff")

    @unittest.skipUnless(hasattr(os, "getuid"), "Unix user-bus fallback")
    async def test_invalid_path_uses_user_bus_fallback(self):
        for value in ["%00bus", "/tmp/bus%00suffix", "/tmp/bus%a", "/tmp/bus%gg", "/tmp/bus with space"]:
            with (self.subTest(value=value),
                  mock.patch.dict(os.environ, {"DBUS_SESSION_BUS_ADDRESS": f"unix:path={value}"}),
                  mock.patch.object(os, "getuid", return_value=42),
                  mock.patch.object(os.path, "exists", return_value=True),
                  mock.patch.object(AsyncDBusWireClient, "connect") as connect):
                await PortalInhibitor.try_connect()
                connect.assert_awaited_once_with("/run/user/42/bus")
