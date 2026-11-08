"""Shared subprocess stream framing, buffering, and flow control."""

import asyncio
import unittest
from unittest import mock

from loki_agent import streams


class ProcessProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_protocol_framing_retains_backpressure_limits_and_eof_tail(self):
        stdout, stderr = mock.Mock(), mock.Mock()
        stdout.is_closing.return_value = stderr.is_closing.return_value = False
        transport = mock.Mock()
        transport.get_pipe_transport.side_effect = lambda fd: {0: None, 1: stdout, 2: stderr}[fd]
        protocol = streams.ProcessProtocol(limit=3)
        protocol.connection_made(transport)
        pending = asyncio.create_task(protocol.stdout.readline())
        try:
            await asyncio.sleep(0)
            protocol.pipe_data_received(1, b'abc\nZ')
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            stdout.pause_reading.assert_called_once_with()
            stdout.resume_reading.assert_not_called()
            self.assertEqual(await protocol.stdout.readline(), b'abc\n')
            stdout.resume_reading.assert_called_once_with()
            protocol.pipe_connection_lost(1, None)
            self.assertEqual(await protocol.stdout.readline(), b'Z')
            protocol.pipe_data_received(2, b'abcd\n')
            with self.assertRaisesRegex(ValueError, 'size limit'):
                await protocol.stderr.readline()
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            protocol.connection_lost(None)
