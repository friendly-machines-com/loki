"""Tests for the IPC seam's POSIX contract and its Windows endpoint types.

The Windows channel is now two anonymous pipes (AF_UNIX is unreachable through
CPython), and its endpoint/encoder/stream classes depend only on import-safe
declarations, so they are exercised here on Linux with the Windows calls
mocked.  The POSIX half runs for real: a connected pair, a channel whose close
is EOF, and the reference/spawn/validation contract the child depends on.
"""

import asyncio
import os
import socket
import unittest
from unittest import mock

from loki_agent import host_ipc


class PairConfirmationTests(unittest.TestCase):
    """The ends handed over must be each other's.

    POSIX ``socketpair()`` cannot be substituted, so these run the same check
    the Windows hand-built pair depends on -- on the only host where it can be
    exercised portably.
    """

    def pair(self):
        first, second = socket.socketpair()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        return first, second

    def test_a_connected_pair_confirms_and_stays_usable(self):
        first, second = self.pair()

        host_ipc.confirm_pair(first, second)

        second.sendall(b"after")
        self.assertEqual(first.recv(5), b"after")

    def test_ends_from_two_different_pairs_are_refused_and_closed(self):
        first, _first_peer = self.pair()
        second, _second_peer = self.pair()

        with mock.patch.object(host_ipc, "_PAIR_CONFIRM_TIMEOUT", 0.05):
            with self.assertRaises(host_ipc.PairConfirmationError):
                host_ipc.confirm_pair(first, second)

        self.assertEqual(first.fileno(), -1)
        self.assertEqual(second.fileno(), -1)


class PrivatePipePairTests(unittest.TestCase):
    """Portable contract of the Windows pipe pair, with the calls mocked.

    The primitive is Windows-only in effect but depends only on declarations,
    so the marshalling, the parent-end inheritance clearing and both cleanup
    paths are checked here; the pipe handles themselves need Windows.  This is
    the transport that replaces the unreachable AF_UNIX emulation.
    """

    def test_only_the_child_ends_stay_inheritable(self):
        created = iter([(0x10, 0x11), (0x12, 0x13)])
        cleared = []
        with mock.patch.object(host_ipc.windows_api, "create_pipe",
                               side_effect=lambda: next(created)), \
                mock.patch.object(host_ipc.windows_api,
                                  "clear_handle_inheritance",
                                  side_effect=cleared.append), \
                mock.patch.object(host_ipc.windows_api, "close_handle"):
            request, response = host_ipc._private_pipe_pair()

        # request is (parent_read, child_write); response is
        # (child_read, parent_write).
        self.assertEqual(request, (0x10, 0x11))
        self.assertEqual(response, (0x12, 0x13))
        # Parent request-read 0x10 and parent response-write 0x13 are cleared;
        # child request-write 0x11 and child response-read 0x12 keep inherit.
        self.assertEqual(cleared, [0x10, 0x13])

    def test_second_pipe_failure_closes_the_first_pipe(self):
        closed = []
        with mock.patch.object(
                host_ipc.windows_api, "create_pipe",
                side_effect=[(0x10, 0x11), OSError("no more handles")]), \
                mock.patch.object(host_ipc.windows_api,
                                  "clear_handle_inheritance"), \
                mock.patch.object(host_ipc.windows_api, "close_handle",
                                  side_effect=closed.append):
            with self.assertRaises(OSError):
                host_ipc._private_pipe_pair()
        self.assertEqual(closed, [0x10, 0x11])

    def test_inheritance_failure_closes_all_four_ends(self):
        closed = []
        with mock.patch.object(
                host_ipc.windows_api, "create_pipe",
                side_effect=[(0x10, 0x11), (0x12, 0x13)]), \
                mock.patch.object(
                    host_ipc.windows_api, "clear_handle_inheritance",
                    side_effect=[None, OSError("cannot clear")]), \
                mock.patch.object(host_ipc.windows_api, "close_handle",
                                  side_effect=closed.append):
            with self.assertRaises(OSError):
                host_ipc._private_pipe_pair()
        self.assertEqual(closed, [0x10, 0x11, 0x12, 0x13])


@unittest.skipUnless(os.name == "posix", "POSIX seam")
class SocketPairTests(unittest.TestCase):
    def test_the_pair_is_connected_in_both_directions(self):
        first, second = host_ipc.socket_pair()
        self.addCleanup(first.close)
        self.addCleanup(second.close)

        first.sendall(b"ping")
        self.assertEqual(second.recv(4), b"ping")
        second.sendall(b"pong")
        self.assertEqual(first.recv(4), b"pong")


@unittest.skipUnless(os.name == "posix", "POSIX seam")
class OwnerChannelTests(unittest.TestCase):
    def test_closing_the_parent_end_is_eof_for_the_child(self):
        parent_end, child_end = host_ipc.owner_channel()
        self.addCleanup(os.close, child_end)

        host_ipc.close_end(parent_end)

        self.assertEqual(os.read(child_end, 1), b"")


@unittest.skipUnless(os.name == "posix", "POSIX seam")
class ReferenceTests(unittest.TestCase):
    def test_a_reference_names_the_child_end(self):
        reader, writer = os.pipe()
        self.addCleanup(os.close, reader)
        self.addCleanup(os.close, writer)

        value = host_ipc.reference(reader)

        self.assertIsInstance(value, int)
        self.assertEqual(host_ipc.child_endpoint(str(value)), reader)

    def test_spawn_kwargs_hand_the_references_over(self):
        self.assertEqual(host_ipc.spawn_kwargs((4, 7)), {"pass_fds": (4, 7)})

    def test_a_descriptor_that_was_not_inherited_is_rejected(self):
        with self.assertRaises(ValueError):
            host_ipc.child_endpoint("1")


class PipeEndpointTests(unittest.TestCase):
    """The Windows endpoint is the only thing that crosses to a child."""

    def test_reference_round_trips_both_directions(self):
        endpoint = host_ipc.PipeEndpoint(read=0x11, write=0x22)
        self.assertEqual(endpoint.handles(), (0x11, 0x22))
        self.assertEqual(endpoint.reference(), "r=17,w=34")
        rebuilt = host_ipc.PipeEndpoint.parse("r=17,w=34")
        self.assertEqual(rebuilt.handles(), (0x11, 0x22))

    def test_a_unidirectional_endpoint_round_trips(self):
        endpoint = host_ipc.PipeEndpoint(read=0x5)
        self.assertEqual(endpoint.handles(), (0x5,))
        self.assertEqual(host_ipc.PipeEndpoint.parse("r=5").read, 0x5)

    def test_malformed_references_are_refused(self):
        for value in ("", "5", "r=", "r=0", "x=5", "r=1,r=2", "r=1,w=2,z=3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    host_ipc.PipeEndpoint.parse(value)

    def test_close_releases_every_handle(self):
        closed = []
        with mock.patch.object(host_ipc.windows_api, "close_handle",
                               side_effect=closed.append):
            host_ipc.PipeEndpoint(read=1, write=2).close()
        self.assertEqual(closed, [1, 2])

    def test_is_endpoint_distinguishes_windows_endpoints(self):
        self.assertTrue(host_ipc.is_endpoint(host_ipc.PipeEndpoint(read=1)))
        self.assertFalse(host_ipc.is_endpoint(3))
        self.assertFalse(host_ipc.is_endpoint(None))

    @unittest.skipUnless(os.name == "posix", "POSIX descriptor")
    def test_a_descriptor_reports_one_handle(self):
        # POSIX path: spawn_kwargs flattens this to pass_fds.
        self.assertEqual(host_ipc.handles(4), (4,))


class HandleWriterTests(unittest.IsolatedAsyncioTestCase):
    """The writer thread keeps a blocking WriteFile off the event loop."""

    async def test_writes_are_flushed_in_order(self):
        written = []

        def write_file(handle, data):
            written.append((handle, bytes(data)))
            return len(data)

        with mock.patch.object(host_ipc.windows_api, "write_file",
                               side_effect=write_file):
            writer = host_ipc._HandleWriter(0xAA, asyncio.get_running_loop())
            writer.start()
            try:
                writer.write(b"one")
                writer.write(b"two")
                await writer.drain()
            finally:
                writer.stop_and_join()

        self.assertEqual(written, [(0xAA, b"one"), (0xAA, b"two")])

    async def test_short_writes_are_completed(self):
        written = bytearray()

        def write_file(handle, data):
            piece = bytes(data)[:2]
            written.extend(piece)
            return len(piece)

        with mock.patch.object(host_ipc.windows_api, "write_file",
                               side_effect=write_file):
            writer = host_ipc._HandleWriter(1, asyncio.get_running_loop())
            writer.start()
            try:
                writer.write(b"abcdef")
                await writer.drain()
            finally:
                writer.stop_and_join()

        self.assertEqual(bytes(written), b"abcdef")

    async def test_a_write_failure_is_raised_by_drain(self):
        def write_file(handle, data):
            raise OSError("pipe closed")

        with mock.patch.object(host_ipc.windows_api, "write_file",
                               side_effect=write_file):
            writer = host_ipc._HandleWriter(1, asyncio.get_running_loop())
            writer.start()
            try:
                writer.write(b"data")
                with self.assertRaises(OSError):
                    await writer.drain()
            finally:
                writer.stop_and_join()

    async def test_drain_of_an_empty_queue_returns(self):
        with mock.patch.object(host_ipc.windows_api, "write_file") as write:
            writer = host_ipc._HandleWriter(1, asyncio.get_running_loop())
            writer.start()
            try:
                await writer.drain()
            finally:
                writer.stop_and_join()
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
