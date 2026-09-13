"""Tests for the IPC seam's POSIX contract.

The Windows branch is AF_UNIX built by hand and cannot even be imported here
(it binds a socket path and uses ``subprocess.STARTUPINFO``), so what is checked
is the half that runs on this host: a connected pair, a channel whose close is
EOF, and the reference/spawn/validation contract the child depends on.
"""

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


if __name__ == "__main__":
    unittest.main()
