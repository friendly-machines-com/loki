"""Tests for the IPC seam's POSIX contract and its Windows endpoint types.

The Windows channel is now two anonymous pipes (AF_UNIX is unreachable through
CPython), and its endpoint/encoder/stream classes depend only on import-safe
declarations, so they are exercised here on Linux with the Windows calls
mocked.  The POSIX half runs for real: a connected pair, a channel whose close
is EOF, and the reference/spawn/validation contract the child depends on.
"""

import asyncio
import contextlib
import ctypes
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from loki_agent import host_ipc
from loki_agent import windows_subprocesses


@contextlib.contextmanager
def mocked_named_pipe(server_handle, *, client_handle=0x22, connect_error=None,
                      open_error=None):
    """The Win32 calls ``windows_subprocesses.pipe()`` makes, with state.

    The named-pipe pair is Windows-only in effect but depends only on
    declarations, like the anonymous pair above, so its creation attributes
    and failure paths are exercised on the same host that runs the rest.
    """
    state = {"created": [], "closed": [], "connected": [], "last_error": 0, 'token_closed': False}
    token = object()

    def close(handle):
        if handle is token:
            state['token_closed'] = True
        else:
            state['closed'].append(handle)

    def named_pipe(address, mode, pipe_mode, instances, obsize, ibsize,
                   timeout, attributes):
        state["created"].append({
            "address": address, "mode": mode, "pipe_mode": pipe_mode,
            "instances": instances,
            "inherit": attributes._obj.bInheritHandle})
        return server_handle

    def connect(server, overlapped):
        state["connected"].append(server)
        if connect_error is not None:
            state["last_error"] = connect_error
            return 0
        return 1

    def convert(sddl, revision, output, size):
        state["sddl"] = sddl
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 0x5EC
        return 1

    def bound(library, symbol, *signature):
        return {
            "CreateNamedPipeW": named_pipe,
            "ConnectNamedPipe": connect,
            "ConvertStringSecurityDescriptorToSecurityDescriptorW": convert,
            "LocalFree": lambda *args: None,
        }[symbol]

    def open_client(*args, **kwargs):
        if open_error is not None:
            raise open_error
        return client_handle

    with mock.patch.object(windows_subprocesses.api, "bind",
                           side_effect=bound), \
            mock.patch.object(windows_subprocesses.api, "current_user_sid",
                              return_value="S-1-5-21-9"), \
            mock.patch.object(windows_subprocesses.api, 'current_process_handle', return_value=10), \
            mock.patch.object(windows_subprocesses.api, 'open_process_token', return_value=token), \
            mock.patch.object(windows_subprocesses.api, 'token_is_app_container', return_value=False), \
            mock.patch.object(windows_subprocesses.api, "open_with_access",
                              side_effect=open_client) as opened, \
            mock.patch.object(windows_subprocesses.api, "close_handle", side_effect=close), \
            mock.patch.object(ctypes, "get_last_error",
                              side_effect=lambda: state["last_error"],
                              create=True):
        state["opened"] = opened
        yield state
        if not state['token_closed']:
            raise AssertionError('pipe identity query retained its token handle')


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


class PrivatePipeCreationTests(unittest.TestCase):
    @contextlib.contextmanager
    def identity(self, contained):
        with mock.patch.multiple(
                host_ipc.windows_api,
                current_user_sid=mock.DEFAULT,
                current_process_handle=mock.DEFAULT,
                open_process_token=mock.DEFAULT,
                token_is_app_container=mock.DEFAULT,
                token_app_container_sid=mock.DEFAULT,
                close_handle=mock.DEFAULT,
                create_pipe=mock.DEFAULT) as calls:
            calls['current_user_sid'].return_value = 'S-1-5-21-9'
            calls['current_process_handle'].return_value = 10
            calls['open_process_token'].return_value = 11
            calls['token_is_app_container'].return_value = contained
            calls['token_app_container_sid'].return_value = 'S-1-15-2-123'
            calls['create_pipe'].return_value = [21, 22]
            yield calls

    def test_uncontained_creator_uses_only_private_trustees(self):
        with self.identity(False) as calls:
            self.assertEqual(host_ipc.private_pipe(), [21, 22])
        calls['open_process_token'].assert_called_once_with(10)
        calls['close_handle'].assert_called_once_with(11)
        calls['token_app_container_sid'].assert_not_called()
        calls['create_pipe'].assert_called_once_with(
            inherit=True,
            security_descriptor='D:P(A;;GA;;;S-1-5-21-9)(A;;GA;;;SY)(A;;GA;;;BA)')

    def test_package_gets_client_reopen_rights_not_pipe_data_read(self):
        with self.identity(True) as calls:
            host_ipc.private_pipe(inherit=False)
        calls['token_app_container_sid'].assert_called_once_with(11)
        calls['close_handle'].assert_called_once_with(11)
        calls['create_pipe'].assert_called_once_with(
            inherit=False,
            security_descriptor=('D:P(A;;GA;;;S-1-5-21-9)(A;;GA;;;SY)(A;;GA;;;BA)'
                                 '(A;;0x120196;;;S-1-15-2-123)'))

    def test_duplex_stdio_read_grant_never_reaches_confidential_one_way_pipes(self):
        with self.identity(True) as calls, mock.patch.object(host_ipc.windows_api, 'set_token_default_dacl') as default:
            policy = host_ipc.pipe_policy(duplex=True)
            host_ipc.private_pipe()
            self.assertEqual(policy['package'], 'S-1-15-2-123')
            self.assertIn('(A;;0x12019f;;;S-1-15-2-123)', policy['descriptor'])
            self.assertIn('(A;;0x120196;;;S-1-15-2-123)', calls['create_pipe'].call_args.kwargs['security_descriptor'])
            default.assert_not_called()

    def test_identity_query_failure_closes_token_and_never_allocates_pipe(self):
        with self.identity(True) as calls:
            calls['token_app_container_sid'].side_effect = OSError('identity unavailable')
            with self.assertRaisesRegex(OSError, 'identity unavailable'):
                host_ipc.private_pipe()
        calls['close_handle'].assert_called_once_with(11)
        calls['create_pipe'].assert_not_called()

    def test_pipe_creation_failure_is_not_retried_with_default_security(self):
        with self.identity(True) as calls:
            calls['create_pipe'].side_effect = OSError('access denied')
            with self.assertRaisesRegex(OSError, 'access denied'):
                host_ipc.private_pipe()
        calls['close_handle'].assert_called_once_with(11)
        self.assertEqual(calls['create_pipe'].call_count, 1)


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
        with mock.patch.object(host_ipc, "private_pipe",
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

        # The worker's stdio pair: the end the child reads is created without
        # overlapped I/O, its partner is the front's overlapped write end, and
        # the pipe is created with our own descriptor -- the default one
        # grants Everyone read -- and refuses remote clients.
        with mocked_named_pipe(0x30) as state:
            server, client = windows_subprocesses.pipe(
                overlapped=(False, True), duplex=True)
        self.assertEqual((server, client), (0x30, 0x22))
        created_pipe = state["created"][0]
        self.assertTrue(
            created_pipe["address"].startswith(r"\\.\pipe\loki-worker-"))
        self.assertEqual(created_pipe["instances"], 1)
        self.assertEqual(created_pipe["mode"] & 0x40000000, 0)  # not overlapped
        self.assertEqual(created_pipe["mode"] & 0x00080000, 0x00080000)
        self.assertEqual(created_pipe["pipe_mode"] & 0x8, 0x8)
        self.assertEqual(created_pipe["inherit"], 0)
        self.assertEqual(state["sddl"],
                         "D:P(A;;GA;;;S-1-5-21-9)(A;;GA;;;SY)(A;;GA;;;BA)")
        self.assertEqual(state["opened"].call_args.kwargs["flags"],
                         0x40000000)
        self.assertEqual(state["connected"], [0x30])
        self.assertEqual(state["closed"], [])

        # The stdout pair is the mirror image: the front's read end is the
        # overlapped one, the child's write end is not.
        with mocked_named_pipe(0x31) as state:
            windows_subprocesses.pipe(overlapped=(True, False))
        self.assertEqual(state["created"][0]["mode"] & 0x40000000, 0x40000000)
        self.assertEqual(state["opened"].call_args.kwargs["flags"], 0)

    def test_second_pipe_failure_closes_the_first_pipe(self):
        closed = []
        with mock.patch.object(
                host_ipc, "private_pipe",
                side_effect=[(0x10, 0x11), OSError("no more handles")]), \
                mock.patch.object(host_ipc.windows_api,
                                  "clear_handle_inheritance"), \
                mock.patch.object(host_ipc.windows_api, "close_handle",
                                  side_effect=closed.append):
            with self.assertRaises(OSError):
                host_ipc._private_pipe_pair()
        self.assertEqual(closed, [0x10, 0x11])

        # Same for the named pair: a client that cannot open leaves no server
        # behind, and a connect the kernel refuses closes both ends.
        with mocked_named_pipe(0x30, open_error=OSError("no client")) as state:
            with self.assertRaises(OSError):
                windows_subprocesses.pipe(overlapped=(True, False))
        self.assertEqual(state["closed"], [0x30])
        with mocked_named_pipe(0x30, connect_error=5) as state:
            with self.assertRaises(windows_subprocesses.api.WindowsApiError):
                windows_subprocesses.pipe(overlapped=(True, False))
        self.assertEqual(state["closed"], [0x30, 0x22])

    def test_inheritance_failure_closes_all_four_ends(self):
        closed = []
        with mock.patch.object(
                host_ipc, "private_pipe",
                side_effect=[(0x10, 0x11), (0x12, 0x13)]), \
                mock.patch.object(
                    host_ipc.windows_api, "clear_handle_inheritance",
                    side_effect=[None, OSError("cannot clear")]), \
                mock.patch.object(host_ipc.windows_api, "close_handle",
                                  side_effect=closed.append):
            with self.assertRaises(OSError):
                host_ipc._private_pipe_pair()
        self.assertEqual(closed, [0x10, 0x11, 0x12, 0x13])


class SocketPairTests(unittest.TestCase):
    def test_the_pair_is_connected_in_both_directions(self):
        if os.name != "posix":
            # Two anonymous pipes: the parent end carries the request read and
            # the response write, the child end the complementary two, and only
            # the parent ends have inheritance cleared.
            created = iter([(0x10, 0x11), (0x12, 0x13)])
            cleared = []
            with mock.patch.object(host_ipc, "private_pipe",
                                   side_effect=lambda: next(created)), \
                    mock.patch.object(host_ipc.windows_api,
                                      "clear_handle_inheritance",
                                      side_effect=cleared.append):
                parent, child = host_ipc.socket_pair()
            self.assertEqual(parent.handles(), [0x10, 0x13])
            self.assertEqual(child.handles(), [0x12, 0x11])
            self.assertEqual(cleared, [0x10, 0x13])
            return
        first, second = host_ipc.socket_pair()
        self.addCleanup(first.close)
        self.addCleanup(second.close)

        first.sendall(b"ping")
        self.assertEqual(second.recv(4), b"ping")
        second.sendall(b"pong")
        self.assertEqual(first.recv(4), b"pong")


class OwnerChannelTests(unittest.TestCase):
    def test_closing_the_parent_end_is_eof_for_the_child(self):
        if os.name != "posix":
            import msvcrt
            parent_end, child_end = host_ipc.owner_channel()
            # The child owns only the read end; the fd takes ownership of that
            # handle, so the endpoint must not close it a second time.
            fd = msvcrt.open_osfhandle(child_end.read, os.O_RDONLY)
            child_end.read = None
            try:
                host_ipc.close_end(parent_end)
                self.assertEqual(os.read(fd, 1), b"")
            finally:
                os.close(fd)
            return
        parent_end, child_end = host_ipc.owner_channel()
        self.addCleanup(os.close, child_end)

        host_ipc.close_end(parent_end)

        self.assertEqual(os.read(child_end, 1), b"")


@unittest.skipUnless(os.name == 'posix', 'POSIX descriptor inheritance')
class ChildReferenceTests(unittest.TestCase):
    def test_borrowed_directory_copy_preserves_closed_standard_streams(self):
        code = r'''
import errno,json,os,sys
from loki_agent import host_ipc
report = os.dup(1)
closed = json.loads(sys.argv[1])
for fd in closed:
    os.close(fd)
source = os.open(sys.argv[2], os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
assert source == min(closed)
original = os.fstat(source)
copy = host_ipc.duplicate_for_child(source)
assert copy >= 3 and copy != source
assert not os.get_inheritable(copy)
assert not os.get_inheritable(source)
assert os.path.samestat(original, os.fstat(copy))
os.close(copy)
assert os.path.samestat(original, os.fstat(source))
for fd in closed:
    if fd == source:
        continue
    try:
        os.fstat(fd)
    except OSError as error:
        assert error.errno == errno.EBADF
    else:
        raise AssertionError('duplication filled a closed standard stream')
os.close(source)
os.write(report, b'borrowed object preserved')
os.close(report)
'''
        with tempfile.TemporaryDirectory() as directory:
            for closed in [[0], [1], [2], [0, 1, 2]]:
                with self.subTest(closed=closed):
                    process = subprocess.run(
                        [sys.executable, '-c', code, json.dumps(closed), directory],
                        stdin=subprocess.DEVNULL, capture_output=True)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    self.assertEqual(process.stdout, b'borrowed object preserved')

    def test_owned_channel_references_avoid_closed_standard_streams(self):
        code = r'''
import os,socket,sys
from loki_agent import host_ipc
report = os.dup(1)
for fd in [0,1,2]:
    os.close(fd)
if sys.argv[1] == 'owner':
    parent,child = host_ipc.owner_channel()
    try:
        assert child >= 3 and not os.get_inheritable(child)
        assert host_ipc.child_endpoint(str(child)) == child
        os.write(parent, b'x')
        assert os.read(child, 1) == b'x'
    finally:
        os.close(parent)
        os.close(child)
else:
    parent,end = host_ipc.socket_pair()
    child = host_ipc.prepare_child_socket(end)
    assert end.fileno() == -1
    try:
        assert child >= 3 and not os.get_inheritable(child)
        assert host_ipc.child_endpoint(str(child)) == child
        with socket.socket(fileno=child) as client:
            parent.sendall(b'x')
            assert client.recv(1) == b'x'
            client.sendall(b'y')
            assert parent.recv(1) == b'y'
    finally:
        parent.close()
os.write(report, b'channel usable')
os.close(report)
'''
        for kind in ['owner', 'credential']:
            with self.subTest(kind=kind):
                process = subprocess.run(
                    [sys.executable, '-c', code, kind],
                    stdin=subprocess.DEVNULL, capture_output=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(process.stdout, b'channel usable')

    def test_credential_copy_failure_cannot_leave_a_server_task(self):
        from loki_agent import authentications, credential_capabilities

        async def scenario():
            tasks = asyncio.all_tasks()
            ends = []
            pair = host_ipc.socket_pair

            def allocate():
                parent, child = pair()
                ends.extend([parent, child])
                return parent, child

            with mock.patch.object(host_ipc, 'socket_pair', side_effect=allocate), \
                    mock.patch.object(host_ipc, 'duplicate_for_child',
                                      side_effect=OSError('copy refused')):
                with self.assertRaisesRegex(OSError, 'copy refused'):
                    await credential_capabilities.CredentialCapabilityServer.create(
                        authentications.CredentialBroker())
            self.assertTrue(ends)
            self.assertTrue(all(end.fileno() == -1 for end in ends))
            await asyncio.sleep(0)
            self.assertFalse(asyncio.all_tasks() - tasks)

        asyncio.run(scenario())

    def test_capability_validation_failure_closes_adopted_transport_and_child_copy(self):
        from loki_agent import credential_capabilities

        async def scenario():
            tasks = asyncio.all_tasks()
            authority = mock.Mock()
            authority.available.side_effect = [frozenset(['revoked']), frozenset()]
            ends, copies = [], []
            pair = host_ipc.socket_pair
            duplicate = host_ipc.duplicate_for_child

            def allocate():
                parent, child = pair()
                ends.extend([parent, child])
                return parent, child

            def copy(fd):
                result = duplicate(fd)
                copies.append(result)
                return result

            with mock.patch.object(host_ipc, 'socket_pair', side_effect=allocate), \
                    mock.patch.object(host_ipc, 'duplicate_for_child', side_effect=copy):
                with self.assertRaisesRegex(ValueError, 'exceed the upstream authority'):
                    await credential_capabilities.CredentialCapabilityServer.create(
                        authority, allowed={'revoked'})
            self.assertTrue(ends)
            self.assertTrue(all(end.fileno() == -1 for end in ends))
            for fd in copies:
                with self.assertRaises(OSError):
                    os.fstat(fd)
            await asyncio.sleep(0)
            self.assertFalse(asyncio.all_tasks() - tasks)

        asyncio.run(scenario())

    def test_copy_failure_releases_owned_ends_but_not_a_borrowed_source(self):
        created = []
        pipe = os.pipe

        def allocate():
            ends = pipe()
            created.extend(ends)
            return ends

        with mock.patch.object(host_ipc.os, 'pipe', side_effect=allocate), \
                mock.patch.object(host_ipc, 'duplicate_for_child',
                                  side_effect=OSError('copy refused')):
            with self.assertRaisesRegex(OSError, 'copy refused'):
                host_ipc.owner_channel()
        for fd in created:
            with self.assertRaises(OSError):
                os.fstat(fd)

        parent, child = socket.socketpair()
        self.addCleanup(parent.close)
        with mock.patch.object(host_ipc, 'duplicate_for_child',
                               side_effect=OSError('copy refused')):
            with self.assertRaisesRegex(OSError, 'copy refused'):
                host_ipc.prepare_child_socket(child)
        self.assertEqual(child.fileno(), -1)
        self.assertEqual(parent.recv(1), b'')

        import fcntl

        with tempfile.TemporaryFile() as source:
            with mock.patch.object(fcntl, 'fcntl', side_effect=OSError('copy refused')):
                with self.assertRaisesRegex(OSError, 'copy refused'):
                    host_ipc.duplicate_for_child(source.fileno())
            os.fstat(source.fileno())


class ReferenceTests(unittest.TestCase):
    def test_a_reference_names_the_child_end(self):
        if os.name != "posix":
            endpoint = host_ipc.PipeEndpoint(read=0x11, write=0x22)
            rebuilt = host_ipc.child_endpoint(host_ipc.reference(endpoint))
            self.assertEqual(rebuilt.handles(), [0x11, 0x22])
            return
        reader, writer = os.pipe()
        self.addCleanup(os.close, reader)
        self.addCleanup(os.close, writer)

        value = host_ipc.reference(reader)

        self.assertIsInstance(value, int)
        self.assertEqual(host_ipc.child_endpoint(str(value)), reader)

    def test_spawn_kwargs_hand_the_references_over(self):
        if os.name != "posix":
            endpoint = host_ipc.PipeEndpoint(read=0x21, write=0x22)
            startup = host_ipc.spawn_kwargs((endpoint,))["startupinfo"]
            self.assertEqual(
                startup.lpAttributeList["handle_list"], [0x21, 0x22])
            return
        self.assertEqual(host_ipc.spawn_kwargs((4, 7)), {"pass_fds": [4, 7]})

    def test_a_descriptor_that_was_not_inherited_is_rejected(self):
        with self.assertRaises(ValueError):
            host_ipc.child_endpoint("1")


class PipeEndpointTests(unittest.TestCase):
    """The Windows endpoint is the only thing that crosses to a child."""

    def test_reference_round_trips_both_directions(self):
        endpoint = host_ipc.PipeEndpoint(read=0x11, write=0x22)
        self.assertEqual(endpoint.handles(), [0x11, 0x22])
        self.assertEqual(endpoint.reference(), "r=17,w=34")
        rebuilt = host_ipc.PipeEndpoint.parse("r=17,w=34")
        self.assertEqual(rebuilt.handles(), [0x11, 0x22])

    def test_a_unidirectional_endpoint_round_trips(self):
        endpoint = host_ipc.PipeEndpoint(read=0x5)
        self.assertEqual(endpoint.handles(), [0x5])
        self.assertEqual(host_ipc.PipeEndpoint.parse("r=5").read, 0x5)

    def test_malformed_references_are_refused(self):
        for value in ["", "5", "r=", "r=0", "x=5", "r=1,r=2", "r=1,w=2,z=3"]:
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

    def test_a_descriptor_reports_one_handle(self):
        if os.name != "posix":
            self.assertEqual(
                host_ipc.handles(host_ipc.PipeEndpoint(read=4)), [4])
            return
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
