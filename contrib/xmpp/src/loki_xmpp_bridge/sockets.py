"""Private multi-client Unix server for Loki bridge protocol v1."""

import asyncio
import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import struct
import uuid

from .configs import ConfigurationError, open_private, private_directory
from .routes import STORAGE_ERRORS


FRAME_BYTES = 1024 * 1024
EVENTS = {'prompt_accepted', 'prompt_rejected', 'turn_started', 'input_chunk',
          'output_chunk', 'turn_finished', 'command_finished', 'session_state',
          'session_closing', 'protocol_error'}
OUTCOMES = {'completed', 'cancelled', 'error', 'max_loops', 'unexecuted'}


def reject_constant(value):
    raise ValueError('Non-finite JSON number')


async def read_frame(reader):
    raw = await reader.readline()
    if not raw:
        raise EOFError
    if len(raw) > FRAME_BYTES or not raw.endswith(b'\n'):
        raise ValueError('Invalid/oversized frame')
    try:
        message = json.loads(raw.decode('utf-8'), parse_constant=reject_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('Invalid JSON') from error
    if not isinstance(message, dict) or not isinstance(message.get('type'), str):
        raise ValueError('Expected a typed JSON object')
    return message


def integer(value, *, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('Invalid integer')
    return value


def identifier(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 128
            or any(ord(char) < 33 or ord(char) > 126 for char in value)):
        raise ValueError('Invalid input ID')
    return value


def uuid_string(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError('Expected a canonical UUID')
    return value


def text(value, maximum=512):
    if not isinstance(value, str) or len(value.encode('utf-8')) > maximum:
        raise ValueError('Invalid text field')
    return value


def boolean(value):
    if type(value) is not bool:
        raise ValueError('Expected a boolean')
    return value


def active_turn(value):
    if not isinstance(value, dict):
        raise ValueError('Expected active turn identity')
    identifier(value['input_id'])
    uuid_string(value['turn_id'])
    if value['origin'] not in ('keyboard', 'bridge'):
        raise ValueError('Invalid input origin')
    for key in ('model', 'provider'):
        if value.get(key) is not None:
            text(value[key])


def registration(message):
    if message['type'] != 'session_registered':
        raise ValueError('Expected session registration')
    uuid_string(message['instance_id'])
    uuid_string(message['conversation_id'])
    text(message['frontend'], 32)
    latest = integer(message['event_seq'])
    integer(message['acknowledged'], maximum=latest)
    integer(message['replay_from'], minimum=1, maximum=latest + 1)
    boolean(message['paused'])
    if message['active'] is not None:
        active_turn(message['active'])
    capabilities = message['capabilities']
    boolean(capabilities['accepts_prompts'])
    boolean(capabilities['publishes_turns'])
    inputs = message['inputs']
    if not isinstance(inputs, list) or len(inputs) > 1024:
        raise ValueError('Invalid retained input list')
    seen = set()
    for item in inputs:
        input_id = identifier(item['input_id'])
        if input_id in seen or item['status'] not in ('queued', 'running', 'finished'):
            raise ValueError('Invalid retained input state')
        seen.add(input_id)
        if item.get('outcome') is not None and item['outcome'] not in OUTCOMES:
            raise ValueError('Invalid retained outcome')


def validate_event(message, instance):
    if message.get('instance_id') != instance or message['type'] not in EVENTS:
        raise ValueError('Wrong instance/unknown event')
    integer(message['event_seq'], minimum=1)
    # Validate every optional field that reaches a database binding, even on
    # event variants which normally omit it. Malformed peers are not disk errors.
    if message.get('input_id') is not None:
        identifier(message['input_id'])
    if message.get('origin') is not None and message['origin'] not in ('keyboard', 'bridge'):
        raise ValueError('Invalid input origin')
    if message.get('turn_id') is not None:
        uuid_string(message['turn_id'])
    if message.get('outcome') is not None and message['outcome'] not in OUTCOMES:
        raise ValueError('Invalid outcome')
    kind = message['type']
    if kind in ('turn_started', 'turn_finished', 'command_finished', 'input_chunk', 'output_chunk'):
        identifier(message['input_id'])
        if message['origin'] not in ('keyboard', 'bridge'):
            raise ValueError('Invalid input origin')
    if kind == 'turn_started':
        active_turn(message)
    if kind == 'turn_finished':
        uuid_string(message['turn_id'])
        boolean(message['paused'])
    if kind in ('turn_finished', 'command_finished') and message['outcome'] not in OUTCOMES:
        raise ValueError('Invalid outcome')
    if kind in ('input_chunk', 'output_chunk'):
        integer(message['chunk_index'], maximum=2**31 - 1)
        text(message['text'], FRAME_BYTES)
        if message['turn_id'] is not None:
            uuid_string(message['turn_id'])
        elif kind == 'input_chunk':
            raise ValueError('Input chunks require a turn ID')
    if kind == 'prompt_accepted':
        identifier(message['input_id'])
        boolean(message['duplicate'])
        if message['status'] not in ('queued', 'running', 'finished'):
            raise ValueError('Invalid acceptance status')
    if kind in ('prompt_rejected', 'protocol_error'):
        text(message['reason'], 128)
        if message.get('input_id') is not None:
            identifier(message['input_id'])
    if kind == 'session_state':
        boolean(message['paused'])


class Peer:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.instance = None
        self.accepts_prompts = False
        self.closing = False
        self.lock = asyncio.Lock()

    async def send(self, message):
        encoded = (json.dumps(message, ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')
        if len(encoded) > FRAME_BYTES:
            raise ValueError('Outgoing frame exceeds limit')
        async with self.lock:
            if self.writer.is_closing():
                raise ConnectionError('Loki disconnected')
            self.writer.write(encoded)
            try:
                await asyncio.wait_for(self.writer.drain(), 5)
            except (OSError, TimeoutError):
                self.writer.close()
                raise

    async def close(self):
        self.writer.close()
        try:
            await asyncio.wait_for(self.writer.wait_closed(), 5)
        except (OSError, TimeoutError):
            pass


class SocketServer:
    def __init__(self, config, router):
        self.config = config
        self.router = router
        self.server = None
        self.lock = None
        self.inode = None
        self.tasks = set()
        self.writers = set()

    async def __aenter__(self):
        path = Path(self.config.socket_path)
        private_directory(path.parent)
        try:
            self.lock = open_private(str(path) + '.lock')
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                info = path.lstat()
            except FileNotFoundError:
                info = None
            if info is not None:
                if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                    raise ConfigurationError('Refusing to remove unsafe socket path')
                try:
                    _reader, writer = await asyncio.wait_for(
                        asyncio.open_unix_connection(str(path)), 1)
                except ConnectionRefusedError:
                    path.unlink()
                else:
                    writer.close()
                    await writer.wait_closed()
                    raise ConfigurationError('Socket already has a listener')
            self.server = await asyncio.start_unix_server(
                self.accept, path=str(path), limit=FRAME_BYTES, start_serving=False)
            os.chmod(path, 0o600)
            self.inode = (path.stat().st_dev, path.stat().st_ino)
            await self.server.start_serving()
        except BaseException:
            await self.__aexit__()
            raise
        return self

    async def __aexit__(self, *exc):
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        for writer in self.writers:
            writer.close()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        path = Path(self.config.socket_path)
        if self.inode is not None:
            try:
                info = path.lstat()
                if (info.st_dev, info.st_ino) == self.inode:
                    path.unlink()
            except FileNotFoundError:
                pass
            self.inode = None
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None

    async def accept(self, reader, writer):
        task = asyncio.current_task()
        if len(self.tasks) >= self.config.max_connections:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            return
        self.tasks.add(task)
        self.writers.add(writer)
        peer = Peer(reader, writer)
        try:
            if hasattr(socket, 'SO_PEERCRED'):
                raw = writer.get_extra_info('socket').getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('iII'))
                _pid, uid, _gid = struct.unpack('iII', raw)
                if uid != os.getuid():
                    raise ValueError('Unexpected Loki peer UID')
            hello = await asyncio.wait_for(read_frame(reader), 3)
            if hello != {'type': 'hello', 'version': 1} or type(hello.get('version')) is not int:
                raise ValueError('Unsupported protocol version')
            await peer.send({'type': 'hello', 'version': 1})
            message = await asyncio.wait_for(read_frame(reader), 5)
            registration(message)
            cursor = self.router.register(peer, message)
            if cursor:
                await peer.send({'type': 'ack', 'instance_id': peer.instance, 'event_seq': cursor})
            while True:
                event = await read_frame(reader)
                if event['type'] == 'event_gap':
                    if event['instance_id'] != peer.instance:
                        raise ValueError('Wrong gap instance')
                    first = integer(event['from_seq'], minimum=1)
                    last = integer(event['to_seq'], minimum=first, maximum=cursor)
                    if last >= message['replay_from']:
                        raise ValueError('Unexpected gap range')
                    continue  # registration already durably recorded this gap
                validate_event(event, peer.instance)
                cursor = self.router.event(peer, event)
                await peer.send({'type': 'ack', 'instance_id': peer.instance, 'event_seq': cursor})
        except STORAGE_ERRORS as error:
            self.router.fail(error)
        except (EOFError, OSError, ValueError, KeyError, TypeError, TimeoutError):
            pass  # malformed/untrusted local peers cannot stop other sessions
        finally:
            try:
                self.router.closed(peer)
            except STORAGE_ERRORS as error:
                self.router.fail(error)
            await peer.close()
            self.writers.discard(writer)
            self.tasks.discard(task)
