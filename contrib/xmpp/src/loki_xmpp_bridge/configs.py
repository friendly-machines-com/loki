"""Explicit configuration and private local file boundaries."""

from dataclasses import dataclass, fields
import os
from pathlib import Path
import stat
import tomllib

from slixmpp import JID


class ConfigurationError(ValueError):
    pass


def bare_jid(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ConfigurationError("Expected an account/room JID")
    jid = JID(value)
    if not jid.user or not jid.domain or jid.resource:
        raise ConfigurationError("Use a bare account/room JID without a resource")
    return jid.bare


def private_directory(path):
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077):
        raise ConfigurationError(f"Directory must be private and user-owned: {path}")


def open_private(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077):
        os.close(fd)
        raise ConfigurationError(f"File must be private and user-owned: {path}")
    return fd


def read_file(path, *, secret=False):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        owners = (os.getuid(),) if secret else (0, os.getuid())
        mask = 0o077 if secret else 0o022
        if (not stat.S_ISREG(info.st_mode) or info.st_uid not in owners
                or info.st_mode & mask or info.st_size > 65536):
            raise ConfigurationError("Unsafe configuration/password file permissions or size")
        with os.fdopen(fd, 'rb', closefd=False) as file:
            return file.read(65537)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Config:
    jid: str
    room: str
    nick: str
    allowed_jids: frozenset[str]
    password_file: str
    socket_path: str
    state_path: str
    host: str | None = None
    port: int = 5222
    ca_file: str | None = None
    max_connections: int = 64
    max_pending_commands: int = 128
    max_outbox_messages: int = 10000
    max_state_bytes: int = 256 * 1024 * 1024
    max_result_bytes: int = 4 * 1024 * 1024
    message_bytes: int = 3000
    network_timeout: int = 30
    echo_timeout: int = 30

    @classmethod
    def load(cls, path):
        document = tomllib.loads(read_file(path).decode('utf-8'))
        if set(document) != {'bridge'} or not isinstance(document['bridge'], dict):
            raise ConfigurationError("Configuration requires one [bridge] table")
        values = dict(document['bridge'])
        known = {field.name for field in fields(cls)}
        if set(values) - known:
            raise ConfigurationError("Unknown configuration option: " +
                                     ', '.join(sorted(set(values) - known)))
        required = ('jid', 'room', 'nick', 'allowed_jids', 'password_file',
                    'socket_path', 'state_path')
        if any(key not in values for key in required):
            raise ConfigurationError("Missing required bridge configuration")
        values['jid'] = bare_jid(values['jid'])
        values['room'] = bare_jid(values['room'])
        allowed = values['allowed_jids']
        if not isinstance(allowed, list) or not allowed:
            raise ConfigurationError("allowed_jids must be a nonempty list")
        values['allowed_jids'] = frozenset(bare_jid(jid) for jid in allowed)
        if values['jid'] in values['allowed_jids']:
            raise ConfigurationError("The bot must not be a command author")
        nick = values['nick']
        if (not isinstance(nick, str) or not 1 <= len(nick) <= 64
                or any(ord(char) < 32 for char in nick)):
            raise ConfigurationError("Invalid room nickname")
        for key in ('password_file', 'socket_path', 'state_path', 'ca_file'):
            value = values.get(key)
            if value is not None and (not isinstance(value, str)
                                      or '\x00' in value or not Path(value).is_absolute()):
                raise ConfigurationError(f"{key} must be an absolute path")
        destinations = [values[key] for key in ('socket_path', 'state_path', 'password_file')]
        if len(set(destinations)) != len(destinations):
            raise ConfigurationError("Socket, database, and password paths must differ")
        if 'host' in values and (not isinstance(values['host'], str) or not values['host']):
            raise ConfigurationError("host must be a nonempty hostname/address")
        result = cls(**values)
        for key in ('port', 'max_connections', 'max_pending_commands',
                    'max_outbox_messages', 'max_state_bytes', 'max_result_bytes',
                    'message_bytes', 'network_timeout', 'echo_timeout'):
            value = getattr(result, key)
            if type(value) is not int or value <= 0:
                raise ConfigurationError(f"{key} must be a positive integer")
        if result.port > 65535 or not 256 <= result.message_bytes <= 16384:
            raise ConfigurationError("Invalid port or message_bytes (256..16384)")
        if result.max_state_bytes < 1024 * 1024:
            raise ConfigurationError("max_state_bytes must be at least 1 MiB")
        return result

    def password(self):
        password = read_file(self.password_file, secret=True).decode('utf-8').rstrip('\r\n')
        if not password or '\x00' in password:
            raise ConfigurationError("Password file is empty/invalid")
        return password
