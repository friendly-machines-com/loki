import json
import os
from pathlib import Path
import tempfile
import uuid

from loki_xmpp_bridge.configs import Config


def config_directory():
    directory = tempfile.TemporaryDirectory(prefix='loki-xmpp-test-')
    root = Path(directory.name)
    password = root / 'password'
    password.write_text('test-password\n', encoding='utf-8')
    password.chmod(0o600)
    config = Config(
        jid='bot@example.test', room='agent@conference.example.test', nick='Bridge',
        allowed_jids=frozenset({'alice@example.test'}), password_file=str(password),
        socket_path=str(root / 'runtime' / 'bridge.sock'),
        state_path=str(root / 'state' / 'bridge.sqlite3'), host='127.0.0.1',
        network_timeout=1, echo_timeout=1)
    return directory, config


def registration(instance=None, conversation=None, *, latest=0, replay_from=1,
                 acknowledged=0, inputs=None, paused=False, active=None, accepts=True):
    return {'type': 'session_registered', 'instance_id': instance or str(uuid.uuid4()),
            'conversation_id': conversation or str(uuid.uuid4()), 'frontend': 'terminal',
            'event_seq': latest, 'replay_from': replay_from, 'acknowledged': acknowledged,
            'inputs': inputs or [], 'paused': paused, 'active': active,
            'capabilities': {'accepts_prompts': accepts, 'publishes_turns': True}}


class FakePeer:
    def __init__(self):
        self.instance = None
        self.closing = False
        self.accepts_prompts = False
        self.sent = []
        self.error = None

    async def send(self, message):
        if self.error:
            raise self.error
        self.sent.append(message)


def config_file(config):
    path = Path(config.password_file).parent / 'config.toml'
    fields = ['[bridge]']
    for name in ('jid', 'room', 'nick', 'password_file', 'socket_path', 'state_path', 'host'):
        fields.append(f'{name} = {json.dumps(getattr(config, name))}')
    fields.append('allowed_jids = ' + json.dumps(sorted(config.allowed_jids)))
    fields.extend(['network_timeout = 1', 'echo_timeout = 1'])
    path.write_text('\n'.join(fields), encoding='utf-8')
    os.chmod(path, 0o600)
    return path
