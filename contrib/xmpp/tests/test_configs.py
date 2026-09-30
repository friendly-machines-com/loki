import asyncio
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import unittest

from loki_xmpp_bridge.configs import Config, ConfigurationError, private_directory
from fixtures import config_directory, config_file


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory, self.config = config_directory()
        self.addCleanup(self.directory.cleanup)
        self.path = config_file(self.config)

    def test_explicit_configuration_and_private_password(self):
        config = Config.load(self.path)
        self.assertEqual(config.jid, self.config.jid)
        self.assertEqual(config.allowed_jids, self.config.allowed_jids)
        self.assertEqual(config.password(), 'test-password')

    def test_unknown_options_empty_authorization_and_relative_paths_are_rejected(self):
        original = self.path.read_text()
        replacements = (
            original + '\nallow_everyone = true\n',
            original.replace('["alice@example.test"]', '[]'),
            original.replace(json.dumps(self.config.socket_path), '"relative.sock"'),
            original + '\nmax_connections = true\n',
            original.replace('alice@example.test', self.config.jid),
        )
        for content in replacements:
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaises(ConfigurationError):
                    Config.load(self.path)

    def test_writable_config_readable_password_and_symlinks_are_rejected(self):
        self.path.chmod(0o666)
        with self.assertRaises(ConfigurationError):
            Config.load(self.path)
        self.path.chmod(0o600)
        password = Path(self.config.password_file)
        password.chmod(0o644)
        with self.assertRaises(ConfigurationError):
            self.config.password()
        password.chmod(0o600)
        target = password.with_name('target-password')
        password.rename(target)
        password.symlink_to(target)
        with self.assertRaises(OSError):
            self.config.password()

    def test_unsafe_runtime_directory_is_rejected(self):
        directory = Path(self.config.socket_path).parent
        directory.mkdir(mode=0o755)
        with self.assertRaises(ConfigurationError):
            private_directory(directory)

    def test_installed_cli_help_check_config_and_invalid_config(self):
        executable = str(Path(sys.executable).with_name('loki-xmpp-bridge'))
        help_result = subprocess.run([executable, '--help'], capture_output=True, text=True, timeout=10)
        self.assertEqual(help_result.returncode, 0)
        result = subprocess.run([executable, '--config', str(self.path), '--check-config'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('test-password', result.stdout + result.stderr)
        self.assertFalse(Path(self.config.state_path).exists())
        self.assertFalse(Path(self.config.socket_path).exists())
        self.path.write_text('[bridge]\n')
        result = subprocess.run([executable, '--config', str(self.path)],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('Traceback', result.stderr)


class EntrypointTests(unittest.IsolatedAsyncioTestCase):
    async def test_installed_daemon_serves_socket_without_xmpp_and_cleans_up_on_sigterm(self):
        directory, config = config_directory()
        process = None
        writer = None
        unused = socket.socket()
        unused.bind(('127.0.0.1', 0))
        port = unused.getsockname()[1]
        unused.close()
        try:
            path = config_file(config)
            with path.open('a') as file:
                file.write(f'\nport = {port}\n')
            executable = str(Path(sys.executable).with_name('loki-xmpp-bridge'))
            process = await asyncio.create_subprocess_exec(
                executable, '--config', str(path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            async with asyncio.timeout(10):
                while not Path(config.socket_path).exists():
                    if process.returncode is not None:
                        _stdout, stderr = await process.communicate()
                        self.fail(stderr.decode())
                    await asyncio.sleep(0.01)
            reader, writer = await asyncio.open_unix_connection(config.socket_path)
            writer.write(b'{"type":"hello","version":1}\n')
            await writer.drain()
            self.assertEqual(json.loads(await reader.readline()), {'type': 'hello', 'version': 1})
            process.send_signal(signal.SIGTERM)
            _stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
            self.assertEqual(process.returncode, 0, stderr.decode())
            self.assertFalse(Path(config.socket_path).exists())
            self.assertTrue(Path(config.state_path).exists())
            self.assertNotIn(b'test-password', stderr)
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            if process is not None and process.returncode is None:
                process.kill()
                await process.communicate()
            directory.cleanup()
