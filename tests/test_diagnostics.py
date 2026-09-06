"""Exercise invoker-facing logging policy in fresh Python processes."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = """\
[loggers]
keys=root,loki,formats
[handlers]
keys=console,trace
[formatters]
keys=brief
[logger_root]
level=WARNING
handlers=console
[logger_loki]
qualname=loki_agent
level=WARNING
handlers=console,trace
propagate=0
[logger_formats]
qualname=loki_agent.formats
level=DEBUG
handlers=
propagate=1
[handler_console]
class=StreamHandler
level=WARNING
formatter=brief
args=(sys.stderr,)
[handler_trace]
class=FileHandler
level=DEBUG
formatter=brief
args=({path!r}, 'a', 'utf-8')
[formatter_brief]
format=%(levelname)s|%(name)s|%(message)s
"""
EMIT = """\
import logging
from loki_agent.diagnostics import configure_logging
from loki_agent.formats import report_unknown
if not configure_logging():
    raise SystemExit(2)
report_unknown('test', 'fields', {'extension': '\\x1b[31m'})
logging.getLogger('loki_agent.protocols').debug('other trace')
logging.getLogger('loki_agent.protocols').warning('visible warning')
"""


class DiagnosticsTests(unittest.TestCase):
    def run_code(self, code=EMIT, **env):
        environment = dict(os.environ)
        environment.pop('LOKI_LOG_CONFIG', None)
        environment.pop('LOKI_TRACE', None)
        environment.update(env)
        return subprocess.run(
            [sys.executable, '-c', code], cwd=ROOT, env=environment,
            capture_output=True, text=True, timeout=15)

    def test_default_silence_and_visible_warning(self):
        result = self.run_code()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertNotIn('Unknown', result.stderr)
        self.assertIn('visible warning', result.stderr)

    def test_trace_and_terminal_safe_payload(self):
        result = self.run_code(LOKI_TRACE='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertIn('Unknown test fields', result.stderr)
        self.assertIn('\\u001b', result.stderr)
        self.assertNotIn('\x1b', result.stderr)
        self.assertIn('other trace', result.stderr)
        self.assertIn('visible warning', result.stderr)

    def test_disabled_trace_does_not_serialize(self):
        result = self.run_code("""\
from unittest.mock import patch
from loki_agent.formats import report_unknown
with patch('loki_agent.diagnostics.json.dumps', side_effect=AssertionError):
    report_unknown('test', 'fields', object())
""")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ini_filtering_destinations_format_and_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'trace.log'
            config = Path(directory) / 'logging.ini'
            config.write_text(CONFIG.format(path=str(log)))
            result = self.run_code(
                LOKI_LOG_CONFIG=str(config), LOKI_TRACE='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '')
            self.assertEqual(
                result.stderr,
                'WARNING|loki_agent.protocols|visible warning\n')
            text = log.read_text()
            self.assertIn('DEBUG|loki_agent.formats|Unknown', text)
            self.assertNotIn('other trace', text)

    def test_repeated_setup_does_not_duplicate(self):
        result = self.run_code(EMIT.replace(
            "report_unknown('test'", "configure_logging()\nreport_unknown('test'"),
            LOKI_TRACE='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count('Unknown test fields'), 1)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'trace.log'
            config = Path(directory) / 'logging.ini'
            config.write_text(CONFIG.format(path=str(log)))
            result = self.run_code(EMIT.replace(
                "report_unknown('test'",
                "configure_logging()\nreport_unknown('test'"),
                LOKI_LOG_CONFIG=str(config))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(log.read_text().count('Unknown test fields'), 1)

    def test_invalid_explicit_config_does_not_fall_back_to_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'logging.ini'
            for content in (None, '', '[invalid]\nvalue=1'):
                if content is not None:
                    path.write_text(content)
                result = self.run_code(
                    LOKI_LOG_CONFIG=str(path), LOKI_TRACE='1')
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')
                self.assertIn('Logging configuration error:', result.stderr)
                self.assertNotIn('Traceback', result.stderr)

    def test_real_headless_supervisor_and_runtime_load_ini(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'logging.ini'
            # Per-process files prove configuration in both execed processes.
            text = CONFIG.format(path=str(Path(directory) / 'trace')).replace(
                repr(str(Path(directory) / 'trace')),
                repr(str(Path(directory) / 'trace-'))
                + " + str(__import__('os').getpid())")
            config.write_text(text)
            environment = {
                key: value for key, value in os.environ.items()
                if not key.startswith('LOKI_')
                and not key.endswith(('_KEY', '_TOKEN', '_PAT'))
            }
            environment.update({
                'HOME': directory,
                'XDG_CONFIG_HOME': str(Path(directory) / 'config'),
                'XDG_STATE_HOME': str(Path(directory) / 'state'),
                'LOKI_LOG_CONFIG': 'logging.ini',
                'LOKI_PROVIDER': 'dummy',
                'LOKI_API_BASE': 'http://dummy.invalid/v1',
                'LOKI_MODEL': 'dummy-model',
                'LOKI_DUMMY_REPLY': 'logging test answer',
            })
            result = subprocess.run(
                [str(ROOT / 'loki.py'), '--headless', '--prompt', 'hello'],
                cwd=directory, env=environment, capture_output=True,
                text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreaterEqual(len(list(Path(directory).glob('trace-*'))), 2)

    def test_relative_filename_is_preserved_in_environment_snapshot(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'logging.ini'
            path.write_text(CONFIG.format(path=str(Path(directory) / 'log')))
            literal = str(path.relative_to(ROOT))
            code = """\
import os
from loki_agent.diagnostics import configure_logging
from loki_agent.credentials import CredentialStore
original = os.environ['LOKI_LOG_CONFIG']
assert configure_logging()
store = CredentialStore.capture(os.environ)
assert store.sanitized_environment()['LOKI_LOG_CONFIG'] == original
assert os.environ['LOKI_LOG_CONFIG'] == original
"""
            result = self.run_code(code, LOKI_LOG_CONFIG=literal)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_tilde_is_a_literal_directory_name(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / '~').mkdir()
            (base / 'home').mkdir()
            log = base / 'literal.log'
            (base / '~' / 'logging.ini').write_text(CONFIG.format(path=str(log)))
            (base / 'home' / 'logging.ini').write_text('not the requested file')
            code = EMIT.replace(
                'if not configure_logging():',
                f'import os\nos.chdir({directory!r})\n'
                'if not configure_logging():')
            result = self.run_code(
                code, LOKI_LOG_CONFIG='~/logging.ini', HOME=str(base / 'home'))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Unknown test fields', log.read_text())

    @unittest.skipUnless(os.name == 'posix', 'POSIX symlink path semantics')
    def test_dotdot_after_symlink_is_not_collapsed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'project').mkdir()
            (base / 'elsewhere' / 'child').mkdir(parents=True)
            (base / 'project' / 'link').symlink_to(base / 'elsewhere' / 'child')
            log = base / 'literal.log'
            (base / 'elsewhere' / 'logging.ini').write_text(
                CONFIG.format(path=str(log)))
            (base / 'project' / 'logging.ini').write_text('not the requested file')
            code = EMIT.replace(
                'if not configure_logging():',
                f'import os\nos.chdir({directory!r})\n'
                'if not configure_logging():')
            for filename in ('project/link/../logging.ini',
                             directory + '/project/link/../logging.ini'):
                with self.subTest(filename=filename):
                    result = self.run_code(code, LOKI_LOG_CONFIG=filename)
                    self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(log.read_text().count('Unknown test fields'), 2)
