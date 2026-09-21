"""Exercise invoker-facing logging policy in fresh Python processes."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from loki_entrypoints import configure_container, entrypoint


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

    def test_a_contained_runtime_ignores_the_invoker_config(self):
        # It cannot be assumed able to read the file the invoker names, nor to
        # write the handlers' targets; it says so and uses stderr.
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'trace.log'
            config = Path(directory) / 'logging.ini'
            config.write_text(CONFIG.format(path=str(log)))
            result = self.run_code(
                "from loki_agent.diagnostics import configure_logging\n"
                "assert configure_logging(contained=True)\n",
                LOKI_LOG_CONFIG=str(config), LOKI_TRACE='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '')
            self.assertIn('Logging configuration ignored', result.stderr)
            self.assertFalse(log.exists())

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

    def test_real_headless_supervisor_loads_ini_and_the_runtime_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            # The supervisor is uncontained and loads this config, writing the
            # trace it names; the contained runtime refuses any configuration
            # the invoker names and logs to stderr instead.  The temp root is
            # created private (0o700, honoured on Windows), which carries no
            # package ACE and so is invisible to the container.
            workspace = os.path.join(directory, "workspace")
            os.makedirs(workspace)
            config = Path(workspace) / 'logging.ini'
            trace = str(Path(workspace) / 'trace')
            # Per-process files prove configuration in both execed processes.
            text = CONFIG.format(path=trace).replace(
                repr(trace),
                repr(trace + '-') + " + str(__import__('os').getpid())")
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
                'LOKI_LOG_CONFIG': str(config),
                'LOKI_PROVIDER': 'dummy',
                'LOKI_API_BASE': 'http://dummy.invalid/v1',
                'LOKI_MODEL': 'dummy-model',
                'LOKI_DUMMY_REPLY': 'logging test answer',
            })
            configure_container(environment, workspace)
            result = subprocess.run(
                [entrypoint('loki'), '--headless', '--prompt', 'hello'],
                cwd=workspace, env=environment, capture_output=True,
                text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Logging configuration ignored", result.stderr)
            self.assertEqual(
                len(list(Path(workspace).glob('trace-*'))), 1)

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

    def test_dotdot_after_symlink_uses_platform_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'project').mkdir()
            (base / 'elsewhere' / 'child').mkdir(parents=True)
            (base / 'project' / 'link').symlink_to(
                base / 'elsewhere' / 'child', target_is_directory=True)
            log = base / 'literal.log'
            # POSIX follows the link and applies `..` in its target, so the
            # elsewhere copy is the one selected; Windows resolves `..`
            # lexically and selects the project copy.  Put the requested config
            # where each platform actually looks.
            selected = base / ('project' if os.name != 'posix' else 'elsewhere')
            other = base / ('elsewhere' if os.name != 'posix' else 'project')
            (selected / 'logging.ini').write_text(
                CONFIG.format(path=str(log)))
            (other / 'logging.ini').write_text('not the requested file')
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
