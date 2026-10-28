"""Supervision/shared contracts must not depend on model/tool execution.

Inspect the source dependency closure, including deferred imports: preloading a
module elsewhere must not conceal a backwards dependency in these tests.
"""

import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from loki_entrypoints import child_environment, entrypoint


PACKAGE = Path(__file__).resolve().parents[1] / 'loki_agent'


def dependencies():
    sources = {path.stem: path for path in PACKAGE.glob('*.py')}
    graph = {}
    for name, path in sources.items():
        edges = set()
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            if node.module is not None:
                candidate = node.module.split('.')[0]
                if candidate in sources:
                    edges.add(candidate)
            else:
                edges.update(alias.name for alias in node.names if alias.name in sources)
        graph[name] = edges
    return graph


def dependency_path(graph, start, forbidden):
    pending = [[start]]
    visited = set()
    while pending:
        route = pending.pop(0)
        current = route[-1]
        if current in forbidden:
            return route
        if current in visited:
            continue
        visited.add(current)
        pending.extend(route + [target] for target in sorted(graph.get(current, [])))
    return None


class ModuleBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = dependencies()

    def test_platform_selection_does_not_use_os_name(self):
        violations = []
        with os.scandir(PACKAGE) as directory:
            for entry in directory:
                if not entry.name.endswith('.py'):
                    continue
                with open(entry.path, encoding='utf-8') as source:
                    tree = ast.parse(source.read())
                os_names = {alias.asname or 'os' for node in ast.walk(tree)
                            if isinstance(node, ast.Import) for alias in node.names
                            if alias.name == 'os'}
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Attribute) and node.attr == 'name'
                            and isinstance(node.value, ast.Name) and node.value.id in os_names):
                        violations.append(f'{entry.name}:{node.lineno}')
                    if (isinstance(node, ast.ImportFrom) and node.module == 'os'
                            and any(alias.name == 'name' for alias in node.names)):
                        violations.append(f'{entry.name}:{node.lineno}')
        self.assertEqual(violations, [], 'Use direct sys.platform guards for platform selection')

    def test_help_needs_no_credential_storage_or_configured_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            blocked = Path(directory) / 'not-a-directory'
            blocked.write_text('untouched', encoding='utf-8')
            result = subprocess.run(
                [entrypoint('loki'), '--help'],
                env=child_environment(
                    PATH=os.environ.get('PATH', os.defpath), HOME=directory,
                    XDG_CONFIG_HOME=str(blocked), XDG_STATE_HOME=str(blocked),
                    LOKI_PROVIDER='invalid-provider'),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage: loki', result.stdout)
            self.assertEqual(result.stderr, '')
            self.assertEqual(blocked.read_text(encoding='utf-8'), 'untouched')
            self.assertEqual(list(Path(directory).iterdir()), [blocked])

    def test_supervision_and_shared_contracts_do_not_load_execution(self):
        execution = {'loki', 'terminal_frontend', 'acp_worker', 'subagents',
                     'files', 'jobs', 'web_tools', 'model_requests'}
        for component in ['credential_supervisors', 'acp', 'connections',
                          'arguments', 'tool_contracts']:
            with self.subTest(component=component):
                route = dependency_path(self.graph, component, execution)
                self.assertIsNone(route, ' -> '.join(route or []))

    def test_extracted_backends_do_not_reach_back_into_composition(self):
        for component in ['files', 'jobs', 'web_tools', 'model_requests']:
            with self.subTest(component=component):
                route = dependency_path(self.graph, component, {'loki'})
                self.assertIsNone(route, ' -> '.join(route or []))
        # HTTP text decoding is shared transport functionality. Inference must
        # not acquire HTML parsing by importing the web-tool implementation.
        route = dependency_path(self.graph, 'model_requests', {'web_tools'})
        self.assertIsNone(route, ' -> '.join(route or []))
