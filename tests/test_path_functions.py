"""Filename policy is part of the one normal verification driver."""

import unittest

import lint_path_functions as policy


class PathCheckerTests(unittest.TestCase):
    def test_helpers_follow_import_assignment_and_getattr_aliases(self):
        sources = [
            'import os as system\nsystem.path.normpath(name)',
            'import os.path as paths\npaths.realpath(name)',
            'from os.path import basename as leaf\nleaf(name)',
            'from os import path as paths\nhelper = paths.dirname\nhelper(name)',
            'import ntpath as paths\nhelper = getattr(paths, "normcase")\nhelper(name)',
            'import os\nattribute = "expanduser"\nhelper = getattr(os.path, attribute)\nhelper(name)',
            'helper = filename.resolve\nhelper()',
            'helper = normpath\nhelper(name)',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(policy.source_violations(source))

    def test_unknown_attribute_classification_does_not_abort_scanning(self):
        for binding in [
                'attribute = "set_mode" if console else "set_flags"',
                'attribute = choose_attribute()',
                'from extension import attribute',
                'attribute = "normpath"\nfrom extension import attribute',
                'attribute = "normpath"\nimport extension as attribute']:
            with self.subTest(binding=binding):
                source = binding + '\nsetter = getattr(native, attribute)\nsetter(value)\nimport os\nos.path.normpath(name)'
                problems = policy.source_violations(source)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn('normpath()', problems[0])

    def test_expanduser_accepts_authored_literals_not_dynamic_operands(self):
        for source in [
                'import os\nos.path.expanduser("~/foo")',
                'from os.path import expanduser as home\nhome("~/foo")',
                'import os\nos.path.expanduser(path="~/foo")']:
            with self.subTest(source=source):
                self.assertEqual(policy.source_violations(source), [])
        for source in [
                'import os\nos.path.expanduser(os.environ["CONFIG"])',
                'import os\nos.path.expanduser(path=os.environ["CONFIG"])',
                'from os.path import expanduser as home\nhome(name)',
                'import os\nos.path.expanduser(f"~/{name}")']:
            with self.subTest(source=source):
                self.assertTrue(policy.source_violations(source))

    def test_path_constructors_and_indirect_spellings_are_checked(self):
        sources = [
            'from pathlib import Path as P\nP(name)',
            'from pathlib import *\nfactory = Path\nfactory(name)',
            'import pathlib as paths\nfactory = paths.Path\nalias = factory\nalias(name)',
            'import pathlib\ngetattr(pathlib, "Path")(name)',
            '__import__("pathlib").Path(name)',
            'import importlib as imports\nimports.import_module("pathlib").Path(name)',
            'from pathlib import PureWindowsPath as P\nP(name)',
            'from pathlib import Path as P\nP.cwd()',
            'from pathlib import Path as P\nparser.add_argument("name", type=P)',
            'from pathlib import Path as P\nmap(P, names)',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(policy.source_violations(source))

    def test_division_and_operator_aliases_cannot_hide_path_construction(self):
        sources = [
            'root / name', 'root /= name',
            'import operator as operations\noperations.truediv(root, name)',
            'from operator import truediv as divide\ndivide(root, name)',
            'root.__truediv__(name)', 'root.__rtruediv__(name)', 'root.joinpath(name)',
            'join = root.joinpath\njoin(name)',
        ]
        prefix = 'from pathlib import Path\nroot = Path(directory)\n'
        for source in sources:
            with self.subTest(source=source):
                problems = policy.source_violations(prefix + source)
                self.assertTrue(any('division' in problem or 'navigation/operator' in problem
                                    for problem in problems), problems)

    def test_numeric_time_conversions_are_not_path_operations(self):
        sources = [
            'elapsed = nanoseconds / 1_000_000_000',
            'milliseconds = nanoseconds / 1_000_000',
            'seconds = milliseconds / 1000.0',
            'backoff /= 2',
            'scale = 1_000_000_000\nseconds = ticks / scale',
            'import operator as operations\noperations.truediv(ticks, 1000)',
            'def ratio(a: float, b: float):\n    return a / b',
            'def ratio(a, b):\n    return a / b',
            'import operator\ndef ratio(a, b):\n    return operator.truediv(a, b)',
            'def ratio(a: float, b: float):\n    return a.__truediv__(b)',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(policy.source_violations(source), [])

    def test_alias_uses_are_checked_before_later_reassignment(self):
        """A later binding must not erase a prohibited call that already occurred."""
        sources = [
            'from pathlib import Path\nmake = Path\nmake("link/../file")\nmake = str',
            'from pathlib import Path as P\ndef f():\n    make = P\n    make("file")\n    make = str',
            'from pathlib import Path\nmake = Path; make("file"); make = str',
            'import os\nhelper = os.path.normpath\nalias = helper\nhelper = str\nalias("file")',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(policy.source_violations(source))

    def test_generated_fstring_worker_source_is_checked(self):
        source = '''
self.write(f"test_{name}.py", f"""
    import pathlib
    root = pathlib.Path({directory!r})
    (root / {name!r}).write_text('ready')
    while not (root / {other!r}).exists():
        pass
""")
'''
        problems = policy.source_violations(source)
        generated = [problem for problem in problems if ':generated:' in problem]
        self.assertEqual(len(generated), 3)
        self.assertEqual(sum('division' in problem for problem in generated), 2)

    def test_exec_and_literal_python_file_writers_are_checked(self):
        for source in [
                'exec("from pathlib import Path; Path(name)")',
                'compile("import os; os.path.normpath(name)", "probe", "exec")',
                '(root / "probe.py").write_text("import os; os.path.normcase(name)")']:
            with self.subTest(source=source):
                self.assertTrue(any(':generated:' in problem
                                    for problem in policy.source_violations(source)))

    def test_python_c_command_source_follows_interpreter_and_source_aliases(self):
        source = '''
import sys as system
interpreter = system.executable
code = 'from pathlib import Path as P; P(name)'
captured = code
code = 'print("not the captured program")'
command = [interpreter, '-I', '-c', captured]
'''
        self.assertTrue(any(':generated:' in problem
                            for problem in policy.source_violations(source)))
        self.assertEqual(policy.source_violations(
            'command = ["bash", "-c", "not Python / data"]'), [])

    def test_generated_source_bindings_do_not_leak_between_functions(self):
        source = '''
import sys
def first():
    code = 'import os; os.path.normpath(name)'
    return [sys.executable, '-c', code]
def second():
    code = 'print("ordinary fixture")'
    return [sys.executable, '-c', code]
'''
        problems = policy.source_violations(source)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('normpath()', problems[0])
        self.assertIn(':5:generated:', problems[0])

    def test_raw_operation_names_and_enumeration_data_are_not_rewritten(self):
        source = '''
import os
with os.scandir(root) as entries:
    for entry in entries:
        with open(entry.path) as stream:
            data = stream.read()
with open('link/../file') as stream:
    data = stream.read()
'''
        self.assertEqual(policy.source_violations(source), [])

    def test_parseable_non_python_file_data_is_not_executable_source(self):
        self.assertEqual(policy.source_violations('file.write_text("a/b")'), [])

    def test_parse_errors_are_not_silent_success(self):
        self.assertIn('could not parse Python', policy.source_violations('def broken(')[0])


class RepositoryPathPolicyTests(unittest.TestCase):
    def test_first_party_sources_follow_filename_policy(self):
        self.assertEqual(policy.main(), 0)
