#!/usr/bin/env python3
"""Check literal filenames without a grandfathered baseline or allowlist.

Check prohibited helpers, Path construction and statically known path
operators, including statically available emitted Python. Unknown arithmetic
is not declared to be filesystem navigation. This syntactic/provenance gate is
not a whole-program Python type proof or a substitute for per-site review.

Run directly, or through test_path_functions in the normal test driver.
"""

from __future__ import annotations

import ast
import os
import sys


BANNED = {
    "abspath", "normpath", "normcase", "expanduser",
    "realpath", "dirname", "basename", "resolve",
}
CONSTRUCTORS = {"Path", "PurePath", "PosixPath", "WindowsPath",
                "PurePosixPath", "PureWindowsPath"}
# The checker can be invoked outside the checkout's cwd. Its own absolute
# module filename selects the tests directory and repository scope; these two
# parent views locate source to inspect, never canonicalize an operation operand.
TESTS, _ = os.path.split(__file__)
ROOT, _ = os.path.split(TESTS)


def _kind(node, bindings, assignments):
    if isinstance(node, ast.Name):
        return bindings.get(node.id, 'helper:' + node.id if node.id in BANNED else None)
    if isinstance(node, ast.Attribute):
        owner = _kind(node.value, bindings, assignments)
        if owner == 'sys' and node.attr == 'executable':
            return 'python-image'
        if (owner == 'pathlib' and node.attr in CONSTRUCTORS
                or owner == 'constructor' and node.attr in ['cwd', 'home']):
            return 'constructor'
        if node.attr in BANNED:
            return 'helper:' + node.attr
        if node.attr == 'joinpath' or (owner == 'path-value' and node.attr in [
                '__truediv__', '__rtruediv__', '__itruediv__']):
            return 'navigation'
        if owner == 'operator' and node.attr in ['truediv', 'itruediv']:
            return 'division'
        if owner == 'importlib' and node.attr == 'import_module':
            return 'importer'
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if any(_kind(value, bindings, assignments) == 'path-value' for value in [node.left, node.right]):
            return 'path-value'
    if isinstance(node, ast.Call):
        if _kind(node.func, bindings, assignments) in ['constructor', 'navigation']:
            return 'path-value'
        if (isinstance(node.func, ast.Name) and node.func.id == 'getattr'
                and len(node.args) == 2):
            name = _text(node.args[1], assignments)
            if isinstance(name, str):
                return _kind(ast.Attribute(value=node.args[0], attr=name), bindings, assignments)
        if node.args and (isinstance(node.func, ast.Name) and node.func.id == '__import__'
                          or _kind(node.func, bindings, assignments) == 'importer'):
            module = node.args[0]
            if isinstance(module, ast.Constant):
                if module.value in ['os', 'pathlib', 'operator']:
                    return module.value
    return None


def _text(node, assignments, seen=None):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        # A placeholder preserves the emitted Python's token boundaries. It
        # does not execute model/fixture interpolation while checking source.
        return ''.join(part.value if isinstance(part, ast.Constant)
                       else '__generated_value__' for part in node.values)
    if isinstance(node, ast.Name) and node.id in assignments:
        seen = set() if seen is None else seen
        if node.id not in seen:
            return _text(assignments[node.id], assignments, seen | {node.id})
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _text(node.left, assignments, seen)
        right = _text(node.right, assignments, seen)
        if left is not None and right is not None:
            return left + right
    return None


def _python_filename(node, assignments, seen=None):
    text = _text(node, assignments)
    if text is not None:
        return text.endswith('.py')
    if isinstance(node, ast.Name) and node.id in assignments:
        seen = set() if seen is None else seen
        if node.id not in seen:
            return _python_filename(assignments[node.id], assignments, seen | {node.id})
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _python_filename(node.right, assignments, seen)
    if isinstance(node, ast.Call) and node.args:
        return _python_filename(node.args[-1], assignments, seen)
    return False


def _scope_nodes(tree):
    nodes = []
    nested = []

    def visit(node):
        nodes.append(node)
        if (isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)
                or isinstance(node, ast.ClassDef)):
            nested.append(node)
            # Defaults, annotations and decorators belong to the containing
            # scope. A function/class body owns its own local bindings.
            for field, value in ast.iter_fields(node):
                if field != 'body':
                    if isinstance(value, ast.AST):
                        visit(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, ast.AST):
                                visit(item)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child)

    for statement in tree.body:
        visit(statement)
    return nodes, nested


def source_violations(source: str, filename='<source>') -> list[str]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        return [f'{filename}:{error.lineno}: could not parse Python: {error.msg}']
    return _scope_violations(tree, source, filename, {}, {})


def _scope_violations(tree, source, filename, outer_bindings, outer_assignments):
    bindings = dict(outer_bindings)
    assignments = dict(outer_assignments)
    nodes, nested = _scope_nodes(tree)
    local_names = set()
    if isinstance(tree, ast.FunctionDef) or isinstance(tree, ast.AsyncFunctionDef):
        for argument in tree.args.posonlyargs + tree.args.args + tree.args.kwonlyargs:
            local_names.add(argument.arg)
        for argument in [tree.args.vararg, tree.args.kwarg]:
            if argument is not None:
                local_names.add(argument.arg)
    for node in nodes:
        if isinstance(node, ast.Assign) or isinstance(node, ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            local_names.update(target.id for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.Import):
            local_names.update(item.asname or item.name.split('.')[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            local_names.update(item.asname or item.name for item in node.names)
    for name in local_names:
        bindings.pop(name, None)
        assignments.pop(name, None)
    if isinstance(tree, ast.FunctionDef) or isinstance(tree, ast.AsyncFunctionDef):
        for argument in tree.args.posonlyargs + tree.args.args + tree.args.kwonlyargs:
            if _kind(argument.annotation, outer_bindings, outer_assignments) == 'constructor':
                bindings[argument.arg] = 'path-value'
    found = []
    emitted = set()
    events = []
    for node in nodes:
        if isinstance(node, ast.Import | ast.ImportFrom | ast.Assign | ast.AnnAssign
                      | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            events.append([node.end_lineno, node.end_col_offset, 1, node])
        if isinstance(node, ast.Call | ast.BinOp | ast.AugAssign | ast.List | ast.Tuple):
            events.append([node.lineno, node.col_offset, 0, node])
    # Check uses before applying an enclosing assignment. Captured alias kinds
    # and literal text belong to that source position; a later reassignment
    # must not erase an earlier call or change already-captured emitted source.
    for _line, _column, binding, node in sorted(events, key=lambda event: event[:3]):
        if binding:
            if isinstance(node, ast.Import):
                for item in node.names:
                    name = item.asname or item.name.split('.')[0]
                    bindings[name] = item.name
                    assignments.pop(name, None)
            elif isinstance(node, ast.ImportFrom):
                for item in node.names:
                    name = item.asname or item.name
                    kind = None
                    if node.module == 'sys' and item.name == 'executable':
                        kind = 'python-image'
                    elif node.module == 'pathlib' and item.name == '*':
                        for constructor in CONSTRUCTORS:
                            bindings[constructor] = 'constructor'
                            assignments.pop(constructor, None)
                    elif node.module == 'pathlib' and item.name in CONSTRUCTORS:
                        kind = 'constructor'
                    elif node.module in ['os.path', 'posixpath', 'ntpath'] and item.name in BANNED:
                        kind = 'helper:' + item.name
                    elif node.module == 'operator' and item.name in ['truediv', 'itruediv']:
                        kind = 'division'
                    bindings[name] = kind
                    assignments.pop(name, None)
            elif isinstance(node, ast.Assign) or isinstance(node, ast.AnnAssign):
                if node.value is not None:
                    kind = _kind(node.value, bindings, assignments)
                    text = _text(node.value, assignments)
                    value = node.value if text is None else ast.Constant(value=text)
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            bindings[target.id] = kind
                            assignments[target.id] = value
            else:
                bindings[node.name] = None
                assignments.pop(node.name, None)
            continue
        problem = None
        generated = None
        if ((isinstance(node, ast.List) or isinstance(node, ast.Tuple))
                and node.elts and _kind(node.elts[0], bindings, assignments) == 'python-image'):
            for index, argument in enumerate(node.elts[:-1]):
                if isinstance(argument, ast.Constant) and argument.value == '-c':
                    generated = _text(node.elts[index + 1], assignments)
                    break
        if ((isinstance(node, ast.BinOp) or isinstance(node, ast.AugAssign))
                and isinstance(node.op, ast.Div)):
            operands = ([node.left, node.right] if isinstance(node, ast.BinOp)
                        else [node.target, node.value])
            if any(_kind(value, bindings, assignments) == 'path-value' for value in operands):
                problem = 'path division'
        elif isinstance(node, ast.Call):
            kind = _kind(node.func, bindings, assignments)
            if kind == 'constructor':
                problem = 'Path construction'
            elif kind == 'navigation':
                problem = 'path navigation/operator'
            elif kind is not None and kind.startswith('helper:'):
                problem = kind[len('helper:'):] + '()'
            elif kind == 'division':
                if any(_kind(argument, bindings, assignments) == 'path-value' for argument in node.args):
                    problem = 'path division'
            elif isinstance(node.func, ast.Name) and node.func.id in BANNED:
                problem = node.func.id + '()'
            callbacks = (node.args[:1] if isinstance(node.func, ast.Name)
                         and node.func.id in ['map', 'filter'] else [])
            callbacks += [keyword.value for keyword in node.keywords
                          if (keyword.arg == 'type' and isinstance(node.func, ast.Attribute)
                              and node.func.attr == 'add_argument')
                          or (keyword.arg == 'key' and isinstance(node.func, ast.Name)
                              and node.func.id in ['sorted', 'min', 'max'])]
            if any(_kind(callback, bindings, assignments) == 'constructor' for callback in callbacks):
                problem = 'indirect Path construction'
            if (isinstance(node.func, ast.Name) and node.func.id in ['exec', 'eval', 'compile']
                    and node.args):
                generated = _text(node.args[0], assignments)
            elif isinstance(node.func, ast.Attribute):
                if (node.func.attr == 'write_text' and node.args
                        and _python_filename(node.func.value, assignments)):
                    generated = _text(node.args[0], assignments)
                elif node.func.attr == 'write' and len(node.args) >= 2:
                    name = _text(node.args[0], assignments)
                    if name is not None and name.endswith('.py'):
                        generated = _text(node.args[1], assignments)
        if problem == 'expanduser()':
            # Source-authored expansion is intentional; dynamic filenames are not.
            operands = node.args + [keyword.value for keyword in node.keywords if keyword.arg == 'path']
            if (len(operands) == 1 and len(node.args) + len(node.keywords) == 1
                    and isinstance(operands[0], ast.Constant)
                    and isinstance(operands[0].value, str)):
                problem = None
        if problem:
            found.append(f'{filename}:{node.lineno}: {problem}')
        if generated is not None and generated not in emitted and generated != source:
            emitted.add(generated)
            # Emitted fixtures commonly indent whole modules for readability.
            import textwrap
            found.extend(source_violations(
                textwrap.dedent(generated), f'{filename}:{node.lineno}:generated'))
    for child in nested:
        # Methods do not close over their class dictionary; enclosing function
        # locals and module globals remain visible to them.
        inherited_bindings = outer_bindings if isinstance(tree, ast.ClassDef) else bindings
        inherited_assignments = outer_assignments if isinstance(tree, ast.ClassDef) else assignments
        found.extend(_scope_violations(
            child, source, filename, inherited_bindings, inherited_assignments))
    return found


def violations(path: str) -> list[str]:
    try:
        with open(path, encoding='utf-8') as stream:
            return source_violations(stream.read(), path)
    except (OSError, UnicodeError) as error:
        return [f'{path}: could not read Python: {error}']


def python_files() -> list[str]:
    files = []

    def visit(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False) and entry.name != '__pycache__':
                    visit(entry.path)
                elif entry.name.endswith('.py') and entry.is_file():
                    files.append(entry.path)

    visit(os.path.join(ROOT, 'loki_agent'))
    visit(TESTS)
    with os.scandir(ROOT) as entries:
        for entry in entries:
            if (entry.name.endswith('.py') or entry.name.endswith('.spec')
                    or entry.name in ['loki-acp', 'loki-setup']) and entry.is_file():
                files.append(entry.path)
    return sorted(files)


def main() -> int:
    problems = [line for path in python_files() for line in violations(path)]
    for line in problems:
        print(line)
    if problems:
        print(f'\n{len(problems)} filename-policy violation(s)', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
