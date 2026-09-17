#!/usr/bin/env python3
"""Fail on any textual path helper: a name is not an identity.

Banned: ``os.path.abspath``, ``normpath``, ``normcase``, ``expanduser``,
``realpath``, ``dirname``, ``basename``, and any ``.resolve()``.

Each of those answers "what is this string?", never "is this the same
object?" -- so every use is either a no-op or a bug that waits for a platform
where the two differ.  Identity comes from ``stat``/``open``; navigation comes
from ``Path.parent``/``Path.name``.  There is no allowlist and no baseline:
delete the call.

Run:  python3 tests/lint_path_functions.py
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

BANNED = (
    "abspath", "normpath", "normcase", "expanduser",
    "realpath", "dirname", "basename",
)
ROOT = Path(__file__).parent.parent
SOURCES = (ROOT / "loki_agent", ROOT / "tests")


def violations(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in BANNED:
            found.append(f"{path}:{node.lineno}: {func.id}()")
        elif isinstance(func, ast.Attribute):
            if (func.attr in BANNED and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "path"):
                found.append(f"{path}:{node.lineno}: os.path.{func.attr}()")
            elif func.attr in BANNED:
                found.append(f"{path}:{node.lineno}: .{func.attr}()")
            elif func.attr == "resolve":
                found.append(f"{path}:{node.lineno}: .resolve()")
    return found


def python_files() -> list[Path]:
    files = [path for base in SOURCES for path in base.rglob("*.py")
             if "__pycache__" not in path.parts]
    files += sorted(ROOT.glob("*.py"))
    return files


def main() -> int:
    problems = [line for path in python_files()
                for line in violations(path)]
    for line in problems:
        print(line)
    if problems:
        print(f"\n{len(problems)} banned path-helper call(s); a name is not an "
              f"identity", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
