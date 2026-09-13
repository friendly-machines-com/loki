"""Keeps the chat/editor module boundary from eroding.

PyInstaller bundles what is reachable from each entry point, so the split is
only real while the import graph stays put; a stray ``import windows_containers``
in a module the chat reaches would quietly undo it.  This is a maintainability
guard, not a security boundary -- see ``windows_setup`` for why -- so it checks
only that the layout still expresses the intent, and claims nothing more.

Portable on purpose: this parses source, so it runs on the machine the mistake
would be made on, not only on Windows.
"""

import ast
import pathlib
import unittest

AGENT = pathlib.Path(__file__).resolve().parent.parent / "loki_agent"

# Modules that carry the editor's capabilities.  Who may import them:
#   * windows_setup is the editor itself; nothing the chat reaches may import it.
#   * windows_containers performs the OS mutation; only the editor may import it.
EDITOR_ONLY = {
    "windows_setup": set(),
    "windows_containers": {"windows_setup"},
}

# The calls that create a profile or rewrite a DACL.  ``derive_app_container_sid``
# and ``dacl_sddl`` are deliberately absent: they only read, and the chat's
# verification needs them.
MUTATION_NAMES = {
    "set_dacl_sddl",
    "create_app_container_profile",
    "delete_app_container_profile",
}

MUTATION_MODULES = {"windows_containers"}


def _sources():
    return sorted(AGENT.glob("*.py"))


def _imported_siblings(tree):
    """Sibling module names imported by ``tree`` (package-relative or absolute)."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                found.add(node.module.split(".")[0])
            elif node.level:
                found.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith("loki_agent."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("loki_agent."):
                    found.add(alias.name.split(".")[1])
    return found


def _defined_functions(tree):
    return {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class ImportBoundaryTests(unittest.TestCase):
    def test_only_the_editor_imports_the_editor_only_modules(self):
        offenders = []
        for path in _sources():
            module = path.stem
            for imported in _imported_siblings(ast.parse(path.read_text())):
                allowed = EDITOR_ONLY.get(imported)
                if allowed is not None and module not in allowed:
                    offenders.append(f"{module} imports {imported}")
        self.assertEqual(
            offenders, [],
            "the chat's bundle would gain container-mutation code: "
            + ", ".join(offenders))

    def test_the_chat_never_imports_the_editor(self):
        self.assertNotIn("windows_setup",
                         _imported_siblings(ast.parse(
                             (AGENT / "loki.py").read_text())))


class FunctionBoundaryTests(unittest.TestCase):
    def test_mutation_lives_only_in_the_editor_modules(self):
        offenders = []
        for path in _sources():
            module = path.stem
            if module in MUTATION_MODULES:
                continue
            leaked = _defined_functions(ast.parse(path.read_text())) & MUTATION_NAMES
            offenders.extend(f"{module} defines {name}" for name in sorted(leaked))
        self.assertEqual(
            offenders, [],
            "read-only modules must not define container mutation: "
            + ", ".join(offenders))

    def test_the_read_api_has_no_mutation_left(self):
        # Regression guard for the split: the mutation was once duplicated here,
        # which kept it in the chat's bundle even though nothing called it.
        defined = _defined_functions(ast.parse(
            (AGENT / "windows_api.py").read_text()))
        self.assertEqual(defined & MUTATION_NAMES, set())


if __name__ == "__main__":
    unittest.main()
