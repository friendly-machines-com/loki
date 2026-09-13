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
import tempfile
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


def _sources(root=AGENT):
    """Every module under ``root`` the guard must read, at any depth.

    A top-level ``glob("*.py")`` scanned only the modules that happen to sit
    directly in the package today, so the first module placed under a
    subpackage would import the editor's code with nobody checking; scanning
    the whole tree keeps the guard's reach equal to the package's shape.
    """
    return sorted(root.rglob("*.py"))


def _module_name(path, root=AGENT):
    """Dotted name of ``path`` under ``root``, with ``__init__`` folded away.

    ``path.stem`` named every nested module by its bare filename, which
    collides across subpackages and hides where an offender lives; the
    relative dotted name is the module a reader has to open.
    """
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or "__init__"


def _imported_siblings(tree):
    """Sibling module names imported by ``tree`` (package-relative or absolute).

    The package-only absolute form may also surface attribute names; only the
    ``EDITOR_ONLY`` module names are ever looked up, so an attribute can never
    turn into an offender -- but it means this is not a list of modules alone.
    """
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                found.add(node.module.split(".")[0])
            elif node.level:
                found.update(alias.name for alias in node.names)
            elif node.module == "loki_agent":
                # ``from loki_agent import windows_setup`` has no trailing dot,
                # so the branch below never matched it and the editor's code
                # could enter the chat's bundle unseen.  None of the callers
                # need the aliases to be real submodules: the ``EDITOR_ONLY``
                # lookup decides, and those names are real either way.
                found.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith("loki_agent."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # A bare ``import loki_agent`` binds the package only and
                # reaches no submodule, so there is nothing to record here.
                if alias.name.startswith("loki_agent."):
                    found.add(alias.name.split(".")[1])
    return found


def _import_offenders(root=AGENT):
    """Modules under ``root`` that import an editor-only module.

    Kept out of the test so a synthetic package can drive the same check: the
    shapes this guard must reject are otherwise only provable by editing the
    real package, which is exactly the mistake the guard exists to prevent.
    """
    offenders = []
    for path in _sources(root):
        module = _module_name(path, root)
        for imported in _imported_siblings(ast.parse(path.read_text())):
            allowed = EDITOR_ONLY.get(imported)
            if allowed is not None and module not in allowed:
                offenders.append(f"{module} imports {imported}")
    return offenders


def _defined_functions(tree):
    return {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class ImportBoundaryTests(unittest.TestCase):
    def test_only_the_editor_imports_the_editor_only_modules(self):
        offenders = _import_offenders()
        self.assertEqual(
            offenders, [],
            "the chat's bundle would gain container-mutation code: "
            + ", ".join(offenders))

    def test_the_chat_never_imports_the_editor(self):
        self.assertNotIn("windows_setup",
                         _imported_siblings(ast.parse(
                             (AGENT / "loki.py").read_text())))


class ImportShapeTests(unittest.TestCase):
    """Regression tests for the import spellings the guard used to miss.

    Each builds a throwaway package and runs the real offender pass over it,
    so these fail when the matcher or the scan regresses rather than when the
    real tree happens to change.
    """

    def offenders(self, sources):
        with tempfile.TemporaryDirectory() as root:
            package = pathlib.Path(root) / "loki_agent"
            package.mkdir()
            (package / "__init__.py").write_text("")
            for name, text in sources.items():
                target = package / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            return _import_offenders(package)

    def test_package_only_absolute_import_is_seen(self):
        # The exact hole: level 0, module "loki_agent", no trailing dot.
        offenders = self.offenders(
            {"chat.py": "from loki_agent import windows_containers\n"})

        self.assertEqual(offenders, ["chat imports windows_containers"])

    def test_a_nested_module_is_scanned(self):
        # The old top-level glob skipped this file entirely.
        offenders = self.offenders(
            {"pkg/deep.py": "from loki_agent import windows_setup\n"})

        self.assertEqual(offenders, ["pkg.deep imports windows_setup"])

    def test_a_package_attribute_is_not_reported(self):
        # ``__version__`` is not in EDITOR_ONLY, so the widened branch must
        # stay quiet about attributes that merely ride the same import form.
        offenders = self.offenders(
            {"chat.py": "from loki_agent import __version__\n"})

        self.assertEqual(offenders, [])

    def test_the_editor_may_import_its_own_mutation(self):
        # The one permitted importer must keep passing after the widening.
        offenders = self.offenders(
            {"windows_setup.py":
             "from loki_agent import windows_containers\n"})

        self.assertEqual(offenders, [])


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
