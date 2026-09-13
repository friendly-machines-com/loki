"""Windows setup editor: the per-workspace container definition.

Windows-only, and deliberately a separate process from the chat:

* The process that holds credentials and drives the model loop must not also
  contain the code that creates profiles and rewrites ACLs.  Keeping mutation
  here means the runtime has no code path that can widen its own sandbox, and
  Tk -- a large C surface -- never loads in the credential-holding process.
* Grants are a **setup-time** property.  There is no runtime grant path:
  widening a container means editing it here and re-running, exactly as a
  bubblewrap invocation cannot be widened after ``unshare``.

Nothing security-relevant crosses the process boundary: the editor writes the
ledger (``<state>/windows-setup.json``) and exits, and the caller re-derives the
package SID and re-runs verification itself rather than trusting this process'
output or its identity.

Two rules this module enforces rather than documents:

* Paths that would invalidate the property are **errors**: anything covering
  Loki's credential directory, its configuration/state trees, or its own
  runtime tree.  We are the only ones who can know those, so refusing is our
  job.
* Conventional secret locations a grant would cover are **warnings**, and only
  when we can check that they exist.  We are not the user's police, and a
  generic "are you sure?" trains them to click through the ones that matter.
"""

from __future__ import annotations

import contextlib
import enum
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Protocol

from . import paths
from . import windows_api


LEDGER_VERSION = 1
PROFILE_NAME_PREFIX = "Loki.Workspace."


class Access(enum.Enum):
    """The two grant levels.  Read-only is the default for added paths.

    There is deliberately no separate "read and execute": as a security
    distinction it is theater -- anything readable can be copied somewhere
    writable and run -- so the boundary is confidentiality (what can be read)
    and integrity/persistence (what can be written).
    """

    READ = "read"
    READ_WRITE = "read-write"


def access_sddl(access: Access) -> str:
    """Return the DACL rights for a level.

    ``FRFX`` is generic read plus generic execute (execute on a directory is
    traverse, which known-path access needs).  Read-write is ``0x1301BF`` --
    the documented "Modify" mask, read/write/execute plus delete -- rather than
    ``FA``, because ``FA`` would also hand over ``WRITE_DAC`` and let a tool
    rewrite the DACL and lock the owner out of their own directory.
    """
    return "FRFX" if access is Access.READ else "0x1301BF"


# -- path rules ----------------------------------------------------------

def _runtime_trees() -> list[str]:
    """Trees the container must never be granted."""
    trees = [os.path.dirname(os.path.abspath(__file__)),
             paths.loki_config_dir(),
             paths.loki_state_dir(),
             paths.credential_directory()]
    if getattr(sys, "frozen", False):
        # A packaged build's own directory holds the runtime and its bundled
        # interpreter; a grant covering it would let a tool rewrite Loki.
        trees.append(os.path.dirname(os.path.abspath(sys.executable)))
    return trees


def protected_path_errors(path: str) -> list[str]:
    """Return the reasons ``path`` may not be granted (empty when fine).

    ``path`` is refused when it *contains* a protected tree (a recursive grant
    would cover it) or when it *sits inside* one.
    """
    target = os.path.normcase(os.path.abspath(path))
    reasons = []
    for tree in _runtime_trees():
        protected = os.path.normcase(os.path.abspath(tree))
        if target == protected:
            reasons.append(f"{path} is {tree}, which Loki never grants")
        elif _contains(target, protected):
            reasons.append(
                f"{path} contains {tree}, which Loki never grants")
        elif _contains(protected, target):
            reasons.append(f"{path} is inside {tree}, which Loki never grants")
    return reasons


def _contains(parent: str, child: str) -> bool:
    """Whether normalized ``child`` lies strictly under ``parent``."""
    return child.startswith(parent.rstrip(os.sep) + os.sep)


# Conventional secret locations, checked for existence before warning: we only
# speak up about what we can see, and never refuse on the user's behalf.
SECRET_HINTS = (
    ".ssh", ".aws", ".azure", ".gnupg", ".netrc", ".git-credentials",
    os.path.join(".config", "gcloud"),
    os.path.join("AppData", "Roaming", "Microsoft", "Credentials"),
    os.path.join("AppData", "Local", "Microsoft", "Credentials"),
)


def covered_secret_warnings(path: str) -> list[str]:
    """Return warnings for existing secret locations a grant would cover."""
    profile = os.path.expanduser("~")
    target = os.path.normcase(os.path.abspath(path))
    warnings = []
    for hint in SECRET_HINTS:
        candidate = os.path.join(profile, hint)
        if not os.path.exists(candidate):
            continue
        if target == os.path.normcase(os.path.abspath(candidate)) or _contains(
                target, os.path.normcase(os.path.abspath(candidate))):
            warnings.append(
                f"{path} also covers {candidate}, which exists")
    return warnings


# -- names ---------------------------------------------------------------

def canonical_workspace(path: str) -> str:
    """Canonical form used as the workspace key.

    Naming only: it folds case and resolves reparse points so one directory has
    one profile, and it is never used to rewrite the operand itself.
    """
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def profile_name_for(workspace: str) -> str:
    """Derive the AppContainer profile name for a workspace.

    Derived, not stored: names are case-insensitive and
    ``DeriveAppContainerSidFromAppContainerName`` is deterministic, so the SID
    can be computed before anything is created.
    """
    digest = hashlib.sha256(canonical_workspace(workspace).encode(
        "utf-8", "surrogatepass")).hexdigest()
    return PROFILE_NAME_PREFIX + digest[:32]


# -- definitions and the ledger -----------------------------------------

@dataclass(frozen=True)
class Grant:
    path: str
    access: Access
    origin: str  # 'workspace' | 'toolchain' | 'temp' | 'user'


@dataclass
class Definition:
    workspace: str
    grants: list[Grant] = field(default_factory=list)


def ledger_path() -> str:
    return os.path.join(paths.loki_state_dir(), "windows-setup.json")


def load_ledger(path: str | None = None) -> dict:
    """Load the ledger, returning an empty one when absent or unreadable."""
    location = ledger_path() if path is None else path
    try:
        with open(location, "r", encoding="utf-8") as stream:
            blob = json.load(stream)
    except (OSError, ValueError):
        return {"version": LEDGER_VERSION, "workspaces": {}}
    if not isinstance(blob, dict) or not isinstance(
            blob.get("workspaces"), dict):
        return {"version": LEDGER_VERSION, "workspaces": {}}
    blob.setdefault("version", LEDGER_VERSION)
    return blob


def save_ledger(blob: dict, path: str | None = None) -> None:
    """Atomically replace the ledger.

    The write is local rather than reusing the agent core's helper because this
    module must stay importable before anything decides about the runtime.
    """
    location = ledger_path() if path is None else path
    directory = os.path.dirname(location) or "."
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=".setup-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(blob, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, location)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def workspace_key(workspace: str) -> str:
    return canonical_workspace(workspace)


def ledger_entry(workspace: str, profile: str,
                 grants: list[Grant]) -> dict:
    return {
        "workspace": os.path.abspath(workspace),
        "profile": profile,
        "grants": [
            {"path": grant.path, "access": grant.access.value,
             "origin": grant.origin}
            for grant in grants
        ],
    }


def entry_grants(entry: dict) -> list[Grant]:
    grants = []
    for item in entry.get("grants", []):
        try:
            access = Access(item["access"])
        except (KeyError, ValueError):
            continue
        grants.append(Grant(str(item.get("path", "")), access,
                            str(item.get("origin", "user"))))
    return grants


# -- SDDL editing --------------------------------------------------------
# User directories keep their existing DACL: we *append* our ACE (and remove
# exactly it again), because replacing the DACL of someone's project directory
# would be repairing their permissions, which this project does not do.

def add_package_ace(sddl: str, package: str, access: Access) -> str:
    """Return ``sddl`` with an inheritable ACE for ``package`` appended."""
    ace = f"(A;OICI;{access_sddl(access)};;;{package})"
    index = sddl.find("(")
    if index < 0:
        if not sddl.startswith("D:"):
            raise windows_api.WindowsApiError(f"not a DACL: {sddl!r}")
        return sddl + ace
    return sddl[:index] + sddl[index:] + ace


def remove_package_aces(sddl: str, package: str) -> str:
    """Return ``sddl`` with every ACE naming ``package`` removed."""
    marker = f";;;{package})"
    result = []
    index = 0
    while True:
        start = sddl.find("(", index)
        if start < 0:
            result.append(sddl[index:])
            return "".join(result)
        result.append(sddl[index:start])
        end = sddl.find(")", start)
        if end < 0:
            result.append(sddl[start:])
            return "".join(result)
        ace = sddl[start:end + 1]
        if not ace.endswith(marker):
            result.append(ace)
        index = end + 1


def grants_package(sddl: str, package: str) -> bool:
    """Whether ``sddl`` names ``package`` at all."""
    return f";;;{package})" in sddl


# -- plan and diff -------------------------------------------------------

@dataclass(frozen=True)
class Change:
    kind: str  # 'add' | 'remove' | 'level'
    path: str
    access: Access


@dataclass
class Plan:
    workspace: str
    profile: str
    changes: list[Change]
    warnings: list[str]
    errors: list[str]

    @property
    def applicable(self) -> bool:
        return not self.errors


def build_plan(definition: Definition, previous: dict | None) -> Plan:
    """Diff a definition against the recorded one, with rules applied."""
    errors: list[str] = []
    warnings: list[str] = []
    for grant in definition.grants:
        errors.extend(protected_path_errors(grant.path))
        if grant.origin == "user":
            warnings.extend(covered_secret_warnings(grant.path))

    current = {canonical_workspace(g.path): g for g in definition.grants}
    recorded = {canonical_workspace(g.path): g
                for g in entry_grants(previous or {})}
    changes = []
    for key, grant in sorted(current.items()):
        if key not in recorded:
            changes.append(Change("add", grant.path, grant.access))
        elif recorded[key].access is not grant.access:
            changes.append(Change("level", grant.path, grant.access))
    for key, grant in sorted(recorded.items()):
        if key not in current:
            changes.append(Change("remove", grant.path, grant.access))
    return Plan(definition.workspace,
                profile_name_for(definition.workspace),
                changes, warnings, errors)


# -- platform operations -------------------------------------------------

@dataclass(frozen=True)
class Check:
    name: str
    status: str  # 'pass' | 'fail' | 'untested'
    detail: str = ""


class Backend(Protocol):
    def apply(self, plan: Plan, definition: Definition) -> list[Check]: ...
    def verify(self, definition: Definition) -> list[Check]: ...
    def uninstall(self, blob: dict) -> list[Check]: ...


class WindowsBackend:
    """Applies the plan through the shared Windows declarations."""

    def __init__(self, ledger: dict, ledger_location: str | None = None):
        self.ledger = ledger
        self.ledger_location = ledger_location

    # -- helpers --
    def _ensure_profile(self, name: str) -> str:
        """Create the profile, or derive the SID when it already exists.

        An existing profile is not a failure: the package SID is a
        deterministic function of the name, so derive it and carry on.  Any
        other creation failure is real and propagates.
        """
        if not windows_api.app_container_profile_name_is_usable(name):
            raise windows_api.WindowsApiError(
                f"generated profile name is not usable: {name!r}")
        try:
            return windows_api.create_app_container_profile(name)
        except windows_api.WindowsApiError as error:
            if error.status != windows_api.PROFILE_ALREADY_EXISTS:
                raise
            return windows_api.derive_app_container_sid(name)

    def _grant_path(self, path: str, access: Access, package: str) -> None:
        current = windows_api.dacl_sddl(path)
        windows_api.set_dacl_sddl(
            path, add_package_ace(current, package, access))

    def _ungrant_path(self, path: str, package: str) -> None:
        current = windows_api.dacl_sddl(path)
        if grants_package(current, package):
            windows_api.set_dacl_sddl(
                path, remove_package_aces(current, package))

    def _private_directory(self, path: str, user: str) -> None:
        os.makedirs(path, exist_ok=True)
        windows_api.set_dacl_sddl(path, private_dacl_sddl(user))

    # -- Backend --
    def apply(self, plan: Plan, definition: Definition) -> list[Check]:
        checks: list[Check] = []
        user = windows_api.current_user_sid()
        self._private_directory(paths.credential_directory(), user)
        self._private_directory(paths.loki_config_dir(), user)
        self._private_directory(paths.loki_state_dir(), user)
        try:
            package = self._ensure_profile(plan.profile)
        except windows_api.WindowsApiError as error:
            # Without a profile there is nothing to grant to; stop rather than
            # record grants that name a SID we could not establish.
            return checks + [Check("profile", "fail", str(error))]
        checks.append(Check("profile", "pass", package))
        for grant in definition.grants:
            self._grant_path(grant.path, grant.access, package)
            checks.append(Check(f"grant {grant.path}", "pass",
                                grant.access.value))
        entry = ledger_entry(definition.workspace, plan.profile,
                             definition.grants)
        self.ledger.setdefault("workspaces", {})[
            workspace_key(definition.workspace)] = entry
        save_ledger(self.ledger, self.ledger_location)
        checks.append(Check("ledger", "pass", ledger_path()))
        return checks

    def verify(self, definition: Definition) -> list[Check]:
        profile = profile_name_for(definition.workspace)
        checks: list[Check] = []
        try:
            package = windows_api.derive_app_container_sid(profile)
            checks.append(Check("profile SID", "pass", package))
        except windows_api.WindowsApiError as error:
            return [Check("profile SID", "fail", str(error))]
        for grant in definition.grants:
            sddl = windows_api.dacl_sddl(grant.path)
            if grants_package(sddl, package):
                checks.append(Check(f"grant {grant.path}", "pass",
                                    grant.access.value))
            else:
                checks.append(Check(f"grant {grant.path}", "fail",
                                    "no ACE for the package SID"))
        for tree, name in ((paths.credential_directory(), "credentials"),
                           (paths.loki_config_dir(), "config"),
                           (paths.loki_state_dir(), "state")):
            try:
                sddl = windows_api.dacl_sddl(tree)
            except windows_api.WindowsApiError as error:
                checks.append(Check(name, "fail", str(error)))
                continue
            if grants_package(sddl, package):
                checks.append(Check(name, "fail", "package SID is granted"))
            else:
                checks.append(Check(name, "pass", "no package ACE"))
        # The token self-test is the only check that proves the container can
        # actually be entered; until it exists, say so instead of implying it
        # passed.
        checks.append(Check("contained probe", "untested",
                            "token self-test not implemented yet"))
        return checks

    def uninstall(self, blob: dict) -> list[Check]:
        checks: list[Check] = []
        for key, entry in sorted(blob.get("workspaces", {}).items()):
            profile = str(entry.get("profile", ""))
            try:
                package = windows_api.derive_app_container_sid(profile)
            except windows_api.WindowsApiError as error:
                checks.append(Check(f"profile {key}", "fail", str(error)))
                package = None
            if package is not None:
                for grant in entry_grants(entry):
                    try:
                        self._ungrant_path(grant.path, package)
                        checks.append(Check(f"ungrant {grant.path}", "pass"))
                    except (windows_api.WindowsApiError, OSError) as error:
                        checks.append(Check(f"ungrant {grant.path}", "fail",
                                            str(error)))
                try:
                    windows_api.delete_app_container_profile(profile)
                    checks.append(Check(f"profile {profile}", "pass",
                                        "deleted"))
                except windows_api.WindowsApiError as error:
                    checks.append(Check(f"profile {profile}", "fail",
                                        str(error)))
        blob["workspaces"] = {}
        save_ledger(blob, self.ledger_location)
        checks.append(Check("ledger", "pass", "cleared"))
        return checks


def private_dacl_sddl(user: str) -> str:
    """A protected DACL granting only the owner, SYSTEM and Administrators.

    Used for the directories Loki creates and owns.  User directories are never
    rewritten to this: their existing DACLs are preserved and our ACE appended.
    """
    return (f"D:P(A;OICI;FA;;;{user})"
            "(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)")


class UnavailableBackend:
    """Refuses every operation: the setup tool is Windows-only."""

    def _refuse(self, *_args, **_kwargs):
        raise windows_api.WindowsUnavailableError(
            "the Windows setup editor runs on Windows only")

    apply = _refuse
    verify = _refuse
    uninstall = _refuse


def backend_for(ledger: dict) -> Backend:
    if sys.platform != "win32":
        return UnavailableBackend()
    return WindowsBackend(ledger)


# -- the editor ----------------------------------------------------------

UNINSTALL_TITLE = "Uninstall Loki"
UNINSTALL_MESSAGE = (
    "This removes the entire Loki coding agent harness:\n\n"
    "  * every recorded workspace container and its grants\n"
    "  * the AppContainer profiles Loki created\n"
    "  * the access Loki added to your directories\n"
    "  * the recorded workspace list itself\n\n"
    "Your files are not deleted. This cannot be undone from here, and any "
    "workspace will have to be set up again.\n\n"
    "Uninstall everything?"
)


def access_label(access: Access) -> str:
    return "read" if access is Access.READ else "read and write"


def automatic_grants(workspace: str) -> list[Grant]:
    """The grants Loki shares for itself, shown but not editable.

    Temporary files are not a grant: ``TEMP``/``TMP`` point inside the
    workspace, which is already granted read-write.  A scratch directory of our
    own was considered and rejected -- anything under the configuration tree
    would put a package ACE inside the tree that holds ``credentials`` and make
    it traversable, which is the same mistake the toolchain placement avoids.
    """
    toolchain = os.path.dirname(os.path.abspath(sys.executable))
    return [
        Grant(workspace, Access.READ_WRITE, "workspace"),
        Grant(toolchain, Access.READ, "toolchain"),
    ]


def definition_for(workspace: str, previous: dict | None) -> Definition:
    """Build the definition to edit: automatic grants plus recorded user ones."""
    grants = automatic_grants(workspace)
    grants.extend(entry_grants(previous or {}))
    return Definition(workspace, grants)


def _tk():
    """Import Tk lazily.

    ``--verify``, ``--list`` and ``--uninstall`` must work without loading
    tcl/tk -- and on a machine where tkinter is not installed at all -- so the
    import happens only here.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    return tk, filedialog, messagebox


class EditorModel:
    """The editor's decisions, with no widgets attached.

    Every rule the UI obeys is a return value here -- which rows are editable,
    what the pending diff says, whether there are unsaved changes -- so the
    widget layer only renders results and the rules can be tested on any
    platform.
    """

    def __init__(self, ledger: dict, workspace: str, backend: Backend):
        self.ledger = ledger
        self.backend = backend
        self.reload(workspace)

    def reload(self, workspace: str) -> None:
        """Load ``workspace``, discarding pending edits."""
        self.workspace = workspace
        self.key = workspace_key(workspace)
        self.previous = self.ledger.get("workspaces", {}).get(self.key)
        self.definition = definition_for(workspace, self.previous)
        self._saved = list(self.definition.grants)

    @property
    def grants(self) -> list[Grant]:
        return list(self.definition.grants)

    @property
    def dirty(self) -> bool:
        return self.definition.grants != self._saved

    def rows(self) -> list[str]:
        rows = []
        for grant in self.definition.grants:
            origin = "" if grant.origin == "user" else f" [{grant.origin}]"
            rows.append(f"{access_label(grant.access)}: {grant.path}{origin}")
        return rows

    def edit_refusal(self, index: int) -> str | None:
        """Why row ``index`` cannot be changed or removed, or None if it can."""
        grant = self.definition.grants[index]
        if grant.origin == "user":
            return None
        return (f"{grant.path} is shared automatically as "
                f"{access_label(grant.access)} and cannot be changed or "
                "removed here. Use Uninstall to remove everything.")

    def add(self, path: str, access: Access) -> None:
        self.definition.grants.append(Grant(path, access, "user"))

    def set_level(self, index: int, access: Access) -> None:
        grant = self.definition.grants[index]
        self.definition.grants[index] = Grant(grant.path, access, grant.origin)

    def remove(self, index: int) -> None:
        del self.definition.grants[index]

    def plan(self) -> Plan:
        return build_plan(self.definition, self.previous)

    def describe(self) -> str:
        return describe_plan(self.plan())

    def apply(self) -> list[Check]:
        return self.backend.apply(self.plan(), self.definition)

    def verify(self) -> list[Check]:
        return self.backend.verify(self.definition)

    def uninstall(self) -> list[Check]:
        checks = self.backend.uninstall(self.ledger)
        self.reload(self.workspace)
        return checks


def describe_plan(plan: Plan) -> str:
    """The pending change, as text to show before anything is applied."""
    verbs = {"add": "will grant", "remove": "will stop granting",
             "level": "will change"}
    lines = [
        f"  * {verbs.get(change.kind, change.kind)} {change.path} "
        f"({access_label(change.access)})"
        for change in plan.changes
    ]
    if not lines:
        lines.append("  * no changes to grants")
    if plan.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  ! {warning}" for warning in plan.warnings)
    return "\n".join(lines)


def _check_text(checks: list[Check]) -> str:
    return "\n".join(
        f"{check.status}: {check.name} {check.detail}".rstrip()
        for check in checks)


class Editor:
    """The Tk layer: builds widgets and delegates every decision to the model.

    The dialog modules are injected so a test can drive the widgets with scripted
    answers instead of blocking on a modal window.
    """

    def __init__(self, root, model: EditorModel, tk, filedialog, messagebox):
        self.root = root
        self.model = model
        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.status = 1
        self._build()
        self.refresh()

    def _build(self) -> None:
        tk = self.tk
        self.root.title("Loki setup - Windows container")
        self.root.minsize(560, 420)
        tk.Label(self.root, anchor="w", justify="left", text=(
            "Workspace (Browse switches which workspace this window edits):")
        ).pack(fill="x", padx=8, pady=(8, 0))
        workspace_row = tk.Frame(self.root)
        workspace_row.pack(fill="x", padx=8)
        self.workspace_var = tk.StringVar(value=self.model.workspace)
        tk.Entry(workspace_row, textvariable=self.workspace_var,
                 state="readonly").pack(side="left", fill="x", expand=True)
        tk.Button(workspace_row, text="Browse...",
                  command=self.browse).pack(side="left", padx=(4, 0))
        tk.Frame(self.root, height=2, bd=1, relief="sunken").pack(
            fill="x", padx=8, pady=8)
        tk.Label(self.root, anchor="w",
                 text="Accessible directories:").pack(fill="x", padx=8)
        list_frame = tk.Frame(self.root)
        list_frame.pack(fill="both", expand=True, padx=8)
        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        self.listing = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                  selectmode="browse", exportselection=False)
        scrollbar.config(command=self.listing.yview)
        scrollbar.pack(side="right", fill="y")
        self.listing.pack(side="left", fill="both", expand=True)
        tk.Label(self.root, anchor="w", justify="left", fg="#444444", text=(
            "Automatically shared: this workspace (read and write) and Loki's "
            "toolchain (read). Temporary files go inside the workspace, which "
            "is already shared.\n"
            f"Always protected and never grantable: "
            f"{paths.credential_directory()}")).pack(
                fill="x", padx=8, pady=(4, 8))
        buttons = tk.Frame(self.root)
        buttons.pack(fill="x", padx=8, pady=8)
        for label, command in (("Add...", self.add),
                               ("Modify...", self.modify),
                               ("Delete...", self.delete)):
            tk.Button(buttons, text=label, command=command).pack(side="left")
        actions = tk.Frame(self.root)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        for label, command in (("Apply", self.apply), ("Verify", self.verify)):
            tk.Button(actions, text=label, command=command).pack(side="left")
        tk.Button(actions, text="Uninstall...",
                  command=self.uninstall).pack(side="right")
        tk.Button(actions, text="Close",
                  command=self.root.destroy).pack(side="right")

    def refresh(self, select: int | None = None) -> None:
        self.listing.delete(0, "end")
        for row in self.model.rows():
            self.listing.insert("end", row)
        if select is not None and 0 <= select < self.listing.size():
            self.listing.selection_clear(0, "end")
            self.listing.selection_set(select)

    def selected(self) -> int | None:
        chosen = self.listing.curselection()
        return int(chosen[0]) if chosen else None

    def add(self) -> None:
        path = self.filedialog.askdirectory(
            title="Add an accessible directory", mustexist=True)
        if not path:
            return
        level = self.messagebox.askyesnocancel(
            "Access level",
            f"How much access to {path}?\n\n"
            "Yes = read and write (including deleting files)\n"
            "No = read only\n"
            "Cancel = do not add")
        if level is None:
            return
        self.model.add(path, Access.READ_WRITE if level else Access.READ)
        self.refresh(self.listing.size())

    def modify(self) -> None:
        index = self.selected()
        if index is None:
            return
        refusal = self.model.edit_refusal(index)
        if refusal is not None:
            self.messagebox.showinfo("Not editable", refusal)
            return
        grant = self.model.grants[index]
        access = (Access.READ_WRITE if grant.access is Access.READ
                  else Access.READ)
        self.model.set_level(index, access)
        self.refresh(index)

    def delete(self) -> None:
        index = self.selected()
        if index is None:
            return
        refusal = self.model.edit_refusal(index)
        if refusal is not None:
            self.messagebox.showinfo("Not removable", refusal)
            return
        if not self.messagebox.askokcancel(
                "Remove", f"Stop sharing {self.model.grants[index].path}?"):
            return
        self.model.remove(index)
        self.refresh()

    def browse(self) -> None:
        # Unsaved edits are never dropped silently: apply them, discard them,
        # or stay here.
        if self.model.dirty:
            answer = self.messagebox.askyesnocancel(
                "Unsaved changes",
                "This workspace has unapplied changes.\n\n"
                "Yes = apply them first\nNo = discard them\n"
                "Cancel = stay here")
            if answer is None:
                return
            if answer and not self._apply_current():
                return
        picked = self.filedialog.askdirectory(
            title="Select the workspace this window edits", mustexist=True)
        if not picked:
            return
        self.model.reload(picked)
        self.workspace_var.set(self.model.workspace)
        self.refresh()

    def _apply_current(self) -> bool:
        """Apply the pending plan; False when refused or declined."""
        plan = self.model.plan()
        if not plan.applicable:
            self.messagebox.showerror(
                "Refused", "Loki never grants these paths:\n\n" +
                "\n".join(f"  * {error}" for error in plan.errors))
            return False
        if not self.messagebox.askokcancel(
                "Apply",
                f"{plan.workspace}\n\n{self.model.describe()}\n\nApply?"):
            return False
        self.messagebox.showinfo("Applied", _check_text(self.model.apply()))
        self.status = 0
        return True

    def apply(self) -> None:
        self._apply_current()

    def verify(self) -> None:
        self.messagebox.showinfo("Verify", _check_text(self.model.verify()))

    def uninstall(self) -> None:
        if not self.messagebox.askokcancel(
                UNINSTALL_TITLE, UNINSTALL_MESSAGE, icon="warning"):
            return
        self.messagebox.showinfo("Uninstalled",
                                 _check_text(self.model.uninstall()))


def run_editor(workspace: str, ledger: dict, backend: Backend) -> int:
    """Run the Tk editor; 0 when the workspace was applied."""
    tk, filedialog, messagebox = _tk()
    root = tk.Tk()
    editor = Editor(root, EditorModel(ledger, workspace, backend),
                    tk, filedialog, messagebox)
    root.mainloop()
    return editor.status


# -- command line --------------------------------------------------------

USAGE = (
    "usage: loki-setup [--edit [WORKSPACE]] [--verify WORKSPACE] "
    "[--list] [--uninstall]\n"
    "\n"
    "Windows-only container setup for Loki. Grants are a setup-time property;\n"
    "there is no way to widen a container from a running session.\n"
)


def list_workspaces(ledger: dict) -> int:
    for key, entry in sorted(ledger.get("workspaces", {}).items()):
        print(f"{key}\t{entry.get('profile', '')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if sys.platform != "win32":
        # Keep the flag vocabulary out of POSIX entirely: an unknown option
        # must fail here rather than quietly doing nothing.
        print("loki-setup: the Windows setup editor runs on Windows only",
              file=sys.stderr)
        return 2
    if not arguments or arguments[0] in ("-h", "--help"):
        print(USAGE, end="")
        return 0
    mode, rest = arguments[0], arguments[1:]
    ledger = load_ledger()
    if mode == "--list":
        return list_workspaces(ledger)
    backend = backend_for(ledger)
    if mode == "--uninstall":
        for check in backend.uninstall(ledger):
            print(f"{check.status}: {check.name} {check.detail}".rstrip())
        return 0
    if mode == "--verify":
        if not rest:
            print("loki-setup: --verify needs a workspace", file=sys.stderr)
            return 2
        definition = definition_for(rest[0], ledger.get(
            "workspaces", {}).get(workspace_key(rest[0])))
        failed = False
        for check in backend.verify(definition):
            print(f"{check.status}: {check.name} {check.detail}".rstrip())
            failed |= check.status == "fail"
        return 1 if failed else 0
    if mode == "--edit":
        workspace = rest[0] if rest else os.path.expanduser("~")
        return run_editor(workspace, ledger, backend)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
