"""Windows setup editor: the per-workspace container definition.

Windows-only, and deliberately a separate process from the chat.  This module
is the one place that explains the split; the other Windows modules refer back
here.

What the split does:

* Keeps profile creation and ACL rewriting out of the process that holds
  credentials and drives the model loop, so the runtime has no built-in grant
  path.  Widening a container means editing it here and re-running, exactly as
  a bubblewrap invocation cannot be widened after ``unshare``; the model loop
  cannot reach ``apply`` at all, only argv and an exit status.
* Keeps Tk -- a large C surface -- out of the credential-holding process.

What the split does *not* do:

* It is not a boundary against code that can be written: anything that can run
  a shell can declare the same ctypes bindings.  What contains the chat is the
  AppContainer token and the DACLs -- grants are Modify, never Full Control, so
  there is no ``WRITE_DAC`` -- and the probe in ``windows_verify`` is what
  checks those hold.  The split buys legibility and a smaller blast radius for
  accidental and indirect calls; it does not remove the capability.

The shared data, rules and read-only verification live in ``windows_state`` and
``windows_verify``; the OS mutation lives in ``windows_containers``.

Nothing security-relevant crosses the process boundary: the editor writes the
ledger and exits, and the caller re-derives the package SID and re-verifies
itself rather than trusting this process' output or its identity.

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

import os
import sys

from . import paths
from . import windows_api
from . import windows_containers
from . import windows_verify
from .windows_state import (
    Access,
    Backend,
    Check,
    Definition,
    Grant,
    Plan,
    add_package_ace,
    build_plan,
    entry_grants,
    ledger_entry,
    ledger_path,
    load_ledger,
    names_package,
    remove_package_aces,
    save_ledger,
    workspace_key,
)


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
            return windows_containers.create_app_container_profile(name)
        except windows_api.WindowsApiError as error:
            if error.status != windows_containers.PROFILE_ALREADY_EXISTS:
                raise
            return windows_api.derive_app_container_sid(name)

    def _grant_path(self, path: str, access: Access, package: str) -> None:
        current = windows_api.dacl_sddl(path)
        windows_containers.set_dacl_sddl(
            path, add_package_ace(current, package, access))

    def _ungrant_path(self, path: str, package: str) -> None:
        current = windows_api.dacl_sddl(path)
        if not names_package(current, package):
            return
        updated = remove_package_aces(current, package)
        # A deny ACE names the package but has no allow to remove; do not rewrite
        # a DACL that this edit did not change.
        if updated != current:
            windows_containers.set_dacl_sddl(path, updated)

    def _private_directory(self, path: str, user: str) -> None:
        os.makedirs(path, exist_ok=True)
        windows_containers.set_dacl_sddl(path, private_dacl_sddl(user))

    def _record_entry(self, key, entry):
        workspaces = dict(self.ledger.get("workspaces", {}))
        if entry is None:
            workspaces.pop(key, None)
        else:
            workspaces[key] = entry
        updated = {**self.ledger, "workspaces": workspaces}
        save_ledger(updated, self.ledger_location)
        self.ledger.clear()
        self.ledger.update(updated)

    # -- Backend --
    def apply(self, plan: Plan, definition: Definition) -> list[Check]:
        key = workspace_key(definition.workspace)
        previous = self.ledger.get("workspaces", {}).get(key)
        checked_plan = build_plan(definition, previous)
        if not checked_plan.applicable or plan.profile != checked_plan.profile:
            return [Check("plan", "fail", "; ".join(checked_plan.errors)
                          or "profile does not match workspace")]
        checks: list[Check] = []
        try:
            user = windows_api.current_user_sid()
            self._private_directory(paths.credential_directory(), user)
            self._private_directory(paths.loki_config_dir(), user)
            self._private_directory(paths.loki_state_dir(), user)
            # Persist recovery coverage BEFORE changing any grant. On a failed
            # or interrupted apply, every old/new path remains recorded, and
            # the startup gate refuses the pending entry. This is not a rollback
            # or a claim of power-loss durability.
            recovery = {workspace_key(g.path): g for g in entry_grants(previous or {})}
            recovery.update({workspace_key(g.path): g for g in definition.grants})
            pending = ledger_entry(definition.workspace, checked_plan.profile,
                                   list(recovery.values()))
            pending["pending"] = True
            self._record_entry(key, pending)
            package = self._ensure_profile(checked_plan.profile)
            checks.append(Check("profile", "pass", package))
            for change in checked_plan.changes:
                if change.kind == "remove":
                    self._ungrant_path(change.path, package)
                    checks.append(Check(f"ungrant {change.path}", "pass"))
            for grant in definition.grants:
                self._grant_path(grant.path, grant.access, package)
                checks.append(Check(f"grant {grant.path}", "pass", grant.access.value))
            self._record_entry(key, ledger_entry(
                definition.workspace, checked_plan.profile, definition.grants))
        except (windows_api.WindowsApiError, OSError, ValueError) as error:
            return checks + [Check("apply", "fail", str(error))]
        return checks + [Check("ledger", "pass", ledger_path())]

    def verify(self, definition: Definition) -> list[Check]:
        return windows_verify.verify_workspace(self.ledger, definition.workspace)

    def uninstall(self, blob: dict) -> list[Check]:
        checks: list[Check] = []
        for key, entry in sorted(list(blob.get("workspaces", {}).items())):
            profile = str(entry.get("profile", ""))
            try:
                # Revocation is retryable. Keep all paths until every removal
                # and profile deletion succeeds; a failed attempt is not an
                # empty/successful installation.
                self._record_entry(key, {**entry, "pending": True})
                package = windows_api.derive_app_container_sid(profile)
                for grant in entry_grants(entry):
                    self._ungrant_path(grant.path, package)
                    checks.append(Check(f"ungrant {grant.path}", "pass"))
                windows_containers.delete_app_container_profile(profile)
                self._record_entry(key, None)
                checks.append(Check(f"profile {profile}", "pass", "removed"))
            except (windows_api.WindowsApiError, OSError, ValueError) as error:
                checks.append(Check(f"uninstall {key}", "fail", str(error)))
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
    """Workspace read-write and toolchain read access, shown as automatic rows.

    This function does not configure TEMP/TMP. Runtime scratch placement is a
    separate launcher responsibility; private configuration trees stay denied.
    """
    toolchain = os.path.dirname(os.path.abspath(sys.executable))
    return [
        Grant(workspace, Access.READ_WRITE, "workspace"),
        Grant(toolchain, Access.READ, "toolchain"),
    ]


def definition_for(workspace: str, previous: dict | None) -> Definition:
    """Build the definition to edit: automatic grants plus recorded user ones."""
    grants = automatic_grants(workspace)
    grants.extend(grant for grant in entry_grants(previous or {})
                  if grant.origin == "user")
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
        checks = self.backend.apply(self.plan(), self.definition)
        if checks and all(check.status == "pass" for check in checks):
            self.previous = self.ledger.get("workspaces", {}).get(self.key)
            self._saved = list(self.definition.grants)
        return checks

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
            "toolchain (read).\n"
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
            self.status = 1
            self.messagebox.showerror(
                "Refused", "Loki never grants these paths:\n\n" +
                "\n".join(f"  * {error}" for error in plan.errors))
            return False
        if not self.messagebox.askokcancel(
                "Apply",
                f"{plan.workspace}\n\n{self.model.describe()}\n\nApply?"):
            return False
        checks = self.model.apply()
        if not checks or any(check.status != "pass" for check in checks):
            self.status = 1
            self.messagebox.showerror("Apply failed", _check_text(checks))
            return False
        self.messagebox.showinfo("Applied", _check_text(checks))
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
        checks = self.model.uninstall()
        self.status = 1  # No configured workspace to start after uninstall.
        if any(check.status != "pass" for check in checks):
            self.messagebox.showerror("Uninstall incomplete", _check_text(checks))
        else:
            self.messagebox.showinfo("Uninstalled", _check_text(checks))


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
        checks = backend.uninstall(ledger)
        for check in checks:
            print(f"{check.status}: {check.name} {check.detail}".rstrip())
        return 1 if any(check.status != "pass" for check in checks) else 0
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
