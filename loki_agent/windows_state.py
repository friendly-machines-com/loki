"""Per-workspace container state, shared by the chat and the editor.

Split out of ``windows_setup`` so the chat can read the ledger, apply the path
rules and verify a container without importing the editor or the OS mutation it
performs; ``windows_setup`` explains why, and what the split does not buy.

Nothing in this module mutates the system: it is data (the ledger), pure rules,
and the declarations the two sides share.
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
from .windows_acl import (
    dacl_parts,
    package_access,
    rights_mask,
)
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


def access_mask(access: Access) -> int:
    """The numeric rights a level means, derived from :func:`access_sddl`."""
    return rights_mask(access_sddl(access))


# -- path rules ----------------------------------------------------------

def _runtime_trees(include_code: bool = True) -> list[str]:
    """Private trees, plus trusted code when checking a writable grant."""
    trees = [paths.loki_config_dir(), paths.loki_state_dir(),
             paths.credential_directory()]
    if include_code:
        trees.append(os.path.dirname(os.path.abspath(__file__)))
    if include_code and getattr(sys, "frozen", False):
        # A packaged build's own directory holds the runtime and its bundled
        # interpreter; a grant covering it would let a tool rewrite Loki.
        trees.append(os.path.dirname(os.path.abspath(sys.executable)))
    return trees


def _final_path(path: str) -> str:
    """The object's canonical path, asked of the kernel about an open handle.

    Not ``realpath``: that resolves per component, and an answer derived from
    the object itself is cheaper and cannot be redirected by a rename during
    the check.
    """
    handle = windows_api.open_directory_handle(path)
    try:
        return windows_api.final_path_from_handle(handle)
    finally:
        windows_api.close_handle(handle)


def _canonical_or_absolute(path: str) -> str:
    """The object's canonical path if it exists, else its absolute spelling.

    A protected tree that does not exist yet still has a location, and a grant
    covering that location must be refused before it is created -- there is no
    handle to ask, so the spelling is the only answer available.
    """
    try:
        return os.path.normcase(_final_path(path))
    except windows_api.WindowsApiError:
        return os.path.normcase(os.path.abspath(path))


def protected_path_errors(path: str, access: Access = Access.READ_WRITE) -> list[str]:
    """Reject overlap with private data; trusted code may be read, not written.

    Both sides are canonicalised from an open handle, so an alias (junction or
    symlink) is seen through.  This is decided here and the ACL is applied
    later; closing that window needs the same handle retained through the ACL
    write, which the backend does not do yet.
    """
    target = _canonical_or_absolute(path)
    reasons = []
    for tree in _runtime_trees(include_code=access is Access.READ_WRITE):
        protected = _canonical_or_absolute(tree)
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
    try:
        target = os.path.normcase(_final_path(path))
    except windows_api.WindowsApiError:
        return []
    warnings = []
    for hint in SECRET_HINTS:
        candidate = os.path.join(profile, hint)
        if not os.path.exists(candidate):
            continue
        protected = os.path.normcase(_final_path(candidate))
        if target == protected or _contains(target, protected):
            warnings.append(
                f"{path} also covers {candidate}, which exists")
    return warnings


# -- names ---------------------------------------------------------------

def canonical_workspace(path: str) -> str:
    """Canonical form used as the workspace key.

    Naming only: it folds case and asks the kernel for the object's canonical
    path so one directory has one profile, and it is never used to rewrite the
    operand itself.  The answer comes from an open handle, so it does not
    depend on the caller's cwd and cannot be changed by a rename during the
    call.
    """
    return os.path.normcase(_final_path(path))


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
    origin: str  # 'workspace' | 'runtime' | 'temp' | 'user'


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
    """Replace this profile's explicit allow ACEs, preserving other entries."""
    header, aces = dacl_parts(remove_package_aces(sddl, package))
    # Explicit entries precede inherited ones. Inserting after inherited denies
    # would make the DACL noncanonical even though the new entry is an allow.
    index = next((i for i, ace in enumerate(aces)
                  if 'ID' in ace[1:-1].split(';')[1]), len(aces))
    aces.insert(index, f"(A;OICI;{access_sddl(access)};;;{package})")
    return header + ''.join(aces)


def remove_package_aces(sddl: str, package: str) -> str:
    """Remove explicit basic allows for Loki's profile, never deny ACEs.

    An inherited allow must be changed at its source. Refuse rather than
    disabling inheritance or claiming to revoke a right the parent still gives.
    """
    header, aces = dacl_parts(sddl)
    kept = []
    for ace in aces:
        kind, flags, _rights, _object, _inherited_object, sid = ace[1:-1].split(';')
        if sid == package and kind != 'D':
            if kind != 'A' or 'ID' in flags:
                raise windows_api.WindowsApiError(
                    "cannot edit inherited or non-basic package grants")
            continue
        kept.append(ace)
    return header + ''.join(kept)


def grants_access(sddl: str, package: str, access: Access) -> bool:
    """Whether ``sddl`` confers at least ``access`` on ``package``.

    Coverage, deny-first: the package must hold every right the level means,
    with no applicable deny clearing a bit.  A bare SID, a deny ACE, or an
    inherit-only allow therefore fails instead of passing on the string match
    that made the old check unusable as a gate.
    """
    required = access_mask(access)
    return package_access(sddl, package) & required == required


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
        errors.extend(protected_path_errors(grant.path, grant.access))
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


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # 'pass' | 'fail' | 'untested'
    detail: str = ""


class Backend(Protocol):
    def apply(self, plan: Plan, definition: Definition) -> list[Check]: ...
    def verify(self, definition: Definition) -> list[Check]: ...
    def uninstall(self, blob: dict) -> list[Check]: ...
