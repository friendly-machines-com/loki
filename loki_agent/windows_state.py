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
import re
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


# Numeric rights (winnt.h), so verification compares masks instead of trusting
# that a SID being *named* means access being *granted*.
FILE_GENERIC_READ = 0x00120089
FILE_GENERIC_WRITE = 0x00120116
FILE_GENERIC_EXECUTE = 0x001200A0
FILE_ALL_ACCESS = 0x001F01FF
DELETE = 0x00010000
READ_CONTROL = 0x00020000
WRITE_DAC = 0x00040000
WRITE_OWNER = 0x00080000
SYNCHRONIZE = 0x00100000
ACCESS_SYSTEM_SECURITY = 0x01000000

# SDDL file-object rights strings (Ace Strings reference) and their masks.
# Generic rights are expanded to the file rights the kernel maps them to, so an
# ACE that literally stores ``GR`` is judged by the access it actually confers.
_RIGHTS = {
    "FA": FILE_ALL_ACCESS,
    "FR": FILE_GENERIC_READ,
    "FW": FILE_GENERIC_WRITE,
    "FX": FILE_GENERIC_EXECUTE,
    "GA": FILE_ALL_ACCESS,
    "GR": FILE_GENERIC_READ,
    "GW": FILE_GENERIC_WRITE,
    "GX": FILE_GENERIC_EXECUTE,
    "RC": READ_CONTROL,
    "SD": DELETE,
    "WD": WRITE_DAC,
    "WO": WRITE_OWNER,
    "SY": SYNCHRONIZE,
    "AS": ACCESS_SYSTEM_SECURITY,
}


def _rights_mask(rights: str) -> int:
    """Parse an SDDL rights field to a numeric access mask.

    Refuse an unknown string instead of under-counting it: this result decides
    whether a grant is present, so guessing low would turn a real grant into a
    false "no access".
    """
    if rights[:2].lower() == "0x":
        try:
            return int(rights, 16)
        except ValueError as error:
            raise windows_api.WindowsApiError(
                f"malformed SDDL rights {rights!r}") from error
    if not rights or len(rights) % 2:
        raise windows_api.WindowsApiError(
            f"unsupported SDDL rights {rights!r}")
    mask = 0
    for index in range(0, len(rights), 2):
        code = rights[index:index + 2]
        if code not in _RIGHTS:
            raise windows_api.WindowsApiError(
                f"unsupported SDDL rights {rights!r}")
        mask |= _RIGHTS[code]
    return mask


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
    return _rights_mask(access_sddl(access))


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


def protected_path_errors(path: str, access: Access = Access.READ_WRITE) -> list[str]:
    """Reject overlap with private data; trusted code may be read, not written.

    Resolve existing aliases before comparing. This detects junction/symlink
    aliases at validation time, not races replacing a path during ACL updates;
    those require retained-handle operations in the Windows backend.
    """
    target = os.path.normcase(os.path.realpath(path))
    reasons = []
    for tree in _runtime_trees(include_code=access is Access.READ_WRITE):
        protected = os.path.normcase(os.path.realpath(tree))
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

def _dacl_parts(sddl: str):
    """Parse basic DACL SDDL; refuse unsupported forms rather than corrupt them."""
    header, separator, rest = sddl.partition("(")
    if not re.fullmatch(r"D:(?:P|AI|AR)*", header):
        raise windows_api.WindowsApiError("unsupported DACL header")
    body = separator + rest
    aces = re.findall(r"\([^()]*\)", body)
    if ''.join(aces) != body or any(len(ace[1:-1].split(';')) != 6 for ace in aces):
        raise windows_api.WindowsApiError("unsupported or malformed DACL ACE")
    return header, aces


def add_package_ace(sddl: str, package: str, access: Access) -> str:
    """Replace this profile's explicit allow ACEs, preserving other entries."""
    header, aces = _dacl_parts(remove_package_aces(sddl, package))
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
    header, aces = _dacl_parts(sddl)
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


def _ace_fields(sddl: str):
    """Yield ``(kind, flags, rights, object, inherited_object, sid)`` per ACE.

    Naming an ACE's fields lets a caller tell an allow from a deny and a SID
    from a rights string without assuming basic (non-object) ACEs: object and
    inherited-object GUIDs occupy the same columns, and the SID is always last.
    """
    _header, aces = _dacl_parts(sddl)
    for ace in aces:
        yield tuple(ace[1:-1].split(';'))


def _flag_codes(flags: str) -> set[str]:
    """Split an ACE-flags field into its two-character codes."""
    if len(flags) % 2:
        raise windows_api.WindowsApiError(f"unsupported ACE flags {flags!r}")
    return {flags[index:index + 2] for index in range(0, len(flags), 2)}


def names_package(sddl: str, package: str) -> bool:
    """Whether any ACE in ``sddl`` names ``package``, allow or deny.

    Presence only, for deciding whether an edit has anything to touch; it says
    nothing about access, which is :func:`package_access`'s question.
    """
    return any(sid == package
               for _kind, _flags, _rights, _object, _inherited, sid
               in _ace_fields(sddl))


def _ace_mask(kind, rights, obj, inherited):
    """The access mask an ACE grants or denies, or ``None`` if it does neither.

    Object ACEs with a GUID restrict the entry to a property or object type,
    which this check cannot evaluate; refuse rather than assume the mask
    applies unqualified.
    """
    if kind in ("A", "D"):
        return _rights_mask(rights)
    if kind in ("OA", "OD"):
        if obj or inherited:
            raise windows_api.WindowsApiError(
                "cannot evaluate an object ACE for the package SID")
        return _rights_mask(rights)
    return None


def package_access(sddl: str, package: str) -> int:
    """The rights ``sddl`` actually confers on ``package``.

    This is the ACL's own answer for one trustee: the union of the allow masks
    that apply to the object, minus the union of the deny masks that apply to
    it.  An inherit-only ACE (``IO``) applies only to children and is not
    counted; an inherited ACE (``ID``) does apply and is.  Audit, alarm and
    mandatory-label entries neither grant nor deny and are skipped.  Group
    membership and implicit owner rights are outside this computation, which
    answers only what the DACL says about this SID.
    """
    allowed = denied = 0
    for kind, flags, rights, obj, inherited, sid in _ace_fields(sddl):
        if sid != package or "IO" in _flag_codes(flags):
            continue
        mask = _ace_mask(kind, rights, obj, inherited)
        if mask is None:
            continue
        if kind in ("D", "OD"):
            denied |= mask
        else:
            allowed |= mask
    return allowed & ~denied


def package_allow(sddl: str, package: str) -> int:
    """The allow rights any ACE naming ``package`` could confer.

    Unlike :func:`package_access`, inherit-only allows are counted: an entry
    that grants only children still hands the package the contents of a
    protected tree, so the private-tree check must not read the tree's own
    DACL as clearing an inheritable grant.  Deny entries alone contribute
    nothing, which is what keeps a deny-only DACL from being reported as a
    grant.
    """
    allowed = 0
    for kind, _flags, rights, obj, inherited, sid in _ace_fields(sddl):
        if sid != package:
            continue
        mask = _ace_mask(kind, rights, obj, inherited)
        if mask is not None and kind in ("A", "OA"):
            allowed |= mask
    return allowed


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
