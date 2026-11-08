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
import secrets
import stat
import sys
from dataclasses import dataclass, field
from typing import Protocol

from . import paths
from . import private_files
from . import windows_api
from .windows_acl import (
    dacl_parts,
    package_access,
    rights_mask,
)
LEDGER_VERSION = 1
LEDGER_NAME = 'windows-setup.json'
_SETUP_TEMP_PREFIX = '.setup-'
_SETUP_TEMP_BYTES = 12
PROFILE_NAME_PREFIX = "Loki.Workspace."
# These are compatibility alternatives in the second (write) access check, not
# an additional private-role conjunct. Managed state must grant none of them
# mutation rights. The current logon SID is added from the actual token.
TOOL_COMPATIBILITY_SIDS = ['S-1-1-0', 'S-1-5-32-545', 'S-1-5-12']


class Access(enum.Enum):
    """The two grant levels.  Read-only is the default for added paths.

    There is deliberately no separate "read and execute": as a security
    distinction it is theater -- anything readable can be copied somewhere
    writable and run -- so the boundary is confidentiality (what can be read)
    and integrity/persistence (what can be written).
    """

    READ = "read"
    READ_WRITE = "read-write"


# SDDL rights text: read plus execute (traverse on a directory), and the
# documented Modify mask.
READ_SDDL_RIGHTS = "FRFX"
MODIFY_SDDL_RIGHTS = "0x1301BF"


def access_sddl(access: Access) -> str:
    """Return the DACL rights for a level.

    ``FRFX`` is generic read plus generic execute (execute on a directory is
    traverse, which known-path access needs).  Read-write is ``0x1301BF`` --
    the documented "Modify" mask, read/write/execute plus delete -- rather than
    ``FA``, because ``FA`` would also hand over ``WRITE_DAC`` and let a tool
    rewrite the DACL and lock the owner out of their own directory.
    """
    return READ_SDDL_RIGHTS if access is Access.READ else MODIFY_SDDL_RIGHTS


def access_mask(access: Access) -> int:
    """The numeric rights a level means, derived from :func:`access_sddl`."""
    return rights_mask(access_sddl(access))


# -- path rules ----------------------------------------------------------

@dataclass(frozen=True)
class ObjectLocation:
    """An opened object's identity and a snapshot of its proper ancestor IDs."""

    key: str
    ancestors: frozenset[str]

    def contains(self, other: ObjectLocation) -> bool:
        return self.key == other.key or self.key in other.ancestors


def object_location(handle) -> ObjectLocation:
    """Inspect a borrowed handle; own and close only the opened parent handles."""
    key = windows_api.file_identity(handle)
    ancestors: set[str] = set()
    current = handle
    with contextlib.ExitStack() as cleanup:
        while True:
            parent = windows_api.open_parent_directory(current)
            if parent is None:
                break
            cleanup.callback(windows_api.close_handle, parent)
            identity = windows_api.file_identity(parent)
            if identity == key or identity in ancestors:
                raise windows_api.WindowsApiError('Location changed while inspecting its parents')
            ancestors.add(identity)
            current = parent
    return ObjectLocation(key, frozenset(ancestors))


def _directory_location_from_handle(handle) -> ObjectLocation:
    information = windows_api.by_handle_file_information(handle)
    if not information.dwFileAttributes & windows_api.FileAttribute.FILE_ATTRIBUTE_DIRECTORY:
        raise windows_api.WindowsApiError('Selected location is not a directory')
    return object_location(handle)


def _directory_location(path: str) -> ObjectLocation:
    handle = windows_api.open_directory_handle(path)
    try:
        return _directory_location_from_handle(handle)
    finally:
        windows_api.close_handle(handle)


def _object_at(path: str) -> ObjectLocation:
    handle = windows_api.open_directory_handle(path)
    try:
        return object_location(handle)
    finally:
        windows_api.close_handle(handle)


def _optional_location(path: str) -> ObjectLocation | None:
    try:
        return _directory_location(path)
    except windows_api.WindowsResourceError:
        raise
    except windows_api.WindowsApiError as error:
        if error.status in [windows_api.ERROR_FILE_NOT_FOUND, windows_api.ERROR_PATH_NOT_FOUND]:
            return None
        raise


def program_directories(*, automatic_read=False) -> list[str]:
    """Resolve setup's program-directory roles through actual opened objects.

    Frozen releases put the two runtime bundles beside the selected editor
    bundle. Source execution needs the selected interpreter and the package's
    enclosing source directory. These are installation-layout selections, not
    rewriting of a workspace/configuration operand. Automatic reads exclude
    the editor-only bundle; protection includes it. Role selection lives here,
    not in a second flag attached to each returned pathname.
    """
    selected: dict[str, str] = {}
    with contextlib.ExitStack() as cleanup:
        def own(handle):
            cleanup.callback(windows_api.close_handle, handle)
            return handle

        def parent(handle):
            opened = windows_api.open_parent_directory(handle)
            return handle if opened is None else own(opened)

        def add(handle):
            key = windows_api.file_identity(handle)
            if key not in selected:
                selected[key] = windows_api.kernel_path_from_handle(handle)

        image = own(windows_api.open_with_access(sys.executable, windows_api.AccessMask.FILE_READ_ATTRIBUTES))
        image_directory = parent(image)
        if getattr(sys, 'frozen', False):
            release = parent(image_directory)
            for name in ['loki', 'loki-acp']:
                bundle = own(windows_api.nt_create_file(
                    release, name, windows_api.AccessMask.FILE_READ_ATTRIBUTES,
                    windows_api.NtCreateDisposition.FILE_OPEN, windows_api.NtCreateOptions.FILE_DIRECTORY_FILE))
                add(bundle)
            if not automatic_read:
                add(image_directory)
        else:
            module = own(windows_api.open_with_access(__file__, windows_api.AccessMask.FILE_READ_ATTRIBUTES))
            add(parent(parent(module)))
            add(image_directory)
    return list(selected.values())


def _root_locations(include_code: bool = True) -> dict[str, ObjectLocation | None]:
    """Existing protected objects; absent automatic roots are checked on Apply."""
    roots = {
        'configuration': _optional_location(paths.loki_config_dir()),
        'state': _optional_location(paths.loki_state_dir()),
        'credentials': _optional_location(paths.credential_directory()),
    }
    if include_code:
        for index, program in enumerate(program_directories(), 1):
            roots[f'program files ({index})'] = _optional_location(program)
    return roots


def _configuration_errors(roots: dict[str, ObjectLocation | None]) -> list[str]:
    config, state, credentials = roots['configuration'], roots['state'], roots['credentials']
    errors = []
    if config is not None and state is not None and (config.contains(state) or state.contains(config)):
        errors.append('Loki configuration and state directories must not overlap')
    if config is not None and credentials is not None and credentials.contains(config):
        errors.append('Loki configuration must not resolve inside credentials')
    return errors


def configuration_path_errors() -> list[str]:
    """Judge existing objects; creation is followed by the same check on Apply."""
    try:
        return _configuration_errors(_root_locations(include_code=False))
    except windows_api.WindowsApiError as error:
        return [f'Could not inspect the configuration layout: {error}']


def _protected_reasons(target: ObjectLocation, path: str, access: Access, origin='user') -> list[str]:
    """Compare opened objects and their ancestry, never pathname spellings."""
    roots = _root_locations(include_code=access is Access.READ_WRITE)
    state, config = roots['state'], roots['configuration']
    managed_state = origin == 'state' and state is not None and target.key == state.key
    if access is Access.READ and config is not None and target.key == config.key:
        # This exact object is deliberately readable; its credential child has
        # a protected DACL. An alias does not change which object is judged.
        return _configuration_errors(roots)
    reasons = _configuration_errors(roots) if managed_state else []
    for role, protected in roots.items():
        if protected is None or (managed_state and role == 'state'):
            continue
        if target.key == protected.key:
            reasons.append(f"{path!r} is Loki's protected {role}")
        elif target.contains(protected):
            reasons.append(f"{path!r} contains Loki's protected {role}")
        elif protected.contains(target):
            reasons.append(f"{path!r} is inside Loki's protected {role}")
    return reasons


def _automatic_default(path: str, access: Access, origin: str) -> bool:
    # Classify only the producer's configured automatic operand, not an object
    # identity. A missing object is never admitted to an ACL write by this test.
    return ((origin == 'config' and access is Access.READ and path == paths.loki_config_dir())
            or (origin == 'state' and path == paths.loki_state_dir()))


def protected_path_errors(path: str, access: Access = Access.READ_WRITE, *, origin='user') -> list[str]:
    """Preflight native lookup; Apply rechecks the handle whose ACL it changes."""
    try:
        return _protected_reasons(_directory_location(path), path, access, origin)
    except windows_api.WindowsApiError as error:
        if (not isinstance(error, windows_api.WindowsResourceError)
                and _automatic_default(path, access, origin)
                and error.status in [windows_api.ERROR_FILE_NOT_FOUND, windows_api.ERROR_PATH_NOT_FOUND]):
            # First setup creates these directories on Apply, then checks the
            # actual grant handle. Do not invent an absolute spelling as identity.
            return []
        return [f'Could not inspect {path!r}: {error}']


def handle_path_errors(handle, path: str,
                       access: Access = Access.READ_WRITE, *, origin='user') -> list[str]:
    """Judge the same opened directory whose ACL the caller will read/write.

    ``path`` labels an error only; it never supplies the target's identity.
    Parent selection verifies each binding through object IDs and retains the
    native namespace. No name-based fallback admits an unverified grant.
    """
    try:
        return _protected_reasons(_directory_location_from_handle(handle), path, access, origin)
    except windows_api.WindowsApiError as error:
        return [f'Could not inspect {path!r}: {error}']


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
        target = _directory_location(path)
    except windows_api.WindowsApiError:
        return []
    warnings = []
    for hint in SECRET_HINTS:
        candidate = os.path.join(profile, hint)
        try:
            protected = _object_at(candidate)
        except windows_api.WindowsApiError:
            # These are advisory hints about objects we can inspect, never an
            # admission decision or a fallback that authorizes a grant.
            continue
        if target.contains(protected):
            warnings.append(f"{path!r} also covers {candidate!r}, which exists")
    return warnings


# -- names ---------------------------------------------------------------

def _directory_identity(path: str) -> str:
    """Select a directory through native lookup, then identify that object."""
    handle = windows_api.open_directory_handle(path)
    try:
        information = windows_api.by_handle_file_information(handle)
        if not information.dwFileAttributes & windows_api.FileAttribute.FILE_ATTRIBUTE_DIRECTORY:
            raise windows_api.WindowsApiError(f"Not a directory: {path!r}")
        return windows_api.file_identity(handle)
    finally:
        windows_api.close_handle(handle)


def profile_name_for_key(key: str) -> str:
    """Derive the profile from an already-selected native directory identity."""
    digest = hashlib.sha256(key.encode("ascii")).hexdigest()
    return PROFILE_NAME_PREFIX + digest[:32]


def profile_name_for(workspace: str) -> str:
    """Aliases and renames of one existing directory select one profile."""
    return profile_name_for_key(workspace_key(workspace))


def tool_role_sid(package: str) -> str:
    """One stable ordinary restricting SID for a kernel-reported package.

    Package-authority SIDs cannot be used as restricting SIDs. A domain-separated
    128-bit identity avoids another stored mapping: setup and launch derive the
    same trustee, while neither argv nor a readable state file selects it.
    """
    if not package.startswith('S-1-15-2-'):
        raise ValueError('tool role requires an AppContainer package SID')
    digest = hashlib.sha256(b'Loki.Tool.Role.v1\0' + package.encode('ascii')).digest()
    return 'S-1-5-21-' + '-'.join(str(int.from_bytes(digest[i:i + 4], 'little')) for i in [0, 4, 8, 12])


# -- definitions and the ledger -----------------------------------------

@dataclass(frozen=True)
class Grant:
    path: str
    access: Access
    origin: str  # 'workspace' | 'runtime' | 'config' | 'state' | 'temp' | 'user'


@dataclass
class Definition:
    workspace: str
    grants: list[Grant] = field(default_factory=list)


def ledger_path() -> str:
    return os.path.join(paths.loki_state_dir(), LEDGER_NAME)


def load_ledger() -> dict:
    """Load the configured setup ledger, empty when absent or unreadable."""
    location = ledger_path()
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


def save_ledger(blob: dict) -> None:
    """Publish the fixed metadata name through one retained state-directory object.

    Setup metadata is private at creation, even while its parent permits owning
    runtimes to maintain other state. The caller never supplies another ledger
    pathname; XDG/configured state selection already has one owner in paths.
    """
    data = (json.dumps(blob, indent=2, sort_keys=True) + '\n').encode('utf-8')
    directory = private_files.open_directory_tree(paths.loki_state_dir())
    temporary = None
    try:
        # The acquired directory must also close if temporary naming fails.
        candidate = _SETUP_TEMP_PREFIX + secrets.token_hex(_SETUP_TEMP_BYTES)
        handle = private_files.create_private_at(directory, candidate)
        # A failed exclusive create never makes another publication ours to delete.
        temporary = candidate
        with private_files.binary_file(handle, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        private_files.replace_at(directory, candidate, LEDGER_NAME)
        temporary = None
        private_files.fsync(directory)
    finally:
        try:
            if temporary is not None:
                with contextlib.suppress(FileNotFoundError):
                    private_files.unlink_at(directory, temporary)
        finally:
            private_files.close(directory)


def workspace_key(workspace: str) -> str:
    return _directory_identity(workspace)


def recovery_keys_for(ledger, key):
    """Pending records which cannot be proved independent of this workspace.

    Setup may retry its exact key; startup must refuse it. Both must also
    detect a stale key naming the same native object, not compare spellings.
    This replaces their global pending checks, not the shared ACL inventory:
    every recorded writer and recovery grant remains owned by the ledger.
    """
    for recorded, entry in ledger.get('workspaces', {}).items():
        if not isinstance(entry, dict):
            yield recorded
            continue
        if not entry.get('pending'):
            continue
        try:
            relevant = recorded == key or workspace_key(entry['workspace']) == key
        except (KeyError, TypeError, ValueError, OSError, windows_api.WindowsApiError):
            # An unreadable recovery operand is not a proven other workspace.
            relevant = True
        if relevant:
            yield recorded


def ledger_entry(workspace: str, profile: str,
                 grants: list[Grant]) -> dict:
    return {
        "workspace": workspace,
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


def setup_asset_name(name: str) -> bool:
    """Classify only the setup publisher's positive leaf-name namespace."""
    if name == LEDGER_NAME:
        return True
    if not name.startswith(_SETUP_TEMP_PREFIX):
        return False
    suffix = name[len(_SETUP_TEMP_PREFIX):]
    return len(suffix) == _SETUP_TEMP_BYTES * 2 and all(character in '0123456789abcdef' for character in suffix)


def state_objects(root=None):
    """Inventory the existing managed tree without following reparse entries.

    Setup/verification inspect every existing object, not merely the root ACL.
    The parent grants' inheritance secures future runtime-created descendants.
    """
    root = paths.loki_state_dir() if root is None else root

    def walk(directory, *, setup_root=False):
        yield directory
        with os.scandir(directory) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            facts = entry.stat(follow_symlinks=False)
            if (stat.S_ISLNK(facts.st_mode) or
                    getattr(facts, 'st_file_attributes', 0) & windows_api.FileAttribute.FILE_ATTRIBUTE_REPARSE_POINT):
                raise windows_api.WindowsApiError('managed state contains a reparse entry: ' + entry.path)
            if setup_root and setup_asset_name(entry.name):
                if not stat.S_ISREG(facts.st_mode):
                    raise windows_api.WindowsApiError('setup metadata is not a regular file: ' + entry.path)
                continue
            if stat.S_ISDIR(facts.st_mode):
                yield from walk(entry.path)
            elif stat.S_ISREG(facts.st_mode):
                yield entry.path
            else:
                raise windows_api.WindowsApiError('managed state contains a non-regular entry: ' + entry.path)
    yield from walk(root, setup_root=True)


def setup_files():
    """Existing authority files that must not inherit runtime state access."""
    with os.scandir(paths.loki_state_dir()) as entries:
        for entry in entries:
            if not setup_asset_name(entry.name):
                continue
            facts = entry.stat(follow_symlinks=False)
            if (not stat.S_ISREG(facts.st_mode) or
                    getattr(facts, 'st_file_attributes', 0) & windows_api.FileAttribute.FILE_ATTRIBUTE_REPARSE_POINT):
                raise windows_api.WindowsApiError('setup metadata is not a regular file: ' + entry.path)
            yield entry.path


def state_writer_sids(ledger):
    """Only recorded state-writing packages may retain managed mutation grants."""
    return {windows_api.derive_app_container_sid(str(entry.get('profile', '')))
            for entry in ledger.get('workspaces', {}).values()
            if any(grant.origin == 'state' for grant in entry_grants(entry))}


STATE_MUTATION_MASK = int(windows_api.AccessMask.FILE_WRITE_DATA | windows_api.AccessMask.FILE_APPEND_DATA |
                          windows_api.AccessMask.FILE_WRITE_EA | windows_api.AccessMask.FILE_WRITE_ATTRIBUTES |
                          windows_api.AccessMask.FILE_DELETE_CHILD | windows_api.AccessMask.DELETE |
                          windows_api.AccessMask.WRITE_DAC | windows_api.AccessMask.WRITE_OWNER)


def state_acl_errors(sddl, user, packages):
    """Fail closed on every write-restriction alternative, including inherited ACEs.

    Normal owning-package access is deliberately writable. No private tool role,
    Everyone/Users/logon alternative or other unowned trustee may supply the
    second access check with mutation rights. Credential privacy is separate.
    """
    trusted = {windows_api.canonical_sid(sid) for sid in [user, 'SY', 'BA', *packages]}
    package_ids = {windows_api.canonical_sid(sid) for sid in packages}
    header, aces = dacl_parts(sddl)
    errors = []
    for ace in aces:
        kind, flags, rights, object_id, inherited_id, sid = ace[1:-1].split(';')
        if kind == 'D':
            continue
        if kind != 'A' or object_id or inherited_id:
            errors.append('unsupported state allow entry')
            continue
        identity = windows_api.canonical_sid(sid)
        mask = rights_mask(rights)
        # Hexadecimal ACE masks can retain generic bits too; they confer the
        # same file mutations as their textual GA/GW spellings.
        if mask & 0x10000000:  # GENERIC_ALL
            mask |= windows_api.AccessMask.FILE_ALL_ACCESS
        if mask & windows_api.AccessMask.GENERIC_WRITE:
            mask |= windows_api.AccessMask.FILE_GENERIC_WRITE
        if identity not in trusted and mask & STATE_MUTATION_MASK:
            errors.append('state grants mutation to ' + sid)
        if identity in package_ids and mask & int(windows_api.AccessMask.FILE_DELETE_CHILD |
                                                  windows_api.AccessMask.WRITE_DAC | windows_api.AccessMask.WRITE_OWNER):
            errors.append('state package can change protected setup entries')
    return errors


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


def _grant_identity(grant: Grant) -> str:
    try:
        return _directory_identity(grant.path)
    except windows_api.WindowsApiError as error:
        if (not isinstance(error, windows_api.WindowsResourceError)
                and _automatic_default(grant.path, grant.access, grant.origin)
                and error.status in [windows_api.ERROR_FILE_NOT_FOUND, windows_api.ERROR_PATH_NOT_FOUND]):
            # This is a pending automatic role, not a fabricated filesystem
            # identity. Apply creates/selects the object and rechecks its handle.
            return 'pending-default:' + grant.origin
        raise


def build_plan(definition: Definition, previous: dict | None) -> Plan:
    """Diff a definition against the recorded one, with rules applied."""
    errors = configuration_path_errors()
    warnings: list[str] = []
    for grant in definition.grants:
        errors.extend(protected_path_errors(grant.path, grant.access, origin=grant.origin))
        if grant.origin == "user":
            warnings.extend(covered_secret_warnings(grant.path))

    profile = ''
    try:
        profile = profile_name_for(definition.workspace)
        current = {_grant_identity(g): g for g in definition.grants}
        recorded = {_grant_identity(g): g for g in entry_grants(previous or {})}
    except windows_api.WindowsApiError as error:
        # An uninspectable recorded grant is not a removal we can safely apply.
        return Plan(definition.workspace, profile, [], warnings, errors + [str(error)])
    changes = []
    for key, grant in sorted(current.items()):
        if key not in recorded:
            changes.append(Change("add", grant.path, grant.access))
        elif recorded[key].access is not grant.access:
            changes.append(Change("level", grant.path, grant.access))
    for key, grant in sorted(recorded.items()):
        if key not in current:
            changes.append(Change("remove", grant.path, grant.access))
    return Plan(definition.workspace, profile, changes, warnings, errors)


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # 'pass' | 'fail' | 'untested'
    detail: str = ""


class Backend(Protocol):
    def apply(self, plan: Plan, definition: Definition) -> list[Check]: ...

    def verify(self, definition: Definition) -> list[Check]: ...

    def uninstall(self, blob: dict) -> list[Check]: ...
