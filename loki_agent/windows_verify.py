"""Container verification: the DACL inventory and the runtime self-check.

Shared by the chat and the editor, and this module must never import
``windows_setup`` or ``windows_containers``: everything here either reads a DACL
or asks for access it must be denied, closing the handle without using it.

``verify_container`` is the uncontained view -- used by the editor's Verify
button, and intended for the launcher once that exists -- and checks that the
grants and the protected trees are what they should be.  ``probe_containment``
is the check for inside the runtime: that its token is the AppContainer, and
that the rights it must not have are refused.  ``windows_setup`` explains why
the import split is not itself the boundary.
"""

from __future__ import annotations

import os

from . import paths
from . import windows_api
from .windows_acl import (
    dacl_parts,
    package_access,
    package_allow,
)
from .windows_state import (
    Access,
    Check,
    Grant,
    access_mask,
    configuration_path_errors,
    entry_grants,
    grants_access,
    profile_name_for,
    ledger_path,
    setup_files,
    state_objects,
    state_writer_sids,
    state_acl_errors,
    tool_role_sid,
    workspace_key,
)


def verify_workspace(ledger: dict, workspace: str) -> list[Check]:
    """Verify the container recorded for ``workspace``, changing nothing."""
    entry = ledger.get("workspaces", {}).get(workspace_key(workspace))
    if not isinstance(entry, dict):
        return [Check("ledger", "fail", "workspace is not configured")]
    if any(item.get('pending') for item in ledger.get('workspaces', {}).values()):
        return [Check("ledger", "fail", "setup or uninstall is incomplete; retry setup")]
    return verify_container(workspace, entry_grants(entry), ledger=ledger)


def verify_container(workspace: str, grants: list[Grant], *, ledger=None) -> list[Check]:
    """Check that ``grants`` are in place for ``workspace``'s profile.

    Read-only and safe to run uncontained, which is why it reads DACLs rather
    than asking for access.  The editor's Verify button calls this; the
    runtime's own check is :func:`probe_containment`.
    """
    errors = configuration_path_errors()
    if errors:
        return [Check("configuration layout", "fail", "; ".join(errors))]
    profile = profile_name_for(workspace)
    checks: list[Check] = []
    try:
        package = windows_api.derive_app_container_sid(profile)
    except windows_api.WindowsApiError as error:
        return [Check("profile SID", "fail", str(error))]
    checks.append(Check("profile SID", "pass", package))
    for grant in grants:
        try:
            sddl = windows_api.dacl_sddl(grant.path)
            granted = grants_access(sddl, package, grant.access)
            if grant.access is Access.READ_WRITE and grant.origin != 'state':
                granted = granted and grants_access(sddl, tool_role_sid(package), Access.READ_WRITE)
        except (windows_api.WindowsApiError, ValueError) as error:
            checks.append(Check(f"grant {grant.path}", "fail", str(error)))
            continue
        checks.append(Check(
            f"grant {grant.path}", "pass" if granted else "fail",
            grant.access.value if granted
            else _grant_failure(sddl, package, grant.access)))
    try:
        sddl = windows_api.dacl_sddl(paths.loki_config_dir())
        read_only = (grants_access(sddl, package, Access.READ)
                     and not package_allow(sddl, package) & ~access_mask(Access.READ))
        checks.append(Check(
            "config", "pass" if read_only else "fail",
            "configuration is readable without write access" if read_only
            else "configuration must grant read access only"))
    except windows_api.WindowsApiError as error:
        checks.append(Check("config", "fail", str(error)))
    for tree, name in [[paths.credential_directory(), "credentials"]]:
        try:
            sddl = windows_api.dacl_sddl(tree)
            allowed = package_allow(sddl, package)
            header, _aces = dacl_parts(sddl)
        except windows_api.WindowsApiError as error:
            checks.append(Check(name, "fail", str(error)))
            continue
        if allowed:
            checks.append(Check(
                name, "fail", f"package SID is granted {allowed:#x}"))
        elif "P" not in header:
            checks.append(Check(
                name, "fail", "private directory must block inherited grants"))
        else:
            checks.append(Check(
                name, "pass", "the package SID holds no access; inheritance is blocked"))
    try:
        writers = state_writer_sids(ledger) if ledger is not None else {package}
        user = windows_api.current_user_sid()
        for path in state_objects():
            try:
                sddl = windows_api.dacl_sddl(path)
                errors = state_acl_errors(sddl, user, writers)
                if path == paths.loki_state_dir() and 'P' not in dacl_parts(sddl)[0]:
                    errors.append('managed state must block ancestor grants')
                if not grants_access(sddl, package, Access.READ_WRITE):
                    errors.append('owning runtime cannot maintain this state object')
                checks.append(Check('state' if path == paths.loki_state_dir() else 'state ' + path, 'fail' if errors else 'pass',
                                    '; '.join(errors) if errors else 'runtime writable; tools read-only'))
            except (windows_api.WindowsApiError, ValueError) as error:
                checks.append(Check('state' if path == paths.loki_state_dir() else 'state ' + path, 'fail', str(error)))
        assets = list(setup_files())
        if ledger_path() not in assets:
            checks.append(Check('setup metadata', 'fail', 'setup metadata is missing'))
        for path in assets:
            sddl = windows_api.dacl_sddl(path)
            header, aces = dacl_parts(sddl)
            trusted = {windows_api.canonical_sid(sid) for sid in [user, 'SY', 'BA']}
            private = 'P' in header and all(
                ace[1:-1].split(';')[0] == 'D' or
                (ace[1:-1].split(';')[0] == 'A' and windows_api.canonical_sid(ace[1:-1].split(';')[5]) in trusted)
                for ace in aces)
            checks.append(Check('setup metadata ' + path, 'pass' if private else 'fail',
                                'setup metadata remains private' if private else 'setup metadata exposes runtime access'))
    except (OSError, windows_api.WindowsApiError, ValueError) as error:
        checks.append(Check('state', 'fail', str(error)))
    return checks


def _grant_failure(sddl: str, package: str, access: Access) -> str:
    """Explain why a grant's DACL does not confer ``access`` on ``package``."""
    held = package_access(sddl, package)
    if held == 0:
        return "no allow ACE applies to the package SID"
    required = access_mask(access)
    return (f"the package SID holds {held:#x}, which does not cover "
            f"{access.value} ({required:#x})")


def probe_containment(workspace: str, expected_package: str) -> list[Check]:
    """Check, from inside the container, that this process is contained.

    ``expected_package`` is the package SID the launcher verified this
    process's token against before resuming it.  The child must not re-derive
    it: that needs ``GetFinalPathNameByHandleW`` on the workspace, which an
    AppContainer token is denied (``ERROR_ACCESS_DENIED``), and re-deriving it
    was what made the gate fail at startup.

    Read-only in effect: each denial is attempted and then closed without being
    used, and nothing is read or written.  A granted path is opened too, as a
    positive control -- a process denied *everything* would otherwise make every
    denial below look like success.
    """
    checks = _identity_checks(expected_package)
    checks.append(_reachable(
        "workspace reachable", workspace,
        windows_api.AccessMask.GENERIC_READ
        | windows_api.AccessMask.GENERIC_WRITE,
        "the workspace opens read-write"))
    checks.extend(_configuration_checks())
    checks.extend(_state_checks())
    checks.extend(_credential_checks())
    checks.append(_denied(
        "cannot rewrite a DACL", workspace, windows_api.AccessMask.WRITE_DAC,
        "requesting WRITE_DAC on a granted path is denied"))
    return checks


# The directions the credential *directory* must refuse.  Listing and creating
# are the entry checks; delete, DACL and owner are the mutations, which matter
# because a container that could delete the directory or rewrite its DACL would
# defeat the store without ever reading a token.
_CREDENTIAL_DIRECTORY_PROBES = (
    ("credentials unlistable", windows_api.AccessMask.GENERIC_READ,
     "listing the credential directory is denied"),
    # Creating an entry is the claim that still holds on a fresh install, when
    # there is no credential file to open.  A directory's FILE_WRITE_DATA is
    # FILE_ADD_FILE, so the open tests create rights without creating anything.
    ("cannot create credentials", windows_api.AccessMask.FILE_WRITE_DATA,
     "creating an entry in the credential directory is denied"),
    ("credential directory not deletable", windows_api.AccessMask.DELETE,
     "the credential directory cannot be opened to delete it"),
    ("credential directory DACL not rewritable",
     windows_api.AccessMask.WRITE_DAC,
     "the credential directory cannot be opened to rewrite its DACL"),
    ("credential directory owner not rewritable",
     windows_api.AccessMask.WRITE_OWNER,
     "the credential directory cannot be opened to change its owner"),
)

# File mutation checks shared by private credentials and readable configuration.
# Opening and closing without using the handle does not mutate the object.
_FILE_MUTATION_PROBES = [
    ["not writable", windows_api.AccessMask.GENERIC_WRITE,
     "cannot be opened for write"],
    ["data not writable", windows_api.AccessMask.FILE_WRITE_DATA,
     "cannot be opened to change its contents"],
    ["attributes not writable", windows_api.AccessMask.FILE_WRITE_ATTRIBUTES,
     "cannot be opened to change its attributes"],
    ["extended attributes not writable", windows_api.AccessMask.FILE_WRITE_EA,
     "cannot be opened to change its extended attributes"],
    ["not appendable", windows_api.AccessMask.FILE_APPEND_DATA,
     "cannot be opened for append"],
    ["not deletable", windows_api.AccessMask.DELETE,
     "cannot be opened to delete it"],
    ["DACL not rewritable", windows_api.AccessMask.WRITE_DAC,
     "cannot be opened to rewrite its DACL"],
    ["owner not rewritable", windows_api.AccessMask.WRITE_OWNER,
     "cannot be opened to change its owner"],
]

# Both files the storage uses, not just the JSON: the lock is a different
# object with its own DACL, and a container that can write or delete it can
# interfere with the read-modify-write transaction even without reading a token.
_CREDENTIAL_FILES = (
    (paths.CREDENTIAL_FILE_NAME, "credential file"),
    (paths.CREDENTIAL_LOCK_FILE_NAME, "credential lock"),
)


def _state_checks() -> list[Check]:
    """Owning runtimes write state; tool tokens only read it by pathname.

    Preopened stdout/stderr are separate file capabilities. Opening a path here
    must never manufacture their exception for another job or metadata object.
    """
    try:
        token = windows_api.open_process_token(windows_api.current_process_handle())
        try:
            package = windows_api.token_app_container_sid(token)
            role = tool_role_sid(package)
            tool = role in [sid for sid, attributes in windows_api.token_restricting_sids(token)]
        finally:
            windows_api.close_handle(token)
    except (windows_api.WindowsApiError, ValueError) as error:
        return [Check('state role', 'fail', str(error))]
    directory = paths.loki_state_dir()
    checks = [_reachable('state readable', directory, windows_api.AccessMask.GENERIC_READ,
                         'runtime state is readable')]
    if tool:
        mutations = {
            'state not writable': windows_api.AccessMask.GENERIC_WRITE,
            'state files not creatable': windows_api.AccessMask.FILE_WRITE_DATA,
            'state directories not creatable': windows_api.AccessMask.FILE_APPEND_DATA,
            'state children not deletable': windows_api.AccessMask.FILE_DELETE_CHILD,
            'state directory not deletable': windows_api.AccessMask.DELETE,
            'state attributes not writable': windows_api.AccessMask.FILE_WRITE_ATTRIBUTES,
            'state extended attributes not writable': windows_api.AccessMask.FILE_WRITE_EA,
            'state DACL not rewritable': windows_api.AccessMask.WRITE_DAC,
            'state owner not rewritable': windows_api.AccessMask.WRITE_OWNER,
        }
        checks.extend(_denied(name, directory, access, name) for name, access in mutations.items())
    else:
        checks.append(_reachable('owning state writable', directory, windows_api.AccessMask.GENERIC_WRITE,
                                 'the owning runtime can maintain its state'))
    checks.append(_denied('setup metadata unreadable', ledger_path(), windows_api.AccessMask.GENERIC_READ,
                          'setup metadata is private to setup and the supervisor'))
    for suffix, access, description in _FILE_MUTATION_PROBES:
        checks.append(_denied('setup metadata ' + suffix, ledger_path(), access, 'setup metadata ' + description))
    return checks


def _configuration_checks() -> list[Check]:
    """Read configuration without gaining persistent mutation rights."""
    directory = paths.loki_config_dir()
    checks = [_reachable(
        "configuration readable", directory, windows_api.AccessMask.GENERIC_READ,
        "the configuration directory opens for read")]
    mutations = {
        "cannot create configuration files": windows_api.AccessMask.FILE_WRITE_DATA,
        "cannot create configuration directories": windows_api.AccessMask.FILE_APPEND_DATA,
        "cannot delete configuration children": windows_api.AccessMask.FILE_DELETE_CHILD,
        "configuration attributes not writable": windows_api.AccessMask.FILE_WRITE_ATTRIBUTES,
        "configuration extended attributes not writable": windows_api.AccessMask.FILE_WRITE_EA,
        "configuration directory not deletable": windows_api.AccessMask.DELETE,
        "configuration DACL not rewritable": windows_api.AccessMask.WRITE_DAC,
        "configuration owner not rewritable": windows_api.AccessMask.WRITE_OWNER,
    }
    checks.extend(_denied(name, directory, access, name)
                  for name, access in mutations.items())
    # Optional files may be missing or deliberately unreadable. Their consumers
    # report read errors; here only mutation rights must always be refused.
    for name in ["SYSTEM.md", "settings.ini", "hooks.json"]:
        file = os.path.join(directory, name)
        error = _attempt(file, windows_api.AccessMask.GENERIC_READ)
        if error is not None and error.status in [
                windows_api.ERROR_FILE_NOT_FOUND, windows_api.ERROR_PATH_NOT_FOUND]:
            continue
        for suffix, access, description in _FILE_MUTATION_PROBES:
            checks.append(_denied(
                f"{name} {suffix}", file, access, f"{name} {description}"))
    return checks


def _credential_checks() -> list[Check]:
    """Denials for the credential directory and the files it holds.

    The directory and each file are different objects with different access
    checks, so listing the directory says nothing about reading ``tokens.json``;
    every object is opened for each right the gate must refuse.  Opening the
    real path is also what governs a read through a workspace hard link or
    junction: the target file's DACL is checked, not the path's.
    """
    directory = paths.credential_directory()
    checks = [_denied(name, directory, access, description)
              for name, access, description in _CREDENTIAL_DIRECTORY_PROBES]
    for file_name, label in _CREDENTIAL_FILES:
        checks.extend(_credential_file_checks(
            os.path.join(directory, file_name), label))
    return checks


def _credential_file_checks(asset: str, label: str) -> list[Check]:
    """Denials for one credential file, or a pass when it does not exist yet."""
    error = _attempt(asset, windows_api.AccessMask.GENERIC_READ)
    if error is not None and error.status in [
            windows_api.ERROR_FILE_NOT_FOUND,
            windows_api.ERROR_PATH_NOT_FOUND]:
        # Nothing to leak, and the directory refuses to let the container make
        # one; the per-direction denials have no object to test.
        return [Check(
            label, "pass",
            f"no {label} exists yet; creating one is denied")]
    checks = [_denied_from(
        f"{label} unreadable", error,
        f"the {label} cannot be opened for read")]
    for suffix, access, description in _FILE_MUTATION_PROBES:
        checks.append(_denied(
            f"{label} {suffix}", asset, access,
            f"the {label} {description}"))
    return checks


def _identity_checks(expected_package: str) -> list[Check]:
    """Check the process token, before asking about anything on disk."""
    try:
        token = windows_api.open_process_token(
            windows_api.current_process_handle())
    except windows_api.WindowsApiError as error:
        return [Check("process token", "fail", str(error))]
    try:
        checks: list[Check] = []
        try:
            contained = windows_api.token_is_app_container(token)
        except windows_api.WindowsApiError as error:
            checks.append(Check("AppContainer", "fail", str(error)))
        else:
            checks.append(Check(
                "AppContainer", "pass" if contained else "fail",
                "the token is an AppContainer" if contained
                else "the token is not an AppContainer"))
        try:
            package = windows_api.token_app_container_sid(token)
        except windows_api.WindowsApiError as error:
            checks.append(Check("package SID", "fail", str(error)))
        else:
            matches = package == expected_package
            checks.append(Check(
                "package SID", "pass" if matches else "fail",
                package if matches
                else f"token has {package}, expected {expected_package}"))
        return checks
    finally:
        windows_api.close_handle(token)


def _attempt(path: str, desired_access: int):
    """Open ``path`` with ``desired_access`` and close it.

    Returns the :class:`WindowsApiError` when the open was refused, or ``None``
    when it succeeded; nothing is done with the handle either way.
    """
    try:
        handle = windows_api.open_with_access(path, desired_access)
    except windows_api.WindowsApiError as error:
        return error
    windows_api.close_handle(handle)
    return None


def _reachable(name: str, path: str, desired_access: int,
               description: str) -> Check:
    """A positive control: the access must be granted."""
    error = _attempt(path, desired_access)
    if error is None:
        return Check(name, "pass", description)
    return Check(name, "fail", f"{description}: {error}")


def _denied(name: str, path: str, desired_access: int,
            description: str) -> Check:
    """A denial: the access must be refused, specifically as access denied."""
    return _denied_from(name, _attempt(path, desired_access), description)


def _denied_from(name: str, error, description: str) -> Check:
    """A denial decided from an already-attempted open.

    Split out so the credential file's read attempt can double as the check
    that it exists: re-opening to learn the same error would be a second scan
    of the same asset.
    """
    if error is None:
        return Check(name, "fail", f"{description}: access was granted")
    if error.status == windows_api.ERROR_ACCESS_DENIED:
        return Check(name, "pass", description)
    return Check(name, "fail", f"{description}: {error}")
