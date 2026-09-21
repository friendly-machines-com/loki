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
    package_access,
    package_allow,
)
from .windows_state import (
    Access,
    Check,
    Grant,
    access_mask,
    entry_grants,
    grants_access,
    profile_name_for,
    workspace_key,
)


def verify_workspace(ledger: dict, workspace: str) -> list[Check]:
    """Verify the container recorded for ``workspace``, changing nothing."""
    entry = ledger.get("workspaces", {}).get(workspace_key(workspace))
    if not isinstance(entry, dict):
        return [Check("ledger", "fail", "workspace is not configured")]
    if entry.get("pending"):
        return [Check("ledger", "fail", "setup or uninstall is incomplete; retry setup")]
    return verify_container(workspace, entry_grants(entry))


def verify_container(workspace: str, grants: list[Grant]) -> list[Check]:
    """Check that ``grants`` are in place for ``workspace``'s profile.

    Read-only and safe to run uncontained, which is why it reads DACLs rather
    than asking for access.  The editor's Verify button calls this; the
    runtime's own check is :func:`probe_containment`.
    """
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
        except windows_api.WindowsApiError as error:
            checks.append(Check(f"grant {grant.path}", "fail", str(error)))
            continue
        checks.append(Check(
            f"grant {grant.path}", "pass" if granted else "fail",
            grant.access.value if granted
            else _grant_failure(sddl, package, grant.access)))
    for tree, name in ((paths.credential_directory(), "credentials"),
                       (paths.loki_config_dir(), "config"),
                       (paths.loki_state_dir(), "state")):
        try:
            sddl = windows_api.dacl_sddl(tree)
            allowed = package_allow(sddl, package)
        except windows_api.WindowsApiError as error:
            checks.append(Check(name, "fail", str(error)))
            continue
        if allowed:
            checks.append(Check(
                name, "fail", f"package SID is granted {allowed:#x}"))
        else:
            checks.append(Check(
                name, "pass", "the package SID holds no access"))
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
        windows_api.GENERIC_READ | windows_api.GENERIC_WRITE,
        "the workspace opens read-write"))
    checks.extend(_credential_checks())
    checks.append(_denied(
        "cannot rewrite a DACL", workspace, windows_api.WRITE_DAC,
        "requesting WRITE_DAC on a granted path is denied"))
    return checks


# The directions the credential *directory* must refuse.  Listing and creating
# are the entry checks; delete, DACL and owner are the mutations, which matter
# because a container that could delete the directory or rewrite its DACL would
# defeat the store without ever reading a token.
_CREDENTIAL_DIRECTORY_PROBES = (
    ("credentials unlistable", windows_api.GENERIC_READ,
     "listing the credential directory is denied"),
    # Creating an entry is the claim that still holds on a fresh install, when
    # there is no credential file to open.  A directory's FILE_WRITE_DATA is
    # FILE_ADD_FILE, so the open tests create rights without creating anything.
    ("cannot create credentials", windows_api.FILE_WRITE_DATA,
     "creating an entry in the credential directory is denied"),
    ("credential directory not deletable", windows_api.DELETE,
     "the credential directory cannot be opened to delete it"),
    ("credential directory DACL not rewritable", windows_api.WRITE_DAC,
     "the credential directory cannot be opened to rewrite its DACL"),
    ("credential directory owner not rewritable", windows_api.WRITE_OWNER,
     "the credential directory cannot be opened to change its owner"),
)

# The directions each credential *file* must refuse, beyond the read that also
# detects whether it exists.  Opening and closing without using the handle
# cannot write, truncate, delete or re-own anything: the access check happens
# at open, so each entry measures permission without performing the act.
_CREDENTIAL_FILE_PROBES = (
    ("not writable", windows_api.GENERIC_WRITE,
     "cannot be opened for write"),
    ("not appendable", windows_api.FILE_APPEND_DATA,
     "cannot be opened for append"),
    ("not deletable", windows_api.DELETE,
     "cannot be opened to delete it"),
    ("DACL not rewritable", windows_api.WRITE_DAC,
     "cannot be opened to rewrite its DACL"),
    ("owner not rewritable", windows_api.WRITE_OWNER,
     "cannot be opened to change its owner"),
)

# Both files the storage uses, not just the JSON: the lock is a different
# object with its own DACL, and a container that can write or delete it can
# interfere with the read-modify-write transaction even without reading a token.
_CREDENTIAL_FILES = (
    (paths.CREDENTIAL_FILE_NAME, "credential file"),
    (paths.CREDENTIAL_LOCK_FILE_NAME, "credential lock"),
)


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
    error = _attempt(asset, windows_api.GENERIC_READ)
    if error is not None and error.status in (
            windows_api.ERROR_FILE_NOT_FOUND,
            windows_api.ERROR_PATH_NOT_FOUND):
        # Nothing to leak, and the directory refuses to let the container make
        # one; the per-direction denials have no object to test.
        return [Check(
            label, "pass",
            f"no {label} exists yet; creating one is denied")]
    checks = [_denied_from(
        f"{label} unreadable", error,
        f"the {label} cannot be opened for read")]
    for suffix, access, description in _CREDENTIAL_FILE_PROBES:
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
