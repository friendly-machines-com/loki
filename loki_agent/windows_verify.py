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

from . import paths
from . import windows_api
from .windows_state import (
    Check,
    Grant,
    entry_grants,
    grants_package,
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
        except windows_api.WindowsApiError as error:
            checks.append(Check(f"grant {grant.path}", "fail", str(error)))
            continue
        granted = grants_package(sddl, package)
        checks.append(Check(
            f"grant {grant.path}", "pass" if granted else "fail",
            grant.access.value if granted else "no ACE for the package SID"))
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
    return checks


def probe_containment(workspace: str) -> list[Check]:
    """Check, from inside the container, that this process is contained.

    Read-only in effect: each denial is attempted and then closed without being
    used, and nothing is read or written.  A granted path is opened too, as a
    positive control -- a process denied *everything* would otherwise make every
    denial below look like success.
    """
    checks = _identity_checks(workspace)
    checks.append(_reachable(
        "workspace reachable", workspace,
        windows_api.GENERIC_READ | windows_api.GENERIC_WRITE,
        "the workspace opens read-write"))
    checks.append(_denied(
        "credentials unreadable", paths.credential_directory(),
        windows_api.GENERIC_READ, "opening the credential directory is denied"))
    checks.append(_denied(
        "cannot rewrite a DACL", workspace, windows_api.WRITE_DAC,
        "requesting WRITE_DAC on a granted path is denied"))
    return checks


def _identity_checks(workspace: str) -> list[Check]:
    """Check the process token, before asking about anything on disk."""
    try:
        expected = windows_api.derive_app_container_sid(
            profile_name_for(workspace))
    except windows_api.WindowsApiError as error:
        return [Check("package SID", "fail", str(error))]
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
            matches = package == expected
            checks.append(Check(
                "package SID", "pass" if matches else "fail",
                package if matches
                else f"token has {package}, expected {expected}"))
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
    error = _attempt(path, desired_access)
    if error is None:
        return Check(name, "fail", f"{description}: access was granted")
    if error.status == windows_api.ERROR_ACCESS_DENIED:
        return Check(name, "pass", description)
    return Check(name, "fail", f"{description}: {error}")
