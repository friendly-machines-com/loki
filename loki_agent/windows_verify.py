"""Read-only container verification, shared by the chat and the editor.

Everything reachable from the chat's start-up check only reads, so this module
must never import ``windows_setup`` or ``windows_containers``.  The probe below
-- a token check plus a deliberate escape attempt -- is meant to prove the
container is real; ``windows_setup`` explains why the import split is not
itself the boundary.
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
    entry = ledger.get("workspaces", {}).get(workspace_key(workspace), {})
    return verify_container(workspace, entry_grants(entry))


def verify_container(workspace: str, grants: list[Grant]) -> list[Check]:
    """Check that ``grants`` are in place for ``workspace``'s profile.

    Read-only: derive the SID from the name, read each DACL, and report the
    token self-test as untested until it exists.  The editor's Verify button and
    the chat's start-up check both come through here.
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
    # The token self-test is the only check that proves the container can
    # actually be entered; until it exists, say so instead of implying it passed.
    checks.append(Check("contained probe", "untested",
                        "token self-test not implemented yet"))
    return checks
