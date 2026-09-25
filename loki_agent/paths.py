"""Shared filesystem locations used by Loki processes.

Credential supervisors and isolated runtimes must calculate the same path
without importing :mod:`loki_agent.loki`: importing the complete agent core
before runtime isolation would make the security boundary depend on import
order.  This small module is therefore the single source of truth for Loki's
locations: XDG on POSIX, the LocalAppData known folder on Windows.
"""

from __future__ import annotations

import os
import sys

from . import windows_api


LOKI_CONFIG_DIR_NAME = "loki"
CREDENTIAL_FILE_NAME = "tokens.json"
CREDENTIAL_LOCK_FILE_NAME = "tokens.lock"


def private_directory_support_error() -> str | None:
    """Return why private directory creation is unsupported here, else None.

    POSIX honors ``os.mkdir(mode)`` everywhere.  Windows only started honoring
    it in 3.11.10 / 3.12.4 / 3.13: before that the mode is ignored and a
    created directory inherits its parent's access, so a credential directory
    would be readable by other users on the machine rather than private.

    ``requires-python`` states the same floor, but only installers enforce it,
    so anything creating credential storage must ask here instead of relying on
    the package metadata.  Returning a message rather than raising keeps this
    module importable everywhere, including on the POSIX-only storage layer's
    future Windows counterpart.
    """
    if sys.platform != "win32":
        return None
    major, minor, micro = sys.version_info[:3]
    honored = (
        (major, minor) >= (3, 13)
        or ((major, minor) == (3, 12) and micro >= 4)
        or ((major, minor) == (3, 11) and micro >= 10)
    )
    if honored:
        return None
    return (
        f"Python {major}.{minor}.{micro} on Windows ignores os.mkdir(mode), so "
        "the credential directory would inherit its parent's access instead of "
        "private permissions; use Python 3.11.10 or later (3.12.4 or later on "
        "the 3.12 branch) on Windows"
    )


def config_base_directory(environ=None) -> str:
    """Return the directory that holds Loki's per-user state.

    ``XDG_CONFIG_HOME`` wins when it is explicitly set, on every platform: it
    is the documented override, and it is how callers and tests inject a
    location.  Otherwise POSIX falls back to ``~/.config`` and Windows asks for
    the LocalAppData known folder.

    Windows uses *local* rather than roaming app data because this tree holds
    credentials: a roaming profile is copied to a server, and secrets must not
    travel with it.
    """
    values = os.environ if environ is None else environ
    configured = values.get("XDG_CONFIG_HOME")
    if configured:
        return os.path.expanduser(configured)
    if sys.platform == "win32":
        return _windows_local_app_data()
    return os.path.expanduser("~/.config")


def _windows_local_app_data() -> str:
    resolved = windows_api.known_folder(
        windows_api.FOLDERID_LOCAL_APP_DATA,
        # Resolve the same path whether or not the caller is packaged or inside
        # an AppContainer, so the credential directory cannot move with the
        # calling context.
        windows_api.KnownFolderFlags.KF_FLAG_NO_PACKAGE_REDIRECTION)
    if resolved.startswith("\\\\"):
        # Folder redirection to a share cannot provide the private-directory
        # property credential storage depends on, so refuse rather than create
        # a credential directory somewhere the premise does not hold.
        raise windows_api.WindowsApiError(
            "LocalAppData is redirected to a network location "
            f"({resolved!r}); Loki's credential directory must be local")
    return resolved


def loki_config_dir(environ=None) -> str:
    return os.path.join(
        config_base_directory(environ), LOKI_CONFIG_DIR_NAME)


def state_base_directory(environ=None) -> str:
    """Return the directory that holds Loki's per-user state.

    POSIX separates state from configuration (``XDG_STATE_HOME``, defaulting to
    ``~/.local/state``), and that override wins when explicitly set.  Windows has
    no such split -- per-user application data lives under LocalAppData -- so
    there state resolves to the same base as the configuration directory.
    """
    values = os.environ if environ is None else environ
    configured = values.get("XDG_STATE_HOME")
    if configured:
        return os.path.expanduser(configured)
    if sys.platform == "win32":
        return _windows_local_app_data()
    return os.path.expanduser("~/.local/state")


def loki_state_dir(environ=None) -> str:
    return os.path.join(
        state_base_directory(environ), LOKI_CONFIG_DIR_NAME)


def credential_directory(environ=None) -> str:
    """Return the directory hidden from credential-consuming runtimes."""
    return os.path.join(loki_config_dir(environ), "credentials")
