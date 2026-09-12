"""Shared filesystem locations used by Loki processes.

Credential supervisors and isolated runtimes must calculate the same path
without importing :mod:`loki_agent.loki`: importing the complete agent core
before runtime isolation would make the security boundary depend on import
order.  This small module is therefore the single source of truth for Loki's
XDG locations.
"""

from __future__ import annotations

import os
import sys


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


def xdg_config_home(environ=None) -> str:
    values = os.environ if environ is None else environ
    configured = values.get("XDG_CONFIG_HOME")
    return os.path.expanduser(configured or "~/.config")


def loki_config_dir(environ=None) -> str:
    return os.path.join(
        xdg_config_home(environ), LOKI_CONFIG_DIR_NAME)


def credential_directory(environ=None) -> str:
    """Return the directory hidden from credential-consuming runtimes."""
    return os.path.join(loki_config_dir(environ), "credentials")
