"""Shared filesystem locations used by Loki processes.

Credential supervisors and isolated runtimes must calculate the same path
without importing :mod:`loki_agent.loki`: importing the complete agent core
before runtime isolation would make the security boundary depend on import
order.  This small module is therefore the single source of truth for Loki's
XDG locations.
"""

from __future__ import annotations

import os


LOKI_CONFIG_DIR_NAME = "loki"
CREDENTIAL_FILE_NAME = "tokens.json"
CREDENTIAL_LOCK_FILE_NAME = "tokens.lock"


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


def credential_file(environ=None) -> str:
    return os.path.join(
        credential_directory(environ), CREDENTIAL_FILE_NAME)


def credential_lock_file(environ=None) -> str:
    return os.path.join(
        credential_directory(environ), CREDENTIAL_LOCK_FILE_NAME)
