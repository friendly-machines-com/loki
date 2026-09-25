"""The sanctioned entrypoints, in the form this platform can execute.

AGENTS.md names ``./loki.py`` and ``./loki-acp`` as the user-facing
entrypoints.  POSIX runs those scripts directly, through their shebang.
Windows has no shebang execution, so the same entrypoints exist there as the
PyInstaller executables from the release bundle, pointed at by the override
variables below.

A test must launch one of those.  Running the bare script on Windows fails with
``WinError 193``; an installed console script is not what ships and is not
tested here.
"""

from __future__ import annotations

import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per entrypoint: the POSIX script, the packaged executable name, and the
# environment variable the packaging job uses to point at its build.
_ENTRYPOINTS = {
    "loki": ("loki.py", "LOKI_LAUNCHER"),
    "loki-acp": ("loki-acp", "LOKI_ACP_LAUNCHER"),
    "loki-setup": ("loki-setup", "LOKI_SETUP_LAUNCHER"),
}


def _packaged(name: str) -> str:
    override_variable = _ENTRYPOINTS[name][1]
    build = os.environ.get(override_variable)
    if not build:
        raise RuntimeError(
            f"{override_variable} must point at the built {name} executable; "
            f"the suite launches the release, not an installed console script")
    if not os.path.exists(build):
        raise RuntimeError(f"{override_variable} does not exist: {build}")
    return build


def entrypoint(name: str) -> str:
    """Absolute path to a sanctioned entrypoint, executable on this platform."""
    script, _override = _ENTRYPOINTS[name]
    if os.name == "nt":
        return _packaged(name)
    return os.path.join(ROOT, script)


def child_environment(**values) -> dict:
    """A reduced environment that a Windows executable can still start in.

    The point of a reduced env is to control what the entrypoint sees, not to
    omit ``SystemRoot``, which the Windows loader needs before any Python runs,
    nor ``LOCALAPPDATA``, which the contained launch the entrypoint performs
    needs in its own block -- without it that ``CreateProcessW`` is refused
    with ``ERROR_ENVVAR_NOT_FOUND`` (203).  Both come from this process, so the
    entrypoint still sees a controlled environment.
    """
    environment = dict(values)
    if os.name == "nt":
        for name in ("SystemRoot", "SystemDrive", "LOCALAPPDATA"):
            if name in os.environ:
                environment.setdefault(name, os.environ[name])
    return environment


def configure_container(environment: dict, cwd: str) -> None:
    """Run the real setup for the environment an entrypoint child will get.

    The Windows gate looks up a ledger entry and verifies the credential,
    config and state trees *at the paths that environment resolves*.  A test
    that relocates them must therefore have setup create them there, exactly as
    a user would -- so this runs the shipped tool and nothing else.  It is
    deliberately not a fixture that writes a ledger or makes directories: the
    configuration must come from the code under test.

    Both ``XDG_CONFIG_HOME`` and ``XDG_STATE_HOME`` are required, so the trees
    and the ledger land inside the test's own directories instead of the
    runner's real user state.  No-op off Windows, where there is no gate.
    """
    if os.name != "nt":
        return
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME"):
        if not environment.get(name):
            raise RuntimeError(
                f"{name} must be set so that setup stays inside the test")
    result = subprocess.run(
        [entrypoint("loki-setup"), "--configure", cwd],
        cwd=cwd, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"loki-setup --configure failed for {cwd!r}: "
            f"{result.stdout}{result.stderr}")
    _report_workspace_access(cwd)


def _powershell_sddl(path: str) -> str:
    """The same object's SDDL as PowerShell renders it, SACL included.

    ``label_sddl`` reads the mandatory label directly; this is the independent
    second opinion, because a single reader that mis-parses would otherwise be
    its own evidence for what the label says.
    """
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             f"(Get-Acl -LiteralPath '{path}').Sddl"],
            capture_output=True, text=True, timeout=60)
    except OSError as error:
        return f"<{type(error).__name__}: {error}>"
    if result.returncode != 0:
        return f"<exit {result.returncode}: {result.stderr.strip()}>"
    return result.stdout.strip()


def _report_workspace_access(workspace: str) -> None:
    """Print the descriptor a contained worker will meet in ``workspace``.

    A failed session reports only the Win32 error, so this prints what the
    error does not say: the package SID this workspace derives, whether it is
    granted on the workspace at all, whether that grant is inheritable, and
    whether ``.loki`` already exists with a descriptor of its own.
    """
    try:
        from loki_agent import windows_api, windows_state

        package = windows_api.derive_app_container_sid(
            windows_state.profile_name_for(workspace))
        print(f"[workspace access] package={package}")
        for path in (workspace, os.path.join(workspace, ".loki")):
            try:
                sddl = windows_api.dacl_sddl(path)
            except Exception as error:  # noqa: BLE001 - diagnostic only
                sddl = f"<{type(error).__name__}: {error}>"
            try:
                label = windows_api.label_sddl(path)
            except Exception as error:  # noqa: BLE001 - diagnostic only
                label = f"<{type(error).__name__}: {error}>"
            print(f"[workspace access] {path} exists="
                  f"{os.path.exists(path)} dacl={sddl} label={label} "
                  f"acl={_powershell_sddl(path)}")
    except Exception as error:  # noqa: BLE001 - never fail a test for this
        print(f"[workspace access] unavailable: "
              f"{type(error).__name__}: {error}")


def loki_command() -> list[str]:
    return [entrypoint("loki")]


def loki_acp_command() -> list[str]:
    return [entrypoint("loki-acp")]
