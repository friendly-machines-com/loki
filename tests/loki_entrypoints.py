"""The sanctioned entrypoints, in the form this platform can execute.

AGENTS.md names ``./loki.py`` and ``./loki-acp`` as the user-facing
entrypoints.  POSIX runs those scripts directly, through their shebang.
Windows has no shebang execution, so the same entrypoints exist there as
packaged executables: the ``loki.exe`` / ``loki-acp.exe`` console scripts that
``pip install`` writes from ``[project.scripts]``, or a PyInstaller build
pointed at by the override variables below.

A test must launch one of those.  Running the bare script on Windows fails with
``WinError 193``; prefixing the interpreter would run the module rather than
the entrypoint, which is not what ships and not what is being tested.
"""

from __future__ import annotations

import os
import shutil
import sysconfig


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per entrypoint: the POSIX script, the packaged executable name, and the
# environment variable a packaging job can use to point at its own build.
_ENTRYPOINTS = {
    "loki": ("loki.py", "LOKI_LAUNCHER"),
    "loki-acp": ("loki-acp", "LOKI_ACP_LAUNCHER"),
}


def _packaged(name: str) -> str:
    override = os.environ.get(_ENTRYPOINTS[name][1])
    candidates = [override] if override else []
    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidates.append(os.path.join(scripts, name + ".exe"))
    found = shutil.which(name)
    if found:
        candidates.append(found)
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        f"no packaged Windows entrypoint for {name!r}: install the package "
        f"(pip install -e .) or set {_ENTRYPOINTS[name][1]} to its build")


def entrypoint(name: str) -> str:
    """Absolute path to a sanctioned entrypoint, executable on this platform."""
    script, _override = _ENTRYPOINTS[name]
    if os.name == "nt":
        return _packaged(name)
    return os.path.join(ROOT, script)


def _on_windows() -> bool:
    return os.name == "nt"


def seed_container_ledger(environment: dict) -> None:
    """Give a Windows child the configured container ledger.

    The entrypoint's gate looks the ledger up under the runtime's *state*
    directory, and ``loki_state_dir`` honours ``XDG_STATE_HOME``.  A test that
    points ``XDG_STATE_HOME`` at a fresh directory is simulating a fresh state,
    so it must carry the configuration there; the container itself (profile and
    ACLs) is machine state and is already in place.  No-op off Windows, where
    there is no gate, and when the child keeps the real state directory.
    """
    if not _on_windows():
        return
    state_home = environment.get("XDG_STATE_HOME")
    if not state_home:
        return
    from loki_agent import windows_state

    target = os.path.join(state_home, "loki", "windows-setup.json")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    windows_state.save_ledger(windows_state.load_ledger(), target)


def child_environment(**values) -> dict:
    """A reduced environment that a Windows executable can still start in.

    The point of a reduced env is to control what the entrypoint sees, not to
    omit ``SystemRoot``, which the Windows loader needs before any Python runs.
    """
    environment = dict(values)
    if os.name == "nt":
        for name in ("SystemRoot", "SystemDrive"):
            if name in os.environ:
                environment.setdefault(name, os.environ[name])
    return environment


def loki_command() -> list[str]:
    return [entrypoint("loki")]


def loki_acp_command() -> list[str]:
    return [entrypoint("loki-acp")]
