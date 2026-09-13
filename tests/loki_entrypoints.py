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
import subprocess
import sys
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
        [sys.executable, "-m", "loki_agent.windows_setup", "--configure", cwd],
        cwd=cwd, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"loki-setup --configure failed for {cwd!r}: "
            f"{result.stdout}{result.stderr}")


def loki_command() -> list[str]:
    return [entrypoint("loki")]


def loki_acp_command() -> list[str]:
    return [entrypoint("loki-acp")]
