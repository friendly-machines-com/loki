# Open Build Service packaging

These files are the OBS package sources for `loki` (Python distribution
`loki-agent`, import package `loki_agent`). They install `loki`, `loki-acp`, and
`loki-setup`. The license is MIT; `COPYING` is the upstream license text.
The project has no Python runtime dependencies.

## Upload to OBS

1. Create a `loki` package in your OBS project. If it currently uses SCM Sync,
   clear its SCM Sync URL before uploading these files directly.
2. In an `osc checkout` of that package, copy the files from this directory
   (except this README) into the package root. Preserve the executable bit on
   `debian.rules`, and include `loki.changes` so OBS can populate RPM changelog
   metadata. Run `osc addremove` and `osc commit`.
3. Run the source service using `osc service remoterun` (or **Run services** in
   the web UI), then wait for it to finish. OBS needs the `obs_scm`, `tar`,
   `recompress`, and `set_version` source services on the source server, not in
   each distribution's build repository. For local builds, download the generated
   files with `osc update -S` after the remote services have finished.
4. Add the desired distribution repositories in the project's **Repositories**
   tab. Package files do not configure repositories automatically.

`_service` fetches the public upstream `master` branch and produces
`loki-0.1.0.tar.gz`, with a `loki-0.1.0/` top-level directory. Tar creation,
compression, and version substitution run server-side before distro builds.
The target build roots do not need `obs-service-tar`, `obs-service-recompress`,
or `obs-service-set_version`. OBS strips service-output filename prefixes before
invoking the distro build tools. For reproducible releases, change
`revision` to the desired upstream commit or tag before running the service.
The Arch checksum is deliberately `SKIP` because OBS generates the source
archive; a standalone AUR release should instead use a pinned archive and hash.

When upstream's `loki_agent/__init__.py` version changes, update `versionformat`
in `_service`, `Version` in the spec and dsc, `pkgver` in PKGBUILD, and the Debian
changelog together. The service version is intentionally explicit: a Git
snapshot timestamp must not silently disagree with the wheel's version.
`set_version` is not a replacement for maintaining the Debian changelog.

## Distribution requirements

- **openSUSE Tumbleweed / Slowroll:** CPython via `%{primary_python}`; no modern
  PyPy interpreter was found in the checked official repositories.
- **Leap 15.x:** the spec selects `python311`. The repository must supply
  Python **3.11.10 or newer** and that interpreter's Flit, pip, and wheel
  packages. A `python311` package name alone does not guarantee the patch level.
- **Fedora 44 / Rawhide:** builds with `python3` and `pyproject-rpm-macros`,
  but installs for the distro's `/usr/bin/pypy3.11` (`pypy3.11` package).
  Other RPM targets default to CPython unless explicitly enabled.
- **RHEL-family:** the default Python on RHEL 9 is too old. This spec requires
  repositories whose `python3` is new enough and which provide the listed build
  dependencies; it does not automatically select RHEL's parallel Python stacks.
- **Debian 13 / Unstable / Ubuntu 26.04:** use distro `pypy3` for the installed
  commands. The ordinary CPython/pybuild installation is retained: Debian PyPy's
  `deb_system` scheme shares `/usr/lib/python3/dist-packages`. `dh_python3`
  rewrites the shebangs; its normal CPython/debhelper dependencies remain.
  The build dependency is named `flit`: that Debian/Ubuntu package contains
  `flit_core/buildapi.py`; there is no separate `python3-flit-core` dependency.
  The flat `debian.*` files and `loki.dsc`
  are OBS debtransform inputs, not an already assembled Debian source package.
  `set_version` substitutes the dsc's `@VERSION@` archive selector before
  debtransform constructs the Debian source package.
- **Arch:** builds with CPython without build isolation, then
  installs for the distro's `/usr/bin/pypy3` using `install-wheels.py`.

Upstream requires `>=3.11.10` and excludes Python `3.12.0` through `3.12.3`.
Unmodified Debian 12 (3.11.2) and Ubuntu 24.04 (3.12.3) therefore are not targets.
Do not relax upstream's requirement merely to make an older repository build.
RPM and Debian runtime dependencies retain these exclusions. Arch rolling
repositories are expected to provide a current Python.

All recipes use **Flit**, not setuptools. Upstream requires `flit_core >=3.8`
with no upper bound. Distribution builds use the installed backend rather than
creating an isolated environment; Arch no longer needs to skip dependency checks.

## Interpreter policy

Prefer **distro-maintained, Loki-compatible PyPy**, not just any package named
`pypy3`. PyPy's own version is different from the Python language version. The
checked packages provide:

| Target | Distro PyPy package | Python language version |
| --- | --- | --- |
| Arch Extra | pypy3 8.0.1 | 3.12.15 |
| Fedora 44 | pypy3.11 8.0.0 | 3.11.16 |
| Debian 13 | pypy3 7.3.19+dfsg-2 | 3.11.11 |
| Debian Unstable | pypy3 8.0.1+dfsg-1 | 3.11.17 |
| Ubuntu 26.04 | pypy3 7.3.20+dfsg-4 | 3.11.13 |

Sources: [Arch](https://archlinux.org/packages/extra/x86_64/pypy3/),
[Fedora spec](https://src.fedoraproject.org/rpms/pypy3.11/raw/f44/f/pypy3.11.spec)
and [its upstream language version](https://raw.githubusercontent.com/pypy/pypy/release-pypy3.11-v8.0.0/pypy/module/sys/version.py),
[Debian 13 source](https://sources.debian.org/src/pypy3/7.3.19%2Bdfsg-2/pypy/module/sys/version.py/),
[Debian Unstable source](https://sources.debian.org/src/pypy3/8.0.1%2Bdfsg-1/pypy/module/sys/version.py/),
[Ubuntu source](https://git.launchpad.net/ubuntu/+source/pypy3/plain/pypy/module/sys/version.py?h=ubuntu/resolute).
These are repository/source checks, not completed native OBS builds.

Interpreter choice is made **when building the distro package**. There is no
runtime PATH probe or launcher fallback. The selected package dependency and
absolute shebang agree; `loki`, `loki-acp`, and the Windows-only `loki-setup`
console script all use that interpreter. Existing internal launches propagate
that interpreter via `sys.executable`. Upstream checkout shebangs are unchanged.

RPM and Arch still build their universal wheel with CPython. `install-wheels.py`
uses CPython's `installer`, but queries the selected runtime for its installation
scheme and generates that runtime's shebangs. It rejects binary wheels and a
runtime outside the wheel's `Requires-Python` range before installing. It does
not generate CPython bytecode for PyPy. No PyPy-specific build backend, interpreter
source download, or locally maintained PyPy package is needed.

For repositories or architectures without a suitable distro PyPy, select CPython
at package-build configuration time:

- **RPM:** `--without pypy` (or `%_without_pypy 1` in OBS project macros).
  openSUSE and RHEL-family targets already default to CPython. Enabling PyPy on
  another verified RPM repository uses `--with pypy`; its package/executable
  names can be set with `loki_pypy_package` and `loki_pypy_executable` macros.
- **Debian:** activate the `cpython` build profile, e.g.
  `DEB_BUILD_PROFILES=cpython` with normal Debian build tools. The profile must
  apply to dependency resolution as well as `debian.rules`; it removes the PyPy
  build dependency and restores the CPython shebangs and runtime dependency.
- **Arch derivatives:** build with `LOKI_PYTHON=python`. Arch Extra itself defaults
  to PyPy. Only `python` and `pypy3` are accepted values.

If PyPy is explicitly selected but unavailable or incompatible, fix the target's
build configuration to select CPython rather than downloading an interpreter or
shipping a broken PyPy shebang.

## OBS repository configuration

Some resolver errors cannot be repaired by Loki's package recipes:

- **Leap 15.7: `python311-base` requires `libexpat1 >= 2.8.5`, but only 2.6.4
  is available.** The selected interpreter and library providers are incompatible.
  Select a Leap-compatible Python repository/package, or provide its matching
  Expat dependency. Do not lower Loki's Python floor or ignore RPM dependencies.
- **Fedora: choice between `wget1-wget` and `wget2-wget` for
  `obs-service-download_files`.** In the OBS project's **Project Config**, add
  `Prefer: wget1-wget` to resolve the provider ambiguity. This is a solver
  preference, not a Loki runtime dependency or an addition to `_service`.
- **AppImage:** this directory supplies no AppImage build description. Disable
  that repository unless an AppImage recipe is separately added.
- **Mageia:** moving services server-side removes the reported service-package
  blocker, but the RPM macros and package names still need a native Mageia build
  check; the Fedora/openSUSE spec has not been qualified for Mageia.

After uploading recipe changes, rerun the **remote** services before rebuilding.
A published repository is not proof that this package's dependencies resolve.

## RPM lint and checks

The RPM recipe removes shebangs from the two imported library modules during
`%prep`, not from the upstream checkout or the installed console entrypoints.
openSUSE's `%pyproject_install` generates normal and optimized bytecode; `%fdupes`
then deduplicates identical files without removing either cache variant.
`loki.changes` and the spec's `%changelog` marker supply OBS changelog metadata.

`%check` imports the public entrypoint modules and the two library modules from
the built wheel under the selected interpreter. Isolated mode prevents importing
from the checkout; no frontend is started, so this check needs neither provider
credentials nor network access. This is a wheel smoke check, not the full upstream
suite or proof that target-specific rpmlint reports are clear. Rerun the native
OBS build and inspect its new lint report after uploading both spec and changes.

## Checks

In an OBS checkout, run `osc service remoterun`, wait for completion, then run
`osc update -S` and `osc build REPOSITORY ARCH` for each configured target.
The `serveronly` services deliberately do not run with local `osc service run`.
RPM macro expansion, debtransform, dependency
availability, and makepkg must be tested in their target repositories; a local
Python wheel build cannot establish that all distro packages build.
