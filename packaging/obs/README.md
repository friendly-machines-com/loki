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
   `debian.rules`. Run `osc addremove` and `osc commit`.
3. Run the source service using `osc service remoterun` (or **Run services** in
   the web UI), then check the build logs. OBS needs the `obs_scm`, `tar`,
   `recompress`, and `set_version` source services available.
4. Add the desired distribution repositories in the project's **Repositories**
   tab. Package files do not configure repositories automatically.

`_service` fetches the public upstream `master` branch and produces
`loki-0.1.0.tar.gz`, with a `loki-0.1.0/` top-level directory. Tar creation,
compression, and version substitution run at build time, without fetching
network dependencies in the package build. For reproducible releases, change
`revision` to the desired upstream commit or tag before running the service.
The Arch checksum is deliberately `SKIP` because OBS generates the source
archive; a standalone AUR release should instead use a pinned archive and hash.

When upstream's `loki_agent/__init__.py` version changes, update `versionformat`
in `_service`, `Version` in the spec and dsc, `pkgver` in PKGBUILD, and the Debian
changelog together. The service version is intentionally explicit: a Git
snapshot timestamp must not silently disagree with the wheel's version.
`set_version` is not a replacement for maintaining the Debian changelog.

## Distribution requirements

- **openSUSE Tumbleweed / Slowroll:** the spec selects `%{primary_python}`.
- **Leap 15.x:** the spec selects `python311`. The repository must supply
  Python **3.11.10 or newer** and that interpreter's Flit, pip, and wheel
  packages. A `python311` package name alone does not guarantee the patch level.
- **Fedora:** uses `python3` and `pyproject-rpm-macros`.
- **RHEL-family:** the default Python on RHEL 9 is too old. This spec requires
  repositories whose `python3` is new enough and which provide the listed build
  dependencies; it does not automatically select RHEL's parallel Python stacks.
- **Debian / Ubuntu:** use a repository with a supported Python, such as Debian
  13, Debian Unstable, or Ubuntu 26.04. The flat `debian.*` files and `loki.dsc`
  are OBS debtransform inputs, not an already assembled Debian source package.
  `set_version` substitutes the dsc's `@VERSION@` archive selector before
  debtransform constructs the Debian source package.
- **Arch:** uses the distribution Python and builds a wheel without isolation
  or dependency checking, then installs it with `python-installer`.

Upstream requires `>=3.11.10` and excludes Python `3.12.0` through `3.12.3`.
Unmodified Debian 12 (3.11.2) and Ubuntu 24.04 (3.12.3) therefore are not targets.
Do not relax upstream's requirement merely to make an older repository build.
RPM and Debian runtime dependencies retain these exclusions. Arch rolling
repositories are expected to provide a current Python.

All recipes use **Flit**, not setuptools. Upstream currently declares
`flit_core >=3.8,<4`; distribution builds use the installed backend rather than
creating an isolated environment. Arch explicitly passes
`--skip-dependency-check` so a distribution Flit 4 does not fail that bound
check. This bypass is packaging policy, not proof of backend compatibility:
check real distribution builds when their backend changes. Ordinary upstream
builds still honor the original bound in `pyproject.toml`.

## Checks

In an OBS checkout, run `osc service run` and then `osc build REPOSITORY ARCH`
for each configured target. RPM macro expansion, debtransform, dependency
availability, and makepkg must be tested in their target repositories; a local
Python wheel build cannot establish that all distro packages build.
