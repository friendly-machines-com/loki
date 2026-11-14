Format: 3.0 (quilt)
Source: loki
Binary: loki
Architecture: all
Version: 0.1.0-1
Maintainer: Danny Milosavljevic <dannym@friendly-machines.com>
Homepage: https://friendly-machines.com/git/loki.git
DEBTRANSFORM-TAR: loki-@VERSION@.tar.gz
DEBTRANSFORM-FILES-TAR:
DEBTRANSFORM-RELEASE: 1
Build-Depends: debhelper-compat (= 13), dh-python, pybuild-plugin-pyproject, python3-all (>= 3.11.10), python3-flit-core (>= 3.8), pypy3 (>= 7.3.19) <!cpython>
