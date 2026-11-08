"""Directory-equivalence fixtures, not symlink-type or target-text oracles."""

import os
import subprocess


def directory_alias(target, alias):
    """Create a real directory alias without requiring Windows symlink privilege.

    The caller owns both paths and their cleanup. Tests of relative/dangling
    targets, readlink, or symlink type must continue to create genuine symlinks.
    """
    if os.name == 'nt':
        subprocess.run(
            ['cmd.exe', '/d', '/c', 'mklink', '/J', os.fspath(alias), os.fspath(target)],
            check=True, capture_output=True, text=True)
    else:
        os.symlink(target, alias, target_is_directory=True)
    if not os.path.samefile(target, alias):
        raise AssertionError('directory alias fixture selected a different target')
