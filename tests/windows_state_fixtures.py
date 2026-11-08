"""Explicit object models and real-directory controls for Windows decisions."""

from contextlib import ExitStack, contextmanager
import os
import stat
import sys
from unittest import mock

import loki_agent
from loki_agent import windows_api
from loki_agent import windows_state


_native_identity = windows_state._directory_identity
_native_directory = windows_state._directory_location
_native_directory_handle = windows_state._directory_location_from_handle
_native_object = windows_state._object_at
_native_location = windows_state.object_location
_native_programs = windows_state.program_directories


def directory_identity_patch():
    """Authored model names select opaque objects, not rewritten filenames.

    The default model says only that credentials are a child of configuration;
    unrelated authored objects have no declared containment. This is decision
    data, not native lookup evidence. Tests of actual ancestry use the real
    directory controls below or the native Windows APIs.
    """
    identities = {}

    def identify(path):
        return identities.setdefault(path, f"fixture-object-{len(identities)}")

    def locate(path):
        key = identify(path)
        parents = set()
        if path == windows_state.paths.credential_directory():
            parent = identify(windows_state.paths.loki_config_dir())
            if parent != key:
                parents.add(parent)
        return windows_state.ObjectLocation(key, frozenset(parents))

    return mock.patch.multiple(
        windows_state, _directory_identity=identify, _directory_location=locate,
        _directory_location_from_handle=locate, _object_at=locate, object_location=locate,
        program_directories=lambda *, automatic_read=False: [loki_agent.__path__[0]])


@contextmanager
def directory_handle(path):
    """A real directory object; never substitute a pathname for its handle."""
    if sys.platform == 'win32':
        handle = windows_api.open_directory_handle(path)
        close = windows_api.close_handle
    else:
        handle = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        close = os.close
    try:
        yield handle
    finally:
        close(handle)


def _posix_directory_location(handle):
    """Use real fstat/openat parent relationships, not a lexical path oracle."""
    def identity(fd):
        facts = os.fstat(fd)
        if not stat.S_ISDIR(facts.st_mode):
            raise AssertionError('real-directory fixture received a non-directory; model file ancestry explicitly')
        return f"posix-object-{facts.st_dev:x}:{facts.st_ino:x}"

    key = identity(handle)
    current, current_key = handle, key
    ancestors = set()
    with ExitStack() as owned:
        while True:
            parent = os.open('..', os.O_RDONLY | os.O_NONBLOCK, dir_fd=current)
            owned.callback(os.close, parent)
            parent_key = identity(parent)
            if parent_key == current_key:
                break
            if parent_key == key or parent_key in ancestors:
                raise AssertionError('cyclic directory ancestry in fixture')
            ancestors.add(parent_key)
            current, current_key = parent, parent_key
        return windows_state.ObjectLocation(key, frozenset(ancestors))


@contextmanager
def real_directory_locations():
    """Directory topology through this platform's actual filesystem objects.

    On win32 these are the production native operations. POSIX controls use
    fstat/openat for directories only; they do not pretend to implement Windows
    pathname parsing or ancestry for arbitrary regular files.
    """
    if sys.platform == 'win32':
        replacements = dict(
            _directory_identity=_native_identity, _directory_location=_native_directory,
            _directory_location_from_handle=_native_directory_handle,
            _object_at=_native_object, object_location=_native_location,
            program_directories=_native_programs)
    else:
        def locate(path):
            try:
                with directory_handle(path) as handle:
                    return _posix_directory_location(handle)
            except OSError as error:
                status = 2 if isinstance(error, FileNotFoundError) else 267 if isinstance(error, NotADirectoryError) else 5
                raise windows_api.WindowsApiError(str(error), status=status) from error

        def identify(path):
            return locate(path).key

        replacements = dict(
            _directory_identity=identify, _directory_location=locate,
            _directory_location_from_handle=_posix_directory_location,
            _object_at=locate, object_location=_posix_directory_location,
            program_directories=lambda *, automatic_read=False: [loki_agent.__path__[0]])
    with mock.patch.multiple(windows_state, **replacements):
        yield
