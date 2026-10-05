"""User preferences and their storage boundary.

Frontends receive immutable values through load_settings(), and explicit saves
use update_user_settings(). No frontend needs to know how settings are stored.
Files contain sparse overrides: defaults are applied once, before the ordered
file layers. Only the user file is discovered today.

Explicit INI updates preserve unrelated values, but ConfigParser rewrites
formatting and does not retain comments. Loading never creates or rewrites a
file. Small local file operations are synchronous; lock contention yields to
asyncio, and the public I/O API can later await centrally managed storage.
"""

from __future__ import annotations

import asyncio
import configparser
import contextlib
import io
import logging
import os
import secrets
from dataclasses import dataclass, field, fields, replace

from . import file_locks, paths, private_files


@dataclass(frozen=True)
class TerminalSettings:
    show_bash_stdout: bool = False
    show_read_stdout: bool = False


@dataclass(frozen=True)
class Settings:
    terminal: TerminalSettings = field(default_factory=TerminalSettings)


class SettingsError(ValueError):
    pass


_MAX_BYTES = 1024 * 1024
_DEFAULTS = Settings()


def _user_path():
    return os.path.join(paths.loki_config_dir(), "settings.ini")


def _definitions():
    for section in fields(_DEFAULTS):
        defaults = getattr(_DEFAULTS, section.name)
        for option in fields(defaults):
            yield {
                "section": section.name,
                "key": option.name,
                "default": getattr(defaults, option.name),
            }


def _parser():
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    return parser


def _read_ini(file):
    parser = _parser()
    try:
        with open(file, "rb") as source:
            data = source.read(_MAX_BYTES + 1)
        if len(data) > _MAX_BYTES:
            raise SettingsError(f"Settings file is too large: {file}")
        parser.read_string(data.decode("utf-8"), source=str(file))
    except FileNotFoundError:
        return parser
    except (OSError, UnicodeError, configparser.Error) as error:
        raise SettingsError(f"Could not read settings {file}: {error}") from error
    if parser.defaults():
        raise SettingsError(f"Use named sections, not [DEFAULT], in {file}")
    return parser


def _overrides(parser, file):
    result = {}
    for definition in _definitions():
        section, key = definition["section"], definition["key"]
        if not parser.has_option(section, key):
            continue
        try:
            # Each supported setting has a typed default. Add its codec here
            # when introducing a setting whose type differs from this one.
            value = parser.getboolean(section, key)
        except ValueError as error:
            raise SettingsError(
                f"Invalid setting {section}.{key} in {file}: {error}") from error
        result.setdefault(section, {})[key] = value
    return result


def _merge(current, overrides):
    return replace(current, **{
        section: replace(getattr(current, section), **values)
        for section, values in overrides.items()
    })


def _report_error(error):
    logging.getLogger(__name__).warning("%s", ascii(str(error)))


async def load_settings(*, files=None, on_error=None) -> Settings:
    """Load optional layers in increasing priority, retaining valid lower layers.

    The default is the user file. Explicit files also support isolated tests
    and an ordered global/local layer list without changing frontend callers.
    Invalid or unreadable layers are reported and ignored, never rewritten.
    """
    current = _DEFAULTS
    report = on_error or _report_error
    for file in [_user_path()] if files is None else files:
        try:
            overrides = _overrides(_read_ini(file), file)
        except SettingsError as error:
            report(error)
            continue
        current = _merge(current, overrides)
    return current


async def update_user_settings(changes, *, file=None) -> Settings:
    """Set qualified user overrides; None removes an override, not a value.

    Reread under the write lock so independent writers keep unrelated changes.
    Never save an expanded effective snapshot, which would pin defaults or
    accidentally copy values from a higher-priority local layer into this file.
    An invalid existing file is left untouched.
    """
    definitions = {
        f"{entry['section']}.{entry['key']}": entry
        for entry in _definitions()
    }
    if not isinstance(changes, dict):
        raise SettingsError("Settings changes must be a mapping")
    for name, value in changes.items():
        if name not in definitions:
            raise SettingsError(f"Unknown setting: {name}")
        if value is not None and type(value) is not type(definitions[name]["default"]):
            raise SettingsError(f"{name} must be a boolean")
    selected = _user_path() if file is None else os.fspath(file)
    if not changes:
        return _merge(_DEFAULTS, _overrides(_read_ini(selected), selected))
    # Keep user-managed dotfile symlinks intact. Both aliases lock and publish
    # to the selected target, rather than replacing the link itself.
    target = os.path.realpath(selected)
    directory, name = os.path.split(target)
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        directory_fd = private_files.open_directory(directory)
    except OSError as error:
        raise SettingsError(f"Could not save settings {selected}: {error}") from error
    lock_fd = None
    temporary_name = None
    try:
        lock_fd = private_files.open_lock_file_at(directory_fd, name + ".lock", 0o600)
        facts = private_files.describe(lock_fd)
        if not facts.regular or facts.reparse_point:
            raise SettingsError("Settings lock is not a regular file")
        deadline = asyncio.get_running_loop().time() + 0.25
        while True:
            try:
                file_locks.try_lock_exclusive(lock_fd)
                break
            except BlockingIOError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise SettingsError("Settings file is busy")
                await asyncio.sleep(0.01)
        parser = _read_ini(target)
        _overrides(parser, target)
        for qualified, value in changes.items():
            definition = definitions[qualified]
            section, key = definition["section"], definition["key"]
            if value is None:
                if parser.has_section(section):
                    parser.remove_option(section, key)
            else:
                if not parser.has_section(section):
                    parser.add_section(section)
                parser.set(section, key, "true" if value else "false")
        output = io.StringIO()
        parser.write(output)
        data = output.getvalue().encode("utf-8")
        if len(data) > _MAX_BYTES:
            raise SettingsError("Settings file is too large")
        temporary_name = f".{name}.{os.getpid()}.{secrets.token_hex(12)}"
        fd = private_files.create_exclusive_at(directory_fd, temporary_name, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = private_files.write(fd, view)
                if written <= 0:
                    raise OSError("short settings write")
                view = view[written:]
            private_files.fsync(fd)
        finally:
            private_files.close(fd)
        private_files.replace_at(directory_fd, temporary_name, name)
        temporary_name = None
        private_files.fsync(directory_fd)
        result = _merge(_DEFAULTS, _overrides(parser, target))
    except OSError as error:
        raise SettingsError(f"Could not save settings {selected}: {error}") from error
    finally:
        if temporary_name is not None:
            with contextlib.suppress(OSError):
                private_files.unlink_at(directory_fd, temporary_name)
        if lock_fd is not None:
            private_files.close(lock_fd)
        private_files.close(directory_fd)
    if file is None:
        return await load_settings()
    return result
