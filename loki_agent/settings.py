"""User configuration and its storage boundary.

Frontends receive one decoded Settings value through load_settings(); explicit
preference saves use update_user_settings(). Hooks and terminal preferences
share this reader, not independent file-loading paths. Consumers borrow the
configuration lists read-only. Files contain sparse overrides and only the
user file is discovered today.

Explicit INI updates preserve unrelated values, but ConfigParser rewrites
formatting and does not retain comments. Loading never creates or rewrites a
file. Small local file operations are synchronous; lock contention yields to
asyncio, and the public I/O API can later await centrally managed storage.
"""

from __future__ import annotations

import asyncio
import configparser
import io
import logging
import os
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace

from . import file_locks, paths, private_files


@dataclass(frozen=True)
class TerminalSettings:
    show_bash_stdout: bool = False
    show_read_stdout: bool = False
    show_skill_stdout: bool = False


@dataclass(frozen=True)
class HookSettings:
    """One configured command, borrowed by ExternalHook after startup.

    Settings owns decoding/defaults; tool_runtime owns execution. Keeping this
    definition separate from its runtime callback avoids a second copy of every
    option in ExternalHook. The loader owns the ordered lists while building
    them; consumers borrow them read-only, never as mutable turn state.
    Field names/defaults and decoding metadata also drive option recognition:
    adding/removing an option must not require a second maintained allowlist.
    """

    phase: str = field(metadata={"section": True})
    hook_id: str = field(metadata={"section": True})
    command: list[str] = field(metadata={"indexed": True})
    cwd: str
    tools: list[str] = field(default_factory=lambda: ["*"], metadata={"indexed": True})
    timeout_ms: int | None = field(default=None, metadata={"codec": int, "minimum": 1})
    on_error: str | None = None
    workspace_side_effects: bool = False


@dataclass(frozen=True)
class Settings:
    terminal: TerminalSettings = field(default_factory=TerminalSettings)
    hooks: list[HookSettings] = field(default_factory=list)


class SettingsError(ValueError):
    pass


class HookConfigurationError(SettingsError):
    """A recognized hook cannot be configured as requested; do not run it."""


# Each phase owns its timeout and error-action defaults here, not in another
# decoder or the execution pipeline. The default is None: no hook execution
# deadline. Hooks legitimately perform local or remote work whose completion
# has no guaranteed wall time, so a configured ``timeout_ms`` is the user's
# deliberate abandonment budget, never an automatic default.
_HOOK_PHASES = {
    "pre_tool_call": [None, "deny"],
    "pre_tool_gate": [None, "deny"],
    # Manual compaction policy is a fail-closed allow/deny gate, not a context
    # producer. Only trusted user configuration can install it; runtime input
    # and saved/provider JSON never create hooks or relax its error default.
    "pre_compact": [None, "deny"],
    "post_tool_call": [None, "continue"],
    "turn_end": [None, None],
    "turn_error": [None, None],
}


_MAX_BYTES = 1024 * 1024
_DEFAULTS = Settings()


def user_path():
    return os.path.join(paths.loki_config_dir(), "settings.ini")


def _definitions():
    for section in fields(_DEFAULTS):
        defaults = getattr(_DEFAULTS, section.name)
        if not is_dataclass(defaults):
            continue
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


def read_ini(file, *, directory_fd=None, name=None):
    """Read one bounded INI file: size-capped, UTF-8, no [DEFAULT] section.

    The configuration decoder and locked writer share this access mechanism.
    The writer supplies its retained directory/entry so a concurrent ancestor
    rename cannot redirect the locked read. Runtime consumers receive Settings,
    not another file-loading interface or a retained filesystem capability.
    """
    parser = _parser()
    try:
        if directory_fd is None:
            with open(file, "rb") as source:
                data = source.read(_MAX_BYTES + 1)
        else:
            # A locked update must read through its retained publication
            # directory. Settings need regularity facts, not credential policy.
            descriptor = private_files.open_read_at(directory_fd, name)
            try:
                facts = private_files.information(descriptor)
                if not facts.regular or facts.reparse_point:
                    raise SettingsError(f"Settings file is not a regular file: {file}")
                data = bytearray()
                while len(data) <= _MAX_BYTES:
                    chunk = private_files.read(descriptor, min(65536, _MAX_BYTES + 1 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
            finally:
                private_files.close(descriptor)
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


def _indexed_values(options, name, location):
    values = {}
    for key, value in options.items():
        base, separator, index = key.partition(".")
        if base != name:
            continue
        if (not separator or not index.isascii() or not index.isdecimal()
                or (len(index) > 1 and index.startswith("0"))):
            raise HookConfigurationError(f"{location}: invalid option {key!r}")
        if not value:
            raise HookConfigurationError(f"{location}: {key} must not be empty")
        values[index] = value
    if not values:
        return None
    if set(values) != {str(index) for index in range(len(values))}:
        raise HookConfigurationError(
            f"{location}: {name} indices must start at zero without gaps")
    return [values[str(index)] for index in range(len(values))]


def _hooks(parser, report):
    definitions = {item.name: item for item in fields(HookSettings)
                   if not item.metadata.get("section")}
    hooks = []
    seen_ids = set()
    for section in parser.sections():
        prefix, separator, rest = section.partition(".")
        if prefix != "hook":
            continue
        phase, separator, hook_id = rest.partition(".")
        location = f"[{section}]"
        if not separator or not hook_id:
            raise HookConfigurationError(
                f"{location} must be named hook.<phase>.<id>")
        if phase not in _HOOK_PHASES:
            report(f"{location}: unsupported hook phase {phase!r}; hook not run")
            continue
        if hook_id in seen_ids:
            raise HookConfigurationError(f"duplicate hook id {hook_id!r}")
        seen_ids.add(hook_id)
        options = dict(parser.items(section))
        values = {"phase": phase, "hook_id": hook_id}
        values["timeout_ms"], values["on_error"] = _HOOK_PHASES[phase]
        for key in options:
            base = key.partition(".")[0]
            if base not in definitions:
                report(f"{location}: unsupported option {key!r}; value not applied")
            elif not definitions[base].metadata.get("indexed") and key != base:
                raise HookConfigurationError(f"{location}: invalid option {key!r}")
        for name, definition in definitions.items():
            if definition.metadata.get("indexed"):
                value = _indexed_values(options, name, location)
                present = value is not None
            else:
                present = name in options
                value = options.get(name)
            if not present:
                if (definition.default is MISSING
                        and definition.default_factory is MISSING):
                    raise HookConfigurationError(
                        f"{location}: {name} is required and must not be empty")
                continue
            if ((phase in ["turn_end", "turn_error"] and name in ["tools", "on_error", "workspace_side_effects"])
                    or (phase == "pre_compact" and name == "tools")):
                raise HookConfigurationError(f"{location} does not support {name}")
            try:
                codec = definition.metadata.get("codec")
                if codec is not None:
                    value = codec(value)
                    minimum = definition.metadata.get("minimum")
                    if minimum is not None and value < minimum:
                        raise ValueError(f"must be at least {minimum}")
                elif type(definition.default) is bool:
                    value = parser.getboolean(section, name)
            except ValueError as error:
                raise HookConfigurationError(
                    f"{location}: invalid {name}: {error}") from error
            if name == "cwd" and not value:
                raise HookConfigurationError(f"{location}: cwd must not be empty")
            values[name] = value
        if values["on_error"] not in [None, "deny", "continue"]:
            raise HookConfigurationError(f"{location}: on_error must be 'deny' or 'continue'")
        if phase == "pre_compact" and values["on_error"] != "deny":
            raise HookConfigurationError(f"{location}: pre_compact errors must deny compaction")
        if phase == "post_tool_call" and values["on_error"] == "deny":
            raise HookConfigurationError(
                f"{location}: on_error cannot be 'deny' for post_tool_call")
        hooks.append(HookSettings(**values))
    return hooks


def from_parser(parser, *, file="settings.ini", on_error=None) -> Settings:
    """Decode one configuration snapshot, without opening any file.

    Startup and locked preference updates use the same schema. Unknown sections
    and options are reported, not discarded: older binaries must preserve newer
    configuration when writing preferences. Unsupported hook phases cannot run;
    unsupported options leave the recognized fields in effect, with a warning.
    """
    report = on_error or _report_error
    known = {}
    for definition in _definitions():
        known.setdefault(definition["section"], set()).add(definition["key"])
    for section in parser.sections():
        if section == "hook" or section.startswith("hook."):
            continue
        if section not in known:
            report(f"[{section}]: unsupported section; values not applied")
        else:
            for option in parser.options(section):
                if option not in known[section]:
                    report(f"[{section}]: unsupported option {option!r}; value not applied")
    return replace(_merge(_DEFAULTS, _overrides(parser, file)),
                   hooks=_hooks(parser, report))


async def load_settings(*, files=None, on_error=None, strict=False) -> Settings:
    """Load the complete configuration, reading each selected layer once.

    Later layers override explicit options; hook identity is its section name
    and first appearance fixes order. Validate hook definitions after layering,
    so a later layer can override just cwd or one argument. No raw parser is
    retained by a frontend or by runtime hooks. Ordinary preference API callers
    retain reported invalid-layer fallback; startup uses strict=True because a
    broken file must not silently remove configured policy hooks.
    """
    merged = _parser()
    report = on_error or _report_error
    for file in [user_path()] if files is None else files:
        try:
            parser = read_ini(file)
            _overrides(parser, file)
        except SettingsError as error:
            if strict:
                raise
            report(error)
            continue
        merged.read_dict({section: dict(parser.items(section))
                          for section in parser.sections()})
    return from_parser(merged, on_error=report)


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
    selected = user_path() if file is None else os.fspath(file)
    if not changes:
        return from_parser(read_ini(selected), file=selected)
    # Only updates need the editor's final-symlink selection. Import that
    # writer dependency here: files imports tool_runtime, whose HookSettings
    # annotation must not become a cycle when settings is imported first.
    from . import files

    # User-managed final symlinks must survive atomic saves. Their selected
    # target may have a different parent, so only after that selection do we
    # obtain the parent/entry pair needed by retained locking and publication.
    # Ancestor lookup belongs to the native directory open, not a normalizer.
    try:
        target = files._write_destination(selected)
        directory, name = os.path.split(target)
        if name in ['', '.', '..']:
            raise SettingsError(f'Settings destination must name a file: {selected}')
        directory = directory or '.'
        os.makedirs(directory, mode=0o700, exist_ok=True)
        directory_fd = private_files.open_directory(directory, follow_symlinks=True)
    except OSError as error:
        raise SettingsError(f"Could not save settings {selected}: {error}") from error
    lock_fd = None
    try:
        lock_fd = private_files.open_lock_file_at(directory_fd, name + ".lock", 0o600)
        facts = private_files.information(lock_fd)
        if not facts.regular or facts.reparse_point:
            raise SettingsError("Settings lock is not a regular file")
        # Waiting for the holder deliberately has no duration: another
        # writer's completion has no guaranteed wall time, and an arbitrary
        # grace budget would report "busy" for a lock that is about to free.
        # Cancellation stops the wait.
        while True:
            try:
                file_locks.try_lock_exclusive(lock_fd)
                break
            except BlockingIOError:
                await asyncio.sleep(0.01)
        parser = read_ini(target, directory_fd=directory_fd, name=name)
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
        result = from_parser(parser, file=selected)
        output = io.StringIO()
        parser.write(output)
        data = output.getvalue().encode("utf-8")
        if len(data) > _MAX_BYTES:
            raise SettingsError("Settings file is too large")
        private_files.publish_synced_bytes_at(directory_fd, name, data)
    except OSError as error:
        raise SettingsError(f"Could not save settings {selected}: {error}") from error
    finally:
        # A lock-close failure must not prevent release of the retained directory.
        try:
            if lock_fd is not None:
                private_files.close(lock_fd)
        finally:
            private_files.close(directory_fd)
    return result
