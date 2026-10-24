"""One skill catalog for advertisement and model-tool loading.

Names are catalog identifiers, not path operands. The catalog retains the
filesystem-provided directory and selected filename so consumers need not
reconstruct a path from a namespaced identifier or recover its parent.
"""

import json
import os
import stat


_MARKER = "SKILL.md"
_DESCRIPTION_CHARS = 32 * 1024
_CONTENT_CHARS = 100_000


def catalog(config_directory: str) -> dict:
    root = os.path.join(config_directory, "skills")
    try:
        with os.scandir(root) as directory:
            entries = sorted(directory, key=lambda entry: entry.name)
    except OSError:
        return {}
    found = {}

    def add(name, directory, path):
        # Two filesystem entries must not make one identifier choose a file
        # by traversal order. Ambiguous names are unavailable in both fronts.
        found[name] = None if name in found else (directory, path)

    for entry in entries:
        directory = entry.path
        if not os.path.isdir(directory):
            continue
        path = os.path.join(directory, _MARKER)
        if os.path.isfile(path):
            add(entry.name, directory, path)
            continue
        try:
            with os.scandir(directory) as children:
                nested = sorted(children, key=lambda child: child.name)
        except OSError:
            continue
        for child in nested:
            base = child.path
            path = os.path.join(base, _MARKER)
            if os.path.isfile(path):
                add(f"{entry.name}:{child.name}", base, path)
    return {name: value for name, value in found.items() if value is not None}


def available(config_directory: str) -> list[dict]:
    """Describe the same registered identifiers used by loading in either front."""
    return [
        {'name': name, 'description': description(path) or f'Skill: {name}'}
        for name, (_directory, path) in catalog(config_directory).items()
    ]


def tools_for_model(definitions: list, config_directory: str) -> list:
    """Attach catalogue data to Skill's description without changing permissions."""
    if not any(tool.get('function', {}).get('name') == 'Skill' for tool in definitions):
        return definitions
    data = json.dumps(available(config_directory), ensure_ascii=True)
    described = []
    for tool in definitions:
        function = tool.get('function', {})
        if function.get('name') == 'Skill':
            described.append({**tool, 'function': {
                **function, 'description': function['description']
                + '\n\nAvailable skills (catalogue data):\n' + data}})
        else:
            described.append(tool)
    return described


def load(config_directory: str, name: str, args: str | None = None) -> str:
    """Load catalog-selected instructions; the identifier is never a pathname."""
    if not name:
        raise ValueError('skill is required')
    entry = catalog(config_directory).get(name)
    if entry is None:
        raise ValueError(f'skill {name!r} is unavailable')
    base_directory, path = entry
    content = read_text(path, _CONTENT_CHARS)
    truncated = '\n\n[Skill content truncated]' if len(content) >= _CONTENT_CHARS else ''
    header = f'<skill_content name="{name}">\n# Skill: {name}\n\n'
    if args:
        content = f'Args: {args}\n\n{content}'
    return (header + content + truncated
            + f'\n\nBase directory for this skill: {base_directory}\n'
            + 'Relative paths in this skill are relative to this base directory.\n</skill_content>')


def read_text(path: str, limit: int, *, errors="strict") -> str:
    flags = os.O_RDONLY
    if os.name == "posix":
        flags |= os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("skill file is not a regular file")
        stream = os.fdopen(descriptor, "r", encoding="utf-8", errors=errors)
        descriptor = None
        with stream:
            return stream.read(limit)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def description(path: str) -> str:
    try:
        head = read_text(path, _DESCRIPTION_CHARS, errors="replace")
    except OSError:
        return ""
    lines = head.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == "---":
                break
            key, separator, value = stripped.partition(":")
            if separator and key.strip() == "description":
                return value.strip().strip('"\'')
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped != "---":
            return stripped
    return ""
