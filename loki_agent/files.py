"""Guarded file operations and searches, independent of application state.

The runtime supplies its single observation history, invocation cwd, and
search executor. Path spelling and editor publication policy stay here.
"""

import asyncio
import contextlib
import hashlib
import os
import stat
import time
import uuid
from dataclasses import dataclass

from . import executables, private_files, process_outputs, tool_contracts, tool_runtime


READ_CHAR_CAP = 10 * 1024 * 1024
READ_PATHS_LIMIT = 1000
GLOB_MAX_RESULTS = 100
GREP_DEFAULT_HEAD_LIMIT = 250
_MAX_LINK_DEPTH = 8
_UMASK = os.umask(0)          # probe: sets umask to 0 momentarily, returns the previous value
os.umask(_UMASK)             # restore immediately; no concurrent code runs at startup


if os.name == "posix":
    def _apply_destination_permissions(file_path, fd, tmp_path, target_mode):
        """Set the captured mode through the open descriptor."""
        fchmod = getattr(os, 'fchmod', None)
        if fchmod is not None:
            fchmod(fd, target_mode)
        else:
            # This legacy fallback lacks the descriptor branch's protection
            # against replacement of the temporary directory entry.
            os.chmod(tmp_path, target_mode)
else:
    def _apply_destination_permissions(file_path, fd, tmp_path, target_mode):
        """Preserve the destination's DACL on the temporary file.

        Windows has no file modes.  The destination's DACL is captured by name
        (mirroring the POSIX mode capture).  The descriptor from ``mkstemp`` was
        not opened with ``WRITE_DAC``, so the DACL is applied by pathname -- the
        same fallback and the same limitation as the POSIX ``os.chmod`` branch:
        a replaced temporary entry could receive it instead.  A destination that
        does not exist yet keeps the inherited default DACL, which is what a
        plain create would have given it.
        """
        from . import windows_api
        try:
            sddl = windows_api.dacl_sddl(file_path)
        except windows_api.WindowsApiError as error:
            if error.status in [windows_api.ERROR_FILE_NOT_FOUND,
                                windows_api.ERROR_PATH_NOT_FOUND]:
                # A destination that does not exist yet keeps the inherited
                # default DACL; any other read failure must not be papered over.
                return
            raise
        if sddl is not None:
            windows_api.set_named_dacl(tmp_path, sddl)


@dataclass(frozen=True)
class FileObservation:
    digest: str
    size: int


def resolve_path(path: str, base_dir: str | None = None) -> str:
    if not path:
        return path
    # This is an operation pathname, not an identity key. Let the kernel
    # traverse symlinks, '..', non-directories and missing components.
    if os.path.isabs(path):
        return path
    if base_dir is None:
        raise ValueError("relative paths require a working directory")
    return os.path.join(base_dir, path)


def observation_key(file_path: str) -> str:
    """The name a read-before-write record is kept under.

    Loki never normalises or resolves a filesystem operand -- never, for any
    purpose, including this one.  A name is what the caller wrote, anchored by
    the caller to the fixed logical cwd; rewriting it substitutes a different
    name for a possibly different object, which is not what any caller asked
    for.  ``realpath``, ``abspath`` and ``normpath`` all do that rewriting.
    Case and separators are folded so one name has one key, and nothing else is
    done to it.

    Identity by ``(st_dev, st_ino)`` is not used either: on volumes that report
    no file index (fat, exFAT, some shares and filters) two unrelated files
    share ``st_ino`` 0 and would share a key.

    What detects a replaced file is the observation recorded under this key,
    compared in ``_stale_file_error``.
    """
    return os.path.normcase(file_path)


def _follow_final_link(file_path: str) -> str:
    """The name to replace: an existing final symlink's target, else the name.

    Following a final symlink is the one thing a write must do with a name:
    editors' atomic saves replace the link's target and leave the link, and
    replacing the link instead breaks pnpm-style links, shared configuration
    and git's view of the file (a typechange from symlink to regular file).
    Loki still never normalises or resolves a *path*: this reads the link
    itself with ``readlink`` and joins at most ``_MAX_LINK_DEPTH`` of them, so
    no ancestor of the operand is opened and no other component is rewritten.
    """
    target = file_path
    for _ in range(_MAX_LINK_DEPTH):
        if not os.path.islink(target):
            return target
        link = os.readlink(target)
        target = (link if os.path.isabs(link)
                  else os.path.join(os.path.dirname(target), link))
    raise OSError(f"too many levels of symbolic links: {file_path}")


def existing_target(file_path: str, observed_stat) -> str:
    """The name to replace, verified against the object the caller observed."""
    target = _follow_final_link(file_path)
    if not os.path.samestat(observed_stat, os.stat(target)):
        raise OSError("path changed during target selection")
    return target


def _write_destination(file_path: str) -> str:
    """Select once, without mutation, before the Read-before-overwrite checks.

    Existing operands must pass kernel lookup before target selection. Missing
    parents may be created later, but a failed traversal of '.' or '..' is not
    permission to discard components. A dangling link is expanded only so its
    missing target, including missing parent directories, can be selected.
    """
    try:
        observed = os.stat(file_path)
    except FileNotFoundError:
        directory, name = os.path.split(file_path)
        if name in ['', '.', '..']:
            raise
        parent = _write_destination(directory or '.')
        try:
            parent_stat = os.stat(parent)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISDIR(parent_stat.st_mode):
                raise NotADirectoryError(parent)
            parent = existing_target(parent, parent_stat)
        target = os.path.join(parent, name)
        try:
            entry = os.lstat(target)
        except FileNotFoundError:
            return target
        if stat.S_ISLNK(entry.st_mode):
            return _write_destination(os.path.join(parent, os.readlink(target)))
        # An entry appeared during selection. Recheck it rather than assume
        # this is still a new-file write. The caller checks the selected name.
        return _write_destination(target)
    return existing_target(file_path, observed)


def atomic_write_text(file_path: str, content: str):
    file_path = _follow_final_link(file_path)
    # A final symlink is followed so its target is replaced and the link stays,
    # the same rule as the write tools apply; callers that already selected a
    # target hand one that is not itself a link.
    directory, name = os.path.split(file_path)
    directory = directory or '.'
    if not name or name in ['.', '..']:
        raise IsADirectoryError(file_path)
    # Capture the desired final mode BEFORE writing so a write/replace failure
    # can never leave the public path's mode corrupted. For an existing file we
    # preserve its current mode (rwx bits, including execute); for a new file
    # we match what plain open(...,'w') would produce, i.e. 0o666 & ~umask.
    try:
        target_mode = os.stat(file_path).st_mode & 0o7777
    except FileNotFoundError:
        target_mode = 0o666 & ~_UMASK
    os.makedirs(directory, exist_ok=True)
    # The directory is opened once, and both the temporary file and the
    # replacement happen inside that open directory: the open *is* the kernel's
    # resolution of it.  Handing the directory's name to the standard allocator
    # instead lets its internal abspath() collapse '..' or a link lexically, so
    # the temporary file lands in a different directory from the destination.
    directory_fd = private_files.open_directory(directory)
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = private_files.create_exclusive_at(
            directory_fd, temporary, 0o600)
        try:
            private_files.write(descriptor, content.encode('utf-8'))
            _apply_destination_permissions(
                file_path, descriptor,
                os.path.join(directory, temporary), target_mode)
        finally:
            private_files.close(descriptor)
        private_files.replace_at(directory_fd, temporary, name)
    except BaseException:
        with contextlib.suppress(OSError):
            private_files.unlink_at(directory_fd, temporary)
        raise
    finally:
        private_files.close(directory_fd)


def atomic_write_text_at(directory_fd, name: str, content: str):
    """Publish private runtime metadata in an already-retained directory.

    Unlike the editor publisher above, these are owned state files: POSIX mode
    is 0600 and Windows inherits the directory's DACL. No pathname-based mode
    or DACL copying may redirect an operation after an ancestor is renamed.
    """
    if not name or name in ['.', '..'] or any(part in name for part in ['/', '\\', ':']):
        raise ValueError("state filename must be a single path component")
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    created = False
    try:
        descriptor = private_files.create_exclusive_at(directory_fd, temporary, 0o600)
        created = True
        try:
            view = memoryview(content.encode('utf-8'))
            while view:
                count = private_files.write(descriptor, view)
                if count <= 0:
                    raise OSError("state write made no progress")
                view = view[count:]
        finally:
            private_files.close(descriptor)
        private_files.replace_at(directory_fd, temporary, name)
    except BaseException:
        if created:
            with contextlib.suppress(OSError):
                private_files.unlink_at(directory_fd, temporary)
        raise


def _stale_file_error(file_path: str, action: str,
                      file_key=None, *, observations) -> str | None:
    if file_key is None:
        file_key = observation_key(file_path)
    observed = observations.get(file_key)
    if file_key not in observations:
        return (
            f"Error: {file_path} has not been read. Read it before "
            f"{action}.")
    if observed is None:
        # Older callers used None for a binary observation. It conveys no
        # identity and therefore cannot safely authorize an overwrite.
        return (
            f"Error: {file_path} has no verifiable read snapshot. "
            f"Read it again before {action}.")
    try:
        if isinstance(observed, FileObservation):
            current = _file_observation(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                current = f.read()
    except Exception as e:
        return f"Error checking current file contents before {action}: {e}"
    if observed != current:
        return (f"Error: {file_path} changed on disk since you last read it. "
                f"Read it again before {action}.")
    return None


def _stat_identity(stat_result) -> tuple:
    identity = (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )
    if os.name == "posix":
        # POSIX ctime is the kernel-maintained metadata-change time: it moves on
        # changes mtime misses (mode, owner, link count) and cannot be set by
        # the process, so it is tamper evidence.  On Windows the field is
        # deprecated and moving from creation time to metadata-change time (or
        # zero), which makes it unstable rather than evidential, so it is left
        # out there; st_birthtime is creation time and constant, so it is not a
        # substitute.  The stored SHA-256 digest remains the content proof.
        identity = identity + (stat_result.st_ctime_ns,)
    return identity


def _file_observation(
        file_path: str, expected_stat=None) -> FileObservation:
    digest = hashlib.sha256()
    size = 0
    with open(file_path, "rb") as file_obj:
        before = os.fstat(file_obj.fileno())
        if (expected_stat is not None
                and _stat_identity(before) != _stat_identity(expected_stat)):
            raise OSError("file changed while it was being read")
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(file_obj.fileno())
        if _stat_identity(after) != _stat_identity(before):
            raise OSError("file changed while it was being read")
    return FileObservation(digest.hexdigest(), size)


def _content_observation(content: str) -> FileObservation:
    encoded = content.encode("utf-8")
    return FileObservation(hashlib.sha256(encoded).hexdigest(), len(encoded))


def _format_numbered_lines(lines: list[str], first_line_number: int = 1) -> str:
    return "\n".join(f"{i}\t{line}" for i, line in enumerate(lines, start=first_line_number))


def run_read(file_path: str, offset: int | None = None, limit: int | None = None, *, structured=False, cwd, observations):
    if not file_path:
        return "Error: file_path is required"
    file_path = resolve_path(file_path, cwd)
    if os.path.isdir(file_path):
        return f"Error: {file_path} is a directory, not a file"
    try:
        st = os.stat(file_path)
        file_key = observation_key(file_path)
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except IsADirectoryError:
        return f"Error: {file_path} is a directory"
    except Exception as e:
        return f"Error: {e}"

    def remember(observation):
        observations[file_key] = observation

    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        try:
            remember(_file_observation(
                file_path, expected_stat=st))
        except OSError as error:
            return f"Error reading file: {error}"
        return f"File {file_path} is an image ({st.st_size} bytes). Visual content rendering not supported in this environment; reading the file is acknowledged."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if _stat_identity(os.fstat(f.fileno())) != _stat_identity(st):
                raise OSError("file changed while it was being read")
            # Read one character past the cap so the response can say whether
            # the file was actually truncated.
            content = f.read(READ_CHAR_CAP + 1)
            if _stat_identity(os.fstat(f.fileno())) != _stat_identity(st):
                raise OSError("file changed while it was being read")
        truncated_chars = len(content) > READ_CHAR_CAP
        if truncated_chars:
            content = content[:READ_CHAR_CAP]
    except UnicodeDecodeError:
        try:
            remember(_file_observation(
                file_path, expected_stat=st))
        except OSError as error:
            return f"Error reading file: {error}"
        return f"File {file_path} is binary ({st.st_size} bytes); cannot display as text."
    except Exception as e:
        return f"Error reading file: {e}"

    if "\x00" in content:
        try:
            remember(_file_observation(
                file_path, expected_stat=st))
        except OSError as error:
            return f"Error reading file: {error}"
        return (
            f"File {file_path} is binary ({st.st_size} bytes); "
            "cannot display as text.")

    if not content:
        remember(_content_observation(""))
        return f"File {file_path} is empty."

    if truncated_chars:
        content += f"\n\n[... file truncated at {READ_CHAR_CAP} characters; file is {st.st_size} bytes on disk -- pass offset/limit to read further]"

    lines = content.splitlines()
    total_lines = len(lines)
    start = int(offset) if offset is not None else 0
    if start < 0 or start >= total_lines:
        return f"Error: offset {offset} out of range (file has {total_lines} lines)"
    if limit is not None:
        lim = int(limit)
        if lim <= 0:
            return "Error: limit must be positive"
    else:
        lim = tool_contracts.READ_DEFAULT_LINES
    sliced = lines[start:start + lim]
    rendered = _format_numbered_lines(sliced, first_line_number=start + 1)
    if start + lim < total_lines:
        rendered += f"\n... ({total_lines - start - lim} more lines not shown)"
    try:
        remember(_file_observation(
            file_path, expected_stat=st))
    except OSError as error:
        return f"Error reading file snapshot: {error}"
    if structured:
        output = process_outputs.ProcessOutput("", rendered, "", shell=False, read=True)
        return tool_runtime.tool_result(True, rendered, process_output=output.to_dict())
    return rendered


def run_write(file_path: str, content: str, *, cwd, observations) -> str:
    if not file_path:
        return "Error: file_path is required"
    if content is None:
        return "Error: content is required"
    operand = resolve_path(file_path, cwd)
    # Authorization is per spelling: the record is looked up under the name the
    # caller wrote, not under the target an existing final symlink resolves to.
    file_key = observation_key(operand)
    try:
        file_path = _write_destination(operand)
    except Exception as error:
        return f"Error: {error}"
    if os.path.exists(operand) and file_key not in observations:
        return (f"Error: You must Read {file_path} before overwriting it. "
                "Read it first, then retry the Write.")
    if file_key in observations:
        stale_error = _stale_file_error(file_path, "overwriting it", file_key, observations=observations)
        if stale_error:
            return stale_error
    try:
        atomic_write_text(file_path, content)
        observations[file_key] = _content_observation(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False, *, cwd, observations) -> str:
    if not file_path:
        return "Error: file_path is required"
    if old_string == new_string:
        return "Error: new_string must be different from old_string"
    operand = resolve_path(file_path, cwd)
    # Authorization is per spelling: looked up under the name the caller wrote.
    file_key = observation_key(operand)
    try:
        file_path = _write_destination(operand)
    except Exception as error:
        return f"Error: {error}"
    if file_key not in observations:
        return f"Error: You must Read {file_path} before editing it."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error: {e}"

    stale_error = _stale_file_error(file_path, "editing it", file_key, observations=observations)
    if stale_error:
        return stale_error

    occurrences = data.count(old_string)
    if occurrences == 0:
        return f"Error: old_string not found in {file_path}."
    if occurrences > 1 and not replace_all:
        return (f"Error: old_string is not unique ({occurrences} occurrences) in {file_path}. "
                "Provide more context to make it unique, or pass replace_all=true.")

    if replace_all:
        new_data = data.replace(old_string, new_string)
        count = occurrences
    else:
        new_data = data.replace(old_string, new_string, 1)
        count = 1

    try:
        atomic_write_text(file_path, new_data)
        observations[file_key] = _content_observation(new_data)
        return f"Successfully edited {file_path} ({count} replacement{'s' if count != 1 else ''})."
    except Exception as e:
        return f"Error: {e}"


async def run_glob_async(
        pattern: str, path: str | None = None,
        cancel_event: asyncio.Event | None = None, *,
        hidden: bool = False, no_ignore: bool = False, cwd, run_exec) -> str:
    if not pattern:
        return "Error: pattern is required"
    rg = executables.RIPGREP
    if not rg:
        return "Error: ripgrep binary not found. Install rg."
    root = resolve_path(path, cwd) if path else cwd
    args = [rg, '--files', '--color=never']
    if hidden:
        args.append('--hidden')
    if no_ignore:
        args.append('--no-ignore')
    args.extend(['--glob', pattern, root])
    start = time.perf_counter()
    job, status, stdout, stderr = await run_exec(
        args, description=f"Glob {pattern!r}",
        cancel_event=cancel_event)
    if status == "cancelled":
        return "Error: glob search was cancelled by the user"
    duration_ms = int((time.perf_counter() - start) * 1000)
    stderr = stderr.strip()
    if job.exit_code not in [0, 1]:
        return f"Error: ripgrep failed with exit code {job.exit_code}" + (f"\n{stderr}" if stderr else "")
    matches = stdout.splitlines()
    num_files = len(matches)
    truncated = len(matches) > GLOB_MAX_RESULTS

    def mtime_or_zero(file_path: str) -> float:
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return 0

    # Put recently changed files first because coding tasks usually care about
    # active work more than lexicographic directory order.
    matches.sort(key=mtime_or_zero, reverse=True)
    matches = matches[:GLOB_MAX_RESULTS]
    if not matches:
        result = f"No files matched pattern {pattern!r} in {root}"
        hints = []
        if not hidden:
            hints.append("hidden paths are excluded by default (set hidden=true to include)")
        if not no_ignore:
            hints.append("ignore rules are respected (set no_ignore=true to bypass)")
        if hints:
            result += "\nNote: " + "; ".join(hints) + "."
        return result
    return "\n".join([
        f"duration_ms: {duration_ms}",
        f"num_files: {num_files}",
        f"truncated: {str(truncated).lower()}",
        "[filenames]",
        *matches,
    ])


def _parse_nonnegative_int(value, name: str, default: int | None = None) -> tuple[int | None, str | None]:
    if value is None:
        return default, None
    try:
        number = int(value)
    except (TypeError, ValueError) as e:
        return None, f"Error: invalid {name}: {e}"
    if number < 0:
        return None, f"Error: {name} must be non-negative"
    return number, None


def _select_limited(lines: list[str], offset: int, head_limit: int) -> tuple[list[str], bool]:
    if head_limit == 0:
        return lines[offset:], False
    return lines[offset:offset + head_limit], len(lines) > offset + head_limit


async def run_grep_async(pattern: str, path: str | None = None, glob: str | None = None,
                         output_mode: str = "files_with_matches",
                         cancel_event: asyncio.Event | None = None,
                         *, cwd, run_exec, **kwargs) -> str:
    if not pattern:
        return "Error: pattern is required"
    if output_mode not in ['content', 'files_with_matches', 'count']:
        return f"Error: invalid output_mode {output_mode!r}"
    rg = executables.RIPGREP
    if not rg:
        return "Error: ripgrep binary not found. Install rg."
    root = resolve_path(path, cwd) if path else cwd

    head_limit, err = _parse_nonnegative_int(kwargs.get('head_limit'), 'head_limit', GREP_DEFAULT_HEAD_LIMIT)
    if err:
        return err
    offset_n, err = _parse_nonnegative_int(kwargs.get('offset'), 'offset', 0)
    if err:
        return err

    args = [rg, '--color=never']
    if output_mode == 'files_with_matches':
        args.append('--files-with-matches')
    elif output_mode == 'count':
        args.extend(['--count-matches', '--with-filename'])
    else:
        args.append('--with-filename')
        if kwargs.get('-n', True):
            args.append('--line-number')
        else:
            args.append('--no-line-number')
        if kwargs.get('-o'):
            args.append('--only-matching')
        context_value = kwargs.get('-C')
        if context_value is None:
            context_value = kwargs.get('context')
        if context_value is not None:
            context_value, err = _parse_nonnegative_int(context_value, '-C/context')
            if err:
                return err
            args.extend(['-C', str(context_value)])
        else:
            before, err = _parse_nonnegative_int(kwargs.get('-B'), '-B')
            if err:
                return err
            after, err = _parse_nonnegative_int(kwargs.get('-A'), '-A')
            if err:
                return err
            if before is not None:
                args.extend(['-B', str(before)])
            if after is not None:
                args.extend(['-A', str(after)])

    if kwargs.get('-i'):
        args.append('--ignore-case')
    if glob:
        args.extend(['--glob', glob])
    if kwargs.get('type'):
        args.extend(['--type', str(kwargs['type'])])
    if kwargs.get('multiline'):
        args.extend(['--multiline', '--multiline-dotall'])
    args.extend(['--', pattern, root])

    start = time.perf_counter()
    job, status, stdout, stderr = await run_exec(
        args, description=f"Grep {pattern!r}",
        cancel_event=cancel_event)
    if status == "cancelled":
        return "Error: grep search was cancelled by the user"
    duration_ms = int((time.perf_counter() - start) * 1000)
    stderr = stderr.strip()
    if job.exit_code not in [0, 1]:
        return f"Error: ripgrep failed with exit code {job.exit_code}" + (f"\n{stderr}" if stderr else "")

    lines = stdout.splitlines()
    selected, truncated = _select_limited(lines, offset_n, head_limit)
    if not selected:
        return f"No matches for {pattern!r}"
    result = "\n".join([
        f"mode: {output_mode}",
        f"duration_ms: {duration_ms}",
        f"num_entries: {len(lines)}",
        f"applied_offset: {offset_n}",
        f"applied_limit: {head_limit}",
        f"truncated: {str(truncated).lower()}",
        "[results]",
        *selected,
    ])
    return process_outputs.truncate_text(result, tool_contracts.BASH_MAX_OUTPUT_CHARS)
