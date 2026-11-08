"""Guarded file operations and searches, independent of application state.

The runtime supplies the conversation's live read history, invocation cwd,
and search executor. Path spelling and editor publication policy stay here.

Operation failures raise: the existing tool exception boundary can then report
failure without guessing from returned prose. Text-only callers use successful
text directly; tool handlers return the existing explicit result envelope.
"""

import asyncio
import contextlib
import errno
import hashlib
import os
import stat
import sys
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


def _apply_destination_permissions(file_path, fd, target_mode):
    """Preserve destination permissions on the owned publication object."""
    if sys.platform != 'win32':
        os.fchmod(fd, target_mode)
        return
    from . import windows_api
    try:
        sddl = windows_api.dacl_sddl(file_path)
    except windows_api.WindowsResourceError:
        raise  # Cleanup failure cannot authorize inheriting a different DACL.
    except windows_api.WindowsApiError as error:
        if error.status in [windows_api.ERROR_FILE_NOT_FOUND,
                            windows_api.ERROR_PATH_NOT_FOUND]:
            # A missing destination keeps normal inheritance. Permission-read
            # errors are not evidence that the destination has no permissions.
            return
        raise
    if sddl is not None:
        # These are both kernel-rendered DACLs, including inheritance controls.
        # An inherited match already preserves the requested permissions; do not
        # require a denied, redundant ACL mutation on OWNER RIGHTS objects.
        # A mismatch still requires the real update, never a weaker fallback.
        if windows_api.handle_dacl_sddl(fd) != sddl:
            windows_api.set_handle_dacl(fd, sddl)


@dataclass(frozen=True)
class FileObservation:
    digest: str
    size: int


def _remember_observation(observations, key, observation):
    # Bound the conversation's live history. Successful Read/Write/Edit refresh
    # a name; eviction asks for another Read rather than retaining unbounded state.
    observations.pop(key, None)
    observations[key] = observation
    if len(observations) > READ_PATHS_LIMIT:
        observations.pop(next(iter(observations)))


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
        try:
            link = os.readlink(target)
        except FileNotFoundError:
            return target
        except OSError as error:
            if error.errno == errno.EINVAL:
                return target  # The selected entry is not a symbolic link.
            raise
        if os.path.isabs(link):
            target = link
        else:
            # readlink's relative operand starts at this link's directory,
            # not the process cwd. Keep its dotdot components for the kernel.
            directory, _name = os.path.split(target)
            target = os.path.join(directory, link)
    raise OSError(f"too many levels of symbolic links: {file_path}")


def existing_target(file_path: str, observed_stat) -> str:
    """Reject an object replaced while this operation selects its destination.

    This is a best-effort check inside one selection, not a lock or an identity
    requirement between separate Read and Write/Edit calls.
    """
    target = _follow_final_link(file_path)
    if not os.path.samestat(observed_stat, os.stat(target)):
        raise OSError('path changed during target selection')
    return target


def _write_destination(file_path: str) -> str:
    """Follow final links without discarding native dot-component traversal.

    Missing parents may be created later. Existing targets are rechecked so a
    replacement during selection is refused; names remain unlocked afterward.
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
            link = os.readlink(target)
        except FileNotFoundError:
            return target
        except OSError as error:
            if error.errno == errno.EINVAL:
                return target
            raise
        return _write_destination(os.path.join(parent, link))
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
    # The editor already needs this native directory acquisition. Recursive
    # creation first is not a harmless existence check: makedirs inspects
    # ancestors even when this directory exists, and a contained Windows writer
    # can use its workspace without permission to inspect C:\. Only a genuinely
    # missing directory needs creation; access/type failures must propagate.
    try:
        directory_fd = private_files.open_directory(directory, follow_symlinks=True)
    except FileNotFoundError:
        os.makedirs(directory, exist_ok=True)
        directory_fd = private_files.open_directory(directory, follow_symlinks=True)
    # Both the temporary file and replacement use this acquired directory: its
    # open is the kernel's resolution. A pathname-based standard allocator can
    # instead collapse '..' or a link lexically and select a different parent.
    temporary = None
    try:
        # Name generation belongs inside the acquired directory's cleanup scope.
        candidate = f'.{name}.{uuid.uuid4().hex}.tmp'
        descriptor = private_files.create_exclusive_at(
            directory_fd, candidate, 0o600, permission_control=True)
        # A colliding entry belongs to another publication; cleanup acquires
        # responsibility for this name only after exclusive creation succeeds.
        temporary = candidate
        if sys.platform == 'win32':
            try:
                # Settle the ACL before bytes can be exposed under a different
                # inherited grant, while this scope still owns the native HANDLE.
                _apply_destination_permissions(file_path, descriptor, target_mode)
            except BaseException:
                private_files.close(descriptor)
                raise
        # Native creation/rename preserve the kernel-selected parent. Content
        # delivery needs no separate raw-write protocol: binary_file already
        # adapts the acquired object to the standard buffered writer. It consumes
        # ownership even if wrapping fails; only the stream closes it afterward.
        with private_files.binary_file(descriptor, 'wb') as stream:
            stream.write(content.encode('utf-8'))
            stream.flush()
            if sys.platform != 'win32':
                # Buffered bytes must reach the file before chmod, because a
                # subsequent write could clear the captured POSIX set-ID bits.
                _apply_destination_permissions(file_path, stream.fileno(), target_mode)
        private_files.replace_at(directory_fd, temporary, name)
        temporary = None
    finally:
        try:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    private_files.unlink_at(directory_fd, temporary)
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
            private_files.write_all(descriptor, content.encode('utf-8'))
        finally:
            private_files.close(descriptor)
        private_files.replace_at(directory_fd, temporary, name)
    except BaseException:
        if created:
            with contextlib.suppress(OSError):
                private_files.unlink_at(directory_fd, temporary)
        raise


def _stale_file_error(file_path: str, action: str,
                      file_key, *, observations, current=None) -> str | None:
    if file_key not in observations:
        return (
            f"Error: {file_path} has not been read. Read it before "
            f"{action}.")
    observed = observations[file_key]
    try:
        if current is None:
            current = _file_observation(file_path)
    except Exception as e:
        return f"Error: checking current file contents before {action}: {e}"
    if observed != current:
        return (f"Error: {file_path} changed on disk since you last read it. "
                f"Read it again before {action}.")
    return None


@contextlib.contextmanager
def _open_regular_file(file_path, mode='r', *, newline=None):
    # POSIX nonblocking open lets us refuse a FIFO without waiting for a writer.
    # Windows descriptors use binary mode; the text wrapper owns translation.
    flags = os.O_RDONLY
    if sys.platform == 'win32':
        flags |= os.O_BINARY
    else:
        flags |= os.O_NONBLOCK
    descriptor = os.open(file_path, flags)
    try:
        file_mode = os.fstat(descriptor).st_mode
        if stat.S_ISDIR(file_mode):
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), file_path)
        if not stat.S_ISREG(file_mode):
            raise OSError('file is not a regular file')
        if sys.platform != 'win32':
            os.set_blocking(descriptor, True)
        # This scope always owns the descriptor. The standard wrapper borrows
        # it, so wrapping failure and read failure have the same close owner.
        with open(descriptor, mode, encoding=None if 'b' in mode else 'utf-8',
                  newline=newline, closefd=False) as source:
            yield source
    finally:
        os.close(descriptor)


def _stat_identity(facts):
    identity = [facts.st_dev, facts.st_ino, facts.st_size, facts.st_mtime_ns]
    if sys.platform != 'win32':
        # POSIX ctime detects changes that a restored mtime misses. Windows'
        # creation/metadata-time meanings differ across supported interpreters.
        identity.append(facts.st_ctime_ns)
    return identity


def _file_observation(file_path: str, expected_stat=None) -> FileObservation:
    # Hash the complete file, including bytes beyond Read's display cap. The
    # metadata checks reject detected changes during this read; they do not
    # make Read/Edit/Write a transaction or authorize filesystem access.
    digest = hashlib.sha256()
    size = 0
    with _open_regular_file(file_path, 'rb') as source:
        before = os.fstat(source.fileno())
        if expected_stat is not None and _stat_identity(before) != _stat_identity(expected_stat):
            raise OSError('file changed while it was being read')
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        if _stat_identity(os.fstat(source.fileno())) != _stat_identity(before):
            raise OSError('file changed while it was being read')
    return FileObservation(digest.hexdigest(), size)


def _content_observation(content: str) -> FileObservation:
    encoded = content.encode("utf-8")
    return FileObservation(hashlib.sha256(encoded).hexdigest(), len(encoded))


def _format_numbered_lines(lines: list[str], first_line_number: int = 1) -> str:
    return "\n".join(f"{i}\t{line}" for i, line in enumerate(lines, start=first_line_number))


def run_read(file_path: str, offset: int | None = None, limit: int | None = None, *, structured=False, cwd, observations):
    if not file_path:
        raise ValueError('file_path is required')
    file_path = resolve_path(file_path, cwd)
    file_key = file_path

    def remember(observation, text, process_output=None):
        _remember_observation(observations, file_key, observation)
        # Empty/binary/image notices are successful observations too. Their
        # wording must never be asked to supply the tool's success flag.
        if structured:
            return tool_runtime.tool_result(True, text, process_output=process_output)
        return text

    ext = os.path.splitext(file_path)[1].lower()
    image = ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    facts = os.stat(file_path)
    content = None
    if not image:
        try:
            # The standard text reader owns UTF-8 and universal-newline
            # decoding. There is no need for a second decoder coordinator
            # merely to record a model's bounded view of the file.
            with _open_regular_file(file_path) as source:
                if _stat_identity(os.fstat(source.fileno())) != _stat_identity(facts):
                    raise OSError('file changed while it was being read')
                content = source.read(READ_CHAR_CAP + 1)
                if _stat_identity(os.fstat(source.fileno())) != _stat_identity(facts):
                    raise OSError('file changed while it was being read')
        except UnicodeDecodeError:
            pass  # Binary files are still observed before Write may replace them.
    observation = _file_observation(file_path, expected_stat=facts)
    if image:
        return remember(observation, f"File {file_path} is an image ({observation.size} bytes). Visual content rendering not supported in this environment; reading the file is acknowledged.")
    if content is None or "\x00" in content:
        return remember(observation, f"File {file_path} is binary ({observation.size} bytes); cannot display as text.")
    if not content:
        return remember(observation, f"File {file_path} is empty.")
    truncated_chars = len(content) > READ_CHAR_CAP
    if truncated_chars:
        content = content[:READ_CHAR_CAP]
        content += f"\n\n[... file truncated at {READ_CHAR_CAP} characters; file is {observation.size} bytes on disk -- pass offset/limit to read further]"

    lines = content.splitlines()
    total_lines = len(lines)
    start = int(offset) if offset is not None else 0
    if start < 0 or start >= total_lines:
        raise ValueError(f'offset {offset} out of range (file has {total_lines} lines)')
    if limit is not None:
        lim = int(limit)
        if lim <= 0:
            raise ValueError('limit must be positive')
    else:
        lim = tool_contracts.READ_DEFAULT_LINES
    sliced = lines[start:start + lim]
    rendered = _format_numbered_lines(sliced, first_line_number=start + 1)
    if start + lim < total_lines:
        rendered += f"\n... ({total_lines - start - lim} more lines not shown)"
    output = process_outputs.ProcessOutput('', rendered, '', shell=False, read=True)
    return remember(observation, rendered, output.to_dict())


def run_write(file_path: str, content: str, *, cwd, observations) -> str:
    if not file_path:
        raise ValueError('file_path is required')
    if content is None:
        raise ValueError('content is required')
    operand = resolve_path(file_path, cwd)
    # The overwrite guard records the caller's spelling, not a final symlink's
    # target name. Reading that target separately does not satisfy this guard.
    file_key = operand
    file_path = _write_destination(operand)
    if os.path.exists(operand) and file_key not in observations:
        raise ValueError(f'You must Read {file_path} before overwriting it. '
                         'Read it first, then retry the Write.')
    if file_key in observations:
        stale_error = _stale_file_error(file_path, 'overwriting it', file_key, observations=observations)
        if stale_error:
            raise ValueError(stale_error)
    atomic_write_text(file_path, content)
    _remember_observation(observations, file_key, _content_observation(content))
    return f'Successfully wrote to {file_path}'


def run_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False, *, cwd, observations) -> str:
    if not file_path:
        raise ValueError('file_path is required')
    if old_string == new_string:
        raise ValueError('new_string must be different from old_string')
    operand = resolve_path(file_path, cwd)
    # The read-before-edit guard is per spelling, as for Write.
    file_key = operand
    file_path = _write_destination(operand)
    if file_key not in observations:
        raise ValueError(f'You must Read {file_path} before editing it.')
    try:
        with _open_regular_file(file_path, newline='') as source:
            data = source.read()
    except UnicodeDecodeError as error:
        raise ValueError(f'{file_path} is not UTF-8 text.') from error

    stale_error = _stale_file_error(
        file_path, 'editing it', file_key, observations=observations,
        current=_content_observation(data))
    if stale_error:
        raise ValueError(stale_error)

    # Compare the raw newlines first, then retain the editor's existing
    # universal-newline behavior.
    data = data.replace('\r\n', '\n').replace('\r', '\n')
    occurrences = data.count(old_string)
    if occurrences == 0:
        raise ValueError(f'old_string not found in {file_path}.')
    if occurrences > 1 and not replace_all:
        raise ValueError(f'old_string is not unique ({occurrences} occurrences) in {file_path}. '
                         'Provide more context to make it unique, or pass replace_all=true.')

    if replace_all:
        new_data = data.replace(old_string, new_string)
        count = occurrences
    else:
        new_data = data.replace(old_string, new_string, 1)
        count = 1

    atomic_write_text(file_path, new_data)
    _remember_observation(observations, file_key, _content_observation(new_data))
    return f"Successfully edited {file_path} ({count} replacement{'s' if count != 1 else ''})."


async def run_glob_async(
        pattern: str, path: str | None = None,
        cancel_event: asyncio.Event | None = None, *,
        hidden: bool = False, no_ignore: bool = False, cwd, run_exec) -> str:
    if not pattern:
        raise ValueError('pattern is required')
    rg = executables.RIPGREP
    if not rg:
        raise FileNotFoundError('ripgrep binary not found. Install rg.')
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
        raise RuntimeError('glob search was cancelled by the user')
    duration_ms = int((time.perf_counter() - start) * 1000)
    stderr = stderr.strip()
    if job.exit_code not in [0, 1]:
        raise RuntimeError(f'ripgrep failed with exit code {job.exit_code}' + (f'\n{stderr}' if stderr else ''))
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
        raise ValueError('pattern is required')
    if output_mode not in ['content', 'files_with_matches', 'count']:
        raise ValueError(f'invalid output_mode {output_mode!r}')
    rg = executables.RIPGREP
    if not rg:
        raise FileNotFoundError('ripgrep binary not found. Install rg.')
    root = resolve_path(path, cwd) if path else cwd

    head_limit, err = _parse_nonnegative_int(kwargs.get('head_limit'), 'head_limit', GREP_DEFAULT_HEAD_LIMIT)
    if err:
        raise ValueError(err)
    offset_n, err = _parse_nonnegative_int(kwargs.get('offset'), 'offset', 0)
    if err:
        raise ValueError(err)

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
                raise ValueError(err)
            args.extend(['-C', str(context_value)])
        else:
            before, err = _parse_nonnegative_int(kwargs.get('-B'), '-B')
            if err:
                raise ValueError(err)
            after, err = _parse_nonnegative_int(kwargs.get('-A'), '-A')
            if err:
                raise ValueError(err)
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
        raise RuntimeError('grep search was cancelled by the user')
    duration_ms = int((time.perf_counter() - start) * 1000)
    stderr = stderr.strip()
    if job.exit_code not in [0, 1]:
        raise RuntimeError(f'ripgrep failed with exit code {job.exit_code}' + (f'\n{stderr}' if stderr else ''))

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
