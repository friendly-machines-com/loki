#!/usr/bin/env python3

# TODO: chat (and file) history rewinding
# TODO: Provide command to set effort level
# TODO: /goal
# TODO: paste support ? maybe not; automatic; weird 4096 Byte length limit ?  It's especially good so pasting something doesnt send 237 requests in a row
# TODO: mouse support; but what for?
# TODO: input with readline support (just print the text you have so far--up to the cursor)
# TODO: maybe sixel bitmap support; but what for?
# TODO: background tasks and job control, maybe
# TODO: also support anthropic protocol (in addition to the old openai chat protocol we do support)
# TODO: make this an actual shell; pipeable and so on like always; history search etc

import sys
import re
import os
import collections
import json
import time
import urllib.request
import urllib.error
import urllib.parse
import subprocess
import signal
import socket
import uuid
import getopt
import io
import tempfile
import shutil
from html.parser import HTMLParser
from pprint import pprint

url = os.environ.get("OPENAI_API_BASE", "https://opencode.ai/zen/go/v1/chat/completions") # "https://api.openai.com/v1/chat/completions"

computer = socket.gethostname()

ERROR_COLOR = 1
STATUS_COLOR = 4
INPUT_COLOR = 7
TOOL_CALL_COLOR = 5

MAX_LOOP_LIMIT = 30
READ_CHAR_CAP = 10 * 1024 * 1024
READ_PATHS_LIMIT = 1000
READ_DEFAULT_LINES = 2000
READ_MAX_LINES = 2000
BASH_DEFAULT_TIMEOUT_MS = 300000
BASH_MAX_TIMEOUT_MS = 600000
BASH_MAX_OUTPUT_CHARS = 10_000_000
WRITE_MAX_OUTPUT_CHARS = 1_000_000
GLOB_MAX_RESULTS = 100
GREP_DEFAULT_HEAD_LIMIT = 250
SEARCH_TIMEOUT_S = 30
SUBAGENT_TIMEOUT_S = 600
TODO_MAX_TODOS = 100
SKILL_MAX_BYTES = 100_000
LOKI_CONFIG_DIR_NAME = "loki"
XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
LOKI_CONFIG_DIR = os.path.join(os.path.expanduser(XDG_CONFIG_HOME), LOKI_CONFIG_DIR_NAME)

WEBFETCH_TIMEOUT_S = 30
WEBFETCH_MAX_BYTES = 10_485_760  # 10 MiB
WEBFETCH_MAX_OUTPUT = 100_000   # 100 KB inline result
WEBFETCH_MAX_PROMPT_CHARS = 200_000
WEBFETCH_CACHE_TTL = 15 * 60    # 15 minutes
_webfetch_cache = {}  # url -> (fetched_at_epoch, content_text, content_type, final_url, status)
WEBSEARCH_TIMEOUT_S = 20
WEBSEARCH_MAX_RESPONSE_BYTES = 2_000_000
WEBSEARCH_MAX_RESULTS = 8
DUCKDUCKGO_HTML_SEARCH_URL = 'https://html.duckduckgo.com/html/'

netloc = urllib.parse.urlparse(url).netloc

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY:
    api_key = OPENAI_API_KEY
    del os.environ['OPENAI_API_KEY']
else:
    res = subprocess.run(['secret-tool', 'lookup', 'domain', netloc], shell=False, capture_output=True, text=True)
    api_key = res.stdout.strip()

if not api_key:
    raise ValueError('API key missing.  Please run secret-tool store --label="opencode API key" domain {!r}'.format(netloc))

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "User-Agent": "TinyAgent/1.0",
}

class LruCache(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = collections.OrderedDict()

    def __setitem__(self, key, value):
        while len(self.items) > self.max_size:
            del self.items[0]

        self.items[key] = value

    def __hasitem__(self, key):
        return key in self.items

    def __contains__(self, key):
        return key in self.items

    def get(self, key, default=None):
        return self.items.get(key, default)

    def __getitem__(self, key):
        self.items.move_to_end(key)
        return self.items[key]


file_state = LruCache(READ_PATHS_LIMIT) # file_path -> last content the agent observed; keys = files Read this session


def _resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(os.getcwd(), path))


class ToolSchemaError(ValueError):
    pass


class ToolValidationError(ValueError):
    pass


SCHEMA_ANNOTATION_KEYS = {"description", "default", "format"}
SCHEMA_VALIDATION_KEYS = {
    "type", "properties", "required", "additionalProperties", "enum", "items",
    "minLength", "maxLength", "minimum", "maximum", "maxItems",
}
SCHEMA_ALLOWED_KEYS = SCHEMA_ANNOTATION_KEYS | SCHEMA_VALIDATION_KEYS


def _schema_path(path: str, key) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


def _json_type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_json_type(value, expected_type: str) -> bool:
    if expected_type == "null":
        return value is None
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    raise ToolSchemaError(f"unsupported type {expected_type!r}")


def _validate_schema(schema: dict, value, path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise ToolSchemaError(f"{path}: schema must be an object")

    unsupported = sorted(set(schema) - SCHEMA_ALLOWED_KEYS)
    if unsupported:
        raise ToolSchemaError(f"{path}: unsupported schema keys: {', '.join(unsupported)}")

    if "type" in schema:
        expected = schema["type"]
        if isinstance(expected, str):
            expected_types = [expected]
        elif isinstance(expected, list) and all(isinstance(t, str) for t in expected):
            expected_types = expected
        else:
            raise ToolSchemaError(f"{path}: type must be a string or list of strings")
        if not any(_matches_json_type(value, expected_type) for expected_type in expected_types):
            expected_label = " or ".join(expected_types)
            raise ToolValidationError(f"{path} must be {expected_label}, got {_json_type_name(value)}")

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(item) for item in schema["enum"])
        raise ToolValidationError(f"{path} must be one of: {allowed}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ToolSchemaError(f"{path}: properties must be an object")

        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ToolSchemaError(f"{path}: required must be a list of strings")
        for key in required:
            if key not in value:
                raise ToolValidationError(f"{_schema_path(path, key)} is required")

        additional = schema.get("additionalProperties", True)
        if additional is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ToolValidationError(f"{_schema_path(path, extra[0])} is not allowed")
        elif additional is not True:
            raise ToolSchemaError(f"{path}: additionalProperties must be true or false")

        for key, subschema in properties.items():
            if key in value:
                _validate_schema(subschema, value[key], _schema_path(path, key))

    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ToolValidationError(f"{path} must contain at most {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                _validate_schema(schema["items"], item, _schema_path(path, i))

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolValidationError(f"{path} must be at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolValidationError(f"{path} must be at most {schema['maxLength']} characters")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolValidationError(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolValidationError(f"{path} must be <= {schema['maximum']}")


def _close_object_schemas(schema: dict) -> None:
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" or "properties" in schema:
        schema.setdefault("additionalProperties", False)
        for subschema in schema.get("properties", {}).values():
            _close_object_schemas(subschema)
    if "items" in schema:
        _close_object_schemas(schema["items"])


def _check_schema_supported(schema: dict, path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise ToolSchemaError(f"{path}: schema must be an object")
    unsupported = sorted(set(schema) - SCHEMA_ALLOWED_KEYS)
    if unsupported:
        raise ToolSchemaError(f"{path}: unsupported schema keys: {', '.join(unsupported)}")
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict):
            raise ToolSchemaError(f"{path}: properties must be an object")
        for key, subschema in properties.items():
            _check_schema_supported(subschema, _schema_path(path, key))
    if "items" in schema:
        _check_schema_supported(schema["items"], f"{path}[]")


def _build_tool_registry(tools: list, handlers: dict) -> dict:
    registry = {}
    seen = set()
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name")
        parameters = function.get("parameters")
        if not name or parameters is None:
            continue
        if name in seen:
            raise ToolSchemaError(f"duplicate tool definition: {name}")
        seen.add(name)
        if name not in handlers:
            raise ToolSchemaError(f"missing handler for tool: {name}")
        _close_object_schemas(parameters)
        _check_schema_supported(parameters)
        handler = handlers[name]
        registry[name] = {
            "definition": tool,
            "schema": parameters,
            "handler": handler["handler"],
            "explore": handler.get("explore", False),
        }

    extra_handlers = sorted(set(handlers) - seen)
    if extra_handlers:
        raise ToolSchemaError(f"handler without tool definition: {', '.join(extra_handlers)}")
    return registry


def validate_tool_args(fn_name: str, args) -> str | None:
    spec = TOOL_REGISTRY.get(fn_name)
    if spec is None:
        return f"Error: unknown tool: {fn_name}"
    try:
        _validate_schema(spec["schema"], args)
    except ToolValidationError as e:
        return f"Error: invalid arguments for {fn_name}: {e}"
    except ToolSchemaError as e:
        return f"Error: invalid schema for {fn_name}: {e}"
    return None


def _truncate_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n... [output truncated: {len(s)} chars total, {max_chars} shown]"


def _format_numbered_lines(lines: list[str], first_line_number: int = 1) -> str:
    return "\n".join(f"{i}\t{line}" for i, line in enumerate(lines, start=first_line_number))


def _format_bash_result(stdout: str, stderr: str, exit_code: int | None,
                        status: str = "completed", no_output_expected: bool = False) -> str:
    parts = [f"status: {status}"]
    if exit_code is not None:
        parts.append(f"exit_code: {exit_code}")
    if no_output_expected:
        parts.append("no_output_expected: true")
    parts.extend([
        "[stdout]",
        stdout if stdout else "(empty)",
        "[stderr]",
        stderr if stderr else "(empty)",
    ])
    return _truncate_text("\n".join(parts), BASH_MAX_OUTPUT_CHARS)


def _atomic_write_text(file_path: str, content: str) -> None:
    directory = os.path.dirname(file_path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(file_path)}.",
        suffix=".tmp",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _stale_file_error(file_path: str, action: str) -> str | None:
    observed = file_state.get(file_path)
    if observed is None:
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            current = f.read()
    except Exception as e:
        return f"Error checking current file contents before {action}: {e}"
    if observed != current:
        return (f"Error: {file_path} changed on disk since you last read it. "
                f"Read it again before {action}.")
    return None


def run_bash(command: str, timeout: int = None, description: str = "",
              run_in_background: bool = False, dangerously_disable_sandbox: bool = False) -> str:
    if command is None:
        return "Error: command is required"
    if command.strip() == "":
        return _format_bash_result("", "", 0, no_output_expected=True)
    if run_in_background:
        try:
            with open(os.devnull, 'w') as devnull:
                subprocess.Popen(command, shell=True, stdout=devnull, stderr=devnull, stdin=subprocess.DEVNULL,
                                 # Calls setsid() in the child before exec: the child becomes
                                 # leader of a new session and process group, detached from
                                 # loki.py's terminal foreground process group. Combined with
                                 # DEVNULL stdin/stdout/stderr, this is fire-and-forget.
                                 start_new_session=True)
            return f"Started in background: {command}"
        except Exception as e:
            return f"Error starting background task: {e}"

    timeout_ms = int(timeout) if timeout else BASH_DEFAULT_TIMEOUT_MS
    timeout_ms = min(timeout_ms, BASH_MAX_TIMEOUT_MS)
    try:
        completed = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_ms / 1000)
        return _format_bash_result(completed.stdout, completed.stderr, completed.returncode)
    except subprocess.TimeoutExpired:
        return _format_bash_result("", f"command timed out after {timeout_ms}ms", None, status="timed_out")
    except Exception as e:
        return f"Error: {e}"


def run_read(file_path: str, offset: int = None, limit: int = None) -> str:
    if not file_path:
        return "Error: file_path is required"
    file_path = _resolve_path(file_path)
    if os.path.isdir(file_path):
        return f"Error: {file_path} is a directory, not a file"
    try:
        st = os.stat(file_path)
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except IsADirectoryError:
        return f"Error: {file_path} is a directory"
    except Exception as e:
        return f"Error: {e}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        file_state[file_path] = None
        return f"File {file_path} is an image ({st.st_size} bytes). Visual content rendering not supported in this environment; reading the file is acknowledged."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(READ_CHAR_CAP + 1)
        truncated_chars = len(content) > READ_CHAR_CAP
        if truncated_chars:
            content = content[:READ_CHAR_CAP]
    except UnicodeDecodeError:
        file_state[file_path] = None
        return f"File {file_path} is binary ({st.st_size} bytes); cannot display as text."
    except Exception as e:
        return f"Error reading file: {e}"

    if not content:
        file_state[file_path] = ""
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
        lim = READ_DEFAULT_LINES
    sliced = lines[start:start + lim]
    rendered = _format_numbered_lines(sliced, first_line_number=start + 1)
    if start + lim < total_lines:
        rendered += f"\n... ({total_lines - start - lim} more lines not shown)"
    file_state[file_path] = content
    return rendered


def run_write(file_path: str, content: str) -> str:
    if not file_path:
        return "Error: file_path is required"
    if not content:
        return "Error: content is required"
    file_path = _resolve_path(file_path)
    existed = os.path.exists(file_path)
    if existed and file_path not in file_state:
        return (f"Error: You must Read {file_path} before overwriting it. "
                "Read it first, then retry the Write.")
    if existed:
        stale_error = _stale_file_error(file_path, "overwriting it")
        if stale_error:
            return stale_error
    try:
        _atomic_write_text(file_path, content)
        file_state[file_path] = content
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    if not file_path:
        return "Error: file_path is required"
    if old_string == new_string:
        return "Error: new_string must be different from old_string"
    file_path = _resolve_path(file_path)
    if file_path not in file_state:
        return f"Error: You must Read {file_path} before editing it."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error: {e}"

    stale_error = _stale_file_error(file_path, "editing it")
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
        _atomic_write_text(file_path, new_data)
        file_state[file_path] = new_data
        return f"Successfully edited {file_path} ({count} replacement{'s' if count != 1 else ''})."
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str, path: str = None) -> str:
    if not pattern:
        return "Error: pattern is required"
    rg = _find_rg_binary()
    if not rg:
        return "Error: ripgrep binary not found. Install rg."
    root = _resolve_path(path) if path else os.getcwd()
    if not os.path.isdir(root):
        return f"Error: {root} is not a directory"
    args = [rg, '--files', '--color=never', '--glob', pattern, root]
    start = time.perf_counter()
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=SEARCH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"Error: ripgrep timed out after {SEARCH_TIMEOUT_S}s"
    except Exception as e:
        return f"Error running ripgrep: {e}"
    duration_ms = int((time.perf_counter() - start) * 1000)
    stderr = res.stderr.strip()
    if res.returncode not in [0, 1]:
        return f"Error: ripgrep failed with exit code {res.returncode}" + (f"\n{stderr}" if stderr else "")
    matches = res.stdout.splitlines()
    truncated = len(matches) > GLOB_MAX_RESULTS

    def mtime_or_zero(file_path: str) -> float:
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return 0

    matches.sort(key=mtime_or_zero, reverse=True)
    matches = matches[:GLOB_MAX_RESULTS]
    if not matches:
        return f"No files matched pattern {pattern!r} in {root}"
    return "\n".join([
        f"duration_ms: {duration_ms}",
        f"num_files: {len(res.stdout.splitlines())}",
        f"truncated: {str(truncated).lower()}",
        "[filenames]",
        *matches,
    ])


def _find_rg_binary() -> str | None:
    return shutil.which('rg')


def _parse_nonnegative_int(value, name: str, default: int = None) -> tuple[int | None, str | None]:
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


def run_grep(pattern: str, path: str = None, glob: str = None,
             output_mode: str = "files_with_matches", **kwargs) -> str:
    if not pattern:
        return "Error: pattern is required"
    if output_mode not in ['content', 'files_with_matches', 'count']:
        return f"Error: invalid output_mode {output_mode!r}"
    rg = _find_rg_binary()
    if not rg:
        return "Error: ripgrep binary not found. Install rg."
    root = _resolve_path(path) if path else os.getcwd()
    if not os.path.exists(root):
        return f"Error: {root} does not exist"

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
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=SEARCH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"Error: ripgrep timed out after {SEARCH_TIMEOUT_S}s"
    except Exception as e:
        return f"Error running ripgrep: {e}"
    duration_ms = int((time.perf_counter() - start) * 1000)
    stderr = res.stderr.strip()
    if res.returncode not in [0, 1]:
        return f"Error: ripgrep failed with exit code {res.returncode}" + (f"\n{stderr}" if stderr else "")

    lines = res.stdout.splitlines()
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
    return _truncate_text(result, BASH_MAX_OUTPUT_CHARS)


def run_todoread() -> str:
    if not session_todos:
        return "No todos for this session."
    out = ["Todos:"]
    for i, t in enumerate(session_todos, start=1):
        out.append(f"  {i}. [{t['status']}] ({t['priority']}) {t['content']}")
    return "\n".join(out)


def run_todowrite(todos: list) -> str:
    global session_todos
    if not isinstance(todos, list):
        return "Error: todos must be an array"
    if len(todos) > TODO_MAX_TODOS:
        return f"Error: too many todos (max {TODO_MAX_TODOS})"
    in_progress = sum(1 for t in todos if t.get('status') == 'in_progress')
    if in_progress > 1:
        return "Error: At most one todo can be in_progress at a time"
    cleaned = []
    for t in todos:
        content = (t.get('content') or '').strip()
        status = t.get('status')
        priority = t.get('priority')
        if not content:
            return "Error: each todo requires non-empty content"
        if status not in ['pending', 'in_progress', 'completed']:
            return f"Error: invalid status {status!r}"
        if priority not in ['high', 'medium', 'low']:
            return f"Error: invalid priority {priority!r}"
        cleaned.append({'content': content, 'status': status, 'priority': priority})
    session_todos = cleaned
    summary = {'total': len(cleaned), 'pending': sum(1 for t in cleaned if t['status'] == 'pending'),
               'in_progress': in_progress,
               'completed': sum(1 for t in cleaned if t['status'] == 'completed')}
    return f"Updated todos: {summary}"


def _handle_bash(args: dict) -> str:
    return run_bash(args["command"],
                    timeout=args.get("timeout"),
                    description=args.get("description"),
                    run_in_background=args.get("run_in_background", False),
                    dangerously_disable_sandbox=args.get("dangerously_disable_sandbox", False))


def _handle_read(args: dict) -> str:
    return run_read(args["file_path"], offset=args.get("offset"), limit=args.get("limit"))


def _handle_write(args: dict) -> str:
    return run_write(args["file_path"], args["content"])


def _handle_edit(args: dict) -> str:
    return run_edit(args["file_path"], args["old_string"],
                    args["new_string"], replace_all=args.get("replace_all", False))


def _handle_glob(args: dict) -> str:
    return run_glob(args["pattern"], args.get("path"))


def _handle_grep(args: dict) -> str:
    return run_grep(args["pattern"],
                    path=args.get("path"),
                    glob=args.get("glob"),
                    output_mode=args.get("output_mode", "files_with_matches"),
                    **{k: v for k, v in args.items()
                       if k in ["-B", "-A", "-C", "context", "-n", "-i", "-o",
                                "type", "head_limit", "offset", "multiline"]})


def _handle_todoread(args: dict) -> str:
    return run_todoread()


def _handle_todowrite(args: dict) -> str:
    return run_todowrite(args["todos"])


def _handle_agent(args: dict) -> str:
    return run_agent(args["description"],
                     args["prompt"],
                     run_in_background=args.get("run_in_background", False),
                     subagent_type=args.get("subagent_type", "Explore"))


def _handle_skill(args: dict) -> str:
    return run_skill(args["skill"], args.get("args"))


def _handle_webfetch(args: dict) -> str:
    return run_webfetch(args["url"], args["prompt"])


def _handle_websearch(args: dict) -> str:
    return run_websearch(args["query"],
                         allowed_domains=args.get("allowed_domains"),
                         blocked_domains=args.get("blocked_domains"))


def _tool_result(ok: bool, content) -> dict:
    return {"ok": ok, "content": str(content)}


def _looks_like_tool_error(content: str) -> bool:
    return content.startswith("Error: ") or content.startswith("Failed")


def with_exception_to_tool_result(context: str, thunk) -> dict:
    try:
        content = thunk()
    except (KeyboardInterrupt, SystemExit):
        raise
    except FileNotFoundError as e:
        return _tool_result(False, f"Error while {context}: file not found: {e}")
    except PermissionError as e:
        return _tool_result(False, f"Error while {context}: permission denied: {e}")
    except TimeoutError as e:
        return _tool_result(False, f"Error while {context}: timed out: {e}")
    except OSError as e:
        return _tool_result(False, f"Error while {context}: OS error: {e}")
    except ValueError as e:
        return _tool_result(False, f"Error while {context}: invalid value: {e}")
    except Exception as e:
        return _tool_result(False, f"Failed while {context}: {type(e).__name__}: {e}")

    text = str(content)
    return _tool_result(not _looks_like_tool_error(text), text)


def dispatch_tool(fn_name: str, args: dict, allowed=None) -> dict:
    spec = TOOL_REGISTRY.get(fn_name)
    if spec is None:
        return _tool_result(False, f"Unknown function: {fn_name}")
    if allowed is not None and fn_name not in allowed:
        return _tool_result(False, f"Tool {fn_name} not available in this subagent (allowed: {sorted(allowed)})")
    return with_exception_to_tool_result(f"executing {fn_name}", lambda: spec["handler"](args))

def run_tool_loop(messages: list, allowed=None, max_loops=MAX_LOOP_LIMIT,
                  chat_fn=None, on_event=None) -> str:
    """Run the raw model/tool protocol loop. Mutates `messages` in place."""
    if chat_fn is None:
        chat_fn = lambda msgs: chat_completion(msgs, tools=TOOLS)
    if on_event is None:
        on_event = lambda event: None

    loop_count = 0
    while True:
        loop_count += 1
        if loop_count > max_loops:
            messages.append({"role": "system",
                             "content": "Max tool loop limit reached. Stop calling tools and respond."})
            on_event({"type": "max_loops"})
        try:
            resp = chat_fn(messages)
        except OSError as e:
            on_event({"type": "network_error", "error": e})
            return ""
        if not resp:
            return ""

        msg = resp["choices"][0]["message"]
        clean_msg = {"role": msg["role"]}
        if msg.get("content"):
            clean_msg["content"] = msg["content"]

        if msg.get("tool_calls"):
            clean_msg["tool_calls"] = msg["tool_calls"]

        messages.append(clean_msg)
        if clean_msg.get("content"):
            on_event({"type": "assistant_message", "content": clean_msg["content"]})

        if not clean_msg.get("tool_calls"):
            return clean_msg.get("content", "")

        for tc in clean_msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception as e:
                args = {}
                result = _tool_result(False, f"Failed to parse arguments: {e}")
            else:
                validation_error = validate_tool_args(fn_name, args)
                if validation_error:
                    result = _tool_result(False, validation_error)
                    on_event({"type": "tool_rejected", "name": fn_name, "args": args})
                else:
                    on_event({"type": "tool_call", "name": fn_name, "args": args})
                    result = dispatch_tool(fn_name, args, allowed=allowed)
            if not result["ok"]:
                on_event({"type": "tool_error", "result": result["content"]})
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": fn_name,
                "content": result["content"],
            })


def _print_tool_args(args) -> None:
    if not isinstance(args, dict):
        pprint(args)
        return
    for k, v in args.items():
        pprint((k, v))


def _terminal_agent_event(event: dict) -> None:
    kind = event.get("type")
    if kind == "max_loops":
        print("\n[!] [Max Loop Limit Reached - Stopping Autonomous Execution]")
    elif kind == "network_error":
        print(f"\n{computer}: NETWORK ERROR: {event['error']}")
    elif kind == "assistant_message":
        rendered_content = terminal.markdown_to_ansi(event["content"])
        print(f"\n{model}: {rendered_content if rendered_content is not None else event['content']}")
    elif kind == "tool_call":
        terminal.set_foreground_color(TOOL_CALL_COLOR)
        print(f"{computer}: Executing Tool: {event['name']} with args:")
        _print_tool_args(event["args"])
        terminal.reset_colors_and_flags()
    elif kind == "tool_rejected":
        terminal.set_foreground_color(TOOL_CALL_COLOR)
        print(f"{computer}: Rejected Tool: {event['name']} with invalid args:")
        _print_tool_args(event["args"])
        terminal.reset_colors_and_flags()
    elif kind == "tool_error":
        terminal.set_background_color(ERROR_COLOR)
        print(event["result"])
        terminal.reset_colors_and_flags()


def run_terminal_turn(messages: list) -> str:
    return run_tool_loop(
        messages,
        chat_fn=lambda msgs: chat_completion(msgs, tools=TOOLS, show_timing=True, report_errors=True),
        on_event=_terminal_agent_event,
    )


def run_toolless_completion(messages: list) -> str:
    resp = chat_completion(messages, tools=[])
    if not resp:
        return ""
    msg = resp["choices"][0]["message"]
    return (msg.get("content") or "").strip()


def _subprocess_stream_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return str(value)


def _format_subagent_result(agent_type: str, description: str, status: str,
                            exit_code: int | None, stdout, stderr) -> str:
    stdout_text = _subprocess_stream_text(stdout).strip()
    stderr_text = _subprocess_stream_text(stderr).strip()
    parts = [
        f"[{agent_type} subagent: {description or 'subagent task'}]",
        f"status: {status}",
    ]
    if exit_code is not None:
        parts.append(f"exit_code: {exit_code}")
    parts.extend([
        "[stdout]",
        stdout_text if stdout_text else "(empty)",
        "[stderr]",
        stderr_text if stderr_text else "(empty)",
    ])
    return "\n".join(parts)


def run_agent(description: str, prompt: str, run_in_background: bool = False,
              subagent_type: str = "Explore") -> str:
    agent_type = subagent_type or "Explore"
    if not prompt:
        return "Error: prompt is required"
    if agent_type != "Explore":
        return f"Error: unknown subagent_type {agent_type!r} (only 'Explore' is supported)"
    if run_in_background:
        return ("Background subagents are not supported in this build. "
                f"Prompt was: {prompt[:200]}")
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             '--subagent', agent_type, '--prompt', prompt],
            text=True, capture_output=True, timeout=SUBAGENT_TIMEOUT_S)
        result = _format_subagent_result(agent_type, description, "completed",
                                         proc.returncode, proc.stdout, proc.stderr)
        if proc.returncode != 0:
            return f"Error: subagent exited with code {proc.returncode}\n{result}"
        return result
    except subprocess.TimeoutExpired as e:
        result = _format_subagent_result(agent_type, description, "timed_out",
                                         None, e.stdout, e.stderr)
        return f"Error: subagent timed out after {SUBAGENT_TIMEOUT_S}s for {description or 'task'}\n{result}"
    except FileNotFoundError:
        return "Error: could not launch subagent (interpreter or script missing)"


def run_skill(skill: str, args: str = None) -> str:
    if not skill:
        return "Error: skill is required"
    skill_root = os.path.join(LOKI_CONFIG_DIR, "skills")
    skill_path = os.path.join(skill_root, skill, "SKILL.md")
    if not os.path.isfile(skill_path):
        return (f"Error: skill {skill!r} not found. Available skills can be discovered by listing "
                f"{skill_root}.")
    try:
        with open(skill_path, 'r', encoding='utf-8') as f:
            content = f.read(SKILL_MAX_BYTES)
    except Exception as e:
        return f"Error loading skill: {e}"
    truncated = len(content) >= SKILL_MAX_BYTES
    base_dir = os.path.dirname(skill_path)
    header = f"<skill_content name=\"{skill}\">\n# Skill: {skill}\n\n"
    body = content
    if args:
        body = f"Args: {args}\n\n{body}"
    footer = ("\n\n[Skill content truncated]" if truncated else "") + \
             f"\n\nBase directory for this skill: {base_dir}\n" \
             "Relative paths in this skill are relative to this base directory.\n</skill_content>"
    return header + body + footer


HTML_TEXT_BLOCK_TAGS = {
    'address', 'article', 'aside', 'blockquote', 'br', 'dd', 'details', 'div',
    'dl', 'dt', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3',
    'h4', 'h5', 'h6', 'header', 'hr', 'li', 'main', 'nav', 'ol', 'p', 'pre',
    'section', 'table', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'ul',
}
HTML_TEXT_SKIP_TAGS = {'script', 'style', 'template', 'noscript', 'svg'}


class HtmlTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0

    def _newline(self):
        if self.parts and self.parts[-1] != '\n':
            self.parts.append('\n')

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in HTML_TEXT_SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in HTML_TEXT_BLOCK_TAGS:
            self._newline()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in HTML_TEXT_SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in HTML_TEXT_BLOCK_TAGS:
            self._newline()

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = ' '.join(data.split())
        if text:
            self.parts.append(text)

    def text(self) -> str:
        lines = []
        current = []
        for part in self.parts:
            if part == '\n':
                if current:
                    lines.append(' '.join(current))
                    current = []
            else:
                current.append(part)
        if current:
            lines.append(' '.join(current))
        return '\n'.join(lines).strip()


def _html_to_text(html: str) -> str:
    parser = HtmlTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def _decode_http_text(raw: bytes, headers, default_charset: str = 'utf-8') -> str:
    candidates = []
    get_content_charset = getattr(headers, 'get_content_charset', None)
    if callable(get_content_charset):
        charset = get_content_charset()
        if charset:
            candidates.append(charset)
    candidates.append(default_charset)

    seen = set()
    for charset in candidates:
        charset = charset.strip()
        if not charset or charset.lower() in seen:
            continue
        seen.add(charset.lower())
        try:
            return raw.decode(charset)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode(default_charset, errors='replace')


def _read_http_text_response(resp, max_bytes: int) -> tuple[str, bool]:
    raw = resp.read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]
    return _decode_http_text(raw, resp.headers), truncated


def _response_status(resp) -> int:
    status = resp.getcode()
    if status is None:
        raise ValueError("HTTP response did not include a status code")
    return status


def _response_content_type(resp) -> str:
    return resp.headers.get('content-type') or ''


def _content_media_type(content_type: str) -> str:
    return content_type.split(';', 1)[0].strip().lower()


def _is_html_content_type(content_type: str) -> bool:
    return _content_media_type(content_type) in {'text/html', 'application/xhtml+xml'}


def _decode_duckduckgo_result_url(raw_href: str) -> str:
    parsed = urllib.parse.urlparse(raw_href)
    query = urllib.parse.parse_qs(parsed.query)
    return query.get('uddg', [raw_href])[0]


class DuckDuckGoResultParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._href = None
        self._text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'a' or self._href is not None:
            return
        attr_map = {name.lower(): value or '' for name, value in attrs}
        classes = attr_map.get('class', '').split()
        href = attr_map.get('href')
        if href and 'result__a' in classes:
            self._href = href
            self._text_parts = []

    def handle_data(self, data):
        if self._href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() != 'a' or self._href is None:
            return
        title = ' '.join(''.join(self._text_parts).split())
        if title:
            self.results.append({
                'title': title,
                'url': _decode_duckduckgo_result_url(self._href),
            })
        self._href = None
        self._text_parts = []


def _parse_duckduckgo_results(html: str) -> list[dict]:
    parser = DuckDuckGoResultParser()
    parser.feed(html)
    parser.close()
    return parser.results


def _fetch_url(url: str) -> dict:
    """GET a URL with redirect tracking, return dict with content/contentType/status/finalUrl/redirects.
    HTTP is upgraded to HTTPS. Cross-host redirects are surfaced, not followed."""
    if url.startswith('http://'):
        url = 'https://' + url[len('http://'):]
    elif not url.startswith('https://'):
        url = 'https://' + url

    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return {'error': f'invalid URL: {url}'}

    req = urllib.request.Request(url, headers={
        'User-Agent': 'loki-WebFetch/0.1 (coding-agent)',
        'Accept': 'text/markdown;q=1.0, text/html;q=0.9, text/plain;q=0.8, application/json;q=0.7, */*;q=0.1',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    try:
        with urllib.request.urlopen(req, timeout=WEBFETCH_TIMEOUT_S) as resp:
            status = _response_status(resp)
            content_type = _response_content_type(resp)
            body, truncated = _read_http_text_response(resp, WEBFETCH_MAX_BYTES)
            final_url = resp.geturl()
            return {'content': body, 'contentType': content_type, 'status': status,
                    'finalUrl': final_url, 'truncated': truncated, 'error': None}
    except urllib.error.HTTPError as e:
        return {'error': f'HTTP {e.code} {e.reason}', 'status': e.code, 'finalUrl': url}
    except urllib.error.URLError as e:
        return {'error': f'URL error: {e.reason}', 'finalUrl': url}
    except Exception as e:
        return {'error': f'fetch failed: {e}', 'finalUrl': url}


def run_webfetch(url: str, prompt: str) -> str:
    if not url:
        return "Error: url is required"
    if not prompt:
        return "Error: prompt is required"
    import time as _time
    now = _time.time()
    cached = _webfetch_cache.get(url) # WHAT, IDIOT? DIY LruCache ??? what the FUCK is going on here?
    if cached and now - cached[0] < WEBFETCH_CACHE_TTL:
        content_text, content_type, final_url, status = cached[1], cached[2], cached[3], cached[4]
        cache_hit = True
    else:
        response = _fetch_url(url)
        if response.get('error'):
            return f"Error: {response['error']}"
        content_type = response['contentType']
        if _is_html_content_type(content_type):
            content_text = _html_to_text(response['content'])
        else:
            content_text = response['content']
        if response.get('truncated'):
            content_text += "\n[... page truncated at fetch limit]"
        # cap to a sane size before sending to the model
        if len(content_text) > WEBFETCH_MAX_PROMPT_CHARS:
            content_text = content_text[:WEBFETCH_MAX_PROMPT_CHARS] + "\n[... content truncated for prompt processing]"
        final_url = response['finalUrl']
        status = response['status']
        _webfetch_cache[url] = (now, content_text, content_type, final_url, status)
        cache_hit = False

    # If the final URL's host differs from the requested one, surface it.
    req_host = urllib.parse.urlparse(url).netloc
    final_host = urllib.parse.urlparse(final_url).netloc
    cross_host_note = ""
    if req_host and final_host and req_host != final_host:
        cross_host_note = (f"\n\nNote: this was a cross-host redirect ({req_host} -> {final_host}). "
                           f"Final URL: {final_url}")

    msgs = [
        {"role": "system",
         "content": "You are processing content fetched by the WebFetch tool. "
                    "Answer only from the fetched page content. "
                    "If the content does not contain the answer, say so plainly. "
                    "Keep quotes short and do not reproduce large copyrighted passages."},
        {"role": "user",
         "content": f"URL: {final_url}\nContent-Type: {content_type}\nPrompt: {prompt}\n\n--- Page content ---\n{content_text}"},
    ]
    answer = run_toolless_completion(msgs) or "(no answer returned)"
    header = f"[WebFetch status={status} cache_hit={cache_hit} bytes~={len(content_text)} url={final_url}]"
    return f"{header}\n{answer}{cross_host_note}"



def run_websearch(query: str, allowed_domains: list = None, blocked_domains: list = None) -> str:
    if not query or len(query) < 2:
        return "Error: query must be at least 2 characters"
    if allowed_domains and blocked_domains:
        return "Error: allowed_domains and blocked_domains cannot both be specified"
    if allowed_domains and len(allowed_domains) > 20:
        return "Error: allowed_domains max 20 entries"
    if blocked_domains and len(blocked_domains) > 20:
        return "Error: blocked_domains max 20 entries"

    form = urllib.parse.urlencode({'q': query, 'b': '', 'kl': 'us-en'})
    req = urllib.request.Request(
        DUCKDUCKGO_HTML_SEARCH_URL,
        data=form.encode('utf-8'),
        headers={'User-Agent': 'loki-WebSearch/0.1 (coding-agent)',
                 'Content-Type': 'application/x-www-form-urlencoded'},
        method='POST')
    try:
        with urllib.request.urlopen(req, timeout=WEBSEARCH_TIMEOUT_S) as resp:
            html, _truncated = _read_http_text_response(resp, WEBSEARCH_MAX_RESPONSE_BYTES)
    except Exception as e:
        return f"Error: web search request failed: {e}"

    results = []
    for result in _parse_duckduckgo_results(html):
        target = result['url']
        host = urllib.parse.urlparse(target).netloc.lower()
        if allowed_domains and not any(host == d.lower() or host.endswith('.' + d.lower())
                                       for d in allowed_domains):
            continue
        if blocked_domains and (any(host == d.lower() or host.endswith('.' + d.lower())
                                    for d in blocked_domains)):
            continue
        results.append(result)
        if len(results) >= WEBSEARCH_MAX_RESULTS:
            break

    if not results:
        return f"No search results for query {query!r}"

    out_lines = [f"WebSearch results for {query!r} ({len(results)} results):"]
    for i, r in enumerate(results, start=1):
        out_lines.append(f"{i}. {r['title']}\n   {r['url']}")

    return "\n".join(out_lines)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "\n".join([
                "Reads a file from the local filesystem.",
                "",
                "- `file_path` may be absolute or relative to the current working directory.",
                "- Reads up to 2000 lines by default.",
                "- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters",
                "- Results are returned using cat -n format, with line numbers starting at 1",
                "- Reads images (PNG, JPG/JPEG, GIF, WEBP) and presents them visually.",
                "- Reading a directory, a missing file, or an empty file returns an error or system reminder rather than content.",
                "- Do NOT re-read a file you just edited to verify -- Edit/Write would have errored if the change failed, and the harness tracks file state for you.",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The absolute or relative path to the file to read"},
                    "offset": {"type": "integer", "minimum": 0,
                               "description": "The line number to start reading from. Only provide if the file is too large to read at once"},
                    "limit": {"type": "integer", "minimum": 1,
                              "description": "The number of lines to read. Only provide if the file is too large to read at once."}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "\n".join([
                "Writes a file to the local filesystem, overwriting if one exists.",
                "",
                "When to use: creating a new file, or fully replacing one you've already Read. Overwriting an existing file you haven't Read will fail. For partial changes, use Edit instead."
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string",
                                  "description": "The absolute or relative path to the file to write"},
                    "content": {"type": "string", "description": "The content to write to the file"}
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "\n".join([
                "Performs exact string replacement in a file.",
                "",
                "- You must Read the file in this conversation before editing, or the call will fail.",
                "- `old_string` must match the file exactly, including indentation, and be unique -- the edit fails otherwise. Strip the Read line prefix (line number + tab) before matching.",
                "- `replace_all: true` replaces every occurrence instead."
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The absolute or relative path to the file to modify"},
                    "old_string": {"type": "string", "description": "The text to replace"},
                    "new_string": {"type": "string",
                                   "description": "The text to replace it with (must be different from old_string)"},
                    "replace_all": {"type": "boolean", "default": False,
                                    "description": "Replace all occurrences of old_string (default false)"}
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "\n".join([
                "Executes a bash command and returns its output.",
                "",
                "- Working directory persists between calls, but prefer absolute paths -- `cd` in a compound command can trigger a permission prompt. Shell state (env vars, functions) does not persist; the shell is initialized from the user's profile.",
                "- IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool as this will provide a much better experience for the user.",
                f"- `timeout` is in milliseconds: default {BASH_DEFAULT_TIMEOUT_MS}, max {BASH_MAX_TIMEOUT_MS}.",
                "- `run_in_background` runs the command detached: it keeps running across turns and re-invokes you when it exits. No `&` needed.",
                "",
                "# Git",
                "- Interactive flags (`-i`, e.g. `git rebase -i`, `git add -i`) are not supported in this environment.",
                "- Commit or push only when the user asks. If on the default branch, branch first."
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute"},
                    "timeout": {"type": "integer", "minimum": 0, "maximum": BASH_MAX_TIMEOUT_MS,
                                "description": f"Optional timeout in milliseconds (max {BASH_MAX_TIMEOUT_MS})"},
                    "description": {"type": "string", "description": "\n".join([
                        'Clear, concise description of what this command does in active voice. Never use words like "complex" or "risk" in the description - just describe what it does.',
                        "",
                        "For simple commands (git, npm, standard CLI tools), keep it brief (5-10 words):",
                        '- ls -> "List files in current directory"',
                        '- git status -> "Show working tree status"',
                        '- npm install -> "Install package dependencies"',
                        "",
                        "For commands that are harder to parse at a glance (piped commands, obscure flags, etc.), add enough context to clarify what it does:",
                        '- find . -name "*.tmp" -exec rm {} \\; -> "Find and delete all .tmp files recursively"',
                        '- git reset --hard origin/main -> "Discard all local changes and match remote main"',
                        "- curl -s url | jq '.data[]' -> \"Fetch JSON from URL and extract data array elements\"",
                    ])},
                    "run_in_background": {"type": "boolean",
                                          "description": "Set to true to run this command in the background."},
                    "dangerously_disable_sandbox": {"type": "boolean",
                                                    "description": "Set this to true to dangerously override sandbox mode and run commands without sandboxing."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Fast file pattern matching. Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\". Returns matching file paths sorted by modification time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The glob pattern to match files against"},
                    "path": {"type": "string",
                             "description": "The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "\n".join([
                "Content search built on ripgrep. Prefer this over `grep`/`rg` via Bash -- results integrate with the permission UI and file links.",
                "",
                "- Full regex syntax (e.g. \"log.*Error\", \"function\\s+\\w+\"). Ripgrep, not grep -- escape literal braces (`interface\\{\\}`).",
                "- Filter with `glob` (e.g. \"**/*.tsx\") or `type` (e.g. \"js\", \"py\", \"rust\").",
                "- `output_mode`: \"content\" (matching lines), \"files_with_matches\" (paths only, default), or \"count\".",
                "- `multiline: true` for patterns that span lines.",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string",
                                "description": "The regular expression pattern to search for in file contents"},
                    "path": {"type": "string",
                             "description": "File or directory to search in (rg PATH). Defaults to current working directory."},
                    "glob": {"type": "string",
                             "description": "Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\") - maps to rg --glob"},
                    "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"],
                                    "description": "Output mode: \"content\" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), \"files_with_matches\" shows file paths (supports head_limit), \"count\" shows match counts (supports head_limit). Defaults to \"files_with_matches\"."},
                    "-B": {"type": "integer",
                           "description": "Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise."},
                    "-A": {"type": "integer",
                           "description": "Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise."},
                    "-C": {"type": "integer", "description": "Alias for context."},
                    "context": {"type": "integer",
                                "description": "Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise."},
                    "-n": {"type": "boolean",
                           "description": "Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise. Defaults to true."},
                    "-i": {"type": "boolean", "description": "Case insensitive search (rg -i)"},
                    "-o": {"type": "boolean",
                           "description": "Print only the matched (non-empty) parts of each matching line, one match per output line (rg -o / --only-matching). Requires output_mode: \"content\", ignored otherwise. Defaults to false."},
                    "type": {"type": "string",
                             "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types."},
                    "head_limit": {"type": "integer",
                                   "description": "Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). Defaults to 250 when unspecified. Pass 0 for unlimited (use sparingly -- large result sets waste context)."},
                    "offset": {"type": "integer",
                               "description": "Skip first N lines/entries before applying head_limit, equivalent to \"| tail -n +N | head -N\". Works across all output modes. Defaults to 0."},
                    "multiline": {"type": "boolean",
                                  "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "TodoRead",
            "description": "Read the current session todo list",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": "\n".join([
                "Create and update a task list for the current session. The list is rendered to the user as your working plan.",
                "",
                "- Each todo has `content`, `status` (\"pending\" | \"in_progress\" | \"completed\"), and `priority` (\"high\" | \"medium\" | \"low\").",
                "- Send the full list each call; it replaces the previous one.",
                "- Keep one item `in_progress` at a time and mark it `completed` when done.",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The complete updated todo list. At most one item may be in_progress at a time.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Brief description of the task"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"],
                                           "description": "Current status of the task"},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"],
                                             "description": "Priority level of the task"}
                            },
                            "required": ["content", "status", "priority"]
                        }
                    }
                },
                "required": ["todos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Agent",
            "description": "\n".join([
                "Launch a new agent to handle complex, multi-step tasks. Each agent type has specific capabilities and tools available to it.",
                "",
                "Available agent types and the tools they have access to:",
                "- Explore: Read-only search agent for broad fan-out searches - when answering means sweeping many files, directories, or naming conventions and you only need the conclusion, not the file dumps. It reads excerpts rather than whole files, so it locates code; it doesn't review or audit it. Specify search breadth: \"medium\" for moderate exploration, \"very thorough\" for multiple locations and naming conventions. (Tools: Glob, Grep, Read, Bash, WebFetch, WebSearch, TodoWrite)",
                "",
                "When using the Agent tool, specify a subagent_type parameter to select which agent type to use. If omitted, the \"Explore\" agent is used.",
                "",
                "## When to use",
                "",
                "Reach for this when the task matches an available agent type, when you have independent work to run in parallel, or when answering would mean reading across several files - delegate it and you keep the conclusion, not the file dumps. For a single-fact lookup where you already know the file, symbol, or value, search directly. Once you've delegated a search, don't also run it yourself - wait for the result.",
                "",
                "- The agent's final message is returned to you as the tool result; it is not shown to the user - relay what matters.",
                "- A new Agent call starts fresh, so the prompt must be self-contained.",
                "- `run_in_background: true` runs the agent asynchronously; you'll be notified when it completes.",
                "- When you launch multiple agents for independent work, send them in a single message with multiple tool uses so they run concurrently",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "A short (3-5 word) description of the task"},
                    "prompt": {"type": "string", "description": "The task for the agent to perform"},
                    "run_in_background": {"type": "boolean",
                                          "description": "Set to true to run this agent in the background. You will be notified when it completes."},
                    "subagent_type": {"type": "string", "enum": ["Explore"],
                                      "description": "The type of specialized agent to use for this task"}
                },
                "required": ["description", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Skill",
            "description": "\n".join([
                "Execute a skill within the main conversation",
                "",
                "When users ask you to perform tasks, check if any of the available skills match. Skills provide specialized capabilities and domain knowledge.",
                "",
                "When users reference a \"slash command\" or \"/<something>\", they are referring to a skill. Use this tool to invoke it.",
                "",
                "How to invoke:",
                "- Set `skill` to the exact name of an available skill (no leading slash). For plugin-namespaced skills use the fully qualified `plugin:skill` form.",
                "- Set `args` to pass optional arguments.",
                "",
                "Important:",
                "- Available skills are listed in system-reminder messages in the conversation",
                "- Only invoke a skill that appears in that list, or one the user explicitly typed as `/<name>` in their message. Never guess or invent a skill name from training data; otherwise do not call this tool",
                "- When a skill matches the user's request, this is a BLOCKING REQUIREMENT: invoke the relevant Skill tool BEFORE generating any other response about the task",
                "- NEVER mention a skill without actually calling this tool",
                "- Do not invoke a skill that is already running",
                "- Do not use this tool for built-in CLI commands (like /help, /clear, etc.)",
                "- If you see a <command-name> tag in the current conversation turn, the skill has ALREADY been loaded - follow the instructions directly instead of calling this tool again",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string",
                              "description": "The name of a skill from the available-skills list. Do not guess names."},
                    "args": {"type": "string", "description": "Optional arguments for the skill"}
                },
                "required": ["skill"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "WebFetch",
            "description": "\n".join([
                "Fetches a URL, converts the page to markdown, and answers `prompt` against it using a small fast model.",
                "",
                "- Fails on authenticated/private URLs -- use an authenticated MCP tool or `gh` for those instead.",
                "- HTTP is upgraded to HTTPS. Cross-host redirects are returned to you rather than followed; call again with the redirect URL.",
                "- Responses are cached for 15 minutes per URL.",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri",
                            "description": "The URL to fetch content from"},
                    "prompt": {"type": "string", "description": "The prompt to run on the fetched content"}
                },
                "required": ["url", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "WebSearch",
            "description": "\n".join([
                "Search the web. Returns result blocks with titles and URLs. US-only.",
                "",
                f"- Current date context: {time.strftime('%B %Y')}. Use this when searching for recent information.",
                "- `allowed_domains` / `blocked_domains` filter results.",
                '- After answering from results, end with a "Sources:" list of the URLs you used as markdown links.',
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 2, "description": "The search query to use"},
                    "allowed_domains": {"type": "array", "items": {"type": "string"}, "maxItems": 20,
                                        "description": "Only include search results from these domains"},
                    "blocked_domains": {"type": "array", "items": {"type": "string"}, "maxItems": 20,
                                        "description": "Never include search results from these domains"}
                },
                "required": ["query"]
            }
        }
    },
]

TOOL_HANDLERS = {
    "Read": {"handler": _handle_read, "explore": True},
    "Write": {"handler": _handle_write},
    "Edit": {"handler": _handle_edit},
    "Bash": {"handler": _handle_bash, "explore": True},
    "Glob": {"handler": _handle_glob, "explore": True},
    "Grep": {"handler": _handle_grep, "explore": True},
    "TodoRead": {"handler": _handle_todoread, "explore": True},
    "TodoWrite": {"handler": _handle_todowrite, "explore": True},
    "Agent": {"handler": _handle_agent},
    "Skill": {"handler": _handle_skill},
    "WebFetch": {"handler": _handle_webfetch, "explore": True},
    "WebSearch": {"handler": _handle_websearch, "explore": True},
}
TOOL_REGISTRY = _build_tool_registry(TOOLS, TOOL_HANDLERS)
TOOLS = [spec["definition"] for spec in TOOL_REGISTRY.values()]
EXPLORE_TOOLS = {name for name, spec in TOOL_REGISTRY.items() if spec["explore"]}

def _chat_request(url: str, payload, report_errors: bool = False,
                  show_timing: bool = False) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8') if payload is not None else None, headers=headers)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except OSError as e:
        if report_errors:
            print(f"API Error for <{url}>: {e}", file=sys.stderr)
        return None

    elapsed = time.perf_counter() - start
    if show_timing:
        print(f"\n[T]  [LLM Response Time: {elapsed:.3f}s]", file=sys.stderr)
    return data

def chat_completion(messages: list, tools=TOOLS, report_errors: bool = False,
                    show_timing: bool = False) -> dict:
    payload = {
        "model": model,
        "messages": messages,
    }
    if tools is not None:
        payload["tools"] = tools
    return _chat_request(url, payload, report_errors=report_errors, show_timing=show_timing)

models = _chat_request(url.replace('/v1/chat/completions', '/v1/models'), None)
models = [item['id'] for item in models['data'] if item['object'] == 'model']

model = 'glm-5.2' if 'glm-5.2' in models else models[0]

#models = ['hy3-preview', 'glm-5.2', 'glm-5.1', 'kimi-k2.7', 'kimi-k2.6', 'deepseek-v4-pro', 'deepseek-v4-flash', 'mimo-v2.5', 'mimo-v2.5-pro']

def count_leading_ones_math(byte: int) -> int:
    # ~byte inverts the bits.
    # & 0xFF ensures we only care about the 8 least-significant bits.
    # .bit_length() tells us where the highest '1' is.
    return 8 - (~byte & 0xFF).bit_length()


if not os.isatty(sys.stdout.fileno()):
  class Terminal:
      def __getattr__(self, x):
          return lambda *args, **kwargs: None
else:
  class Terminal:
    BOLD = '\033[1m'
    CYAN = '\033[36m'
    RESET = '\033[0m'
    def __init__(self):
        self.bracketed_paste = False

    def clear_screen(self):
        print('\033[2J', end='')

    def clear_to_end_of_screen(self): # always, not relative.
        print('\033[J', end='')

    def goto_position(self, row, column):
        print('\033[{};{}H'.format(row, column), end='')

    def set_clipping_region(self, first_row, last_row): # note: after that, cursor position is (1,1) ABSOLUTE OR RELATIVE DEPENDING ON origin_mode
        assert last_row - first_row >= 2 # otherwise not supported.
        print('\033[{};{}r'.format(first_row, last_row - 1), end='')
		#goto_position(1, 1)

    def disable_clipping_regions(self): # note: after that, cursor position is (1,1) either absolute or relative dependig on origin_mode.
        print('\033[r', end='')
		#goto_position(1, 1)

    def save_cursor_position(self):
        print('\033[s', end='')

    def restore_cursor_position(self):
        print('\033[u', end='')

    def flush(self):
        sys.stdout.flush()

    def enable_origin_mode(self): # relative coordinates
        print('\033[?6h', end='')

    def disable_origin_mode(self):
        print('\033[?6l', end='')

    def set_foreground_color(self, index):
        print('\033[{}m'.format(30 + index), end='')

    def set_background_color(self, index):
        print('\033[{}m'.format(40 + index), end='')

    def reset_colors_and_flags(self):
        print('\033[m', end='')

    def enable_bracketed_paste_mode(self): # \e[200~ ... \e[201~
        print('\033[?2004h', end='')

    def disable_bracketed_paste_mode(self):
        print('\033[?2004l', end='')

	# TODO: set raw; set nonecho maybe

    def read_key(self):
       result = io.BytesIO()
       ansi_escape = None
       utf8_count = None
       while True:
           c = os.read(0, 1)
           if c == b'': # EOF
               return 'EOF'

           if ansi_escape is None:
               if c == b'\n' and not self.bracketed_paste:
                   return '\n'

               if c == b'\033':
                   ansi_escape = io.BytesIO()
               else:
                   if c[0] >= 0xc0: # on purpose this will start another utf-8 run (in case there was intermittent connection loss)
                       utf8_count = count_leading_ones_math(c[0])
                       # note: assert utf8_count <= 4
                       # fallthrough

                   result.write(c)
                   if c[0] < 0x80: # ASCII
                       break
                   elif c[0] >= 0x80 and c[0] < 0xc0: # 0b10... ; not 0b11... # not a transmission error
                       if utf8_count is not None:
                           utf8_count = utf8_count - 1

                       if utf8_count == 0: # finished utf-8
                           break
                   else: # transmission error
                       pass  # uhh now what? I got a random extra character I don't want.
           else:
               ansi_escape.write(c)
               if c[0] in b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~':
                   ansi = ansi_escape.getvalue().decode('utf-8', errors='replace')
                   ansi_escape = None
                   if len(ansi) > 0:
                       match ansi[0]:
                           case '[':
                               match ansi[-1]:
                                 case 'A':
                                    return 'CURSOR_UP'
                                 case 'B':
                                    return 'CURSOR_DOWN'
                                 case 'C':
                                    return 'CURSOR_RIGHT'
                                 case 'D':
                                    return 'CURSOR_LEFT'
                                 case '~':
                                   for v in ansi[1:-1].split():
                                       try:
                                           v = int(v)
                                           match v:
                                               case 3: # Delete
                                                   return 'DELETE'
                                               case 5: # page up
                                                   return 'PAGE_UP'
                                               case 6: # page down
                                                   return 'PAGE_DOWN'
                                               case 200: # beginning of bracketed paste
                                                   self.bracketed_paste = True
                                               case 201: # end of bracketed paste
                                                   self.bracketed_paste = False
                                       except ValueError:
                                           pass
                               pass
                           case _:
                               pass

                   break

       return result.getvalue().decode('utf-8', errors='replace')

    def markdown_to_ansi(self, text: str) -> str:
        # We split the text by inline code blocks.
        # Using a capture group `(`.*?`)` ensures the code segments are kept in the resulting list.
        parts = re.split(r'(`.*?`)', text)
        for i, part in enumerate(parts):
            # Check if the current part is a code block
            if part.startswith('`') and part.endswith('`') and len(part) >= 2:
                inner_text = part[1:-1] # Strip the backticks
                parts[i] = f"{CYAN}{inner_text}{RESET}"
            else: # normal text
                 # .*? is non-greedy so it correctly matches isolated **bold** pairs
                part = re.sub(r'\*\*(.*?)\*\*', f'{BOLD}\\1{RESET}', part) # bold; TODO: also __foo__ also bold!
                part = re.sub(r'(?<!\*)\*(.*?)\*(?!\*)', f'\033[3m\\1{RESET}', part) # italics
                # TODO: maybe underline.
                # TODO: maybe strikethrough text: ~
                # TODO: # headline, ## headline, ### headline; also extra line ===== or ----
                # TODO: handle >blockquote blocks
                # TODO: ![Tux, the Linux mascot](/assets/images/tux.png)
                # TODO: \* Without the backslash, this would be a bullet in an unordered list.
                parts[i] = part

        return "".join(parts)

terminal = Terminal()

try:
    terminal_size = os.get_terminal_size()
    terminal_lines = terminal_size.lines
except OSError:
    terminal_lines = 25

terminal_lines = terminal_lines + 1 # last line in 1-based indices is missing otherwise.
output_area = 1, terminal_lines - 4
input_area = terminal_lines - 4, terminal_lines - 2
status_area = terminal_lines - 2, terminal_lines # too big, but that's the minimum supported height of set_clipping_region

'''
    output area
    question area
        regular question
        diff viewer (maybe huge with actual scrolling need!)
    input area
    tab to switch to next area
    status line
'''

def update_status_bar():
    terminal.set_clipping_region(*status_area)
    terminal.goto_position(1, 1)
    terminal.set_background_color(STATUS_COLOR)
    terminal.clear_to_end_of_screen()
    print('Model: {}; Hint: Use /quit to quit, /model to switch model, !foo to execute shell command foo'.format(model), end='')

def get_input(prompt=None):
    terminal.set_clipping_region(*input_area)
    terminal.goto_position(1, 1)
    terminal.set_background_color(INPUT_COLOR)
    terminal.clear_to_end_of_screen() # also clears status bar
    update_status_bar()
    terminal.set_clipping_region(*input_area)
    terminal.goto_position(1, 1)
    terminal.set_background_color(INPUT_COLOR)
    terminal.flush()
    # TODO: read raw keys
    text = input(prompt or 'User: ')
    #terminal.save_cursor_position()
    terminal.set_clipping_region(*output_area)
    terminal.restore_cursor_position()
    return text

# TODO: use.
def run_menu(items):
    # Menus can be large--so show it in the output area.
    terminal.save_cursor_position()
    while True:
        terminal.restore_cursor_position()
        for i, item in enumerate(items):
            print(i + 1, '.', item)
        #terminal.save_cursor_position()
        selection = input('User choice: ') # TODO: just use get_input; and do terminal.save_cursor_position() before here.
        selection = selection.strip()
        if selection.endswith('.'):
            selection = selection.rstrip('.')

        if selection:
            terminal.save_cursor_position()
            try:
                return items[int(selection) - 1]
            except:
                return selection

    terminal.save_cursor_position()
    return 'None of the above'

def run_input_loop():
  while True:
    input_text = get_input()

    terminal.reset_colors_and_flags()
    terminal.set_clipping_region(*output_area)
    terminal.restore_cursor_position()

    yield input_text
    #print('blahblah')
    # Update cursor position for next output
    terminal.save_cursor_position()

    #terminal.disable_origin_mode()
    #break

messages = []
session_todos = []

def new_chat_log(filename):
    global chat_log
    chat_log = open(filename, 'w')

def save_chat_log():
    chat_log.seek(0)
    json.dump({
        "messages": messages,
        "session_todos": session_todos,
    }, chat_log, indent=4)
    chat_log.truncate()
    chat_log.flush()
    print('Note: Saved chat log to {}'.format(chat_log.name), file=sys.stderr)
    sys.stderr.flush()

def load_chat_log(filename):
    global chat_log
    global messages
    global session_todos
    chat_log = open(filename, 'r')
    try:
        blob = json.load(chat_log)
        messages = blob["messages"]
        session_todos = blob["session_todos"]
        for message in messages:
            for k, v in message.items():
                pprint((k, v)) # TODO: nicer

            print()

        print('----')
    finally:
        chat_log.close()

    chat_log = open(filename, 'w')
    save_chat_log()

def run_subagent_prompt(subagent_type: str, prompt: str) -> str:
    if subagent_type != "Explore":
        return f"Error: unknown subagent_type {subagent_type!r} (only 'Explore' is supported)"
    if not prompt:
        return ""
    msgs = [
        {"role": "system",
         "content": "You are a focused Explore subagent. Use Glob/Grep/Read/Bash to investigate, then write a concise final answer."},
        {"role": "user", "content": prompt},
    ]
    return run_tool_loop(msgs, allowed=EXPLORE_TOOLS)


def run_subagent_cli(subagent_type: str, prompt: str = None):
    prompt = prompt if prompt is not None else sys.stdin.read().strip()
    result = run_subagent_prompt(subagent_type, prompt)
    if result:
        print(result)


def main():
    global model
    global messages

    options, args = getopt.getopt(sys.argv[1:], 'r:p:', ['resume=', 'prompt=', 'subagent=', 'headless', 'toolset='])
    prompt_arg = None
    subagent_type = None
    headless = False
    toolset = None
    for option_name, option_value in options:
        if option_name in ['--prompt', '-p']:
            prompt_arg = option_value
        elif option_name == '--subagent':
            subagent_type = option_value
        elif option_name == '--headless':
            headless = True
        elif option_name == '--toolset':
            toolset = option_value

    if subagent_type or headless:
        run_subagent_cli(subagent_type or toolset or "Explore", prompt_arg)
        return

    try:
        log_filename = None
        for option_name, option_value in options:
            if option_name == '--resume' or option_name == '-r':
                log_filename = option_value

        if args[0:1] == ['resume']:
            log_filename = args[1]

        if log_filename:
            if not os.path.exists(log_filename):
                log_filename = 'chat-{}.json'.format(log_filename)
            load_chat_log(log_filename)
        else:
            log_filename = 'chat-{}.json'.format(str(uuid.uuid4()))
            new_chat_log(log_filename)

    except IndexError: # TODO: remove
        log_filename = 'chat-{}.json'.format(str(uuid.uuid4()))
        new_chat_log(log_filename)

    messages = [{
        "role": "system",
        "content": ("You are a helpful system agent running in a terminal. You have these tools: "
                    "Read, Write, Edit, Bash, Glob, Grep, TodoRead, TodoWrite, Agent, Skill, WebFetch, WebSearch. "
                    "Prefer Glob/Grep/Read over Bash equivalents (find/grep/cat). "
                    "Always Read a file before editing or overwriting it. "
                    "Use TodoWrite to plan multi-step work. Keep responses concise.")
    }]
    for user_in in run_input_loop():
        if not user_in:
            continue

        print('User:', user_in)
        match user_in.strip():
            case '/quit':
                break
            case '/model':
                model = run_menu(models)
                continue
            case _:
                if user_in.strip().startswith('!'): # direct command execution
                    cmd = user_in[1:].strip()
                    print(f"{computer}: [Running local command: {cmd}]")
                    cmd_output = run_bash(cmd)
                    print(cmd_output) # Show output to you in the terminal
                    # Morph the user input so the AI sees exactly what you did and the result
                    user_in = f"I ran the local command `{cmd}`.\nOutput:\n```\n{cmd_output}\n```"
                else:
                    pass

        messages.append({"role": "user", "content": user_in})

        try:
            run_terminal_turn(messages)
        except KeyboardInterrupt:
            terminal.reset_colors_and_flags()
            # ? EMERGENCY BRAKE
            print("\n\n? [EMERGENCY STOP] Agent execution cancelled by user!")
            # If we interrupt while a tool call was requested but unanswered, the API will crash on
            # the next request. We must surgically remove the unanswered tool request from history.
            if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
                messages.pop()

            messages.append({
                "role": "system",
                "content": "CRITICAL: The user forcefully stopped your execution via KeyboardInterrupt (Ctrl+C). You were likely looping, making a mistake, or doing something dangerous. Await new instructions."
            })
            continue # Drop immediately back to the User> prompt

if __name__ == '__main__':
    cleanup_done = False

    def clean_up_step(thunk):
        try:
            thunk()
        except Exception as e:
            print(f"Cleanup error: {type(e).__name__}: {e}", file=sys.stderr)

    def clean_up(*args, **kwargs):
        global cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        if 'chat_log' in globals():
            clean_up_step(save_chat_log)
        clean_up_step(terminal.disable_bracketed_paste_mode)
        clean_up_step(terminal.disable_clipping_regions)
        clean_up_step(terminal.disable_origin_mode)
        clean_up_step(terminal.reset_colors_and_flags)
        clean_up_step(terminal.clear_screen)

    def clean_up_and_exit(*args, **kwargs):
        clean_up(*args, **kwargs)
        sys.exit(1)

    signal.signal(signal.SIGTERM, clean_up_and_exit)

    terminal.enable_bracketed_paste_mode()
    terminal.enable_origin_mode()
    terminal.clear_screen()

    terminal.reset_colors_and_flags()
    terminal.set_clipping_region(*output_area)
    terminal.goto_position(1, 1)
    terminal.save_cursor_position()

    try:
        main()
    finally:
        clean_up()
