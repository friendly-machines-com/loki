#!/usr/bin/env python3

# TODO: chat (and file) history rewinding
# TODO: Provide command to set effort level
# TODO: /goal
# TODO: paste support ? maybe not; automatic; weird 4096 Byte length limit ?  It's especially good so pasting something doesnt send 237 requests in a row
# TODO: mouse support; but what for?
# TODO: input with readline support (just print the text you have so far--up to the cursor)
# TODO: maybe sixel bitmap support; but what for?
# TODO: background tasks and job control, maybe
# TODO: implement OpenAI Responses provider rendering/parsing on top of the neutral transcript
# TODO: make this an actual shell; pipeable and so on like always; history search etc

import sys
import os
import asyncio
import collections
import json
import time
import urllib.parse
import subprocess
import signal
import socket
import uuid
import getopt
import tempfile
import shutil
import threading
import ssl
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pprint import pprint

import terminals
from terminals import get_input_async, restore_output_area_after_input, run_menu_async, terminal
import formats
import protocols

url = os.environ.get("LOKI_API_BASE") or os.environ.get("OPENAI_API_BASE", "https://opencode.ai/zen/go/v1/chat/completions") # "https://api.openai.com/v1/chat/completions"
provider_override = os.environ.get("LOKI_PROVIDER", "auto")
provider_kind = protocols.resolve_protocol(url, provider_override)

computer = socket.gethostname()

ERROR_COLOR = 1
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
XDG_STATE_HOME = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
LOKI_STATE_DIR = os.path.join(os.path.expanduser(XDG_STATE_HOME), LOKI_CONFIG_DIR_NAME)
LOKI_JOB_STATE_DIR = os.path.join(LOKI_STATE_DIR, "jobs")
JOB_TAIL_CHARS = 20_000

WEBFETCH_TIMEOUT_S = 30
WEBFETCH_MAX_BYTES = 10_485_760  # 10 MiB
WEBFETCH_MAX_OUTPUT = 100_000   # 100 KB inline result
WEBFETCH_MAX_PROMPT_CHARS = 200_000
WEBFETCH_CACHE_TTL = 15 * 60    # 15 minutes
WEBFETCH_CACHE_MAX_ENTRIES = 128
WEBSEARCH_TIMEOUT_S = 20
WEBSEARCH_MAX_RESPONSE_BYTES = 2_000_000
WEBSEARCH_MAX_RESULTS = 8
DUCKDUCKGO_HTML_SEARCH_URL = 'https://html.duckduckgo.com/html/'
HTTP_HEADER_MAX_BYTES = 64 * 1024
HTTP_MAX_RESPONSE_BYTES = 50 * 1024 * 1024

netloc = urllib.parse.urlparse(url).netloc

def _pop_env_api_keys(names):
    values = {}
    for name in names:
        value = os.environ.pop(name, "")
        if value:
            values[name] = value
    return values


def _int_env(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


env_api_keys = _pop_env_api_keys(['LOKI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'])
if provider_kind == protocols.ANTHROPIC_MESSAGES:
    api_key = (env_api_keys.get('LOKI_API_KEY') or
               env_api_keys.get('ANTHROPIC_API_KEY') or
               env_api_keys.get('OPENAI_API_KEY') or "")
else:
    api_key = (env_api_keys.get('LOKI_API_KEY') or
               env_api_keys.get('OPENAI_API_KEY') or
               env_api_keys.get('ANTHROPIC_API_KEY') or "")
if not api_key:
    res = subprocess.run(['secret-tool', 'lookup', 'domain', netloc], shell=False, capture_output=True, text=True)
    api_key = res.stdout.strip()

if not api_key:
    raise ValueError('API key missing.  Please run secret-tool store --label="opencode API key" domain {!r}'.format(netloc))

chat_provider = protocols.make_provider(
    url,
    provider=provider_kind,
    api_key=api_key,
    models_url=os.environ.get("LOKI_MODELS_URL"),
    max_tokens=_int_env("LOKI_MAX_TOKENS", 4096),
    anthropic_version=os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
    auth_header=os.environ.get("LOKI_AUTH_HEADER"),
)
headers = chat_provider.headers

class LruCache(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = collections.OrderedDict()

    def __setitem__(self, key, value):
        self.items[key] = value
        self.items.move_to_end(key)
        while len(self.items) > self.max_size:
            self.items.popitem(last=False)

    def __hasitem__(self, key):
        return key in self.items

    def __contains__(self, key):
        return key in self.items

    def get(self, key, default=None):
        if key not in self.items:
            return default
        self.items.move_to_end(key)
        return self.items[key]

    def __getitem__(self, key):
        self.items.move_to_end(key)
        return self.items[key]


file_state = LruCache(READ_PATHS_LIMIT) # file_path -> last content the agent observed; keys = files Read this session
_webfetch_cache = LruCache(WEBFETCH_CACHE_MAX_ENTRIES)  # url -> (fetched_at_epoch, content_text, content_type, final_url, status)


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


def _validate_schema(schema: dict, value, path: str = "$"):
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


def _close_object_schemas(schema: dict):
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" or "properties" in schema:
        schema.setdefault("additionalProperties", False)
        for subschema in schema.get("properties", {}).values():
            _close_object_schemas(subschema)
    if "items" in schema:
        _close_object_schemas(schema["items"])


def _check_schema_supported(schema: dict, path: str = "$"):
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
        sync_handler = handler.get("handler")
        async_handler = handler.get("async_handler")
        if sync_handler is None and async_handler is None:
            raise ToolSchemaError(f"tool {name} has neither handler nor async_handler")
        registry[name] = {
            "definition": tool,
            "schema": parameters,
            "handler": sync_handler,
            "async_handler": async_handler,
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


def _atomic_write_text(file_path: str, content: str):
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
            #os.fsync(f.fileno())
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


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _decode_spooled_output(data: bytes) -> str:
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('utf-8', errors='replace')


def _read_spool_tail(path: str, max_chars: int = JOB_TAIL_CHARS) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            f.seek(max(0, size - max_chars * 4))
            text = _decode_spooled_output(f.read())
    except FileNotFoundError:
        return ""
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


@dataclass
class Job:
    id: str
    command: str
    argv: list[str] | None
    shell: bool
    description: str
    background: bool
    spool_dir: str
    stdout_path: str
    stderr_path: str
    metadata_path: str
    started_at: float
    started_at_iso: str
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    pid: int | None = None
    pgid: int | None = None
    status: str = "starting"
    exit_code: int | None = None
    signal: int | None = None
    finished_at: float | None = None
    finished_at_iso: str | None = None
    timeout_ms: int | None = None


class JobManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.session_id = str(uuid.uuid4())
        self.session_dir = os.path.join(base_dir, self.session_id)
        self.jobs = {}
        self._counter = 0
        self._lock = threading.RLock()

    def _next_job_id(self) -> str:
        with self._lock:
            self._counter += 1
            return str(self._counter)

    def _job_dir(self, job_id: str) -> str:
        return os.path.join(self.session_dir, job_id)

    def _job_metadata(self, job: Job) -> dict:
        return {
            "id": job.id,
            "command": job.command,
            "argv": job.argv,
            "shell": job.shell,
            "description": job.description,
            "background": job.background,
            "pid": job.pid,
            "pgid": job.pgid,
            "status": job.status,
            "exit_code": job.exit_code,
            "signal": job.signal,
            "started_at": job.started_at_iso,
            "finished_at": job.finished_at_iso,
            "timeout_ms": job.timeout_ms,
            "stdout_path": job.stdout_path,
            "stderr_path": job.stderr_path,
        }

    def _write_metadata(self, job: Job):
        _atomic_write_text(job.metadata_path, json.dumps(self._job_metadata(job), indent=2) + "\n")

    def _record_exit(self, job: Job, exit_code: int):
        with self._lock:
            if job.status in ["timed_out", "stopped"]:
                pass
            elif exit_code < 0:
                job.status = "signaled"
                job.signal = -exit_code
            else:
                job.status = "exited"
            job.exit_code = exit_code
            job.finished_at = time.time()
            job.finished_at_iso = _now_iso()
            self._write_metadata(job)

    def _refresh_job(self, job: Job):
        if job.status not in ["running", "stopping"]:
            return
        exit_code = job.process.returncode
        if exit_code is not None:
            self._record_exit(job, exit_code)

    async def _spawn(self, command, display_command: str, description: str,
                     background: bool, timeout_ms: int | None, shell: bool,
                     env: dict | None = None) -> Job:
        os.makedirs(self.session_dir, exist_ok=True)
        job_id = self._next_job_id()
        spool_dir = self._job_dir(job_id)
        os.makedirs(spool_dir, exist_ok=True)
        stdout_path = os.path.join(spool_dir, "stdout")
        stderr_path = os.path.join(spool_dir, "stderr")
        metadata_path = os.path.join(spool_dir, "job.json")
        job = Job(
            id=job_id,
            command=display_command,
            argv=None if shell else list(command),
            shell=shell,
            description=description,
            background=background,
            spool_dir=spool_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata_path=metadata_path,
            started_at=time.time(),
            started_at_iso=_now_iso(),
            timeout_ms=timeout_ms,
        )
        with open(stdout_path, 'wb') as stdout_file, open(stderr_path, 'wb') as stderr_file:
            if shell:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                    env=env,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                    env=env,
                )
        job.process = proc
        job.pid = proc.pid
        try:
            job.pgid = os.getpgid(proc.pid)
        except OSError:
            job.pgid = proc.pid
        job.status = "running"
        with self._lock:
            self.jobs[job.id] = job
            self._write_metadata(job)
        return job

    async def _wait_for_job(self, job: Job) -> int:
        return await job.process.wait()

    async def _monitor_background_job(self, job: Job):
        try:
            exit_code = await self._wait_for_job(job)
            self._record_exit(job, exit_code)
        except Exception as e:
            with self._lock:
                job.status = "monitor_error"
                job.finished_at = time.time()
                job.finished_at_iso = _now_iso()
                with open(job.stderr_path, 'ab') as stderr_file:
                    stderr_file.write(f"\n[job monitor error: {type(e).__name__}: {e}]\n".encode('utf-8'))
                self._write_metadata(job)

    async def run_foreground(self, command, display_command: str, timeout_ms: int,
                             description: str = "", shell: bool = False,
                             output_chars: int = BASH_MAX_OUTPUT_CHARS,
                             env: dict | None = None):
        job = await self._spawn(command, display_command, description, False, timeout_ms, shell, env=env)
        try:
            exit_code = await asyncio.wait_for(self._wait_for_job(job), timeout=timeout_ms / 1000)
        except asyncio.TimeoutError:
            with self._lock:
                job.status = "timed_out"
                self._write_metadata(job)
            try:
                os.killpg(job.pgid or job.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                exit_code = await asyncio.wait_for(self._wait_for_job(job), timeout=2)
            except asyncio.TimeoutError:
                try:
                    os.killpg(job.pgid or job.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                exit_code = await self._wait_for_job(job)
            self._record_exit(job, exit_code)
            return job, "timed_out", _read_spool_tail(job.stdout_path, output_chars), _read_spool_tail(job.stderr_path, output_chars)

        self._record_exit(job, exit_code)
        return job, "completed", _read_spool_tail(job.stdout_path, output_chars), _read_spool_tail(job.stderr_path, output_chars)

    async def run_shell(self, command: str, timeout: int = None, description: str = "",
                        run_in_background: bool = False) -> str:
        if command is None:
            return "Error: command is required"
        if command.strip() == "":
            return _format_bash_result("", "", 0, no_output_expected=True)

        timeout_ms = int(timeout) if timeout else BASH_DEFAULT_TIMEOUT_MS
        timeout_ms = min(timeout_ms, BASH_MAX_TIMEOUT_MS)

        if run_in_background:
            job = await self._spawn(command, command, description, True, None, True)
            try:
                asyncio.get_running_loop().create_task(self._monitor_background_job(job))
            except RuntimeError:
                pass
            return "\n".join([
                f"Started background job {job.id}",
                f"pid: {job.pid}",
                f"pgid: {job.pgid}",
                f"status: {job.status}",
                f"stdout: {job.stdout_path}",
                f"stderr: {job.stderr_path}",
            ])

        job, status, stdout, stderr = await self.run_foreground(
            command, command, timeout_ms, description=description, shell=True)
        if status == "timed_out":
            if stderr:
                stderr += "\n"
            stderr += f"command timed out after {timeout_ms}ms"
            return _format_bash_result(stdout, stderr, job.exit_code, status="timed_out")

        return _format_bash_result(stdout, stderr, job.exit_code)

    async def run_exec(self, argv: list[str], timeout_ms: int, description: str = "",
                       output_chars: int = BASH_MAX_OUTPUT_CHARS,
                       env: dict | None = None):
        if not argv:
            raise ValueError("argv must not be empty")
        return await self.run_foreground(argv, " ".join(argv), timeout_ms,
                                         description=description, shell=False,
                                         output_chars=output_chars, env=env)

    async def run_background_exec(self, argv: list[str], description: str = "",
                                  env: dict | None = None) -> Job:
        if not argv:
            raise ValueError("argv must not be empty")
        job = await self._spawn(argv, " ".join(argv), description, True, None, False, env=env)
        asyncio.get_running_loop().create_task(self._monitor_background_job(job))
        return job

    def _get_job(self, job_id: str) -> Job | None:
        with self._lock:
            job = self.jobs.get(str(job_id))
        if job is not None:
            self._refresh_job(job)
        return job

    def list_jobs(self) -> str:
        with self._lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            self._refresh_job(job)
        if not jobs:
            return "No jobs."
        lines = ["Jobs:"]
        for job in jobs:
            lines.append(
                f"{job.id}. status={job.status} pid={job.pid} exit={job.exit_code} "
                f"started={job.started_at_iso} command={job.command!r}"
            )
        return "\n".join(lines)

    def job_status(self, job_id: str, tail_chars: int = JOB_TAIL_CHARS) -> str:
        job = self._get_job(job_id)
        if job is None:
            return f"Error: unknown job id {job_id!r}"
        stdout = _read_spool_tail(job.stdout_path, tail_chars)
        stderr = _read_spool_tail(job.stderr_path, tail_chars)
        return "\n".join([
            f"job_id: {job.id}",
            f"status: {job.status}",
            f"pid: {job.pid}",
            f"pgid: {job.pgid}",
            f"exit_code: {job.exit_code}",
            f"signal: {job.signal}",
            f"started_at: {job.started_at_iso}",
            f"finished_at: {job.finished_at_iso}",
            f"stdout_path: {job.stdout_path}",
            f"stderr_path: {job.stderr_path}",
            "[stdout_tail]",
            stdout if stdout else "(empty)",
            "[stderr_tail]",
            stderr if stderr else "(empty)",
        ])

    def stop_job(self, job_id: str, force: bool = False) -> str:
        job = self._get_job(job_id)
        if job is None:
            return f"Error: unknown job id {job_id!r}"
        if job.status != "running":
            return f"Job {job.id} is not running (status={job.status})."
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(job.pgid or job.pid, sig)
        except ProcessLookupError:
            return f"Job {job.id} is no longer running."
        with self._lock:
            job.status = "stopping"
            self._write_metadata(job)
        return f"Sent {sig.name} to job {job.id} (pgid={job.pgid})."


job_manager = JobManager(LOKI_JOB_STATE_DIR)


def run_bash(command: str, timeout: int = None, description: str = "",
              run_in_background: bool = False, dangerously_disable_sandbox: bool = False) -> str:
    return asyncio.run(run_bash_async(command, timeout, description,
                                      run_in_background, dangerously_disable_sandbox))


async def run_bash_async(command: str, timeout: int = None, description: str = "",
                         run_in_background: bool = False,
                         dangerously_disable_sandbox: bool = False) -> str:
    return await job_manager.run_shell(command, timeout=timeout, description=description,
                                       run_in_background=run_in_background)


def run_jobs() -> str:
    return job_manager.list_jobs()


def run_job_status(job_id: str, tail_chars: int = JOB_TAIL_CHARS) -> str:
    if not job_id:
        return "Error: job_id is required"
    try:
        tail_chars = int(tail_chars)
    except (TypeError, ValueError) as e:
        return f"Error: invalid tail_chars: {e}"
    if tail_chars < 0:
        return "Error: tail_chars must be non-negative"
    return job_manager.job_status(job_id, tail_chars=tail_chars)


def run_job_stop(job_id: str, force: bool = False) -> str:
    if not job_id:
        return "Error: job_id is required"
    return job_manager.stop_job(job_id, force=bool(force))


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
    return asyncio.run(run_glob_async(pattern, path))


async def run_glob_async(pattern: str, path: str = None) -> str:
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
    job, status, stdout, stderr = await job_manager.run_exec(
        args, SEARCH_TIMEOUT_S * 1000, description=f"Glob {pattern!r}")
    if status == "timed_out":
        return f"Error: ripgrep timed out after {SEARCH_TIMEOUT_S}s"
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

    matches.sort(key=mtime_or_zero, reverse=True)
    matches = matches[:GLOB_MAX_RESULTS]
    if not matches:
        return f"No files matched pattern {pattern!r} in {root}"
    return "\n".join([
        f"duration_ms: {duration_ms}",
        f"num_files: {num_files}",
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
    return asyncio.run(run_grep_async(pattern, path, glob, output_mode, **kwargs))


async def run_grep_async(pattern: str, path: str = None, glob: str = None,
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
    job, status, stdout, stderr = await job_manager.run_exec(
        args, SEARCH_TIMEOUT_S * 1000, description=f"Grep {pattern!r}")
    if status == "timed_out":
        return f"Error: ripgrep timed out after {SEARCH_TIMEOUT_S}s"
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


async def _handle_bash_async(args: dict) -> str:
    return await run_bash_async(args["command"],
                                timeout=args.get("timeout"),
                                description=args.get("description", ""),
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


async def _handle_glob_async(args: dict) -> str:
    return await run_glob_async(args["pattern"], args.get("path"))


def _handle_grep(args: dict) -> str:
    return run_grep(args["pattern"],
                    path=args.get("path"),
                    glob=args.get("glob"),
                    output_mode=args.get("output_mode", "files_with_matches"),
                    **{k: v for k, v in args.items()
                       if k in ["-B", "-A", "-C", "context", "-n", "-i", "-o",
                                "type", "head_limit", "offset", "multiline"]})


async def _handle_grep_async(args: dict) -> str:
    extra = {k: v for k, v in args.items()
             if k not in ["pattern", "path", "glob", "output_mode"]}
    return await run_grep_async(args["pattern"],
                                path=args.get("path"),
                                glob=args.get("glob"),
                                output_mode=args.get("output_mode", "files_with_matches"),
                                **extra)


def _handle_jobs(args: dict) -> str:
    return run_jobs()


def _handle_job_status(args: dict) -> str:
    return run_job_status(args["job_id"], args.get("tail_chars", JOB_TAIL_CHARS))


def _handle_job_stop(args: dict) -> str:
    return run_job_stop(args["job_id"], args.get("force", False))


def _handle_todoread(args: dict) -> str:
    return run_todoread()


def _handle_todowrite(args: dict) -> str:
    return run_todowrite(args["todos"])


def _handle_agent(args: dict) -> str:
    return run_agent(args["description"],
                     args["prompt"],
                     run_in_background=args.get("run_in_background", False),
                     subagent_type=args.get("subagent_type", "Explore"))


async def _handle_agent_async(args: dict) -> str:
    return await run_agent_async(args.get("description", ""),
                                 args["prompt"],
                                 args.get("run_in_background", False),
                                 args.get("subagent_type", "Explore"))


def _handle_skill(args: dict) -> str:
    return run_skill(args["skill"], args.get("args"))


async def _handle_webfetch_async(args: dict) -> str:
    return await run_webfetch_async(args["url"], args["prompt"])


async def _handle_websearch_async(args: dict) -> str:
    return await run_websearch_async(args["query"],
                                     allowed_domains=args.get("allowed_domains"),
                                     blocked_domains=args.get("blocked_domains"))


def _tool_result(ok: bool, content) -> dict:
    return {"ok": ok, "content": str(content)}


def _looks_like_tool_error(content: str) -> bool:
    return content.startswith("Error: ") or content.startswith("Failed")


async def with_exception_to_tool_result_async(context: str, thunk) -> dict:
    try:
        content = await thunk()
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


async def dispatch_tool_async(fn_name: str, args: dict, allowed=None) -> dict:
    spec = TOOL_REGISTRY.get(fn_name)
    if spec is None:
        return _tool_result(False, f"Unknown function: {fn_name}")
    if allowed is not None and fn_name not in allowed:
        return _tool_result(False, f"Tool {fn_name} not available in this subagent (allowed: {sorted(allowed)})")

    async def run_handler():
        if spec.get("async_handler") is not None:
            return await spec["async_handler"](args)
        return spec["handler"](args)

    return await with_exception_to_tool_result_async(f"executing {fn_name}", run_handler)


async def run_tool_loop_async(transcript_items: list, allowed=None, max_loops=MAX_LOOP_LIMIT,
                              chat_fn=None, on_event=None) -> str:
    """Run the model/tool loop over the neutral transcript. Mutates `transcript_items` in place."""
    if chat_fn is None:
        chat_fn = lambda items: async_chat_completion(items, tools=TOOLS)
    if on_event is None:
        on_event = lambda event: None

    loop_count = 0
    while True:
        loop_count += 1
        if loop_count > max_loops:
            transcript_items.append(formats.instruction_item(
                "Max tool loop limit reached. Stop calling tools and respond."))
            on_event({"type": "max_loops"})
        try:
            response_items = await chat_fn(transcript_items)
        except OSError as e:
            on_event({"type": "network_error", "error": e})
            return ""
        if not response_items:
            return ""

        for item in response_items:
            transcript_items.append(item)

        assistant_items = [
            item for item in response_items
            if item.get("type") == "message" and item.get("role") == "assistant"
        ]
        if not assistant_items:
            return ""

        assistant_item = assistant_items[-1]
        assistant_text = formats.item_text(assistant_item)
        tool_calls = formats.item_tool_calls(assistant_item)
        if assistant_text:
            on_event({"type": "assistant_message", "content": assistant_text})

        if not tool_calls:
            return assistant_text

        for tc in tool_calls:
            fn_name = tc.get("name")
            args = tc.get("input", {})
            if tc.get("parse_error"):
                result = _tool_result(False, f"Failed to parse arguments: {tc['parse_error']}")
            elif not isinstance(args, dict):
                result = _tool_result(False, f"Tool arguments must be an object, got {type(args).__name__}")
            else:
                validation_error = validate_tool_args(fn_name, args)
                if validation_error:
                    result = _tool_result(False, validation_error)
                    on_event({"type": "tool_rejected", "name": fn_name, "args": args})
                else:
                    on_event({"type": "tool_call", "name": fn_name, "args": args})
                    result = await dispatch_tool_async(fn_name, args, allowed=allowed)
            if not result["ok"]:
                on_event({"type": "tool_error", "result": result["content"]})
            transcript_items.append(formats.tool_result_item(
                tc.get("id"),
                result["content"],
                name=fn_name,
                is_error=not result["ok"],
            ))


def _print_tool_args(args):
    if not isinstance(args, dict):
        pprint(args)
        return
    for k, v in args.items():
        pprint((k, v))


def _terminal_agent_event(event: dict):
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


async def run_terminal_turn_async(transcript_items: list) -> str:
    async def chat_fn(items):
        return await async_chat_completion(items, TOOLS, True, True)

    return await run_tool_loop_async(
        transcript_items,
        chat_fn=chat_fn,
        on_event=_terminal_agent_event,
    )


async def run_toolless_completion_async(transcript_items: list) -> str:
    response_items = await async_chat_completion(transcript_items, tools=[])
    if not response_items:
        return ""
    for item in response_items:
        if item.get("type") == "message" and item.get("role") == "assistant":
            return formats.item_text(item).strip()
    return ""


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


def _subagent_env() -> dict:
    env = os.environ.copy()
    env['LOKI_PROVIDER'] = chat_provider.kind
    env['LOKI_API_BASE'] = chat_provider.input_url
    env['LOKI_MODEL'] = model
    env['LOKI_API_KEY'] = api_key
    return env


def _format_started_background_job(job: Job, kind: str = "job") -> str:
    return "\n".join([
        f"Started background {kind} {job.id}",
        f"pid: {job.pid}",
        f"pgid: {job.pgid}",
        f"status: {job.status}",
        f"stdout: {job.stdout_path}",
        f"stderr: {job.stderr_path}",
    ])


def run_agent(description: str, prompt: str, run_in_background: bool = False,
              subagent_type: str = "Explore") -> str:
    return asyncio.run(run_agent_async(description, prompt, run_in_background, subagent_type))


async def run_agent_async(description: str, prompt: str, run_in_background: bool = False,
                          subagent_type: str = "Explore") -> str:
    agent_type = subagent_type or "Explore"
    if not prompt:
        return "Error: prompt is required"
    if agent_type != "Explore":
        return f"Error: unknown subagent_type {agent_type!r} (only 'Explore' is supported)"
    argv = [
        sys.executable,
        os.path.abspath(__file__),
        '--subagent',
        agent_type,
        '--prompt',
        prompt,
    ]
    if run_in_background:
        job = await job_manager.run_background_exec(
            argv,
            description=description or "subagent task",
            env=_subagent_env())
        return _format_started_background_job(job, "subagent")

    job, status, stdout, stderr = await job_manager.run_exec(
        argv, SUBAGENT_TIMEOUT_S * 1000,
        description=description or "subagent task",
        env=_subagent_env())
    if status == "timed_out":
        result = _format_subagent_result(agent_type, description, "timed_out",
                                         job.exit_code, stdout, stderr)
        return f"Error: subagent timed out after {SUBAGENT_TIMEOUT_S}s for {description or 'task'}\n{result}"
    result = _format_subagent_result(agent_type, description, "completed",
                                     job.exit_code, stdout, stderr)
    if job.exit_code != 0:
        return f"Error: subagent exited with code {job.exit_code}\n{result}"
    return result


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
    elif isinstance(headers, dict):
        content_type = headers.get('content-type') or headers.get('Content-Type') or ''
        for part in content_type.split(';')[1:]:
            name, sep, value = part.strip().partition('=')
            if sep and name.lower() == 'charset':
                candidates.append(value.strip('"'))
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


@dataclass
class HttpResponse:
    url: str
    status: int
    reason: str
    headers: dict
    body: bytes
    truncated: bool = False
    redirect_url: str | None = None

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name.lower(), default)


def _host_header(parsed) -> str:
    default_port = 443 if parsed.scheme == 'https' else 80
    if parsed.port and parsed.port != default_port:
        return f"{parsed.hostname}:{parsed.port}"
    return parsed.hostname or ''


def _request_target(parsed) -> str:
    path = parsed.path or '/'
    if parsed.params:
        path += ';' + parsed.params
    if parsed.query:
        path += '?' + parsed.query
    return path


async def _read_headers(reader: asyncio.StreamReader) -> tuple[str, dict]:
    total = 0
    status_line = await reader.readline()
    if not status_line:
        raise OSError("empty HTTP response")
    total += len(status_line)
    header_lines = []
    while True:
        line = await reader.readline()
        if not line:
            break
        total += len(line)
        if total > HTTP_HEADER_MAX_BYTES:
            raise OSError("HTTP headers too large")
        if line in [b'\r\n', b'\n']:
            break
        header_lines.append(line.decode('iso-8859-1').rstrip('\r\n'))

    headers_out = {}
    for line in header_lines:
        if ':' not in line:
            continue
        name, value = line.split(':', 1)
        key = name.strip().lower()
        value = value.strip()
        if key in headers_out:
            headers_out[key] += ', ' + value
        else:
            headers_out[key] = value
    return status_line.decode('iso-8859-1').rstrip('\r\n'), headers_out


async def _read_until_eof(reader: asyncio.StreamReader, max_bytes: int) -> tuple[bytes, bool]:
    chunks = []
    total = 0
    truncated = False
    while True:
        chunk = await reader.read(min(65536, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            truncated = True
            break
    body = b''.join(chunks)
    if len(body) > max_bytes:
        return body[:max_bytes], True
    return body, truncated


async def _read_content_length(reader: asyncio.StreamReader, length: int,
                               max_bytes: int) -> tuple[bytes, bool]:
    to_read = min(length, max_bytes + 1)
    body = await reader.readexactly(to_read) if to_read else b''
    truncated = length > max_bytes or len(body) > max_bytes
    if len(body) > max_bytes:
        body = body[:max_bytes]
    return body, truncated


async def _read_chunked_body(reader: asyncio.StreamReader, max_bytes: int) -> tuple[bytes, bool]:
    chunks = []
    total = 0
    truncated = False
    while True:
        line = await reader.readline()
        if not line:
            break
        size_text = line.split(b';', 1)[0].strip()
        try:
            size = int(size_text, 16)
        except ValueError:
            raise OSError(f"invalid chunk size {size_text!r}")
        if size == 0:
            await reader.readline()
            break
        data = await reader.readexactly(size)
        await reader.readexactly(2)  # CRLF
        keep = 0
        if total < max_bytes:
            keep = min(size, max_bytes - total)
            chunks.append(data[:keep])
            total += keep
        if size > keep:
            truncated = True
            break
    return b''.join(chunks), truncated


async def async_http_request(method: str, request_url: str, *, headers_in: dict = None,
                             body: bytes = b'', timeout: int = 30,
                             max_bytes: int = HTTP_MAX_RESPONSE_BYTES) -> HttpResponse:
    async def request_once() -> HttpResponse:
        parsed = urllib.parse.urlparse(request_url)
        if parsed.scheme not in ['http', 'https'] or not parsed.hostname:
            raise ValueError(f"unsupported URL: {request_url}")
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        ssl_context = ssl.create_default_context() if parsed.scheme == 'https' else None
        reader, writer = await asyncio.open_connection(
            parsed.hostname,
            port,
            ssl=ssl_context,
            server_hostname=parsed.hostname if ssl_context else None,
        )
        try:
            request_headers = {
                'Host': _host_header(parsed),
                'Connection': 'close',
            }
            if headers_in:
                request_headers.update(headers_in)
            if body:
                request_headers['Content-Length'] = str(len(body))
            lines = [f"{method.upper()} {_request_target(parsed)} HTTP/1.1"]
            lines.extend(f"{name}: {value}" for name, value in request_headers.items())
            raw_request = ("\r\n".join(lines) + "\r\n\r\n").encode('iso-8859-1') + body
            writer.write(raw_request)
            await writer.drain()

            status_line, response_headers = await _read_headers(reader)
            parts = status_line.split(' ', 2)
            if len(parts) < 2 or not parts[1].isdigit():
                raise OSError(f"invalid HTTP status line: {status_line!r}")
            status = int(parts[1])
            reason = parts[2] if len(parts) > 2 else ''
            transfer_encoding = response_headers.get('transfer-encoding', '').lower()
            if 'chunked' in transfer_encoding:
                response_body, truncated = await _read_chunked_body(reader, max_bytes)
            elif response_headers.get('content-length', '').isdigit():
                response_body, truncated = await _read_content_length(
                    reader, int(response_headers['content-length']), max_bytes)
            else:
                response_body, truncated = await _read_until_eof(reader, max_bytes)
            return HttpResponse(request_url, status, reason, response_headers, response_body, truncated)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    return await asyncio.wait_for(request_once(), timeout=timeout)


def _redirect_location(response: HttpResponse) -> str | None:
    location = response.header('location')
    if response.status in range(300, 400) and location:
        return urllib.parse.urljoin(response.url, location)
    return None


async def async_http_request_follow_same_host(method: str, request_url: str, *,
                                              headers_in: dict = None, body: bytes = b'',
                                              timeout: int = 30,
                                              max_bytes: int = HTTP_MAX_RESPONSE_BYTES,
                                              max_redirects: int = 5) -> HttpResponse:
    current_url = request_url
    original_host = urllib.parse.urlparse(request_url).netloc
    for _ in range(max_redirects + 1):
        response = await async_http_request(method, current_url, headers_in=headers_in,
                                            body=body, timeout=timeout, max_bytes=max_bytes)
        next_url = _redirect_location(response)
        if not next_url:
            return response
        next_host = urllib.parse.urlparse(next_url).netloc
        if next_host != original_host:
            response.redirect_url = next_url
            return response
        current_url = next_url
    return response


async def _fetch_url_async(url: str) -> dict:
    """GET a URL with redirect tracking, return dict with content/contentType/status/finalUrl/redirects.
    HTTP is upgraded to HTTPS. Cross-host redirects are surfaced, not followed."""
    if url.startswith('http://'):
        url = 'https://' + url[len('http://'):]
    elif not url.startswith('https://'):
        url = 'https://' + url

    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return {'error': f'invalid URL: {url}'}

    request_headers = {
        'User-Agent': 'loki-WebFetch/0.1 (coding-agent)',
        'Accept': 'text/markdown;q=1.0, text/html;q=0.9, text/plain;q=0.8, application/json;q=0.7, */*;q=0.1',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        response = await async_http_request_follow_same_host(
            'GET', url, headers_in=request_headers,
            timeout=WEBFETCH_TIMEOUT_S, max_bytes=WEBFETCH_MAX_BYTES)
        if response.redirect_url:
            return {'redirectUrl': response.redirect_url, 'status': response.status,
                    'finalUrl': response.url, 'error': None}
        content_type = response.header('content-type')
        body = _decode_http_text(response.body, response.headers)
        return {'content': body, 'contentType': content_type, 'status': response.status,
                'finalUrl': response.url, 'truncated': response.truncated, 'error': None}
    except Exception as e:
        return {'error': f'fetch failed: {e}', 'finalUrl': url}


async def run_webfetch_async(url: str, prompt: str) -> str:
    if not url:
        return "Error: url is required"
    if not prompt:
        return "Error: prompt is required"
    now = time.time()
    cached = _webfetch_cache.get(url)
    if cached and now - cached[0] < WEBFETCH_CACHE_TTL:
        content_text, content_type, final_url, status = cached[1], cached[2], cached[3], cached[4]
        cache_hit = True
    else:
        response = await _fetch_url_async(url)
        if response.get('error'):
            return f"Error: {response['error']}"
        if response.get('redirectUrl'):
            return "\n".join([
                f"WebFetch redirect: HTTP {response['status']}",
                f"requested_url: {response['finalUrl']}",
                f"redirect_url: {response['redirectUrl']}",
                "Call WebFetch again with redirect_url if you want to fetch that page.",
            ])
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

    msgs = [
        formats.instruction_item(
            "You are processing content fetched by the WebFetch tool. "
            "Answer only from the fetched page content. "
            "If the content does not contain the answer, say so plainly. "
            "Keep quotes short and do not reproduce large copyrighted passages."),
        formats.message_item(
            "user",
            f"URL: {final_url}\nContent-Type: {content_type}\nPrompt: {prompt}\n\n--- Page content ---\n{content_text}"),
    ]
    answer = await run_toolless_completion_async(msgs) or "(no answer returned)"
    header = f"[WebFetch status={status} cache_hit={cache_hit} bytes~={len(content_text)} url={final_url}]"
    return f"{header}\n{answer}"


async def run_websearch_async(query: str, allowed_domains: list = None,
                              blocked_domains: list = None) -> str:
    if not query or len(query) < 2:
        return "Error: query must be at least 2 characters"
    if allowed_domains and blocked_domains:
        return "Error: allowed_domains and blocked_domains cannot both be specified"
    if allowed_domains and len(allowed_domains) > 20:
        return "Error: allowed_domains max 20 entries"
    if blocked_domains and len(blocked_domains) > 20:
        return "Error: blocked_domains max 20 entries"

    form = urllib.parse.urlencode({'q': query, 'b': '', 'kl': 'us-en'})
    try:
        response = await async_http_request(
            'POST',
            DUCKDUCKGO_HTML_SEARCH_URL,
            body=form.encode('utf-8'),
            headers_in={'User-Agent': 'loki-WebSearch/0.1 (coding-agent)',
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.1'},
            timeout=WEBSEARCH_TIMEOUT_S,
            max_bytes=WEBSEARCH_MAX_RESPONSE_BYTES,
        )
        if response.status >= 400:
            return f"Error: web search request failed: HTTP {response.status} {response.reason}"
        html = _decode_http_text(response.body, response.headers)
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
                "- `run_in_background` starts a detached job with stdout/stderr spooled to files. Use Jobs/JobStatus/JobStop to inspect or control it. No `&` needed.",
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
            "name": "Jobs",
            "description": "List background jobs with their status, pid, exit code, and command.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "JobStatus",
            "description": "Inspect one shell job, including status, exit code, spool paths, and stdout/stderr tails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The job id returned by Bash, Agent, or Jobs"},
                    "tail_chars": {"type": "integer", "minimum": 0,
                                   "description": f"Maximum characters to show from each spool file. Defaults to {JOB_TAIL_CHARS}."}
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "JobStop",
            "description": "Stop a running shell job by sending SIGTERM to its process group, or SIGKILL if force is true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The job id returned by Bash, Agent, or Jobs"},
                    "force": {"type": "boolean",
                              "description": "Use SIGKILL instead of SIGTERM. Default false."}
                },
                "required": ["job_id"]
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
                "- Explore: Read-only search agent for broad fan-out searches - when answering means sweeping many files, directories, or naming conventions and you only need the conclusion, not the file dumps. It reads excerpts rather than whole files, so it locates code; it doesn't review or audit it. Specify search breadth: \"medium\" for moderate exploration, \"very thorough\" for multiple locations and naming conventions. (Tools: Glob, Grep, Read, Bash, Jobs, JobStatus, JobStop, WebFetch, WebSearch, TodoWrite)",
                "",
                "When using the Agent tool, specify a subagent_type parameter to select which agent type to use. If omitted, the \"Explore\" agent is used.",
                "",
                "## When to use",
                "",
                "Reach for this when the task matches an available agent type, when you have independent work to run in parallel, or when answering would mean reading across several files - delegate it and you keep the conclusion, not the file dumps. For a single-fact lookup where you already know the file, symbol, or value, search directly. Once you've delegated a search, don't also run it yourself - wait for the result.",
                "",
                "- The agent's final message is returned to you as the tool result; it is not shown to the user - relay what matters.",
                "- A new Agent call starts fresh, so the prompt must be self-contained.",
                "- `run_in_background` starts the subagent as a background job with stdout/stderr spooled to files. Use Jobs/JobStatus/JobStop to inspect or control it.",
                "- When you launch multiple agents for independent work, send them in a single message with multiple tool uses so they run concurrently",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "A short (3-5 word) description of the task"},
                    "prompt": {"type": "string", "description": "The task for the agent to perform"},
                    "run_in_background": {"type": "boolean",
                                          "description": "Set to true to run this agent as a background job. Use Jobs/JobStatus/JobStop to inspect or control it."},
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
    "Bash": {"handler": _handle_bash, "async_handler": _handle_bash_async, "explore": True},
    "Jobs": {"handler": _handle_jobs, "explore": True},
    "JobStatus": {"handler": _handle_job_status, "explore": True},
    "JobStop": {"handler": _handle_job_stop, "explore": True},
    "Glob": {"handler": _handle_glob, "async_handler": _handle_glob_async, "explore": True},
    "Grep": {"handler": _handle_grep, "async_handler": _handle_grep_async, "explore": True},
    "TodoRead": {"handler": _handle_todoread, "explore": True},
    "TodoWrite": {"handler": _handle_todowrite, "explore": True},
    "Agent": {"handler": _handle_agent, "async_handler": _handle_agent_async},
    "Skill": {"handler": _handle_skill},
    "WebFetch": {"async_handler": _handle_webfetch_async, "explore": True},
    "WebSearch": {"async_handler": _handle_websearch_async, "explore": True},
}
TOOL_REGISTRY = _build_tool_registry(TOOLS, TOOL_HANDLERS)
TOOLS = [spec["definition"] for spec in TOOL_REGISTRY.values()]
EXPLORE_TOOLS = {name for name, spec in TOOL_REGISTRY.items() if spec["explore"]}

async def async_chat_request(request_url: str, payload, request_headers: dict = None,
                             report_errors: bool = False, show_timing: bool = False) -> dict:
    start = time.perf_counter()
    try:
        body = json.dumps(payload).encode('utf-8') if payload is not None else b''
        method = 'POST' if payload is not None else 'GET'
        response = await async_http_request(
            method,
            request_url,
            body=body,
            headers_in=request_headers or headers,
            timeout=WEBFETCH_TIMEOUT_S,
            max_bytes=HTTP_MAX_RESPONSE_BYTES,
        )
        response_text = _decode_http_text(response.body, response.headers)
        if response.status >= 400:
            if report_errors:
                print(f"API Error for <{request_url}>: HTTP {response.status} {response.reason}: "
                      f"{response_text[:1000]}", file=sys.stderr)
            return None
        data = json.loads(response_text)
    except OSError as e:
        if report_errors:
            print(f"API Error for <{request_url}>: {e}", file=sys.stderr)
        return None

    elapsed = time.perf_counter() - start
    if show_timing:
        print(f"\n[T]  [LLM Response Time: {elapsed:.3f}s]", file=sys.stderr)
    return data


async def async_chat_completion(transcript_items: list, tools=TOOLS, report_errors: bool = False,
                                show_timing: bool = False) -> list:
    payload = chat_provider.chat_payload(transcript_items, tools, model)
    data = await async_chat_request(
        chat_provider.chat_url,
        payload,
        request_headers=chat_provider.headers,
        report_errors=report_errors,
        show_timing=show_timing,
    )
    if not data:
        return []
    detected = protocols.detect_protocol_from_response(data)
    if detected and detected != chat_provider.kind:
        raise protocols.ProtocolError(
            f"configured provider {chat_provider.kind!r} but response looks like {detected!r}")
    return chat_provider.parse_chat_response(data)


models = []
model = os.environ.get('LOKI_MODEL', 'glm-5.2')


async def load_models_async():
    global models
    global model
    if not chat_provider.models_url:
        models = [model]
        return
    data = await async_chat_request(
        chat_provider.models_url,
        None,
        request_headers=chat_provider.headers,
        report_errors=True,
    )
    if not data:
        models = [model]
        return
    loaded = chat_provider.parse_model_ids(data)
    if not loaded:
        models = [model]
        return
    models = loaded
    if model not in models:
        model = 'glm-5.2' if 'glm-5.2' in models else models[0]

#models = ['hy3-preview', 'glm-5.2', 'glm-5.1', 'kimi-k2.7', 'kimi-k2.6', 'deepseek-v4-pro', 'deepseek-v4-flash', 'mimo-v2.5', 'mimo-v2.5-pro']

terminals.set_status_text_provider(
    lambda: 'Model: {}; Hint: Use /quit to quit, /model to switch model, !foo to execute shell command foo'.format(model)
)

transcript_items = []
session_todos = []


def initial_transcript_items():
    return [formats.instruction_item(
        "You are a helpful system agent running in a terminal. You have these tools: "
        "Read, Write, Edit, Bash, Jobs, JobStatus, JobStop, Glob, Grep, TodoRead, TodoWrite, Agent, Skill, WebFetch, WebSearch. "
        "Prefer Glob/Grep/Read over Bash equivalents (find/grep/cat). "
        "Always Read a file before editing or overwriting it. "
        "Use TodoWrite to plan multi-step work. Keep responses concise."
    )]


def user_prompt_history(items):
    return formats.user_prompt_history(items)


def new_chat_log(filename):
    global chat_log
    global transcript_items
    global session_todos
    transcript_items = initial_transcript_items()
    session_todos = []
    chat_log = open(filename, 'w')

def save_chat_log():
    chat_log.seek(0)
    json.dump(formats.new_log_blob(transcript_items, session_todos), chat_log, indent=4)
    chat_log.truncate()
    chat_log.flush()
    print('Note: Saved chat log to {}'.format(chat_log.name), file=sys.stderr)
    sys.stderr.flush()

def load_chat_log(filename):
    global chat_log
    global transcript_items
    global session_todos
    chat_log = open(filename, 'r')
    try:
        blob = json.load(chat_log)
        transcript_items, session_todos = formats.load_log_blob(blob)
        for item in transcript_items:
            for k, v in item.items():
                pprint((k, v)) # TODO: nicer

            print()

        print('----')
    finally:
        chat_log.close()

    chat_log = open(filename, 'w')
    save_chat_log()

def run_subagent_prompt(subagent_type: str, prompt: str) -> str:
    return asyncio.run(run_subagent_prompt_async(subagent_type, prompt))


async def run_subagent_prompt_async(subagent_type: str, prompt: str) -> str:
    if subagent_type != "Explore":
        return f"Error: unknown subagent_type {subagent_type!r} (only 'Explore' is supported)"
    if not prompt:
        return ""
    msgs = [
        formats.instruction_item(
            "You are a focused Explore subagent. Use Glob/Grep/Read/Bash to investigate, then write a concise final answer."),
        formats.message_item("user", prompt),
    ]
    return await run_tool_loop_async(msgs, allowed=EXPLORE_TOOLS)


def run_subagent_cli(subagent_type: str, prompt: str = None):
    asyncio.run(run_subagent_cli_async(subagent_type, prompt))


async def run_subagent_cli_async(subagent_type: str, prompt: str = None):
    prompt = prompt if prompt is not None else sys.stdin.read().strip()
    result = await run_subagent_prompt_async(subagent_type, prompt)
    if result:
        print(result)


async def async_main(args):
    global model
    global transcript_items

    options, args = getopt.getopt(args, 'r:p:', ['resume=', 'prompt=', 'subagent=', 'headless', 'toolset=', 'dangerously-skip-permissions'])
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

    await load_models_async()

    if subagent_type or headless:
        await run_subagent_cli_async(subagent_type or toolset or "Explore", prompt_arg)
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

    while True:
        try:
            user_in = await get_input_async(history=user_prompt_history(transcript_items))
        except EOFError:
            break
        restore_output_area_after_input()

        if not user_in:
            terminal.save_cursor_position()
            continue

        print('User:', user_in)
        match user_in.strip():
            case '/quit':
                break
            case '/model':
                model = await run_menu_async(models)
                terminal.save_cursor_position()
                continue
            case _:
                if user_in.strip().startswith('!'): # direct command execution
                    cmd = user_in[1:].strip()
                    print(f"{computer}: [Running local command: {cmd}]")
                    cmd_output = await run_bash_async(cmd)
                    print(cmd_output) # Show output to you in the terminal
                    # Morph the user input so the AI sees exactly what you did and the result
                    user_in = f"I ran the local command `{cmd}`.\nOutput:\n```\n{cmd_output}\n```"
                else:
                    pass

        transcript_items.append(formats.message_item("user", user_in))

        try:
            await run_terminal_turn_async(transcript_items)
        except KeyboardInterrupt:
            terminal.reset_colors_and_flags()
            # ? EMERGENCY BRAKE
            print("\n\n? [EMERGENCY STOP] Agent execution cancelled by user!")
            # If we interrupt while a tool call was requested but unanswered, remove that
            # assistant item so every later render has matched tool call/result pairs.
            if transcript_items and formats.item_has_tool_calls(transcript_items[-1]):
                transcript_items.pop()

            transcript_items.append(formats.instruction_item(
                "CRITICAL: The user forcefully stopped your execution via KeyboardInterrupt (Ctrl+C). You were likely looping, making a mistake, or doing something dangerous. Await new instructions."))
            continue # Drop immediately back to the User> prompt
        terminal.save_cursor_position()

def main():
    asyncio.run(async_main(sys.argv[1:]))

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
    terminal.set_clipping_region(*terminals.output_area)
    terminal.goto_position(1, 1)
    terminal.save_cursor_position()

    try:
        main()
    finally:
        clean_up()
