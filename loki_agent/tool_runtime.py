import asyncio
import contextlib
import copy
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass, field


class ToolSchemaError(ValueError):
    pass


class HookConfigurationError(ValueError):
    pass


class HookExecutionError(RuntimeError):
    pass


SCHEMA_ANNOTATION_KEYS = {"description", "default", "format"}
SCHEMA_VALIDATION_KEYS = {
    "type", "properties", "required", "additionalProperties", "enum", "items",
    "minLength", "maxLength", "minimum", "maximum", "maxItems",
}
SCHEMA_ALLOWED_KEYS = SCHEMA_ANNOTATION_KEYS | SCHEMA_VALIDATION_KEYS

FILESYSTEM_PATH = "filesystem_path"
_PATH_AUTOLINK = re.compile(
    r"\[(?P<label>[^\[\]\r\n/]+)\]\(https?://(?P=label)\)")
_UNSET = object()


def format_path(path) -> str:
    result = "$"
    for key in path:
        if isinstance(key, int):
            result += f"[{key}]"
        else:
            result += f".{key}"
    return result


def json_type_name(value) -> str:
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


def matches_json_type(value, expected_type: str) -> bool:
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


def _expected_types(schema, path):
    if "type" not in schema:
        return []
    expected = schema["type"]
    if isinstance(expected, str):
        result = [expected]
    elif (isinstance(expected, list)
          and all(isinstance(item, str) for item in expected)):
        result = list(expected)
    else:
        raise ToolSchemaError(
            f"{format_path(path)}: type must be a string or list of strings")
    for expected_type in result:
        matches_json_type(None, expected_type)
    return result


@dataclass(frozen=True)
class ValidationIssue:
    path: tuple
    code: str
    message: str
    expected: object = None
    actual_type: str | None = None

    def to_dict(self):
        result = {
            "path": list(self.path),
            "display_path": format_path(self.path),
            "code": self.code,
            "message": self.message,
        }
        if self.expected is not None:
            result["expected"] = copy.deepcopy(self.expected)
        if self.actual_type is not None:
            result["actual_type"] = self.actual_type
        return result


def validate_schema(schema: dict, value, path=()) -> list[ValidationIssue]:
    if not isinstance(schema, dict):
        raise ToolSchemaError(
            f"{format_path(path)}: schema must be an object")
    unsupported = sorted(set(schema) - SCHEMA_ALLOWED_KEYS)
    if unsupported:
        raise ToolSchemaError(
            f"{format_path(path)}: unsupported schema keys: "
            + ", ".join(unsupported))

    expected_types = _expected_types(schema, path)
    if (expected_types
            and not any(matches_json_type(value, expected_type)
                        for expected_type in expected_types)):
        expected_label = " or ".join(expected_types)
        actual_type = json_type_name(value)
        return [ValidationIssue(
            tuple(path),
            "type",
            f"{format_path(path)} must be {expected_label}, "
            f"got {actual_type}",
            expected=expected_types,
            actual_type=actual_type,
        )]

    issues = []
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(item) for item in schema["enum"])
        issues.append(ValidationIssue(
            tuple(path),
            "enum",
            f"{format_path(path)} must be one of: {allowed}",
            expected=copy.deepcopy(schema["enum"]),
            actual_type=json_type_name(value),
        ))

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ToolSchemaError(
                f"{format_path(path)}: properties must be an object")
        required = schema.get("required", [])
        if (not isinstance(required, list)
                or not all(isinstance(item, str) for item in required)):
            raise ToolSchemaError(
                f"{format_path(path)}: required must be a list of strings")
        for key in required:
            if key not in value:
                child_path = tuple(path) + (key,)
                issues.append(ValidationIssue(
                    child_path,
                    "required",
                    f"{format_path(child_path)} is required",
                    expected="present",
                    actual_type="missing",
                ))
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for key in sorted(set(value) - set(properties)):
                child_path = tuple(path) + (key,)
                issues.append(ValidationIssue(
                    child_path,
                    "additional_property",
                    f"{format_path(child_path)} is not allowed",
                    expected="absent",
                    actual_type=json_type_name(value[key]),
                ))
        elif additional is not True:
            raise ToolSchemaError(
                f"{format_path(path)}: additionalProperties must be "
                "true or false")
        for key, subschema in properties.items():
            if key in value:
                issues.extend(validate_schema(
                    subschema, value[key], tuple(path) + (key,)))

    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            issues.append(ValidationIssue(
                tuple(path),
                "max_items",
                f"{format_path(path)} must contain at most "
                f"{schema['maxItems']} items",
                expected=schema["maxItems"],
                actual_type="array",
            ))
        if "items" in schema:
            for index, item in enumerate(value):
                issues.extend(validate_schema(
                    schema["items"], item, tuple(path) + (index,)))

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            issues.append(ValidationIssue(
                tuple(path),
                "min_length",
                f"{format_path(path)} must be at least "
                f"{schema['minLength']} characters",
                expected=schema["minLength"],
                actual_type="string",
            ))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            issues.append(ValidationIssue(
                tuple(path),
                "max_length",
                f"{format_path(path)} must be at most "
                f"{schema['maxLength']} characters",
                expected=schema["maxLength"],
                actual_type="string",
            ))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            issues.append(ValidationIssue(
                tuple(path),
                "minimum",
                f"{format_path(path)} must be >= {schema['minimum']}",
                expected=schema["minimum"],
                actual_type=json_type_name(value),
            ))
        if "maximum" in schema and value > schema["maximum"]:
            issues.append(ValidationIssue(
                tuple(path),
                "maximum",
                f"{format_path(path)} must be <= {schema['maximum']}",
                expected=schema["maximum"],
                actual_type=json_type_name(value),
            ))
    return issues


def _value_at(value, path):
    current = value
    for key in path:
        current = current[key]
    return current


def _schema_at(schema, path):
    current = schema
    for key in path:
        if isinstance(key, int):
            current = current.get("items", {})
        else:
            current = current.get("properties", {}).get(key, {})
    return current


def _parent_schema_at(schema, path):
    return _schema_at(schema, path[:-1])


def _replace_at(value, path, replacement):
    if not path:
        return copy.deepcopy(replacement)
    parent = _value_at(value, path[:-1])
    parent[path[-1]] = copy.deepcopy(replacement)
    return value


def _remove_at(value, path):
    if not path:
        raise ValueError("cannot remove the root tool input")
    parent = _value_at(value, path[:-1])
    del parent[path[-1]]


def _semantic_issues(value, semantics):
    issues = []
    for raw_path, semantic in (semantics or {}).items():
        path = tuple(raw_path)
        if semantic != FILESYSTEM_PATH:
            raise ToolSchemaError(
                f"unsupported argument semantic {semantic!r} at "
                f"{format_path(path)}")
        try:
            current = _value_at(value, path)
        except (KeyError, IndexError, TypeError):
            continue
        if (isinstance(current, str)
                and _PATH_AUTOLINK.search(current) is not None):
            issues.append(ValidationIssue(
                path,
                "path_markdown_autolink",
                f"{format_path(path)} contains a Markdown auto-link where "
                "a plain filesystem path is required",
                expected="plain filesystem path",
                actual_type="string",
            ))
    return issues


def validate_tool_input(schema, value, semantics=None):
    issues = validate_schema(schema, value)
    if not issues:
        issues.extend(_semantic_issues(value, semantics))
    return issues


@dataclass
class ToolAdjustment:
    hook: str
    rule: str
    path: tuple
    operation: str
    value: object = _UNSET

    def to_dict(self):
        result = {
            "hook": self.hook,
            "rule": self.rule,
            "path": list(self.path),
            "display_path": format_path(self.path),
            "operation": self.operation,
        }
        if self.value is not _UNSET:
            result["value"] = copy.deepcopy(self.value)
        return result


@dataclass
class RepairResult:
    value: object
    issues: list[ValidationIssue]
    adjustments: list[ToolAdjustment]


def _array_item_accepts(schema, value):
    item_schema = schema.get("items")
    return item_schema is None or not validate_schema(item_schema, value)


def repair_tool_input(schema, value, semantics=None,
                      hook_id="loki.input-repair") -> RepairResult:
    effective = copy.deepcopy(value)
    adjustments = []
    issues = validate_tool_input(schema, effective, semantics)
    if not issues:
        return RepairResult(effective, [], [])

    for _pass in range(8):
        changed = False
        for issue in list(issues):
            try:
                current = _value_at(effective, issue.path)
            except (KeyError, IndexError, TypeError):
                continue
            field_schema = _schema_at(schema, issue.path)
            if issue.code == "path_markdown_autolink":
                replacement = _PATH_AUTOLINK.sub(
                    lambda match: match.group("label"), current)
                if replacement != current:
                    effective = _replace_at(
                        effective, issue.path, replacement)
                    adjustments.append(ToolAdjustment(
                        hook_id,
                        "path_markdown_autolink",
                        issue.path,
                        "replace",
                        replacement,
                    ))
                    changed = True
                continue
            if issue.code != "type":
                continue
            expected_types = _expected_types(field_schema, issue.path)
            if current is None and issue.path:
                parent_schema = _parent_schema_at(schema, issue.path)
                required = parent_schema.get("required", [])
                if (isinstance(issue.path[-1], str)
                        and issue.path[-1] not in required
                        and "null" not in expected_types):
                    _remove_at(effective, issue.path)
                    adjustments.append(ToolAdjustment(
                        hook_id,
                        "optional_null_omission",
                        issue.path,
                        "remove",
                    ))
                    changed = True
                continue
            if "array" not in expected_types:
                continue
            replacement = _UNSET
            rule = None
            if isinstance(current, str):
                try:
                    parsed = json.loads(current)
                except json.JSONDecodeError:
                    parsed = _UNSET
                if isinstance(parsed, list):
                    replacement = parsed
                    rule = "json_encoded_array"
                elif _array_item_accepts(field_schema, current):
                    replacement = [current]
                    rule = "bare_string_array"
            elif current == {}:
                replacement = []
                rule = "empty_object_array"
            if replacement is not _UNSET:
                effective = _replace_at(
                    effective, issue.path, replacement)
                adjustments.append(ToolAdjustment(
                    hook_id,
                    rule,
                    issue.path,
                    "replace",
                    replacement,
                ))
                changed = True
        new_issues = validate_tool_input(schema, effective, semantics)
        if not new_issues:
            return RepairResult(effective, [], adjustments)
        if not changed:
            return RepairResult(effective, new_issues, adjustments)
        issues = new_issues
    return RepairResult(effective, issues, adjustments)


def diff_values(before, after, hook_id, path=()):
    if type(before) is not type(after):
        return [ToolAdjustment(
            hook_id, "custom_hook", tuple(path), "replace", after)]
    if isinstance(before, dict):
        changes = []
        for key in sorted(set(before) - set(after)):
            changes.append(ToolAdjustment(
                hook_id, "custom_hook", tuple(path) + (key,), "remove"))
        for key in sorted(set(after) - set(before)):
            changes.append(ToolAdjustment(
                hook_id, "custom_hook", tuple(path) + (key,),
                "add", after[key]))
        for key in sorted(set(before) & set(after)):
            changes.extend(diff_values(
                before[key], after[key], hook_id, tuple(path) + (key,)))
        return changes
    if isinstance(before, list):
        if before == after:
            return []
        return [ToolAdjustment(
            hook_id, "custom_hook", tuple(path), "replace", after)]
    if before != after:
        return [ToolAdjustment(
            hook_id, "custom_hook", tuple(path), "replace", after)]
    return []


@dataclass
class HookRecord:
    hook: str
    phase: str
    status: str
    message: str | None = None

    def to_dict(self):
        result = {
            "hook": self.hook,
            "phase": self.phase,
            "status": self.status,
        }
        if self.message:
            result["message"] = self.message
        return result


@dataclass
class ToolInvocation:
    call_id: str | None
    tool_name: str
    raw_arguments: object
    original_arguments: object
    effective_arguments: object
    schema: dict
    semantics: dict = field(default_factory=dict)
    cwd: str | None = None
    model: str | None = None
    provider: str | None = None
    parse_error: str | None = None
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    adjustments: list[ToolAdjustment] = field(default_factory=list)
    hook_records: list[HookRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    denied_reason: str | None = None
    denied_by: str | None = None
    changed_paths: list[str] = field(default_factory=list)
    invalidate_all_files: bool = False

    def to_hook_dict(self):
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "raw_arguments": copy.deepcopy(self.raw_arguments),
            "original_arguments": copy.deepcopy(self.original_arguments),
            "effective_arguments": copy.deepcopy(self.effective_arguments),
            "schema": copy.deepcopy(self.schema),
            "cwd": self.cwd,
            "model": self.model,
            "provider": self.provider,
            "parse_error": self.parse_error,
            "validation_issues": [
                issue.to_dict() for issue in self.validation_issues],
            "adjustments": [
                adjustment.to_dict()
                for adjustment in self.adjustments
            ],
        }


@dataclass
class PreHookDecision:
    action: str = "continue"
    arguments: object = _UNSET
    message: str | None = None
    note: str | None = None
    changed_paths: list[str] = field(default_factory=list)
    invalidate_all_files: bool = False


@dataclass
class PostHookDecision:
    note: str | None = None
    changed_paths: list[str] = field(default_factory=list)
    invalidate_all_files: bool = False


@dataclass
class ToolOutcome:
    status: str
    executed: bool
    ok: bool
    content: str

    def to_hook_dict(self):
        return {
            "status": self.status,
            "executed": self.executed,
            "ok": self.ok,
            "content": self.content,
        }


@dataclass
class RegisteredHook:
    hook_id: str
    callback: object
    on_error: str
    matcher: object = None


async def _call_hook(callback, *args):
    result = callback(*args)
    if inspect.isawaitable(result):
        return await result
    return result


class ToolHookPipeline:
    def __init__(self):
        self.pre_hooks = []
        self.gate_hooks = []
        self.post_hooks = []

    def add_pre(self, hook_id, callback, on_error="deny", matcher=None):
        self.pre_hooks.append(RegisteredHook(
            hook_id, callback, on_error, matcher))

    def add_gate(self, hook_id, callback, on_error="deny", matcher=None):
        self.gate_hooks.append(RegisteredHook(
            hook_id, callback, on_error, matcher))

    def add_post(self, hook_id, callback, on_error="continue", matcher=None):
        self.post_hooks.append(RegisteredHook(
            hook_id, callback, on_error, matcher))

    @property
    def has_custom_hooks(self):
        return bool(self.pre_hooks or self.gate_hooks or self.post_hooks)

    @staticmethod
    def _record_side_effects(invocation, decision):
        invocation.changed_paths.extend(decision.changed_paths)
        if decision.invalidate_all_files:
            invocation.invalidate_all_files = True

    async def _run_pre_hook(self, invocation, registered, gate=False):
        phase = "pre_tool_gate" if gate else "pre_tool_call"
        try:
            decision = await _call_hook(
                registered.callback, copy.deepcopy(invocation))
            if decision is None:
                decision = PreHookDecision()
            if not isinstance(decision, PreHookDecision):
                raise HookExecutionError(
                    "pre-tool hook returned an invalid decision")
            if decision.action not in ["continue", "deny"]:
                raise HookExecutionError(
                    f"unknown pre-tool action {decision.action!r}")
            if gate and decision.arguments is not _UNSET:
                raise HookExecutionError(
                    "a gate hook cannot replace tool arguments")
            self._record_side_effects(invocation, decision)
            if decision.note:
                invocation.notes.append(decision.note)
            if decision.action == "deny":
                invocation.denied_reason = (
                    decision.message
                    or f"Tool call denied by hook {registered.hook_id}.")
                invocation.denied_by = registered.hook_id
                invocation.hook_records.append(HookRecord(
                    registered.hook_id, phase, "denied",
                    invocation.denied_reason))
                return
            if decision.arguments is not _UNSET:
                before = invocation.effective_arguments
                after = copy.deepcopy(decision.arguments)
                invocation.adjustments.extend(diff_values(
                    before, after, registered.hook_id))
                invocation.effective_arguments = after
                invocation.parse_error = None
                invocation.validation_issues = validate_tool_input(
                    invocation.schema,
                    invocation.effective_arguments,
                    invocation.semantics,
                )
            invocation.hook_records.append(HookRecord(
                registered.hook_id, phase, "ok"))
        except Exception as error:
            message = (
                f"{type(error).__name__}: {error}")
            invocation.hook_records.append(HookRecord(
                registered.hook_id, phase, "error", message))
            if registered.on_error == "deny":
                invocation.denied_reason = (
                    f"Tool call was not executed because pre-tool hook "
                    f"{registered.hook_id!r} failed: {message}")
                invocation.denied_by = registered.hook_id
            else:
                invocation.notes.append(
                    f"Pre-tool hook {registered.hook_id!r} failed; "
                    f"execution continued: {message}")

    async def prepare(self, invocation):
        if invocation.parse_error is None:
            repaired = repair_tool_input(
                invocation.schema,
                invocation.effective_arguments,
                invocation.semantics,
            )
            invocation.effective_arguments = repaired.value
            invocation.adjustments.extend(repaired.adjustments)
            invocation.validation_issues = repaired.issues
        else:
            invocation.validation_issues = [ValidationIssue(
                (),
                "json_parse",
                f"$ is not valid JSON: {invocation.parse_error}",
                expected="object",
                actual_type="invalid_json",
            )]

        for registered in self.pre_hooks:
            if (registered.matcher is not None
                    and not registered.matcher(invocation.tool_name)):
                continue
            await self._run_pre_hook(invocation, registered)
            if invocation.denied_reason:
                return invocation

        if invocation.parse_error is None:
            invocation.validation_issues = validate_tool_input(
                invocation.schema,
                invocation.effective_arguments,
                invocation.semantics,
            )
        if invocation.validation_issues:
            return invocation

        for registered in self.gate_hooks:
            if (registered.matcher is not None
                    and not registered.matcher(invocation.tool_name)):
                continue
            await self._run_pre_hook(
                invocation, registered, gate=True)
            if invocation.denied_reason:
                return invocation
        return invocation

    async def finish(self, invocation, outcome):
        for registered in self.post_hooks:
            if (registered.matcher is not None
                    and not registered.matcher(invocation.tool_name)):
                continue
            try:
                decision = await _call_hook(
                    registered.callback,
                    copy.deepcopy(invocation),
                    copy.deepcopy(outcome),
                )
                if decision is None:
                    decision = PostHookDecision()
                if not isinstance(decision, PostHookDecision):
                    raise HookExecutionError(
                        "post-tool hook returned an invalid decision")
                self._record_side_effects(invocation, decision)
                if decision.note:
                    invocation.notes.append(decision.note)
                invocation.hook_records.append(HookRecord(
                    registered.hook_id, "post_tool_call", "ok"))
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                invocation.hook_records.append(HookRecord(
                    registered.hook_id,
                    "post_tool_call",
                    "error",
                    message,
                ))
                timing = (
                    "after the tool had already executed"
                    if outcome.executed else "after rejection")
                invocation.notes.append(
                    f"Post-tool hook {registered.hook_id!r} failed "
                    f"{timing}: {message}")
        return invocation, outcome


async def _read_stream_limited(stream, limit):
    chunks = []
    size = 0
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > limit:
            raise HookExecutionError(
                f"hook output exceeded {limit} bytes")
        chunks.append(chunk)


def _hook_environment(environ=None):
    source_environ = os.environ if environ is None else environ
    allowed = {
        "HOME", "LANG", "LC_ALL", "LOGNAME", "PATH", "SHELL",
        "TERM", "TMPDIR", "USER",
    }
    return {
        key: value for key, value in source_environ.items()
        if key in allowed or key.startswith("LC_")
    }


async def _run_hook_command(command, payload, cwd, timeout_ms,
                            stdout_limit=1_000_000,
                            stderr_limit=100_000,
                            stderr_reporter=None):
    try:
        stdin_data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HookExecutionError(
            f"hook input is not JSON-compatible: {error}") from error
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=_hook_environment(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
    except OSError as error:
        raise HookExecutionError(
            f"could not start hook command: {error}") from error
    stdout_task = asyncio.create_task(
        _read_stream_limited(process.stdout, stdout_limit))
    stderr_task = asyncio.create_task(
        _read_stream_limited(process.stderr, stderr_limit))

    async def write_stdin():
        try:
            process.stdin.write(stdin_data)
            await process.stdin.drain()
        finally:
            process.stdin.close()
            # Python 3.10 records a broken child stdin on the stream's close
            # waiter. Await it so timeout cancellation consumes that result
            # instead of leaking an unretrieved-future warning at loop exit.
            with contextlib.suppress(
                    BrokenPipeError, ConnectionError, OSError):
                await process.stdin.wait_closed()

    stdin_task = asyncio.create_task(write_stdin())

    async def terminate():
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
        for task in [stdin_task, stdout_task, stderr_task]:
            if not task.done():
                task.cancel()
        await asyncio.gather(
            stdin_task, stdout_task, stderr_task, return_exceptions=True)

    try:
        _stdin_result, stdout, stderr, returncode = await asyncio.wait_for(
            asyncio.gather(
                stdin_task, stdout_task, stderr_task, process.wait()),
            timeout=timeout_ms / 1000,
        )
    except asyncio.TimeoutError as error:
        await terminate()
        raise HookExecutionError(
            f"hook timed out after {timeout_ms}ms") from error
    except BaseException:
        await terminate()
        raise
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    if returncode != 0:
        detail = f": {stderr_text}" if stderr_text else ""
        raise HookExecutionError(
            f"hook exited with status {returncode}{detail}")
    if stderr_text:
        if stderr_reporter is None:
            sys.stdout.flush()
            print(
                f"Hook {command[0]!r} stderr:\n{stderr_text}",
                file=sys.stderr,
            )
            sys.stderr.flush()
        else:
            stderr_reporter(command, stderr_text)
    if not stdout.strip():
        return {}
    try:
        result = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HookExecutionError(
            f"hook stdout is not one JSON object: {error}") from error
    if not isinstance(result, dict):
        raise HookExecutionError(
            "hook stdout JSON must be an object")
    return result


@dataclass
class ExternalHook:
    hook_id: str
    tools: tuple
    command: tuple
    timeout_ms: int
    workspace_side_effects: bool = False
    event_name: str = "pre_tool_call"
    stderr_reporter: object = None

    def matches(self, tool_name):
        return "*" in self.tools or tool_name in self.tools

    def _side_effects(self, result):
        changed_paths = result.get("changed_paths", [])
        if (not isinstance(changed_paths, list)
                or not all(isinstance(path, str)
                           for path in changed_paths)):
            raise HookExecutionError(
                "changed_paths must be a list of strings")
        invalidate_all = (
            self.workspace_side_effects and not changed_paths)
        return changed_paths, invalidate_all

    async def pre(self, invocation):
        if not self.matches(invocation.tool_name):
            return PreHookDecision()
        result = await _run_hook_command(
            self.command,
            {
                "event": self.event_name,
                "invocation": invocation.to_hook_dict(),
            },
            invocation.cwd,
            self.timeout_ms,
            stderr_reporter=self.stderr_reporter,
        )
        action = result.get("action", "continue")
        arguments = result.get("arguments", _UNSET)
        message = result.get("message")
        note = result.get("note")
        if message is not None and not isinstance(message, str):
            raise HookExecutionError("message must be a string")
        if note is not None and not isinstance(note, str):
            raise HookExecutionError("note must be a string")
        changed_paths, invalidate_all = self._side_effects(result)
        return PreHookDecision(
            action=action,
            arguments=arguments,
            message=message,
            note=note,
            changed_paths=changed_paths,
            invalidate_all_files=invalidate_all,
        )

    async def post(self, invocation, outcome):
        if not self.matches(invocation.tool_name):
            return PostHookDecision()
        result = await _run_hook_command(
            self.command,
            {
                "event": "post_tool_call",
                "invocation": invocation.to_hook_dict(),
                "outcome": outcome.to_hook_dict(),
            },
            invocation.cwd,
            self.timeout_ms,
            stderr_reporter=self.stderr_reporter,
        )
        note = result.get("note")
        if note is not None and not isinstance(note, str):
            raise HookExecutionError("note must be a string")
        changed_paths, invalidate_all = self._side_effects(result)
        return PostHookDecision(
            note=note,
            changed_paths=changed_paths,
            invalidate_all_files=invalidate_all,
        )


def _external_hook_from_dict(
        value, path, default_timeout, stderr_reporter=None):
    if not isinstance(value, dict):
        raise HookConfigurationError(f"{path} must be an object")
    hook_id = value.get("id")
    if not isinstance(hook_id, str) or not hook_id:
        raise HookConfigurationError(
            f"{path}.id must be a nonempty string")
    tools = value.get("tools", ["*"])
    if (not isinstance(tools, list) or not tools
            or not all(isinstance(tool, str) and tool
                       for tool in tools)):
        raise HookConfigurationError(
            f"{path}.tools must be a nonempty list of strings")
    command = value.get("command")
    if (not isinstance(command, list) or not command
            or not all(isinstance(part, str) and part
                       for part in command)):
        raise HookConfigurationError(
            f"{path}.command must be a nonempty argv list")
    timeout_ms = value.get("timeout_ms", default_timeout)
    if (not isinstance(timeout_ms, int)
            or isinstance(timeout_ms, bool)
            or timeout_ms < 1):
        raise HookConfigurationError(
            f"{path}.timeout_ms must be a positive integer")
    side_effects = value.get("workspace_side_effects", False)
    if not isinstance(side_effects, bool):
        raise HookConfigurationError(
            f"{path}.workspace_side_effects must be boolean")
    on_error = value.get("on_error")
    if on_error is not None and on_error not in ["deny", "continue"]:
        raise HookConfigurationError(
            f"{path}.on_error must be 'deny' or 'continue'")
    return ExternalHook(
        hook_id,
        tuple(tools),
        tuple(command),
        timeout_ms,
        workspace_side_effects=side_effects,
        stderr_reporter=stderr_reporter,
    ), on_error


def load_hook_pipeline(path, stderr_reporter=None):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise HookConfigurationError(
            f"could not load hook configuration {path!r}: {error}") from error
    if not isinstance(config, dict):
        raise HookConfigurationError(
            "hook configuration must be an object")
    allowed = {"pre_tool_call", "pre_tool_gate", "post_tool_call"}
    unexpected = sorted(set(config) - allowed)
    if unexpected:
        raise HookConfigurationError(
            "unknown hook configuration keys: "
            + ", ".join(unexpected))
    pipeline = ToolHookPipeline()
    seen_ids = set()
    for section, default_timeout in [
            ("pre_tool_call", 2000),
            ("pre_tool_gate", 2000),
            ("post_tool_call", 10000)]:
        values = config.get(section, [])
        if not isinstance(values, list):
            raise HookConfigurationError(
                f"{section} must be a list")
        for index, value in enumerate(values):
            hook, on_error = _external_hook_from_dict(
                value,
                f"{section}[{index}]",
                default_timeout,
                stderr_reporter=stderr_reporter,
            )
            hook.event_name = section
            if section == "post_tool_call" and on_error == "deny":
                raise HookConfigurationError(
                    f"{section}[{index}].on_error cannot be 'deny': "
                    "post-tool failures cannot undo an attempted tool")
            if hook.hook_id in seen_ids:
                raise HookConfigurationError(
                    f"duplicate hook id {hook.hook_id!r}")
            seen_ids.add(hook.hook_id)
            if section == "pre_tool_call":
                pipeline.add_pre(
                    hook.hook_id,
                    hook.pre,
                    on_error=on_error or "deny",
                    matcher=hook.matches,
                )
            elif section == "pre_tool_gate":
                pipeline.add_gate(
                    hook.hook_id,
                    hook.pre,
                    on_error=on_error or "deny",
                    matcher=hook.matches,
                )
            else:
                pipeline.add_post(
                    hook.hook_id,
                    hook.post,
                    on_error=on_error or "continue",
                    matcher=hook.matches,
                )
    return pipeline


def adjustment_summary(adjustment):
    path = format_path(adjustment.path)
    messages = {
        "optional_null_omission":
            f"{path}: omitted optional null field",
        "json_encoded_array":
            f"{path}: parsed JSON-encoded array",
        "empty_object_array":
            f"{path}: replaced empty object placeholder with an empty array",
        "bare_string_array":
            f"{path}: wrapped a bare string as a one-item array",
        "path_markdown_autolink":
            f"{path}: unwrapped a degenerate Markdown path auto-link",
        "custom_hook":
            f"{path}: changed by hook {adjustment.hook!r}",
    }
    return messages.get(
        adjustment.rule,
        f"{path}: adjusted by hook {adjustment.hook!r}",
    )


def invalid_input_message(tool_name, issues):
    lines = [
        f"{tool_name} was not executed because its arguments are invalid:"
    ]
    lines.extend(f"- {issue.message}" for issue in issues)
    lines.append(f"Retry {tool_name} with corrected arguments.")
    return "\n".join(lines)
