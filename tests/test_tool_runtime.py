import asyncio
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from loki_entrypoints import child_environment, configure_container, entrypoint
from response_header_fixtures import setUpModule  # noqa: F401 - unittest hook


from loki_agent import formats
from loki_agent import tool_runtime
from loki_agent import loki


ARRAY_SCHEMA = {
    "type": "object",
    "properties": {
        "values": {
            "type": "array",
            "items": {"type": "string"},
        },
        "optional": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["values"],
    "additionalProperties": False,
}


class StructuredValidationTests(unittest.TestCase):
    def test_collects_all_independent_issues(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "mode": {"type": "string", "enum": ["a", "b"]},
            },
            "required": ["count", "mode", "missing"],
            "additionalProperties": False,
        }

        issues = tool_runtime.validate_schema(
            schema,
            {"count": "three", "mode": "c", "extra": True},
        )

        self.assertEqual(
            [(issue.path, issue.code) for issue in issues],
            [
                (("missing",), "required"),
                (("extra",), "additional_property"),
                (("count",), "type"),
                (("mode",), "enum"),
            ],
        )

    def test_valid_json_shaped_string_is_never_preprocessed(self):
        original = {
            "values": ["a"],
            "content": '{"looks":["like","json"]}',
        }

        repaired = tool_runtime.repair_tool_input(
            ARRAY_SCHEMA, original)

        self.assertEqual(repaired.value, original)
        self.assertEqual(repaired.adjustments, [])
        self.assertEqual(repaired.issues, [])


class InputRepairTests(unittest.TestCase):
    def repair(self, value, semantics=None):
        return tool_runtime.repair_tool_input(
            ARRAY_SCHEMA, value, semantics)

    def test_optional_null_is_omitted(self):
        result = self.repair({
            "values": ["a"],
            "optional": None,
        })

        self.assertEqual(result.value, {"values": ["a"]})
        self.assertEqual(
            [item.rule for item in result.adjustments],
            ["optional_null_omission"],
        )

    def test_json_encoded_array_is_parsed_before_string_wrapping(self):
        result = self.repair({"values": '["a","b"]'})

        self.assertEqual(result.value["values"], ["a", "b"])
        self.assertEqual(
            [item.rule for item in result.adjustments],
            ["json_encoded_array"],
        )

    def test_empty_object_placeholder_becomes_empty_array(self):
        result = self.repair({"values": {}})

        self.assertEqual(result.value["values"], [])
        self.assertEqual(
            [item.rule for item in result.adjustments],
            ["empty_object_array"],
        )

    def test_bare_string_becomes_one_item_array(self):
        result = self.repair({"values": "a"})

        self.assertEqual(result.value["values"], ["a"])
        self.assertEqual(
            [item.rule for item in result.adjustments],
            ["bare_string_array"],
        )

    def test_required_null_is_not_removed(self):
        schema = copy.deepcopy(ARRAY_SCHEMA)
        schema["required"].append("optional")

        result = tool_runtime.repair_tool_input(
            schema, {"values": [], "optional": None})

        self.assertEqual(result.adjustments, [])
        self.assertEqual(
            [(issue.path, issue.code) for issue in result.issues],
            [(("optional",), "type")],
        )

    def test_nonempty_object_is_not_guessed_as_array(self):
        result = self.repair({"values": {"a": "b"}})

        self.assertEqual(result.adjustments, [])
        self.assertEqual(result.issues[0].code, "type")

    def test_repair_that_reveals_bad_array_item_does_not_execute_cleanly(self):
        result = self.repair({"values": '["a", 3]'})

        self.assertEqual(
            [item.rule for item in result.adjustments],
            ["json_encoded_array"],
        )
        self.assertEqual(
            [(issue.path, issue.code) for issue in result.issues],
            [(("values", 1), "type")],
        )

    def test_degenerate_markdown_path_component_is_unwrapped(self):
        schema = {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
            "additionalProperties": False,
        }
        original = {
            "file_path":
                "/tmp/project/[notes.md](http://notes.md)",
        }

        result = tool_runtime.repair_tool_input(
            schema,
            original,
            {("file_path",): tool_runtime.FILESYSTEM_PATH},
        )

        self.assertEqual(
            result.value["file_path"],
            "/tmp/project/notes.md",
        )
        self.assertEqual(
            [item.rule for item in result.adjustments],
            ["path_markdown_autolink"],
        )
        self.assertEqual(
            original["file_path"],
            "/tmp/project/[notes.md](http://notes.md)",
        )

    def test_real_markdown_link_and_non_path_strings_are_untouched(self):
        schema = {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
            "additionalProperties": False,
        }
        original = {
            "file_path": "[click](https://example.com)",
            "content": "[notes.md](http://notes.md)",
        }

        result = tool_runtime.repair_tool_input(
            schema,
            original,
            {("file_path",): tool_runtime.FILESYSTEM_PATH},
        )

        self.assertEqual(result.value, original)
        self.assertEqual(result.adjustments, [])

    def test_successful_repairs_are_valid_idempotent_and_nonmutating(self):
        cases = [
            {"values": "a"},
            {"values": '["a","b"]'},
            {"values": {}},
            {"values": [], "optional": None},
        ]
        for original in cases:
            with self.subTest(original=original):
                preserved = copy.deepcopy(original)
                first = self.repair(original)
                second = self.repair(first.value)

                self.assertEqual(original, preserved)
                self.assertEqual(first.issues, [])
                self.assertEqual(
                    tool_runtime.validate_schema(
                        ARRAY_SCHEMA, first.value),
                    [],
                )
                self.assertEqual(second.value, first.value)
                self.assertEqual(second.adjustments, [])


class HookPipelineTests(unittest.TestCase):
    def invocation(self, args):
        return tool_runtime.ToolInvocation(
            call_id="call_1",
            tool_name="Example",
            raw_arguments=copy.deepcopy(args),
            original_arguments=copy.deepcopy(args),
            effective_arguments=copy.deepcopy(args),
            schema=ARRAY_SCHEMA,
            cwd=os.getcwd(),
        )

    def test_invalid_custom_mutation_is_rejected_before_gate(self):
        pipeline = tool_runtime.ToolHookPipeline()
        gate_calls = []

        def transform(invocation):
            return tool_runtime.PreHookDecision(
                arguments={"values": [3]})

        def gate(invocation):
            gate_calls.append(invocation)
            return tool_runtime.PreHookDecision()

        pipeline.add_pre("bad.transform", transform)
        pipeline.add_gate("must.not.run", gate)

        result = asyncio.run(pipeline.prepare(
            self.invocation({"values": ["a"]})))

        self.assertEqual(gate_calls, [])
        self.assertEqual(
            [(issue.path, issue.code)
             for issue in result.validation_issues],
            [(("values", 0), "type")],
        )

    def test_post_hook_failure_preserves_executed_outcome(self):
        pipeline = tool_runtime.ToolHookPipeline()

        def broken(invocation, outcome):
            raise RuntimeError("post broke")

        pipeline.add_post("broken.post", broken)
        invocation = self.invocation({"values": []})
        outcome = tool_runtime.ToolOutcome(
            "success", True, True, "actual result")

        invocation, outcome = asyncio.run(
            pipeline.finish(invocation, outcome))

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.content, "actual result")
        self.assertIn("already executed", invocation.notes[0])


class ExternalHookTests(unittest.TestCase):
    def test_invalid_hook_configuration_exits_before_terminal_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "hooks.json")
            with open(config_path, "w", encoding="utf-8") as stream:
                stream.write("{invalid")
            env = child_environment(
                HOME=directory,
                PATH=os.environ.get("PATH", ""),
                TERM="dumb",
                LOKI_HOOKS=config_path,
                XDG_CONFIG_HOME=os.path.join(directory, "config"),
                XDG_STATE_HOME=os.path.join(directory, "state"),
            )

            workspace = os.path.join(directory, "workspace")
            os.makedirs(workspace)
            configure_container(env, workspace)
            result = subprocess.run(
                [entrypoint("loki"), "--headless"],
                cwd=workspace,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Hook configuration error:", result.stderr)
        self.assertNotIn("\x1b", result.stdout + result.stderr)

    def test_hook_environment_carries_scratch_but_not_credentials(self):
        source = dict(os.environ)
        source.update({
            "LOKI_API_KEY": "secret",
            "UNRELATED_SECRET": "secret",
            "TEMP": "/workspace/.loki/tmp",
            "TMP": "/workspace/.loki/tmp",
        })
        environment = tool_runtime._hook_environment(source)
        # Windows hook children have no profile TEMP to fall back on.
        self.assertEqual(environment["TEMP"], "/workspace/.loki/tmp")
        self.assertEqual(environment["TMP"], "/workspace/.loki/tmp")
        self.assertNotIn("LOKI_API_KEY", environment)
        self.assertNotIn("UNRELATED_SECRET", environment)

    def test_external_hook_is_bounded_to_minimal_environment(self):
        script = (
            "import json, os, sys; "
            "p=json.load(sys.stdin); "
            "assert 'LOKI_API_KEY' not in os.environ; "
            "assert 'UNRELATED_SECRET' not in os.environ; "
            "a=p['invocation']['effective_arguments']; "
            "a['content']='from hook'; "
            "json.dump({'arguments':a}, sys.stdout)"
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "hooks.json")
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "pre_tool_call": [{
                        "id": "external.transform",
                        "tools": ["Example"],
                        "command": [sys.executable, "-c", script],
                    }],
                }, stream)
            source_environ = dict(os.environ)
            source_environ.update({
                "LOKI_API_KEY": "secret",
                "UNRELATED_SECRET": "secret",
            })
            with mock.patch.object(
                    tool_runtime,
                    "_hook_environment",
                    return_value=tool_runtime._hook_environment(
                        source_environ)):
                pipeline = tool_runtime.load_hook_pipeline(config_path)
                invocation = tool_runtime.ToolInvocation(
                    call_id="call_external",
                    tool_name="Example",
                    raw_arguments={"values": []},
                    original_arguments={"values": []},
                    effective_arguments={"values": []},
                    schema=ARRAY_SCHEMA,
                    cwd=directory,
                )
                result = asyncio.run(pipeline.prepare(invocation))

        self.assertEqual(
            result.effective_arguments,
            {"values": [], "content": "from hook"},
        )
        self.assertEqual(
            result.adjustments[-1].hook,
            "external.transform",
        )

    def test_hook_command_holds_only_its_standard_descriptors(self):
        """A hook child holds nothing beyond its own standard descriptors.

        The parent holds an *inheritable* descriptor, as the runtime does: its
        own owner and capability descriptors reached it that way.  Without one,
        nothing could cross and the assertion could not fail, because Python's
        descriptors are non-inheritable by default.  The child enumerates the
        descriptors *it* holds, so the claim is about its own table rather than
        a query against a number the parent owns.
        """
        runtime_read, runtime_write = os.pipe()
        os.set_inheritable(runtime_read, True)
        script = (
            "import json, os, sys\n"
            "held = []\n"
            "for fd in range(0, 4096):\n"
            "    try:\n"
            "        os.fstat(fd)\n"
            "    except OSError:\n"
            "        continue\n"
            "    held.append(fd)\n"
            "json.dump({'held': held}, sys.stdout)\n"
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = asyncio.run(tool_runtime._run_hook_command(
                    [sys.executable, "-c", script],
                    {},
                    directory,
                    1000,
                ))
        finally:
            os.close(runtime_read)
            os.close(runtime_write)
        self.assertEqual(result, {"held": [0, 1, 2]})

    def test_external_post_hook_can_only_append_note(self):
        script = (
            "import json, sys; "
            "p=json.load(sys.stdin); "
            "assert p['outcome']['content']=='real result'; "
            "json.dump({'note':'post note'}, sys.stdout)"
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "hooks.json")
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "post_tool_call": [{
                        "id": "external.post",
                        "tools": ["Example"],
                        "command": [sys.executable, "-c", script],
                    }],
                }, stream)
            pipeline = tool_runtime.load_hook_pipeline(config_path)
            invocation = tool_runtime.ToolInvocation(
                call_id="call_external",
                tool_name="Example",
                raw_arguments={"values": []},
                original_arguments={"values": []},
                effective_arguments={"values": []},
                schema=ARRAY_SCHEMA,
                cwd=directory,
            )
            outcome = tool_runtime.ToolOutcome(
                "success", True, True, "real result")
            invocation, outcome = asyncio.run(
                pipeline.finish(invocation, outcome))

        self.assertEqual(outcome.content, "real result")
        self.assertEqual(invocation.notes, ["post note"])

    def test_pre_hook_timeout_denies_without_tool_execution(self):
        script = "import time; time.sleep(1)"
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "hooks.json")
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "pre_tool_call": [{
                        "id": "external.timeout",
                        "command": [sys.executable, "-c", script],
                        "timeout_ms": 10,
                    }],
                }, stream)
            pipeline = tool_runtime.load_hook_pipeline(config_path)
            invocation = tool_runtime.ToolInvocation(
                call_id="call_timeout",
                tool_name="Example",
                raw_arguments={"values": []},
                original_arguments={"values": []},
                effective_arguments={"values": []},
                schema=ARRAY_SCHEMA,
                cwd=directory,
            )
            result = asyncio.run(pipeline.prepare(invocation))

        self.assertEqual(result.denied_by, "external.timeout")
        self.assertIn("timed out", result.denied_reason)

    def test_hook_timeout_includes_blocked_stdin_transfer(self):
        script = "import time; time.sleep(10)"

        async def scenario(directory):
            started = asyncio.get_running_loop().time()
            with self.assertRaisesRegex(
                    tool_runtime.HookExecutionError,
                    "hook timed out after 20ms"):
                await asyncio.wait_for(
                    tool_runtime._run_hook_command(
                        [sys.executable, "-c", script],
                        {"large": "x" * 2_000_000},
                        directory,
                        20,
                    ),
                    timeout=1,
                )
            return asyncio.get_running_loop().time() - started

        with tempfile.TemporaryDirectory() as directory:
            elapsed = asyncio.run(scenario(directory))

        self.assertLess(elapsed, 0.5)

    def test_invalid_hook_config_is_rejected_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "hooks.json")
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "pre_tool_call": [{
                        "id": "bad",
                        "command": "shell string is forbidden",
                    }],
                }, stream)

            with self.assertRaisesRegex(
                    tool_runtime.HookConfigurationError,
                    "argv list"):
                tool_runtime.load_hook_pipeline(config_path)

            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "post_tool_call": [{
                        "id": "bad.post",
                        "command": [sys.executable, "-c", "print('{}')"],
                        "on_error": "deny",
                    }],
                }, stream)
            with self.assertRaisesRegex(
                    tool_runtime.HookConfigurationError,
                    "cannot be 'deny'"):
                tool_runtime.load_hook_pipeline(config_path)


class LokiToolRuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        loki.file_state.clear()

    def execute_with_fake_dispatch(self, call, pipeline=None):
        dispatched = []
        events = []

        async def dispatch(name, args, allowed=None, extra_context=None):
            dispatched.append((name, copy.deepcopy(args)))
            return {"ok": True, "content": "real result"}

        with mock.patch.object(
                loki, "dispatch_tool_async", new=dispatch):
            result, execution = asyncio.run(
                loki.execute_tool_call_async(
                    call,
                    on_event=events.append,
                    hook_pipeline=(
                        pipeline
                        if pipeline is not None
                        else tool_runtime.ToolHookPipeline()),
                ))
        return result, execution, dispatched, events

    def test_custom_tool_executes_with_opaque_string_input(self):
        received = []

        def execute(input_text):
            received.append(input_text)
            return "custom result"

        registry = loki._build_tool_registry(
            [{
                "type": "custom",
                "name": "exec",
                "description": "Execute freeform input.",
            }],
            {"exec": {"handler": execute}},
        )
        call = formats.openai_responses_response_to_items({
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "exec",
                "input": "echo custom",
            }],
        }).items[0]

        with mock.patch.object(loki, "TOOL_REGISTRY", registry):
            result, _execution = asyncio.run(
                loki.execute_tool_call_async(
                    call,
                    hook_pipeline=tool_runtime.ToolHookPipeline(),
                ))

        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "custom result")
        self.assertEqual(received, ["echo custom"])

    def test_custom_call_cannot_enter_function_handler(self):
        call = formats.tool_call_item(
            "call_1",
            "Bash",
            raw_arguments="id",
            tool_kind="custom",
        )

        result, execution = asyncio.run(
            loki.execute_tool_call_async(
                call,
                hook_pipeline=tool_runtime.ToolHookPipeline(),
            ))

        self.assertFalse(result["ok"])
        self.assertIsNone(execution)
        self.assertIn("registered as function", result["content"])

    def test_bash_uses_the_cwd_shown_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = loki.current_cwd()
            pipeline = tool_runtime.ToolHookPipeline()

            def change_session_cwd(invocation):
                loki.current_session().shell_cwd = directory

            pipeline.add_pre("change-cwd", change_session_cwd)
            events = []
            call = formats.tool_call_item(
                "call_cwd", "Bash", {"command": "pwd"})
            try:
                with mock.patch.object(loki, "run_bash_async",
                                       new_callable=mock.AsyncMock,
                                       return_value="done") as run_bash:
                    asyncio.run(loki.execute_tool_call_async(
                        call, hook_pipeline=pipeline, on_event=events.append))
                shown_cwd = next(event["cwd"] for event in events
                                 if event["type"] == "tool_call")
                self.assertEqual(shown_cwd, initial)
                self.assertEqual(run_bash.await_args.kwargs["cwd"], shown_cwd)
            finally:
                loki.current_session().shell_cwd = initial

    def test_read_relational_defaults_are_transparent_not_errors(self):
        call = formats.tool_call_item(
            "call_read", "Read",
            {"file_path": "README.md", "offset": 3})

        result, execution, dispatched, _ = (
            self.execute_with_fake_dispatch(call))

        self.assertEqual(
            dispatched,
            [("Read", {"file_path": "README.md", "offset": 3})],
        )
        self.assertTrue(result["ok"])
        self.assertIn(
            "limit was omitted; Read used the default of 2000 lines",
            result["content"],
        )
        self.assertEqual(
            execution["defaults"],
            [{"field": "limit", "value": 2000}],
        )

    def test_post_hook_failure_does_not_reexecute_handler(self):
        pipeline = tool_runtime.ToolHookPipeline()

        def broken(invocation, outcome):
            raise RuntimeError("broken post")

        pipeline.add_post("broken.post", broken)
        call = formats.tool_call_item(
            "call_once", "TodoRead", {})

        result, execution, dispatched, _ = (
            self.execute_with_fake_dispatch(call, pipeline))

        self.assertEqual(dispatched, [("TodoRead", {})])
        self.assertTrue(result["ok"])
        self.assertIn("already executed", result["content"])
        self.assertEqual(
            execution["hooks"][0]["status"], "error")

    def test_pre_hook_denial_skips_dispatch_and_reaches_post_hook(self):
        pipeline = tool_runtime.ToolHookPipeline()
        observed = []

        def deny(invocation):
            return tool_runtime.PreHookDecision(
                action="deny",
                message="Denied by test policy.",
            )

        def observe(invocation, outcome):
            observed.append((
                outcome.status, outcome.executed, outcome.content))
            return tool_runtime.PostHookDecision()

        pipeline.add_pre("deny.test", deny)
        pipeline.add_post("observe.test", observe)
        call = formats.tool_call_item(
            "call_denied", "TodoRead", {})

        result, execution, dispatched, _ = (
            self.execute_with_fake_dispatch(call, pipeline))

        self.assertEqual(dispatched, [])
        self.assertFalse(result["ok"])
        self.assertEqual(result["content"], "Denied by test policy.")
        self.assertEqual(
            observed,
            [("rejected", False, "Denied by test policy.")],
        )
        self.assertEqual(execution["denied_by"], "deny.test")

    def test_custom_pre_hook_can_replace_malformed_raw_json(self):
        pipeline = tool_runtime.ToolHookPipeline()

        def repair_raw(invocation):
            self.assertEqual(
                invocation.raw_arguments, '{"query":')
            self.assertIsNotNone(invocation.parse_error)
            return tool_runtime.PreHookDecision(arguments={
                "query": "loki",
            })

        pipeline.add_pre("repair.raw", repair_raw)
        call = formats.tool_call_item(
            "call_raw",
            "WebSearch",
            raw_arguments='{"query":',
            parse_error="incomplete JSON",
        )

        result, execution, dispatched, _ = (
            self.execute_with_fake_dispatch(call, pipeline))

        self.assertTrue(result["ok"])
        self.assertEqual(
            dispatched, [("WebSearch", {"query": "loki"})])
        self.assertEqual(call["arguments"], '{"query":')
        self.assertEqual(
            execution["adjustments"][0]["hook"], "repair.raw")

    def test_invalid_input_lists_all_issues_and_never_dispatches(self):
        call = formats.tool_call_item(
            "call_invalid",
            "WebSearch",
            {
                "query": "x",
                "allowed_domains": [3],
                "blocked_domains": [4],
            },
        )

        result, _, dispatched, events = (
            self.execute_with_fake_dispatch(call))

        self.assertEqual(dispatched, [])
        self.assertFalse(result["ok"])
        self.assertIn("$.query", result["content"])
        self.assertIn("$.allowed_domains[0]", result["content"])
        self.assertIn("$.blocked_domains[0]", result["content"])
        self.assertEqual(
            [event["type"] for event in events],
            ["tool_input_invalid", "tool_rejected"],
        )

    def test_malformed_execution_metadata_is_rejected(self):
        call = formats.tool_call_item('call_saved', 'WebSearch', {
            'query': 'loki', 'allowed_domains': 'example.com'})
        blob = formats.new_log_blob([
            formats.message_item('user', 'search'),
            formats.model_response_event(formats.OPENAI_CHAT, [call]),
            formats.tool_result_for_call(call, 'parser fixture result'),
        ], [])
        blob['events'][2]['execution'] = []
        with self.assertRaisesRegex(
                formats.TranscriptFormatError,
                'execution metadata must be an object'):
            formats.load_log_blob(blob)


class TurnEndHookTests(unittest.TestCase):
    def test_no_hooks_do_not_start_subprocess_or_build_payload(self):
        with mock.patch.object(loki, "TOOL_HOOK_PIPELINE",
                               tool_runtime.ToolHookPipeline()):
            with mock.patch.object(tool_runtime.asyncio,
                                   "create_subprocess_exec") as spawn:
                asyncio.run(loki.run_turn_end_hooks_async([], "hello"))
                spawn.assert_not_called()

    def test_turn_end_command_has_no_default_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "hooks.json")
            output = os.path.join(directory, "payload.json")
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump({"turn_end": [{
                    "id": "record",
                    "command": [sys.executable, "-c",
                                "import json,os,sys,time; time.sleep(.05); "
                                "payload=json.load(sys.stdin); "
                                "payload['child_cwd']=os.getcwd(); "
                                "json.dump(payload, open(sys.argv[1], 'w')); "
                                "print('{}')", output],
                }]}, stream)
            pipeline = tool_runtime.load_hook_pipeline(config_path)
            self.assertIsNone(pipeline.turn_end_hooks[0].timeout_ms)
            with mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline):
                with mock.patch.object(loki, "current_cwd", return_value=directory):
                    asyncio.run(loki.run_turn_end_hooks_async(
                        [{"type": "max_loops"}], "last text"))
            with open(output, encoding="utf-8") as stream:
                payload = json.load(stream)
            self.assertEqual(payload["cwd"], directory)
            self.assertEqual(payload["child_cwd"], directory)
            self.assertEqual(payload["event"], "turn_end")
            self.assertEqual(payload["reason"], "max_loops")
            self.assertEqual(payload["text"], "last text")

    def test_turn_end_timeout_and_failure_do_not_change_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "hooks.json")
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump({"turn_end": [{
                    "id": "slow", "command": [
                        sys.executable, "-c", "import time; time.sleep(1)"],
                    "timeout_ms": 20,
                }]}, stream)
            pipeline = tool_runtime.load_hook_pipeline(config_path)
            with mock.patch.object(loki, "TOOL_HOOK_PIPELINE", pipeline):
                with mock.patch("sys.stderr") as stderr:
                    asyncio.run(loki.run_turn_end_hooks_async(
                        [{"type": "response_cancelled"}]))
                self.assertTrue(stderr.write.called)

    def test_turn_end_rejects_tool_options_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "hooks.json")
            for value in (
                    {"turn_end": [{"id": "x", "command": ["true"],
                                   "tools": ["Bash"]}]},
                    {"pre_tool_call": [{"id": "x", "command": ["true"]}],
                     "turn_end": [{"id": "x", "command": ["true"]}]}):
                with open(config_path, "w", encoding="utf-8") as stream:
                    json.dump(value, stream)
                with self.assertRaises(tool_runtime.HookConfigurationError):
                    tool_runtime.load_hook_pipeline(config_path)


if __name__ == "__main__":
    unittest.main()
