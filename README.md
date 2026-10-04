# Agent

Really really minimal-dependency coding agent.

You need to use your own VM (do) or container (don't) to isolate it.

It doesn't ask you every 2 seconds whether it is allowed to do some command.

Runs on ECMA-48 console (tested with "foot" terminal on Linux) and via many editors (emacs, zed, ...) via ACP.

Supports Anthropic and OpenAI protocols.

## Features

* Glob
* Grep
* (ephemeral) Bash
* File editing
* Subagent
* History (stored on disk, in cwd)
* Web Search
* Web Fetch
* Background jobs
* Task planning
* Skills

## How to run

On POSIX, requires Python 3.11.10 or later (on the 3.12 branch, 3.12.4 or
later). On Windows, requires Python 3.12.4 or later.

Run it in a VM or container.

On Windows there is no VM requirement: ``loki-setup`` configures an
AppContainer for a workspace. ``TEMP`` and ``TMP`` point at
``<workspace>\.loki\tmp``. Loki creates this directory when needed but does not
clean it up automatically; you can remove its contents between runs.

```
export LOKI_API_KEY=xxx
export LOKI_API_BASE="https://opencode.ai/zen/go/v1/chat/completions"
export LOKI_MODEL="glm-5.2"
./loki.py
```

To use an OpenAI ChatGPT subscription instead of Platform API billing, log in
before starting a Loki session:

```
./loki.py auth login openai
```

Login prints a URL and attempts to open it in a browser. It needs localhost
port 1455 or 1457 to receive the login confirmation. On a headless or remote
machine, use device-code login:

```
./loki.py auth login openai --device-code
```

Device-code login must be enabled for the ChatGPT account or workspace.
`./loki.py auth status openai` reports the stored login without displaying
tokens, and `./loki.py auth logout openai` removes it locally. Logout does not
immediately revoke access in already-running sessions; close those sessions
as well.

Loki stores subscription tokens in
`$XDG_CONFIG_HOME/loki/credentials/tokens.json` (normally
`~/.config/loki/credentials/tokens.json`).

No desktop credential service or message bus is required.

Use `/image PATH` to attach a local PNG, JPEG, GIF, or WebP image to
the next prompt. Relative paths use Loki's current `/pwd`; quote paths
containing spaces. Several `/image` commands may be used before the prompt,
and an empty prompt sends the staged images without additional text. Each
image is limited to 20 MiB and is snapshotted when `/image` is entered, so a
later change to the file does not change the conversation. The selected
provider and model must support image input.

Loki reads credentials at startup and removes variables ending in `_KEY`,
`_TOKEN`, or `_PAT` from the environment inherited by tools and hooks. On Linux,
Loki also hides `$XDG_CONFIG_HOME/loki/credentials` from tools and subagents.
These protections do not replace the VM or container needed to isolate tool
activity. An interrupted credential refresh may require logging in again.

## Models and connections

The `/model` picker uses models.dev and shows providers for which credentials
were supplied at startup, such as `OPENROUTER_API_KEY`. Deprecated models remain
selectable but are labeled in the picker and status bar.

`OpenAI Platform API` uses API billing. `OpenAI ChatGPT subscription` uses your
saved ChatGPT login and lists the models available to that account; these may
differ from the Platform API models. Subscription credentials are not sent to
models.dev or the Platform API.

Loki does not select a built-in provider connection at startup. Without an
explicit `LOKI_API_BASE` or a saved session connection, it starts disconnected
and `/model` can be used to choose among providers represented by captured
credentials. An explicitly configured endpoint uses `LOKI_API_KEY` when it is
set; when it is absent, Loki sends no authentication header. Loki never
substitutes another provider's credential based merely on the endpoint's wire
protocol.

All explicit connection settings are Loki-namespaced: `LOKI_API_BASE`,
`LOKI_PROVIDER`, `LOKI_MODEL`, `LOKI_API_KEY`, `LOKI_MODELS_URL`,
`LOKI_MAX_TOKENS`, `LOKI_CONTEXT_WINDOW`, `LOKI_AUTH_HEADER`, `LOKI_AUTH_SCHEME`,
`LOKI_ANTHROPIC_VERSION`, and `LOKI_STREAM`. `LOKI_PROMPT_CACHE` controls
Anthropic Messages prompt-cache metadata.
Loki never chooses the first model returned by a provider. A new explicit
connection needs `LOKI_MODEL`, or the model must be selected with `/model`
before sending a chat request. A complete captured `LOKI_*` connection also
appears in `/model` as `Explicit LOKI_* connection`, so it can be selected
again after switching to a catalog provider or while models.dev is
unavailable.

Use `/thinking` to see and change the current model's thinking settings:

- `/thinking effort high` chooses an available effort level.
- `/thinking mode off` turns thinking off when the model allows it.
- `/thinking mode adaptive` lets a model that supports it manage its thinking.
- `/thinking mode manual budget 2048` enables manual thinking with an explicit
  token allowance. The output limit must be larger than the allowance.
- `/thinking retention preserve` asks a supported provider to reuse earlier
  thinking; this may increase input tokens and cost.

Use `default` instead of a value to clear a setting. Preferences are remembered
with the conversation. Settings that do not apply to a different model stay
inactive, and switching back restores them. ACP clients offer the same settings;
use the commands for numerical allowances or changes that need to happen together.
Changes made during a response apply to the next turn.

Use `/trace thinking on` to show thinking or summaries returned by the provider,
and `/trace thinking off` to hide them. These commands do not turn thinking on
or change its settings. Hidden thinking and continuation data remain in history.
Some models do not return readable thinking; OpenAI returns summaries, not its
complete internal reasoning.

Set `LOKI_STREAM=1` to display assistant text as it arrives. Streaming is
disabled by default. If the server rejects it, set `LOKI_STREAM=0`.

The Remote row's `Context: 37%` shows the last reported context usage, not an
exact count of the next prompt. `*` marks an older or incomplete measurement;
`unknown` means usage or capacity is unavailable. To override the model's
context capacity, set `LOKI_CONTEXT_WINDOW` to a positive token count. This
changes the display only, not the model's limits.

## User settings

Terminal preferences are read from the optional `~/.config/loki/settings.ini`
(or `$XDG_CONFIG_HOME/loki/settings.ini`). No file is needed: every preference
has a built-in default, and startup never creates or rewrites the file.

Bash stdout is hidden from automatic terminal tool-result display by default.
To show it, add:

```ini
[terminal]
show_bash_stdout = true
```

Restart the terminal session after editing the file. Stderr, status, and tool
notes remain visible. This also applies to automatic shell-job status results,
but not subagent output or explicit inspection with `!command` or `/ps ID`.
Stdout is still captured, saved, and supplied to the model; this is a display
preference, not redaction, and it does not change ACP output.

New saved results retain stream identity for terminal replay. Older combined
Bash results cannot be reliably split and show a notice when stdout is hidden;
enable the setting to display those older results in full.

Application code reads and updates preferences through `loki_agent.settings`,
not through frontend file parsing. Explicit updates preserve unrelated INI
values but rewrite formatting and do not retain comments.

## Tool hooks

Tool-call output shows the session shell cwd separately from the model-supplied
arguments; Bash runs in that directory, and relative file paths are resolved
against it. The ACP tool-call title also shows the cwd. Background Jobs and
JobStatus report the cwd captured at launch.

In the terminal, `/ps` lists running, starting, and failed jobs;
`/ps all` includes finished jobs. Lists are ordered oldest first.
Use `/ps ID` to see a job's status and recent stdout/stderr, `/ps stop ID`
to request a graceful stop, or `/ps kill ID` to force termination.
These commands execute immediately, without queueing behind a running turn,
and are also available during terminal pickers and confirmations. `/ps` is
not an ACP command.

The same immediate delivery covers `/status` (with `--json`, `all`, and
`save` forms), the read-only `/account CONTROL` form, and `/queue`: they
answer while a turn is running and their output never becomes conversation.
Delayed answers display immediately on completion as separate, labelled
blocks, without waiting for the running turn. `/queue` inspects and edits
what is waiting: bare `/queue` lists its
subcommands, `/queue texts` lists queued prompts in send order, and
`/queue images` lists images staged for the next prompt. Each entry has a
stable ID, not a position: IDs survive edits and moves and are not reused
when entries are consumed or deleted. Identical submissions have different
IDs. Use `/queue texts delete ID`, `/queue texts edit ID TEXT`,
`/queue texts move ID before OTHER_ID`, or `/queue texts move ID end`;
images support the same delete and move forms. Both IDs in a move must
still be pending. A stale ID is rejected without changing another entry.
Every successful edit re-lists the entries. A
staged image cannot be re-pointed; delete it and stage another with
`/image PATH`. Listing and editing never touch a running turn. `/account`
without a control id, and `/account CONTROL ACTION`, stay queued behind a
running turn because they interact. Immediate commands
are declared in one place (`loki_agent/command_deliveries.py`); on ACP every
command is an ordinary prompt -- the worker rejects it while another prompt
runs -- until an out-of-band output channel exists there.

Loki checks tool input before execution, corrects some unambiguous formatting
mistakes, and reports any corrections. Other invalid calls are rejected.

Loki also supports trusted external tool hooks. Set `LOKI_HOOKS` to an explicit
JSON configuration path, or place user-owned configuration at
`~/.config/loki/hooks.json`. Repository hook files are never loaded
automatically. Set `LOKI_HOOKS=off` to disable external hooks while retaining
Loki's built-in input repair.

```json
{
    "pre_tool_call": [
        {
            "id": "normalize",
            "tools": ["Write", "Edit"],
            "command": ["/home/me/bin/loki-normalize"],
            "timeout_ms": 2000,
            "on_error": "deny"
        }
    ],
    "pre_tool_gate": [
        {
            "id": "policy",
            "tools": ["Bash"],
            "command": ["/home/me/bin/loki-policy"]
        }
    ],
    "post_tool_call": [
        {
            "id": "format",
            "tools": ["Write", "Edit"],
            "command": ["/home/me/bin/loki-format"],
            "timeout_ms": 10000,
            "on_error": "continue",
            "workspace_side_effects": true
        }
    ]
}
```

Hooks run sequentially in configuration order and receive one JSON object on
stdin. Pre-hook input has `event: "pre_tool_call"` and an `invocation` object
containing the call ID, tool name, original and effective arguments, schema,
current validation issues, cwd, model, provider, and prior adjustments. A
pre-transformer writes one of:

```json
{"action": "continue"}
{"action": "continue", "arguments": {"replacement": "input"}, "note": "why"}
{"action": "deny", "message": "model-readable reason"}
```

A `pre_tool_gate` has the same input and decisions but cannot replace
arguments. It sees the final validated input after all transformers.
Post-hook input additionally contains the terminal `outcome`, including
whether the tool executed and its real result. A post-hook may return a
model-visible note and workspace paths it changed:

```json
{"note": "formatter ran", "changed_paths": ["src/example.py"]}
```

Post-hooks cannot replace the real tool result or cause automatic
re-execution. A pre-hook error denies execution by default. A post-hook error
preserves the outcome and tells the model that the tool had already executed.

For user-facing terminal and ACP turns, an optional `turn_end` section runs
once after the entire model/tool loop (including cancellation or failure), not
once per model response. It does not run for Explore subagent turns or local
commands. For example:

```json
{"turn_end": [{"id": "notify", "command": ["/home/me/bin/loki-notify"]}]}
```

Each command receives `{"event":"turn_end","reason":"completed",
"cwd":"...","text":"..."}` on stdin. `cwd` is the session shell cwd
at turn end and is also the hook subprocess's working directory. `reason` may
also be `cancelled`, `error`, or `max_loops`; `text` is the last returned
assistant text, if any.
Hooks run in order; their JSON stdout is validated but cannot change the
completed turn. Failures are reported to stderr and do not change the result.
There is **no default timeout** for turn-end hooks; set a positive `timeout_ms`
on an entry if a limit is desired. With no `turn_end` entries, no hook process
is started. Turn-end entries do not accept `tools`, `on_error`, or
`workspace_side_effects`.
Commands are argv arrays, not shell strings; stdout is reserved for the single
JSON response, while stderr remains diagnostic. Hook subprocesses receive a
minimal environment without Loki API credentials. A hook configured with
`workspace_side_effects: true` must return `changed_paths` listing the files it
changed.

## Saved sessions

Chat logs save the conversation, selected model, connection settings, and other
session state, but not credential values. Use `./loki.py --resume` to choose a
saved chat, or `./loki.py --resume LOG` to open a specific log. Older savefile
formats are not migrated automatically.

Changing `/model` preserves the conversation, though provider-specific reasoning
data may not transfer to the new model. To resume an authenticated connection,
supply its credentials again. Loki asks for confirmation before using saved
endpoints. Missing credentials do not erase the saved connection.

`LOKI_*` settings initialize new sessions. On resume they temporarily override
the saved connection; selecting a model with `/model` updates the saved choice.

Direct Anthropic API connections enable prompt caching by default. For other
Anthropic-compatible servers, enable it with `LOKI_PROMPT_CACHE=1`. Set
`LOKI_PROMPT_CACHE=0` to disable it. Cache reuse depends on the provider and can
be affected by changes to the conversation or working directory.

## Diagnostic logging

By default, developer traces (unknown provider fields, response timing, raw error
payloads and tracebacks) are hidden. User-facing errors and hook stderr remain
visible. Enable Loki DEBUG logging to stderr with `LOKI_TRACE=1 ./loki.py` or
`LOKI_TRACE=1 ./loki-acp`.

For more control, set `LOKI_LOG_CONFIG=/absolute/path/to/logging.ini` to a
standard Python logging INI file. This takes precedence over `LOKI_TRACE`;
an unreadable or invalid file prevents startup. Use an absolute path accessible
to Loki and its subagents; Loki does not expand `~` in this setting.

The following standard INI configuration sends warnings to stderr and only
provider-format DEBUG diagnostics to a file. Change `loki_agent.formats` to
`loki_agent` to enable all Loki diagnostics, or name another module such as
`loki_agent.terminal_frontend` for terminal timing and error details.

```ini
[loggers]
keys=root,loki,formats

[handlers]
keys=console,trace

[formatters]
keys=diagnostic

[logger_root]
level=WARNING
handlers=console

[logger_loki]
qualname=loki_agent
level=WARNING
handlers=console,trace
propagate=0

[logger_formats]
qualname=loki_agent.formats
level=DEBUG
handlers=
propagate=1

[handler_console]
class=StreamHandler
level=WARNING
formatter=diagnostic
args=(sys.stderr,)

[handler_trace]
class=FileHandler
level=DEBUG
formatter=diagnostic
# Choose an absolute, private path; each process gets its own file.
args=('/tmp/loki-trace-' + str(__import__('os').getpid()) + '.log', 'a', 'utf-8')

[formatter_diagnostic]
format=%(asctime)s %(levelname)s %(name)s[%(process)d]: %(message)s
```

In agent-shell, stderr appears in Notices even when its ACP logging toggle is
off. Use a file destination to avoid verbose notices. Never send ACP logs to
stdout, which is reserved for communication with the editor. Subagent stderr
is available through its job output.

Logging configuration can execute code. Use only a configuration you trust,
not one supplied by a model or remote client. Log paths are relative to the
process's working directory, not the INI file; prefer absolute paths in private
directories. Traces may contain conversation content and provider secrets, so
review them before sharing. Give each process its own log file, as in the
example, rather than sharing a rotating log. Logger levels do not hide ordinary
UI messages, hook output, or user-facing errors.

## HTTP response status

`/status` shows the last observed HTTP response headers for the current
connection. Use `/status all` for all known connections, or add `--json` for
JSON output. These commands use saved and locally observed data, not a live
provider lookup.

Outside a session, use `./loki.py status` or `./loki.py status --json` to inspect
all saved connections. Filter by endpoint with
`./loki.py status --endpoint https://example.com/v1/chat/completions`.

For ChatGPT subscriptions, status also shows last-reported quota usage and
reset times when available. `window: 7 days` is the allowance's duration, not
time remaining. `[retained observation]` marks older data. Subscription quotas
are separate from the conversation's context usage.

Observations are saved on normal exit; use `/status save` to save them sooner.
Other running sessions' unsaved data is not visible, and a crash can lose
unsaved observations. Replacing a credential with another account can leave
old status data visible; use `/account` for a live lookup.

Saved status is at `$XDG_STATE_HOME/loki/response-headers.json` (normally
`~/.local/state/loki/response-headers.json`; on Windows,
`%LOCALAPPDATA%\loki\response-headers.json`). Known secret headers are redacted,
but review the file before sharing it: providers may send other sensitive
information. If the file is invalid, Loki reports an error; remove it manually
to start over. This is diagnostic data, not a billing record.

## Account controls

`/account` lists the controls supported by the active connection. Selecting a
control makes a live provider request:

* OpenAI ChatGPT subscription: `usage` shows usage windows and available resets;
  `resets` lists banked limit-reset credits and lets you redeem them.
* OpenRouter: `balance` shows the key's spend limit and prepaid credit.
* DeepSeek: `balance` shows prepaid account balance.

Redeeming a reset is irreversible and never automatic. Account results are not
saved to the response-header file.

## Running tests

Local development and CI use the same root-level driver:

```sh
python3 run_tests.py
python3 run_tests.py -p 'test_acp*' -j 4 -v -k prompt
```

Files run concurrently in separate Python processes. `-p` selects filenames,
`-j` limits parallel workers, and unittest options such as `-v`, `-q`, `-b`,
`-f`, and `-k` are forwarded to each selected file. `-s` selects another test
directory. Live output is tagged with its test file, including partial
progress and stack diagnostics; any failing file makes the driver exit
nonzero.

Execution and diagnostic timers are **unlimited/off by default**.
`LOKI_SUITE_STALL_SECONDS=60` explicitly enables repeating worker stack dumps;
it does not kill tests or change their verdicts. For a deliberately bounded
debug run, `--timeout SECONDS` enables a per-file execution limit. CI's job
budget and diagnostic interval are configured in `.github/workflows/python-app.yml`,
not hardcoded as test-driver defaults.
