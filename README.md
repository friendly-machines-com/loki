# Agent

Really really minimal-dependency coding agent.

You need to use your own VM (do) or container (don't) to isolate it.

It doesn't ask you every 2 seconds whether it is allowed to do some command.

Runs on ECMA-48 console (tested with "foot" terminal on Linux).

Supports Anthropic and OpenAI protocols.

## How to run

Run it in a VM or container.

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

The default authorization-code flow prints a URL, attempts to open it in a
browser, and listens only on localhost ports 1455 or 1457 for the PKCE
callback. On a headless or remote machine, use OpenAI's device-code flow:

```
./loki.py auth login openai --device-code
```

Device-code login must be enabled for the ChatGPT account or workspace.
`./loki.py auth status openai` reports the stored login without displaying
tokens, and `./loki.py auth logout openai` removes it locally. Logout prevents
new supervisors from loading the credential. A supervisor that was already
running can continue using its cached access token until it next needs a
refresh; neither that cached token nor a token already leased to a request can
be recalled locally, and both remain subject to server-side expiry.

Loki stores subscription tokens in
`$XDG_CONFIG_HOME/loki/credentials/tokens.json` (normally
`~/.config/loki/credentials/tokens.json`). The directory is private to the
user and token files are created with mode 0600. Writes use same-directory
atomic replacement and filesystem synchronization. A separate lock
serializes login, logout, and token rotation among independently running
terminal and ACP supervisors.

Use `/image PATH` to attach a local PNG, JPEG, GIF, or WebP image to
the next prompt. Relative paths use Loki's current `/pwd`; quote paths
containing spaces. Several `/image` commands may be used before the prompt,
and an empty prompt sends the staged images without additional text. Each
image is limited to 20 MiB and is snapshotted when `/image` is entered, so a
later change to the file does not change the conversation. The selected
provider and model must support image input.

At startup Loki captures the environment and removes variables ending in
`_KEY`, `_TOKEN`, or `_PAT` from the environment inherited by tools. The
Linux and macOS startup paths also overwrite each such variable's original
process-startup record with same-length `x` bytes while preserving its
terminating NUL and the framing of later records. This removes the credential
from Linux `/proc/PID/environ` and macOS `KERN_PROCARGS2` inspection. The
macOS path uses the documented `_NSGetEnviron()` interface and refuses to
write unless every target record is within the initial main-thread stack.
Every public entrypoint first becomes a credential-owning supervisor. For the
terminal and headless interfaces it starts a separate runtime through the same
executable; the ACP front starts one runtime worker per session. A runtime
receives only a sanitized environment, an owner-lifetime pipe, and an
anonymous socket capability restricted to the brokered credentials it may
request. Losing either supervisor channel cancels the runtime. Subagents
receive a fresh capability restricted to the current provider, and nested
subagents get a newly relayed capability rather than inheriting their parent's
descriptor. Rotating refresh tokens stay in the top-level broker; delegated
processes can lease an access token but cannot obtain the refresh token.

The supervisor/runtime split also gives persistent credentials a pathname
boundary. Every supervisor creates Loki's dedicated
`$XDG_CONFIG_HOME/loki/credentials` directory before starting a runtime. On
Linux, each runtime enters a private user and mount namespace before importing
the agent core and covers that directory with an empty read-only filesystem.
Tools and nested subagents inherit the covered view; only the supervisor
retains the original view needed to load and later update credentials.
Runtimes then discard their namespace capabilities and enable
`NO_NEW_PRIVS`, so they cannot remove the cover mount. This is not a general
sandbox: Loki still expects the surrounding VM or container described above
to confine arbitrary tool activity.

OpenAI refresh tokens may rotate. Before sending one, the supervisor
atomically changes its stored record to a refresh-in-progress tombstone that
does not contain the refresh token. A crash, cancellation after possible
delivery, or ambiguous HTTP result therefore requires a new login instead of
allowing another Loki process to replay the old token. Only a provably
pre-delivery transport failure restores the old active record.

Ordinary commands and hooks are started with other descriptors closed.
Credential-owning supervisors and credential-consuming runtimes also make
themselves non-dumpable on Linux, so same-UID tool children cannot inspect
their memory or open descriptors through ptrace-governed `/proc` interfaces.

The `/model` picker fetches models.dev lazily and shows only providers for
which a captured credential is available. Provider-specific variables such as
`OPENROUTER_API_KEY` can therefore be supplied by the VM/container launcher
without also exposing them to ordinary tool subprocesses. Deprecated catalog
entries remain selectable but are labeled in the picker and status bar.
Models.dev omits the OpenAI Platform endpoint because its native JavaScript
SDK supplies that default internally. When the catalog entry exactly matches
the canonical OpenAI provider signature, Loki supplies
`https://api.openai.com/v1` and labels the provider
`OpenAI Platform API [endpoint supplied by Loki]`. A changed signature or
non-OpenAI endpoint is rejected rather than receiving `OPENAI_API_KEY`.
For a stored ChatGPT login, Loki also constructs the separate provider
`OpenAI ChatGPT subscription [endpoint supplied by Loki]`, using the private
ChatGPT Codex Responses endpoint and the brokered subscription credential.
Its selectable models come only from that account's authenticated ChatGPT
Codex model catalog; Loki does not copy OpenAI Platform models from models.dev
into the subscription provider. Subscription authorization is permitted only
for the exact ChatGPT Codex Responses URL and the model-list request with
Loki's pinned, tested Codex compatibility level. The authenticated account's
picker-visible models remain authoritative; Loki does not silently remove
models based on their request metadata. Models advertising Responses-Lite
are sent with that protocol's header, all-turn reasoning context, developer
input items, and namespaced client-function tools; the selected framing is
stored with the connection so resume, ACP, and delegated subagents agree.
Subscription authorization is never attached to the models.dev catalog
request or to the OpenAI Platform API.

Loki does not select a built-in provider connection at startup. Without an
explicit `LOKI_API_BASE` or a saved session connection, it starts disconnected
and `/model` can be used to choose among providers represented by captured
credentials. An explicitly configured endpoint uses `LOKI_API_KEY` when it is
set; when it is absent, Loki sends no authentication header. Loki never
substitutes another provider's credential based merely on the endpoint's wire
protocol.

All explicit connection settings are Loki-namespaced: `LOKI_API_BASE`,
`LOKI_PROVIDER`, `LOKI_MODEL`, `LOKI_API_KEY`, `LOKI_MODELS_URL`,
`LOKI_MAX_TOKENS`, `LOKI_AUTH_HEADER`, `LOKI_AUTH_SCHEME`,
`LOKI_ANTHROPIC_VERSION`, and `LOKI_STREAM`. `LOKI_PROMPT_CACHE` controls
Anthropic Messages prompt-cache metadata.
Loki never chooses the first model returned by a provider. A new explicit
connection needs `LOKI_MODEL`, or the model must be selected with `/model`
before sending a chat request. A complete captured `LOKI_*` connection also
appears in `/model` as `Explicit LOKI_* connection`, so it can be selected
again after switching to a catalog provider or while models.dev is
unavailable.

For catalog models whose exact provider/model entry exposes a verified
reasoning-effort control, `/effort` selects `none`, `minimal`, `low`,
`medium`, `high`, `xhigh`, or `max` only when that entry advertises the
value. `Model default` is the initial setting and omits an effort override
for ordinary providers; ChatGPT subscription requests retain their
authenticated catalog default. An explicit selection is a conversation
preference: after switching to a model which does not support it, Loki uses
that model's default without coercion and restores the selection if a later
model supports it. The selected value is fixed for every provider request in
one logical tool turn.

ACP sessions expose the same selector as a `thought_level` session config
option. The complete option list is returned after model or effort changes,
so clients such as emacs-agent-shell can add, remove, or refresh their
reasoning selector as the model changes. An ACP effort change accepted while
a response is running applies to the next logical turn.

Streaming is disabled by default because compatible servers are not required
to implement it. Set `LOKI_STREAM=1` to request streaming for the selected
connection. Loki streams assistant text as it arrives but waits for a
protocol-confirmed final response before adding it to the session or executing
tool calls. Interrupted transport output is shown but is not invented as a
completed assistant response. If a server ignores the request and returns
ordinary JSON, Loki accepts that same response without resending the inference
request. If the server rejects streaming, set `LOKI_STREAM=0`.

Tool input is validated before execution. When validation identifies one of a
small set of unambiguous representation mistakes, Loki repairs a copy of the
input, validates the result again, and reports the adjustment to both the user
and model. The original provider tool call is never rewritten. Supported
repairs are optional `null` omission, JSON-encoded arrays, an empty-object
placeholder for an array, a bare string for a string array, and degenerate
Markdown auto-links in explicitly marked filesystem-path fields. Other invalid
input is rejected with a path-by-path retry message. A repaired call is
executed at most once.

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
Commands are argv arrays, not shell strings; stdout is reserved for the single
JSON response, while stderr remains diagnostic. Hook subprocesses receive a
minimal environment without Loki API credentials. A hook configured with
`workspace_side_effects: true` must return `changed_paths`, or Loki
conservatively invalidates all remembered file state.

Chat logs are session savefiles. They persist the selected model, its known
catalog status, protocol, concrete endpoints, and other session state, but
never credential values.
The v4 savefile is an editable ordered event stream. Direct messages, complete
model responses, and tool results are the only event types. Each model response
contains its originating protocol and ordered canonical output items; common
text, media, and function-call meaning is represented once, while native
continuation data such as Anthropic thinking signatures and OpenAI Responses
reasoning remains attached to that response. There are no positional call
ranges or external provenance records.

Every request is a pure projection of the complete current event stream into
the selected wire protocol. Changing `/model` does not convert or rewrite the
savefile: portable history is projected to the new protocol, native
continuation data is replayed only to the same provider endpoint and model,
and foreign continuation-only data remains stored but is omitted. Unknown
ordered provider output is printed on stderr, retained for its originating
connection, and omitted from foreign projections rather than synthesized as
dialogue. Each distinct tool schema snapshot is stored once and never inserted
into model-visible history. Older savefile schemas are intentionally not
migrated.
Resuming an authenticated connection requires the same credential variable to
be supplied again. Credentialless connections resume without inventing a
credential. Loki asks for confirmation before sending either kind of
connection to the saved endpoints. A temporarily unavailable credential does
not remove the saved connection.
`LOKI_*` config initializes new sessions; on resumed sessions it is a runtime
override and does not replace the saved connection. A successful `/model`
selection does replace the session's saved connection.

During ordinary chat/tool turns, prompt history is append-only and tool
definitions contain no changing date text. Direct Anthropic API connections
default to automatic ephemeral prompt caching. Other Anthropic-compatible
servers receive cache metadata only when `LOKI_PROMPT_CACHE=1`; set
`LOKI_PROMPT_CACHE=0` to disable it explicitly. OpenAI-compatible servers can
apply their own automatic prefix cache to the same stable tools, instructions,
and history. A later operator instruction, such as the context update produced
by `/cd`, can still invalidate the instruction-and-history portion of a cache.

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
