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

At startup Loki captures the environment and removes variables ending in
`_KEY`, `_TOKEN`, or `_PAT` from the environment inherited by tools. The
`/model` picker fetches models.dev lazily and shows only providers for which a
captured credential is available. Provider-specific variables such as
`OPENROUTER_API_KEY` can therefore be supplied by the VM/container launcher
without also exposing them to ordinary tool subprocesses. Deprecated catalog
entries remain selectable but are labeled in the picker and status bar.

Loki has no built-in provider connection. Without an explicit
`LOKI_API_BASE` or a saved session connection, it starts disconnected and
`/model` can be used to choose among providers represented by captured
credentials. An explicitly configured endpoint uses `LOKI_API_KEY` when it is
set; when it is absent, Loki sends no authentication header. Loki never
substitutes another provider's credential based merely on the endpoint's wire
protocol.

All explicit connection settings are Loki-namespaced: `LOKI_API_BASE`,
`LOKI_PROVIDER`, `LOKI_MODEL`, `LOKI_API_KEY`, `LOKI_MODELS_URL`,
`LOKI_MAX_TOKENS`, `LOKI_AUTH_HEADER`, and `LOKI_ANTHROPIC_VERSION`.
Loki never chooses the first model returned by a provider. A new explicit
connection needs `LOKI_MODEL`, or the model must be selected with `/model`
before sending a chat request. A complete captured `LOKI_*` connection also
appears in `/model` as `Explicit LOKI_* connection`, so it can be selected
again after switching to a catalog provider or while models.dev is
unavailable.

Chat logs are session savefiles. They persist the selected model, its known
catalog status, protocol, concrete endpoints, and other session state, but
never credential values.
Resuming an authenticated connection requires the same credential variable to
be supplied again. Credentialless connections resume without inventing a
credential. Loki asks for confirmation before sending either kind of
connection to the saved endpoints. A temporarily unavailable credential does
not remove the saved connection.
`LOKI_*` config initializes new sessions; on resumed sessions it is a runtime
override and does not replace the saved connection. A successful `/model`
selection does replace the session's saved connection.

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
