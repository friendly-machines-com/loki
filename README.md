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
./loki.py
```

At startup Loki captures the environment and removes variables ending in
`_KEY`, `_TOKEN`, or `_PAT` from the environment inherited by tools. The
`/model` picker fetches models.dev lazily and shows only providers for which a
captured credential is available. Provider-specific variables such as
`OPENROUTER_API_KEY` can therefore be supplied by the VM/container launcher
without also exposing them to ordinary tool subprocesses.

Loki has no built-in provider connection. Without an explicit
`LOKI_API_BASE` or a saved session connection, it starts disconnected and
`/model` can be used to choose among providers represented by captured
credentials. An explicitly configured endpoint accepts only `LOKI_API_KEY`;
Loki never substitutes another provider's credential based merely on the
endpoint's wire protocol.

Chat logs persist the selected model, protocol, and concrete endpoints, but
never the credential value. Resuming that connection requires the same
credential variable to be supplied again and asks for confirmation before
sending it to the saved endpoints.

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
