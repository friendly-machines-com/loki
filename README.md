# Agent

Really really minimal-dependency coding agent.

You need to use your own VM (do) or container (don't) to isolate it.

It doesn't ask you every 2 seconds whether it is allowed to do some command.

Runs on ECMA-48 console (tested with "foot" terminal on Linux).

Supports Anthropic and OpenAI protocols.

## How to run

Run it in a VM or container.

```
export LOKI_API_KEY=xxx # or secret-tool store --label='opencode.ai api key' domain opencode.ai
export LOKI_API_BASE="https://opencode.ai/zen/go/v1/chat/completions"
./loki.py
```

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
